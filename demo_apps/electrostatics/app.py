# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This demo solves an electrostatics problem around a MEMS comb-drive actuator
using WoSX's GPU walk on stars solver. The lower anchored comb is grounded,
while the upper comb has a prescribed voltage and is driven by a sinusoidal
displacement. For each frame, the demo estimates the electric potential and
electric field on a slice plane and visualizes the result in Polyscope when
'visualizeProblem' is enabled in the config; otherwise, it saves the results
as PNG images.
'''

import argparse
import json
import sys
import math
import os
import numpy as np
import wosx
import polyscope as ps
import polyscope.imgui as psim
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

################################################################################################################
# Utility structs and functions

class MovableCombState:
    def __init__(self):
        self.voltage = 0.0
        self.displacement_amplitude = 0.025
        self.frame_index = 0
        self.n_frames = 120
        self.is_moving = False
        self.has_plotted_solution = False

    def increment_frame_index(self):
        # increment frame index and wrap around if necessary
        self.frame_index += 1
        if self.frame_index >= self.n_frames:
            self.frame_index -= self.n_frames

class MeshData:
    def __init__(self):
        self.base_positions = wosx.Float3List()
        self.current_positions = wosx.Float3List()
        self.indices = wosx.Int3List()
        self.is_movable_face = None
        self.is_movable_vertex = None

        self.bounding_box = None
        self.box_positions = wosx.Float3List()
        self.box_indices = wosx.Int3List()

    def compute_movable_comb_displacement(self, comb_state):
        time = float(comb_state.frame_index) / float(comb_state.n_frames)
        return comb_state.displacement_amplitude * np.sin(2.0 * np.pi * time)

    def update_movable_comb(self, displacement):
        positions = wosx.convert_list_to_numpy_array(self.base_positions)
        positions[self.is_movable_vertex, 1] += displacement
        self.current_positions = wosx.Float3List(positions)

class SlicePlaneData:
    def __init__(self):
        self.positions = None
        self.indices = None

    def build(self, bounding_box, resolution_pow2):
        # compute positions and indices
        size = 1 << int(resolution_pow2)
        h, w = np.meshgrid(np.arange(size + 1), np.arange(size + 1), indexing="ij")

        positions = np.zeros(((size + 1) * (size + 1), 3), dtype=np.float32)
        positions[:, 0] = w.reshape(-1)
        positions[:, 1] = h.reshape(-1)

        cell_h, cell_w = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        index = cell_h.reshape(-1) * (size + 1) + cell_w.reshape(-1)
        self.indices = np.ascontiguousarray(
            np.stack([index, index + 1, index + size + 2, index + size + 1], axis=1),
            dtype=np.int64)

        # normalize, scale and shift positions
        center = 0.5 * (bounding_box[0] + bounding_box[1])
        extent = bounding_box[1] - bounding_box[0]
        center[2] = 0.0
        extent[2] = 0.0
        scale = 0.125 * np.linalg.norm(extent)

        positions -= np.mean(positions, axis=0)
        positions /= np.max(np.linalg.norm(positions, axis=1))
        positions *= scale
        positions += center
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)

    def get_cell_centers(self):
        return np.mean(self.positions[self.indices], axis=1).astype(np.float32)

def get_filename(directory_path, config, filename,
                 is_required=True, default_value=""):
    if filename in config:
        return str(Path(directory_path) / config[filename])

    if is_required:
        raise ValueError(f"Missing required config entry: {filename}")

    return str(Path(directory_path) / default_value)

def get_frame_filename(filename, frame_index):
    path = Path(filename)
    return str(path.parent / f"{path.stem}_{frame_index:04d}{path.suffix}")

def save_colormapped_image(filename, image_height, image_width, values,
                           colormap, colormap_min_val, colormap_max_val):
    # create the output directory if it doesn't exist
    output_directory = os.path.dirname(filename)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    # fill the image
    image_buffer = np.zeros((image_height, image_width), dtype=np.float32)
    for i, value in enumerate(values):
        row = image_height - i // image_width - 1
        col = i % image_width
        image_buffer[row, col] = value

    # apply the requested colormap and save the RGB image
    colormap_range = colormap_max_val - colormap_min_val
    if colormap_range == 0.0:
        colormap_range = 1.0
    image = np.clip((image_buffer - colormap_min_val) / colormap_range, 0.0, 1.0)
    cmap = plt.get_cmap(colormap)
    colormapped_image = cmap(image, bytes=True)
    colormapped_image = np.clip(colormapped_image[:, :, :3].astype(np.uint8), 0, 255)
    Image.fromarray(colormapped_image).save(filename)

def save_solution_and_gradient_norm_images(directory_path, output_config,
                                           frame_index, slice_resolution_pow2,
                                           electric_potential, electric_field_magnitude):
    # get the filenames and colormap parameters for the electric potential
    electric_potential_file = get_frame_filename(
        get_filename(directory_path, output_config, "electricPotentialFile",
                     False, "solutions/electric_potential.png"), frame_index)
    electric_potential_colormap = output_config["electricPotentialColormap"]\
        if "electricPotentialColormap" in output_config else "plasma"
    electric_potential_colormap_min_val = output_config["electricPotentialColormapMinVal"]\
        if "electricPotentialColormapMinVal" in output_config else 0.0
    electric_potential_colormap_max_val = output_config["electricPotentialColormapMaxVal"]\
        if "electricPotentialColormapMaxVal" in output_config else 100.0

    # get the filenames and colormap parameters for the electric field magnitude
    electric_field_magnitude_file = get_frame_filename(
        get_filename(directory_path, output_config, "electricFieldMagnitudeFile",
                     False, "solutions/electric_field_magnitude.png"), frame_index)
    electric_field_magnitude_colormap = output_config["electricFieldMagnitudeColormap"]\
        if "electricFieldMagnitudeColormap" in output_config else "turbo"
    electric_field_magnitude_colormap_min_val = output_config["electricFieldMagnitudeColormapMinVal"]\
        if "electricFieldMagnitudeColormapMinVal" in output_config else 0.0
    electric_field_magnitude_colormap_max_val = output_config["electricFieldMagnitudeColormapMaxVal"]\
        if "electricFieldMagnitudeColormapMaxVal" in output_config else 15000.0

    # save the images
    slice_resolution = 1 << int(slice_resolution_pow2)
    save_colormapped_image(electric_potential_file, slice_resolution, slice_resolution,
                           electric_potential, electric_potential_colormap,
                           electric_potential_colormap_min_val,
                           electric_potential_colormap_max_val)
    save_colormapped_image(electric_field_magnitude_file, slice_resolution, slice_resolution,
                           electric_field_magnitude, electric_field_magnitude_colormap,
                           electric_field_magnitude_colormap_min_val,
                           electric_field_magnitude_colormap_max_val)

################################################################################################################
# Problem specification - load mesh data, setup the GPU geometric queries and PDE objects

def partition_into_components(n_positions, indices):
    # build a map of adjacent faces for each vertex
    adjacent_faces = [[] for _ in range(n_positions)]
    for face_index, face in enumerate(indices):
        for vertex_index in face:
            adjacent_faces[int(vertex_index)].append(face_index)

    # partition the faces into components
    components = np.full(len(indices), -1, dtype=np.int32)
    component = 0
    for face_index in range(len(indices)):
        if components[face_index] >= 0:
            continue

        queue = deque([face_index])
        components[face_index] = component
        while queue:
            current_face = queue.popleft()
            for vertex_index in indices[current_face]:
                for neighbor_face in adjacent_faces[int(vertex_index)]:
                    if components[neighbor_face] < 0:
                        components[neighbor_face] = component
                        queue.append(neighbor_face)

        component += 1

    return components.astype(np.uint32)

def load_mesh_data(directory_path, problem_config):
    # load the mesh
    geometry_filename = get_filename(directory_path, problem_config, "geometry")
    mesh_data = MeshData()
    wosx.Utils.load_boundary_mesh(geometry_filename, mesh_data.base_positions,
                                  mesh_data.indices, dim=3)
    wosx.Utils.normalize(mesh_data.base_positions, dim=3) # normalize to a unit sphere

    n_positions = len(mesh_data.base_positions)
    mesh_data.current_positions = wosx.Float3List(mesh_data.base_positions)
    indices_numpy = wosx.convert_list_to_numpy_array(mesh_data.indices)
    mesh_data.is_movable_face = partition_into_components(n_positions, indices_numpy)
    mesh_data.is_movable_vertex = np.zeros(n_positions, dtype=bool)
    for face_index, face in enumerate(indices_numpy):
        if mesh_data.is_movable_face[face_index] == 1:
            mesh_data.is_movable_vertex[face] = True

    # compute a bounding box for the mesh
    mesh_data.bounding_box = wosx.Utils.compute_bounding_box(mesh_data.base_positions,
                                                             True, 1.5, dim=3)
    wosx.Utils.build_bounding_box_mesh(mesh_data.bounding_box[0], mesh_data.bounding_box[1],
                                       mesh_data.box_positions, mesh_data.box_indices, dim=3)

    return mesh_data

def setup_geometric_queries(mesh_data):
    # setup an absorbing boundary handler for the mesh
    absorbing_boundary_handler = wosx.Utils.GPUFcpwDirichletBoundaryHandler(
        mesh_data.current_positions, mesh_data.indices, dim=3)

    # setup a reflecting boundary handler for the mesh's bounding box
    ignore_candidate_silhouette = wosx.Utils.get_ignore_candidate_silhouette_callback(False)
    reflecting_boundary_handler = wosx.Utils.GPUFcpwNeumannBoundaryHandler(
        mesh_data.box_positions, mesh_data.box_indices,
        ignore_candidate_silhouette, dim=3)

    # create a geometric queries object from the handlers
    geometric_queries = wosx.Core.GPUGeometricQueries(
        absorbing_boundary_handler, reflecting_boundary_handler,
        mesh_data.bounding_box[0], mesh_data.bounding_box[1], True, dim=3)

    return geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler

class GPUElectrostaticsPDE(wosx.Core.GPUPDE):
    def __init__(self, acceleration_structure,
                 is_movable_face, movable_comb_voltage):
        super().__init__()
        self._acceleration_structure = acceleration_structure
        self._is_movable_face = wosx.UintList(is_movable_face)
        self._is_movable_face_buffer = wosx.Utils.GPUUintBuffer(False)
        self._movable_comb_voltage = movable_comb_voltage

    def update_acceleration_structure(self, acceleration_structure):
        self._acceleration_structure = acceleration_structure

    def update_movable_comb_voltage(self, movable_comb_voltage):
        self._movable_comb_voltage = movable_comb_voltage

    def allocate(self, context):
        self._is_movable_face_buffer.allocate(context, self._is_movable_face)

    def set_resources(self, cursor, print_logs):
        self._acceleration_structure.set_resources(cursor["mAccelerationStructure"], print_logs)
        self._is_movable_face_buffer.set_binding(cursor["mIsMovableFace"])
        cursor["mMovableCombVoltage"].set_data_float(self._movable_comb_voltage)
        if print_logs:
            wosx.Utils.print_gpu_reflection_info(cursor, 3, self.get_reflection_type())

    def get_reflection_type(self):
        return "ElectrostaticsPDE"

    def get_type(self):
        return wosx.Core.GPUPDEType.Poisson

def update_movable_comb_position(mesh_data, displacement, task_handle,
                                 geometric_queries, pde):
    # update the movable comb position
    mesh_data.update_movable_comb(displacement)

    # update the absorbing boundary handler for the geometric queries object
    absorbing_boundary_handler = wosx.Utils.GPUFcpwDirichletBoundaryHandler(
        mesh_data.current_positions, mesh_data.indices, False, dim=3)
    geometric_queries.reallocate_absorbing_boundary(task_handle.get_context(),
                                                    absorbing_boundary_handler,
                                                    mesh_data.bounding_box[0],
                                                    mesh_data.bounding_box[1], True)

    # update the PDE acceleration structure
    pde.update_acceleration_structure(absorbing_boundary_handler)

################################################################################################################
# Walk on stars solver - create sample points, run the solver and extract the solution and gradient

def create_sample_points(slice_plane_data):
    sample_pts = []
    solve_locations = slice_plane_data.get_cell_centers()
    for location in solve_locations:
        sample_pt = wosx.Solvers.GPUSamplePoint(dim=3)
        sample_pt.pt = wosx.GPUFloat3(location[0], location[1], location[2])
        sample_pt.normal = wosx.GPUFloat3(0.0, 0.0, 1.0)
        sample_pt.type = wosx.Solvers.SampleType.InDomain
        sample_pt.estimation_quantity = wosx.Solvers.EstimationQuantity.SolutionAndGradient
        sample_pts.append(sample_pt)

    return wosx.Solvers.GPUSamplePointList(sample_pts, dim=3)

def setup_solver(solver_config, task_handle, sample_pts):
    # load config settings for walk on stars
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3

    n_walks = solver_config["nWalks"]\
        if "nWalks" in solver_config else 128
    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 10000
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length
    n_resident_threads = solver_config["nResidentThreads"]\
        if "nResidentThreads" in solver_config else 131072

    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    disable_persistent_threads = solver_config["disablePersistentThreads"]\
        if "disablePersistentThreads" in solver_config else False
    if not disable_persistent_threads and task_handle.get_device_backend() != "cuda":
        print("Persistent threads require CUDA backend, disabling")
        disable_persistent_threads = True

    # setup walk settings for the solver
    walk_settings = wosx.Solvers.GPUWalkSettings()
    walk_settings.epsilon_shell_for_absorbing_boundary = epsilon_shell_for_absorbing_boundary
    walk_settings.epsilon_shell_for_reflecting_boundary = epsilon_shell_for_reflecting_boundary
    walk_settings.silhouette_precision = silhouette_precision
    walk_settings.russian_roulette_threshold = 0.0
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = steps_before_using_maximal_spheres
    walk_settings.solve_double_sided = 0
    walk_settings.use_gradient_control_variates = 1
    walk_settings.use_gradient_antithetic_variates = 1
    walk_settings.ignore_absorbing_boundary_contribution = 0
    walk_settings.ignore_reflecting_boundary_contribution = 1 # demo uses perfectly insulating (i.e., zero) Neumann conditions
    walk_settings.ignore_source_contribution = 1

    # initialize the solver
    walk_on_stars = wosx.Solvers.GPUWalkOnStarsSolver(task_handle, walk_settings,
                                                      n_resident_threads,
                                                      not disable_persistent_threads,
                                                      print_logs, dim=3, channels=1)

    # compute workload parameters--the number of sample copies and number of
    # walks per copy--to help improve GPU occupancy when running the solver
    if disable_persistent_threads:
        workload_parameters = walk_on_stars.compute_workload_parameters(len(sample_pts), n_walks)

    else:
        workload_parameters = (1, n_walks)

    # populate sample points on the GPU
    walk_on_stars.populate_sample_points(sample_pts, workload_parameters[0], False)

    return walk_on_stars, workload_parameters

def run_solver(walk_on_stars, workload_parameters):
    # update the boundary distance for the populated sample points
    walk_on_stars.update_populated_sample_points_boundary_distance()

    # reset the sample statistics from the previous solve
    walk_on_stars.reset_sample_statistics()

    # solve
    if walk_on_stars.using_persistent_threads():
        walk_on_stars.solve(workload_parameters[1])

    else:
        walk_on_stars.solve(1, workload_parameters[1])

    # extract sample statistics from the GPU
    sample_statistics = wosx.Solvers.GPUSampleStatisticsList(dim=3, channels=1)
    walk_on_stars.get_sample_statistics(sample_statistics)

    return sample_statistics

def get_solution_and_gradient_norm(sample_statistics):
    n_samples = len(sample_statistics)
    electric_potential = np.empty(n_samples, dtype=np.float32)
    electric_field_magnitude = np.empty(n_samples, dtype=np.float32)

    for i, statistics in enumerate(sample_statistics):
        electric_potential[i] = statistics.get_estimated_solution()
        gx = statistics.get_estimated_gradient(0)
        gy = statistics.get_estimated_gradient(1)
        gz = statistics.get_estimated_gradient(2)
        electric_field_magnitude[i] = math.sqrt(math.sqrt(gx*gx + gy*gy + gz*gz)) # NOTE: compressing high-field peaks for visualization

    return electric_potential, electric_field_magnitude

################################################################################################################
# Problem Visualization

def set_camera_view(mesh_data):
    p_min, p_max = mesh_data.bounding_box
    center = 0.5 * (p_min + p_max)
    view_distance = 0.6 * max(p_max[0] - p_min[0], p_max[1] - p_min[1])

    camera = center.copy()
    camera[2] += view_distance
    ps.look_at(camera, center)

def plot_comb_voltage(output_config, mesh_data, voltage):
    mesh = ps.get_surface_mesh("Mesh")
    voltages = np.zeros(len(mesh_data.is_movable_face), dtype=np.float32)
    voltages[mesh_data.is_movable_face == 1] = voltage
    voltage_colormap_min_val = output_config["electricPotentialColormapMinVal"]\
        if "electricPotentialColormapMinVal" in output_config else 0.0
    voltage_colormap_max_val = output_config["electricPotentialColormapMaxVal"]\
        if "electricPotentialColormapMaxVal" in output_config else 100.0

    mesh.add_scalar_quantity("Voltage", voltages,
                             defined_on="faces", cmap="reds",
                             vminmax=(voltage_colormap_min_val,
                                      voltage_colormap_max_val),
                             enabled=True)

def plot_solution_and_gradient(output_config, electric_potential,
                               electric_field_magnitude, plot_potential):
    slice_plane = ps.get_surface_mesh("Slice Plane")

    # plot electric potential
    electric_potential_colormap = output_config["electricPotentialColormap"]\
        if "electricPotentialColormap" in output_config else "plasma"
    electric_potential_colormap_min_val = output_config["electricPotentialColormapMinVal"]\
        if "electricPotentialColormapMinVal" in output_config else 0.0
    electric_potential_colormap_max_val = output_config["electricPotentialColormapMaxVal"]\
        if "electricPotentialColormapMaxVal" in output_config else 100.0
    if plot_potential:
        slice_plane.add_scalar_quantity("Electric Potential", electric_potential,
                                        defined_on="faces", cmap=electric_potential_colormap,
                                        vminmax=(electric_potential_colormap_min_val,
                                                 electric_potential_colormap_max_val),
                                        enabled=True)
    else:
        slice_plane.add_scalar_quantity("Electric Potential", electric_potential,
                                        defined_on="faces", cmap=electric_potential_colormap,
                                        vminmax=(electric_potential_colormap_min_val,
                                                 electric_potential_colormap_max_val))

    # plot electric field magnitude
    electric_field_magnitude_colormap = output_config["electricFieldMagnitudeColormap"]\
        if "electricFieldMagnitudeColormap" in output_config else "turbo"
    electric_field_magnitude_colormap_min_val = output_config["electricFieldMagnitudeColormapMinVal"]\
        if "electricFieldMagnitudeColormapMinVal" in output_config else 0.0
    electric_field_magnitude_colormap_max_val = output_config["electricFieldMagnitudeColormapMaxVal"]\
        if "electricFieldMagnitudeColormapMaxVal" in output_config else 15000.0
    slice_plane.add_scalar_quantity("Electric Field Magnitude", electric_field_magnitude,
                                    defined_on="faces", cmap=electric_field_magnitude_colormap,
                                    vminmax=(electric_field_magnitude_colormap_min_val,
                                             electric_field_magnitude_colormap_max_val))

def gui_callback(output_config, mesh_data, comb_state,
                 task_handle, geometric_queries, pde,
                 walk_on_stars, workload_parameters):
    if comb_state.is_moving:
        if psim.Button("Stop Movable Comb"):
            comb_state.is_moving = False

    else:
        changed, voltage = psim.SliderFloat("Movable Comb Voltage",
                                            comb_state.voltage, 0.0, 100.0)
        if changed:
            # remove the existing quantities from the slice plane
            comb_state.voltage = voltage
            comb_state.has_plotted_solution = False
            slice_plane = ps.get_surface_mesh("Slice Plane")
            slice_plane.remove_quantity("Electric Potential")
            slice_plane.remove_quantity("Electric Field Magnitude")

            # update the PDE and plot the voltage on the comb
            pde.update_movable_comb_voltage(comb_state.voltage)
            plot_comb_voltage(output_config, mesh_data, comb_state.voltage)

        if psim.Button("Solve with Movable Comb"):
            comb_state.is_moving = True

    if comb_state.is_moving:
        # update the movable comb position
        displacement = mesh_data.compute_movable_comb_displacement(comb_state)
        update_movable_comb_position(mesh_data, displacement, task_handle,
                                     geometric_queries, pde)

        # run the solver and get the sample statistics
        sample_statistics = run_solver(walk_on_stars, workload_parameters)

        # extract electric potential and field magnitude estimates
        electric_potential, electric_field_magnitude =\
            get_solution_and_gradient_norm(sample_statistics)

        # update the mesh positions in polyscope
        mesh_positions = wosx.convert_list_to_numpy_array(mesh_data.current_positions)
        ps.get_surface_mesh("Mesh").update_vertex_positions(mesh_positions)

        # plot the estimated results on the slice plane
        plot_solution_and_gradient(output_config, electric_potential,
                                   electric_field_magnitude,
                                   not comb_state.has_plotted_solution)
        comb_state.has_plotted_solution = True

        # increment the frame index
        comb_state.increment_frame_index()

def visualize_problem(output_config, mesh_data, slice_plane_data,
                      comb_state, task_handle, geometric_queries,
                      pde, walk_on_stars, workload_parameters):
    # set a few options
    ps.set_program_name("electrostatics demo")
    ps.set_verbosity(0)
    ps.set_use_prefs_file(False)
    ps.set_autocenter_structures(False)
    ps.set_ground_plane_mode("none")

    # initialize polyscope
    ps.init()

    # set camera view
    set_camera_view(mesh_data)

    # register the mesh
    mesh_positions = wosx.convert_list_to_numpy_array(mesh_data.current_positions)
    mesh_indices = wosx.convert_list_to_numpy_array(mesh_data.indices)
    ps.register_surface_mesh("Mesh", mesh_positions, mesh_indices)
    plot_comb_voltage(output_config, mesh_data, comb_state.voltage)

    # register the slice plane
    ps.register_surface_mesh("Slice Plane",
                             slice_plane_data.positions,
                             slice_plane_data.indices)

    # bind the gui callback
    ps.set_user_callback(lambda: gui_callback(output_config, mesh_data, comb_state,
                                              task_handle, geometric_queries, pde,
                                              walk_on_stars, workload_parameters))

    # give control to polyscope gui
    try:
        ps.show()
    finally:
        ps.set_user_callback(None)

################################################################################################################
# Demo entry point - load config, setup the problem, run the solver, save the solution,
# or optionally visualize the problem

def run_demo():
    # parse arguments
    parser = argparse.ArgumentParser(description="electrostatics demo")
    parser.add_argument("--config", type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    try:
        # load the configuration file
        with open(args.config, 'r') as file:
            config = json.load(file)

        # load config settings
        wosx_directory_path = Path.cwd()
        demo_directory_path = wosx_directory_path / "demo_apps" / "electrostatics"
        wosx_directory_path_str = str(wosx_directory_path)
        demo_directory_path_str = str(demo_directory_path)
        problem_config = config["problem"]
        solver_config = config["solver"]
        output_config = config["output"]

        # load mesh data
        mesh_data = load_mesh_data(demo_directory_path_str, problem_config)

        # setup a geometric queries object for the mesh
        geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler =\
            setup_geometric_queries(mesh_data)

        # setup the demo PDE
        comb_state = MovableCombState()
        comb_state.voltage = problem_config["movableCombVoltage"]\
            if "movableCombVoltage" in problem_config else 75.0
        comb_state.n_frames = max(1, problem_config["nFrames"]\
            if "nFrames" in problem_config else 120)
        pde = GPUElectrostaticsPDE(absorbing_boundary_handler,
                                   mesh_data.is_movable_face,
                                   comb_state.voltage)

        # setup a task handle to run the demo on the GPU
        device_backend = config["deviceBackend"]\
            if "deviceBackend" in config else "cuda"
        task_handle = wosx.GPUTaskHandle(wosx_directory_path_str,
                                         demo_directory_path_str,
                                         device_backend, dim=3, channels=1)
        task_handle.set_geometric_queries(geometric_queries)
        task_handle.set_pde(pde)
        task_handle.init()

        # create sample points to run the solver on; here we use a slice plane through the mesh
        slice_resolution_pow2 = problem_config["sliceResolutionPow2"]
        slice_plane_data = SlicePlaneData()
        slice_plane_data.build(mesh_data.bounding_box, slice_resolution_pow2)
        sample_pts = create_sample_points(slice_plane_data)

        # setup the walk on stars solver
        walk_on_stars, workload_parameters =\
            setup_solver(solver_config, task_handle, sample_pts)

        should_visualize_problem = output_config["visualizeProblem"]\
            if "visualizeProblem" in output_config else False
        if should_visualize_problem:
            # visualize the problem
            visualize_problem(output_config, mesh_data, slice_plane_data, comb_state,
                              task_handle, geometric_queries, pde, walk_on_stars,
                              workload_parameters)

        else:
            # run the solver and save the results as images
            for _ in range(comb_state.n_frames):
                print(f"Solving frame {comb_state.frame_index}")

                # update the movable comb position
                displacement = mesh_data.compute_movable_comb_displacement(comb_state)
                update_movable_comb_position(mesh_data, displacement, task_handle,
                                             geometric_queries, pde)

                # run the solver and get the sample statistics
                sample_statistics = run_solver(walk_on_stars, workload_parameters)

                # extract electric potential and field magnitude estimates
                electric_potential, electric_field_magnitude =\
                    get_solution_and_gradient_norm(sample_statistics)

                # save the estimated results as frame-indexed colormapped images
                save_solution_and_gradient_norm_images(demo_directory_path_str,
                                                       output_config,
                                                       comb_state.frame_index,
                                                       slice_resolution_pow2,
                                                       electric_potential,
                                                       electric_field_magnitude)

                # increment the frame index
                comb_state.increment_frame_index()

    except FileNotFoundError:
        sys.exit("Configuration file not found")

    except json.JSONDecodeError:
        sys.exit("Invalid configuration file")

    except ValueError as error:
        sys.exit(str(error))

if __name__ == "__main__":
    run_demo()
