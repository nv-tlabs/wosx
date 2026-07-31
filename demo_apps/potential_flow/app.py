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
This demo solves an exterior potential-flow problem around a closed 3D body
using WoSX's GPU boundary value caching solver. A uniform freestream velocity
is prescribed at infinity, and the body surface enforces a no-through-flow
Neumann condition for the perturbation potential. The exterior domain is
handled with a Kelvin transform, which maps the unbounded problem to a bounded
inverted domain. When 'visualizeSetup' is enabled, the demo opens Polyscope to
inspect the input geometry, active slice plane cells, and freestream direction.
Otherwise, it estimates the perturbation potential and flow velocity on a
slice plane and saves the results to file via Polyscope's screenshot tool.
'''

import argparse
import json
import sys
import os
import numpy as np
import wosx
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path

################################################################################################################
# Utility structs and functions

class MeshData:
    def __init__(self):
        self.positions = wosx.Float3List()
        self.indices = wosx.Int3List()

        self.inverted_positions = wosx.Float3List()
        self.inverted_min_robin_coeff_values = wosx.FloatList()
        self.inverted_max_robin_coeff_values = wosx.FloatList()

        self.bounding_box = None
        self.inverted_bounding_box = None

    def get_inside_bounding_box_callback(self):
        return wosx.Utils.get_inside_bounding_box_callback(self.bounding_box[0],
                                                           self.bounding_box[1],
                                                           dim=3)

    def get_inside_inverted_bounding_box_callback(self):
        return wosx.Utils.get_inside_bounding_box_callback(self.inverted_bounding_box[0],
                                                           self.inverted_bounding_box[1],
                                                           dim=3)

class SlicePlaneData:
    def __init__(self):
        self.positions = None
        self.indices = None
        self.cell_centers = None
        self.active_cells = None

    def build(self, problem_config, bounding_box):
        # compute positions and indices
        resolution_pow2 = 6
        size = 1 << resolution_pow2
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
        scale = 0.5 * np.linalg.norm(extent)

        positions -= np.mean(positions, axis=0)
        positions /= np.max(np.linalg.norm(positions, axis=1))
        positions *= scale
        positions += center
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)

        # compute cell centers
        self.cell_centers = np.mean(self.positions[self.indices], axis=1).astype(np.float32)

        # set active cells
        self.active_cells = np.ones(len(self.indices), dtype=bool)
        slice_crop_min_y = problem_config["sliceCropMinY"]\
            if "sliceCropMinY" in problem_config else -0.3
        slice_crop_max_y = problem_config["sliceCropMaxY"]\
            if "sliceCropMaxY" in problem_config else 0.65
        self.active_cells &= self.cell_centers[:, 1] <= slice_crop_max_y
        self.active_cells &= self.cell_centers[:, 1] >= slice_crop_min_y

    def deactivate_interior_cells(self, task_handle):
        # compute signed distance to the boundary from cell centers
        cell_centers = wosx.Float3List(self.cell_centers)
        dist_to_boundary = wosx.FloatList()
        stub = wosx.FloatList()
        distance_queries = wosx.Utils.GPUDistanceQueries(task_handle, dim=3, channels=1)
        distance_queries.compute_dist_to_boundary(cell_centers, dist_to_boundary, stub)

        # deactivate interior cells using signed distance
        boundary_distance_margin = 0.035
        dist_to_boundary_numpy = wosx.convert_list_to_numpy_array(dist_to_boundary)
        self.active_cells &= dist_to_boundary_numpy > boundary_distance_margin

def get_filename(directory_path, config, filename,
                 is_required=True, default_value=""):
    if filename in config:
        return str(Path(directory_path) / config[filename])

    if is_required:
        raise ValueError(f"Missing required config entry: {filename}")

    return str(Path(directory_path) / default_value)

################################################################################################################
# Problem specification - load mesh data, setup the GPU geometric queries and PDE objects

def load_mesh_data(directory_path, problem_config):
    # load the mesh
    geometry_filename = get_filename(directory_path, problem_config, "geometry")
    mesh_data = MeshData()
    wosx.Utils.load_boundary_mesh(geometry_filename, mesh_data.positions,
                                  mesh_data.indices, dim=3)
    wosx.Utils.normalize(mesh_data.positions, dim=3) # normalize to a unit sphere

    # compute a bounding box for the mesh
    mesh_data.bounding_box = wosx.Utils.compute_bounding_box(mesh_data.positions,
                                                             True, 1.5, dim=3)

    return mesh_data

def invert_mesh(kelvin_transform, mesh_data):
    # invert mesh positions using the Kelvin transform
    kelvin_transform.transform_points(mesh_data.positions, mesh_data.inverted_positions)

    # compute the Robin coefficients for the inverted domain
    min_robin_coeff_values = wosx.FloatList([0.0] * len(mesh_data.indices))
    max_robin_coeff_values = wosx.FloatList([0.0] * len(mesh_data.indices))
    kelvin_transform.compute_robin_coefficients(mesh_data.inverted_positions, mesh_data.indices,
                                                min_robin_coeff_values, max_robin_coeff_values,
                                                mesh_data.inverted_min_robin_coeff_values,
                                                mesh_data.inverted_max_robin_coeff_values)

    # compute a bounding box for the inverted mesh
    mesh_data.inverted_bounding_box = wosx.Utils.compute_bounding_box(mesh_data.inverted_positions,
                                                                      True, 1.0, dim=3)

def setup_geometric_queries(mesh_data, use_inverted_mesh):
    # setup boundary handlers and bounding box based on the use_inverted_mesh flag
    absorbing_boundary_handler = None
    reflecting_boundary_handler = None
    bounding_box = None

    if use_inverted_mesh:
        # setup an empty absorbing boundary handler
        absorbing_boundary_handler = wosx.Core.GPUEmptyAbsorbingBoundaryHandler()

        # setup a robin reflecting boundary handler
        ignore_candidate_silhouette = wosx.Utils.get_ignore_candidate_silhouette_callback(False)
        reflecting_boundary_handler = wosx.Utils.GPUFcpwRobinBoundaryHandler(
            mesh_data.inverted_positions, mesh_data.indices,
            ignore_candidate_silhouette, mesh_data.inverted_min_robin_coeff_values,
            mesh_data.inverted_max_robin_coeff_values, dim=3)

        # set the inverted bounding box
        bounding_box = mesh_data.inverted_bounding_box

    else:
        # setup an absorbing boundary handler
        absorbing_boundary_handler = wosx.Utils.GPUFcpwDirichletBoundaryHandler(
            mesh_data.positions, mesh_data.indices, dim=3)

        # setup an empty reflecting boundary handler
        reflecting_boundary_handler = wosx.Core.GPUEmptyReflectingBoundaryHandler()

        # set the bounding box
        bounding_box = mesh_data.bounding_box

    # create a geometric queries object from the handlers
    geometric_queries = wosx.Core.GPUGeometricQueries(
        absorbing_boundary_handler, reflecting_boundary_handler,
        bounding_box[0], bounding_box[1], True, dim=3)

    return geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler

class FreestreamState:
    def __init__(self, speed, angle):
        self.speed = speed
        self.angle = angle

    def get_velocity(self):
        return np.asarray([self.speed * np.cos(self.angle),
                           self.speed * np.sin(self.angle),
                           0.0], dtype=np.float32)

class GPUPotentialFlowPDE(wosx.Core.GPUPDE):
    def __init__(self, freestream_state):
        super().__init__()
        self._freestream_state = freestream_state

    def allocate(self, context):
        # nothing to do
        pass

    def set_resources(self, cursor, print_logs):
        cursor["mFreestreamSpeed"].set_data_float(self._freestream_state.speed)
        cursor["mFreestreamAngle"].set_data_float(self._freestream_state.angle)
        if print_logs:
            wosx.Utils.print_gpu_reflection_info(cursor, 2, self.get_reflection_type())

    def get_reflection_type(self):
        return "PotentialFlowPDE"

    def get_type(self):
        return wosx.Core.GPUPDEType.Poisson

def compute_boundary_sampling_weights(mesh_data, freestream_state):
    # set primitive weights to a mixture of two densities: one proportional to the
    # Kelvin-transformed Robin RHS, and one uniform with respect to the original surface
    # area (a weight of 1/|y|^4 per unit inverted area, since an inversion scales areas
    # by 1/|x|^4). The latter keeps the sample density from vanishing where the freestream
    # is tangent to the surface, as samples there carry a large 1/pdf during splatting.
    positions = wosx.convert_list_to_numpy_array(mesh_data.inverted_positions)
    indices = wosx.convert_list_to_numpy_array(mesh_data.indices)
    freestream_velocity = freestream_state.get_velocity()

    # compute triangle centers, normals and areas
    tri_positions = positions[indices]
    centers = np.mean(tri_positions, axis=1)
    normals = np.cross(tri_positions[:, 1] - tri_positions[:, 0],
                       tri_positions[:, 2] - tri_positions[:, 0])
    normal_norms = np.linalg.norm(normals, axis=1)
    areas = 0.5 * normal_norms
    valid_normals = normal_norms > np.finfo(np.float32).eps
    normals[valid_normals] /= normal_norms[valid_normals, None]
    normals[~valid_normals] = 0.0

    # compute the squared distances to the centers
    r2 = np.sum(centers * centers, axis=1)
    valid_radii = r2 > np.finfo(np.float32).eps

    # compute the weights for each density
    rhs_weights = np.zeros(len(indices), dtype=np.float32)
    area_weights = np.zeros(len(indices), dtype=np.float32)
    if np.any(valid_radii):
        centers_valid = centers[valid_radii]
        normals_valid = normals[valid_radii]
        r2_valid = r2[valid_radii]

        normal_dot_center = np.sum(normals_valid * centers_valid, axis=1)
        transformed_normals = normals_valid - \
            (2.0 * normal_dot_center / r2_valid)[:, None] * centers_valid
        weights = np.abs(-np.sum(transformed_normals * freestream_velocity, axis=1))
        weights /= r2_valid
        weights /= np.sqrt(r2_valid)
        rhs_weights[valid_radii] = weights
        area_weights[valid_radii] = 1.0 / (r2_valid * r2_valid)

    # the sampler builds its cdf from primitive weights scaled by primitive areas,
    # so normalize by the sampling mass of each density to combine them evenly
    rhs_sampling_mass = max(float(np.sum(areas * rhs_weights)), 1e-8)
    area_sampling_mass = max(float(np.sum(areas * area_weights)), 1e-8)
    primitive_weights = 0.75 * rhs_weights / rhs_sampling_mass + 0.25 * area_weights / area_sampling_mass

    # return the mixture primitive weights
    return wosx.FloatList(primitive_weights.astype(np.float32))

################################################################################################################
# Boundary value caching solver - create evaluation points, run the solver and
# extract the solution and gradient

def create_evaluation_points(slice_plane_data, kelvin_transform):
    # use inverted cell centers as solve locations
    cell_centers = wosx.Float3List(slice_plane_data.cell_centers)
    solve_locations = wosx.Float3List()
    kelvin_transform.transform_points(cell_centers, solve_locations)

    # create evaluation points that are exterior to the
    # input mesh and interior to the inverted mesh
    eval_pts = []
    for i, is_active in enumerate(slice_plane_data.active_cells):
        if is_active:
            solve_location = solve_locations[i]
            eval_pt = wosx.Solvers.GPUBVCEvaluationPoint(dim=3)
            eval_pt.pt = wosx.GPUFloat3(solve_location[0], solve_location[1], solve_location[2])
            eval_pt.normal = wosx.GPUFloat3(0.0, 0.0, 0.0)
            eval_pt.type = wosx.Solvers.SampleType.InDomain
            eval_pts.append(eval_pt)

    return wosx.Solvers.GPUBVCEvaluationPointList(eval_pts, dim=3)

def run_solver(solver_config, task_handle, eval_pts):
    # load config settings for random walks
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.99

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

    # load config settings for boundary value caching
    reflecting_boundary_cache_size = solver_config["reflectingBoundaryCacheSize"]\
        if "reflectingBoundaryCacheSize" in solver_config else 10000
    n_walks_for_cached_solution_estimates = solver_config["nWalksForCachedSolutionEstimates"]\
        if "nWalksForCachedSolutionEstimates" in solver_config else 128
    radius_clamp_for_kernels = solver_config["radiusClampForKernels"]\
        if "radiusClampForKernels" in solver_config else 0.0
    regularization_for_kernels = solver_config["regularizationForKernels"]\
        if "regularizationForKernels" in solver_config else 0.0

    # setup walk settings for the solver
    walk_settings = wosx.Solvers.GPUWalkSettings()
    walk_settings.epsilon_shell_for_absorbing_boundary = epsilon_shell_for_absorbing_boundary
    walk_settings.epsilon_shell_for_reflecting_boundary = epsilon_shell_for_reflecting_boundary
    walk_settings.silhouette_precision = silhouette_precision
    walk_settings.russian_roulette_threshold = russian_roulette_threshold
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = steps_before_using_maximal_spheres
    walk_settings.solve_double_sided = 0
    walk_settings.use_gradient_control_variates = 1
    walk_settings.use_gradient_antithetic_variates = 1
    walk_settings.ignore_absorbing_boundary_contribution = 1
    walk_settings.ignore_reflecting_boundary_contribution = 0 # solving an exterior Neumann problem
    walk_settings.ignore_source_contribution = 1

    # initialize the solver
    boundary_value_caching = wosx.Solvers.GPUBoundaryValueCachingSolver(
        task_handle, walk_settings, n_resident_threads,
        not disable_persistent_threads, print_logs,
        dim=3, channels=1)

    # populate evaluation points on the GPU
    boundary_value_caching.populate_evaluation_points(eval_pts)

    # generate samples on the reflecting boundary (we do not require
    # domain or absorbing boundary samples for this problem)
    absorbing_boundary_cache_size = 0
    domain_cache_size = 0
    boundary_value_caching.generate_samples(absorbing_boundary_cache_size,
                                            reflecting_boundary_cache_size,
                                            domain_cache_size)

    # compute sample estimates on the boundary (we do not require
    # gradient estimates as there is no absorbing boundary)
    n_walks_for_cached_gradient_estimates = 0
    boundary_value_caching.compute_sample_estimates(n_walks_for_cached_solution_estimates,
                                                    n_walks_for_cached_gradient_estimates)

    # splat boundary sample estimates to evaluation points
    boundary_value_caching.splat(radius_clamp_for_kernels, regularization_for_kernels)

    # extract evaluation outputs from the GPU
    evaluation_outputs = wosx.Solvers.GPUBVCEvaluationOutputsList(dim=3, channels=1)
    boundary_value_caching.get_evaluation_outputs(evaluation_outputs)

    return evaluation_outputs

def get_solution_and_gradient(slice_plane_data, kelvin_transform,
                              eval_pts, eval_outputs, freestream_velocity):
    n_cells = len(slice_plane_data.cell_centers)
    perturbation_potential = np.zeros(n_cells, dtype=np.float32)
    flow_velocity = np.zeros((n_cells, 3), dtype=np.float32)
    counter = 0

    for i, is_active in enumerate(slice_plane_data.active_cells):
        if is_active:
            # get the solve location
            solve_pt = eval_pts[counter].pt
            solve_location = np.asarray([solve_pt.x, solve_pt.y, solve_pt.z], dtype=np.float32)

            # compute the perturbation potential at the solve location
            solution = eval_outputs[counter].get_estimated_solution()
            perturbation_potential[i] =\
                kelvin_transform.transform_solution_estimate(solution, solve_location)

            # compute the flow velocity at the solve location
            gradient = np.asarray([eval_outputs[counter].get_estimated_gradient(j)
                                   for j in range(3)], dtype=np.float32)
            gradient = kelvin_transform.transform_gradient_estimate(solution, gradient,
                                                                    solve_location)
            flow_velocity[i] = freestream_velocity + gradient
            counter += 1

    return perturbation_potential, flow_velocity

################################################################################################################
# Problem Visualization

def plot_freestream_velocity(slice_plane_data, freestream_state):
    # assign freestream velocity to active cells; inactive cells remain zero
    n_cells = len(slice_plane_data.cell_centers)
    freestream_velocity = np.zeros((n_cells, 3), dtype=np.float32)
    freestream_velocity[slice_plane_data.active_cells] = freestream_state.get_velocity()

    # plot freestream velocity on the slice plane
    vector_length = 0.02 * freestream_state.speed / 10.0
    slice_plane = ps.get_surface_mesh("Slice Plane")
    slice_plane.add_vector_quantity("Freestream Velocity", freestream_velocity,
                                    defined_on="faces", enabled=True,
                                    length=vector_length)

def gui_callback(slice_plane_data, freestream_state):
    changed, speed = psim.SliderFloat("Freestream Speed",
                                      freestream_state.speed, 1.0, 10.0)
    if changed:
        freestream_state.speed = speed
        plot_freestream_velocity(slice_plane_data, freestream_state)

    changed, angle = psim.SliderFloat("Freestream Angle",
                                      freestream_state.angle, 0.0, 2.0*np.pi)
    if changed:
        freestream_state.angle = angle
        plot_freestream_velocity(slice_plane_data, freestream_state)

def set_camera_view(mesh_data):
    p_min, p_max = mesh_data.bounding_box
    center = 0.5 * (p_min + p_max)
    view_distance = 1.25 * max(p_max[0] - p_min[0], p_max[1] - p_min[1])

    camera = center.copy()
    camera[2] += view_distance
    ps.look_at(camera, center)

def visualize_setup(mesh_data, slice_plane_data, freestream_state):
    # set a few options
    ps.set_program_name("potential flow demo - problem setup")
    ps.set_verbosity(0)
    ps.set_use_prefs_file(False)
    ps.set_autocenter_structures(False)
    ps.set_ground_plane_mode("none")

    # initialize polyscope
    ps.init()

    # set camera view
    set_camera_view(mesh_data)

    # register the mesh
    mesh_positions = wosx.convert_list_to_numpy_array(mesh_data.positions)
    mesh_indices = wosx.convert_list_to_numpy_array(mesh_data.indices)
    ps.register_surface_mesh("Mesh", mesh_positions, mesh_indices)

    # register the slice plane
    slice_plane = ps.register_surface_mesh("Slice Plane",
                                           slice_plane_data.positions,
                                           slice_plane_data.indices)
    slice_plane.set_color((1.0, 1.0, 1.0))
    slice_plane.set_material("flat")
    slice_plane.set_back_face_policy("cull")

    # plot freestream velocity on slice plane
    plot_freestream_velocity(slice_plane_data, freestream_state)

    # bind the gui callback
    ps.set_user_callback(lambda: gui_callback(slice_plane_data, freestream_state))

    # give control to polyscope gui
    try:
        ps.show()
    finally:
        ps.set_user_callback(None)

def save_solution_and_gradient(directory_path, output_config,
                               mesh_data, slice_plane_data, freestream_state,
                               perturbation_potential, flow_velocity):
    # get the filename and screenshot dimensions for the flow visualization
    flow_file = get_filename(directory_path, output_config, "flowFile",
                             False, "solutions/flow.png")
    screenshot_width = output_config["screenshotWidth"]\
        if "screenshotWidth" in output_config else 1024
    screenshot_height = output_config["screenshotHeight"]\
        if "screenshotHeight" in output_config else 1024

    # set a few options
    ps.set_program_name("potential flow demo")
    ps.set_verbosity(0)
    ps.set_use_prefs_file(False)
    ps.set_autocenter_structures(False)
    ps.set_ground_plane_mode("none")
    ps.set_SSAA_factor(4)

    # initialize polyscope
    ps.init()

    try:
        # set window size and camera view
        ps.set_window_size(screenshot_width, screenshot_height)
        set_camera_view(mesh_data)

        # register the mesh
        mesh_positions = wosx.convert_list_to_numpy_array(mesh_data.positions)
        mesh_indices = wosx.convert_list_to_numpy_array(mesh_data.indices)
        ps.register_surface_mesh("Mesh", mesh_positions, mesh_indices)

        # register the slice plane
        slice_plane = ps.register_surface_mesh("Slice Plane",
                                               slice_plane_data.positions,
                                               slice_plane_data.indices)
        slice_plane.set_color((1.0, 1.0, 1.0))
        slice_plane.set_material("flat")
        slice_plane.set_back_face_policy("cull")

        # plot perturbation potential on the slice plane
        perturbation_potential_colormap = output_config["perturbationPotentialColormap"]\
            if "perturbationPotentialColormap" in output_config else "coolwarm"
        perturbation_potential_colormap_min_val = output_config["perturbationPotentialColormapMinVal"]\
            if "perturbationPotentialColormapMinVal" in output_config else -1.0
        perturbation_potential_colormap_max_val = output_config["perturbationPotentialColormapMaxVal"]\
            if "perturbationPotentialColormapMaxVal" in output_config else 1.0
        slice_plane.add_scalar_quantity("Perturbation Potential", perturbation_potential,
                                        defined_on="faces", cmap=perturbation_potential_colormap,
                                        vminmax=(perturbation_potential_colormap_min_val,
                                        perturbation_potential_colormap_max_val), enabled=True)

        # plot flow velocity on the slice plane
        vector_length = 0.02 * freestream_state.speed / 10.0
        slice_plane.add_vector_quantity("Flow Velocity", flow_velocity,
                                        defined_on="faces", color=(0.0, 0.0, 0.0),
                                        enabled=True, length=vector_length)

        # save a screenshot
        output_directory = os.path.dirname(flow_file)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        ps.screenshot(flow_file, transparent_bg=False, include_UI=False)

    finally:
        # shutdown polyscope
        ps.shutdown()

################################################################################################################
# Demo entry point - load config, setup the problem, run the solver, save the solution,
# or optionally visualize the problem

def run_demo():
    # parse arguments
    parser = argparse.ArgumentParser(description="potential flow demo")
    parser.add_argument("--config", type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    try:
        # load the configuration file
        with open(args.config, 'r') as file:
            config = json.load(file)

        # load config settings
        wosx_directory_path = Path.cwd()
        demo_directory_path = wosx_directory_path / "demo_apps" / "potential_flow"
        wosx_directory_path_str = str(wosx_directory_path)
        demo_directory_path_str = str(demo_directory_path)
        problem_config = config["problem"]
        solver_config = config["solver"]
        output_config = config["output"]

        # setup a task handle to run the demo on the GPU
        device_backend = config["deviceBackend"]\
            if "deviceBackend" in config else "cuda"
        task_handle = wosx.GPUTaskHandle(wosx_directory_path_str,
                                         demo_directory_path_str,
                                         device_backend, dim=3, channels=1)

        # load mesh data
        mesh_data = load_mesh_data(demo_directory_path_str, problem_config)

        # setup a geometric queries object for the mesh and initialize
        # a task handle with it to determine the exterior of the mesh
        geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler =\
            setup_geometric_queries(mesh_data, False)
        task_handle.set_geometric_queries(geometric_queries)
        task_handle.init()

        # build a slice plane mesh and deactivate cells interior to the mesh
        slice_plane_data = SlicePlaneData()
        slice_plane_data.build(problem_config, mesh_data.bounding_box)
        slice_plane_data.deactivate_interior_cells(task_handle)

        # setup the freestream state
        speed = problem_config["freestreamSpeed"]\
            if "freestreamSpeed" in problem_config else 5.0
        angle = problem_config["freestreamAngle"]\
            if "freestreamAngle" in problem_config else 0.0
        freestream_state = FreestreamState(speed, angle)

        should_visualize_setup = output_config["visualizeSetup"]\
            if "visualizeSetup" in output_config else False
        if should_visualize_setup:
            # visualize the problem setup
            visualize_setup(mesh_data, slice_plane_data, freestream_state)

        else:
            # invert the mesh about the origin using the Kelvin transform;
            # here we assume the origin is contained within the mesh
            inversion_point = np.zeros(3, dtype=np.float32)
            kelvin_transform = wosx.Solvers.KelvinTransform(inversion_point, dim=3, channels=1)
            invert_mesh(kelvin_transform, mesh_data)

            # setup a geometric queries object for the inverted mesh
            inverted_geometric_queries, inverted_absorbing_boundary_handler,\
                inverted_reflecting_boundary_handler = setup_geometric_queries(mesh_data, True)

            # setup the demo PDE and the corresponding Kelvin transform PDE
            # for the inverted mesh
            pde = GPUPotentialFlowPDE(freestream_state)
            kelvin_pde = wosx.Core.GPUKelvinPDE(pde, inversion_point, dim=3, channels=1)

            # setup a reflecting boundary sampler (we do not require samplers
            # for the domain and/or absorbing boundary for this problem)
            inside_bounding_domain = mesh_data.get_inside_inverted_bounding_box_callback()
            boundary_sampling_weights = compute_boundary_sampling_weights(mesh_data, freestream_state)
            reflecting_boundary_sampler = wosx.Samplers.create_gpu_uniform_boundary_sampler(
                inverted_geometric_queries, mesh_data.inverted_positions,
                mesh_data.indices, boundary_sampling_weights,
                inside_bounding_domain, 0.0, False, dim=3)

            # reinitialize the task handle with the inverted geometric queries,
            # reflecting boundary sampler and Kelvin PDE
            task_handle.set_geometric_queries(inverted_geometric_queries)
            task_handle.set_reflecting_boundary_sampler(reflecting_boundary_sampler)
            task_handle.set_pde(kelvin_pde)
            task_handle.init()

            # create evaluation points to run the solver on; here we use
            # evaluation points on the slice plane in the exterior of the
            # input mesh (equivalently, the interior of the inverted mesh)
            eval_pts = create_evaluation_points(slice_plane_data, kelvin_transform)

            # run the solver
            eval_outputs = run_solver(solver_config, task_handle, eval_pts)

            # extract the solution and gradient
            perturbation_potential, flow_velocity = get_solution_and_gradient(
                slice_plane_data, kelvin_transform, eval_pts,
                eval_outputs, freestream_state.get_velocity())

            # save the estimated results
            save_solution_and_gradient(demo_directory_path_str, output_config,
                                       mesh_data, slice_plane_data, freestream_state,
                                       perturbation_potential, flow_velocity)

    except FileNotFoundError:
        sys.exit("Configuration file not found")

    except json.JSONDecodeError:
        sys.exit("Invalid configuration file")

    except ValueError as error:
        sys.exit(str(error))

if __name__ == "__main__":
    run_demo()
