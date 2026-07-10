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
This demo solves a thermal conduction problem on the surface of a Mars rover mesh
using WoSX. It loads the rover geometry, UVs, and grayscale textures that define
spatially varying Robin boundary values and coefficients, then traces camera rays
to select visible surface sample points. The GPU walk on stars solver estimates
the solution at those samples and writes the extracted values as images.
When 'visualizeSetup' is enabled in the config, Polyscope is used to inspect
the problem setup: geometry, robin boundary data, camera views, and sample points.
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
import matplotlib.pyplot as plt
from PIL import Image

################################################################################################################
# Utility structs and functions

class MeshData:
    def __init__(self):
        self.positions = wosx.Float3List()
        self.uvs = wosx.Float2List()
        self.indices = wosx.Int3List()
        self.uv_indices = wosx.Int3List()

        self.bounding_box = None
        self.box_positions = wosx.Float3List()
        self.box_indices = wosx.Int3List()

        self.robin_value = None
        self.robin_value_shape = None
        self.robin_coefficient = None
        self.robin_coefficient_shape = None

    def get_corner_uvs_as_numpy_array(self, flatten=True):
        uvs = np.asarray(wosx.convert_list_to_numpy_array(self.uvs))
        uv_indices = np.asarray(wosx.convert_list_to_numpy_array(self.uv_indices))
        if np.any((uv_indices < 0) | (uv_indices >= len(self.uvs))):
            raise ValueError("Invalid mesh UV index")

        return uvs[uv_indices.reshape(-1)] if flatten else uvs[uv_indices]

    def get_corner_uvs(self):
        corner_uvs = self.get_corner_uvs_as_numpy_array()
        return wosx.Float2List(np.ascontiguousarray(corner_uvs, dtype=np.float32))

    def compute_robin_coefficient_bounds_as_numpy_array(self):
        # load_image_buffer() stores data in transposed dense-grid order.
        # Transpose back here to match Image::get(uv, false) in app.cpp.
        robin_coefficient = self.robin_coefficient.reshape(int(self.robin_coefficient_shape[0]),
                                                           int(self.robin_coefficient_shape[1])).T
        h, w = robin_coefficient.shape

        corner_uvs = self.get_corner_uvs_as_numpy_array(False)
        i = np.clip((corner_uvs[:, :, 1] * h).astype(np.int64), 0, h - 1)
        j = np.clip((corner_uvs[:, :, 0] * w).astype(np.int64), 0, w - 1)
        values = robin_coefficient[i, j]

        min_values = np.ascontiguousarray(values.min(axis=1), dtype=np.float32)
        max_values = np.ascontiguousarray(values.max(axis=1), dtype=np.float32)

        return min_values, max_values

    def compute_robin_coefficient_bounds(self):
        min_values, max_values = self.compute_robin_coefficient_bounds_as_numpy_array()
        return wosx.FloatList(min_values), wosx.FloatList(max_values)

def get_filename(directory_path, config, filename,
                 is_required=True, default_value=""):
    if filename in config:
        return str(Path(directory_path) / config[filename])

    if is_required:
        raise ValueError(f"Missing required config entry: {filename}")

    return str(Path(directory_path) / default_value)

def load_image_buffer(image_file, channels):
    # PIL does not support pfm files, so default to png
    if image_file.endswith(".pfm"):
        image_file = image_file.replace(".pfm", ".png")

    # load image and convert to the expected channel layout
    image = Image.open(image_file)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image = image.transpose(Image.Transpose.TRANSPOSE)
    image_shape = np.array([image.height, image.width], dtype=np.int32)
    if channels == 1:
        image = image.convert("L")
        image = np.array(image).flatten().astype(np.float32) / 255.0
    else:
        image = image.convert("RGBA")
        image = np.array(image).reshape(-1, channels).astype(np.float32) / 255.0

    return image, image_shape

def save_image_buffer(output_config, image_file, image_buffer, channels):
    # load the output configuration
    colormap = output_config["colormap"]\
        if "colormap" in output_config else "turbo"
    colormap_min_val = output_config["colormapMinVal"]\
        if "colormapMinVal" in output_config else 0.0
    colormap_max_val = output_config["colormapMaxVal"]\
        if "colormapMaxVal" in output_config else 1.0
    save_colormapped = output_config["saveColormapped"]\
        if "saveColormapped" in output_config else True

    # PIL does not support pfm files, so default to png
    if image_file.endswith(".pfm"):
        image_file = image_file.replace(".pfm", ".png")

    output_directory = os.path.dirname(image_file)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    # save image
    image = np.clip(image_buffer, 0.0, 1.0)
    output_image = (image * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.save(image_file)

    # save colormapped image
    if channels == 1 and save_colormapped:
        image = np.clip((image_buffer - colormap_min_val) / (colormap_max_val - colormap_min_val), 0.0, 1.0)
        cmap = plt.get_cmap(colormap)
        colormapped_image = cmap(image, bytes=True)
        colormapped_image = np.clip(colormapped_image[:, :, :3].astype(np.uint8), 0, 255)
        colormapped_image = Image.fromarray(colormapped_image)
        base, ext = os.path.splitext(image_file)
        colormap_file = f"{base}_color{ext}"
        colormapped_image.save(colormap_file)

def save_solution_images(directory_path, problem_config, output_config,
                         sample_index_map, solution):
    camera_width = problem_config["cameraWidth"]\
        if "cameraWidth" in problem_config else 512
    camera_height = problem_config["cameraHeight"]\
        if "cameraHeight" in problem_config else 512
    solution_file = get_filename(directory_path, output_config,
                                 "solutionFile", False,
                                 "solutions/solution.png")

    solution_image = np.zeros((camera_height, camera_width), dtype=np.float32)
    for i, value in enumerate(solution):
        index = sample_index_map[i]
        row = index // camera_width
        col = index % camera_width
        solution_image[row, col] = value

    save_image_buffer(output_config, solution_file, solution_image, 1)

################################################################################################################
# Problem specification - load mesh data, generate rays from a camera view, and
# setup the GPU geometric queries and PDE objects

def load_mesh_data(directory_path, problem_config):
    # load the mesh
    geometry_filename = get_filename(directory_path, problem_config, "geometry")
    mesh_data = MeshData()
    wosx.Utils.load_textured_boundary_mesh(geometry_filename, mesh_data.positions, mesh_data.uvs,
                                           mesh_data.indices, mesh_data.uv_indices, dim=3)
    wosx.Utils.normalize(mesh_data.positions, dim=3)

    # compute a bounding box for the mesh
    mesh_data.bounding_box = wosx.Utils.compute_bounding_box(mesh_data.positions, True, 1.25, dim=3)
    wosx.Utils.build_bounding_box_mesh(mesh_data.bounding_box[0], mesh_data.bounding_box[1],
                                       mesh_data.box_positions, mesh_data.box_indices, dim=3)

    # load the robin textures
    robin_value_filename =\
        get_filename(directory_path, problem_config, "robinValueTexture")
    robin_coefficient_filename =\
        get_filename(directory_path, problem_config, "robinCoefficientTexture")
    mesh_data.robin_value, mesh_data.robin_value_shape =\
        load_image_buffer(robin_value_filename, 1)
    mesh_data.robin_coefficient, mesh_data.robin_coefficient_shape =\
        load_image_buffer(robin_coefficient_filename, 1)

    # invert the robin coefficient texture
    mesh_data.robin_coefficient = 1.0 - mesh_data.robin_coefficient

    return mesh_data

def generate_rays_from_camera_view(view_mat, fov, camera_width, camera_height):
    # setup the camera parameters and generate camera rays
    aspect_ratio = float(camera_width) / float(camera_height)
    camera_parameters = ps.CameraParameters(
        ps.CameraIntrinsics(fov_vertical_deg=fov, aspect=aspect_ratio),
        ps.CameraExtrinsics(mat=np.asarray(view_mat, dtype=np.float32)))

    camera_position = camera_parameters.get_position().astype(np.float32)
    camera_ray_directions = camera_parameters.generate_camera_rays(
        (camera_width, camera_height), image_origin="upper_left")

    # populate the ray origins and directions
    directions = np.ascontiguousarray(camera_ray_directions.reshape(-1, 3),
                                      dtype=np.float32)
    origins = np.ascontiguousarray(np.repeat(camera_position[None, :],
                                             directions.shape[0], axis=0),
                                   dtype=np.float32)

    return wosx.Float3List(origins), wosx.Float3List(directions)

def generate_rays_from_camera_view_file(directory_path, problem_config):
    # parse the camera view from file
    camera_view_filename = get_filename(directory_path, problem_config,
                                        "cameraView", False,
                                        "data/camera_view_0.json")
    with open(camera_view_filename, "r") as camera_view_file:
        camera_view = json.load(camera_view_file)

    # get the camera view matrix, fov and image resolution
    view_mat = np.asarray(camera_view["viewMat"], dtype=np.float32).reshape(4, 4)
    fov = float(camera_view["fov"])
    camera_width = problem_config["cameraWidth"]\
        if "cameraWidth" in problem_config else 512
    camera_height = problem_config["cameraHeight"]\
        if "cameraHeight" in problem_config else 512

    # generate rays from the camera view
    return generate_rays_from_camera_view(view_mat, fov, camera_width, camera_height)

def setup_geometric_queries(mesh_data):
    # setup an absorbing boundary handler for the mesh's bounding box
    absorbing_boundary_handler = wosx.Utils.GPUFcpwDirichletBoundaryHandler(
        mesh_data.box_positions, mesh_data.box_indices, dim=3)

    # setup a reflecting boundary handler for the mesh
    ignore_candidate_silhouette = wosx.Utils.get_ignore_candidate_silhouette_callback(True)
    min_robin_coeff_values, max_robin_coeff_values = mesh_data.compute_robin_coefficient_bounds()
    reflecting_boundary_handler = wosx.Utils.GPUFcpwRobinBoundaryHandler(
        mesh_data.positions, mesh_data.indices, ignore_candidate_silhouette,
        min_robin_coeff_values, max_robin_coeff_values, dim=3)

    # create a geometric queries object from the handlers
    geometric_queries = wosx.Core.GPUGeometricQueries(
        absorbing_boundary_handler, reflecting_boundary_handler,
        mesh_data.bounding_box[0], mesh_data.bounding_box[1], False, dim=3)

    return geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler

class GPUThermalConductionPDE(wosx.Core.GPUPDE):
    def __init__(self, acceleration_structure, mesh_data):
        super().__init__()
        self._acceleration_structure = acceleration_structure
        self._corner_uvs_data = mesh_data.get_corner_uvs()
        self._corner_uvs = wosx.Utils.GPUFloat2Buffer(False)
        self._sampler = wosx.Utils.GPUSampler()

        uv_min = np.zeros(2, dtype=np.float32)
        uv_max = np.ones(2, dtype=np.float32)
        self._robin_value = wosx.Utils.GPUDenseGrid(self._sampler, mesh_data.robin_value,
                                                    mesh_data.robin_value_shape,
                                                    uv_min, uv_max, False, dim=2, channels=1)
        self._robin_coefficient = wosx.Utils.GPUDenseGrid(self._sampler, mesh_data.robin_coefficient,
                                                          mesh_data.robin_coefficient_shape,
                                                          uv_min, uv_max, False, dim=2, channels=1)

    def allocate(self, context):
        self._corner_uvs.allocate(context, self._corner_uvs_data)
        self._sampler.allocate(context, wosx.Utils.GPUTextureFilteringMode.Linear,
                               wosx.Utils.GPUTextureAddressingMode.ClampToEdge)
        self._robin_value.allocate(context)
        self._robin_coefficient.allocate(context)

    def set_resources(self, cursor, print_logs):
        self._acceleration_structure.set_resources(cursor["mAccelerationStructure"], print_logs)
        self._corner_uvs.set_binding(cursor["mCornerUVs"])
        self._robin_value.set_resources(cursor["mRobinValue"], print_logs)
        self._robin_coefficient.set_resources(cursor["mRobinCoefficient"], print_logs)
        if print_logs:
            wosx.Utils.print_gpu_reflection_info(cursor, 4, self.get_reflection_type())

    def get_reflection_type(self):
        return "ThermalConductionPDE"

    def get_type(self):
        return wosx.Core.GPUPDEType.Poisson

################################################################################################################
# Walk on stars solver - create sample points, run the solver and extract the solution

def create_sample_points(origins, directions, task_handle):
    # run intersection queries
    dist_along_ray = wosx.FloatList()
    intersection_points = wosx.Float3List()
    intersection_normals = wosx.Float3List()
    distance_queries = wosx.Utils.GPUDistanceQueries(task_handle, dim=3, channels=1)
    distance_queries.intersect_reflecting_boundary(origins, directions, dist_along_ray,
                                                   intersection_points, intersection_normals)

    # create the sample points
    sample_pts = []
    sample_index_map = []

    for i in range(len(origins)):
        if dist_along_ray[i] >= 0.0:
            sample_pt = wosx.Solvers.GPUSamplePoint(dim=3)
            sample_pt.pt = wosx.GPUFloat3(intersection_points[i][0],
                                          intersection_points[i][1],
                                          intersection_points[i][2])
            sample_pt.normal = wosx.GPUFloat3(intersection_normals[i][0],
                                              intersection_normals[i][1],
                                              intersection_normals[i][2])
            sample_pt.type = wosx.Solvers.SampleType.OnReflectingBoundary
            sample_pt.estimation_quantity = wosx.Solvers.EstimationQuantity.Solution
            sample_pts.append(sample_pt)
            sample_index_map.append(i)

    return wosx.Solvers.GPUSamplePointList(sample_pts, dim=3), sample_index_map

def run_solver(solver_config, task_handle, sample_pts):
    # load config settings for the walk on stars solver
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.99

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
    walk_settings.russian_roulette_threshold = russian_roulette_threshold
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = steps_before_using_maximal_spheres
    walk_settings.solve_double_sided = 1
    walk_settings.ignore_absorbing_boundary_contribution = 0
    walk_settings.ignore_reflecting_boundary_contribution = 0
    walk_settings.ignore_source_contribution = 1

    # initialize the solver
    walk_on_stars = wosx.Solvers.GPUWalkOnStarsSolver(task_handle, walk_settings, n_resident_threads,
                                                      not disable_persistent_threads, print_logs,
                                                      dim=3, channels=1)

    # run the solver
    if disable_persistent_threads:
        # compute workload parameters--the number of sample copies and number of
        # walks per copy--to help improve GPU occupancy when running the solver
        workload_parameters = walk_on_stars.compute_workload_parameters(len(sample_pts), n_walks)

        # populate sample points on the GPU
        walk_on_stars.populate_sample_points(sample_pts, workload_parameters[0])

        # solve
        walk_on_stars.solve(1, workload_parameters[1])

    else:
        # populate sample points on the GPU
        walk_on_stars.populate_sample_points(sample_pts)

        # solve
        walk_on_stars.solve(n_walks)

    # extract sample statistics from the GPU
    sample_statistics = wosx.Solvers.GPUSampleStatisticsList(dim=3, channels=1)
    walk_on_stars.get_sample_statistics(sample_statistics)

    return sample_statistics

def get_solution(sample_statistics):
    solution = np.zeros(len(sample_statistics))
    for i in range(len(sample_statistics)):
        solution[i] = sample_statistics[i].get_estimated_solution()

    return solution

################################################################################################################
# Visualization - read and write camera views, plot sample points and set the GUI callback

def set_camera_view(camera_view_filename):
    with open(camera_view_filename, "r") as camera_view_file:
        json_data = camera_view_file.read()
    ps.set_view_from_json(json_data, False)

def write_camera_view(camera_view_filename):
    with open(camera_view_filename, "w") as camera_view_file:
        camera_view_file.write(ps.get_view_as_json())
        camera_view_file.write("\n")

    print(f"Wrote camera view to: {camera_view_filename}")

def plot_sample_points(sample_pts, enabled=False):
    points = np.zeros((len(sample_pts), 3), dtype=np.float32)
    normals = np.zeros((len(sample_pts), 3), dtype=np.float32)

    for i, sample_pt in enumerate(sample_pts):
        points[i, :] = [sample_pt.pt.x, sample_pt.pt.y, sample_pt.pt.z]
        normals[i, :] = [sample_pt.normal.x, sample_pt.normal.y,
                         sample_pt.normal.z]

    point_cloud = ps.register_point_cloud("Sample Points", points)
    point_cloud.add_vector_quantity("Normals", normals)
    point_cloud.set_enabled(enabled)

def gui_callback(directory_path, problem_config, output_config,
                 task_handle, sample_pts):
    if psim.Button("Generate sample points"):
        # generate rays from the current camera view
        camera_width = problem_config["cameraWidth"]\
            if "cameraWidth" in problem_config else 512
        camera_height = problem_config["cameraHeight"]\
            if "cameraHeight" in problem_config else 512
        origins, directions = generate_rays_from_camera_view(
            ps.get_camera_view_matrix(), ps.get_vertical_fov_degrees(),
            camera_width, camera_height)

        # create and plot sample points
        new_sample_pts, _ = create_sample_points(origins, directions, task_handle)
        sample_pts.clear()
        sample_pts.extend(new_sample_pts)
        plot_sample_points(sample_pts, True)

    if psim.Button("Write camera view"):
        camera_view_filename = get_filename(directory_path, output_config,
                                            "cameraView", False,
                                            "data/camera_view_out.json")
        write_camera_view(camera_view_filename)

def visualize_setup(directory_path, problem_config, output_config,
                    mesh_data, task_handle, sample_pts):
    # set a few options
    ps.set_program_name("thermal conduction demo - problem setup")
    ps.set_verbosity(0)
    ps.set_use_prefs_file(False)
    ps.set_autocenter_structures(False)
    ps.set_ground_plane_mode("none")

    # initialize polyscope
    ps.init()

    # set the camera view from file
    camera_view_filename = get_filename(directory_path, problem_config,
                                        "cameraView", False,
                                        "data/camera_view_0.json")
    set_camera_view(camera_view_filename)

    # register the mesh and its robin coefficients and values
    positions = np.asarray(wosx.convert_list_to_numpy_array(mesh_data.positions))
    indices = np.asarray(wosx.convert_list_to_numpy_array(mesh_data.indices))
    mesh = ps.register_surface_mesh("Mesh", positions, indices)

    corner_uvs = mesh_data.get_corner_uvs_as_numpy_array()
    mesh.add_parameterization_quantity("UVs", corner_uvs, defined_on="corners")

    robin_value = mesh_data.robin_value.reshape(int(mesh_data.robin_value_shape[0]),
                                                int(mesh_data.robin_value_shape[1])).T
    mesh.add_scalar_quantity("Robin Value", robin_value, defined_on="texture",
                             param_name="UVs", image_origin="lower_left",
                             cmap="plasma", enabled=True)

    robin_coefficient = mesh_data.robin_coefficient.reshape(int(mesh_data.robin_coefficient_shape[0]),
                                                            int(mesh_data.robin_coefficient_shape[1])).T
    mesh.add_scalar_quantity("Robin Coefficient", robin_coefficient, defined_on="texture",
                             param_name="UVs", image_origin="lower_left",
                             cmap="viridis", enabled=False)

    min_robin_coeff_values, max_robin_coeff_values =\
        mesh_data.compute_robin_coefficient_bounds_as_numpy_array()
    mesh.add_scalar_quantity("Min Robin Coefficient", min_robin_coeff_values,
                             defined_on="faces", cmap="viridis", enabled=False)
    mesh.add_scalar_quantity("Max Robin Coefficient", max_robin_coeff_values,
                             defined_on="faces", cmap="viridis", enabled=False)

    # plot the sample points
    plot_sample_points(sample_pts)

    # set the GUI callback
    ps.set_user_callback(lambda: gui_callback(directory_path, problem_config, output_config,
                                              task_handle, sample_pts))

    # give control to polyscope gui
    try:
        ps.show()
    finally:
        ps.set_user_callback(None)

################################################################################################################
# Demo entry point - load config, setup the problem, run the solver, save the solution,
# or optionally visualize the problem setup

def run_demo():
    # parse arguments
    parser = argparse.ArgumentParser(description="thermal conduction demo")
    parser.add_argument("--config", type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    try:
        # load the configuration file
        with open(args.config, 'r') as file:
            config = json.load(file)

        # load config settings
        wosx_directory_path = Path.cwd()
        demo_directory_path = wosx_directory_path / "demo_apps" / "thermal_conduction"
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
        pde = GPUThermalConductionPDE(reflecting_boundary_handler, mesh_data)

        # setup a task handle to run the demo on the GPU
        device_backend = config["deviceBackend"]\
            if "deviceBackend" in config else "cuda"
        task_handle = wosx.GPUTaskHandle(wosx_directory_path_str,
                                         demo_directory_path_str,
                                         device_backend, dim=3, channels=1)
        task_handle.set_geometric_queries(geometric_queries)
        task_handle.set_pde(pde)
        task_handle.init()

        # generate rays from a camera view
        origins, directions = generate_rays_from_camera_view_file(demo_directory_path_str,
                                                                  problem_config)

        # create sample points to run the solver on
        sample_pts, sample_index_map = create_sample_points(origins, directions, task_handle)

        should_visualize_setup = output_config["visualizeSetup"]\
            if "visualizeSetup" in output_config else False
        if should_visualize_setup:
            visualize_setup(demo_directory_path_str, problem_config, output_config,
                            mesh_data, task_handle, sample_pts)

        else:
            # run the solver
            sample_statistics = run_solver(solver_config, task_handle, sample_pts)

            # extract the solution and save it as images
            solution = get_solution(sample_statistics)
            save_solution_images(demo_directory_path_str, problem_config,
                                 output_config, sample_index_map, solution)

    except FileNotFoundError:
        sys.exit("Configuration file not found")

    except json.JSONDecodeError:
        sys.exit("Invalid configuration file")

    except ValueError as error:
        sys.exit(str(error))

if __name__ == "__main__":
    run_demo()
