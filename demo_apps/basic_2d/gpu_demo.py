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
This file is the entry point for a 2D GPU demo application demonstrating how to use WoSX.
It reads a 'model problem' description from a JSON file, runs the WalkOnSpheres, WalkOnStars
or BoundaryValueCaching solvers, and saves the result as a PNG file.

The PDE is defined via a custom Python subclass of wosx.Core.GPUPDE, which implements
the allocate(), set_resources(), get_reflection_type(), and get_type() methods. The
corresponding Slang shader definition is in wosx-pde-definition.slang.

The full WoSX GPU API, including treatment of 3D domains and/or vector-valued PDEs, can be viewed
using the following commands in the Python console:
>>> import wosx
>>> help(wosx)
'''

import argparse
import json
import sys
import os
import numpy as np
import wosx
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

##############################################################################################
# Grid generation and image I/O utility functions

def create_grid_points(output_config, bounding_box):
    grid_res = output_config["gridRes"]
    grid_min = bounding_box[0]
    grid_max = bounding_box[1]
    grid_points = np.zeros((grid_res * grid_res, 2))
    extent = grid_max - grid_min

    for i in range(grid_res):
        for j in range(grid_res):
            index = i * grid_res + j
            grid_points[index][0] = (i / float(grid_res)) * extent[0] + grid_min[0]
            grid_points[index][1] = (j / float(grid_res)) * extent[1] + grid_min[1]

    return grid_points

def create_grid_values(output_config, distance_info, values, channels):
    grid_res = output_config["gridRes"]
    boundary_dist_mask = output_config["boundaryDistanceMask"]\
        if "boundaryDistanceMask" in output_config else 0.0
    grid_values = np.zeros((grid_res, grid_res))\
        if channels == 1 else np.zeros((grid_res, grid_res, channels))

    for i in range(grid_res):
        for j in range(grid_res):
            index = i * grid_res + j
            in_valid_solve_region = distance_info[index][0]
            dist_to_boundary = min(distance_info[index][1], distance_info[index][2])

            if in_valid_solve_region and dist_to_boundary > boundary_dist_mask:
                grid_values[j][i] = values[index]

    return grid_values

def load_image_buffer(image_file, channels):
    # PIL does not support pfm files, so default to png
    if image_file.endswith(".pfm"):
        image_file = image_file.replace(".pfm", ".png")

    # load image and convert to the expected channel layout
    image = Image.open(image_file)
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

##############################################################################################
# PDE definition - Python subclass of wosx.Core.GPUPDE
#
# The PDE is defined in a separate wosx-pde-definition.slang file. The necessary
# resources needed for the PDE, e.g., the source and boundary texture data, are
# allocated on the GPU using the GPUModelProblemPDE class below.

class GPUModelProblemPDE(wosx.Core.GPUPDE):
    def __init__(self, absorption_coeff, robin_coeff, grid_min, grid_max,
                 source_value_buffer, source_value_shape,
                 absorbing_boundary_value_buffer,
                 absorbing_boundary_normal_aligned_value_buffer,
                 absorbing_boundary_value_shape,
                 reflecting_boundary_value_buffer,
                 reflecting_boundary_normal_aligned_value_buffer,
                 reflecting_boundary_value_shape, dim, channels):
        super().__init__()
        self._absorption_coeff = absorption_coeff
        self._robin_coeff = robin_coeff
        self._sampler = wosx.Utils.GPUSampler()

        self._source_value = wosx.Utils.GPUDenseGrid(
            self._sampler, source_value_buffer, source_value_shape,
            grid_min, grid_max, False, dim=dim, channels=channels)
        self._absorbing_boundary_value = wosx.Utils.GPUDenseGrid(
            self._sampler, absorbing_boundary_value_buffer,
            absorbing_boundary_value_shape, grid_min, grid_max, False,
            dim=dim, channels=channels)
        self._reflecting_boundary_value = wosx.Utils.GPUDenseGrid(
            self._sampler, reflecting_boundary_value_buffer,
            reflecting_boundary_value_shape, grid_min, grid_max, False,
            dim=dim, channels=channels)
        self._absorbing_boundary_normal_aligned_value = None
        if absorbing_boundary_normal_aligned_value_buffer is not None:
            self._absorbing_boundary_normal_aligned_value = wosx.Utils.GPUDenseGrid(
                self._sampler, absorbing_boundary_normal_aligned_value_buffer,
                absorbing_boundary_value_shape, grid_min, grid_max, False,
                dim=dim, channels=channels)
        self._reflecting_boundary_normal_aligned_value = None
        if reflecting_boundary_normal_aligned_value_buffer is not None:
            self._reflecting_boundary_normal_aligned_value = wosx.Utils.GPUDenseGrid(
                self._sampler, reflecting_boundary_normal_aligned_value_buffer,
                reflecting_boundary_value_shape, grid_min, grid_max, False,
                dim=dim, channels=channels)

    def allocate(self, context):
        self._sampler.allocate(context, wosx.Utils.GPUTextureFilteringMode.Linear,
                               wosx.Utils.GPUTextureAddressingMode.ClampToEdge)
        self._source_value.allocate(context)
        self._absorbing_boundary_value.allocate(context)
        self._reflecting_boundary_value.allocate(context)
        if self._absorbing_boundary_normal_aligned_value is not None:
            self._absorbing_boundary_normal_aligned_value.allocate(context)
        if self._reflecting_boundary_normal_aligned_value is not None:
            self._reflecting_boundary_normal_aligned_value.allocate(context)

    def set_resources(self, cursor, print_logs):
        n_resources = 4
        if self._absorption_coeff > 0:
            cursor["mAbsorptionCoeff"].set_data_float(self._absorption_coeff)
            n_resources += 1
        cursor["mRobinCoeff"].set_data_float(self._robin_coeff)
        self._source_value.set_resources(cursor["mSourceValue"], print_logs)
        self._absorbing_boundary_value.set_resources(cursor["mAbsorbingBoundaryValue"], print_logs)
        self._reflecting_boundary_value.set_resources(cursor["mReflectingBoundaryValue"], print_logs)
        if self._absorbing_boundary_normal_aligned_value is not None:
            self._absorbing_boundary_normal_aligned_value.set_resources(
                cursor["mAbsorbingBoundaryNormalAlignedValue"], print_logs)
            n_resources += 1
        if self._reflecting_boundary_normal_aligned_value is not None:
            self._reflecting_boundary_normal_aligned_value.set_resources(
                cursor["mReflectingBoundaryNormalAlignedValue"], print_logs)
            n_resources += 1
        if print_logs:
            wosx.Utils.print_gpu_reflection_info(cursor, n_resources, self.get_reflection_type())

    def get_reflection_type(self):
        return "ModelProblemPDE"

    def get_type(self):
        if self._absorption_coeff > 0:
            return wosx.Core.GPUPDEType.ScreenedPoisson
        return wosx.Core.GPUPDEType.Poisson

##############################################################################################
# Problem specification - geometry loading and PDE setup

def get_channels(model_problem_config):
    channels = model_problem_config["channels"]\
        if "channels" in model_problem_config else 1

    if channels not in (1, 4):
        raise ValueError("channels must be 1 or 4")

    return channels

def load_boundary_mesh(model_problem_config, dim, normalize=True, flip_orientation=True):
    # load the model problem configuration
    obj_file = model_problem_config["geometry"]

    # load obj file, and optionally normalize and flip mesh orientation
    positions = wosx.FloatNList(dim=dim)
    indices = wosx.IntNList(dim=dim)
    wosx.Utils.load_boundary_mesh(obj_file, positions, indices, dim=dim)

    if normalize:
        wosx.Utils.normalize(positions, dim=dim)

    if flip_orientation:
        wosx.Utils.flip_orientation(indices, dim=dim)

    # compute the bounding box for the domain
    bounding_box = wosx.Utils.compute_bounding_box(positions, True, 1.0, dim=dim)

    return positions, indices, bounding_box

def setup_pde(model_problem_config, bounding_box, dim, channels):
    # load the model problem configuration
    robin_coeff = model_problem_config["robinCoeff"]\
        if "robinCoeff" in model_problem_config else 0.0
    absorption_coeff = model_problem_config["absorptionCoeff"]\
        if "absorptionCoeff" in model_problem_config else 0.0
    solve_double_sided = model_problem_config["solveDoubleSided"]\
        if "solveDoubleSided" in model_problem_config else False
    solve_exterior = model_problem_config["solveExterior"]\
        if "solveExterior" in model_problem_config else False
    source_value_buffer, source_value_shape =\
        load_image_buffer(model_problem_config["sourceValue"], channels)
    absorbing_boundary_value_buffer, absorbing_boundary_value_shape =\
        load_image_buffer(model_problem_config["absorbingBoundaryValue"], channels)
    reflecting_boundary_value_buffer, reflecting_boundary_value_shape =\
        load_image_buffer(model_problem_config["reflectingBoundaryValue"], channels)
    absorbing_boundary_normal_aligned_value_buffer = None
    reflecting_boundary_normal_aligned_value_buffer = None
    if solve_double_sided:
        absorbing_boundary_normal_aligned_value_buffer, _ =\
            load_image_buffer(model_problem_config["absorbingBoundaryNormalAlignedValue"], channels)
        reflecting_boundary_normal_aligned_value_buffer, _ =\
            load_image_buffer(model_problem_config["reflectingBoundaryNormalAlignedValue"], channels)
    is_reflecting_boundary_buffer = None
    is_reflecting_boundary_shape = None
    if solve_exterior:
        absorption_coeff = 0.0 # kelvin transform requires absorption coefficient to be 0
        is_reflecting_boundary_buffer, is_reflecting_boundary_shape =\
            load_image_buffer(model_problem_config["reflectingBoundaryValue"], 1)
    else:
        is_reflecting_boundary_buffer, is_reflecting_boundary_shape =\
            load_image_buffer(model_problem_config["isReflectingBoundary"], 1)
    domain_min = bounding_box[0]
    domain_max = bounding_box[1]

    # setup the PDE
    pde = GPUModelProblemPDE(absorption_coeff, robin_coeff, domain_min, domain_max,
                             source_value_buffer, source_value_shape,
                             absorbing_boundary_value_buffer,
                             absorbing_boundary_normal_aligned_value_buffer,
                             absorbing_boundary_value_shape,
                             reflecting_boundary_value_buffer,
                             reflecting_boundary_normal_aligned_value_buffer,
                             reflecting_boundary_value_shape, dim, channels)
    has_reflecting_boundary_conditions = wosx.Utils.get_dense_grid_indicator_callback(
        is_reflecting_boundary_buffer, is_reflecting_boundary_shape,
        domain_min, domain_max, dim=dim)

    return pde, has_reflecting_boundary_conditions

def partition_boundary_mesh(has_reflecting_boundary_conditions, positions, indices, dim):
    # use WoSX's default partitioning function, which assumes the boundary discretization
    # is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    # and the user might want to override this function for their specific problem
    absorbing_boundary_positions = wosx.FloatNList(dim=dim)
    absorbing_boundary_indices = wosx.IntNList(dim=dim)
    reflecting_boundary_positions = wosx.FloatNList(dim=dim)
    reflecting_boundary_indices = wosx.IntNList(dim=dim)
    wosx.Utils.partition_boundary_mesh(has_reflecting_boundary_conditions, positions, indices,
                                       absorbing_boundary_positions, absorbing_boundary_indices,
                                       reflecting_boundary_positions, reflecting_boundary_indices,
                                       dim=dim)

    return absorbing_boundary_positions, absorbing_boundary_indices,\
           reflecting_boundary_positions, reflecting_boundary_indices

def setup_geometric_queries(model_problem_config, bounding_box,
                            absorbing_boundary_positions, absorbing_boundary_indices,
                            reflecting_boundary_positions, reflecting_boundary_indices,
                            min_robin_coeff_values, max_robin_coeff_values,
                            are_robin_conditions_pure_neumann, solve_double_sided, dim):
    # load the model problem configuration
    domain_is_watertight = model_problem_config["domainIsWatertight"]\
        if "domainIsWatertight" in model_problem_config else False

    # create the absorbing boundary handler
    absorbing_boundary_handler = None
    if len(absorbing_boundary_positions) > 0:
        absorbing_boundary_handler = wosx.Utils.GPUFcpwDirichletBoundaryHandler(
            absorbing_boundary_positions, absorbing_boundary_indices, dim=dim)

    else:
        absorbing_boundary_handler = wosx.Core.GPUEmptyAbsorbingBoundaryHandler()

    # create the reflecting boundary handler
    reflecting_boundary_handler = None
    if len(reflecting_boundary_positions) > 0:
        ignore_candidate_silhouette = wosx.Utils.get_ignore_candidate_silhouette_callback(solve_double_sided)
        if are_robin_conditions_pure_neumann:
            reflecting_boundary_handler = wosx.Utils.GPUFcpwNeumannBoundaryHandler(
                reflecting_boundary_positions, reflecting_boundary_indices,
                ignore_candidate_silhouette, dim=dim)

        else:
            reflecting_boundary_handler = wosx.Utils.GPUFcpwRobinBoundaryHandler(
                reflecting_boundary_positions, reflecting_boundary_indices, ignore_candidate_silhouette,
                min_robin_coeff_values, max_robin_coeff_values, dim=dim)

    else:
        reflecting_boundary_handler = wosx.Core.GPUEmptyReflectingBoundaryHandler()

    # create the geometric queries object
    geometric_queries = wosx.Core.GPUGeometricQueries(
        absorbing_boundary_handler, reflecting_boundary_handler,
        bounding_box[0], bounding_box[1], domain_is_watertight, dim=dim)

    return geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler

def compute_distance_info(device_backend, solve_locations, geometric_queries,
                          domain_is_watertight, solve_double_sided,
                          solve_exterior, dim, channels):
    # create GPU task handle for distance computation
    wosx_directory_path = str(Path.cwd())
    task_handle = wosx.GPUTaskHandle(wosx_directory_path, "", device_backend,
                                     dim=dim, channels=channels)
    task_handle.set_geometric_queries(geometric_queries)
    task_handle.init()

    # compute distance info on the GPU
    dist_to_absorbing_boundary = wosx.FloatList()
    dist_to_reflecting_boundary = wosx.FloatList()
    distance_queries = wosx.Utils.GPUDistanceQueries(task_handle, dim=dim, channels=channels)
    distance_queries.compute_dist_to_boundary(solve_locations,
                                              dist_to_absorbing_boundary,
                                              dist_to_reflecting_boundary)

    # determine solve locations in the valid solve region
    distance_info = [None] * len(solve_locations)

    for i in range(len(solve_locations)):
        d1 = abs(dist_to_absorbing_boundary[i])
        d2 = abs(dist_to_reflecting_boundary[i])
        dist_to_boundary = dist_to_absorbing_boundary[i] if d1 < d2 else dist_to_reflecting_boundary[i]
        inside_domain = True if not domain_is_watertight else dist_to_boundary < 0.0
        if domain_is_watertight and solve_exterior:
            inside_domain = not inside_domain
        in_valid_solve_region = inside_domain or solve_double_sided
        distance_info[i] = (in_valid_solve_region, d1, d2)

    return distance_info

##############################################################################################
# Exterior problem utilities - uses a Kelvin transform to convert an exterior problem
# into an equivalent interior problem with a modified PDE on the inverted domain

def invert_exterior_problem(kelvin_transform, positions, absorbing_boundary_positions,
                            reflecting_boundary_positions, reflecting_boundary_indices,
                            pde, robin_coeff, dim, channels):
    # invert the domain
    inverted_positions = wosx.FloatNList(dim=dim)
    inverted_absorbing_boundary_positions = wosx.FloatNList(dim=dim)
    inverted_reflecting_boundary_positions = wosx.FloatNList(dim=dim)
    kelvin_transform.transform_points(positions, inverted_positions)
    kelvin_transform.transform_points(absorbing_boundary_positions, inverted_absorbing_boundary_positions)
    kelvin_transform.transform_points(reflecting_boundary_positions, inverted_reflecting_boundary_positions)

    # compute the bounding box for the inverted domain
    inverted_bounding_box = wosx.Utils.compute_bounding_box(inverted_positions, True, 1.0, dim=dim)

    # setup the modified PDE on the inverted domain
    pde_inverted_domain = wosx.Core.GPUKelvinPDE(pde, kelvin_transform.get_origin(),
                                                 dim=dim, channels=channels)

    # compute the modified Robin coefficients on the inverted domain
    min_robin_coeff_values_inverted_domain = wosx.FloatList()
    max_robin_coeff_values_inverted_domain = wosx.FloatList()
    if robin_coeff != 0.0:
        min_robin_coeff_values = wosx.FloatList([robin_coeff] * len(reflecting_boundary_indices))
        max_robin_coeff_values = wosx.FloatList([robin_coeff] * len(reflecting_boundary_indices))
        kelvin_transform.compute_robin_coefficients(inverted_reflecting_boundary_positions,
                                                    reflecting_boundary_indices,
                                                    min_robin_coeff_values, max_robin_coeff_values,
                                                    min_robin_coeff_values_inverted_domain,
                                                    max_robin_coeff_values_inverted_domain)

    return inverted_bounding_box, inverted_positions, inverted_absorbing_boundary_positions,\
           inverted_reflecting_boundary_positions, pde_inverted_domain,\
           min_robin_coeff_values_inverted_domain, max_robin_coeff_values_inverted_domain

def invert_solve_locations(kelvin_transform, solve_locations, dim):
    inverted_solve_locations = np.zeros((len(solve_locations), dim))
    for i in range(len(solve_locations)):
        inverted_solve_locations[i] = kelvin_transform.transform_point(solve_locations[i])

    return inverted_solve_locations

def create_solution_values(count, channels):
    return np.zeros(count) if channels == 1 else np.zeros((count, channels))

def get_solution_value(solution, channels):
    if channels == 1:
        return solution

    return np.asarray(solution, dtype=np.float32).reshape(channels)

def compute_exterior_solution(kelvin_transform, interior_solution,
                              inverted_solve_locations, channels):
    exterior_solution = create_solution_values(len(interior_solution), channels)
    for i in range(len(interior_solution)):
        solution = kelvin_transform.transform_solution_estimate(interior_solution[i],
                                                                inverted_solve_locations[i])
        exterior_solution[i] = get_solution_value(solution, channels)

    return exterior_solution

##############################################################################################
# Walk on Spheres and Walk on Stars solvers - note that the latter is a strict generalization
# of the former, and reduces to it when the PDE only has Dirichlet boundary conditions

def create_sample_points(solve_locations, distance_info, dim):
    sample_points = []

    for i in range(len(solve_locations)):
        if distance_info[i][0]: # in valid solve region
            sample_pt = wosx.Solvers.GPUSamplePoint(dim=dim)
            if dim == 2:
                sample_pt.pt = wosx.GPUFloat2(solve_locations[i][0],
                                              solve_locations[i][1])
            else:
                sample_pt.pt = wosx.GPUFloat3(solve_locations[i][0],
                                              solve_locations[i][1],
                                              solve_locations[i][2])
            sample_pt.type = wosx.Solvers.SampleType.InDomain
            sample_pt.estimation_quantity = wosx.Solvers.EstimationQuantity.Solution
            sample_pt.dist_to_absorbing_boundary = distance_info[i][1]
            sample_pt.dist_to_reflecting_boundary = distance_info[i][2]
            sample_points.append(sample_pt)

    return wosx.Solvers.GPUSamplePointList(sample_points, dim=dim)

def run_walk_on_spheres(solver_config, task_handle, sample_pts,
                        geometric_queries, pde, solve_double_sided,
                        dim, channels):
    # load config settings
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    n_walks = solver_config["nWalks"]\
        if "nWalks" in solver_config else 128
    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    n_resident_threads = solver_config["nResidentThreads"]\
        if "nResidentThreads" in solver_config else 131072

    disable_gradient_control_variates = solver_config["disableGradientControlVariates"]\
        if "disableGradientControlVariates" in solver_config else False
    disable_gradient_antithetic_variates = solver_config["disableGradientAntitheticVariates"]\
        if "disableGradientAntitheticVariates" in solver_config else False
    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    disable_persistent_threads = solver_config["disablePersistentThreads"]\
        if "disablePersistentThreads" in solver_config else False
    if not disable_persistent_threads and task_handle.get_device_backend() != "cuda":
        print("Persistent threads require CUDA backend, disabling")
        disable_persistent_threads = True

    # initialize GPU task handle
    task_handle.set_geometric_queries(geometric_queries)
    task_handle.set_pde(pde)
    task_handle.init()

    # initialize GPU walk settings
    walk_settings = wosx.Solvers.GPUWalkSettings()
    walk_settings.epsilon_shell_for_absorbing_boundary = epsilon_shell_for_absorbing_boundary
    walk_settings.epsilon_shell_for_reflecting_boundary = 0.0
    walk_settings.silhouette_precision = 0.0
    walk_settings.russian_roulette_threshold = russian_roulette_threshold
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = 0
    walk_settings.solve_double_sided = 1 if solve_double_sided else 0
    walk_settings.use_gradient_control_variates = 0 if disable_gradient_control_variates else 1
    walk_settings.use_gradient_antithetic_variates = 0 if disable_gradient_antithetic_variates else 1
    walk_settings.ignore_absorbing_boundary_contribution = 1 if ignore_absorbing_boundary_contribution else 0
    walk_settings.ignore_reflecting_boundary_contribution = 1
    walk_settings.ignore_source_contribution = 1 if ignore_source_contribution else 0

    # initialize GPU walk on spheres solver
    walk_on_spheres = wosx.Solvers.GPUWalkOnSpheresSolver(task_handle, walk_settings, n_resident_threads,
                                                          not disable_persistent_threads, print_logs,
                                                          dim=dim, channels=channels)

    if disable_persistent_threads:
        # compute workload parameters--the number of sample copies and number of
        # walks per copy--to help improve GPU occupancy when running the solver
        workload_parameters = walk_on_spheres.compute_workload_parameters(len(sample_pts), n_walks)

        # populate sample points on the GPU
        walk_on_spheres.populate_sample_points(sample_pts, workload_parameters[0], False)

        # run solver on the GPU
        walk_on_spheres.solve(1, workload_parameters[1])

    else:
        # populate sample points on the GPU
        walk_on_spheres.populate_sample_points(sample_pts, 1, False)

        # run solver on the GPU
        walk_on_spheres.solve(n_walks)

    # extract sample statistics from the GPU
    sample_statistics = wosx.Solvers.GPUSampleStatisticsList(dim=dim, channels=channels)
    walk_on_spheres.get_sample_statistics(sample_statistics)

    return sample_statistics

def run_walk_on_stars(solver_config, task_handle, sample_pts,
                      geometric_queries, pde, solve_double_sided,
                      dim, channels):
    # load config settings
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    n_walks = solver_config["nWalks"]\
        if "nWalks" in solver_config else 128
    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length
    n_resident_threads = solver_config["nResidentThreads"]\
        if "nResidentThreads" in solver_config else 131072

    disable_gradient_control_variates = solver_config["disableGradientControlVariates"]\
        if "disableGradientControlVariates" in solver_config else False
    disable_gradient_antithetic_variates = solver_config["disableGradientAntitheticVariates"]\
        if "disableGradientAntitheticVariates" in solver_config else False
    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_reflecting_boundary_contribution = solver_config["ignoreReflectingBoundaryContribution"]\
        if "ignoreReflectingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    disable_persistent_threads = solver_config["disablePersistentThreads"]\
        if "disablePersistentThreads" in solver_config else False
    if not disable_persistent_threads and task_handle.get_device_backend() != "cuda":
        print("Persistent threads require CUDA backend, disabling")
        disable_persistent_threads = True

    # initialize GPU task handle
    task_handle.set_geometric_queries(geometric_queries)
    task_handle.set_pde(pde)
    task_handle.init()

    # initialize GPU walk settings
    walk_settings = wosx.Solvers.GPUWalkSettings()
    walk_settings.epsilon_shell_for_absorbing_boundary = epsilon_shell_for_absorbing_boundary
    walk_settings.epsilon_shell_for_reflecting_boundary = epsilon_shell_for_reflecting_boundary
    walk_settings.silhouette_precision = silhouette_precision
    walk_settings.russian_roulette_threshold = russian_roulette_threshold
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = steps_before_using_maximal_spheres
    walk_settings.solve_double_sided = 1 if solve_double_sided else 0
    walk_settings.use_gradient_control_variates = 0 if disable_gradient_control_variates else 1
    walk_settings.use_gradient_antithetic_variates = 0 if disable_gradient_antithetic_variates else 1
    walk_settings.ignore_absorbing_boundary_contribution = 1 if ignore_absorbing_boundary_contribution else 0
    walk_settings.ignore_reflecting_boundary_contribution = 1 if ignore_reflecting_boundary_contribution else 0
    walk_settings.ignore_source_contribution = 1 if ignore_source_contribution else 0

    # initialize GPU walk on stars solver
    walk_on_stars = wosx.Solvers.GPUWalkOnStarsSolver(task_handle, walk_settings, n_resident_threads,
                                                      not disable_persistent_threads, print_logs,
                                                      dim=dim, channels=channels)

    if disable_persistent_threads:
        # compute workload parameters--the number of sample copies and number of
        # walks per copy--to help improve GPU occupancy when running the solver
        workload_parameters = walk_on_stars.compute_workload_parameters(len(sample_pts), n_walks)

        # populate sample points on the GPU
        walk_on_stars.populate_sample_points(sample_pts, workload_parameters[0], False)

        # run solver on the GPU
        walk_on_stars.solve(1, workload_parameters[1])

    else:
        # populate sample points on the GPU
        walk_on_stars.populate_sample_points(sample_pts, 1, False)

        # run solver on the GPU
        walk_on_stars.solve(n_walks)

    # extract sample statistics from the GPU
    sample_statistics = wosx.Solvers.GPUSampleStatisticsList(dim=dim, channels=channels)
    walk_on_stars.get_sample_statistics(sample_statistics)

    return sample_statistics

def get_solution_from_sample_points(distance_info, sample_statistics, channels):
    solution = create_solution_values(len(distance_info), channels)
    counter = 0

    for i in range(len(distance_info)):
        if distance_info[i][0]: # in valid solve region
            solution[i] = get_solution_value(sample_statistics[counter].get_estimated_solution(), channels)
            counter += 1

    return solution

##############################################################################################
# Boundary Value Caching solver

def create_bvc_evaluation_points(solve_locations, dim):
    evaluation_points = [None] * len(solve_locations)

    for i in range(len(solve_locations)):
        evaluation_points[i] = wosx.Solvers.GPUBVCEvaluationPoint(dim=dim)
        if dim == 2:
            evaluation_points[i].pt = wosx.GPUFloat2(solve_locations[i][0],
                                                     solve_locations[i][1])
        else:
            evaluation_points[i].pt = wosx.GPUFloat3(solve_locations[i][0],
                                                     solve_locations[i][1],
                                                     solve_locations[i][2])
        evaluation_points[i].type = wosx.Solvers.SampleType.InDomain

    return wosx.Solvers.GPUBVCEvaluationPointList(evaluation_points, dim=dim)

def run_boundary_value_caching(solver_config, task_handle, evaluation_pts, bounding_box,
                               absorbing_boundary_positions, absorbing_boundary_indices,
                               reflecting_boundary_positions, reflecting_boundary_indices,
                               geometric_queries, pde, solve_double_sided, dim, channels):
    # load config settings for walk on stars
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length
    n_resident_threads = solver_config["nResidentThreads"]\
        if "nResidentThreads" in solver_config else 131072

    disable_gradient_control_variates = solver_config["disableGradientControlVariates"]\
        if "disableGradientControlVariates" in solver_config else False
    disable_gradient_antithetic_variates = solver_config["disableGradientAntitheticVariates"]\
        if "disableGradientAntitheticVariates" in solver_config else False
    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_reflecting_boundary_contribution = solver_config["ignoreReflectingBoundaryContribution"]\
        if "ignoreReflectingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    disable_persistent_threads = solver_config["disablePersistentThreads"]\
        if "disablePersistentThreads" in solver_config else False
    if not disable_persistent_threads and task_handle.get_device_backend() != "cuda":
        print("Persistent threads require CUDA backend, disabling")
        disable_persistent_threads = True

    # load config settings for boundary value caching
    n_walks_for_cached_solution_estimates = solver_config["nWalksForCachedSolutionEstimates"]\
        if "nWalksForCachedSolutionEstimates" in solver_config else 128
    n_walks_for_cached_gradient_estimates = solver_config["nWalksForCachedGradientEstimates"]\
        if "nWalksForCachedGradientEstimates" in solver_config else 640
    absorbing_boundary_cache_size = solver_config["absorbingBoundaryCacheSize"]\
        if "absorbingBoundaryCacheSize" in solver_config else 1024
    reflecting_boundary_cache_size = solver_config["reflectingBoundaryCacheSize"]\
        if "reflectingBoundaryCacheSize" in solver_config else 1024
    domain_cache_size = solver_config["domainCacheSize"]\
        if "domainCacheSize" in solver_config else 1024

    normal_offset_for_absorbing_boundary = solver_config["normalOffsetForAbsorbingBoundary"]\
        if "normalOffsetForAbsorbingBoundary" in solver_config else 5.0 * epsilon_shell_for_absorbing_boundary
    normal_offset_for_reflecting_boundary = solver_config["normalOffsetForReflectingBoundary"]\
        if "normalOffsetForReflectingBoundary" in solver_config else 0.0
    radius_clamp_for_kernels = solver_config["radiusClampForKernels"]\
        if "radiusClampForKernels" in solver_config else 0.0
    regularization_for_kernels = solver_config["regularizationForKernels"]\
        if "regularizationForKernels" in solver_config else 0.0

    # initialize boundary samplers: wrap the Python callable in the opaque callback type
    inside_bounding_domain = wosx.Utils.get_inside_bounding_box_callback(bounding_box[0],
                                                                         bounding_box[1],
                                                                         dim=dim)
    absorbing_boundary_sampler = wosx.Samplers.create_gpu_uniform_boundary_sampler(
        geometric_queries, absorbing_boundary_positions, absorbing_boundary_indices,
        inside_bounding_domain, normal_offset_for_absorbing_boundary,
        solve_double_sided, dim=dim)
    reflecting_boundary_sampler = wosx.Samplers.create_gpu_uniform_boundary_sampler(
        geometric_queries, reflecting_boundary_positions, reflecting_boundary_indices,
        inside_bounding_domain, normal_offset_for_reflecting_boundary,
        solve_double_sided, dim=dim)

    # initialize solve region and domain sampler
    solve_region = None
    if solve_double_sided:
        solve_region = wosx.Samplers.GPUBoundingBoxSolveRegion(bounding_box[0],
                                                               bounding_box[1],
                                                               dim=dim)

    else:
        signed_volume_absorbing = wosx.Utils.compute_signed_volume(
            absorbing_boundary_positions, absorbing_boundary_indices, dim=dim)
        signed_volume_reflecting = wosx.Utils.compute_signed_volume(
            reflecting_boundary_positions, reflecting_boundary_indices, dim=dim)
        region_volume = abs(signed_volume_absorbing + signed_volume_reflecting)
        solve_region = wosx.Samplers.GPUWatertightDomainSolveRegion(
            geometric_queries, bounding_box[0], bounding_box[1], region_volume, dim=dim)

    domain_sampler = wosx.Samplers.GPUUniformDomainSampler(solve_region, geometric_queries, dim=dim)
    if ignore_source_contribution:
        domain_cache_size = 0

    # initialize GPU task handle
    task_handle.set_geometric_queries(geometric_queries)
    task_handle.set_absorbing_boundary_sampler(absorbing_boundary_sampler)
    task_handle.set_reflecting_boundary_sampler(reflecting_boundary_sampler)
    task_handle.set_domain_sampler(domain_sampler)
    task_handle.set_pde(pde)
    task_handle.init()

    # initialize GPU walk settings
    walk_settings = wosx.Solvers.GPUWalkSettings()
    walk_settings.epsilon_shell_for_absorbing_boundary = epsilon_shell_for_absorbing_boundary
    walk_settings.epsilon_shell_for_reflecting_boundary = epsilon_shell_for_reflecting_boundary
    walk_settings.silhouette_precision = silhouette_precision
    walk_settings.russian_roulette_threshold = russian_roulette_threshold
    walk_settings.max_walk_length = max_walk_length
    walk_settings.steps_before_using_maximal_spheres = steps_before_using_maximal_spheres
    walk_settings.solve_double_sided = 1 if solve_double_sided else 0
    walk_settings.use_gradient_control_variates = 0 if disable_gradient_control_variates else 1
    walk_settings.use_gradient_antithetic_variates = 0 if disable_gradient_antithetic_variates else 1
    walk_settings.ignore_absorbing_boundary_contribution = 1 if ignore_absorbing_boundary_contribution else 0
    walk_settings.ignore_reflecting_boundary_contribution = 1 if ignore_reflecting_boundary_contribution else 0
    walk_settings.ignore_source_contribution = 1 if ignore_source_contribution else 0

    # initialize GPU boundary value caching solver
    boundary_value_caching = wosx.Solvers.GPUBoundaryValueCachingSolver(
        task_handle, walk_settings, n_resident_threads,
        not disable_persistent_threads, print_logs,
        dim=dim, channels=channels)

    # populate evaluation points on the GPU
    boundary_value_caching.populate_evaluation_points(evaluation_pts)

    # generate boundary and domain samples
    boundary_value_caching.generate_samples(absorbing_boundary_cache_size,
                                            reflecting_boundary_cache_size,
                                            domain_cache_size)

    # compute sample estimates on the boundary
    boundary_value_caching.compute_sample_estimates(n_walks_for_cached_solution_estimates,
                                                    n_walks_for_cached_gradient_estimates)

    # splat boundary sample estimates and domain data to evaluation points
    boundary_value_caching.splat(radius_clamp_for_kernels, regularization_for_kernels)

    # extract evaluation outputs from the GPU
    evaluation_outputs = wosx.Solvers.GPUBVCEvaluationOutputsList(dim=dim, channels=channels)
    boundary_value_caching.get_evaluation_outputs(evaluation_outputs)

    return evaluation_outputs

def get_solution_from_bvc_evaluation_outputs(evaluation_outputs, channels):
    solution = create_solution_values(len(evaluation_outputs), channels)

    for i in range(len(evaluation_outputs)):
        solution[i] = get_solution_value(evaluation_outputs[i].get_estimated_solution(), channels)

    return solution

##############################################################################################
# Solver execution

def run_solver(solver_type, device_backend, solver_config, bounding_box,
               absorbing_boundary_positions, absorbing_boundary_indices,
               reflecting_boundary_positions, reflecting_boundary_indices,
               geometric_queries, pde, solve_locations, distance_info,
               solve_double_sided, dim, channels):
    # create GPU task handle
    wosx_directory_path = str(Path.cwd())
    wosx_pde_directory_path = wosx_directory_path + "/demo_apps/basic_2d"
    task_handle = wosx.GPUTaskHandle(wosx_directory_path, wosx_pde_directory_path,
                                     device_backend, dim=dim, channels=channels)

    if solver_type == "wos":
        # create sample points to estimate solution at
        sample_pts = create_sample_points(solve_locations, distance_info, dim)

        # run walk on spheres
        sample_statistics = run_walk_on_spheres(solver_config, task_handle,
                                                sample_pts, geometric_queries, pde,
                                                solve_double_sided, dim, channels)

        # extract solution from sample points
        return get_solution_from_sample_points(distance_info, sample_statistics, channels)

    elif solver_type == "wost":
        # create sample points to estimate solution at
        sample_pts = create_sample_points(solve_locations, distance_info, dim)

        # run walk on stars
        sample_statistics = run_walk_on_stars(solver_config, task_handle,
                                              sample_pts, geometric_queries, pde,
                                              solve_double_sided, dim, channels)

        # extract solution from sample points
        return get_solution_from_sample_points(distance_info, sample_statistics, channels)

    elif solver_type == "bvc":
        # create evaluation points to estimate solution at
        evaluation_pts = create_bvc_evaluation_points(solve_locations, dim)

        # run boundary value caching
        evaluation_outputs = run_boundary_value_caching(
            solver_config, task_handle, evaluation_pts, bounding_box,
            absorbing_boundary_positions, absorbing_boundary_indices,
            reflecting_boundary_positions, reflecting_boundary_indices,
            geometric_queries, pde, solve_double_sided, dim, channels)

        # extract solution from evaluation points
        return get_solution_from_bvc_evaluation_outputs(evaluation_outputs, channels)

    else:
        raise ValueError(f"Invalid solver type: {solver_type}")

def run_solver_exterior(solver_type, device_backend, solver_config, model_problem_config,
                        positions, absorbing_boundary_positions, absorbing_boundary_indices,
                        reflecting_boundary_positions, reflecting_boundary_indices,
                        pde, robin_coeff, solve_locations, solve_double_sided, dim, channels):
    # initialize a Kelvin transform: ensure origin lies inside the default domain
    # used for the demo, which is a requirement for solving exterior problems
    origin = np.zeros(dim)
    origin[1] = 0.125
    kelvin_transform = wosx.Solvers.KelvinTransform(origin, dim=dim, channels=channels)

    # invert the exterior problem into an equivalent interior problem
    inverted_bounding_box, inverted_positions, inverted_absorbing_boundary_positions,\
        inverted_reflecting_boundary_positions, pde_inverted_domain,\
            min_robin_coeff_values_inverted_domain, max_robin_coeff_values_inverted_domain =\
                invert_exterior_problem(kelvin_transform, positions, absorbing_boundary_positions,
                                        reflecting_boundary_positions, reflecting_boundary_indices,
                                        pde, robin_coeff, dim, channels)

    # set up the geometric queries for the inverted absorbing and reflecting boundary
    geometric_queries_inverted_domain, inverted_absorbing_boundary_handler,\
        inverted_reflecting_boundary_handler = setup_geometric_queries(
            model_problem_config, inverted_bounding_box,
            inverted_absorbing_boundary_positions, absorbing_boundary_indices,
            inverted_reflecting_boundary_positions, reflecting_boundary_indices,
            min_robin_coeff_values_inverted_domain, max_robin_coeff_values_inverted_domain,
            robin_coeff == 0.0, solve_double_sided, dim)

    # invert the solve locations and update the distance info
    domain_is_watertight = model_problem_config["domainIsWatertight"]\
        if "domainIsWatertight" in model_problem_config else False
    inverted_solve_locations = invert_solve_locations(kelvin_transform, solve_locations, dim)
    distance_info_inverted_domain = compute_distance_info(
        device_backend, inverted_solve_locations, geometric_queries_inverted_domain,
        domain_is_watertight, solve_double_sided, False, dim, channels)

    # run the solver
    solution = run_solver(solver_type, device_backend, solver_config, inverted_bounding_box,
                          inverted_absorbing_boundary_positions, absorbing_boundary_indices,
                          inverted_reflecting_boundary_positions, reflecting_boundary_indices,
                          geometric_queries_inverted_domain, pde_inverted_domain, inverted_solve_locations,
                          distance_info_inverted_domain, solve_double_sided, dim, channels)

    # map the solution values back to the exterior domain
    return compute_exterior_solution(kelvin_transform, solution, inverted_solve_locations, channels)

def run_demo():
    # parse arguments
    parser = argparse.ArgumentParser(description="wosx 2d gpu demo application")
    parser.add_argument("--config", type=str, required=True, help="path to the configuration file")
    args = parser.parse_args()

    try:
        # load the configuration file
        with open(args.config, 'r') as file:
            config = json.load(file)

        # set problem parameters
        model_problem_config = config["modelProblem"]
        channels = get_channels(model_problem_config)
        dim = 2

        # load a boundary mesh
        positions, indices, bounding_box = load_boundary_mesh(model_problem_config, dim)

        # setup the PDE
        pde, has_reflecting_boundary_conditions = setup_pde(model_problem_config, bounding_box,
                                                            dim, channels)

        # partition the boundary mesh into absorbing and reflecting boundary elements
        absorbing_boundary_positions, absorbing_boundary_indices,\
            reflecting_boundary_positions, reflecting_boundary_indices =\
                partition_boundary_mesh(has_reflecting_boundary_conditions, positions, indices, dim)

        # specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
        # we use a constant value for all elements in this demo, but WoSX supports variable coefficients
        robin_coeff = model_problem_config["robinCoeff"]\
            if "robinCoeff" in model_problem_config else 0.0
        min_robin_coeff_values = wosx.FloatList([abs(robin_coeff)] * len(reflecting_boundary_indices))
        max_robin_coeff_values = wosx.FloatList([abs(robin_coeff)] * len(reflecting_boundary_indices))

        # set up the geometric queries for the absorbing and reflecting boundary
        solve_double_sided = model_problem_config["solveDoubleSided"]\
            if "solveDoubleSided" in model_problem_config else False
        geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler =\
            setup_geometric_queries(model_problem_config, bounding_box,
                                    absorbing_boundary_positions, absorbing_boundary_indices,
                                    reflecting_boundary_positions, reflecting_boundary_indices,
                                    min_robin_coeff_values, max_robin_coeff_values,
                                    robin_coeff == 0.0, solve_double_sided, dim)

        # create solve locations on a grid for this demo
        output_config = config["output"]
        device_backend = config["deviceBackend"]\
            if "deviceBackend" in config else "cuda"
        solve_exterior = model_problem_config["solveExterior"]\
            if "solveExterior" in model_problem_config else False
        domain_is_watertight = model_problem_config["domainIsWatertight"]\
            if "domainIsWatertight" in model_problem_config else False
        solve_locations = create_grid_points(output_config, bounding_box)
        distance_info = compute_distance_info(device_backend, solve_locations, geometric_queries,
                                              domain_is_watertight, solve_double_sided,
                                              solve_exterior, dim, channels)

        # run the solver
        solver_type = config["solverType"]
        solver_config = config["solver"]
        solution = None
        if solve_exterior:
            solution = run_solver_exterior(solver_type, device_backend, solver_config, model_problem_config,
                                           positions, absorbing_boundary_positions, absorbing_boundary_indices,
                                           reflecting_boundary_positions, reflecting_boundary_indices,
                                           pde, robin_coeff, solve_locations, solve_double_sided, dim, channels)

        else:
            solution = run_solver(solver_type, device_backend, solver_config, bounding_box,
                                  absorbing_boundary_positions, absorbing_boundary_indices,
                                  reflecting_boundary_positions, reflecting_boundary_indices,
                                  geometric_queries, pde, solve_locations, distance_info,
                                  solve_double_sided, dim, channels)

        # save the solution to disk
        grid_values = create_grid_values(output_config, distance_info, solution, channels)
        solution_file = output_config["solutionFile"]\
            if "solutionFile" in output_config else "solution.png"
        save_image_buffer(output_config, solution_file, grid_values, channels)

    except FileNotFoundError:
        sys.exit("Configuration file not found")

    except json.JSONDecodeError:
        sys.exit("Invalid configuration file")

    except ValueError as error:
        sys.exit(str(error))

if __name__ == "__main__":
    run_demo()
