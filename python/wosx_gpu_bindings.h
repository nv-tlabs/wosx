/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file contains helper functions to create python bindings for WoSX using nanobind.
// Functions are templated on a value type T which can be, e.g., a float for scalar-valued
// PDE problems or an Array<float, CHANNELS> for vector-valued problems. The template parameter
// DIM specifies the dimension of the problem, and is typically 2 or 3. Users can use these
// functions to create bindings specialized to value types and dimensions of their choice.

#pragma once

#include "wosx_bindings.h"
#include <nanobind/trampoline.h>
#include <wosx/wosx_gpu.h>

// vector types
template <size_t DIM>
using GPUSamplePointList = std::vector<wosx::GPUSamplePoint<DIM>>;
template <typename T, size_t DIM>
using GPUSampleStatisticsList = std::vector<wosx::GPUSampleStatistics<T, DIM>>;
template <size_t DIM>
using GPUBVCEvaluationPointList = std::vector<wosx::GPUBVCEvaluationPoint<DIM>>;
template <typename T, size_t DIM>
using GPUBVCEvaluationOutputsList = std::vector<wosx::GPUBVCEvaluationOutputs<T, DIM>>;
using StringPairList = std::vector<std::pair<std::string, std::string>>;

// trampoline for Python subclassing
struct PyGPUPDE: wosx::GPUPDE {
    NB_TRAMPOLINE(wosx::GPUPDE, 5);

    void allocate(wosx::GPUContext& context) override {
        NB_OVERRIDE_PURE_NAME("allocate", allocate, context);
    }

    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const override {
        NB_OVERRIDE_PURE_NAME("set_resources", setResources, cursor, printLogs);
    }

    std::string getReflectionType() const override {
        NB_OVERRIDE_PURE_NAME("get_reflection_type", getReflectionType);
    }

    wosx::GPUPDEType getType() const override {
        NB_OVERRIDE_PURE_NAME("get_type", getType);
    }

    bool usesKelvinTransform() const override {
        NB_OVERRIDE_NAME("uses_kelvin_transform", usesKelvinTransform);
    }
};

// wrapper for GPUBuffer class
template <typename T>
class GPUStructuredBuffer {
public:
    // constructors
    GPUStructuredBuffer(bool unorderedAccess_): unorderedAccess(unorderedAccess_) {
        // do nothing
    }

    // allocates buffer
    void allocate(wosx::GPUContext& context, const std::vector<T>& data) {
        buffer.allocate<T>(context, unorderedAccess, data);
    }

    // reads buffer data
    void read(wosx::GPUContext& context, std::vector<T>& result) const {
        buffer.read<T>(context, result);
    }

    // sets binding
    void setBinding(const wosx::ShaderCursor& cursor) const {
        cursor.setBinding(buffer.buffer);
    }

private:
    // members
    wosx::GPUBuffer buffer = {};
    bool unorderedAccess;
};

using GPUUintBuffer = GPUStructuredBuffer<uint32_t>;
using GPUFloatBuffer = GPUStructuredBuffer<float>;
using GPUFloat2Buffer = GPUStructuredBuffer<wosx::Vector2>;
using GPUFloat3Buffer = GPUStructuredBuffer<wosx::Vector3>;

// binding functions
void bindGPUNonTemplatedLibraryResources(nb::module_ m);

template <size_t DIM>
void bindGPUGeometricQueries(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindGPUSamplers(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindGPUSampleAndEvaluationPoints(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindGPUGeometryUtilityStructures(nb::module_ m, std::string typeStr="");

template <typename T, size_t DIM>
void bindGPUTaskHandle(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUKelvinPDE(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUSampleStatistics(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUPointEstimatorSolvers(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUBoundaryValueCachingSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUDistanceQueries(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindGPUDenseGrid(nb::module_ m, std::string typeStr="");

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename BufferType>
void bindGPUStructuredBuffer(nb::module_ m, const std::string& name)
{
    nb::class_<BufferType>(m, name.c_str())
        .def(nb::init<bool>(),
            "unordered_access"_a=false)
        .def("allocate", &BufferType::allocate,
            "Allocates GPU resources.",
            "context"_a, "data"_a)
        .def("read", &BufferType::read,
            "Reads buffer data from the GPU.",
            "context"_a, "result"_a)
        .def("set_binding", &BufferType::setBinding,
            "Sets GPU resources.",
            "cursor"_a);
}

inline void bindGPUNonTemplatedLibraryResources(nb::module_ m)
{
    nb::bind_vector<StringPairList>(m, "StringPairList");

    nb::class_<wosx::float2>(m, "GPUFloat2")
        .def(nb::init<>())
        .def(nb::init<float, float>(),
            "x"_a, "y"_a)
        .def_rw("x", &wosx::float2::x)
        .def_rw("y", &wosx::float2::y);

    nb::class_<wosx::float3>(m, "GPUFloat3")
        .def(nb::init<>())
        .def(nb::init<float, float, float>(),
            "x"_a, "y"_a, "z"_a)
        .def_rw("x", &wosx::float3::x)
        .def_rw("y", &wosx::float3::y)
        .def_rw("z", &wosx::float3::z);

    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    nb::class_<wosx::GPUContext>(opaque_types_m, "GPUContext"); // opaque -- users receive it in allocate() and forward it
    nb::class_<wosx::GPUShaderObject>(opaque_types_m, "GPUShaderObject"); // must be registered for nanobind inheritance

    nb::module_ core_m = m.def_submodule("Core", "Core module");

    nb::class_<wosx::ShaderCursor>(core_m, "GPUShaderCursor")
        .def("__getitem__", [](const wosx::ShaderCursor& c, const char* name) { return c[name]; })
        .def("set_data_float", [](const wosx::ShaderCursor& c, float v) { c.setData(v); })
        .def("set_data_uint", [](const wosx::ShaderCursor& c, uint32_t v) { c.setData(v); })
        .def("set_data_int", [](const wosx::ShaderCursor& c, int32_t v) { c.setData(v); })
        .def("set_data_float2", [](const wosx::ShaderCursor& c, const wosx::float2& v) { c.setData(v); })
        .def("set_data_float3", [](const wosx::ShaderCursor& c, const wosx::float3& v) { c.setData(v); });

    nb::enum_<wosx::GPUPDEType>(core_m, "GPUPDEType")
        .value("Poisson", wosx::GPUPDEType::Poisson)
        .value("ScreenedPoisson", wosx::GPUPDEType::ScreenedPoisson);

    nb::class_<wosx::GPUPDE, wosx::GPUShaderObject, PyGPUPDE>(core_m, "GPUPDE")
        .def(nb::init<>()) // TODO: is this needed for the trampoline to work? GPUPDE is an abstract class...
        .def("allocate", &wosx::GPUPDE::allocate,
            "Allocates GPU resources.",
            "context"_a)
        .def("set_resources", &wosx::GPUPDE::setResources,
            "Sets GPU resources.",
            "cursor"_a, "print_logs"_a)
        .def("get_reflection_type", &wosx::GPUPDE::getReflectionType,
            "Returns the reflection type.")
        .def("get_type", &wosx::GPUPDE::getType,
            "Returns the PDE type.")
        .def("uses_kelvin_transform", &wosx::GPUPDE::usesKelvinTransform,
            "Checks if the PDE uses a Kelvin transform.");

    nb::class_<wosx::GPUBoundaryHandler, wosx::GPUShaderObject>(
        core_m, "GPUBoundaryHandler");
    nb::class_<wosx::GPUAbsorbingBoundaryHandler, wosx::GPUBoundaryHandler>(
        core_m, "GPUAbsorbingBoundaryHandler");
    nb::class_<wosx::GPUEmptyAbsorbingBoundaryHandler, wosx::GPUAbsorbingBoundaryHandler>(
        core_m, "GPUEmptyAbsorbingBoundaryHandler")
        .def(nb::init<>());
    nb::class_<wosx::GPUReflectingBoundaryHandler, wosx::GPUBoundaryHandler>(
        core_m, "GPUReflectingBoundaryHandler");
    nb::class_<wosx::GPUEmptyReflectingBoundaryHandler, wosx::GPUReflectingBoundaryHandler>(
        core_m, "GPUEmptyReflectingBoundaryHandler")
        .def(nb::init<>());

    nb::module_ samplers_m = m.def_submodule("Samplers", "Samplers module");

    nb::class_<wosx::GPUSolveRegion, wosx::GPUShaderObject>(samplers_m, "GPUSolveRegion");
    nb::class_<wosx::GPUDomainSampler, wosx::GPUShaderObject>(samplers_m, "GPUDomainSampler");
    nb::class_<wosx::GPUBoundarySampler, wosx::GPUShaderObject>(samplers_m, "GPUBoundarySampler")
        .def("get_normal_offset_for_boundary", &wosx::GPUBoundarySampler::getNormalOffsetForBoundary,
            "Returns the normal offset for the boundary.")
        .def("get_sample_count", &wosx::GPUBoundarySampler::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false);

    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::GPUWalkSettings, wosx::GPUShaderObject>(solvers_m, "GPUWalkSettings")
        .def(nb::init<>())
        .def_rw("epsilon_shell_for_absorbing_boundary", &wosx::GPUWalkSettings::epsilonShellForAbsorbingBoundary)
        .def_rw("epsilon_shell_for_reflecting_boundary", &wosx::GPUWalkSettings::epsilonShellForReflectingBoundary)
        .def_rw("silhouette_precision", &wosx::GPUWalkSettings::silhouettePrecision)
        .def_rw("russian_roulette_threshold", &wosx::GPUWalkSettings::russianRouletteThreshold)
        .def_rw("max_walk_length", &wosx::GPUWalkSettings::maxWalkLength)
        .def_rw("steps_before_using_maximal_spheres", &wosx::GPUWalkSettings::stepsBeforeUsingMaximalSpheres)
        .def_rw("solve_double_sided", &wosx::GPUWalkSettings::solveDoubleSided)
        .def_rw("use_gradient_control_variates", &wosx::GPUWalkSettings::useGradientControlVariates)
        .def_rw("use_gradient_antithetic_variates", &wosx::GPUWalkSettings::useGradientAntitheticVariates)
        .def_rw("ignore_absorbing_boundary_contribution", &wosx::GPUWalkSettings::ignoreAbsorbingBoundaryContribution)
        .def_rw("ignore_reflecting_boundary_contribution", &wosx::GPUWalkSettings::ignoreReflectingBoundaryContribution)
        .def_rw("ignore_source_contribution", &wosx::GPUWalkSettings::ignoreSourceContribution);

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    bindGPUStructuredBuffer<GPUUintBuffer>(utils_m, "GPUUintBuffer");
    bindGPUStructuredBuffer<GPUFloatBuffer>(utils_m, "GPUFloatBuffer");
    bindGPUStructuredBuffer<GPUFloat2Buffer>(utils_m, "GPUFloat2Buffer");
    bindGPUStructuredBuffer<GPUFloat3Buffer>(utils_m, "GPUFloat3Buffer");

    nb::enum_<wosx::TextureFilteringMode>(utils_m, "GPUTextureFilteringMode")
        .value("Point", wosx::TextureFilteringMode::Point)
        .value("Linear", wosx::TextureFilteringMode::Linear);
    nb::enum_<wosx::TextureAddressingMode>(utils_m, "GPUTextureAddressingMode")
        .value("Wrap", wosx::TextureAddressingMode::Wrap)
        .value("ClampToEdge", wosx::TextureAddressingMode::ClampToEdge)
        .value("ClampToBorder", wosx::TextureAddressingMode::ClampToBorder)
        .value("MirrorRepeat", wosx::TextureAddressingMode::MirrorRepeat)
        .value("MirrorOnce", wosx::TextureAddressingMode::MirrorOnce);

    nb::class_<wosx::GPUSampler>(utils_m, "GPUSampler")
        .def(nb::init<>())
        .def("allocate", nb::overload_cast<wosx::GPUContext&,
                                           wosx::TextureFilteringMode,
                                           wosx::TextureAddressingMode>(&wosx::GPUSampler::allocate),
            "Allocates GPU resources.",
            "context"_a, "filter"_a, "address"_a);

    utils_m.def("print_gpu_reflection_info", &wosx::printReflectionInfo,
        "Prints reflection info for a GPU shader object.",
        "cursor"_a, "n_fields"_a, "reflection_type"_a);
}

template <size_t DIM>
void bindGPUGeometricQueries(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    nb::class_<wosx::GPUGeometricQueries<DIM>, wosx::GPUShaderObject>(
        core_m, ("GPUGeometricQueries" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler>,
                      std::shared_ptr<wosx::GPUReflectingBoundaryHandler>,
                      const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool>(),
            "absorbing_boundary_handler"_a, "reflecting_boundary_handler"_a,
            "domain_min"_a, "domain_max"_a, "domain_is_watertight"_a)
        .def("reallocate", &wosx::GPUGeometricQueries<DIM>::reallocate,
            "Reallocates GPU resources.",
            "context"_a, "absorbing_boundary_handler"_a, "reflecting_boundary_handler"_a,
            "domain_min"_a, "domain_max"_a, "domain_is_watertight"_a)
        .def("reallocate_absorbing_boundary",
            &wosx::GPUGeometricQueries<DIM>::reallocateAbsorbingBoundary,
            "Reallocates absorbing boundary GPU resources.",
            "context"_a, "absorbing_boundary_handler"_a,
            "domain_min"_a, "domain_max"_a, "domain_is_watertight"_a)
        .def("reallocate_reflecting_boundary",
            &wosx::GPUGeometricQueries<DIM>::reallocateReflectingBoundary,
            "Reallocates reflecting boundary GPU resources.",
            "context"_a, "reflecting_boundary_handler"_a,
            "domain_min"_a, "domain_max"_a, "domain_is_watertight"_a);
}

template <size_t DIM>
void bindGPUSamplers(nb::module_ m, std::string typeStr)
{
    nb::module_ samplers_m = m.def_submodule("Samplers", "Samplers module");

    nb::class_<wosx::GPUBoundingBoxSolveRegion<DIM>, wosx::GPUSolveRegion>(
        samplers_m, ("GPUBoundingBoxSolveRegion" + typeStr).c_str())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&>(),
            "region_min"_a, "region_max"_a);

    nb::class_<wosx::GPUWatertightDomainSolveRegion<DIM>, wosx::GPUSolveRegion>(
        samplers_m, ("GPUWatertightDomainSolveRegion" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUGeometricQueries<DIM>>,
                      const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, float>(),
            "geometric_queries"_a, "region_min"_a, "region_max"_a, "region_volume"_a);

    nb::class_<wosx::GPUEmptyDomainSampler<DIM>, wosx::GPUDomainSampler>(
        samplers_m, ("GPUEmptyDomainSampler" + typeStr).c_str())
        .def(nb::init<>());

    nb::class_<wosx::GPUUniformDomainSampler<DIM>, wosx::GPUDomainSampler>(
        samplers_m, ("GPUUniformDomainSampler" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUSolveRegion>,
                      std::shared_ptr<wosx::GPUGeometricQueries<DIM>>>(),
            "solve_region"_a, "geometric_queries"_a);

    nb::class_<wosx::GPUEmptyBoundarySampler<DIM>, wosx::GPUBoundarySampler>(
        samplers_m, ("GPUEmptyBoundarySampler" + typeStr).c_str())
        .def(nb::init<>())
        .def("get_normal_offset_for_boundary", &wosx::GPUEmptyBoundarySampler<DIM>::getNormalOffsetForBoundary,
            "Returns the normal offset for the boundary.")
        .def("get_sample_count", &wosx::GPUEmptyBoundarySampler<DIM>::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false);

    if (DIM == 2) {
        nb::class_<wosx::GPUUniformLineSegmentBoundarySampler, wosx::GPUBoundarySampler>(
            samplers_m, ("GPUUniformLineSegmentBoundarySampler" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUGeometricQueries<2>>,
                      const FloatNList<2>&, const IntNList<2>&,
                      const FloatList&, const FloatList&,
                      FloatNToTypeFunc<bool, 2>, float, bool, bool>(),
            "geometric_queries"_a, "positions"_a, "indices"_a,
            "primitive_weights"_a, "primitive_weights_normal_aligned"_a,
            "inside_solve_region"_a, "normal_offset_for_boundary"_a,
            "solve_double_sided"_a, "compute_weighted_normals"_a=false)
        .def("get_normal_offset_for_boundary", &wosx::GPUUniformLineSegmentBoundarySampler::getNormalOffsetForBoundary,
            "Returns the normal offset for the boundary.")
        .def("get_sample_count", &wosx::GPUUniformLineSegmentBoundarySampler::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false);

        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<2>>,
                                        const FloatNList<2>&, const IntNList<2>&,
                                        FloatNToTypeFunc<bool, 2>, float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "inside_solve_region"_a, "normal_offset_for_boundary"_a,
                      "solve_double_sided"_a, "compute_weighted_normals"_a=false);
        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<2>>,
                                        const FloatNList<2>&, const IntNList<2>&,
                                        const FloatList&, FloatNToTypeFunc<bool, 2>,
                                        float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "primitive_weights"_a, "inside_solve_region"_a,
                      "normal_offset_for_boundary"_a, "solve_double_sided"_a,
                      "compute_weighted_normals"_a=false);
        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<2>>,
                                        const FloatNList<2>&, const IntNList<2>&,
                                        const FloatList&, const FloatList&,
                                        FloatNToTypeFunc<bool, 2>, float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "primitive_weights"_a, "primitive_weights_normal_aligned"_a,
                      "inside_solve_region"_a, "normal_offset_for_boundary"_a,
                      "solve_double_sided"_a, "compute_weighted_normals"_a=false);

    } else if (DIM == 3) {
        nb::class_<wosx::GPUUniformTriangleBoundarySampler, wosx::GPUBoundarySampler>(
            samplers_m, ("GPUUniformTriangleBoundarySampler" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUGeometricQueries<3>>,
                      const FloatNList<3>&, const IntNList<3>&,
                      const FloatList&, const FloatList&,
                      FloatNToTypeFunc<bool, 3>, float, bool, bool>(),
            "geometric_queries"_a, "positions"_a, "indices"_a,
            "primitive_weights"_a, "primitive_weights_normal_aligned"_a,
            "inside_solve_region"_a, "normal_offset_for_boundary"_a,
            "solve_double_sided"_a, "compute_weighted_normals"_a=false)
        .def("get_normal_offset_for_boundary", &wosx::GPUUniformTriangleBoundarySampler::getNormalOffsetForBoundary,
            "Returns the normal offset for the boundary.")
        .def("get_sample_count", &wosx::GPUUniformTriangleBoundarySampler::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false);

        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<3>>,
                                        const FloatNList<3>&, const IntNList<3>&,
                                        FloatNToTypeFunc<bool, 3>, float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "inside_solve_region"_a, "normal_offset_for_boundary"_a,
                      "solve_double_sided"_a, "compute_weighted_normals"_a=false);
        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<3>>,
                                        const FloatNList<3>&, const IntNList<3>&,
                                        const FloatList&, FloatNToTypeFunc<bool, 3>,
                                        float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "primitive_weights"_a, "inside_solve_region"_a,
                      "normal_offset_for_boundary"_a, "solve_double_sided"_a,
                      "compute_weighted_normals"_a=false);
        samplers_m.def(("create_gpu_uniform_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<std::shared_ptr<wosx::GPUGeometricQueries<3>>,
                                        const FloatNList<3>&, const IntNList<3>&,
                                        const FloatList&, const FloatList&,
                                        FloatNToTypeFunc<bool, 3>, float, bool, bool>(
                                            &wosx::createUniformBoundarySampler),
                      "geometric_queries"_a, "positions"_a, "indices"_a,
                      "primitive_weights"_a, "primitive_weights_normal_aligned"_a,
                      "inside_solve_region"_a, "normal_offset_for_boundary"_a,
                      "solve_double_sided"_a, "compute_weighted_normals"_a=false);
    }
}

template <size_t DIM>
void bindGPUSampleAndEvaluationPoints(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::GPUSamplePoint<DIM>>(solvers_m, ("GPUSamplePoint" + typeStr).c_str())
        .def(nb::init<>())
        .def_rw("pt", &wosx::GPUSamplePoint<DIM>::pt)
        .def_rw("normal", &wosx::GPUSamplePoint<DIM>::normal)
        .def_rw("type", &wosx::GPUSamplePoint<DIM>::type)
        .def_rw("estimation_quantity", &wosx::GPUSamplePoint<DIM>::estimationQuantity)
        .def_rw("pdf", &wosx::GPUSamplePoint<DIM>::pdf)
        .def_rw("dist_to_absorbing_boundary", &wosx::GPUSamplePoint<DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &wosx::GPUSamplePoint<DIM>::distToReflectingBoundary)
        .def_rw("estimate_boundary_normal_aligned", &wosx::GPUSamplePoint<DIM>::estimateBoundaryNormalAligned);

    nb::bind_vector<GPUSamplePointList<DIM>>(solvers_m, ("GPUSamplePointList" + typeStr).c_str());

    nb::class_<wosx::GPUBVCEvaluationPoint<DIM>>(solvers_m, ("GPUBVCEvaluationPoint" + typeStr).c_str())
        .def(nb::init<>())
        .def_rw("pt", &wosx::GPUBVCEvaluationPoint<DIM>::pt)
        .def_rw("normal", &wosx::GPUBVCEvaluationPoint<DIM>::normal)
        .def_rw("type", &wosx::GPUBVCEvaluationPoint<DIM>::type);

    nb::bind_vector<GPUBVCEvaluationPointList<DIM>>(solvers_m, ("GPUBVCEvaluationPointList" + typeStr).c_str());
}

template <size_t DIM>
void bindGPUGeometryUtilityStructures(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    utils_m.def(("get_inside_bounding_box_callback" + typeStr).c_str(),
               [](const wosx::Vector<DIM>& boundingBoxMin,
                  const wosx::Vector<DIM>& boundingBoxMax) -> FloatNToTypeFunc<bool, DIM> {
                    return [boundingBoxMin, boundingBoxMax](const wosx::Vector<DIM>& x) -> bool {
                        return (x.array() >= boundingBoxMin.array()).all() &&
                               (x.array() <= boundingBoxMax.array()).all();
                    };
              },
              "bounding_box_min"_a, "bounding_box_max"_a,
              "Returns an inside bounding box indicator callback.");

    nb::class_<wosx::GPUSdfGrid<DIM>, wosx::GPUShaderObject>(utils_m, ("GPUSdfGrid" + typeStr).c_str())
        .def(nb::init<const Eigen::VectorXf&, const wosx::Vectori<DIM>&,
                      const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool>(),
            "sdf_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a, "unordered_access"_a)
        .def("allocate", &wosx::GPUSdfGrid<DIM>::allocate,
            "Allocates GPU resources.", "context"_a)
        .def("set_resources", &wosx::GPUSdfGrid<DIM>::setResources,
            "Sets GPU resources.", "cursor"_a, "print_logs"_a);

    nb::class_<wosx::GPUFcpwDirichletBoundaryHandler<DIM>, wosx::GPUAbsorbingBoundaryHandler>(
        utils_m, ("GPUFcpwDirichletBoundaryHandler" + typeStr).c_str())
        .def(nb::init<const FloatNList<DIM>&, const IntNList<DIM>&, bool>(),
            "positions"_a, "indices"_a, "print_stats"_a=true)
        .def("allocate", &wosx::GPUFcpwDirichletBoundaryHandler<DIM>::allocate,
            "Allocates GPU resources.", "context"_a)
        .def("set_resources", &wosx::GPUFcpwDirichletBoundaryHandler<DIM>::setResources,
            "Sets GPU resources.", "cursor"_a, "print_logs"_a);

    nb::class_<wosx::GPUSdfDirichletBoundaryHandler<DIM>, wosx::GPUAbsorbingBoundaryHandler>(
        utils_m, ("GPUSdfDirichletBoundaryHandler" + typeStr).c_str())
        .def(nb::init<const Eigen::VectorXf&, const wosx::Vectori<DIM>&,
                      const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool>(),
            "sdf_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a, "unordered_access"_a=false)
        .def("allocate", &wosx::GPUSdfDirichletBoundaryHandler<DIM>::allocate,
            "Allocates GPU resources.", "context"_a)
        .def("set_resources", &wosx::GPUSdfDirichletBoundaryHandler<DIM>::setResources,
            "Sets GPU resources.", "cursor"_a, "print_logs"_a);

    nb::class_<wosx::GPUFcpwNeumannBoundaryHandler<DIM>, wosx::GPUReflectingBoundaryHandler>(
        utils_m, ("GPUFcpwNeumannBoundaryHandler" + typeStr).c_str())
        .def(nb::init<const FloatNList<DIM>&, const IntNList<DIM>&, FloatIntToBoolFunc, bool>(),
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a, "print_stats"_a=true)
        .def("allocate", &wosx::GPUFcpwNeumannBoundaryHandler<DIM>::allocate,
            "Allocates GPU resources.", "context"_a)
        .def("set_resources", &wosx::GPUFcpwNeumannBoundaryHandler<DIM>::setResources,
            "Sets GPU resources.", "cursor"_a, "print_logs"_a);

    nb::class_<wosx::GPUFcpwRobinBoundaryHandler<DIM>, wosx::GPUReflectingBoundaryHandler>(
        utils_m, ("GPUFcpwRobinBoundaryHandler" + typeStr).c_str())
        .def(nb::init<const FloatNList<DIM>&, const IntNList<DIM>&, FloatIntToBoolFunc,
                      const FloatList&, const FloatList&, bool>(),
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a,
            "min_robin_coeff_values"_a, "max_robin_coeff_values"_a, "print_stats"_a=true)
        .def("allocate", nb::overload_cast<wosx::GPUContext&>(
            &wosx::GPUFcpwRobinBoundaryHandler<DIM>::allocate),
            "Allocates GPU resources.", "context"_a)
        .def("reallocate", &wosx::GPUFcpwRobinBoundaryHandler<DIM>::reallocate,
            "Reallocates GPU resources.",
            "context"_a, "min_robin_coeff_values"_a, "max_robin_coeff_values"_a)
        .def("set_resources", &wosx::GPUFcpwRobinBoundaryHandler<DIM>::setResources,
            "Sets GPU resources.", "cursor"_a, "print_logs"_a);
}

template <typename T, size_t DIM>
void bindGPUTaskHandle(nb::module_ m, std::string typeStr)
{
    nb::class_<wosx::GPUTaskHandle<T, DIM>>(m, ("GPUTaskHandle" + typeStr).c_str())
        .def(nb::init<const std::string&, const std::string&, const std::string&>(),
            "wosx_directory_path"_a, "wosx_pde_directory_path"_a=std::string(""),
            "device_backend"_a=std::string("default"))
        .def("set_geometric_queries", &wosx::GPUTaskHandle<T, DIM>::setGeometricQueries,
            "Sets the geometric queries.",
            "geometric_queries"_a)
        .def("set_absorbing_boundary_sampler", &wosx::GPUTaskHandle<T, DIM>::setAbsorbingBoundarySampler,
            "Sets the absorbing boundary sampler.",
            "sampler"_a)
        .def("set_reflecting_boundary_sampler", &wosx::GPUTaskHandle<T, DIM>::setReflectingBoundarySampler,
            "Sets the reflecting boundary sampler.",
            "sampler"_a)
        .def("set_domain_sampler", &wosx::GPUTaskHandle<T, DIM>::setDomainSampler,
            "Sets the domain sampler.",
            "sampler"_a)
        .def("set_pde", &wosx::GPUTaskHandle<T, DIM>::setPDE,
            "Sets the PDE.",
            "pde"_a)
        .def("init", &wosx::GPUTaskHandle<T, DIM>::init,
            "Initializes the handle (call after setting WoSX structures)",
            // release the GIL during device setup and shader compilation; the PyGPUPDE
            // trampoline re-acquires the GIL (via NB_OVERRIDE) before re-entering Python
            "user_macros"_a=StringPairList{},
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_context", &wosx::GPUTaskHandle<T, DIM>::getContext,
            "Returns the GPU context.",
            nb::rv_policy::reference_internal)
        .def("get_wosx_directory_path", &wosx::GPUTaskHandle<T, DIM>::getWosxDirectoryPath,
            "Returns the WoSX directory path.")
        .def("get_wosx_gpu_directory_path", &wosx::GPUTaskHandle<T, DIM>::getWosxGpuDirectoryPath,
            "Returns the WoSX GPU directory path.")
        .def("get_wosx_pde_directory_path", &wosx::GPUTaskHandle<T, DIM>::getWosxPdeDirectoryPath,
            "Returns the WoSX PDE directory path.")
        .def("get_device_backend", &wosx::GPUTaskHandle<T, DIM>::getDeviceBackend,
            "Returns the device backend.")
        .def("get_geometric_queries", &wosx::GPUTaskHandle<T, DIM>::getGeometricQueries,
            "Returns the geometric queries.")
        .def("get_absorbing_boundary_sampler", &wosx::GPUTaskHandle<T, DIM>::getAbsorbingBoundarySampler,
            "Returns the absorbing boundary sampler.")
        .def("get_reflecting_boundary_sampler", &wosx::GPUTaskHandle<T, DIM>::getReflectingBoundarySampler,
            "Returns the reflecting boundary sampler.")
        .def("get_domain_sampler", &wosx::GPUTaskHandle<T, DIM>::getDomainSampler,
            "Returns the domain sampler.")
        .def("get_pde", &wosx::GPUTaskHandle<T, DIM>::getPDE,
            "Returns the PDE.");
}

template <typename T, size_t DIM>
void bindGPUKelvinPDE(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    nb::class_<wosx::GPUKelvinPDE<T, DIM>, wosx::GPUPDE>(core_m, ("GPUKelvinPDE" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUPDE>, const wosx::Vector<DIM>&>(),
            "pde_exterior_domain"_a, "origin"_a=wosx::Vector<DIM>::Zero())
        .def("update_origin", &wosx::GPUKelvinPDE<T, DIM>::updateOrigin,
            "Updates the origin of the Kelvin transform.",
            "origin"_a);
}

template <typename T, size_t DIM>
void bindGPUSampleStatistics(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::GPUSampleStatistics<T, DIM>>(solvers_m, ("GPUSampleStatistics" + typeStr).c_str())
        .def(nb::init<>())
        .def("get_estimated_solution", &wosx::GPUSampleStatistics<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", nb::overload_cast<int>(
            &wosx::GPUSampleStatistics<T, DIM>::getEstimatedGradient, nb::const_),
            "Returns estimated gradient for specified channel.",
            "channel"_a)
        .def("get_solution_estimate_count", &wosx::GPUSampleStatistics<T, DIM>::getSolutionEstimateCount,
            "Returns number of solution estimates.")
        .def("get_gradient_estimate_count", &wosx::GPUSampleStatistics<T, DIM>::getGradientEstimateCount,
            "Returns number of gradient estimates.")
        .def("get_mean_walk_length", &wosx::GPUSampleStatistics<T, DIM>::getMeanWalkLength,
            "Returns mean walk length.");

    nb::bind_vector<GPUSampleStatisticsList<T, DIM>>(solvers_m, ("GPUSampleStatisticsList" + typeStr).c_str());
}

template <typename T, size_t DIM>
void bindGPUPointEstimatorSolvers(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::GPUPointEstimator<T, DIM>>(solvers_m, ("GPUPointEstimator" + typeStr).c_str())
        .def("compute_workload_parameters", &wosx::GPUPointEstimator<T, DIM>::computeWorkloadParameters,
            "Utility function to help improve GPU occupancy when running the solver:\ngiven the number of input sample points and number of walks per point,\ncomputes the number of sample copies and number of walks per copy.",
            "n_input_sample_points"_a, "n_walks"_a)
        .def("populate_sample_points", &wosx::GPUPointEstimator<T, DIM>::populateSamplePoints,
            "Populates sample points.",
            "input_sample_points"_a, "n_sample_copies"_a=1, "compute_distance_info"_a=true)
        .def("update_populated_sample_points_boundary_distance",
            &wosx::GPUPointEstimator<T, DIM>::updatePopulatedSamplePointsBoundaryDistance,
            "Updates the boundary distance for populated sample points.")
        .def("solve", &wosx::GPUPointEstimator<T, DIM>::solve,
            "Runs point estimator; nDispatchCalls is a utility parameter to help\nreduce GPU divergence when random walks have varying lengths:\nsetting nDispatchCalls > 1 (and nWalks := nWalksPerSampleCopy / nDispatchCalls)\nwill run the solver in multiple dispatch calls.",
            "n_walks"_a, "n_dispatch_calls"_a=1,
            nb::call_guard<nb::gil_scoped_release>())
        .def("reset_sample_statistics", &wosx::GPUPointEstimator<T, DIM>::resetSampleStatistics,
            "Resets sample statistics.")
        .def("get_sample_statistics", &wosx::GPUPointEstimator<T, DIM>::getSampleStatistics,
            "Returns sample statistics (resized to be the same size as input sample points).",
            "output_statistics"_a)
        .def("using_persistent_threads", &wosx::GPUPointEstimator<T, DIM>::usingPersistentThreads,
            "Returns true if using persistent threads.")
        .def("enable_logging", &wosx::GPUPointEstimator<T, DIM>::enableLogging,
            "Enables/disables logging.",
            "enable"_a);

    nb::class_<wosx::GPUWalkOnSpheresSolver<T, DIM>, wosx::GPUPointEstimator<T, DIM>>(
        solvers_m, ("GPUWalkOnSpheresSolver" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUTaskHandle<T, DIM>>,
                      std::shared_ptr<wosx::GPUWalkSettings>,
                      uint32_t, bool, bool>(),
            "task_handle"_a, "walk_settings"_a, "n_resident_threads"_a=131072,
            "use_persistent_threads"_a=true, "print_logs"_a=false);

    nb::class_<wosx::GPUWalkOnStarsSolver<T, DIM>, wosx::GPUPointEstimator<T, DIM>>(
        solvers_m, ("GPUWalkOnStarsSolver" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUTaskHandle<T, DIM>>,
                      std::shared_ptr<wosx::GPUWalkSettings>,
                      uint32_t, bool, bool>(),
            "task_handle"_a, "walk_settings"_a, "n_resident_threads"_a=131072,
            "use_persistent_threads"_a=true, "print_logs"_a=false);
}

template <typename T, size_t DIM>
void bindGPUBoundaryValueCachingSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::GPUBVCEvaluationOutputs<T, DIM>>(solvers_m, ("GPUBVCEvaluationOutputs" + typeStr).c_str())
        .def(nb::init<>())
        .def("get_estimated_solution", &wosx::GPUBVCEvaluationOutputs<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", &wosx::GPUBVCEvaluationOutputs<T, DIM>::getEstimatedGradient,
            "Returns estimated gradient for specified channel.",
            "channel"_a);

    nb::bind_vector<GPUBVCEvaluationOutputsList<T, DIM>>(solvers_m, ("GPUBVCEvaluationOutputsList" + typeStr).c_str());

    nb::class_<wosx::GPUBoundaryValueCachingSolver<T, DIM>>(
        solvers_m, ("GPUBoundaryValueCachingSolver" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUTaskHandle<T, DIM>>,
                      std::shared_ptr<wosx::GPUWalkSettings>,
                      uint32_t, bool, bool>(),
            "task_handle"_a, "walk_settings"_a, "n_resident_threads"_a=131072,
            "use_persistent_threads"_a=true, "print_logs"_a=false)
        .def("populate_evaluation_points", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::populateEvaluationPoints,
            "Populates evaluation points.",
            "input_evaluation_points"_a)
        .def("generate_samples", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::generateSamples,
            "Generates boundary and domain samples.",
            "absorbing_boundary_cache_size"_a, "reflecting_boundary_cache_size"_a, "domain_cache_size"_a)
        .def("compute_sample_estimates", nb::overload_cast<uint32_t, uint32_t>(
            &wosx::GPUBoundaryValueCachingSolver<T, DIM>::computeSampleEstimates),
            "Computes sample estimates on the boundary.",
            "n_walks_for_solution_estimates"_a, "n_walks_for_gradient_estimates"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("splat", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::splat,
            "Splat solution and gradient estimates into the interior.",
            "radius_clamp"_a, "kernel_regularization"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("reset_evaluation_statistics", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::resetEvaluationStatistics,
            "Resets evaluation statistics.")
        .def("get_evaluation_outputs", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::getEvaluationOutputs,
            "Returns evaluation outputs (resized to be the same size as input evaluation points).",
            "evaluation_outputs"_a)
        .def("update_normal_offset_for_absorbing_boundary",
            &wosx::GPUBoundaryValueCachingSolver<T, DIM>::updateNormalOffsetForAbsorbingBoundary,
            "Updates normal offset for the absorbing boundary.",
            "offset"_a)
        .def("get_absorbing_boundary_samples", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::getAbsorbingBoundarySamples,
            "Returns the absorbing boundary samples.",
            "sample_points"_a, "return_boundary_normal_aligned"_a=false)
        .def("get_reflecting_boundary_samples", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::getReflectingBoundarySamples,
            "Returns the reflecting boundary samples.",
            "sample_points"_a, "return_boundary_normal_aligned"_a=false)
        .def("get_domain_samples", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::getDomainSamples,
            "Returns the domain samples.",
            "sample_points"_a)
        .def("using_persistent_threads", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::usingPersistentThreads,
            "Returns true if using persistent threads.")
        .def("enable_logging", &wosx::GPUBoundaryValueCachingSolver<T, DIM>::enableLogging,
            "Enables/disables logging.",
            "enable"_a);
}

template <typename T, size_t DIM>
void bindGPUDistanceQueries(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    nb::class_<wosx::GPUDistanceQueries<T, DIM>>(utils_m, ("GPUDistanceQueries" + typeStr).c_str())
        .def(nb::init<std::shared_ptr<wosx::GPUTaskHandle<T, DIM>>, bool>(),
            "task_handle"_a, "print_logs"_a=false)
        .def("compute_dist_to_boundary", &wosx::GPUDistanceQueries<T, DIM>::computeDistToBoundary,
            "Computes distance to boundary for a set of points.",
            "points"_a, "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("intersect_boundary", &wosx::GPUDistanceQueries<T, DIM>::intersectBoundary,
            "Computes intersection distance to boundary for a set of rays (returns -1.0 if no intersection).",
            "origins"_a, "directions"_a, "dist_along_ray"_a, "intersection_points"_a, "intersection_normals"_a)
        .def("intersect_absorbing_boundary", &wosx::GPUDistanceQueries<T, DIM>::intersectAbsorbingBoundary,
            "Computes intersection distance to absorbing boundary for a set of rays (returns -1.0 if no intersection).",
            "origins"_a, "directions"_a, "dist_along_ray"_a, "intersection_points"_a, "intersection_normals"_a)
        .def("intersect_reflecting_boundary", &wosx::GPUDistanceQueries<T, DIM>::intersectReflectingBoundary,
            "Computes intersection distance to reflecting boundary for a set of rays (returns -1.0 if no intersection).",
            "origins"_a, "directions"_a, "dist_along_ray"_a, "intersection_points"_a, "intersection_normals"_a);
}

template <typename T, size_t DIM>
void bindGPUDenseGrid(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        nb::class_<wosx::GPUDenseGrid<T, 1, DIM>, wosx::GPUShaderObject>(
            utils_m, ("GPUDenseGrid" + typeStr).c_str())
            .def(nb::init<std::shared_ptr<wosx::GPUSampler>,
                          const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                          const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                          const wosx::Vector<DIM>&, bool>(),
                "sampler"_a, "grid_data"_a, "grid_shape"_a,
                "grid_min"_a, "grid_max"_a, "unordered_access"_a)
            .def("allocate", &wosx::GPUDenseGrid<T, 1, DIM>::allocate,
                "Allocates GPU resources.", "context"_a)
            .def("set_resources", &wosx::GPUDenseGrid<T, 1, DIM>::setResources,
                "Sets GPU resources.", "cursor"_a, "print_logs"_a);

    } else {
        nb::class_<wosx::GPUDenseGrid<float, T::RowsAtCompileTime, DIM>, wosx::GPUShaderObject>(
            utils_m, ("GPUDenseGrid" + typeStr).c_str())
            .def(nb::init<std::shared_ptr<wosx::GPUSampler>,
                          const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                          const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                          const wosx::Vector<DIM>&, bool>(),
                "sampler"_a, "grid_data"_a, "grid_shape"_a,
                "grid_min"_a, "grid_max"_a, "unordered_access"_a)
            .def("allocate", &wosx::GPUDenseGrid<float, T::RowsAtCompileTime, DIM>::allocate,
                "Allocates GPU resources.", "context"_a)
            .def("set_resources", &wosx::GPUDenseGrid<float, T::RowsAtCompileTime, DIM>::setResources,
                "Sets GPU resources.", "cursor"_a, "print_logs"_a);
    }
}
