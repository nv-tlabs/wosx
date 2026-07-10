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

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <wosx/wosx.h>

namespace nb = nanobind;
using namespace nb::literals;

// vector types
using BoolList = std::vector<bool>;
using IntList = std::vector<int>;
using UintList = std::vector<uint32_t>;
using FloatList = std::vector<float>;

template <size_t DIM>
using IntNList = std::vector<wosx::Vectori<DIM>>;
template <size_t DIM>
using FloatNList = std::vector<wosx::Vector<DIM>>;

template <typename T, size_t DIM>
using SamplePointList = std::vector<wosx::SamplePoint<T, DIM>>;
template <typename T, size_t DIM>
using SampleStatisticsList = std::vector<wosx::SampleStatistics<T, DIM>>;
template <typename T, size_t DIM>
using BVCEvaluationPointList = std::vector<wosx::bvc::EvaluationPoint<T, DIM>>;
template <typename T, size_t DIM>
using RWSEvaluationPointList = std::vector<wosx::rws::EvaluationPoint<T, DIM>>;

// opaque types
using FloatIntToBoolFunc = std::function<bool(float, int)>;
using FloatToFloatFunc = std::function<float(float)>;
using IntIntToVoidFunc = std::function<void(int, int)>;
using VoidToFloatFunc = std::function<float()>;
template <size_t DIM>
using IntersectBoundaryFunc = std::function<bool(const wosx::Vector<DIM>&, const wosx::Vector<DIM>&,
                                                 const wosx::Vector<DIM>&, float, bool,
                                                 wosx::IntersectionPoint<DIM>&)>;
template <size_t DIM>
using IntersectUnionedBoundaryFunc = std::function<bool(const wosx::Vector<DIM>&, const wosx::Vector<DIM>&,
                                                        const wosx::Vector<DIM>&, float, bool, bool,
                                                        wosx::IntersectionPoint<DIM>&)>;
template <typename T, size_t DIM>
using FloatNToTypeFunc = std::function<T(const wosx::Vector<DIM>&)>;
template <typename T, size_t DIM>
using FloatNBoolToTypeFunc = std::function<T(const wosx::Vector<DIM>&, bool)>;
template <typename T, size_t DIM>
using FloatNFloatNBoolToTypeFunc = std::function<T(const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool)>;
template <typename T, size_t DIM>
using WalkStateToVoidFunc = std::function<void(const wosx::WalkState<T, DIM>&)>;
template <typename T, size_t DIM>
using WalkCodeStateToTypeFunc = std::function<T(wosx::WalkCompletionCode, const wosx::WalkState<T, DIM>&)>;

NB_MAKE_OPAQUE(FloatIntToBoolFunc)                                     // ignoreCandidateSilhouette
NB_MAKE_OPAQUE(FloatToFloatFunc)                                       // branchTraversalWeight
NB_MAKE_OPAQUE(IntIntToVoidFunc)                                       // reportProgress
NB_MAKE_OPAQUE(VoidToFloatFunc)                                        // computeSignedVolume
NB_MAKE_OPAQUE(IntersectBoundaryFunc<2>)                               // intersectAbsorbingBoundary, intersectReflectingBoundary
NB_MAKE_OPAQUE(IntersectBoundaryFunc<3>)                               // intersectAbsorbingBoundary, intersectReflectingBoundary
NB_MAKE_OPAQUE(IntersectUnionedBoundaryFunc<2>)                        // intersectBoundary
NB_MAKE_OPAQUE(IntersectUnionedBoundaryFunc<3>)                        // intersectBoundary
NB_MAKE_OPAQUE(FloatNToTypeFunc<bool, 2>)                              // insideDomain, insideBoundingDomain, outsideBoundingDomain, hasReflectingBoundaryConditions, insideSolveRegion
NB_MAKE_OPAQUE(FloatNToTypeFunc<bool, 3>)                              // insideDomain, insideBoundingDomain, outsideBoundingDomain, hasReflectingBoundaryConditions, insideSolveRegion

// 1-channel
NB_MAKE_OPAQUE(FloatNToTypeFunc<float, 2>)                             // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<float, 3>)                             // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<float, 2>)                         // computeDistToBoundary, dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<float, 3>)                         // computeDistToBoundary, dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<float, 2>)                   // robinCoeff, robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<float, 3>)                   // robinCoeff, robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<float, 2>)                          // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<float, 3>)                          // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<float, 2>)                      // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<float, 3>)                      // terminalContributionCallback

// 4-channels
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 4>, 2>)             // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 4>, 3>)             // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 4>, 2>)         // dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 4>, 3>)         // dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 4>, 2>)   // robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 4>, 3>)   // robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 4>, 2>)          // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 4>, 3>)          // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 4>, 2>)      // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 4>, 3>)      // terminalContributionCallback

// 16-channels
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 16>, 2>)            // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 16>, 3>)            // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 16>, 2>)        // dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 16>, 3>)        // dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 16>, 2>)  // robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 16>, 3>)  // robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 16>, 2>)         // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 16>, 3>)         // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 16>, 2>)     // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 16>, 3>)     // terminalContributionCallback

// 64-channels
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 64>, 2>)            // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 64>, 3>)            // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 64>, 2>)        // dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 64>, 3>)        // dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 64>, 2>)  // robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 64>, 3>)  // robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 64>, 2>)         // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 64>, 3>)         // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 64>, 2>)     // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 64>, 3>)     // terminalContributionCallback

// 256-channels
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 256>, 2>)           // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<wosx::Array<float, 256>, 3>)           // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 256>, 2>)       // dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<wosx::Array<float, 256>, 3>)       // dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 256>, 2>) // robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<wosx::Array<float, 256>, 3>) // robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 256>, 2>)        // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<wosx::Array<float, 256>, 3>)        // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 256>, 2>)    // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<wosx::Array<float, 256>, 3>)    // terminalContributionCallback

// binding functions
void bindNonTemplatedLibraryResources(nb::module_ m);

template <size_t DIM>
void bindIntersectBoundaryFuncs(nb::module_ m);
template <typename T, size_t DIM>
void bindFloatNToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindFloatNFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkStateFuncs(nb::module_ m, std::string typeStr="");

template <typename T, size_t DIM>
void bindDenseGrid(nb::module_ m, std::string typeStr="");

template <size_t DIM>
void bindCoreGeometryStructures(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindGeometryUtilityFunctions(nb::module_ m, std::string typeStr=""); // currently valid only for 2D or 3D

template <size_t DIM>
void bindPDEIndicatorCallbacks(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindPDECoefficientCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDESouceCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDEDirichletCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDERobinCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDEStructure(nb::module_ m, std::string typeStr="");

template <typename T, size_t DIM>
void bindRandomWalkStructures(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkOnSpheresSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkOnStarsSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindSamplers(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindBoundaryValueCachingSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindReverseWalkOnStarsSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindKelvinTransform(nb::module_ m, std::string typeStr="");

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename List, typename T>
nb::ndarray<nb::numpy, T> convertListToNumpyArray(const List& v)
{
    std::cerr << "convertListToNumpyArray: Unsupported type" << std::endl;
    exit(EXIT_FAILURE);
}

template <>
inline nb::ndarray<nb::numpy, bool> convertListToNumpyArray(const BoolList& v)
{
    // allocate a memory region an initialize it
    bool *data = new bool[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (bool *)p;
    });

    return nb::ndarray<nb::numpy, bool>(data, { v.size() }, owner);
}

template <>
inline nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntList& v)
{
    // allocate a memory region an initialize it
    int *data = new int[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size() }, owner);
}

template <>
inline nb::ndarray<nb::numpy, uint32_t> convertListToNumpyArray(const UintList& v)
{
    // allocate a memory region an initialize it
    uint32_t *data = new uint32_t[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (uint32_t *)p;
    });

    return nb::ndarray<nb::numpy, uint32_t>(data, { v.size() }, owner);
}

template <>
inline nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatList& v)
{
    // allocate a memory region an initialize it
    float *data = new float[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size() }, owner);
}

template <>
inline nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntNList<2>& v)
{
    // allocate a memory region an initialize it
    int *data = new int[2*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[2*i + 0] = v[i][0];
        data[2*i + 1] = v[i][1];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size(), 2 }, owner);
}

template <>
inline nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntNList<3>& v)
{
    // allocate a memory region an initialize it
    int *data = new int[3*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[3*i + 0] = v[i][0];
        data[3*i + 1] = v[i][1];
        data[3*i + 2] = v[i][2];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size(), 3 }, owner);
}

template <>
inline nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatNList<2>& v)
{
    // allocate a memory region an initialize it
    float *data = new float[2*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[2*i + 0] = v[i][0];
        data[2*i + 1] = v[i][1];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size(), 2 }, owner);
}

template <>
inline nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatNList<3>& v)
{
    // allocate a memory region an initialize it
    float *data = new float[3*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[3*i + 0] = v[i][0];
        data[3*i + 1] = v[i][1];
        data[3*i + 2] = v[i][2];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size(), 3 }, owner);
}

inline void bindNonTemplatedLibraryResources(nb::module_ m)
{
    nb::bind_vector<BoolList>(m, "BoolList");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<BoolList, bool>);

    nb::bind_vector<IntList>(m, "IntList");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<IntList, int>);

    nb::bind_vector<UintList>(m, "UintList");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<UintList, uint32_t>);

    nb::bind_vector<FloatList>(m, "FloatList");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<FloatList, float>);

    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    nb::class_<FloatIntToBoolFunc>(opaque_types_m, "func<bool(float, int)>")
        .def("__call__", [](const FloatIntToBoolFunc& callback,
                            float a, int b) -> bool {
            return callback(a, b);
        });

    nb::class_<FloatToFloatFunc>(opaque_types_m, "func<float(float)>")
        .def("__call__", [](const FloatToFloatFunc& callback,
                            float a) -> float {
            return callback(a);
        });

    nb::class_<IntIntToVoidFunc>(opaque_types_m, "func<void(int, int)>")
        .def("__call__", [](const IntIntToVoidFunc& callback,
                            int a, int b) -> void {
            callback(a, b);
        });

    nb::class_<VoidToFloatFunc>(opaque_types_m, "func<float(void)>")
        .def("__call__", [](const VoidToFloatFunc& callback) -> float {
            return callback();
        });

    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::enum_<wosx::SampleType>(solvers_m, "SampleType")
        .value("InDomain", wosx::SampleType::InDomain)
        .value("OnAbsorbingBoundary", wosx::SampleType::OnAbsorbingBoundary)
        .value("OnReflectingBoundary", wosx::SampleType::OnReflectingBoundary);

    nb::enum_<wosx::EstimationQuantity>(solvers_m, "EstimationQuantity")
        .value("Solution", wosx::EstimationQuantity::Solution)
        .value("SolutionAndGradient", wosx::EstimationQuantity::SolutionAndGradient)
        .value("Skip", wosx::EstimationQuantity::Skip);

    nb::enum_<wosx::WalkCompletionCode>(solvers_m, "WalkCompletionCode")
        .value("ReachedAbsorbingBoundary", wosx::WalkCompletionCode::ReachedAbsorbingBoundary)
        .value("TerminatedWithRussianRoulette", wosx::WalkCompletionCode::TerminatedWithRussianRoulette)
        .value("ExceededMaxWalkLength", wosx::WalkCompletionCode::ExceededMaxWalkLength)
        .value("EscapedDomain", wosx::WalkCompletionCode::EscapedDomain);

    nb::class_<wosx::WalkSettings>(solvers_m, "WalkSettings")
        .def(nb::init<float, float, int, bool>(),
            "epsilon_shell_for_absorbing_boundary"_a, "epsilon_shell_for_reflecting_boundary"_a,
            "max_walk_length"_a, "solve_double_sided"_a)
        .def(nb::init<float, float, float, float, float, int, int, int,
                      bool, bool, bool, bool, bool, bool, bool, bool>(),
            "epsilon_shell_for_absorbing_boundary"_a, "epsilon_shell_for_reflecting_boundary"_a,
            "silhouette_precision"_a, "russian_roulette_threshold"_a, "splitting_threshold"_a,
            "max_walk_length"_a, "steps_before_applying_tikhonov"_a,
            "steps_before_using_maximal_spheres"_a, "solve_double_sided"_a,
            "use_gradient_control_variates"_a, "use_gradient_antithetic_variates"_a,
            "use_cosine_sampling_for_derivatives"_a, "ignore_absorbing_boundary_contribution"_a,
            "ignore_reflecting_boundary_contribution"_a, "ignore_source_contribution"_a, "print_logs"_a)
        .def_rw("epsilon_shell_for_absorbing_boundary", &wosx::WalkSettings::epsilonShellForAbsorbingBoundary)
        .def_rw("epsilon_shell_for_reflecting_boundary", &wosx::WalkSettings::epsilonShellForReflectingBoundary)
        .def_rw("silhouette_precision", &wosx::WalkSettings::silhouettePrecision)
        .def_rw("russian_roulette_threshold", &wosx::WalkSettings::russianRouletteThreshold)
        .def_rw("splitting_threshold", &wosx::WalkSettings::splittingThreshold)
        .def_rw("max_walk_length", &wosx::WalkSettings::maxWalkLength)
        .def_rw("steps_before_applying_tikhonov", &wosx::WalkSettings::stepsBeforeApplyingTikhonov)
        .def_rw("steps_before_using_maximal_spheres", &wosx::WalkSettings::stepsBeforeUsingMaximalSpheres)
        .def_rw("solve_double_sided", &wosx::WalkSettings::solveDoubleSided)
        .def_rw("use_gradient_control_variates", &wosx::WalkSettings::useGradientControlVariates)
        .def_rw("use_gradient_antithetic_variates", &wosx::WalkSettings::useGradientAntitheticVariates)
        .def_rw("use_cosine_sampling_for_derivatives", &wosx::WalkSettings::useCosineSamplingForDerivatives)
        .def_rw("ignore_absorbing_boundary_contribution", &wosx::WalkSettings::ignoreAbsorbingBoundaryContribution)
        .def_rw("ignore_reflecting_boundary_contribution", &wosx::WalkSettings::ignoreReflectingBoundaryContribution)
        .def_rw("ignore_source_contribution", &wosx::WalkSettings::ignoreSourceContribution)
        .def_rw("print_logs", &wosx::WalkSettings::printLogs);

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    utils_m.def("get_ignore_candidate_silhouette_callback",
               &wosx::getIgnoreCandidateSilhouetteCallback,
               "solve_double_sided"_a=false, "silhouette_precision"_a=1e-3f,
               "Returns a callback that ignores silhouette candidates on the reflecting boundary based on the given parameters.");

    utils_m.def("get_branch_traversal_weight_callback",
               &wosx::getBranchTraversalWeightCallback,
               "min_radial_distance"_a=1e-2f,
               "Returns a branch traversal weight function for a reflecting boundary.");

    nb::class_<wosx::ProgressBar>(utils_m, "ProgressBar")
        .def(nb::init<int, int>(),
            "total_work"_a, "display_width"_a=80)
        .def("report", &wosx::ProgressBar::report,
            "Reports progress on the progress bar.",
            "new_work_completed"_a, "thread_id"_a)
        .def("finish", &wosx::ProgressBar::finish,
            "Finishes the progress bar.");

    utils_m.def("get_report_progress_callback",
               &wosx::getReportProgressCallback,
               "progress_bar"_a,
               "Returns a callback that reports progress using a progress bar.\nNOTE: the returned callback references the progress bar, which must be kept alive while the callback is in use.");
}

template <size_t DIM>
void bindIntersectBoundaryFuncs(nb::module_ m)
{
    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string intersectionPointStr = "IntersectionPoint_" + std::to_string(DIM) + "d";
    std::string funcStr = "func<bool(" + vectorStr + ", " + vectorStr + ", " + vectorStr + ", float, bool, " + intersectionPointStr + ")>";
    nb::class_<IntersectBoundaryFunc<DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const IntersectBoundaryFunc<DIM>& callback,
                            const wosx::Vector<DIM>& a, const wosx::Vector<DIM>& b,
                            const wosx::Vector<DIM>& c, float d, bool e,
                            wosx::IntersectionPoint<DIM>& f) -> bool {
            return callback(a, b, c, d, e, f);
        });

    funcStr = "func<bool(" + vectorStr + ", " + vectorStr + ", " + vectorStr + ", float, bool, bool, " + intersectionPointStr + ")>";
    nb::class_<IntersectUnionedBoundaryFunc<DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const IntersectUnionedBoundaryFunc<DIM>& callback,
                            const wosx::Vector<DIM>& a, const wosx::Vector<DIM>& b,
                            const wosx::Vector<DIM>& c, float d, bool e, bool f,
                            wosx::IntersectionPoint<DIM>& g) -> bool {
            return callback(a, b, c, d, e, f, g);
        });
}

template <typename T, size_t DIM>
void bindFloatNToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ")>";
    nb::class_<FloatNToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNToTypeFunc<T, DIM>& callback,
                            const wosx::Vector<DIM>& a) -> T {
            return callback(a);
        });
}

template <typename T, size_t DIM>
void bindFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ", bool)>";
    nb::class_<FloatNBoolToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNBoolToTypeFunc<T, DIM>& callback,
                            const wosx::Vector<DIM>& a, bool b) -> T {
            return callback(a, b);
        });
}

template <typename T, size_t DIM>
void bindFloatNFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ", " + vectorStr + ", bool)>";
    nb::class_<FloatNFloatNBoolToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNFloatNBoolToTypeFunc<T, DIM>& callback,
                            const wosx::Vector<DIM>& a,
                            const wosx::Vector<DIM>& b, bool c) -> T {
            return callback(a, b, c);
        });
}

template <typename T, size_t DIM>
void bindWalkStateFuncs(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("OpaqueTypes", "Opaque types module");

    std::string funcStr = "func<void(WalkState_" + typeStr + "_" + std::to_string(DIM) + "d)>";
    nb::class_<WalkStateToVoidFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const WalkStateToVoidFunc<T, DIM>& callback,
                            const wosx::WalkState<T, DIM>& a) -> void {
            callback(a);
        });

    funcStr = "func<" + typeStr + "(WalkCompletionCode, WalkState_" + typeStr + "_" + std::to_string(DIM) + "d)>";
    nb::class_<WalkCodeStateToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const WalkCodeStateToTypeFunc<T, DIM>& callback,
                            wosx::WalkCompletionCode a,
                            const wosx::WalkState<T, DIM>& b) -> T {
            return callback(a, b);
        });
}

template <size_t DIM>
void bindCoreGeometryStructures(nb::module_ m, std::string typeStr)
{
    nb::bind_vector<IntNList<DIM>>(m, ("Int" + std::to_string(DIM) + "List").c_str());
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<IntNList<DIM>, int>);

    nb::bind_vector<FloatNList<DIM>>(m, ("Float" + std::to_string(DIM) + "List").c_str());
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<FloatNList<DIM>, float>);

    nb::module_ core_m = m.def_submodule("Core", "Core module");

    nb::class_<wosx::IntersectionPoint<DIM>>(core_m, ("IntersectionPoint" + typeStr).c_str())
        .def(nb::init<>())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, float>(),
            "pt"_a, "normal"_a, "dist"_a)
        .def_rw("pt", &wosx::IntersectionPoint<DIM>::pt)
        .def_rw("normal", &wosx::IntersectionPoint<DIM>::normal)
        .def_rw("dist", &wosx::IntersectionPoint<DIM>::dist);

    nb::class_<wosx::GeometricQueries<DIM>>(core_m, ("GeometricQueries" + typeStr).c_str())
        .def(nb::init<>())
        .def(nb::init<bool, const wosx::Vector<DIM>&, const wosx::Vector<DIM>&>(),
            "domain_is_watertight"_a, "domain_min"_a, "domain_max"_a)
        .def_ro("has_non_empty_absorbing_boundary", &wosx::GeometricQueries<DIM>::hasNonEmptyAbsorbingBoundary)
        .def_ro("has_non_empty_reflecting_boundary", &wosx::GeometricQueries<DIM>::hasNonEmptyReflectingBoundary)
        .def_ro("domain_is_watertight", &wosx::GeometricQueries<DIM>::domainIsWatertight)
        .def_ro("domain_min", &wosx::GeometricQueries<DIM>::domainMin)
        .def_ro("domain_max", &wosx::GeometricQueries<DIM>::domainMax)
        .def_ro("compute_dist_to_absorbing_boundary", &wosx::GeometricQueries<DIM>::computeDistToAbsorbingBoundary)
        .def_ro("compute_dist_to_reflecting_boundary", &wosx::GeometricQueries<DIM>::computeDistToReflectingBoundary)
        .def_ro("compute_dist_to_boundary", &wosx::GeometricQueries<DIM>::computeDistToBoundary)
        .def_ro("intersect_absorbing_boundary", &wosx::GeometricQueries<DIM>::intersectAbsorbingBoundary)
        .def_ro("intersect_reflecting_boundary", &wosx::GeometricQueries<DIM>::intersectReflectingBoundary)
        .def_ro("intersect_boundary", &wosx::GeometricQueries<DIM>::intersectBoundary)
        .def_ro("inside_domain", &wosx::GeometricQueries<DIM>::insideDomain)
        .def_ro("inside_bounding_domain", &wosx::GeometricQueries<DIM>::insideBoundingDomain)
        .def_ro("outside_bounding_domain", &wosx::GeometricQueries<DIM>::outsideBoundingDomain)
        .def_ro("compute_domain_signed_volume", &wosx::GeometricQueries<DIM>::computeDomainSignedVolume);
}

template <typename T, size_t DIM>
void bindDenseGrid(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        nb::class_<wosx::DenseGrid<T, 1, DIM>>(utils_m, ("DenseGrid" + typeStr).c_str())
            .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool>(),
                "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false)
            .def(nb::init<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                          const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                          const wosx::Vector<DIM>&, bool>(),
                "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                "enable_interpolation"_a=false)
            .def("set", nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                          const wosx::Vectori<DIM>&>(
                &wosx::DenseGrid<T, 1, DIM>::set),
                "sets the grid data.",
                "grid_data"_a, "grid_shape"_a)
            .def("set", nb::overload_cast<int, const wosx::Array<T, 1>&>(
                &wosx::DenseGrid<T, 1, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("set", nb::overload_cast<const wosx::Vectori<DIM>&,
                                          const wosx::Array<T, 1>&>(
                &wosx::DenseGrid<T, 1, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("get", nb::overload_cast<int>(
                &wosx::DenseGrid<T, 1, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("get", nb::overload_cast<const wosx::Vectori<DIM>&>(
                &wosx::DenseGrid<T, 1, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("__call__", &wosx::DenseGrid<T, 1, DIM>::operator(),
                "returns the grid value at a point.",
                "x"_a)
            .def("min", &wosx::DenseGrid<T, 1, DIM>::min,
                "returns the minimum grid value.")
            .def("max", &wosx::DenseGrid<T, 1, DIM>::max,
                "returns the maximum grid value.")
            .def_ro("data", &wosx::DenseGrid<T, 1, DIM>::data)
            .def_ro("shape", &wosx::DenseGrid<T, 1, DIM>::shape)
            .def_ro("origin", &wosx::DenseGrid<T, 1, DIM>::origin)
            .def_ro("extent", &wosx::DenseGrid<T, 1, DIM>::extent);

    } else {
        nb::class_<wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>>(utils_m, ("DenseGrid" + typeStr).c_str())
            .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, bool>(),
                "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false)
            .def(nb::init<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                          const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                          const wosx::Vector<DIM>&, bool>(),
                "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                "enable_interpolation"_a=false)
            .def("set", nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                          const wosx::Vectori<DIM>&>(
                &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid data.",
                "grid_data"_a, "grid_shape"_a)
            .def("set", nb::overload_cast<int, const wosx::Array<float, T::RowsAtCompileTime>&>(
                &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("set", nb::overload_cast<const wosx::Vectori<DIM>&,
                                          const wosx::Array<float, T::RowsAtCompileTime>&>(
                &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("get", nb::overload_cast<int>(
                &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("get", nb::overload_cast<const wosx::Vectori<DIM>&>(
                &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("__call__", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::operator(),
                "returns the grid value at a point.",
                "x"_a)
            .def("min", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::min,
                "returns the minimum grid value.")
            .def("max", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::max,
                "returns the maximum grid value.")
            .def_ro("data", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::data)
            .def_ro("shape", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::shape)
            .def_ro("origin", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::origin)
            .def_ro("extent", &wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>::extent);
    }
}

template <size_t DIM>
void bindGeometryUtilityFunctions(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    utils_m.def(("load_boundary_mesh" + typeStr).c_str(),
               &wosx::loadBoundaryMesh<DIM>,
               "obj_file"_a, "positions"_a, "indices"_a,
               "Loads boundary mesh from OBJ file.");

    if (DIM == 3) {
        utils_m.def(("load_textured_boundary_mesh" + typeStr).c_str(),
                    &wosx::loadTexturedBoundaryMesh<DIM>,
                    "obj_file"_a, "positions"_a, "texture_coordinates"_a,
                    "indices"_a, "texture_indices"_a,
                    "Loads textured boundary mesh from OBJ file.");
    }

    utils_m.def(("normalize" + typeStr).c_str(),
               &wosx::normalize<DIM>,
               "positions"_a,
               "Normalizes positions to the unit sphere.");

    utils_m.def(("apply_shift" + typeStr).c_str(),
               &wosx::applyShift<DIM>,
               "shift"_a, "positions"_a,
               "Applies a shift to a set of positions.");

    utils_m.def(("flip_orientation" + typeStr).c_str(),
               &wosx::flipOrientation<DIM>,
               "indices"_a,
               "Flips the orientation of a boundary mesh.");

    utils_m.def(("compute_bounding_box" + typeStr).c_str(),
               &wosx::computeBoundingBox<DIM>,
               "positions"_a, "make_square"_a, "scale"_a,
               "Computes the bounding box of a boundary mesh.");

    utils_m.def(("build_bounding_box_mesh" + typeStr).c_str(),
               &wosx::buildBoundingBoxMesh<DIM>,
               "bounding_box_min"_a, "bounding_box_max"_a, "positions"_a, "indices"_a,
               "Builds a bounding box mesh.");

    utils_m.def(("add_bounding_box_to_boundary_mesh" + typeStr).c_str(),
               &wosx::addBoundingBoxToBoundaryMesh<DIM>,
               "bounding_box_min"_a, "bounding_box_max"_a, "positions"_a, "indices"_a,
               "Adds a bounding box to a boundary mesh.");

    utils_m.def(("compute_signed_volume" + typeStr).c_str(),
               &wosx::computeSignedVolume<DIM>,
               "boundary_positions"_a, "boundary_indices"_a,
               "Computes the signed volume of a boundary mesh.");

    utils_m.def(("compute_dist_to_boundary" + typeStr).c_str(),
               [](const wosx::GeometricQueries<DIM>& geometricQueries,
                  const FloatNList<DIM>& solveLocations,
                  FloatList& distToAbsorbingBoundary,
                  FloatList& distToReflectingBoundary) {
                   distToAbsorbingBoundary.resize(solveLocations.size());
                   distToReflectingBoundary.resize(solveLocations.size());

                   for (size_t i = 0; i < solveLocations.size(); i++) {
                       distToAbsorbingBoundary[i] =
                           geometricQueries.computeDistToAbsorbingBoundary(solveLocations[i], false);
                       distToReflectingBoundary[i] =
                           geometricQueries.computeDistToReflectingBoundary(solveLocations[i], false);
                   }
               },
               "geometric_queries"_a, "solve_locations"_a,
               "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a,
               "Computes distance to absorbing and reflecting boundaries for a set of locations.");

    utils_m.def(("partition_boundary_mesh" + typeStr).c_str(),
               &wosx::partitionBoundaryMesh<DIM>,
               "on_reflecting_boundary"_a, "positions"_a, "indices"_a,
               "absorbing_positions"_a, "absorbing_indices"_a,
               "reflecting_positions"_a, "reflecting_indices"_a,
               "Partitions a boundary mesh into absorbing and reflecting parts using primitive centroids---\nthis assumes the boundary discretization is perfectly adapted to the boundary conditions,\nwhich isn't always a correct assumption.");

    nb::class_<wosx::FcpwDirichletBoundaryHandler<DIM>>(utils_m, ("FcpwDirichletBoundaryHandler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &wosx::FcpwDirichletBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions and indices.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "build_bvh"_a=true,
            "enable_bvh_vectorization"_a=false, "print_stats"_a=true);

    nb::class_<wosx::FcpwNeumannBoundaryHandler<DIM>>(utils_m, ("FcpwNeumannBoundaryHandler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &wosx::FcpwNeumannBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions and indices.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a,
            "build_bvh"_a=true, "enable_bvh_vectorization"_a=false, "print_stats"_a=true);

    nb::class_<wosx::FcpwRobinBoundaryHandler<DIM>>(utils_m, ("FcpwRobinBoundaryHandler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &wosx::FcpwRobinBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions, indices, and min and max absolute coefficient values per mesh face.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a,
            "min_robin_coeff_values"_a, "max_robin_coeff_values"_a,
            "build_bvh"_a=true, "enable_bvh_vectorization"_a=false, "print_stats"_a=true)
        .def("update_coefficient_values", &wosx::FcpwRobinBoundaryHandler<DIM>::updateCoefficientValues,
            "updates the Robin coefficients on the boundary mesh.",
            "min_robin_coeff_values"_a, "max_robin_coeff_values"_a);

    nb::class_<wosx::SdfGrid<DIM>, wosx::DenseGrid<float, 1, DIM>>(utils_m, ("SDFGrid" + typeStr).c_str())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&>(),
            "grid_min"_a, "grid_max"_a)
        .def(nb::init<const Eigen::VectorXf&, const wosx::Vectori<DIM>&,
                      const wosx::Vector<DIM>&, const wosx::Vector<DIM>&>(),
            "sdf_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a)
        .def("compute_gradient", &wosx::SdfGrid<DIM>::computeGradient,
            "Computes the gradient of the SDF at the given point.",
            "x"_a)
        .def("compute_normal", &wosx::SdfGrid<DIM>::computeNormal,
            "Computes the normal of the level set at the given point.", 
            "x"_a)
        .def("project_to_zero_level_set", &wosx::SdfGrid<DIM>::projectToZeroLevelSet,
            "Projects the given point to the zero level set.",
            "x"_a, "normal"_a, "max_iterations"_a=8, "epsilon"_a=1e-6f)
        .def("intersect_zero_level_set", &wosx::SdfGrid<DIM>::intersectZeroLevelSet,
            "Intersects the zero level set with the given ray.",
            "origin"_a, "direction"_a, "t_max"_a, "intersection_pt"_a,
            "max_iterations"_a=128, "epsilon"_a=1e-6f);

    utils_m.def(("populate_sdf_grid" + typeStr).c_str(),
               &wosx::populateSdfGrid<DIM>,
               "dirichlet_boundary_handler"_a, "sdf_grid"_a, "grid_shape"_a,
               "compute_unsigned_distance"_a=false,
               "Populates an SDF grid from a Dirichlet boundary handler.");

    utils_m.def(("populate_geometric_queries_for_dirichlet_boundary" + typeStr).c_str(),
               nb::overload_cast<wosx::FcpwDirichletBoundaryHandler<DIM>&,
                                 wosx::GeometricQueries<DIM>&>(
               &wosx::populateGeometricQueriesForDirichletBoundary<DIM>),
               "fcpw_dirichlet_boundary_handler"_a, "geometric_queries"_a,
               "Populates geometric queries for an absorbing Dirichlet boundary.\nNOTE: the populated queries reference the boundary handler, which must be kept alive for the lifetime of the geometric queries.");

    utils_m.def(("populate_geometric_queries_for_dirichlet_boundary" + typeStr).c_str(),
               nb::overload_cast<const wosx::SdfGrid<DIM>&, wosx::GeometricQueries<DIM>&>(
               &wosx::populateGeometricQueriesForDirichletBoundary<wosx::SdfGrid<DIM>, DIM>),
               "sdf_grid"_a, "geometric_queries"_a,
               "Populates geometric queries for an absorbing Dirichlet boundary.\nNOTE: the populated queries reference the sdf grid, which must be kept alive for the lifetime of the geometric queries.");

    utils_m.def(("populate_geometric_queries_for_neumann_boundary" + typeStr).c_str(),
               &wosx::populateGeometricQueriesForNeumannBoundary<DIM>,
               "fcpw_neumann_boundary_handler"_a, "branch_traversal_weight"_a, "geometric_queries"_a,
               "Populates geometric queries for a reflecting Neumann boundary.\nNOTE: the populated queries reference the boundary handler, which must be kept alive for the lifetime of the geometric queries.");

    utils_m.def(("populate_geometric_queries_for_robin_boundary" + typeStr).c_str(),
               &wosx::populateGeometricQueriesForRobinBoundary<DIM>,
               "fcpw_robin_boundary_handler"_a, "branch_traversal_weight"_a, "geometric_queries"_a,
               "Populates geometric queries for a reflecting Robin boundary.\nNOTE: the populated queries reference the boundary handler, which must be kept alive for the lifetime of the geometric queries.");

    utils_m.def(("get_spatially_sorted_point_indices" + typeStr).c_str(),
               &wosx::getSpatiallySortedPointIndices<DIM>,
               "points"_a, "out_indices"_a,
               "Outputs a list of indices that spatially sort the input points.");
}

template <size_t DIM>
void bindPDEIndicatorCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    core_m.def(("get_constant_indicator_callback" + typeStr).c_str(),
              [](bool value) -> FloatNToTypeFunc<bool, DIM> {
                return [value](const wosx::Vector<DIM>& a) -> bool { return value; };
              },
              "value"_a,
              "Returns a constant indicator callback.");

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    utils_m.def(("get_dense_grid_indicator_callback" + typeStr).c_str(),
               nb::overload_cast<const wosx::DenseGrid<bool, 1, DIM>&>(
               &wosx::getDenseGridCallback0<bool, bool, 1, DIM>),
               "grid"_a,
               "Returns a dense grid indicator callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
    utils_m.def(("get_dense_grid_indicator_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<bool, Eigen::Dynamic, 1>&,
                                 const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                 const wosx::Vector<DIM>&, bool>(
               &wosx::getDenseGridCallback0<bool, bool, 1, DIM>),
               "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
               "enable_interpolation"_a=false,
               "Returns a dense grid indicator callback.");
}

template <size_t DIM>
void bindPDECoefficientCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    core_m.def(("get_constant_robin_coefficient_callback" + typeStr).c_str(),
              [](float value) -> FloatNFloatNBoolToTypeFunc<float, DIM> {
                return [value](const wosx::Vector<DIM>& a,
                               const wosx::Vector<DIM>& b, bool c) -> float {
                    return value;
                };
              },
              "value"_a,
              "Returns a constant coefficient callback.");
    core_m.def(("get_constant_robin_coefficient_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNFloatNBoolToTypeFunc<float, DIM> {
                return [value, valueBoundaryNormalAligned](const wosx::Vector<DIM>& a,
                                                           const wosx::Vector<DIM>& b, bool c) -> float {
                    return c ? valueBoundaryNormalAligned : value;
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant coefficient callback.");

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const wosx::DenseGrid<float, 1, DIM>&>(
               &wosx::getDenseGridCallback3<float, float, 1, DIM>),
               "grid"_a,
               "Returns a dense grid coefficient callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                 const wosx::Vector<DIM>&, bool>(
               &wosx::getDenseGridCallback3<float, float, 1, DIM>),
               "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
               "enable_interpolation"_a=false,
               "Returns a dense grid coefficient callback.");

    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const wosx::DenseGrid<float, 1, DIM>&,
                                 const wosx::DenseGrid<float, 1, DIM>&>(
               &wosx::getDenseGridCallback4<float, float, 1, DIM>),
               "grid"_a, "grid_boundary_normal_aligned"_a,
               "Returns a dense grid coefficient callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                 const wosx::Vector<DIM>&, bool>(
               &wosx::getDenseGridCallback4<float, float, 1, DIM>),
               "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
               "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
               "Returns a dense grid coefficient callback.");
}

template <typename T, size_t DIM>
void bindPDESouceCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    core_m.def(("get_constant_source_callback" + typeStr).c_str(),
              [](float value) -> FloatNToTypeFunc<T, DIM> {
                return [value](const wosx::Vector<DIM>& a) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant source callback.");

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<T, 1, DIM>&>(
                   &wosx::getDenseGridCallback0<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid source callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback0<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid source callback.");

    } else {
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &wosx::getDenseGridCallback0<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid source callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback0<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid source callback.");
    }
}

template <typename T, size_t DIM>
void bindPDEDirichletCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    core_m.def(("get_constant_dirichlet_callback" + typeStr).c_str(),
              [](float value) -> FloatNBoolToTypeFunc<T, DIM> {
                return [value](const wosx::Vector<DIM>& a, bool b) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant dirichlet boundary condition callback.");
    core_m.def(("get_constant_dirichlet_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNBoolToTypeFunc<T, DIM> {
                return [value, valueBoundaryNormalAligned](const wosx::Vector<DIM>& a, bool b) -> T {
                    return b ? T(valueBoundaryNormalAligned) : T(value);
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant dirichlet boundary condition callback.");

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<T, 1, DIM>&>(
                   &wosx::getDenseGridCallback1<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid dirichlet boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback1<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<T, 1, DIM>&,
                                     const wosx::DenseGrid<T, 1, DIM>&>(
                   &wosx::getDenseGridCallback2<T, T, 1, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid dirichlet boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback2<T, T, 1, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

    } else {
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &wosx::getDenseGridCallback1<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid dirichlet boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback1<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&,
                                     const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &wosx::getDenseGridCallback2<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid dirichlet boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback2<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");
    }
}

template <typename T, size_t DIM>
void bindPDERobinCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    core_m.def(("get_constant_robin_callback" + typeStr).c_str(),
              [](float value) -> FloatNFloatNBoolToTypeFunc<T, DIM> {
                return [value](const wosx::Vector<DIM>& a,
                               const wosx::Vector<DIM>& b, bool c) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant robin boundary condition callback.");
    core_m.def(("get_constant_robin_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNFloatNBoolToTypeFunc<T, DIM> {
                return [value, valueBoundaryNormalAligned](const wosx::Vector<DIM>& a,
                                                           const wosx::Vector<DIM>& b, bool c) -> T {
                    return c ? T(valueBoundaryNormalAligned) : T(value);
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant robin boundary condition callback.");

    nb::module_ utils_m = m.def_submodule("Utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<T, 1, DIM>&>(
                   &wosx::getDenseGridCallback3<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid robin boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback3<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<T, 1, DIM>&,
                                     const wosx::DenseGrid<T, 1, DIM>&>(
                   &wosx::getDenseGridCallback4<T, T, 1, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid robin boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback4<T, T, 1, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

    } else {
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &wosx::getDenseGridCallback3<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid robin boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback3<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&,
                                     const wosx::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &wosx::getDenseGridCallback4<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid robin boundary condition callback.\nNOTE: the returned callback references the grid(s), which must be kept alive while the callback is in use.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const wosx::Vectori<DIM>&, const wosx::Vector<DIM>&,
                                     const wosx::Vector<DIM>&, bool>(
                   &wosx::getDenseGridCallback4<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");
    }
}

template <typename T, size_t DIM>
void bindPDEStructure(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("Core", "Core module");

    nb::class_<wosx::PDE<T, DIM>>(core_m, ("PDE" + typeStr).c_str())
        .def(nb::init<>())
        .def_rw("absorption_coeff", &wosx::PDE<T, DIM>::absorptionCoeff)
        .def_rw("is_source_constant", &wosx::PDE<T, DIM>::isSourceConstant)
        .def_rw("are_robin_conditions_pure_neumann", &wosx::PDE<T, DIM>::areRobinConditionsPureNeumann)
        .def_rw("are_robin_coeffs_nonnegative", &wosx::PDE<T, DIM>::areRobinCoeffsNonnegative)
        .def_rw("source", &wosx::PDE<T, DIM>::source)
        .def_rw("dirichlet", &wosx::PDE<T, DIM>::dirichlet)
        .def_rw("robin", &wosx::PDE<T, DIM>::robin)
        .def_rw("robin_coeff", &wosx::PDE<T, DIM>::robinCoeff)
        .def_rw("has_reflecting_boundary_conditions", &wosx::PDE<T, DIM>::hasReflectingBoundaryConditions);
}

template <typename T, size_t DIM>
void bindRandomWalkStructures(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::WalkState<T, DIM>>(solvers_m, ("WalkState" + typeStr).c_str())
        .def_ro("current_pt", &wosx::WalkState<T, DIM>::currentPt)
        .def_ro("current_normal", &wosx::WalkState<T, DIM>::currentNormal)
        .def_ro("prev_direction", &wosx::WalkState<T, DIM>::prevDirection)
        .def_ro("prev_distance", &wosx::WalkState<T, DIM>::prevDistance)
        .def_ro("walk_length", &wosx::WalkState<T, DIM>::walkLength)
        .def_ro("on_reflecting_boundary", &wosx::WalkState<T, DIM>::onReflectingBoundary);

    nb::class_<wosx::SampleStatistics<T, DIM>>(solvers_m, ("SampleStatistics" + typeStr).c_str())
        .def(nb::init<>())
        .def("reset", &wosx::SampleStatistics<T, DIM>::reset,
            "Resets statistics.")
        .def("get_estimated_solution", &wosx::SampleStatistics<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", nb::overload_cast<int>(
            &wosx::SampleStatistics<T, DIM>::getEstimatedGradient, nb::const_),
            "Returns estimated gradient for specified channel.",
            "channel"_a)
        .def("get_estimated_derivative", &wosx::SampleStatistics<T, DIM>::getEstimatedDerivative,
            "Returns estimated directional derivative.")
        .def("get_solution_estimate_count", &wosx::SampleStatistics<T, DIM>::getSolutionEstimateCount,
            "Returns number of solution estimates.")
        .def("get_gradient_estimate_count", &wosx::SampleStatistics<T, DIM>::getGradientEstimateCount,
            "Returns number of gradient estimates.")
        .def("get_mean_walk_length", &wosx::SampleStatistics<T, DIM>::getMeanWalkLength,
            "Returns mean walk length.");

    nb::class_<wosx::SamplePoint<T, DIM>>(solvers_m, ("SamplePoint" + typeStr).c_str())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, wosx::SampleType,
                      wosx::EstimationQuantity, float, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "estimation_quantity"_a, "pdf"_a,
            "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("reset", &wosx::SamplePoint<T, DIM>::reset,
            "Resets the solution data.")
        .def_rw("pt", &wosx::SamplePoint<T, DIM>::pt)
        .def_rw("normal", &wosx::SamplePoint<T, DIM>::normal)
        .def_rw("direction_for_derivative", &wosx::SamplePoint<T, DIM>::directionForDerivative)
        .def_rw("type", &wosx::SamplePoint<T, DIM>::type)
        .def_rw("estimation_quantity", &wosx::SamplePoint<T, DIM>::estimationQuantity)
        .def_rw("pdf", &wosx::SamplePoint<T, DIM>::pdf)
        .def_rw("dist_to_absorbing_boundary", &wosx::SamplePoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &wosx::SamplePoint<T, DIM>::distToReflectingBoundary)
        .def_ro("first_sphere_radius", &wosx::SamplePoint<T, DIM>::firstSphereRadius)
        .def_rw("estimate_boundary_normal_aligned", &wosx::SamplePoint<T, DIM>::estimateBoundaryNormalAligned);

    nb::bind_vector<SamplePointList<T, DIM>>(solvers_m, ("SamplePointList" + typeStr).c_str());
    nb::bind_vector<SampleStatisticsList<T, DIM>>(solvers_m, ("SampleStatisticsList" + typeStr).c_str());

    solvers_m.def(("create_sample_statistics_list" + typeStr).c_str(),
                 [](int size) -> SampleStatisticsList<T, DIM> {
                    return SampleStatisticsList<T, DIM>(static_cast<size_t>(size));
                 },
                 "size"_a,
                 "Creates a SampleStatisticsList with the given size.");

    solvers_m.def(("get_empty_walk_state_callback" + typeStr).c_str(),
                 []() -> WalkStateToVoidFunc<T, DIM> { return {}; },
                 "Returns an empty walk state callback.");

    solvers_m.def(("get_empty_terminal_contribution_callback" + typeStr).c_str(),
                 []() -> WalkCodeStateToTypeFunc<T, DIM> { return {}; },
                 "Returns an empty terminal contribution callback.");
}

template <typename T, size_t DIM>
void bindWalkOnSpheresSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::WalkOnSpheres<T, DIM>>(solvers_m, ("WalkOnSpheres" + typeStr).c_str())
        .def(nb::init<const wosx::GeometricQueries<DIM>&>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a)
        .def(nb::init<const wosx::GeometricQueries<DIM>&, WalkStateToVoidFunc<T, DIM>, WalkCodeStateToTypeFunc<T, DIM>>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a, "walk_state_callback"_a, "terminal_contribution_callback"_a)
        .def("solve", nb::overload_cast<const wosx::PDE<T, DIM>&, const wosx::WalkSettings&, int,
                                        wosx::SamplePoint<T, DIM>&, wosx::SampleStatistics<T, DIM>&>(
            &wosx::WalkOnSpheres<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input point.\nAssumes the point does not lie on the boundary when estimating the gradient.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pt"_a, "statistics"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("solve", nb::overload_cast<const wosx::PDE<T, DIM>&, const wosx::WalkSettings&, const IntList&,
                                        SamplePointList<T, DIM>&, SampleStatisticsList<T, DIM>&, bool, IntIntToVoidFunc>(
            &wosx::WalkOnSpheres<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input points.\nAssumes points do not lie on the boundary when estimating gradients.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pts"_a, "statistics"_a,
            "run_single_threaded"_a=false, "report_progress"_a,
            nb::call_guard<nb::gil_scoped_release>());
}

template <typename T, size_t DIM>
void bindWalkOnStarsSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::WalkOnStars<T, DIM>>(solvers_m, ("WalkOnStars" + typeStr).c_str())
        .def(nb::init<const wosx::GeometricQueries<DIM>&>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a)
        .def(nb::init<const wosx::GeometricQueries<DIM>&, WalkStateToVoidFunc<T, DIM>, WalkCodeStateToTypeFunc<T, DIM>>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a, "walk_state_callback"_a, "terminal_contribution_callback"_a)
        .def("solve", nb::overload_cast<const wosx::PDE<T, DIM>&, const wosx::WalkSettings&, int,
                                        wosx::SamplePoint<T, DIM>&, wosx::SampleStatistics<T, DIM>&>(
            &wosx::WalkOnStars<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input point.\nAssumes the point does not lie on the boundary when estimating the gradient.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pt"_a, "statistics"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("solve", nb::overload_cast<const wosx::PDE<T, DIM>&, const wosx::WalkSettings&, const IntList&,
                                        SamplePointList<T, DIM>&, SampleStatisticsList<T, DIM>&, bool, IntIntToVoidFunc>(
            &wosx::WalkOnStars<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input points.\nAssumes points do not lie on the boundary when estimating gradients.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pts"_a, "statistics"_a,
            "run_single_threaded"_a=false, "report_progress"_a,
            nb::call_guard<nb::gil_scoped_release>());
}

template <typename T, size_t DIM>
void bindSamplers(nb::module_ m, std::string typeStr)
{
    nb::module_ samplers_m = m.def_submodule("Samplers", "Samplers module");

    nb::class_<wosx::BoundarySampler<T, DIM>>(samplers_m, ("BoundarySampler" + typeStr).c_str())
        .def("initialize", &wosx::BoundarySampler<T, DIM>::initialize,
            "Performs any sampler specific initialization.",
            "normal_offset_for_boundary"_a, "solve_double_sided"_a)
        .def("get_sample_count", &wosx::BoundarySampler<T, DIM>::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false)
        .def("generate_samples", &wosx::BoundarySampler<T, DIM>::generateSamples,
            "Generates sample points on the boundary.",
            "n_samples"_a, "sample_type"_a, "normal_offset_for_boundary"_a,
            "geometric_queries"_a, "sample_pts"_a,
            "generate_boundary_normal_aligned_samples"_a=false);

    if (DIM == 2) {
        samplers_m.def(("create_uniform_line_segment_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<2>&, const IntNList<2>&,
                                        FloatNToTypeFunc<bool, 2>, bool>(
                                            &wosx::createUniformLineSegmentBoundarySampler<T>),
                      "positions"_a, "indices"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform line segment boundary sampler.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");
        samplers_m.def(("create_uniform_line_segment_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<2>&, const IntNList<2>&,
                                        const FloatList&, FloatNToTypeFunc<bool, 2>, bool>(
                                            &wosx::createUniformLineSegmentBoundarySampler<T>),
                      "positions"_a, "indices"_a, "primitive_weights"_a,
                      "inside_solve_region"_a, "compute_weighted_normals"_a=false,
                      "Creates a uniform line segment boundary sampler with primitive weights.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");
        samplers_m.def(("create_uniform_line_segment_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<2>&, const IntNList<2>&,
                                        const FloatList&, const FloatList&,
                                        FloatNToTypeFunc<bool, 2>, bool>(
                                            &wosx::createUniformLineSegmentBoundarySampler<T>),
                      "positions"_a, "indices"_a, "primitive_weights"_a,
                      "primitive_weights_normal_aligned"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform line segment boundary sampler with side-specific primitive weights.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");

    } else if (DIM == 3) {
        samplers_m.def(("create_uniform_triangle_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<3>&, const IntNList<3>&,
                                        FloatNToTypeFunc<bool, 3>, bool>(
                                            &wosx::createUniformTriangleBoundarySampler<T>),
                      "positions"_a, "indices"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform triangle boundary sampler.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");
        samplers_m.def(("create_uniform_triangle_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<3>&, const IntNList<3>&,
                                        const FloatList&, FloatNToTypeFunc<bool, 3>, bool>(
                                            &wosx::createUniformTriangleBoundarySampler<T>),
                      "positions"_a, "indices"_a, "primitive_weights"_a,
                      "inside_solve_region"_a, "compute_weighted_normals"_a=false,
                      "Creates a uniform triangle boundary sampler with primitive weights.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");
        samplers_m.def(("create_uniform_triangle_boundary_sampler" + typeStr).c_str(),
                      nb::overload_cast<const FloatNList<3>&, const IntNList<3>&,
                                        const FloatList&, const FloatList&,
                                        FloatNToTypeFunc<bool, 3>, bool>(
                                            &wosx::createUniformTriangleBoundarySampler<T>),
                      "positions"_a, "indices"_a, "primitive_weights"_a,
                      "primitive_weights_normal_aligned"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform triangle boundary sampler with side-specific primitive weights.\nNOTE: the sampler stores references to positions and indices, which must be kept alive for the lifetime of the sampler.");
    }

    nb::class_<wosx::DomainSampler<T, DIM>>(samplers_m, ("DomainSampler" + typeStr).c_str())
        .def("generate_samples", &wosx::DomainSampler<T, DIM>::generateSamples,
            "Generates sample points inside the user-specified solve region.",
            "n_samples"_a, "geometric_queries"_a, "sample_pts"_a);

    samplers_m.def(("create_uniform_domain_sampler" + typeStr).c_str(),
                  &wosx::createUniformDomainSampler<T, DIM>,
                  "inside_solve_region"_a, "solve_region_min"_a,
                  "solve_region_max"_a, "solve_region_volume"_a,
                  "Creates a uniform domain sampler.");
}

template <typename T, size_t DIM>
void bindBoundaryValueCachingSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::bvc::EvaluationPoint<T, DIM>>(solvers_m, ("BVCEvaluationPoint" + typeStr).c_str())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, wosx::SampleType, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("get_estimated_solution", &wosx::bvc::EvaluationPoint<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", nb::overload_cast<int>(
            &wosx::bvc::EvaluationPoint<T, DIM>::getEstimatedGradient, nb::const_),
            "Returns estimated gradient for specified channel.",
            "channel"_a)
        .def("reset", &wosx::bvc::EvaluationPoint<T, DIM>::reset,
            "Resets statistics.")
        .def_rw("pt", &wosx::bvc::EvaluationPoint<T, DIM>::pt)
        .def_rw("normal", &wosx::bvc::EvaluationPoint<T, DIM>::normal)
        .def_rw("type", &wosx::bvc::EvaluationPoint<T, DIM>::type)
        .def_rw("dist_to_absorbing_boundary", &wosx::bvc::EvaluationPoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &wosx::bvc::EvaluationPoint<T, DIM>::distToReflectingBoundary);

    nb::bind_vector<BVCEvaluationPointList<T, DIM>>(solvers_m, ("BVCEvaluationPointList" + typeStr).c_str());

    nb::class_<wosx::bvc::BoundaryValueCachingSolver<T, DIM>>(solvers_m, ("BoundaryValueCaching" + typeStr).c_str())
        .def(nb::init<const wosx::GeometricQueries<DIM>&,
                      std::shared_ptr<wosx::BoundarySampler<T, DIM>>,
                      std::shared_ptr<wosx::BoundarySampler<T, DIM>>,
                      std::shared_ptr<wosx::DomainSampler<T, DIM>>>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a, "absorbing_boundary_sampler"_a, "reflecting_boundary_sampler"_a, "domain_sampler"_a)
        .def("generate_samples", &wosx::bvc::BoundaryValueCachingSolver<T, DIM>::generateSamples,
            "Generates boundary and domain samples.",
            "absorbing_boundary_cache_size"_a, "reflecting_boundary_cache_size"_a, "domain_cache_size"_a,
            "normal_offset_for_absorbing_boundary"_a, "normal_offset_for_reflecting_boundary"_a, "solve_double_sided"_a)
        .def("compute_sample_estimates", &wosx::bvc::BoundaryValueCachingSolver<T, DIM>::computeSampleEstimates,
            "Computes sample estimates on the boundary.",
            "pde"_a, "walk_settings"_a, "n_walks_for_solution_estimates"_a, "n_walks_for_gradient_estimates"_a,
            "robin_coeff_cutoff_for_normal_derivative"_a, "use_finite_differences"_a=false,
            "run_single_threaded"_a=false, "report_progress"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("splat", &wosx::bvc::BoundaryValueCachingSolver<T, DIM>::splat,
            "Splat solution and gradient estimates into the interior.",
            "pde"_a, "radius_clamp"_a, "kernel_regularization"_a, "robin_coeff_cutoff_for_normal_derivative"_a,
            "cutoff_dist_to_absorbing_boundary"_a, "cutoff_dist_to_reflecting_boundary"_a, "eval_pts"_a,
            "report_progress"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("estimate_solution_near_boundary",
            &wosx::bvc::BoundaryValueCachingSolver<T, DIM>::estimateSolutionNearBoundary,
            "Estimates the solution at the input evaluation points near the boundary.",
            "pde"_a, "walk_settings"_a, "cutoff_dist_to_absorbing_boundary"_a, "cutoff_dist_to_reflecting_boundary"_a,
            "n_walks_for_solution_estimates"_a, "eval_pts"_a, "run_single_threaded"_a=false,
            nb::call_guard<nb::gil_scoped_release>());
}

template <typename T, size_t DIM>
void bindReverseWalkOnStarsSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::rws::EvaluationPoint<T, DIM>>(solvers_m, ("RWSEvaluationPoint" + typeStr).c_str())
        .def(nb::init<const wosx::Vector<DIM>&, const wosx::Vector<DIM>&, wosx::SampleType, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("get_estimated_solution", &wosx::rws::EvaluationPoint<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.",
            "n_absorbing_boundary_samples"_a, "n_absorbing_boundary_normal_aligned_samples"_a,
            "n_reflecting_boundary_samples"_a, "n_reflecting_boundary_normal_aligned_samples"_a,
            "n_source_samples"_a)
        .def("reset", &wosx::rws::EvaluationPoint<T, DIM>::reset,
            "Resets state.")
        .def_rw("pt", &wosx::rws::EvaluationPoint<T, DIM>::pt)
        .def_rw("normal", &wosx::rws::EvaluationPoint<T, DIM>::normal)
        .def_rw("type", &wosx::rws::EvaluationPoint<T, DIM>::type)
        .def_rw("dist_to_absorbing_boundary", &wosx::rws::EvaluationPoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &wosx::rws::EvaluationPoint<T, DIM>::distToReflectingBoundary);

    nb::bind_vector<RWSEvaluationPointList<T, DIM>>(solvers_m, ("RWSEvaluationPointList" + typeStr).c_str());

    nb::class_<wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>>(
        solvers_m, ("ReverseWalkOnStars" + typeStr).c_str())
        .def(nb::init<const wosx::GeometricQueries<DIM>&,
                      std::shared_ptr<wosx::BoundarySampler<T, DIM>>,
                      std::shared_ptr<wosx::BoundarySampler<T, DIM>>,
                      std::shared_ptr<wosx::DomainSampler<T, DIM>>>(),
            "NOTE: the solver stores a reference to geometric_queries, which must be kept alive for the lifetime of the solver.",
            "geometric_queries"_a, "absorbing_boundary_sampler"_a, "reflecting_boundary_sampler"_a, "domain_sampler"_a)
        .def("generate_samples",
            &wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>::generateSamples,
            "Generates boundary and domain samples.",
            "absorbing_boundary_sample_count"_a, "reflecting_boundary_sample_count"_a, "domain_sample_count"_a,
            "normal_offset_for_absorbing_boundary"_a, "solve_double_sided"_a)
        .def("solve",
            &wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>::solve,
            "Solves the PDE using the reverse walk on stars algorithm.",
            "pde"_a, "walk_settings"_a, "normal_offset_for_absorbing_boundary"_a, "radius_clamp"_a,
            "kernel_regularization"_a, "eval_pts"_a, "updated_eval_pt_locations"_a=true,
            "run_single_threaded"_a=false, "report_progress"_a,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_absorbing_boundary_sample_count",
            &wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>::getAbsorbingBoundarySampleCount,
            "Returns the number of absorbing boundary sample points.",
            "return_boundary_normal_aligned"_a=false)
        .def("get_reflecting_boundary_sample_count",
            &wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>::getReflectingBoundarySampleCount,
            "Returns the number of reflecting boundary sample points.",
            "return_boundary_normal_aligned"_a=false)
        .def("get_domain_sample_count",
            &wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>>::getDomainSampleCount,
            "Returns the number of domain sample points.");
}

template <typename T, size_t DIM>
void bindKelvinTransform(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("Solvers", "Solvers module");

    nb::class_<wosx::KelvinTransform<T, DIM>> kelvinTransform(solvers_m, ("KelvinTransform" + typeStr).c_str());
    kelvinTransform.def(nb::init<const wosx::Vector<DIM>&>(),
                        "origin"_a=wosx::Vector<DIM>::Zero())
                    .def("set_origin", &wosx::KelvinTransform<T, DIM>::setOrigin,
                        "Sets the origin of the Kelvin transform.",
                        "origin"_a)
                    .def("get_origin", &wosx::KelvinTransform<T, DIM>::getOrigin,
                        "Returns the origin of the Kelvin transform.")
                    .def("transform_point", &wosx::KelvinTransform<T, DIM>::transformPoint,
                        "Applies the Kelvin transform to a point in the exterior domain.",
                        "x"_a)
                    .def("transform_pde", &wosx::KelvinTransform<T, DIM>::transformPde,
                        "Sets up the PDE for the inverted domain given the PDE for the exterior domain.",
                        "pde_exterior_domain"_a, "pde_inverted_domain"_a)
                    .def("transform_solution_estimate", &wosx::KelvinTransform<T, DIM>::transformSolutionEstimate,
                        "Returns the estimated solution in the exterior domain, given the solution estimate at a transformed point.",
                        "V"_a, "y"_a);

    if constexpr (std::is_same_v<T, float>) {
        kelvinTransform.def("transform_gradient_estimate",
            [](const wosx::KelvinTransform<T, DIM>& self,
               const T& V,
               const wosx::Vector<DIM>& dV,
               const wosx::Vector<DIM>& y) {
                FloatList dVList(DIM);
                for (size_t i = 0; i < DIM; i++) {
                    dVList[i] = dV[i];
                }

                FloatList gradientList(DIM);
                self.transformGradientEstimate(V, dVList, y, gradientList);

                wosx::Vector<DIM> gradient;
                for (size_t i = 0; i < DIM; i++) {
                    gradient[i] = gradientList[i];
                }

                return gradient;
            },
            "Returns the estimated gradient in the exterior domain, given solution and gradient estimates at a transformed point.",
            "V"_a, "dV"_a, "y"_a);

    } else {
        using GradientMatrix = Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime, Eigen::RowMajor>;
        kelvinTransform.def("transform_gradient_estimate",
            [](const wosx::KelvinTransform<T, DIM>& self,
               const T& V,
               const GradientMatrix& dV,
               const wosx::Vector<DIM>& y) {
                std::vector<T> dVList(DIM);
                constexpr int channels = GradientMatrix::ColsAtCompileTime;
                for (size_t i = 0; i < DIM; i++) {
                    for (int c = 0; c < channels; c++) {
                        dVList[i][c] = dV((Eigen::Index)i, c);
                    }
                }

                std::vector<T> gradientList(DIM);
                self.transformGradientEstimate(V, dVList, y, gradientList);

                GradientMatrix gradient(DIM, channels);
                for (size_t i = 0; i < DIM; i++) {
                    for (int c = 0; c < channels; c++) {
                        gradient((Eigen::Index)i, c) = gradientList[i][c];
                    }
                }

                return gradient;
            },
            "Returns the estimated gradient in the exterior domain, given solution and gradient estimates at a transformed point.",
            "V"_a, "dV"_a, "y"_a);
    }

    kelvinTransform.def("transform_points", &wosx::KelvinTransform<T, DIM>::transformPoints,
                        "Applies the Kelvin transform to a set of points in the exterior domain.",
                        "points"_a, "transformed_points"_a)
                    .def("compute_robin_coefficients", &wosx::KelvinTransform<T, DIM>::computeRobinCoefficients,
                        "Computes the modified Robin coefficients for the transformed reflecting boundary represented by line segments in 2D and triangles in 3D:\na boundary with Neumann conditions typically has non-zero Robin coefficients on the inverted domain in 3D, but it continues to have\nNeumann conditions in 2D.",
                        "transformed_points"_a, "indices"_a, "min_robin_coeff_values"_a, "max_robin_coeff_values"_a,
                        "transformed_min_robin_coeff_values"_a, "transformed_max_robin_coeff_values"_a);
}
