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

// This file is the entry point for a 2D demo application demonstrating how to use WoSX.
// It reads a 'model problem' description from a JSON file, runs the WalkOnSpheres, WalkOnStars,
// BoundaryValueCaching or ReverseWalkOnStars solvers, and writes the result to a PFM or PNG file.

#include "model_problem.h"
#include "grid.h"

template <size_t DIM>
void computeDistanceInfo(const std::vector<wosx::Vector<DIM>>& solveLocations,
                         const wosx::GeometricQueries<DIM>& queries,
                         bool solveDoubleSided, bool solveExterior,
                         std::vector<DistanceInfo>& distanceInfo)
{
    distanceInfo.resize(solveLocations.size());
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        wosx::Vector<DIM> pt = solveLocations[i];
        bool insideDomain = queries.insideDomain(pt);
        if (queries.domainIsWatertight && solveExterior) insideDomain = !insideDomain;
        distanceInfo[i].inValidSolveRegion = insideDomain || solveDoubleSided;
        distanceInfo[i].distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
        distanceInfo[i].distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);
    }
}

template <typename T, size_t DIM>
void createSamplePoints(const std::vector<wosx::Vector<DIM>>& solveLocations,
                        const std::vector<DistanceInfo>& distanceInfo,
                        std::vector<wosx::SamplePoint<T, DIM>>& samplePts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            wosx::Vector<DIM> pt = solveLocations[i];
            wosx::Vector<DIM> normal = wosx::Vector<DIM>::Zero();
            wosx::SampleType sampleType = wosx::SampleType::InDomain;
            wosx::EstimationQuantity estimationQuantity = wosx::EstimationQuantity::Solution;
            float pdf = 1.0f;
            float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
            float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

            samplePts.emplace_back(wosx::SamplePoint<T, DIM>(pt, normal, sampleType,
                                                             estimationQuantity, pdf,
                                                             distToAbsorbingBoundary,
                                                             distToReflectingBoundary));
        }
    }
}

template <typename T, size_t DIM>
void runWalkOnSpheres(const json& solverConfig,
                      const wosx::GeometricQueries<DIM>& queries,
                      const wosx::PDE<T, DIM>& pde,
                      bool solveDoubleSided,
                      std::vector<wosx::SamplePoint<T, DIM>>& samplePts,
                      std::vector<wosx::SampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // initialize solver and estimate solution
    wosx::ProgressBar pb(samplePts.size());
    std::function<void(int, int)> reportProgress = wosx::getReportProgressCallback(pb);

    wosx::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                    0.0f, 0.0f, russianRouletteThreshold,
                                    splittingThreshold, maxWalkLength,
                                    stepsBeforeApplyingTikhonov,
                                    0, solveDoubleSided,
                                    !disableGradientControlVariates,
                                    !disableGradientAntitheticVariates,
                                    useCosineSamplingForDirectionalDerivatives,
                                    ignoreAbsorbingBoundaryContribution, true,
                                    ignoreSourceContribution, printLogs);
    std::vector<int> nWalksVector(samplePts.size(), nWalks);
    wosx::WalkOnSpheres<T, DIM> walkOnSpheres(queries);
    walkOnSpheres.solve(pde, walkSettings, nWalksVector, samplePts, sampleStatistics,
                        runSingleThreaded, reportProgress);
    pb.finish();
}

template <typename T, size_t DIM>
void runWalkOnStars(const json& solverConfig,
                    const wosx::GeometricQueries<DIM>& queries,
                    const wosx::PDE<T, DIM>& pde,
                    bool solveDoubleSided,
                    std::vector<wosx::SamplePoint<T, DIM>>& samplePts,
                    std::vector<wosx::SampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // initialize solver and estimate solution
    wosx::ProgressBar pb(samplePts.size());
    std::function<void(int, int)> reportProgress = wosx::getReportProgressCallback(pb);

    wosx::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                    epsilonShellForReflectingBoundary,
                                    silhouettePrecision,
                                    russianRouletteThreshold,
                                    splittingThreshold, maxWalkLength,
                                    stepsBeforeApplyingTikhonov,
                                    stepsBeforeUsingMaximalSpheres,
                                    solveDoubleSided,
                                    !disableGradientControlVariates,
                                    !disableGradientAntitheticVariates,
                                    useCosineSamplingForDirectionalDerivatives,
                                    ignoreAbsorbingBoundaryContribution,
                                    ignoreReflectingBoundaryContribution,
                                    ignoreSourceContribution, printLogs);
    std::vector<int> nWalksVector(samplePts.size(), nWalks);
    wosx::WalkOnStars<T, DIM> walkOnStars(queries);
    walkOnStars.solve(pde, walkSettings, nWalksVector, samplePts, sampleStatistics,
                      runSingleThreaded, reportProgress);
    pb.finish();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<DistanceInfo>& distanceInfo,
                 const std::vector<wosx::SampleStatistics<T, DIM>>& sampleStatistics,
                 std::vector<T>& solution)
{
    solution.resize(distanceInfo.size(), T(0.0f));
    int counter = 0;
    for (int i = 0; i < (int)distanceInfo.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            solution[i] = sampleStatistics[counter++].getEstimatedSolution();
        }
    }
}

template <typename T, size_t DIM>
void createBvcEvaluationPoints(const std::vector<wosx::Vector<DIM>>& solveLocations,
                               const std::vector<DistanceInfo>& distanceInfo,
                               std::vector<wosx::bvc::EvaluationPoint<T, DIM>>& evalPts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        wosx::Vector<DIM> pt = solveLocations[i];
        wosx::Vector<DIM> normal = wosx::Vector<DIM>::Zero();
        wosx::SampleType sampleType = wosx::SampleType::InDomain;
        float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
        float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

        evalPts.emplace_back(wosx::bvc::EvaluationPoint<T, DIM>(pt, normal, sampleType,
                                                                distToAbsorbingBoundary,
                                                                distToReflectingBoundary));
    }
}

template <typename T>
std::shared_ptr<wosx::BoundarySampler<T, 2>> createBoundarySampler(const std::vector<wosx::Vector2>& boundaryPositions,
                                                                   const std::vector<wosx::Vector2i>& boundaryIndices,
                                                                   const wosx::GeometricQueries<2>& queries)
{
    return wosx::createUniformLineSegmentBoundarySampler<T>(boundaryPositions, boundaryIndices,
                                                            queries.insideBoundingDomain);
}

template <typename T>
std::shared_ptr<wosx::BoundarySampler<T, 3>> createBoundarySampler(const std::vector<wosx::Vector3>& boundaryPositions,
                                                                   const std::vector<wosx::Vector3i>& boundaryIndices,
                                                                   const wosx::GeometricQueries<3>& queries)
{
    return wosx::createUniformTriangleBoundarySampler<T>(boundaryPositions, boundaryIndices,
                                                         queries.insideBoundingDomain);
}

template <typename T, size_t DIM>
void runBoundaryValueCaching(const json& solverConfig,
                             const std::vector<wosx::Vector<DIM>>& absorbingBoundaryPositions,
                             const std::vector<wosx::Vectori<DIM>>& absorbingBoundaryIndices,
                             const std::vector<wosx::Vector<DIM>>& reflectingBoundaryPositions,
                             const std::vector<wosx::Vectori<DIM>>& reflectingBoundaryIndices,
                             const wosx::GeometricQueries<DIM>& queries,
                             const wosx::PDE<T, DIM>& pde,
                             bool solveDoubleSided,
                             std::vector<wosx::bvc::EvaluationPoint<T, DIM>>& evalPts)
{
    // load config settings for wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // load config settings for boundary value caching
    const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
    const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
    const int absorbingBoundaryCacheSize = getOptional<int>(solverConfig, "absorbingBoundaryCacheSize", 1024);
    const int reflectingBoundaryCacheSize = getOptional<int>(solverConfig, "reflectingBoundaryCacheSize", 1024);
    int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);

    const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);

    const float robinCoeffCutoffForNormalDerivative = getOptional<float>(solverConfig, "robinCoeffCutoffForNormalDerivative",
                                                                         std::numeric_limits<float>::max());
    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary",
                                                                      5.0f*epsilonShellForAbsorbingBoundary);
    const float normalOffsetForReflectingBoundary = getOptional<float>(solverConfig, "normalOffsetForReflectingBoundary", 0.0f);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<wosx::BoundarySampler<T, DIM>> absorbingBoundarySampler =
        createBoundarySampler<T>(absorbingBoundaryPositions, absorbingBoundaryIndices, queries);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);

    std::shared_ptr<wosx::BoundarySampler<T, DIM>> reflectingBoundarySampler =
        createBoundarySampler<T>(reflectingBoundaryPositions, reflectingBoundaryIndices, queries);
    reflectingBoundarySampler->initialize(normalOffsetForReflectingBoundary, solveDoubleSided);

    // initialize domain sampler
    std::function<bool(const wosx::Vector<DIM>&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<wosx::DomainSampler<T, DIM>> domainSampler =
        wosx::createUniformDomainSampler<T, DIM>(insideSolveRegionDomainSampler,
                                                 queries.domainMin, queries.domainMax,
                                                 solveRegionVolume);
    if (ignoreSourceContribution) domainCacheSize = 0;

    // solve using boundary value caching
    int totalWork = 2*(absorbingBoundaryCacheSize + reflectingBoundaryCacheSize) + domainCacheSize;
    wosx::ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = wosx::getReportProgressCallback(pb);

    wosx::bvc::BoundaryValueCachingSolver<T, DIM> boundaryValueCaching(
        queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler);

    // generate boundary and domain samples
    boundaryValueCaching.generateSamples(absorbingBoundaryCacheSize, reflectingBoundaryCacheSize,
                                         domainCacheSize, normalOffsetForAbsorbingBoundary,
                                         normalOffsetForReflectingBoundary, solveDoubleSided);

    // compute sample estimates
    wosx::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                    epsilonShellForReflectingBoundary,
                                    silhouettePrecision,
                                    russianRouletteThreshold,
                                    splittingThreshold, maxWalkLength,
                                    stepsBeforeApplyingTikhonov,
                                    stepsBeforeUsingMaximalSpheres,
                                    solveDoubleSided,
                                    !disableGradientControlVariates,
                                    !disableGradientAntitheticVariates,
                                    useCosineSamplingForDirectionalDerivatives,
                                    ignoreAbsorbingBoundaryContribution,
                                    ignoreReflectingBoundaryContribution,
                                    ignoreSourceContribution, printLogs);
    boundaryValueCaching.computeSampleEstimates(pde, walkSettings,
                                                nWalksForCachedSolutionEstimates,
                                                nWalksForCachedGradientEstimates,
                                                robinCoeffCutoffForNormalDerivative,
                                                useFiniteDifferencesForBoundaryDerivatives,
                                                runSingleThreaded, reportProgress);

    // splat boundary sample estimates and domain data to evaluation points
    boundaryValueCaching.splat(pde, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);

    // estimate solution near boundary
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings,
                                                      normalOffsetForAbsorbingBoundary,
                                                      normalOffsetForReflectingBoundary,
                                                      nWalksForCachedSolutionEstimates,
                                                      evalPts, runSingleThreaded);
    pb.finish();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<wosx::bvc::EvaluationPoint<T, DIM>>& evalPts,
                 std::vector<T>& solution)
{
    solution.resize(evalPts.size(), T(0.0f));
    for (int i = 0; i < (int)evalPts.size(); i++) {
        solution[i] = evalPts[i].getEstimatedSolution();
    }
}

template <typename T, size_t DIM>
void createRwsEvaluationPoints(const std::vector<wosx::Vector<DIM>>& solveLocations,
                               const std::vector<DistanceInfo>& distanceInfo,
                               std::vector<wosx::rws::EvaluationPoint<T, DIM>>& evalPts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        wosx::Vector<DIM> pt = solveLocations[i];
        wosx::Vector<DIM> normal = wosx::Vector<DIM>::Zero();
        wosx::SampleType sampleType = wosx::SampleType::InDomain;
        float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
        float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

        evalPts.emplace_back(wosx::rws::EvaluationPoint<T, DIM>(pt, normal, sampleType,
                                                                distToAbsorbingBoundary,
                                                                distToReflectingBoundary));
    }
}

template <typename T, size_t DIM>
void runReverseWalkOnStars(const json& solverConfig,
                           const std::vector<wosx::Vector<DIM>>& absorbingBoundaryPositions,
                           const std::vector<wosx::Vectori<DIM>>& absorbingBoundaryIndices,
                           const std::vector<wosx::Vector<DIM>>& reflectingBoundaryPositions,
                           const std::vector<wosx::Vectori<DIM>>& reflectingBoundaryIndices,
                           const wosx::GeometricQueries<DIM>& queries,
                           const wosx::PDE<T, DIM>& pde,
                           bool solveDoubleSided,
                           std::vector<wosx::rws::EvaluationPoint<T, DIM>>& evalPts,
                           std::vector<int>& sampleCounts)
{
    // load config settings for reverse wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // load config settings for reverse walk splatting
    int absorbingBoundarySampleCount = getOptional<int>(solverConfig, "absorbingBoundarySampleCount", 1024);
    int reflectingBoundarySampleCount = getOptional<int>(solverConfig, "reflectingBoundarySampleCount", 1024);
    int domainSampleCount = getOptional<int>(solverConfig, "domainSampleCount", 1024);

    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary",
                                                                      5.0f*epsilonShellForAbsorbingBoundary);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<wosx::BoundarySampler<T, DIM>> absorbingBoundarySampler =
        createBoundarySampler<T>(absorbingBoundaryPositions, absorbingBoundaryIndices, queries);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
    if (ignoreAbsorbingBoundaryContribution) absorbingBoundarySampleCount = 0;

    std::shared_ptr<wosx::BoundarySampler<T, DIM>> reflectingBoundarySampler =
        createBoundarySampler<T>(reflectingBoundaryPositions, reflectingBoundaryIndices, queries);
    reflectingBoundarySampler->initialize(0.0f, solveDoubleSided);
    if (ignoreReflectingBoundaryContribution) reflectingBoundarySampleCount = 0;

    // initialize domain sampler
    std::function<bool(const wosx::Vector<DIM>&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<wosx::DomainSampler<T, DIM>> domainSampler =
        wosx::createUniformDomainSampler<T, DIM>(insideSolveRegionDomainSampler,
                                                 queries.domainMin, queries.domainMax,
                                                 solveRegionVolume);
    if (ignoreSourceContribution) domainSampleCount = 0;

    // solve using reverse walk on stars
    int totalWork = absorbingBoundarySampleCount + reflectingBoundarySampleCount + domainSampleCount;
    wosx::ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = wosx::getReportProgressCallback(pb);

    wosx::rws::ReverseWalkOnStarsSolver<T, DIM, wosx::NearestNeighborFinder<DIM>> reverseWalkOnStars(
        queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler);

    // generate boundary and domain samples
    reverseWalkOnStars.generateSamples(absorbingBoundarySampleCount, reflectingBoundarySampleCount,
                                       domainSampleCount, normalOffsetForAbsorbingBoundary, solveDoubleSided);

    // splat contributions to evaluation points
    wosx::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                    epsilonShellForReflectingBoundary,
                                    silhouettePrecision,
                                    russianRouletteThreshold,
                                    splittingThreshold, maxWalkLength,
                                    stepsBeforeApplyingTikhonov,
                                    stepsBeforeUsingMaximalSpheres,
                                    solveDoubleSided, false, false, false,
                                    ignoreAbsorbingBoundaryContribution,
                                    ignoreReflectingBoundaryContribution,
                                    ignoreSourceContribution, printLogs);
    reverseWalkOnStars.solve(pde, walkSettings, normalOffsetForAbsorbingBoundary,
                             radiusClampForKernels, regularizationForKernels, evalPts,
                             true, runSingleThreaded, reportProgress);
    pb.finish();

    // save samples counts
    sampleCounts.resize(5);
    sampleCounts[0] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(false);
    sampleCounts[1] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(true);
    sampleCounts[2] = reverseWalkOnStars.getReflectingBoundarySampleCount(false);
    sampleCounts[3] = reverseWalkOnStars.getReflectingBoundarySampleCount(true);
    sampleCounts[4] = reverseWalkOnStars.getDomainSampleCount();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<wosx::rws::EvaluationPoint<T, DIM>>& evalPts,
                 const std::vector<int>& sampleCounts,
                 std::vector<T>& solution)
{
    solution.resize(evalPts.size(), T(0.0f));
    int absorbingBoundarySampleCount = sampleCounts[0];
    int absorbingBoundaryNormalAlignedSampleCount = sampleCounts[1];
    int reflectingBoundarySampleCount = sampleCounts[2];
    int reflectingBoundaryNormalAlignedSampleCount = sampleCounts[3];
    int domainSampleCount = sampleCounts[4];

    for (int i = 0; i < (int)evalPts.size(); i++) {
        solution[i] = evalPts[i].getEstimatedSolution(absorbingBoundarySampleCount,
                                                      absorbingBoundaryNormalAlignedSampleCount,
                                                      reflectingBoundarySampleCount,
                                                      reflectingBoundaryNormalAlignedSampleCount,
                                                      domainSampleCount);
    }
}

template <typename T, size_t DIM>
void runSolver(const std::string& solverType, const json& solverConfig,
               const std::vector<wosx::Vector<DIM>>& absorbingBoundaryPositions,
               const std::vector<wosx::Vectori<DIM>>& absorbingBoundaryIndices,
               const std::vector<wosx::Vector<DIM>>& reflectingBoundaryPositions,
               const std::vector<wosx::Vectori<DIM>>& reflectingBoundaryIndices,
               const wosx::GeometricQueries<DIM>& queries,
               const wosx::PDE<T, DIM>& pde, bool solveDoubleSided,
               const std::vector<wosx::Vector<DIM>>& solveLocations,
               const std::vector<DistanceInfo>& distanceInfo,
               std::vector<T>& solution)
{
    if (solverType == "wos") {
        // create sample points to estimate solution at
        std::vector<wosx::SamplePoint<T, DIM>> samplePts;
        createSamplePoints<T, DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on spheres
        std::vector<wosx::SampleStatistics<T, DIM>> sampleStatistics;
        runWalkOnSpheres<T, DIM>(solverConfig, queries, pde, solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);

    } else if (solverType == "wost") {
        // create sample points to estimate solution at
        std::vector<wosx::SamplePoint<T, DIM>> samplePts;
        createSamplePoints<T, DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on stars
        std::vector<wosx::SampleStatistics<T, DIM>> sampleStatistics;
        runWalkOnStars<T, DIM>(solverConfig, queries, pde, solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);

    } else if (solverType == "bvc") {
        // create evaluation points to estimate solution at
        std::vector<wosx::bvc::EvaluationPoint<T, DIM>> evalPts;
        createBvcEvaluationPoints<T, DIM>(solveLocations, distanceInfo, evalPts);

        // run boundary value caching
        runBoundaryValueCaching<T, DIM>(solverConfig, absorbingBoundaryPositions, absorbingBoundaryIndices,
                                        reflectingBoundaryPositions, reflectingBoundaryIndices,
                                        queries, pde, solveDoubleSided, evalPts);

        // extract solution from evaluation points
        getSolution<T, DIM>(evalPts, solution);

    } else if (solverType == "rws") {
        // ccreate evaluation points to estimate solution at
        std::vector<wosx::rws::EvaluationPoint<T, DIM>> evalPts;
        createRwsEvaluationPoints<T, DIM>(solveLocations, distanceInfo, evalPts);

        // run reverse walk on stars
        std::vector<int> sampleCounts;
        runReverseWalkOnStars<T, DIM>(solverConfig, absorbingBoundaryPositions, absorbingBoundaryIndices,
                                      reflectingBoundaryPositions, reflectingBoundaryIndices,
                                      queries, pde, solveDoubleSided, evalPts, sampleCounts);

        // extract solution from evaluation points
        getSolution<T, DIM>(evalPts, sampleCounts, solution);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
int runDemo(const json& config)
{
    // load config settings
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();

    // initialize the model problem
    ModelProblem<T> modelProblem(modelProblemConfig, wosxDirectoryPath.string());
    const std::vector<Vector2i>& absorbingBoundaryIndices = modelProblem.getAbsorbingBoundaryIndices();
    const std::vector<Vector2i>& reflectingBoundaryIndices = modelProblem.getReflectingBoundaryIndices();
    const std::pair<Vector2, Vector2>& boundingBox = modelProblem.getBoundingBox();
    const wosx::GeometricQueries<2>& queries = modelProblem.getGeometricQueries();
    bool solveDoubleSided = modelProblem.solveDoubleSided();
    bool solveExterior = modelProblem.solveExterior();

    // create solve locations on a grid for this demo
    std::vector<Vector2> solveLocations;
    std::vector<DistanceInfo> distanceInfo;
    createGridPoints(outputConfig, boundingBox, solveLocations);
    computeDistanceInfo<2>(solveLocations, queries, solveDoubleSided, solveExterior, distanceInfo);

    // solve the model problem
    std::vector<T> solution;
    if (solveExterior) {
        const wosx::KelvinTransform<T, 2>& kelvinTransform = modelProblem.getKelvinTransform();
        const std::vector<Vector2>& invertedAbsorbingBoundaryPositions = modelProblem.getInvertedAbsorbingBoundaryPositions();
        const std::vector<Vector2>& invertedReflectingBoundaryPositions = modelProblem.getInvertedReflectingBoundaryPositions();
        const wosx::PDE<T, 2>& pdeInvertedDomain = modelProblem.getPDEInvertedDomain();
        const wosx::GeometricQueries<2>& queriesInvertedDomain = modelProblem.getGeometricQueriesInvertedDomain();

        // invert the solve locations and update the distance info
        int nSolveLocations = (int)solveLocations.size();
        std::vector<Vector2> invertedSolveLocations(nSolveLocations, Vector2::Zero());
        for (int i = 0; i < nSolveLocations; i++) {
            invertedSolveLocations[i] = kelvinTransform.transformPoint(solveLocations[i]);
        }
        std::vector<DistanceInfo> distanceInfoInvertedDomain;
        computeDistanceInfo<2>(invertedSolveLocations, queriesInvertedDomain,
                               solveDoubleSided, false, distanceInfoInvertedDomain);

        // run the solver on the inverted domain
        runSolver<T, 2>(solverType, solverConfig,
                        invertedAbsorbingBoundaryPositions, absorbingBoundaryIndices,
                        invertedReflectingBoundaryPositions, reflectingBoundaryIndices,
                        queriesInvertedDomain, pdeInvertedDomain, solveDoubleSided,
                        invertedSolveLocations, distanceInfoInvertedDomain, solution);

        // map the solution values back to the exterior domain
        for (int i = 0; i < nSolveLocations; i++) {
            solution[i] = kelvinTransform.transformSolutionEstimate(solution[i], invertedSolveLocations[i]);
        }

    } else {
        const std::vector<Vector2>& absorbingBoundaryPositions = modelProblem.getAbsorbingBoundaryPositions();
        const std::vector<Vector2>& reflectingBoundaryPositions = modelProblem.getReflectingBoundaryPositions();
        const wosx::PDE<T, 2>& pde = modelProblem.getPDE();

        // run the solver on the input domain
        runSolver<T, 2>(solverType, solverConfig,
                        absorbingBoundaryPositions, absorbingBoundaryIndices,
                        reflectingBoundaryPositions, reflectingBoundaryIndices,
                        queries, pde, solveDoubleSided, solveLocations,
                        distanceInfo, solution);
    }

    // save the solution to disk
    saveGridValues<T>(outputConfig, wosxDirectoryPath.string(), distanceInfo, solution);
    return 0;
}

int main(int argc, const char *argv[])
{
    if (argc != 2) {
        std::cerr << "must provide config filename" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    json config = json::parse(configFile);
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const int channels = getOptional<int>(modelProblemConfig, "channels", 1);
    if (channels == 1) {
        return runDemo<float>(config);

    } else if (channels == 4) {
        return runDemo<Array4>(config);
    }

    std::cerr << "C++ basic_2d demo only supports channels = 1 or channels = 4" << std::endl;
    return EXIT_FAILURE;
}
