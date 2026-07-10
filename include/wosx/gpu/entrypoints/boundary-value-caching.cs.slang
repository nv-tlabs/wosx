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

import wosx;
import wosx_pde_definition;

uniform ParameterBlock<AbsorbingBoundarySamplerType> gAbsorbingBoundarySampler;
uniform ParameterBlock<ReflectingBoundarySamplerType> gReflectingBoundarySampler;
uniform ParameterBlock<DomainSamplerType> gDomainSampler;

typedef WalkOnStars<DataType, PDEType, GeometricQueriesType, WOSX_PROBLEM_DIMENSION> PointEstimatorType;
uniform ParameterBlock<PointEstimatorType> gPointEstimator;

typedef BoundaryValueCaching<DataType, PDEType, WOSX_PROBLEM_DIMENSION> BoundaryValueCachingType;
uniform ParameterBlock<BoundaryValueCachingType> gBoundaryValueCaching;

uniform ParameterBlock<WalkSettings> gWalkSettings;
uniform ParameterBlock<PDEType> gPDE;

#include "common-shaders/point-estimation.cs.slang"
#include "common-shaders/rng-seeding.cs.slang"
#include "common-shaders/sample-points-generation.cs.slang"
#include "common-shaders/sample-statistics-utils.cs.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void getBoundaryData(uint3 threadId: SV_DispatchThreadID,
                     uniform StructuredBuffer<SamplePoint<WOSX_PROBLEM_DIMENSION>> samplePoints,
                     uniform StructuredBuffer<SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION>> sampleStatistics,
                     uniform RWStructuredBuffer<BVCBoundaryData<DataType>> boundaryData,
                     uniform uint nInputSamplePoints,
                     uniform uint nSamplePoints)
{
    uint index = threadId.x;
    if (index >= nInputSamplePoints)
    {
        return;
    }

    uint nSampleCopies = nSamplePoints / nInputSamplePoints;
    SampleStatistics<DataType, WOSX_PROBLEM_DIMENSION> localStatistics = sampleStatistics[nSampleCopies * index];
    for (uint i = 1; i < nSampleCopies; i++)
    {
        localStatistics.merge(sampleStatistics[nSampleCopies * index + i]);
    }

    SamplePoint<WOSX_PROBLEM_DIMENSION> samplePt = samplePoints[index];
    vector<float, WOSX_PROBLEM_DIMENSION> directionForDerivative = samplePt.normal;
    if (samplePt.estimateBoundaryNormalAligned != 0)
    {
        directionForDerivative = directionForDerivative * -1.0;
    }

    DataType solutionEstimate = localStatistics.getEstimatedSolution();
    DataType derivativeEstimate = localStatistics.getEstimatedDerivative(directionForDerivative);
    boundaryData[index] = gBoundaryValueCaching.getBoundaryData(samplePt, solutionEstimate, derivativeEstimate);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatBoundaryData(uint3 threadId: SV_DispatchThreadID,
                       uniform StructuredBuffer<SamplePoint<WOSX_PROBLEM_DIMENSION>> samplePoints,
                       uniform StructuredBuffer<BVCBoundaryData<DataType>> boundaryData,
                       uniform StructuredBuffer<BVCEvaluationPoint<WOSX_PROBLEM_DIMENSION>> evaluationPoints,
                       uniform RWStructuredBuffer<BVCEvaluationStatistics<DataType, WOSX_PROBLEM_DIMENSION>> evaluationStatistics,
                       uniform float radiusClamp,
                       uniform float kernelRegularization,
                       uniform uint nSamplesSplatted,
                       uniform uint sampleIndex,
                       uniform uint nEvaluationPoints)
{
    uint index = threadId.x;
    if (index >= nEvaluationPoints)
    {
        return;
    }

    gBoundaryValueCaching.splatBoundaryData(samplePoints[sampleIndex],
                                            boundaryData[sampleIndex],
                                            evaluationPoints[index],
                                            radiusClamp, kernelRegularization,
                                            nSamplesSplatted + sampleIndex + 1,
                                            evaluationStatistics[index]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatSourceData(uint3 threadId: SV_DispatchThreadID,
                     uniform StructuredBuffer<SamplePoint<WOSX_PROBLEM_DIMENSION>> samplePoints,
                     uniform StructuredBuffer<BVCEvaluationPoint<WOSX_PROBLEM_DIMENSION>> evaluationPoints,
                     uniform RWStructuredBuffer<BVCEvaluationStatistics<DataType, WOSX_PROBLEM_DIMENSION>> evaluationStatistics,
                     uniform float radiusClamp,
                     uniform float kernelRegularization,
                     uniform uint nSamplesSplatted,
                     uniform uint sampleIndex,
                     uniform uint nEvaluationPoints)
{
    uint index = threadId.x;
    if (index >= nEvaluationPoints)
    {
        return;
    }

    gBoundaryValueCaching.splatSourceData(samplePoints[sampleIndex],
                                          evaluationPoints[index],
                                          radiusClamp, kernelRegularization,
                                          nSamplesSplatted + sampleIndex + 1,
                                          evaluationStatistics[index]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void resetEvaluationStatistics(uint3 threadId: SV_DispatchThreadID,
                               uniform RWStructuredBuffer<BVCEvaluationStatistics<DataType, WOSX_PROBLEM_DIMENSION>> evaluationStatistics,
                               uniform uint nEvaluationPoints)
{
    uint index = threadId.x;
    if (index >= nEvaluationPoints)
    {
        return;
    }

    evaluationStatistics[index].reset();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void getEvaluationOutputs(uint3 threadId: SV_DispatchThreadID,
                          uniform StructuredBuffer<BVCEvaluationPoint<WOSX_PROBLEM_DIMENSION>> evaluationPoints,
                          uniform StructuredBuffer<BVCEvaluationStatistics<DataType, WOSX_PROBLEM_DIMENSION>> evaluationStatistics,
                          uniform RWStructuredBuffer<BVCEvaluationOutputs<DataType, WOSX_PROBLEM_DIMENSION>> evaluationOutputs,
                          uniform uint nEvaluationPoints)
{
    uint index = threadId.x;
    if (index >= nEvaluationPoints)
    {
        return;
    }

    if (evaluationPoints[index].type == SampleType::OnAbsorbingBoundary)
    {
        evaluationOutputs[index].solution = gWalkSettings.ignoreAbsorbingBoundaryContribution == 0 ?
                                            gPDE.dirichlet(evaluationPoints[index].pt) : DataType(0.0);
    }
    else
    {
        evaluationOutputs[index] = evaluationStatistics[index].getOutputs();
    }
}
