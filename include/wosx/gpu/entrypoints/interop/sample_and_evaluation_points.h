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

#pragma once

#include <wosx/point_estimation/common.h>
#include <wosx/gpu/entrypoints/interop/supported_datatypes.h>
#include <fcpw/fcpw_gpu.h>

namespace wosx {

using fcpw::float2;
using fcpw::float3;
using namespace fcpw;

template <size_t DIM>
struct GPUSamplePoint {
    // empty
};

template <>
struct GPUSamplePoint<2> {
    // members
    float2 pt = float2{0.0f, 0.0f};
    float2 normal = float2{0.0f, 0.0f};
    SampleType type = SampleType::InDomain;
    EstimationQuantity estimationQuantity = EstimationQuantity::Solution;
    float pdf = 1.0f;
    float distToAbsorbingBoundary = 0.0f;
    float distToReflectingBoundary = 0.0f;
    uint32_t estimateBoundaryNormalAligned = 0;
};

template <>
struct GPUSamplePoint<3> {
    // members
    float3 pt = float3{0.0f, 0.0f, 0.0f};
    float3 normal = float3{0.0f, 0.0f, 0.0f};
    SampleType type = SampleType::InDomain;
    EstimationQuantity estimationQuantity = EstimationQuantity::Solution;
    float pdf = 1.0f;
    float distToAbsorbingBoundary = 0.0f;
    float distToReflectingBoundary = 0.0f;
    uint32_t estimateBoundaryNormalAligned = 0;
};

template <typename T, size_t DIM>
struct GPUSampleStatistics {
    // members
    T solutionMean;
    T gradientMean[DIM];
    T totalFirstSourceContribution;
    uint32_t nSolutionEstimates;
    uint32_t nGradientEstimates;
    uint32_t totalWalkLength;
    uint32_t _padding;

    // methods to get average values
    T getEstimatedSolution() const;
    const T* getEstimatedGradient() const;
    T getEstimatedGradient(int channel) const;
    int getSolutionEstimateCount() const;
    int getGradientEstimateCount() const;
    float getMeanWalkLength() const;
};

class GPUComputeSampleBoundaryDistance: public GPUShaderEntryPoint {
public:
    // constructor
    GPUComputeSampleBoundaryDistance(GPUBuffer& samplePoints_,
                                     uint32_t nSamplePoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& samplePoints;
    uint32_t nSamplePoints;
};

class GPUSeedRngs: public GPUShaderEntryPoint {
public:
    // constructor
    GPUSeedRngs();

    // allocates GPU resources
    void allocate(GPUContext& context, uint32_t nRngs_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // returns the rngs buffer
    GPUBuffer& getBuffer();

private:
    // members
    GPUBuffer randomSeeds = {};
    GPUBuffer rngs = {};
    uint32_t nRngs;
};

class GPUResetSampleStatistics: public GPUShaderEntryPoint {
public:
    // constructor
    GPUResetSampleStatistics(GPUBuffer& sampleStatistics_,
                             uint32_t nSamplePoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& sampleStatistics;
    uint32_t nSamplePoints;
};

template <typename T, size_t DIM>
class GPUGetSampleStatistics: public GPUShaderEntryPoint {
public:
    // constructor
    GPUGetSampleStatistics(GPUBuffer& sampleStatistics_,
                           uint32_t nSamplePoints_,
                           uint32_t nOutputSamplePoints_);

    // allocates GPU resources
    void allocate(GPUContext& context);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // reads GPU resources
    void read(GPUContext& context,
              std::vector<GPUSampleStatistics<T, DIM>>& outputSampleStatisticsData) const;

private:
    // members
    GPUBuffer& sampleStatistics;
    GPUBuffer outputSampleStatistics = {};
    uint32_t nSamplePoints;
    uint32_t nOutputSamplePoints;
};

template <typename T>
struct GPUBVCBoundaryData {
    // members
    T solution;
    T normalDerivative;
};

template <size_t DIM>
struct GPUBVCEvaluationPoint {
    // empty
};

template <>
struct GPUBVCEvaluationPoint<2> {
    // members
    float2 pt = float2{0.0f, 0.0f};
    float2 normal = float2{0.0f, 0.0f};
    SampleType type = SampleType::InDomain;
};

template <>
struct GPUBVCEvaluationPoint<3> {
    // members
    float3 pt = float3{0.0f, 0.0f, 0.0f};
    float3 normal = float3{0.0f, 0.0f, 0.0f};
    SampleType type = SampleType::InDomain;
};

template <typename T, size_t DIM>
struct GPUBVCSampleStatistics {
    // members
    T solutionMean;
    T gradientMean[DIM];
};

template <typename T, size_t DIM>
struct GPUBVCEvaluationStatistics {
    // members
    GPUBVCSampleStatistics<T, DIM> absorbingBoundaryStatistics;
    GPUBVCSampleStatistics<T, DIM> absorbingBoundaryNormalAlignedStatistics;
    GPUBVCSampleStatistics<T, DIM> reflectingBoundaryStatistics;
    GPUBVCSampleStatistics<T, DIM> reflectingBoundaryNormalAlignedStatistics;
    GPUBVCSampleStatistics<T, DIM> sourceStatistics;
};

template <typename T, size_t DIM>
struct GPUBVCEvaluationOutputs {
    // members
    T solution;
    T gradient[DIM];

    // convenience methods to return solution and gradient
    T getEstimatedSolution() const;
    T getEstimatedGradient(int channel) const;
};

class GPUResetBVCEvaluationStatistics: public GPUShaderEntryPoint {
public:
    // constructor
    GPUResetBVCEvaluationStatistics(GPUBuffer& evaluationStatistics_,
                                    uint32_t nEvaluationPoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& evaluationStatistics;
    uint32_t nEvaluationPoints;
};

template <typename T, size_t DIM>
class GPUGetBVCEvaluationOutputs: public GPUShaderEntryPoint {
public:
    // constructor
    GPUGetBVCEvaluationOutputs(GPUBuffer& evaluationPoints_,
                               GPUBuffer& evaluationStatistics_,
                               uint32_t nEvaluationPoints_);

    // allocates GPU resources
    void allocate(GPUContext& context);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

    // reads GPU resources
    void read(GPUContext& context,
              std::vector<GPUBVCEvaluationOutputs<T, DIM>>& evaluationOutputsData) const;

private:
    // members
    GPUBuffer& evaluationPoints;
    GPUBuffer& evaluationStatistics;
    GPUBuffer evaluationOutputs = {};
    uint32_t nEvaluationPoints;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
T GPUSampleStatistics<T, DIM>::getEstimatedSolution() const
{
    return solutionMean;
}

template <typename T, size_t DIM>
const T* GPUSampleStatistics<T, DIM>::getEstimatedGradient() const
{
    return gradientMean;
}

template <typename T, size_t DIM>
T GPUSampleStatistics<T, DIM>::getEstimatedGradient(int channel) const
{
    return gradientMean[channel];
}

template <typename T, size_t DIM>
int GPUSampleStatistics<T, DIM>::getSolutionEstimateCount() const
{
    return nSolutionEstimates;
}

template <typename T, size_t DIM>
int GPUSampleStatistics<T, DIM>::getGradientEstimateCount() const
{
    return nGradientEstimates;
}

template <typename T, size_t DIM>
float GPUSampleStatistics<T, DIM>::getMeanWalkLength() const
{
    int N = std::max(1, (int)nSolutionEstimates);
    return (float)totalWalkLength/N;
}

inline GPUComputeSampleBoundaryDistance::GPUComputeSampleBoundaryDistance(GPUBuffer& samplePoints_,
                                                                          uint32_t nSamplePoints_):
samplePoints(samplePoints_),
nSamplePoints(nSamplePoints_)
{
    // do nothing
}

inline void GPUComputeSampleBoundaryDistance::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    if (printLogs) printReflectionInfo(cursor, 3, "computeSampleBoundaryDistance");
}

inline GPUSeedRngs::GPUSeedRngs():
nRngs(0)
{
    // do nothing
}

inline void GPUSeedRngs::allocate(GPUContext& context, uint32_t nRngs_)
{
    auto generateRandomSeeds = [](uint32_t nRandomSeeds) -> std::vector<uint64_t> {
        // read the clock once for the base state seed; each rng is decorrelated by a
        // unique per-thread stream (the thread index) supplied in the seeding shader,
        // so identical base states across threads still produce independent sequences
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t baseSeed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        std::vector<uint64_t> randomSeeds(nRandomSeeds);
        for (uint32_t i = 0; i < nRandomSeeds; i++) {
            randomSeeds[i] = baseSeed + i;
        }

        return randomSeeds;
    };

    nRngs = nRngs_;
    std::vector<uint64_t> randomSeedsData = generateRandomSeeds(nRngs);
    randomSeeds.allocate<uint64_t>(context, false, randomSeedsData);
    rngs.allocate<pcg32>(context.device, true, nullptr, nRngs);
}

inline void GPUSeedRngs::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("randomSeeds").setBinding(randomSeeds.buffer);
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("nRngs").setData(nRngs);
    if (printLogs) printReflectionInfo(cursor, 4, "seedRngs");
}

inline GPUBuffer& GPUSeedRngs::getBuffer()
{
    return rngs;
}

inline GPUResetSampleStatistics::GPUResetSampleStatistics(GPUBuffer& sampleStatistics_,
                                                          uint32_t nSamplePoints_):
sampleStatistics(sampleStatistics_),
nSamplePoints(nSamplePoints_)
{
    // do nothing
}

inline void GPUResetSampleStatistics::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("sampleStatistics").setBinding(sampleStatistics.buffer);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    if (printLogs) printReflectionInfo(cursor, 3, "resetSampleStatistics");
}

template <typename T, size_t DIM>
GPUGetSampleStatistics<T, DIM>::GPUGetSampleStatistics(GPUBuffer& sampleStatistics_,
                                                       uint32_t nSamplePoints_,
                                                       uint32_t nOutputSamplePoints_):
sampleStatistics(sampleStatistics_),
nSamplePoints(nSamplePoints_),
nOutputSamplePoints(nOutputSamplePoints_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUGetSampleStatistics<T, DIM>::allocate(GPUContext& context)
{
    std::vector<GPUSampleStatistics<T, DIM>> outputSampleStatisticsData(nOutputSamplePoints);
    outputSampleStatistics.allocate<GPUSampleStatistics<T, DIM>>(context, true, outputSampleStatisticsData);
}

template <typename T, size_t DIM>
void GPUGetSampleStatistics<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("sampleStatistics").setBinding(sampleStatistics.buffer);
    cursor.getPath("outputSampleStatistics").setBinding(outputSampleStatistics.buffer);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    cursor.getPath("nOutputSamplePoints").setData(nOutputSamplePoints);
    if (printLogs) printReflectionInfo(cursor, 5, "getSampleStatistics");
}

template <typename T, size_t DIM>
void GPUGetSampleStatistics<T, DIM>::read(GPUContext& context,
                                          std::vector<GPUSampleStatistics<T, DIM>>& outputSampleStatisticsData) const
{
    outputSampleStatistics.read<GPUSampleStatistics<T, DIM>>(context, outputSampleStatisticsData);
}

template <typename T, size_t DIM>
T GPUBVCEvaluationOutputs<T, DIM>::getEstimatedSolution() const
{
    return solution;
}

template <typename T, size_t DIM>
T GPUBVCEvaluationOutputs<T, DIM>::getEstimatedGradient(int channel) const
{
    return gradient[channel];
}

inline GPUResetBVCEvaluationStatistics::GPUResetBVCEvaluationStatistics(GPUBuffer& evaluationStatistics_,
                                                                        uint32_t nEvaluationPoints_):
evaluationStatistics(evaluationStatistics_),
nEvaluationPoints(nEvaluationPoints_)
{
    // do nothing
}

inline void GPUResetBVCEvaluationStatistics::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("evaluationStatistics").setBinding(evaluationStatistics.buffer);
    cursor.getPath("nEvaluationPoints").setData(nEvaluationPoints);
    if (printLogs) printReflectionInfo(cursor, 3, "resetEvaluationStatistics");
}

template <typename T, size_t DIM>
GPUGetBVCEvaluationOutputs<T, DIM>::GPUGetBVCEvaluationOutputs(GPUBuffer& evaluationPoints_,
                                                               GPUBuffer& evaluationStatistics_,
                                                               uint32_t nEvaluationPoints_):
evaluationPoints(evaluationPoints_),
evaluationStatistics(evaluationStatistics_),
nEvaluationPoints(nEvaluationPoints_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUGetBVCEvaluationOutputs<T, DIM>::allocate(GPUContext& context)
{
    std::vector<GPUBVCEvaluationOutputs<T, DIM>> evaluationOutputsData(nEvaluationPoints);
    evaluationOutputs.allocate<GPUBVCEvaluationOutputs<T, DIM>>(context, true, evaluationOutputsData);
}

template <typename T, size_t DIM>
void GPUGetBVCEvaluationOutputs<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("evaluationPoints").setBinding(evaluationPoints.buffer);
    cursor.getPath("evaluationStatistics").setBinding(evaluationStatistics.buffer);
    cursor.getPath("evaluationOutputs").setBinding(evaluationOutputs.buffer);
    cursor.getPath("nEvaluationPoints").setData(nEvaluationPoints);
    if (printLogs) printReflectionInfo(cursor, 5, "getEvaluationOutputs");
}

template <typename T, size_t DIM>
void GPUGetBVCEvaluationOutputs<T, DIM>::read(GPUContext& context,
                                              std::vector<GPUBVCEvaluationOutputs<T, DIM>>& evaluationOutputsData) const
{
    evaluationOutputs.read<GPUBVCEvaluationOutputs<T, DIM>>(context, evaluationOutputsData);
}

} // wosx
