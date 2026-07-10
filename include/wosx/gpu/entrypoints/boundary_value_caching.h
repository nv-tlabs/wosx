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

#include <wosx/gpu/entrypoints/interop/task_handle.h>

namespace wosx {

template <typename T, size_t DIM>
class GPUBoundaryValueCachingSolver {
public:
    // constructor
    GPUBoundaryValueCachingSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                  std::shared_ptr<GPUWalkSettings> walkSettings_,
                                  uint32_t nResidentThreads_=131072,
                                  bool usePersistentThreads_=true,
                                  bool printLogs_=false);

    // populates evaluation points
    void populateEvaluationPoints(const std::vector<GPUBVCEvaluationPoint<DIM>>& inputEvaluationPoints);

    // generates boundary and domain samples
    void generateSamples(uint32_t absorbingBoundaryCacheSize,
                         uint32_t reflectingBoundaryCacheSize,
                         uint32_t domainCacheSize);

    // computes sample estimates on the boundary
    void computeSampleEstimates(uint32_t nWalksForSolutionEstimates,
                                uint32_t nWalksForGradientEstimates);

    // splats solution and gradient estimates into the interior
    void splat(float radiusClamp, float kernelRegularization);

    // resets evaluation statistics
    void resetEvaluationStatistics();

    // returns evaluation outputs (resized to be the same size as input evaluation points)
    void getEvaluationOutputs(std::vector<GPUBVCEvaluationOutputs<T, DIM>>& evaluationOutputs);

    // update normal offset for the absorbing boundary
    void updateNormalOffsetForAbsorbingBoundary(float offset);

    // returns the boundary and domain samples
    void getAbsorbingBoundarySamples(std::vector<GPUSamplePoint<DIM>>& samplePoints,
                                     bool returnBoundaryNormalAligned=false);
    void getReflectingBoundarySamples(std::vector<GPUSamplePoint<DIM>>& samplePoints,
                                      bool returnBoundaryNormalAligned=false);
    void getDomainSamples(std::vector<GPUSamplePoint<DIM>>& samplePoints);

    // returns true if using persistent threads
    bool usingPersistentThreads() const;

    // enables/disables logging
    void enableLogging(bool enable);

private:
    // seeds rngs
    void seedRngs(uint32_t nRngs, GPUSeedRngs& entryPoint);

    // generates absorbing boundary samples
    void generateAbsorbingBoundarySamples(uint32_t nAbsorbingBoundarySamples);

    // generates reflecting boundary samples
    void generateReflectingBoundarySamples(uint32_t nReflectingBoundarySamples);

    // generates domain samples
    void generateDomainSamples(uint32_t nDomainSamples);

    // resets sample statistics for each sample point
    void resetSampleStatistics(GPUBuffer& sampleStatistics,
                               uint32_t nSamplePoints);

    // runs point estimator for each sample point to compute boundary estimates
    void runPointEstimator(GPUBuffer& samplePoints,
                           GPUBuffer& rngs,
                           GPUBuffer& sampleStatistics,
                           uint32_t nInputSamplePoints,
                           uint32_t nSamplePoints,
                           uint32_t nWalks,
                           uint32_t nDispatchCalls);

    // runs persistent point estimator for each sample point to compute boundary estimates
    void runPersistentPointEstimator(GPUBuffer& samplePoints,
                                     GPUBuffer& rngs,
                                     GPUBuffer& sampleStatistics,
                                     GPUBuffer& locks,
                                     uint32_t nInputSamplePoints,
                                     uint32_t nWalks);

    // gets boundary data from sample statistics
    void getBoundaryData(GPUBuffer& samplePoints,
                         GPUBuffer& sampleStatistics,
                         GPUBuffer& boundaryData,
                         uint32_t nInputSamplePoints,
                         uint32_t nSamplePoints);

    // computes sample estimates on the boundary
    void computeSampleEstimates(GPUBuffer& samplePoints,
                                GPUBuffer& sampleStatistics,
                                GPUBuffer& boundaryData,
                                GPUBuffer& locks,
                                GPUSeedRngs& entryPoint,
                                uint32_t nInputSamplePoints,
                                uint32_t& nSamplePoints,
                                uint32_t nWalks);

    // splats boundary data
    void splatBoundaryData(GPUBuffer& samplePoints,
                           GPUBuffer& boundaryData,
                           float radiusClamp,
                           float kernelRegularization,
                           uint32_t nSamplePoints,
                           uint32_t& nSamplesSplatted);

    // splats source data
    void splatSourceData(float radiusClamp,
                         float kernelRegularization);

    // members
    std::shared_ptr<GPUTaskHandle<T, DIM>> handle;
    GPUContext& context;
    std::shared_ptr<GPUWalkSettings> walkSettings;
    std::shared_ptr<GPUPDE> pde;
    std::shared_ptr<GPUBoundarySampler> absorbingBoundarySampler;
    std::shared_ptr<GPUBoundarySampler> reflectingBoundarySampler;
    std::shared_ptr<GPUDomainSampler> domainSampler;
    GPUWalkOnStars<T, DIM> walkOnStars;
    GPUBoundaryValueCaching<T, DIM> boundaryValueCaching;
    uint32_t nEvaluationPoints;
    uint32_t nAbsorbingBoundarySamples;
    uint32_t nAbsorbingBoundarySamplesSplatted;
    uint32_t nAbsorbingBoundarySamplesNormalAligned;
    uint32_t nAbsorbingBoundarySamplesNormalAlignedSplatted;
    uint32_t nReflectingBoundarySamples;
    uint32_t nReflectingBoundarySamplesSplatted;
    uint32_t nReflectingBoundarySamplesNormalAligned;
    uint32_t nReflectingBoundarySamplesNormalAlignedSplatted;
    uint32_t nDomainSamplesSplatted;
    GPUBuffer evaluationPoints;
    GPUBuffer evaluationStatistics;
    GPUBuffer absorbingBoundarySamplesStatistics;
    GPUBuffer absorbingBoundarySamplesNormalAlignedStatistics;
    GPUBuffer reflectingBoundarySamplesStatistics;
    GPUBuffer reflectingBoundarySamplesNormalAlignedStatistics;
    GPUBuffer absorbingBoundarySamplesData;
    GPUBuffer absorbingBoundarySamplesNormalAlignedData;
    GPUBuffer reflectingBoundarySamplesData;
    GPUBuffer reflectingBoundarySamplesNormalAlignedData;
    GPUBuffer globalCounter;
    GPUBuffer absorbingBoundarySamplesLocks;
    GPUBuffer absorbingBoundarySamplesNormalAlignedLocks;
    GPUBuffer reflectingBoundarySamplesLocks;
    GPUBuffer reflectingBoundarySamplesNormalAlignedLocks;
    ComputeShader seedRngsShader;
    ComputeShader generateAbsorbingBoundarySamplesShader;
    ComputeShader generateReflectingBoundarySamplesShader;
    ComputeShader generateDomainSamplesShader;
    ComputeShader resetSampleStatisticsShader;
    ComputeShader runPointEstimatorShader;
    ComputeShader getBoundaryDataShader;
    ComputeShader splatBoundaryDataShader;
    ComputeShader splatSourceDataShader;
    ComputeShader resetEvaluationStatisticsShader;
    ComputeShader getEvaluationOutputsShader;
    GPUSeedRngs seedAbsorbingBoundarySamplerRngsEntryPoint;
    GPUSeedRngs seedReflectingBoundarySamplerRngsEntryPoint;
    GPUSeedRngs seedDomainSamplerRngsEntryPoint;
    GPUSeedRngs seedAbsorbingBoundarySamplesRngsEntryPoint;
    GPUSeedRngs seedAbsorbingBoundarySamplesNormalAlignedRngsEntryPoint;
    GPUSeedRngs seedReflectingBoundarySamplesRngsEntryPoint;
    GPUSeedRngs seedReflectingBoundarySamplesNormalAlignedRngsEntryPoint;
    GPUGenerateBoundarySamples<DIM> generateAbsorbingBoundarySamplesEntryPoint;
    GPUGenerateBoundarySamples<DIM> generateAbsorbingBoundarySamplesNormalAlignedEntryPoint;
    GPUGenerateBoundarySamples<DIM> generateReflectingBoundarySamplesEntryPoint;
    GPUGenerateBoundarySamples<DIM> generateReflectingBoundarySamplesNormalAlignedEntryPoint;
    GPUGenerateDomainSamples<DIM> generateDomainSamplesEntryPoint;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindGenerateAbsorbingBoundarySamplesResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindGenerateReflectingBoundarySamplesResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindGenerateDomainSamplesResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindWalkOnStarsResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindBoundaryValueCachingResources;
    std::function<void(const ComputeShader&, const ShaderCursor&)> bindEvaluationOutputsResources;
    uint32_t nResidentThreads;
    uint32_t nThreadsPerGroup;
    bool usePersistentThreads;
    bool printLogs;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
GPUBoundaryValueCachingSolver<T, DIM>::GPUBoundaryValueCachingSolver(std::shared_ptr<GPUTaskHandle<T, DIM>> handle_,
                                                                     std::shared_ptr<GPUWalkSettings> walkSettings_,
                                                                     uint32_t nResidentThreads_,
                                                                     bool usePersistentThreads_,
                                                                     bool printLogs_):
handle(handle_),
context(handle->getContext()),
walkSettings(walkSettings_),
pde(handle->getPDE()),
absorbingBoundarySampler(handle->getAbsorbingBoundarySampler()),
reflectingBoundarySampler(handle->getReflectingBoundarySampler()),
domainSampler(handle->getDomainSampler()),
walkOnStars(handle->getPDE(), handle->getGeometricQueries(), walkSettings_),
boundaryValueCaching(handle->getPDE(), walkSettings_),
nEvaluationPoints(0),
nAbsorbingBoundarySamples(0),
nAbsorbingBoundarySamplesSplatted(0),
nAbsorbingBoundarySamplesNormalAligned(0),
nAbsorbingBoundarySamplesNormalAlignedSplatted(0),
nReflectingBoundarySamples(0),
nReflectingBoundarySamplesSplatted(0),
nReflectingBoundarySamplesNormalAligned(0),
nReflectingBoundarySamplesNormalAlignedSplatted(0),
nDomainSamplesSplatted(0),
generateAbsorbingBoundarySamplesEntryPoint(seedAbsorbingBoundarySamplerRngsEntryPoint.getBuffer(),
                                           SampleType::OnAbsorbingBoundary, false),
generateAbsorbingBoundarySamplesNormalAlignedEntryPoint(seedAbsorbingBoundarySamplerRngsEntryPoint.getBuffer(),
                                                        SampleType::OnAbsorbingBoundary, true),
generateReflectingBoundarySamplesEntryPoint(seedReflectingBoundarySamplerRngsEntryPoint.getBuffer(),
                                            SampleType::OnReflectingBoundary, false),
generateReflectingBoundarySamplesNormalAlignedEntryPoint(seedReflectingBoundarySamplerRngsEntryPoint.getBuffer(),
                                                        SampleType::OnReflectingBoundary, true),
generateDomainSamplesEntryPoint(seedDomainSamplerRngsEntryPoint.getBuffer()),
nResidentThreads(nResidentThreads_),
nThreadsPerGroup(256),
usePersistentThreads(usePersistentThreads_),
printLogs(printLogs_)
{
    // initialize shader programs
    GPUModule mainModule;
    mainModule.load(context, handle->getWosxGpuDirectoryPath() + "/entrypoints/boundary-value-caching.cs.slang");
    const std::vector<GPUModule>& libraryModules = handle->getLibraryModules();
    seedRngsShader.loadProgram(context, mainModule, libraryModules, "seedRngs");
    generateAbsorbingBoundarySamplesShader.loadProgram(context, mainModule, libraryModules, "generateAbsorbingBoundarySamples");
    generateReflectingBoundarySamplesShader.loadProgram(context, mainModule, libraryModules, "generateReflectingBoundarySamples");
    generateDomainSamplesShader.loadProgram(context, mainModule, libraryModules, "generateDomainSamples");
    resetSampleStatisticsShader.loadProgram(context, mainModule, libraryModules, "resetSampleStatistics");
    std::string runPointEstimatorShaderName = usePersistentThreads ? "runPersistentPointEstimator" : "runPointEstimator";
    runPointEstimatorShader.loadProgram(context, mainModule, libraryModules, runPointEstimatorShaderName);
    getBoundaryDataShader.loadProgram(context, mainModule, libraryModules, "getBoundaryData");
    splatBoundaryDataShader.loadProgram(context, mainModule, libraryModules, "splatBoundaryData");
    splatSourceDataShader.loadProgram(context, mainModule, libraryModules, "splatSourceData");
    resetEvaluationStatisticsShader.loadProgram(context, mainModule, libraryModules, "resetEvaluationStatistics");
    getEvaluationOutputsShader.loadProgram(context, mainModule, libraryModules, "getEvaluationOutputs");

    // set normal offsets for the absorbing and reflecting boundaries
    float absorbingBoundaryNormalOffset = absorbingBoundarySampler->getNormalOffsetForBoundary();
    generateAbsorbingBoundarySamplesEntryPoint.setNormalOffsetForBoundary(absorbingBoundaryNormalOffset);
    generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.setNormalOffsetForBoundary(absorbingBoundaryNormalOffset);    
    float reflectingBoundaryNormalOffset = reflectingBoundarySampler->getNormalOffsetForBoundary();
    generateReflectingBoundarySamplesEntryPoint.setNormalOffsetForBoundary(reflectingBoundaryNormalOffset);
    generateReflectingBoundarySamplesNormalAlignedEntryPoint.setNormalOffsetForBoundary(reflectingBoundaryNormalOffset);

    // bind shader resources
    bindGenerateAbsorbingBoundarySamplesResources = [this](const ComputeShader& shader,
                                                           const ShaderCursor& cursor) {
        ComPtr<IShaderObject> samplerShaderObject = shader.createShaderObject(
            context, absorbingBoundarySampler->getReflectionType());
        ShaderCursor samplerShaderCursor(samplerShaderObject);
        absorbingBoundarySampler->setResources(samplerShaderCursor, printLogs);
        cursor.getPath("gAbsorbingBoundarySampler").setObject(samplerShaderObject);
    };
    bindGenerateReflectingBoundarySamplesResources = [this](const ComputeShader& shader,
                                                            const ShaderCursor& cursor) {
        ComPtr<IShaderObject> samplerShaderObject = shader.createShaderObject(
            context, reflectingBoundarySampler->getReflectionType());
        ShaderCursor samplerShaderCursor(samplerShaderObject);
        reflectingBoundarySampler->setResources(samplerShaderCursor, printLogs);
        cursor.getPath("gReflectingBoundarySampler").setObject(samplerShaderObject);
    };
    bindGenerateDomainSamplesResources = [this](const ComputeShader& shader,
                                                const ShaderCursor& cursor) {
        ComPtr<IShaderObject> samplerShaderObject = shader.createShaderObject(
            context, domainSampler->getReflectionType());
        ShaderCursor samplerShaderCursor(samplerShaderObject);
        domainSampler->setResources(samplerShaderCursor, printLogs);
        cursor.getPath("gDomainSampler").setObject(samplerShaderObject);
    };
    bindWalkOnStarsResources = [this](const ComputeShader& shader,
                                      const ShaderCursor& cursor) {
        ComPtr<IShaderObject> walkOnStarsShaderObject = shader.createShaderObject(
            context, walkOnStars.getReflectionType());
        ShaderCursor walkOnStarsShaderCursor(walkOnStarsShaderObject);
        walkOnStars.setResources(walkOnStarsShaderCursor, printLogs);
        cursor.getPath("gPointEstimator").setObject(walkOnStarsShaderObject);
    };
    bindBoundaryValueCachingResources = [this](const ComputeShader& shader,
                                               const ShaderCursor& cursor) {
        ComPtr<IShaderObject> boundaryValueCachingShaderObject = shader.createShaderObject(
            context, boundaryValueCaching.getReflectionType());
        ShaderCursor boundaryValueCachingShaderCursor(boundaryValueCachingShaderObject);
        boundaryValueCaching.setResources(boundaryValueCachingShaderCursor, printLogs);
        cursor.getPath("gBoundaryValueCaching").setObject(boundaryValueCachingShaderObject);
    };
    bindEvaluationOutputsResources = [this](const ComputeShader& shader,
                                            const ShaderCursor& cursor) {
        ComPtr<IShaderObject> walkSettingsShaderObject = shader.createShaderObject(
            context, walkSettings->getReflectionType());
        ShaderCursor walkSettingsShaderCursor(walkSettingsShaderObject);
        walkSettings->setResources(walkSettingsShaderCursor, printLogs);
        cursor.getPath("gWalkSettings").setObject(walkSettingsShaderObject);

        ComPtr<IShaderObject> pdeShaderObject = shader.createShaderObject(
            context, pde->getReflectionType());
        ShaderCursor pdeShaderCursor(pdeShaderObject);
        pde->setResources(pdeShaderCursor, printLogs);
        cursor.getPath("gPDE").setObject(pdeShaderObject);
    };
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::populateEvaluationPoints(const std::vector<GPUBVCEvaluationPoint<DIM>>& inputEvaluationPoints)
{
    // allocate evaluation points and statistics
    nEvaluationPoints = (uint32_t)inputEvaluationPoints.size();
    evaluationPoints.allocate<GPUBVCEvaluationPoint<DIM>>(context, false, inputEvaluationPoints);
    evaluationStatistics.allocate<GPUBVCEvaluationStatistics<T, DIM>>(context.device, true, nullptr, nEvaluationPoints);

    // reset evaluation statistics
    resetEvaluationStatistics();
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::seedRngs(uint32_t nRngs, GPUSeedRngs& entryPoint)
{
    // allocate entry point data
    entryPoint.allocate(context, nRngs);

    // run shader
    if (nRngs > 0) {
        uint32_t nThreadGroups = countThreadGroups(nRngs, nThreadsPerGroup, printLogs);
        runShader<GPUSeedRngs>(context, seedRngsShader, entryPoint, {}, nThreadGroups, 1, printLogs);
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::generateAbsorbingBoundarySamples(uint32_t nAbsorbingBoundarySamples)
{
    uint32_t nSamples = absorbingBoundarySampler->getSampleCount(nAbsorbingBoundarySamples, false);
    uint32_t nSamplesNormalAligned = absorbingBoundarySampler->getSampleCount(nAbsorbingBoundarySamples, true);
    if (nSamples != generateAbsorbingBoundarySamplesEntryPoint.getBufferSize() ||
        nSamplesNormalAligned != generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize()) {
        // seed sampler rngs
        uint32_t nRngs = std::max(nSamples, nSamplesNormalAligned);
        seedRngs(nRngs, seedAbsorbingBoundarySamplerRngsEntryPoint);

        // allocate entry point data
        generateAbsorbingBoundarySamplesEntryPoint.allocate(context, nSamples);
        generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.allocate(context, nSamplesNormalAligned);

        // allocate boundary data
        absorbingBoundarySamplesData.allocate<GPUBVCBoundaryData<T>>(context.device, true, nullptr, nSamples);
        absorbingBoundarySamplesNormalAlignedData.allocate<GPUBVCBoundaryData<T>>(context.device, true, nullptr, nSamplesNormalAligned);
    }

    // generate samples
    if (nSamples > 0) {
        uint32_t nThreadGroups = countThreadGroups(nSamples, nThreadsPerGroup, printLogs);
        runShader<GPUGenerateBoundarySamples<DIM>>(context, generateAbsorbingBoundarySamplesShader,
                                                   generateAbsorbingBoundarySamplesEntryPoint,
                                                   bindGenerateAbsorbingBoundarySamplesResources,
                                                   nThreadGroups, 1, printLogs);
    }
    if (nSamplesNormalAligned > 0) {
        uint32_t nThreadGroups = countThreadGroups(nSamplesNormalAligned, nThreadsPerGroup, printLogs);
        runShader<GPUGenerateBoundarySamples<DIM>>(context, generateAbsorbingBoundarySamplesShader,
                                                   generateAbsorbingBoundarySamplesNormalAlignedEntryPoint,
                                                   bindGenerateAbsorbingBoundarySamplesResources,
                                                   nThreadGroups, 1, printLogs);
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::generateReflectingBoundarySamples(uint32_t nReflectingBoundarySamples)
{
    uint32_t nSamples = reflectingBoundarySampler->getSampleCount(nReflectingBoundarySamples, false);
    uint32_t nSamplesNormalAligned = reflectingBoundarySampler->getSampleCount(nReflectingBoundarySamples, true);
    if (nSamples != generateReflectingBoundarySamplesEntryPoint.getBufferSize() ||
        nSamplesNormalAligned != generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize()) {
        // seed sampler rngs
        uint32_t nRngs = std::max(nSamples, nSamplesNormalAligned);
        seedRngs(nRngs, seedReflectingBoundarySamplerRngsEntryPoint);

        // allocate entry point data
        generateReflectingBoundarySamplesEntryPoint.allocate(context, nSamples);
        generateReflectingBoundarySamplesNormalAlignedEntryPoint.allocate(context, nSamplesNormalAligned);

        // allocate boundary data
        reflectingBoundarySamplesData.allocate<GPUBVCBoundaryData<T>>(context.device, true, nullptr, nSamples);
        reflectingBoundarySamplesNormalAlignedData.allocate<GPUBVCBoundaryData<T>>(context.device, true, nullptr, nSamplesNormalAligned);
    }

    // generate samples
    if (nSamples > 0) {
        uint32_t nThreadGroups = countThreadGroups(nSamples, nThreadsPerGroup, printLogs);
        runShader<GPUGenerateBoundarySamples<DIM>>(context, generateReflectingBoundarySamplesShader,
                                                   generateReflectingBoundarySamplesEntryPoint,
                                                   bindGenerateReflectingBoundarySamplesResources,
                                                   nThreadGroups, 1, printLogs);
    }
    if (nSamplesNormalAligned > 0) {
        uint32_t nThreadGroups = countThreadGroups(nSamplesNormalAligned, nThreadsPerGroup, printLogs);
        runShader<GPUGenerateBoundarySamples<DIM>>(context, generateReflectingBoundarySamplesShader,
                                                   generateReflectingBoundarySamplesNormalAlignedEntryPoint,
                                                   bindGenerateReflectingBoundarySamplesResources,
                                                   nThreadGroups, 1, printLogs);
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::generateDomainSamples(uint32_t nDomainSamples)
{
    if (nDomainSamples != generateDomainSamplesEntryPoint.getBufferSize()) {
        // seed sampler rngs
        seedRngs(nDomainSamples, seedDomainSamplerRngsEntryPoint);

        // allocate entry point data
        generateDomainSamplesEntryPoint.allocate(context, nDomainSamples);
    }

    // generate samples
    if (nDomainSamples > 0) {
        uint32_t nThreadGroups = countThreadGroups(nDomainSamples, nThreadsPerGroup, printLogs);
        runShader<GPUGenerateDomainSamples<DIM>>(context, generateDomainSamplesShader,
                                                 generateDomainSamplesEntryPoint,
                                                 bindGenerateDomainSamplesResources,
                                                 nThreadGroups, 1, printLogs);
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::generateSamples(uint32_t absorbingBoundaryCacheSize,
                                                            uint32_t reflectingBoundaryCacheSize,
                                                            uint32_t domainCacheSize)
{
    generateAbsorbingBoundarySamples(absorbingBoundaryCacheSize);
    generateReflectingBoundarySamples(reflectingBoundaryCacheSize);
    generateDomainSamples(domainCacheSize);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::resetSampleStatistics(GPUBuffer& sampleStatistics,
                                                                  uint32_t nSamplePoints)
{
    // initialize entry point data
    GPUResetSampleStatistics entryPoint(sampleStatistics, nSamplePoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPUResetSampleStatistics>(context, resetSampleStatisticsShader, entryPoint,
                                        {}, nThreadGroups, 1, printLogs);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::runPointEstimator(GPUBuffer& samplePoints,
                                                              GPUBuffer& rngs,
                                                              GPUBuffer& sampleStatistics,
                                                              uint32_t nInputSamplePoints,
                                                              uint32_t nSamplePoints,
                                                              uint32_t nWalks,
                                                              uint32_t nDispatchCalls)
{
    // initialize entry point data
    GPURunPointEstimator entryPoint(samplePoints, rngs, sampleStatistics,
                                    nInputSamplePoints, nSamplePoints, nWalks);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPURunPointEstimator>(context, runPointEstimatorShader,
                                    entryPoint, bindWalkOnStarsResources,
                                    nThreadGroups, nDispatchCalls, printLogs);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::runPersistentPointEstimator(GPUBuffer& samplePoints,
                                                                        GPUBuffer& rngs,
                                                                        GPUBuffer& sampleStatistics,
                                                                        GPUBuffer& locks,
                                                                        uint32_t nInputSamplePoints,
                                                                        uint32_t nWalks)
{
    uint64_t nTotalWorkItems = (uint64_t)(nInputSamplePoints)*nWalks;
    uint32_t nPersistentThreads = nResidentThreads;

    // reset global counter to 0
    std::vector<uint64_t> counterData = {0ULL};
    globalCounter.allocate<uint64_t>(context, true, counterData);

    // initialize entry point data
    GPURunPersistentPointEstimator entryPoint(samplePoints, rngs, sampleStatistics,
                                              globalCounter, locks, nTotalWorkItems,
                                              nInputSamplePoints, nPersistentThreads);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nPersistentThreads, nThreadsPerGroup, printLogs);
    runShader<GPURunPersistentPointEstimator>(context, runPointEstimatorShader,
                                              entryPoint, bindWalkOnStarsResources,
                                              nThreadGroups, 1, printLogs);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::getBoundaryData(GPUBuffer& samplePoints,
                                                            GPUBuffer& sampleStatistics,
                                                            GPUBuffer& boundaryData,
                                                            uint32_t nInputSamplePoints,
                                                            uint32_t nSamplePoints)
{
    // initialize entry point data
    GPUBVCGetBoundaryData entryPoint(samplePoints, sampleStatistics, boundaryData,
                                     nInputSamplePoints, nSamplePoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nInputSamplePoints, nThreadsPerGroup, printLogs);
    runShader<GPUBVCGetBoundaryData>(context, getBoundaryDataShader, entryPoint,
                                     bindBoundaryValueCachingResources,
                                     nThreadGroups, 1, printLogs);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::computeSampleEstimates(GPUBuffer& samplePoints,
                                                                   GPUBuffer& sampleStatistics,
                                                                   GPUBuffer& boundaryData,
                                                                   GPUBuffer& locks,
                                                                   GPUSeedRngs& entryPoint,
                                                                   uint32_t nInputSamplePoints,
                                                                   uint32_t& nSamplePoints,
                                                                   uint32_t nWalks)
{
    if (usePersistentThreads) {
        if (nInputSamplePoints != nSamplePoints) {
            // seed rngs (one per persistent thread)
            nSamplePoints = nInputSamplePoints;
            seedRngs(nResidentThreads, entryPoint);

            // allocate sample statistics and locks
            sampleStatistics.allocate<GPUSampleStatistics<T, DIM>>(context.device, true, nullptr, nSamplePoints);
            std::vector<uint32_t> locksData(nInputSamplePoints, 0);
            locks.allocate<uint32_t>(context, true, locksData);
        }

        // reset sample statistics and run persistent point estimator
        resetSampleStatistics(sampleStatistics, nSamplePoints);
        runPersistentPointEstimator(samplePoints, entryPoint.getBuffer(),
                                    sampleStatistics, locks, nInputSamplePoints, nWalks);

    } else {
        // decide how many copies of every sample we need: we want at least
        // ceil(nResidentThreads / nInputSamplePoints) threads per sample point,
        // but never more than the number of walks we eventually want
        uint32_t nSampleCopies = 1;
        if (nInputSamplePoints > 0 && nInputSamplePoints < nResidentThreads) {
            nSampleCopies = (nResidentThreads + nInputSamplePoints - 1)/nInputSamplePoints;
            nSampleCopies = std::min(nSampleCopies, nWalks);
        }

        if (nSampleCopies*nInputSamplePoints != nSamplePoints) {
            // seed rngs
            nSamplePoints = nSampleCopies*nInputSamplePoints;
            seedRngs(nSamplePoints, entryPoint);

            // allocate sample statistics
            sampleStatistics.allocate<GPUSampleStatistics<T, DIM>>(context.device, true, nullptr, nSamplePoints);
        }

        // reset sample statistics
        resetSampleStatistics(sampleStatistics, nSamplePoints);

        // decide how many walks does each sample copy performs, and run point estimator
        uint32_t nWalksPerSampleCopy = (nWalks + nSampleCopies - 1)/nSampleCopies;
        runPointEstimator(samplePoints, entryPoint.getBuffer(), sampleStatistics,
                          nInputSamplePoints, nSamplePoints, 1, nWalksPerSampleCopy);
    }

    // get boundary data
    getBoundaryData(samplePoints, sampleStatistics, boundaryData,
                    nInputSamplePoints, nSamplePoints);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::computeSampleEstimates(uint32_t nWalksForSolutionEstimates,
                                                                   uint32_t nWalksForGradientEstimates)
{
    if (generateAbsorbingBoundarySamplesEntryPoint.getBufferSize() > 0) {
        computeSampleEstimates(generateAbsorbingBoundarySamplesEntryPoint.getBuffer(),
                               absorbingBoundarySamplesStatistics,
                               absorbingBoundarySamplesData,
                               absorbingBoundarySamplesLocks,
                               seedAbsorbingBoundarySamplesRngsEntryPoint,
                               generateAbsorbingBoundarySamplesEntryPoint.getBufferSize(),
                               nAbsorbingBoundarySamples, nWalksForGradientEstimates);
    }
    if (generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
        computeSampleEstimates(generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBuffer(),
                               absorbingBoundarySamplesNormalAlignedStatistics,
                               absorbingBoundarySamplesNormalAlignedData,
                               absorbingBoundarySamplesNormalAlignedLocks,
                               seedAbsorbingBoundarySamplesNormalAlignedRngsEntryPoint,
                               generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize(),
                               nAbsorbingBoundarySamplesNormalAligned, nWalksForGradientEstimates);
    }
    if (generateReflectingBoundarySamplesEntryPoint.getBufferSize() > 0) {
        computeSampleEstimates(generateReflectingBoundarySamplesEntryPoint.getBuffer(),
                               reflectingBoundarySamplesStatistics,
                               reflectingBoundarySamplesData,
                               reflectingBoundarySamplesLocks,
                               seedReflectingBoundarySamplesRngsEntryPoint,
                               generateReflectingBoundarySamplesEntryPoint.getBufferSize(),
                               nReflectingBoundarySamples, nWalksForSolutionEstimates);
    }
    if (generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
        computeSampleEstimates(generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBuffer(),
                               reflectingBoundarySamplesNormalAlignedStatistics,
                               reflectingBoundarySamplesNormalAlignedData,
                               reflectingBoundarySamplesNormalAlignedLocks,
                               seedReflectingBoundarySamplesNormalAlignedRngsEntryPoint,
                               generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize(),
                               nReflectingBoundarySamplesNormalAligned, nWalksForSolutionEstimates);
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::splatBoundaryData(GPUBuffer& samplePoints,
                                                              GPUBuffer& boundaryData,
                                                              float radiusClamp,
                                                              float kernelRegularization,
                                                              uint32_t nSamplePoints,
                                                              uint32_t& nSamplesSplatted)
{
    // initialize entry point data
    GPUBVCSplatBoundaryData entryPoint(samplePoints, boundaryData,
                                       evaluationPoints, evaluationStatistics,
                                       radiusClamp, kernelRegularization,
                                       nSamplesSplatted, nEvaluationPoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nEvaluationPoints, nThreadsPerGroup, printLogs);
    runShader<GPUBVCSplatBoundaryData>(context, splatBoundaryDataShader, entryPoint,
                                       bindBoundaryValueCachingResources,
                                       nThreadGroups, nSamplePoints, printLogs);

    // update number of samples splatted
    nSamplesSplatted += nSamplePoints;
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::splatSourceData(float radiusClamp,
                                                            float kernelRegularization)
{
    // initialize entry point data
    GPUBVCSplatSourceData entryPoint(generateDomainSamplesEntryPoint.getBuffer(),
                                     evaluationPoints, evaluationStatistics,
                                     radiusClamp, kernelRegularization,
                                     nDomainSamplesSplatted, nEvaluationPoints);

    // run shader
    uint32_t nSamplePoints = generateDomainSamplesEntryPoint.getBufferSize();
    uint32_t nThreadGroups = countThreadGroups(nEvaluationPoints, nThreadsPerGroup, printLogs);
    runShader<GPUBVCSplatSourceData>(context, splatSourceDataShader, entryPoint,
                                     bindBoundaryValueCachingResources,
                                     nThreadGroups, nSamplePoints, printLogs);

    // update number of samples splatted
    nDomainSamplesSplatted += nSamplePoints;
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::splat(float radiusClamp,
                                                  float kernelRegularization)
{
    if (nEvaluationPoints > 0) {
        if (generateAbsorbingBoundarySamplesEntryPoint.getBufferSize() > 0) {
            splatBoundaryData(generateAbsorbingBoundarySamplesEntryPoint.getBuffer(),
                              absorbingBoundarySamplesData, radiusClamp, kernelRegularization,
                              generateAbsorbingBoundarySamplesEntryPoint.getBufferSize(),
                              nAbsorbingBoundarySamplesSplatted);
        }
        if (generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
            splatBoundaryData(generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBuffer(),
                              absorbingBoundarySamplesNormalAlignedData, radiusClamp, kernelRegularization,
                              generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize(),
                              nAbsorbingBoundarySamplesNormalAlignedSplatted);
        }
        if (generateReflectingBoundarySamplesEntryPoint.getBufferSize() > 0) {
            splatBoundaryData(generateReflectingBoundarySamplesEntryPoint.getBuffer(),
                              reflectingBoundarySamplesData, radiusClamp, kernelRegularization,
                              generateReflectingBoundarySamplesEntryPoint.getBufferSize(),
                              nReflectingBoundarySamplesSplatted);
        }
        if (generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
            splatBoundaryData(generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBuffer(),
                              reflectingBoundarySamplesNormalAlignedData, radiusClamp, kernelRegularization,
                              generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize(),
                              nReflectingBoundarySamplesNormalAlignedSplatted);
        }
        if (generateDomainSamplesEntryPoint.getBufferSize() > 0) {
            splatSourceData(radiusClamp, kernelRegularization);
        }
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::resetEvaluationStatistics()
{
    // initialize entry point data
    GPUResetBVCEvaluationStatistics entryPoint(evaluationStatistics, nEvaluationPoints);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nEvaluationPoints, nThreadsPerGroup, printLogs);
    runShader<GPUResetBVCEvaluationStatistics>(context, resetEvaluationStatisticsShader,
                                               entryPoint, {}, nThreadGroups, 1, printLogs);

    // reset number of samples splatted
    nAbsorbingBoundarySamplesSplatted = 0;
    nAbsorbingBoundarySamplesNormalAlignedSplatted = 0;
    nReflectingBoundarySamplesSplatted = 0;
    nReflectingBoundarySamplesNormalAlignedSplatted = 0;
    nDomainSamplesSplatted = 0;
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::getEvaluationOutputs(std::vector<GPUBVCEvaluationOutputs<T, DIM>>& evaluationOutputs)
{
    // allocate entry point data
    GPUGetBVCEvaluationOutputs<T, DIM> entryPoint(evaluationPoints, evaluationStatistics, nEvaluationPoints);
    entryPoint.allocate(context);

    // run shader
    uint32_t nThreadGroups = countThreadGroups(nEvaluationPoints, nThreadsPerGroup, printLogs);
    runShader<GPUGetBVCEvaluationOutputs<T, DIM>>(context, getEvaluationOutputsShader,
                                                  entryPoint, bindEvaluationOutputsResources,
                                                  nThreadGroups, 1, printLogs);

    // read results from GPU
    entryPoint.read(context, evaluationOutputs);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::updateNormalOffsetForAbsorbingBoundary(float offset)
{
    absorbingBoundarySampler->reallocate(context, offset);
    generateAbsorbingBoundarySamplesEntryPoint.setNormalOffsetForBoundary(offset);
    generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.setNormalOffsetForBoundary(offset);
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::getAbsorbingBoundarySamples(std::vector<GPUSamplePoint<DIM>>& samplePoints,
                                                                        bool returnBoundaryNormalAligned)
{
    if (returnBoundaryNormalAligned) {
        if (generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
            generateAbsorbingBoundarySamplesNormalAlignedEntryPoint.read(context, samplePoints);
        }

    } else {
        if (generateAbsorbingBoundarySamplesEntryPoint.getBufferSize() > 0) {
            generateAbsorbingBoundarySamplesEntryPoint.read(context, samplePoints);
        }
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::getReflectingBoundarySamples(std::vector<GPUSamplePoint<DIM>>& samplePoints,
                                                                         bool returnBoundaryNormalAligned)
{
    if (returnBoundaryNormalAligned) {
        if (generateReflectingBoundarySamplesNormalAlignedEntryPoint.getBufferSize() > 0) {
            generateReflectingBoundarySamplesNormalAlignedEntryPoint.read(context, samplePoints);
        }

    } else {
        if (generateReflectingBoundarySamplesEntryPoint.getBufferSize() > 0) {
            generateReflectingBoundarySamplesEntryPoint.read(context, samplePoints);
        }
    }
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::getDomainSamples(std::vector<GPUSamplePoint<DIM>>& samplePoints)
{
    if (generateDomainSamplesEntryPoint.getBufferSize() > 0) {
        generateDomainSamplesEntryPoint.read(context, samplePoints);
    }
}

template <typename T, size_t DIM>
bool GPUBoundaryValueCachingSolver<T, DIM>::usingPersistentThreads() const
{
    return usePersistentThreads;
}

template <typename T, size_t DIM>
void GPUBoundaryValueCachingSolver<T, DIM>::enableLogging(bool enable)
{
    printLogs = enable;
}

} // wosx
