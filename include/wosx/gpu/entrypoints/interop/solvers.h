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

#include <wosx/gpu/entrypoints/interop/pde.h>
#include <wosx/gpu/entrypoints/interop/samplers.h>

namespace wosx {

enum class GPUPointEstimatorType: int {
    WalkOnSpheres = 1,
    WalkOnStars = 2
};

struct GPUWalkSettings: public GPUShaderObject {
    // members
    float epsilonShellForAbsorbingBoundary = 1e-3f;
    float epsilonShellForReflectingBoundary = 1e-3f;
    float silhouettePrecision = 1e-3f;
    float russianRouletteThreshold = 0.0f;
    uint32_t maxWalkLength = 10000;
    uint32_t stepsBeforeUsingMaximalSpheres = 10000;
    uint32_t solveDoubleSided = 0;
    uint32_t useGradientControlVariates = 1;
    uint32_t useGradientAntitheticVariates = 1;
    uint32_t ignoreAbsorbingBoundaryContribution = 0;
    uint32_t ignoreReflectingBoundaryContribution = 0;
    uint32_t ignoreSourceContribution = 0;

    // sets GPU resources, and returns reflection type
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
};

template <typename T, size_t DIM>
class GPUWalkOnSpheres: public GPUShaderObject {
public:
    // constructor
    GPUWalkOnSpheres(std::shared_ptr<GPUPDE> pde_,
                     std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                     std::shared_ptr<GPUWalkSettings> walkSettings_);

    // sets GPU resources, and returns reflection type
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    std::shared_ptr<GPUPDE> pde;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    std::shared_ptr<GPUWalkSettings> walkSettings;
};

template <typename T, size_t DIM>
class GPUWalkOnStars: public GPUShaderObject {
public:
    // constructor
    GPUWalkOnStars(std::shared_ptr<GPUPDE> pde_,
                   std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                   std::shared_ptr<GPUWalkSettings> walkSettings_);

    // sets GPU resources, and returns reflection type
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    std::shared_ptr<GPUPDE> pde;
    std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries;
    std::shared_ptr<GPUWalkSettings> walkSettings;
};

class GPURunPointEstimator: public GPUShaderEntryPoint {
public:
    // constructor
    GPURunPointEstimator(GPUBuffer& samplePoints_,
                         GPUBuffer& rngs_,
                         GPUBuffer& sampleStatistics_,
                         uint32_t nInputSamplePoints_,
                         uint32_t nSamplePoints_,
                         uint32_t nWalks_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& samplePoints;
    GPUBuffer& rngs;
    GPUBuffer& sampleStatistics;
    uint32_t nInputSamplePoints;
    uint32_t nSamplePoints;
    uint32_t nWalks;
};

class GPURunPersistentPointEstimator: public GPUShaderEntryPoint {
public:
    // constructor
    GPURunPersistentPointEstimator(GPUBuffer& samplePoints_,
                                   GPUBuffer& rngs_,
                                   GPUBuffer& sampleStatistics_,
                                   GPUBuffer& globalCounter_,
                                   GPUBuffer& locks_,
                                   uint64_t nTotalWorkItems_,
                                   uint32_t nInputSamplePoints_,
                                   uint32_t nPersistentThreads_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& samplePoints;
    GPUBuffer& rngs;
    GPUBuffer& sampleStatistics;
    GPUBuffer& globalCounter;
    GPUBuffer& locks;
    uint64_t nTotalWorkItems;
    uint32_t nInputSamplePoints;
    uint32_t nPersistentThreads;
};

template <typename T, size_t DIM>
class GPUBoundaryValueCaching: public GPUShaderObject {
public:
    // constructor
    GPUBoundaryValueCaching(std::shared_ptr<GPUPDE> pde_,
                            std::shared_ptr<GPUWalkSettings> walkSettings_);

    // sets GPU resources, and returns reflection type
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    std::shared_ptr<GPUPDE> pde;
    std::shared_ptr<GPUWalkSettings> walkSettings;
};

class GPUBVCGetBoundaryData: public GPUShaderEntryPoint {
public:
    // constructor
    GPUBVCGetBoundaryData(GPUBuffer& samplePoints_,
                          GPUBuffer& sampleStatistics_,
                          GPUBuffer& boundaryData_,
                          uint32_t nInputSamplePoints_,
                          uint32_t nSamplePoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;

private:
    // members
    GPUBuffer& samplePoints;
    GPUBuffer& sampleStatistics;
    GPUBuffer& boundaryData;
    uint32_t nInputSamplePoints;
    uint32_t nSamplePoints;
};

class GPUBVCSplatBoundaryData: public GPUShaderEntryPoint {
public:
    // constructor
    GPUBVCSplatBoundaryData(GPUBuffer& samplePoints_,
                            GPUBuffer& boundaryData_,
                            GPUBuffer& evaluationPoints_,
                            GPUBuffer& evaluationStatistics_,
                            float radiusClamp_,
                            float kernelRegularization_,
                            uint32_t nSamplesSplatted_,
                            uint32_t nEvaluationPoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    void setDispatchResources(const ShaderCursor& cursor, uint32_t dispatchIndex) const;

private:
    // members
    GPUBuffer& samplePoints;
    GPUBuffer& boundaryData;
    GPUBuffer& evaluationPoints;
    GPUBuffer& evaluationStatistics;
    float radiusClamp;
    float kernelRegularization;
    uint32_t nSamplesSplatted;
    uint32_t nEvaluationPoints;
};

class GPUBVCSplatSourceData: public GPUShaderEntryPoint {
public:
    // constructor
    GPUBVCSplatSourceData(GPUBuffer& samplePoints_,
                          GPUBuffer& evaluationPoints_,
                          GPUBuffer& evaluationStatistics_,
                          float radiusClamp_,
                          float kernelRegularization_,
                          uint32_t nSamplesSplatted_,
                          uint32_t nEvaluationPoints_);

    // sets GPU resources
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    void setDispatchResources(const ShaderCursor& cursor, uint32_t dispatchIndex) const;

private:
    // members
    GPUBuffer& samplePoints;
    GPUBuffer& evaluationPoints;
    GPUBuffer& evaluationStatistics;
    float radiusClamp;
    float kernelRegularization;
    uint32_t nSamplesSplatted;
    uint32_t nEvaluationPoints;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

inline void GPUWalkSettings::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["epsilonShellForAbsorbingBoundary"].setData(epsilonShellForAbsorbingBoundary);
    cursor["epsilonShellForReflectingBoundary"].setData(epsilonShellForReflectingBoundary);
    cursor["silhouettePrecision"].setData(silhouettePrecision);
    cursor["russianRouletteThreshold"].setData(russianRouletteThreshold);
    cursor["maxWalkLength"].setData(maxWalkLength);
    cursor["stepsBeforeUsingMaximalSpheres"].setData(stepsBeforeUsingMaximalSpheres);
    cursor["solveDoubleSided"].setData(solveDoubleSided);
    cursor["useGradientControlVariates"].setData(useGradientControlVariates);
    cursor["useGradientAntitheticVariates"].setData(useGradientAntitheticVariates);
    cursor["ignoreAbsorbingBoundaryContribution"].setData(ignoreAbsorbingBoundaryContribution);
    cursor["ignoreReflectingBoundaryContribution"].setData(ignoreReflectingBoundaryContribution);
    cursor["ignoreSourceContribution"].setData(ignoreSourceContribution);
    if (printLogs) printReflectionInfo(cursor, 12, getReflectionType());
}

inline std::string GPUWalkSettings::getReflectionType() const
{
    return "WalkSettings";
}

template <typename T, size_t DIM>
GPUWalkOnSpheres<T, DIM>::GPUWalkOnSpheres(std::shared_ptr<GPUPDE> pde_,
                                           std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                                           std::shared_ptr<GPUWalkSettings> walkSettings_):
pde(pde_),
geometricQueries(geometricQueries_),
walkSettings(walkSettings_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUWalkOnSpheres<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    pde->setResources(cursor["pde"], printLogs);
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    walkSettings->setResources(cursor["walkSettings"], printLogs);
    if (printLogs) printReflectionInfo(cursor, 3, getReflectionType());
}

template <typename T, size_t DIM>
std::string GPUWalkOnSpheres<T, DIM>::getReflectionType() const
{
    std::string arguments = getDataTypeName<T>() + ", " +
                            pde->getReflectionType() + ", " +
                            geometricQueries->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "WalkOnSpheres<" + arguments + ">";
}

template <typename T, size_t DIM>
GPUWalkOnStars<T, DIM>::GPUWalkOnStars(std::shared_ptr<GPUPDE> pde_,
                                       std::shared_ptr<GPUGeometricQueries<DIM>> geometricQueries_,
                                       std::shared_ptr<GPUWalkSettings> walkSettings_):
pde(pde_),
geometricQueries(geometricQueries_),
walkSettings(walkSettings_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUWalkOnStars<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    pde->setResources(cursor["pde"], printLogs);
    geometricQueries->setResources(cursor["geometricQueries"], printLogs);
    walkSettings->setResources(cursor["walkSettings"], printLogs);
    if (printLogs) printReflectionInfo(cursor, 3, getReflectionType());
}

template <typename T, size_t DIM>
std::string GPUWalkOnStars<T, DIM>::getReflectionType() const
{
    std::string arguments = getDataTypeName<T>() + ", " +
                            pde->getReflectionType() + ", " +
                            geometricQueries->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "WalkOnStars<" + arguments + ">";
}

inline GPURunPointEstimator::GPURunPointEstimator(GPUBuffer& samplePoints_,
                                                  GPUBuffer& rngs_,
                                                  GPUBuffer& sampleStatistics_,
                                                  uint32_t nInputSamplePoints_,
                                                  uint32_t nSamplePoints_,
                                                  uint32_t nWalks_):
samplePoints(samplePoints_),
rngs(rngs_),
sampleStatistics(sampleStatistics_),
nInputSamplePoints(nInputSamplePoints_),
nSamplePoints(nSamplePoints_),
nWalks(nWalks_)
{
    // do nothing
}

inline void GPURunPointEstimator::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("sampleStatistics").setBinding(sampleStatistics.buffer);
    cursor.getPath("nInputSamplePoints").setData(nInputSamplePoints);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    cursor.getPath("nWalks").setData(nWalks);
    if (printLogs) printReflectionInfo(cursor, 7, "runPointEstimator");
}

inline GPURunPersistentPointEstimator::GPURunPersistentPointEstimator(GPUBuffer& samplePoints_,
                                                                      GPUBuffer& rngs_,
                                                                      GPUBuffer& sampleStatistics_,
                                                                      GPUBuffer& globalCounter_,
                                                                      GPUBuffer& locks_,
                                                                      uint64_t nTotalWorkItems_,
                                                                      uint32_t nInputSamplePoints_,
                                                                      uint32_t nPersistentThreads_):
samplePoints(samplePoints_),
rngs(rngs_),
sampleStatistics(sampleStatistics_),
globalCounter(globalCounter_),
locks(locks_),
nTotalWorkItems(nTotalWorkItems_),
nInputSamplePoints(nInputSamplePoints_),
nPersistentThreads(nPersistentThreads_)
{
    // do nothing
}

inline void GPURunPersistentPointEstimator::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("rngs").setBinding(rngs.buffer);
    cursor.getPath("sampleStatistics").setBinding(sampleStatistics.buffer);
    cursor.getPath("globalCounter").setBinding(globalCounter.buffer);
    cursor.getPath("locks").setBinding(locks.buffer);
    cursor.getPath("nTotalWorkItems").setData(nTotalWorkItems);
    cursor.getPath("nInputSamplePoints").setData(nInputSamplePoints);
    cursor.getPath("nPersistentThreads").setData(nPersistentThreads);
    if (printLogs) printReflectionInfo(cursor, 8, "runPersistentPointEstimator");
}

template <typename T, size_t DIM>
GPUBoundaryValueCaching<T, DIM>::GPUBoundaryValueCaching(std::shared_ptr<GPUPDE> pde_,
                                                         std::shared_ptr<GPUWalkSettings> walkSettings_):
pde(pde_),
walkSettings(walkSettings_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUBoundaryValueCaching<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    pde->setResources(cursor["pde"], printLogs);
    walkSettings->setResources(cursor["walkSettings"], printLogs);
    if (printLogs) printReflectionInfo(cursor, 2, getReflectionType());
}

template <typename T, size_t DIM>
std::string GPUBoundaryValueCaching<T, DIM>::getReflectionType() const
{
    std::string arguments = getDataTypeName<T>() + ", " +
                            pde->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "BoundaryValueCaching<" + arguments + ">";
}

inline GPUBVCGetBoundaryData::GPUBVCGetBoundaryData(GPUBuffer& samplePoints_,
                                                    GPUBuffer& sampleStatistics_,
                                                    GPUBuffer& boundaryData_,
                                                    uint32_t nInputSamplePoints_,
                                                    uint32_t nSamplePoints_):
samplePoints(samplePoints_),
sampleStatistics(sampleStatistics_),
boundaryData(boundaryData_),
nInputSamplePoints(nInputSamplePoints_),
nSamplePoints(nSamplePoints_)
{
    // do nothing
}

inline void GPUBVCGetBoundaryData::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("sampleStatistics").setBinding(sampleStatistics.buffer);
    cursor.getPath("boundaryData").setBinding(boundaryData.buffer);
    cursor.getPath("nInputSamplePoints").setData(nInputSamplePoints);
    cursor.getPath("nSamplePoints").setData(nSamplePoints);
    if (printLogs) printReflectionInfo(cursor, 6, "getBoundaryData");
}

inline GPUBVCSplatBoundaryData::GPUBVCSplatBoundaryData(GPUBuffer& samplePoints_,
                                                        GPUBuffer& boundaryData_,
                                                        GPUBuffer& evaluationPoints_,
                                                        GPUBuffer& evaluationStatistics_,
                                                        float radiusClamp_,
                                                        float kernelRegularization_,
                                                        uint32_t nSamplesSplatted_,
                                                        uint32_t nEvaluationPoints_):
samplePoints(samplePoints_),
boundaryData(boundaryData_),
evaluationPoints(evaluationPoints_),
evaluationStatistics(evaluationStatistics_),
radiusClamp(radiusClamp_),
kernelRegularization(kernelRegularization_),
nSamplesSplatted(nSamplesSplatted_),
nEvaluationPoints(nEvaluationPoints_)
{
    // do nothing
}

inline void GPUBVCSplatBoundaryData::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("boundaryData").setBinding(boundaryData.buffer);
    cursor.getPath("evaluationPoints").setBinding(evaluationPoints.buffer);
    cursor.getPath("evaluationStatistics").setBinding(evaluationStatistics.buffer);
    cursor.getPath("radiusClamp").setData(radiusClamp);
    cursor.getPath("kernelRegularization").setData(kernelRegularization);
    cursor.getPath("nSamplesSplatted").setData(nSamplesSplatted);
    cursor.getPath("sampleIndex").setData(0);
    cursor.getPath("nEvaluationPoints").setData(nEvaluationPoints);
    if (printLogs) printReflectionInfo(cursor, 10, "splatBoundaryData");
}

inline void GPUBVCSplatBoundaryData::setDispatchResources(const ShaderCursor& cursor, uint32_t dispatchIndex) const
{
    cursor.getPath("sampleIndex").setData(dispatchIndex);
}

inline GPUBVCSplatSourceData::GPUBVCSplatSourceData(GPUBuffer& samplePoints_,
                                                    GPUBuffer& evaluationPoints_,
                                                    GPUBuffer& evaluationStatistics_,
                                                    float radiusClamp_,
                                                    float kernelRegularization_,
                                                    uint32_t nSamplesSplatted_,
                                                    uint32_t nEvaluationPoints_):
samplePoints(samplePoints_),
evaluationPoints(evaluationPoints_),
evaluationStatistics(evaluationStatistics_),
radiusClamp(radiusClamp_),
kernelRegularization(kernelRegularization_),
nSamplesSplatted(nSamplesSplatted_),
nEvaluationPoints(nEvaluationPoints_)
{
    // do nothing
}

inline void GPUBVCSplatSourceData::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor.getPath("samplePoints").setBinding(samplePoints.buffer);
    cursor.getPath("evaluationPoints").setBinding(evaluationPoints.buffer);
    cursor.getPath("evaluationStatistics").setBinding(evaluationStatistics.buffer);
    cursor.getPath("radiusClamp").setData(radiusClamp);
    cursor.getPath("kernelRegularization").setData(kernelRegularization);
    cursor.getPath("nSamplesSplatted").setData(nSamplesSplatted);
    cursor.getPath("sampleIndex").setData(0);
    cursor.getPath("nEvaluationPoints").setData(nEvaluationPoints);
    if (printLogs) printReflectionInfo(cursor, 9, "splatSourceData");
}

inline void GPUBVCSplatSourceData::setDispatchResources(const ShaderCursor& cursor, uint32_t dispatchIndex) const
{
    cursor.getPath("sampleIndex").setData(dispatchIndex);
}

} // wosx
