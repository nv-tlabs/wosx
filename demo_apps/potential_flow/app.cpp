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

// This demo solves an exterior potential-flow problem around a closed 3D body
// using WoSX's GPU boundary value caching solver. A uniform freestream velocity
// is prescribed at infinity, and the body surface enforces a no-through-flow
// Neumann condition for the perturbation potential. The exterior domain is
// handled with a Kelvin transform, which maps the unbounded problem to a bounded
// inverted domain. When 'visualizeSetup' is enabled, the demo opens Polyscope to
// inspect the input geometry, active slice plane cells, and freestream direction.
// Otherwise, it estimates the perturbation potential and flow velocity on a
// slice plane and saves the results to file via Polyscope's screenshot tool.

#include <wosx/wosx_gpu.h>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "config.h"
#include <filesystem>
#include <fstream>
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility structs and functions

struct MeshData {
    // members
    std::vector<wosx::Vector3> positions;
    std::vector<wosx::Vector3i> indices;

    std::vector<wosx::Vector3> invertedPositions;
    std::vector<float> invertedMinRobinCoeffValues;
    std::vector<float> invertedMaxRobinCoeffValues;

    std::pair<wosx::Vector3, wosx::Vector3> boundingBox;
    std::pair<wosx::Vector3, wosx::Vector3> invertedBoundingBox;

    // helper functions
    std::function<bool(const wosx::Vector<3>&)> getInsideBoundingBoxCallback() const;
    std::function<bool(const wosx::Vector<3>&)> getInsideInvertedBoundingBoxCallback() const;
};

std::function<bool(const wosx::Vector<3>&)> MeshData::getInsideBoundingBoxCallback() const
{
    return [this](const wosx::Vector<3>& x) -> bool {
        return (x.array() >= boundingBox.first.array()).all() &&
               (x.array() <= boundingBox.second.array()).all();
    };
}

std::function<bool(const wosx::Vector<3>&)> MeshData::getInsideInvertedBoundingBoxCallback() const
{
    return [this](const wosx::Vector<3>& x) -> bool {
        return (x.array() >= invertedBoundingBox.first.array()).all() &&
               (x.array() <= invertedBoundingBox.second.array()).all();
    };
}

struct SlicePlaneData {
    // members
    std::vector<wosx::Vector3> positions;
    std::vector<wosx::Vector3> cellCenters;
    std::vector<std::vector<size_t>> indices;
    std::vector<bool> activeCells;

    // builds slice plane mesh
    void build(const json& problemConfig,
               const std::pair<wosx::Vector3, wosx::Vector3>& boundingBox);

    // deactivates interior cells of the slice plane
    void deactivateInteriorCells(std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle);

private:
    // computes cell centers from positions and indices
    void computeCellCenters();
};

void SlicePlaneData::computeCellCenters()
{
    cellCenters.resize(indices.size());
    for (int i = 0; i < (int)indices.size(); i++) {
        const std::vector<size_t>& index = indices[i];
        wosx::Vector3 center = wosx::Vector3::Zero();
        int nIndex = (int)index.size();
        for (int j = 0; j < nIndex; j++) {
            center += positions[index[j]];
        }

        center = center/nIndex;
        cellCenters[i] = center;
    }
}

void SlicePlaneData::build(const json& problemConfig,
                           const std::pair<wosx::Vector3, wosx::Vector3>& boundingBox)
{
    // compute positions and indices
    indices.clear();
    positions.clear();
    const int resolutionPow2 = 6;
    int size = 1 << resolutionPow2;

    for (size_t h = 0; h < size + 1; h++) {
        for (size_t w = 0; w < size + 1; w++) {
            size_t index = positions.size();
            positions.emplace_back(wosx::Vector3::Zero());
            positions[index](0) = w;
            positions[index](1) = h;

            if (h != size && w != size) {
                indices.emplace_back(std::vector<size_t>{index, index + 1,
                                                         index + size + 2,
                                                         index + size + 1});
            }
        }
    }

    // normalize, scale and shift positions
    wosx::Vector3 center = 0.5f*(boundingBox.first + boundingBox.second);
    wosx::Vector3 extent = boundingBox.second - boundingBox.first;
    center(2) = 0.0f;
    extent(2) = 0.0f;
    float scale = 0.5f*extent.norm();

    wosx::normalize<3>(positions);
    for (int i = 0; i < (int)positions.size(); i++) {
        positions[i] *= scale;
        positions[i] += center;
    }

    // compute cell centers
    computeCellCenters();

    // set active cells
    activeCells.assign(indices.size(), true);
    const float sliceCropMinY = getOptional<float>(problemConfig, "sliceCropMinY", -0.3f);
    const float sliceCropMaxY = getOptional<float>(problemConfig, "sliceCropMaxY", 0.65f);

    for (int i = 0; i < (int)cellCenters.size(); i++) {
        if (cellCenters[i][1] > sliceCropMaxY || cellCenters[i][1] < sliceCropMinY) {
            activeCells[i] = false;
        }
    }
}

void SlicePlaneData::deactivateInteriorCells(std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle)
{
    // compute signed distance to the boundary from cell centers
    std::vector<float> distToBoundary, stub;
    wosx::GPUDistanceQueries<float, 3> distanceQueries(taskHandle);
    distanceQueries.computeDistToBoundary(cellCenters, distToBoundary, stub);

    // deactivate interior cells using signed distance
    const float boundaryDistanceMargin = 0.035f;
    for (int i = 0; i < (int)cellCenters.size(); i++) {
        if (activeCells[i] && distToBoundary[i] <= boundaryDistanceMargin) {
            activeCells[i] = false;
        }
    }
}

std::string getFilename(const std::string& directoryPath,
                        const json& config,
                        const std::string& filename,
                        bool isRequired=true,
                        const std::string& defaultValue="")
{
    return directoryPath + "/" + (isRequired ? getRequired<std::string>(config, filename) :
                                               getOptional<std::string>(config, filename, defaultValue));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem specification - load mesh data, setup the GPU geometric queries and PDE objects

void loadMeshData(const std::string& directoryPath,
                  const json& problemConfig,
                  MeshData& meshData)
{
    // load the mesh
    std::string geometryFilename = getFilename(directoryPath, problemConfig, "geometry");
    wosx::loadBoundaryMesh<3>(geometryFilename, meshData.positions, meshData.indices);
    wosx::normalize<3>(meshData.positions); // normalize to a unit sphere

    // compute a bounding box for the mesh
    meshData.boundingBox = wosx::computeBoundingBox<3>(meshData.positions, true, 1.5f);
}

void invertMesh(const wosx::KelvinTransform<float, 3>& kelvinTransform,
                MeshData& meshData)
{
    // invert mesh positions using the Kelvin transform
    kelvinTransform.transformPoints(meshData.positions, meshData.invertedPositions);

    // compute the Robin coefficients for the inverted domain
    std::vector<float> minRobinCoeffValues(meshData.indices.size(), 0.0f);
    std::vector<float> maxRobinCoeffValues(meshData.indices.size(), 0.0f);
    kelvinTransform.computeRobinCoefficients(meshData.invertedPositions, meshData.indices,
                                             minRobinCoeffValues, maxRobinCoeffValues,
                                             meshData.invertedMinRobinCoeffValues,
                                             meshData.invertedMaxRobinCoeffValues);

    // compute a bounding box for the inverted mesh
    meshData.invertedBoundingBox = wosx::computeBoundingBox<3>(meshData.invertedPositions,
                                                               true, 1.0f);
}

std::shared_ptr<wosx::GPUGeometricQueries<3>> setupGeometricQueries(const MeshData& meshData,
                                                                    bool useInvertedMesh)
{
    // setup boundary handlers and bounding box based on the useInvertedMesh flag
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler = nullptr;
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflectingBoundaryHandler = nullptr;
    std::pair<wosx::Vector3, wosx::Vector3> boundingBox;

    if (useInvertedMesh) {
        // setup an empty absorbing boundary handler
        absorbingBoundaryHandler = std::make_shared<wosx::GPUEmptyAbsorbingBoundaryHandler>();

        // setup a robin reflecting boundary handler
        std::function<bool(float, int)> ignoreCandidateSilhouette =
            wosx::getIgnoreCandidateSilhouetteCallback(false);
        reflectingBoundaryHandler = std::make_shared<wosx::GPUFcpwRobinBoundaryHandler<3>>(
            meshData.invertedPositions, meshData.indices, ignoreCandidateSilhouette,
            meshData.invertedMinRobinCoeffValues, meshData.invertedMaxRobinCoeffValues);

        // set the inverted bounding box
        boundingBox = meshData.invertedBoundingBox;

    } else {
        // setup an absorbing boundary handler
        absorbingBoundaryHandler = std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
            meshData.positions, meshData.indices);

        // setup an empty reflecting boundary handler
        reflectingBoundaryHandler = std::make_shared<wosx::GPUEmptyReflectingBoundaryHandler>();

        // set the bounding box
        boundingBox = meshData.boundingBox;
    }

    // create a geometric queries object from the handlers
    return std::make_shared<wosx::GPUGeometricQueries<3>>(
        absorbingBoundaryHandler, reflectingBoundaryHandler,
        boundingBox.first, boundingBox.second, true);
}

struct FreestreamState {
    // members
    float speed;
    float angle;

    // computes the freestream velocity from the speed and angle
    wosx::Vector3 getVelocity() const;
};

wosx::Vector3 FreestreamState::getVelocity() const
{
    return wosx::Vector3{speed*std::cos(angle), speed*std::sin(angle), 0.0f};
}

class GPUPotentialFlowPDE: public wosx::GPUPDE {
public:
    // constructor
    GPUPotentialFlowPDE(const FreestreamState& freestreamState_);

    // allocates and sets GPU resources, and returns type info
    void allocate(wosx::GPUContext& context);
    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    wosx::GPUPDEType getType() const;

private:
    // members
    FreestreamState freestreamState;
};

GPUPotentialFlowPDE::GPUPotentialFlowPDE(const FreestreamState& freestreamState_):
freestreamState(freestreamState_)
{
    // nothing to do
}

void GPUPotentialFlowPDE::allocate(wosx::GPUContext& context)
{
    // nothing to do
}

void GPUPotentialFlowPDE::setResources(const wosx::ShaderCursor& cursor, bool printLogs) const
{
    cursor["mFreestreamSpeed"].setData(freestreamState.speed);
    cursor["mFreestreamAngle"].setData(freestreamState.angle);
    if (printLogs) wosx::printReflectionInfo(cursor, 2, getReflectionType());
}

std::string GPUPotentialFlowPDE::getReflectionType() const
{
    return "PotentialFlowPDE";
}

wosx::GPUPDEType GPUPotentialFlowPDE::getType() const
{
    return wosx::GPUPDEType::Poisson;
}

std::vector<float> computeBoundarySamplingWeights(const MeshData& meshData,
                                                  const FreestreamState& freestreamState)
{
    // set primitive weights to a mixture of two densities: one proportional to the
    // Kelvin-transformed Robin RHS, and one uniform with respect to the original surface
    // area (a weight of 1/|y|^4 per unit inverted area, since an inversion scales areas
    // by 1/|x|^4). The latter keeps the sample density from vanishing where the freestream
    // is tangent to the surface, as samples there carry a large 1/pdf during splatting.
    int nPrimitives = (int)meshData.indices.size();
    std::vector<float> rhsWeights(nPrimitives, 0.0f);
    std::vector<float> areaWeights(nPrimitives, 0.0f);
    wosx::Vector3 freestreamVelocity = freestreamState.getVelocity();
    float rhsSamplingMass = 0.0f;
    float areaSamplingMass = 0.0f;

    for (int i = 0; i < nPrimitives; i++) {
        const wosx::Vector3i& index = meshData.indices[i];
        const wosx::Vector3& p0 = meshData.invertedPositions[index[0]];
        const wosx::Vector3& p1 = meshData.invertedPositions[index[1]];
        const wosx::Vector3& p2 = meshData.invertedPositions[index[2]];
        wosx::Vector3 y = (p0 + p1 + p2)/3.0f;

        wosx::Vector3 N = wosx::computePrimitiveNormal<3>(meshData.invertedPositions,
                                                          meshData.indices, i);
        float r2 = y.squaredNorm();
        if (r2 <= std::numeric_limits<float>::epsilon()) continue;

        wosx::Vector3 n = N - 2.0f*N.dot(y)*y/r2;
        rhsWeights[i] = std::fabs(-n.dot(freestreamVelocity))/(r2*std::sqrt(r2));
        areaWeights[i] = 1.0f/(r2*r2);

        // the sampler builds its cdf from primitive weights scaled by primitive areas,
        // so accumulate the sampling mass of each density to combine them evenly
        float area = wosx::computeTriangleSurfaceArea(p0, p1, p2);
        rhsSamplingMass += area*rhsWeights[i];
        areaSamplingMass += area*areaWeights[i];
    }

    std::vector<float> primitiveWeights(nPrimitives, 0.0f);
    for (int i = 0; i < nPrimitives; i++) {
        primitiveWeights[i] = 0.75f*rhsWeights[i]/std::max(rhsSamplingMass, 1e-8f) +
                              0.25f*areaWeights[i]/std::max(areaSamplingMass, 1e-8f);
    }

    return primitiveWeights;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Boundary value caching solver - create evaluation points, run the solver and
// extract the solution and gradient

void createEvaluationPoints(const SlicePlaneData& slicePlaneData,
                            const wosx::KelvinTransform<float, 3>& kelvinTransform,
                            std::vector<wosx::GPUBVCEvaluationPoint<3>>& evalPts)
{
    // use inverted cell centers as solve locations
    std::vector<wosx::Vector3> solveLocations;
    kelvinTransform.transformPoints(slicePlaneData.cellCenters, solveLocations);

    // create the evaluation points
    evalPts.clear();
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        if (slicePlaneData.activeCells[i]) {
            // select solve locations that are exterior to the
            // input mesh and interior to the inverted mesh
            wosx::GPUBVCEvaluationPoint<3> evalPt;
            evalPt.pt = wosx::float3{solveLocations[i][0],
                                     solveLocations[i][1],
                                     solveLocations[i][2]};
            evalPt.normal = wosx::float3{0.0f, 0.0f, 0.0f};
            evalPt.type = wosx::SampleType::InDomain;
            evalPts.emplace_back(evalPt);
        }
    }
}

void runSolver(const json& solverConfig,
               std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
               const std::vector<wosx::GPUBVCEvaluationPoint<3>>& evalPts,
               std::vector<wosx::GPUBVCEvaluationOutputs<float, 3>>& evalOutputs)
{
    // load config settings for random walks
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.99f);

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 10000);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // load config settings for boundary value caching
    const int reflectingBoundaryCacheSize = getOptional<int>(solverConfig, "reflectingBoundaryCacheSize", 10000);
    const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // setup walk settings for the solver
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = epsilonShellForReflectingBoundary;
    walkSettings->silhouettePrecision = silhouettePrecision;
    walkSettings->russianRouletteThreshold = russianRouletteThreshold;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = 0;
    walkSettings->useGradientControlVariates = 1;
    walkSettings->useGradientAntitheticVariates = 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = 1;
    walkSettings->ignoreReflectingBoundaryContribution = 0; // solving an exterior Neumann problem
    walkSettings->ignoreSourceContribution = 1;

    // initialize the solver
    std::shared_ptr<wosx::GPUBoundaryValueCachingSolver<float, 3>> boundaryValueCaching =
        std::make_shared<wosx::GPUBoundaryValueCachingSolver<float, 3>>(
            taskHandle, walkSettings, nResidentThreads, !disablePersistentThreads, printLogs);

    // populate evaluation points on the GPU
    boundaryValueCaching->populateEvaluationPoints(evalPts);

    // generate samples on the reflecting boundary (we do not require
    // domain or absorbing boundary samples for this problem)
    const int absorbingBoundaryCacheSize = 0;
    const int domainCacheSize = 0;
    boundaryValueCaching->generateSamples(absorbingBoundaryCacheSize,
                                          reflectingBoundaryCacheSize,
                                          domainCacheSize);

    // compute sample estimates on the boundary (we do not require
    // gradient estimates as there is no absorbing boundary)
    const int nWalksForCachedGradientEstimates = 0;
    boundaryValueCaching->computeSampleEstimates(nWalksForCachedSolutionEstimates,
                                                 nWalksForCachedGradientEstimates);

    // splat boundary sample estimates to evaluation points
    boundaryValueCaching->splat(radiusClampForKernels, regularizationForKernels);

    // extract evaluation outputs from the GPU
    boundaryValueCaching->getEvaluationOutputs(evalOutputs);
}

void getSolutionAndGradient(const SlicePlaneData& slicePlaneData,
                            const wosx::KelvinTransform<float, 3>& kelvinTransform,
                            const std::vector<wosx::GPUBVCEvaluationPoint<3>>& evalPts,
                            const std::vector<wosx::GPUBVCEvaluationOutputs<float, 3>>& evalOutputs,
                            const wosx::Vector3& freestreamVelocity,
                            std::vector<float>& perturbationPotential,
                            std::vector<wosx::Vector3>& flowVelocity)
{
    int nCells = (int)slicePlaneData.cellCenters.size();
    perturbationPotential.assign(nCells, 0.0f);
    flowVelocity.assign(nCells, wosx::Vector3::Zero());
    int counter = 0;

    for (int i = 0; i < nCells; i++) {
        if (slicePlaneData.activeCells[i]) {
            // get the solve location
            wosx::Vector3 solveLocation = wosx::Vector3{evalPts[counter].pt.x,
                                                        evalPts[counter].pt.y,
                                                        evalPts[counter].pt.z};

            // compute the perturbation potential at the solve location
            float solution = evalOutputs[counter].getEstimatedSolution();
            perturbationPotential[i] = kelvinTransform.transformSolutionEstimate(solution, solveLocation);

            // compute the flow velocity at the solve location
            std::vector<float> gradient(3);
            for (int j = 0; j < 3; j++) {
                gradient[j] = evalOutputs[counter].getEstimatedGradient(j);
            }

            kelvinTransform.transformGradientEstimate(solution, gradient, solveLocation, gradient);
            flowVelocity[i] = freestreamVelocity + wosx::Vector3{gradient[0], gradient[1], gradient[2]};
            counter++;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem Visualization

void plotFreestreamVelocity(const SlicePlaneData& slicePlaneData,
                            const wosx::Vector3& velocity)
{
    int nCells = (int)slicePlaneData.cellCenters.size();
    std::vector<wosx::Vector3> freestreamVelocity(nCells, wosx::Vector3::Zero());
    for (int i = 0; i < nCells; i++) {
        if (slicePlaneData.activeCells[i]) {
            freestreamVelocity[i] = velocity;
        }
    }

    auto slicePlane = polyscope::getSurfaceMesh("Slice Plane");
    auto velocityQuantity = slicePlane->addFaceVectorQuantity("Freestream Velocity",
                                                              freestreamVelocity);
    velocityQuantity->setVectorLengthRange(10.0f);
    velocityQuantity->setEnabled(true);
}

void guiCallback(const SlicePlaneData& slicePlaneData,
                 FreestreamState& freestreamState)
{
    if (ImGui::SliderFloat("Freestream Speed", &freestreamState.speed, 1.0f, 10.0f)) {
        plotFreestreamVelocity(slicePlaneData, freestreamState.getVelocity());
    }

    if (ImGui::SliderFloat("Freestream Angle", &freestreamState.angle, 0.0f, 2.0f*M_PI)) {
        plotFreestreamVelocity(slicePlaneData, freestreamState.getVelocity());
    }
}

void setCameraView(const MeshData& meshData)
{
    const wosx::Vector3& pMin = meshData.boundingBox.first;
    const wosx::Vector3& pMax = meshData.boundingBox.second;
    wosx::Vector3 center = 0.5f*(pMin + pMax);
    float viewDistance = 1.25f*std::max(pMax[0] - pMin[0], pMax[1] - pMin[1]);

    glm::vec3 target(center[0], center[1], center[2]);
    glm::vec3 camera(center[0], center[1], center[2] + viewDistance);
    polyscope::view::lookAt(camera, target, glm::vec3(0.0f, 1.0f, 0.0f), false);
}

void visualizeSetup(const MeshData& meshData,
                    const SlicePlaneData& slicePlaneData,
                    FreestreamState& freestreamState)
{
    // set a few options
    polyscope::options::programName = "potential flow demo - problem setup";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneEnabled = false;

    // initialize polyscope
    polyscope::init();

    // set camera view
    setCameraView(meshData);

    // register the mesh
    polyscope::registerSurfaceMesh("Mesh", meshData.positions, meshData.indices);

    // register the slice plane
    auto slicePlane = polyscope::registerSurfaceMesh("Slice Plane",
                                                     slicePlaneData.positions,
                                                     slicePlaneData.indices);
    slicePlane->setSurfaceColor(glm::vec3{1.0f, 1.0f, 1.0f});
    slicePlane->setMaterial("flat");
    slicePlane->setBackFacePolicy(polyscope::BackFacePolicy::Cull);

    // plot freestream velocity on slice plane
    plotFreestreamVelocity(slicePlaneData, freestreamState.getVelocity());

    // bind the gui callback
    polyscope::state::userCallback = std::bind(&guiCallback,
                                               std::cref(slicePlaneData),
                                               std::ref(freestreamState));

    // give control to polyscope gui
    polyscope::show();
    polyscope::state::userCallback = nullptr;
}

void saveSolutionAndGradient(const std::string& directoryPath,
                             const json& outputConfig,
                             const MeshData& meshData,
                             const SlicePlaneData& slicePlaneData,
                             const std::vector<float>& perturbationPotential,
                             const std::vector<wosx::Vector3>& flowVelocity)
{
    // get the filename and screenshot dimensions for the flow visualization
    const std::string flowFile = getFilename(directoryPath, outputConfig, "flowFile",
                                             false, "solutions/flow.png");
    const int screenshotWidth = getOptional<int>(outputConfig, "screenshotWidth", 1024);
    const int screenshotHeight = getOptional<int>(outputConfig, "screenshotHeight", 1024);

    // set a few options
    polyscope::options::programName = "potential flow demo";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneEnabled = false;
    polyscope::options::ssaaFactor = 4;

    // initialize polyscope
    polyscope::init();

    // set window size and camera view
    polyscope::view::setWindowSize(screenshotWidth, screenshotHeight);
    setCameraView(meshData);

    // register the mesh
    polyscope::registerSurfaceMesh("Mesh", meshData.positions, meshData.indices);

    // register the slice plane
    auto slicePlane = polyscope::registerSurfaceMesh("Slice Plane",
                                                     slicePlaneData.positions,
                                                     slicePlaneData.indices);
    slicePlane->setSurfaceColor(glm::vec3{1.0f, 1.0f, 1.0f});
    slicePlane->setMaterial("flat");
    slicePlane->setBackFacePolicy(polyscope::BackFacePolicy::Cull);

    // plot perturbation potential on the slice plane
    const std::string perturbationPotentialColormap = getOptional<std::string>(
        outputConfig, "perturbationPotentialColormap", "coolwarm");
    const float perturbationPotentialColormapMinVal = getOptional<float>(
        outputConfig, "perturbationPotentialColormapMinVal", -1.0f);
    const float perturbationPotentialColormapMaxVal = getOptional<float>(
        outputConfig, "perturbationPotentialColormapMaxVal", 1.0f);

    auto potentialQuantity = slicePlane->addFaceScalarQuantity("Perturbation Potential",
                                                               perturbationPotential);
    potentialQuantity->setColorMap(perturbationPotentialColormap);
    potentialQuantity->setMapRange({perturbationPotentialColormapMinVal,
                                    perturbationPotentialColormapMaxVal});
    potentialQuantity->setEnabled(true);

    // plot flow velocity on the slice plane
    auto velocityQuantity = slicePlane->addFaceVectorQuantity("Flow Velocity", flowVelocity);
    velocityQuantity->setVectorLengthRange(10.0f);
    velocityQuantity->setVectorColor(glm::vec3{0.0f, 0.0f, 0.0f});
    velocityQuantity->setEnabled(true);

    // save a screenshot
    std::filesystem::path path(flowFile);
    std::filesystem::create_directories(path.parent_path());
    polyscope::ScreenshotOptions screenshotOptions;
    screenshotOptions.transparentBackground = false;
    screenshotOptions.includeUI = false;
    polyscope::screenshot(flowFile, screenshotOptions);

    // shutdown polyscope
    polyscope::shutdown();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Demo entry point - load config, setup the problem, run the solver, save the solution,
// or optionally visualize the problem

void runDemo(const json& config)
{
    // load config settings
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::filesystem::path demoDirectoryPath = wosxDirectoryPath / "demo_apps" / "potential_flow";
    std::string wosxDirectoryPathStr = wosxDirectoryPath.string();
    std::string demoDirectoryPathStr = demoDirectoryPath.string();
    const json problemConfig = getRequired<json>(config, "problem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");

    // setup a task handle to run the demo on the GPU
    const std::string deviceBackend = getOptional<std::string>(config, "deviceBackend", "cuda");
    std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<float, 3>>(wosxDirectoryPathStr,
                                                        demoDirectoryPathStr,
                                                        deviceBackend);

    // load mesh data
    MeshData meshData;
    loadMeshData(demoDirectoryPathStr, problemConfig, meshData);

    // setup a geometric queries object for the mesh and initialize
    // a task handle with it to determine the exterior of the mesh
    std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries =
        setupGeometricQueries(meshData, false);
    taskHandle->setGeometricQueries(geometricQueries);
    taskHandle->init();

    // build a slice plane mesh and deactivate cells interior to the mesh
    SlicePlaneData slicePlaneData;
    slicePlaneData.build(problemConfig, meshData.boundingBox);
    slicePlaneData.deactivateInteriorCells(taskHandle);

    // setup the freestream state
    FreestreamState freestreamState;
    freestreamState.speed = getOptional<float>(problemConfig, "freestreamSpeed", 5.0f);
    freestreamState.angle = getOptional<float>(problemConfig, "freestreamAngle", 0.0f);

    const bool shouldVisualizeSetup = getOptional<bool>(outputConfig, "visualizeSetup", false);
    if (shouldVisualizeSetup) {
        // visualize the problem setup
        visualizeSetup(meshData, slicePlaneData, freestreamState);

    } else {
        // invert the mesh about the origin using the Kelvin transform;
        // here we assume the origin is contained within the mesh
        const wosx::Vector3 inversionPoint = wosx::Vector3::Zero();
        wosx::KelvinTransform<float, 3> kelvinTransform(inversionPoint);
        invertMesh(kelvinTransform, meshData);

        // setup a geometric queries object for the inverted mesh
        std::shared_ptr<wosx::GPUGeometricQueries<3>> invertedGeometricQueries =
            setupGeometricQueries(meshData, true);

        // setup the demo PDE and the corresponding Kelvin transform PDE
        // for the inverted mesh
        std::shared_ptr<GPUPotentialFlowPDE> pde =
            std::make_shared<GPUPotentialFlowPDE>(freestreamState);
        std::shared_ptr<wosx::GPUKelvinPDE<float, 3>> kelvinPde =
            std::make_shared<wosx::GPUKelvinPDE<float, 3>>(pde, inversionPoint);

        // setup a reflecting boundary sampler (we do not require samplers
        // for the domain and/or absorbing boundary for this problem)
        std::function<bool(const wosx::Vector<3>&)> insideBoundingDomain =
            meshData.getInsideInvertedBoundingBoxCallback();
        std::vector<float> boundarySamplingWeights =
            computeBoundarySamplingWeights(meshData, freestreamState);
        std::shared_ptr<wosx::GPUBoundarySampler> reflectingBoundarySampler =
            wosx::createUniformBoundarySampler(invertedGeometricQueries, meshData.invertedPositions,
                                               meshData.indices, boundarySamplingWeights,
                                               insideBoundingDomain, 0.0f, false);

        // reinitialize the task handle with the inverted geometric queries,
        // reflecting boundary sampler and Kelvin PDE
        taskHandle->setGeometricQueries(invertedGeometricQueries);
        taskHandle->setReflectingBoundarySampler(reflectingBoundarySampler);
        taskHandle->setPDE(kelvinPde);
        taskHandle->init();

        // create evaluation points to run the solver on; here we use
        // evaluation points on the slice plane in the exterior of the
        // input mesh (equivalently, the interior of the inverted mesh)
        std::vector<wosx::GPUBVCEvaluationPoint<3>> evalPts;
        createEvaluationPoints(slicePlaneData, kelvinTransform, evalPts);

        // run the solver
        std::vector<wosx::GPUBVCEvaluationOutputs<float, 3>> evalOutputs;
        runSolver(solverConfig, taskHandle, evalPts, evalOutputs);

        // extract the solution and gradient
        std::vector<float> perturbationPotential;
        std::vector<wosx::Vector3> flowVelocity;
        getSolutionAndGradient(slicePlaneData, kelvinTransform, evalPts,
                               evalOutputs, freestreamState.getVelocity(),
                               perturbationPotential, flowVelocity);

        // save the estimated results
        saveSolutionAndGradient(demoDirectoryPathStr, outputConfig, meshData,
                                slicePlaneData, perturbationPotential, flowVelocity);
    }
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
    runDemo(config);

    return 0;
}
