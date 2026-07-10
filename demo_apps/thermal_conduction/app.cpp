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

// This demo solves a thermal conduction problem on the surface of a Mars rover mesh
// using WoSX. It loads the rover geometry, UVs, and grayscale textures that define
// spatially varying Robin boundary values and coefficients, then traces camera rays
// to select visible surface sample points. The GPU walk on stars solver estimates
// the solution at those samples and writes the extracted values as images.
// When 'visualizeSetup' is enabled in the config, Polyscope is used to inspect
// the problem setup: geometry, robin boundary data, camera views, and sample points.

#include <wosx/wosx_gpu.h>
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "config.h"
#include "image.h"
#include "colormap.h"
#include <filesystem>
#include <fstream>
#include <limits>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility structs and functions

struct MeshData {
    // members
    std::vector<wosx::Vector3> positions;
    std::vector<wosx::Vector2> uvs;
    std::vector<wosx::Vector3i> indices;
    std::vector<wosx::Vector3i> uvIndices;

    std::pair<wosx::Vector3, wosx::Vector3> boundingBox;
    std::vector<wosx::Vector3> boxPositions;
    std::vector<wosx::Vector3i> boxIndices;

    Image<1> robinValue;
    Image<1> robinCoefficient;

    // helper functions
    std::vector<wosx::Vector2> getCornerUVs() const;
    void computeRobinCoefficientBounds(std::vector<float>& minRobinCoeffValues,
                                       std::vector<float>& maxRobinCoeffValues) const;
};

std::vector<wosx::Vector2> MeshData::getCornerUVs() const
{
    std::vector<wosx::Vector2> cornerUVs;
    cornerUVs.reserve(3*uvIndices.size());

    for (const wosx::Vector3i& faceUVIndices: uvIndices) {
        for (int c = 0; c < 3; c++) {
            int uvIndex = faceUVIndices[c];
            if (uvIndex < 0 || uvIndex >= (int)uvs.size()) {
                std::cerr << "Invalid mesh UV index: " << uvIndex << std::endl;
                exit(EXIT_FAILURE);
            }

            cornerUVs.emplace_back(uvs[uvIndex]);
        }
    }

    return cornerUVs;
}

void MeshData::computeRobinCoefficientBounds(std::vector<float>& minRobinCoeffValues,
                                             std::vector<float>& maxRobinCoeffValues) const
{
    minRobinCoeffValues.resize(uvIndices.size());
    maxRobinCoeffValues.resize(uvIndices.size());

    for (int t = 0; t < (int)uvIndices.size(); t++) {
        float minValue = std::numeric_limits<float>::infinity();
        float maxValue = -std::numeric_limits<float>::infinity();

        for (int c = 0; c < 3; c++) {
            int uvIndex = uvIndices[t][c];
            if (uvIndex < 0 || uvIndex >= (int)uvs.size()) {
                std::cerr << "Invalid mesh UV index: " << uvIndex << std::endl;
                exit(EXIT_FAILURE);
            }

            float value = robinCoefficient.get(uvs[uvIndex], false)[0];
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }

        minRobinCoeffValues[t] = minValue;
        maxRobinCoeffValues[t] = maxValue;
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

void saveSolutionImages(const std::string& directoryPath,
                        const json& problemConfig,
                        const json& outputConfig,
                        const std::vector<int>& indexMap,
                        const std::vector<float>& solution)
{
    const int cameraWidth = getOptional<int>(problemConfig, "cameraWidth", 512);
    const int cameraHeight = getOptional<int>(problemConfig, "cameraHeight", 512);
    const std::string solutionFile = getFilename(directoryPath, outputConfig, "solutionFile",
                                                 false, "solutions/solution.png");
    const bool saveColormapped = getOptional<bool>(outputConfig, "saveColormapped", true);
    const std::string colormap = getOptional<std::string>(outputConfig, "colormap", "");
    const float colormapMinVal = getOptional<float>(outputConfig, "colormapMinVal", 0.0f);
    const float colormapMaxVal = getOptional<float>(outputConfig, "colormapMaxVal", 1.0f);

    // fill the solution image
    std::shared_ptr<Image<1>> solutionImage = std::make_shared<Image<1>>(cameraHeight, cameraWidth);
    for (int i = 0; i < (int)solution.size(); i++) {
        int index = indexMap[i];
        int row = index/cameraWidth;
        int col = index%cameraWidth;
        solutionImage->get(row, col)[0] = solution[i];
    }

    // write solution to disk
    std::filesystem::path path(solutionFile);
    std::filesystem::create_directories(path.parent_path());
    solutionImage->write(solutionFile);

    if (saveColormapped) {
        std::string basePath = (path.parent_path() / path.stem()).string();
        std::string ext = path.extension().string();
        getColormappedImage(solutionImage, colormap, colormapMinVal, colormapMaxVal)->write(basePath + "_color" + ext);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Problem specification - load mesh data, generate rays from a camera view, and
// setup the GPU geometric queries and PDE objects

void loadMeshData(const std::string& directoryPath,
                  const json& problemConfig,
                  MeshData& meshData)
{
    // load the mesh
    std::string geometryFilename = getFilename(directoryPath, problemConfig, "geometry");
    wosx::loadTexturedBoundaryMesh<3>(geometryFilename, meshData.positions, meshData.uvs,
                                      meshData.indices, meshData.uvIndices);
    wosx::normalize<3>(meshData.positions); // normalize to a unit sphere

    // compute a bounding box for the mesh
    meshData.boundingBox = wosx::computeBoundingBox<3>(meshData.positions, true, 1.25f);
    wosx::buildBoundingBoxMesh<3>(meshData.boundingBox.first, meshData.boundingBox.second,
                                  meshData.boxPositions, meshData.boxIndices);

    // load the robin textures
    std::string robinValueFilename = getFilename(directoryPath, problemConfig,
                                                 "robinValueTexture");
    std::string robinCoefficientFilename = getFilename(directoryPath, problemConfig,
                                                       "robinCoefficientTexture");
    meshData.robinValue = Image<1>(robinValueFilename);
    meshData.robinCoefficient = Image<1>(robinCoefficientFilename);

    // invert the robin coefficient texture
    for (int i = 0; i < meshData.robinCoefficient.h; i++) {
        for (int j = 0; j < meshData.robinCoefficient.w; j++) {
            meshData.robinCoefficient.get(i, j)[0] = 1.0f - meshData.robinCoefficient.get(i, j)[0];
        }
    }
}

void generateRaysFromCameraView(const glm::mat4& viewMat, float fov,
                                int cameraWidth, int cameraHeight,
                                std::vector<wosx::Vector3>& origins,
                                std::vector<wosx::Vector3>& directions)
{
    // setup the camera parameters and generate camera rays
    float aspectRatio = (float)cameraWidth/(float)cameraHeight;
    polyscope::CameraParameters cameraParameters(
        polyscope::CameraIntrinsics::fromFoVDegVerticalAndAspect(fov, aspectRatio),
        polyscope::CameraExtrinsics::fromMatrix(viewMat));

    glm::vec3 cameraPosition = cameraParameters.getPosition();
    std::vector<glm::vec3> cameraRayDirections = cameraParameters.generateCameraRays(
        cameraWidth, cameraHeight, polyscope::ImageOrigin::UpperLeft);

    // populate the ray origins and directions
    origins.clear();
    wosx::Vector3 origin(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    origins.resize(cameraRayDirections.size(), origin);

    directions.clear();
    for (const glm::vec3& direction: cameraRayDirections) {
        directions.emplace_back(direction.x, direction.y, direction.z);
    }
}

void generateRaysFromCameraView(const std::string& directoryPath,
                                const json& problemConfig,
                                std::vector<wosx::Vector3>& origins,
                                std::vector<wosx::Vector3>& directions)
{
    // parse the camera view from file
    const std::string cameraViewFilename = getFilename(directoryPath, problemConfig,
                                                       "cameraView", false,
                                                       "data/camera_view_0.json");
    std::ifstream cameraViewFile(cameraViewFilename);
    if (!cameraViewFile.is_open()) {
        std::cerr << "Error opening camera view file for reading: "
                  << cameraViewFilename << std::endl;
        exit(EXIT_FAILURE);
    }

    json cameraView = json::parse(cameraViewFile);
    if (!cameraView.contains("fov") || !cameraView.contains("viewMat")) {
        std::cerr << "Camera view file must contain fov and viewMat entries: "
                  << cameraViewFilename << std::endl;
        exit(EXIT_FAILURE);
    }

    // get the camera view matrix, fov and image resolution
    glm::mat4x4 viewMat(1.0f);
    auto matData = cameraView["viewMat"];
    auto it = matData.begin();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            viewMat[j][i] = *it;
            it++;
        }
    }

    float fov = cameraView["fov"];
    int cameraWidth = getOptional<int>(problemConfig, "cameraWidth", 512);
    int cameraHeight = getOptional<int>(problemConfig, "cameraHeight", 512);

    // generate rays from the camera view
    generateRaysFromCameraView(viewMat, fov, cameraWidth, cameraHeight,
                               origins, directions);
}

std::shared_ptr<wosx::GPUGeometricQueries<3>> setupGeometricQueries(const MeshData& meshData)
{
    // setup an absorbing boundary handler for the mesh's bounding box
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<3>>(
            meshData.boxPositions, meshData.boxIndices);

    // setup a reflecting boundary handler for the mesh
    std::function<bool(float, int)> ignoreCandidateSilhouette =
        wosx::getIgnoreCandidateSilhouetteCallback(true);
    std::vector<float> minRobinCoeffValues, maxRobinCoeffValues;
    meshData.computeRobinCoefficientBounds(minRobinCoeffValues, maxRobinCoeffValues);
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflectingBoundaryHandler =
        std::make_shared<wosx::GPUFcpwRobinBoundaryHandler<3>>(
            meshData.positions, meshData.indices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);

    // create a geometric queries object from the handlers
    return std::make_shared<wosx::GPUGeometricQueries<3>>(
        absorbingBoundaryHandler, reflectingBoundaryHandler,
        meshData.boundingBox.first, meshData.boundingBox.second, false);
}

class GPUThermalConductionPDE: public wosx::GPUPDE {
public:
    // constructor
    GPUThermalConductionPDE(std::shared_ptr<wosx::GPUReflectingBoundaryHandler> accelerationStructure_,
                            const MeshData& meshData_);

    // allocates and sets GPU resources, and returns type info
    void allocate(wosx::GPUContext& context);
    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    wosx::GPUPDEType getType() const;

private:
    // members
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> accelerationStructure;
    const MeshData& meshData;
    wosx::GPUBuffer cornerUVs = {};
    std::shared_ptr<wosx::GPUSampler> sampler;
    std::shared_ptr<wosx::GPUDenseGrid<float, 1, 2>> robinValue;
    std::shared_ptr<wosx::GPUDenseGrid<float, 1, 2>> robinCoefficient;
};

GPUThermalConductionPDE::GPUThermalConductionPDE(std::shared_ptr<wosx::GPUReflectingBoundaryHandler> accelerationStructure_,
                                                 const MeshData& meshData_):
accelerationStructure(accelerationStructure_),
meshData(meshData_),
sampler(std::make_shared<wosx::GPUSampler>()),
robinValue(nullptr),
robinCoefficient(nullptr)
{
    wosx::Vector2 uvMin = wosx::Vector2::Zero();
    wosx::Vector2 uvMax = wosx::Vector2::Ones();
    wosx::Vector2i robinValueGridShape = wosx::Vector2i(meshData.robinValue.w,
                                                        meshData.robinValue.h);
    robinValue = std::make_unique<wosx::GPUDenseGrid<float, 1, 2>>(
        sampler, meshData.robinValue.toEigen(true, false),
        robinValueGridShape, uvMin, uvMax, false);
    wosx::Vector2i robinCoefficientGridShape = wosx::Vector2i(meshData.robinCoefficient.w,
                                                              meshData.robinCoefficient.h);
    robinCoefficient = std::make_unique<wosx::GPUDenseGrid<float, 1, 2>>(
        sampler, meshData.robinCoefficient.toEigen(true, false),
        robinCoefficientGridShape, uvMin, uvMax, false);
}

void GPUThermalConductionPDE::allocate(wosx::GPUContext& context)
{
    cornerUVs.allocate<wosx::Vector2>(context, false, meshData.getCornerUVs());
    sampler->allocate(context, wosx::TextureFilteringMode::Linear,
                      wosx::TextureAddressingMode::ClampToEdge);
    robinValue->allocate(context);
    robinCoefficient->allocate(context);
}

void GPUThermalConductionPDE::setResources(const wosx::ShaderCursor& cursor, bool printLogs) const
{
    accelerationStructure->setResources(cursor["mAccelerationStructure"], printLogs);
    cursor["mCornerUVs"].setBinding(cornerUVs.buffer);
    robinValue->setResources(cursor["mRobinValue"], printLogs);
    robinCoefficient->setResources(cursor["mRobinCoefficient"], printLogs);
    if (printLogs) wosx::printReflectionInfo(cursor, 4, getReflectionType());
}

std::string GPUThermalConductionPDE::getReflectionType() const
{
    return "ThermalConductionPDE";
}

wosx::GPUPDEType GPUThermalConductionPDE::getType() const
{
    return wosx::GPUPDEType::Poisson;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Walk on stars solver - create sample points, run the solver and extract the solution

void createSamplePoints(const std::vector<wosx::Vector3>& origins,
                        const std::vector<wosx::Vector3>& directions,
                        std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                        std::vector<wosx::GPUSamplePoint<3>>& samplePts,
                        std::vector<int>& sampleIndexMap)
{
    // run intersection queries
    std::vector<float> distAlongRay;
    std::vector<wosx::Vector3> intersectionPoints;
    std::vector<wosx::Vector3> intersectionNormals;
    wosx::GPUDistanceQueries<float, 3> distanceQueries(taskHandle);
    distanceQueries.intersectReflectingBoundary(origins, directions, distAlongRay,
                                                intersectionPoints, intersectionNormals);

    // create the sample points
    samplePts.clear();
    sampleIndexMap.clear();
    for (int i = 0; i < (int)origins.size(); i++) {
        if (distAlongRay[i] >= 0.0f) {
            wosx::GPUSamplePoint<3> samplePt;
            samplePt.pt = wosx::float3{intersectionPoints[i][0],
                                       intersectionPoints[i][1],
                                       intersectionPoints[i][2]};
            samplePt.normal = wosx::float3{intersectionNormals[i][0],
                                           intersectionNormals[i][1],
                                           intersectionNormals[i][2]};
            samplePt.type = wosx::SampleType::OnReflectingBoundary;
            samplePt.estimationQuantity = wosx::EstimationQuantity::Solution;
            samplePts.emplace_back(samplePt);
            sampleIndexMap.emplace_back(i);
        }
    }
}

void runSolver(const json& solverConfig,
               std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
               const std::vector<wosx::GPUSamplePoint<3>>& samplePts,
               std::vector<wosx::GPUSampleStatistics<float, 3>>& sampleStatistics)
{
    // load config settings for the walk on stars solver
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.99f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 10000);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int nResidentThreads = getOptional<int>(solverConfig, "nResidentThreads", 131072);

    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    bool disablePersistentThreads = getOptional<bool>(solverConfig, "disablePersistentThreads", false);
    if (!disablePersistentThreads && taskHandle->getDeviceBackend() != "cuda") {
        std::cerr << "Persistent threads require CUDA backend, disabling" << std::endl;
        disablePersistentThreads = true;
    }

    // setup walk settings for the solver
    std::shared_ptr<wosx::GPUWalkSettings> walkSettings = std::make_shared<wosx::GPUWalkSettings>();
    walkSettings->epsilonShellForAbsorbingBoundary = epsilonShellForAbsorbingBoundary;
    walkSettings->epsilonShellForReflectingBoundary = epsilonShellForReflectingBoundary;
    walkSettings->silhouettePrecision = silhouettePrecision;
    walkSettings->russianRouletteThreshold = russianRouletteThreshold;
    walkSettings->maxWalkLength = maxWalkLength;
    walkSettings->stepsBeforeUsingMaximalSpheres = stepsBeforeUsingMaximalSpheres;
    walkSettings->solveDoubleSided = 1;
    walkSettings->ignoreAbsorbingBoundaryContribution = 0;
    walkSettings->ignoreReflectingBoundaryContribution = 0;
    walkSettings->ignoreSourceContribution = 1;

    // initialize the solver
    wosx::GPUWalkOnStarsSolver<float, 3> walkOnStars(taskHandle, walkSettings, nResidentThreads,
                                                     !disablePersistentThreads, printLogs);

    // run the solver
    if (disablePersistentThreads) {
        // compute workload parameters--the number of sample copies and number of
        // walks per copy--to help improve GPU occupancy when running the solver
        std::pair<uint32_t, uint32_t> workloadParameters =
            walkOnStars.computeWorkloadParameters(samplePts.size(), nWalks);

        // populate sample points on the GPU
        walkOnStars.populateSamplePoints(samplePts, workloadParameters.first);

        // solve
        walkOnStars.solve(1, workloadParameters.second);

    } else {
        // populate sample points on the GPU
        walkOnStars.populateSamplePoints(samplePts);

        // solve
        walkOnStars.solve(nWalks);
    }

    // extract sample statistics from the GPU
    walkOnStars.getSampleStatistics(sampleStatistics);
}

void getSolution(const std::vector<wosx::GPUSampleStatistics<float, 3>>& sampleStatistics,
                 std::vector<float>& solution)
{
    solution.resize(sampleStatistics.size(), 0.0f);
    for (int i = 0; i < (int)sampleStatistics.size(); i++) {
        solution[i] = sampleStatistics[i].getEstimatedSolution();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Visualization - read and write camera views, plot sample points and set the GUI callback

void setCameraView(const std::string& cameraViewFilename)
{
    std::ifstream cameraViewFile(cameraViewFilename);
    if (!cameraViewFile.is_open()) {
        std::cerr << "Error opening camera view file for reading: "
                  << cameraViewFilename << std::endl;
        return;
    }

    std::string jsonData;
    std::string line;
    while (std::getline(cameraViewFile, line)) {
        jsonData += line + "\n";
    }

    if (!cameraViewFile.eof()) {
        std::cerr << "Error reading camera view file: "
                  << cameraViewFilename << std::endl;
        return;
    }

    polyscope::view::setViewFromJson(jsonData, false);
}

void writeCameraView(const std::string& cameraViewFilename)
{
    std::ofstream cameraViewFile(cameraViewFilename);
    if (!cameraViewFile.is_open()) {
        std::cerr << "Error opening camera view file for writing: "
                  << cameraViewFilename << std::endl;
        return;
    }

    cameraViewFile << polyscope::view::getViewAsJson() << std::endl;
    if (!cameraViewFile.good()) {
        std::cerr << "Error writing camera view file: "
                  << cameraViewFilename << std::endl;
        return;
    }

    std::cout << "Wrote camera view to: " << cameraViewFilename << std::endl;
}

void plotSamplePoints(const std::vector<wosx::GPUSamplePoint<3>>& samplePts,
                      bool enabled=false)
{
    std::vector<wosx::Vector3> points, normals;
    for (const auto& samplePoint: samplePts) {
        points.emplace_back(wosx::Vector3(samplePoint.pt.x,
                                          samplePoint.pt.y,
                                          samplePoint.pt.z));
        normals.emplace_back(wosx::Vector3(samplePoint.normal.x,
                                           samplePoint.normal.y,
                                           samplePoint.normal.z));
    }

    auto pointCloud = polyscope::registerPointCloud("Sample Points", points);
    pointCloud->addVectorQuantity("Normals", normals);
    pointCloud->setEnabled(enabled);
}

void guiCallback(const std::string& directoryPath,
                 const json& problemConfig,
                 const json& outputConfig,
                 std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                 std::vector<wosx::GPUSamplePoint<3>>& samplePts)
{
    if (ImGui::Button("Generate sample points")) {
        // generate rays from a camera view
        std::vector<wosx::Vector3> origins;
        std::vector<wosx::Vector3> directions;
        int cameraWidth = getOptional<int>(problemConfig, "cameraWidth", 512);
        int cameraHeight = getOptional<int>(problemConfig, "cameraHeight", 512);
        generateRaysFromCameraView(polyscope::view::viewMat, polyscope::view::fov,
                                   cameraWidth, cameraHeight, origins, directions);

        // create and plot sample points
        std::vector<int> sampleIndexMap;
        createSamplePoints(origins, directions, taskHandle, samplePts, sampleIndexMap);
        plotSamplePoints(samplePts, true);
    }

    if (ImGui::Button("Write camera view")) {
        const std::string cameraViewFilename = getFilename(directoryPath, outputConfig,
                                                           "cameraView", false,
                                                           "data/camera_view_out.json");
        writeCameraView(cameraViewFilename);
    }
}

void visualizeSetup(const std::string& directoryPath,
                    const json& problemConfig,
                    const json& outputConfig,
                    const MeshData& meshData,
                    std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle,
                    std::vector<wosx::GPUSamplePoint<3>>& samplePts)
{
    // set a few options
    polyscope::options::programName = "thermal conduction demo - problem setup";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneEnabled = false;

    // initialize polyscope
    polyscope::init();

    // set the camera view from file
    const std::string cameraViewFilename = getFilename(directoryPath, problemConfig,
                                                       "cameraView", false,
                                                       "data/camera_view_0.json");
    setCameraView(cameraViewFilename);

    // register the mesh and its robin coefficients and values
    auto mesh = polyscope::registerSurfaceMesh("Mesh", meshData.positions, meshData.indices);

    auto uvParameterization = mesh->addParameterizationQuantity("UVs", meshData.getCornerUVs());
    auto robinValueQuantity = mesh->addTextureScalarQuantity("Robin Value", *uvParameterization,
                                                             meshData.robinValue.w, meshData.robinValue.h,
                                                             meshData.robinValue.toEigen(false, false),
                                                             polyscope::ImageOrigin::LowerLeft);
    robinValueQuantity->setColorMap("plasma");
    robinValueQuantity->setEnabled(true);

    auto robinCoefficientQuantity = mesh->addTextureScalarQuantity("Robin Coefficient", *uvParameterization,
                                                                   meshData.robinCoefficient.w,
                                                                   meshData.robinCoefficient.h,
                                                                   meshData.robinCoefficient.toEigen(false, false),
                                                                   polyscope::ImageOrigin::LowerLeft);
    robinCoefficientQuantity->setColorMap("viridis");
    robinCoefficientQuantity->setEnabled(false);

    std::vector<float> minRobinCoeffValues;
    std::vector<float> maxRobinCoeffValues;
    meshData.computeRobinCoefficientBounds(minRobinCoeffValues, maxRobinCoeffValues);

    auto minCoefficientQuantity = mesh->addFaceScalarQuantity("Min Robin Coefficient", minRobinCoeffValues);
    minCoefficientQuantity->setColorMap("viridis");
    minCoefficientQuantity->setEnabled(false);

    auto maxCoefficientQuantity = mesh->addFaceScalarQuantity("Max Robin Coefficient", maxRobinCoeffValues);
    maxCoefficientQuantity->setColorMap("viridis");
    maxCoefficientQuantity->setEnabled(false);

    // plot the sample points
    plotSamplePoints(samplePts);

    // bind the gui callback
    polyscope::state::userCallback = std::bind(&guiCallback, directoryPath,
                                               problemConfig, outputConfig,
                                               taskHandle, std::ref(samplePts));

    // give control to polyscope gui
    polyscope::show();
    polyscope::state::userCallback = nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Demo entry point - load config, setup the problem, run the solver, save the solution,
// or optionally visualize the problem setup

void runDemo(const json& config)
{
    // load config settings
    std::filesystem::path wosxDirectoryPath = std::filesystem::current_path().parent_path();
    std::filesystem::path demoDirectoryPath = wosxDirectoryPath / "demo_apps" / "thermal_conduction";
    std::string wosxDirectoryPathStr = wosxDirectoryPath.string();
    std::string demoDirectoryPathStr = demoDirectoryPath.string();
    const json problemConfig = getRequired<json>(config, "problem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");

    // load mesh data
    MeshData meshData;
    loadMeshData(demoDirectoryPathStr, problemConfig, meshData);

    // setup a geometric queries object for the mesh
    std::shared_ptr<wosx::GPUGeometricQueries<3>> geometricQueries =
        setupGeometricQueries(meshData);

    // setup the demo PDE
    std::shared_ptr<GPUThermalConductionPDE> pde = std::make_shared<GPUThermalConductionPDE>(
        geometricQueries->getReflectingBoundaryHandler(), meshData);

    // setup a task handle to run the demo on the GPU
    const std::string deviceBackend = getOptional<std::string>(config, "deviceBackend", "cuda");
    std::shared_ptr<wosx::GPUTaskHandle<float, 3>> taskHandle =
        std::make_shared<wosx::GPUTaskHandle<float, 3>>(wosxDirectoryPathStr,
                                                        demoDirectoryPathStr,
                                                        deviceBackend);
    taskHandle->setGeometricQueries(geometricQueries);
    taskHandle->setPDE(pde);
    taskHandle->init();

    // generate rays from a camera view
    std::vector<wosx::Vector3> origins;
    std::vector<wosx::Vector3> directions;
    generateRaysFromCameraView(demoDirectoryPathStr, problemConfig, origins, directions);

    // create sample points to run the solver on
    std::vector<wosx::GPUSamplePoint<3>> samplePts;
    std::vector<int> sampleIndexMap;
    createSamplePoints(origins, directions, taskHandle, samplePts, sampleIndexMap);

    const bool shouldVisualizeSetup = getOptional<bool>(outputConfig, "visualizeSetup", false);
    if (shouldVisualizeSetup) {
        // visualize the problem setup
        visualizeSetup(demoDirectoryPathStr, problemConfig, outputConfig,
                       meshData, taskHandle, samplePts);

    } else {
        // run the solver
        std::vector<wosx::GPUSampleStatistics<float, 3>> sampleStatistics;
        runSolver(solverConfig, taskHandle, samplePts, sampleStatistics);

        // extract the solution and save it as images
        std::vector<float> solution;
        getSolution(sampleStatistics, solution);
        saveSolutionImages(demoDirectoryPathStr, problemConfig, outputConfig,
                           sampleIndexMap, solution);
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
