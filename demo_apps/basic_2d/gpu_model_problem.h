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

// This file defines a GPUModelProblem class, which is used to describe a scalar- or
// vector-valued Poisson or screened Poisson PDE on a 2D domain via a boundary mesh,
// associated boundary conditions, source term, and robin and absorption coefficients.
//
// The boundary mesh is read from an OBJ file, while the input PDE data is read from
// images for the purposes of this demo. Compared to the CPU version of this demo, the
// PDE for this GPU version is defined in a separate wosx-pde-definition.slang file.
// The necessary resources needed for the PDE, e.g., the source and boundary texture data,
// are allocated on the GPU using the GPUModelProblemPDE class below.
//
// NOTE: Users may analogously define a GPUModelProblem class for 3D domains and/or
// vector-valued PDEs, as all functionality in WoSX is templated on the dimension
// and value type of the PDE.

#pragma once

#include <wosx/wosx_gpu.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

template <typename T>
class GPUModelProblemPDE: public wosx::GPUPDE {
public:
    static constexpr size_t CHANNELS = ValueTraits<T>::channels;
    static constexpr size_t IMAGE_CHANNELS = ValueTraits<T>::imageChannels;

    // constructor
    GPUModelProblemPDE(float absorptionCoeff_, float robinCoeff_,
                       const Vector2& gridMin_, const Vector2& gridMax_,
                       const Image<IMAGE_CHANNELS>& sourceValue_,
                       const Image<IMAGE_CHANNELS>& absorbingBoundaryValue_,
                       const Image<IMAGE_CHANNELS>& reflectingBoundaryValue_,
                       const Image<IMAGE_CHANNELS>& absorbingBoundaryNormalAlignedValue_,
                       const Image<IMAGE_CHANNELS>& reflectingBoundaryNormalAlignedValue_);

    // allocates and sets GPU resources, and returns type info
    void allocate(wosx::GPUContext& context);
    void setResources(const wosx::ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    wosx::GPUPDEType getType() const;

private:
    using TextureData = Eigen::Matrix<float, Eigen::Dynamic, CHANNELS>;

    // converts demo image storage to GPU PDE texture data
    TextureData getTextureData(const Image<IMAGE_CHANNELS>& image) const;

    // members
    float mAbsorptionCoeff;
    float mRobinCoeff;
    std::shared_ptr<wosx::GPUSampler> mSampler;
    std::unique_ptr<wosx::GPUDenseGrid<float, CHANNELS, 2>> mSourceValue;
    std::unique_ptr<wosx::GPUDenseGrid<float, CHANNELS, 2>> mAbsorbingBoundaryValue;
    std::unique_ptr<wosx::GPUDenseGrid<float, CHANNELS, 2>> mReflectingBoundaryValue;
    std::unique_ptr<wosx::GPUDenseGrid<float, CHANNELS, 2>> mAbsorbingBoundaryNormalAlignedValue;
    std::unique_ptr<wosx::GPUDenseGrid<float, CHANNELS, 2>> mReflectingBoundaryNormalAlignedValue;
};

template <typename T>
class GPUModelProblem {
public:
    static constexpr size_t CHANNELS = ValueTraits<T>::channels;
    static constexpr size_t IMAGE_CHANNELS = ValueTraits<T>::imageChannels;

    // constructor
    GPUModelProblem(const json& config, std::string directoryPath);

    // getters
    bool domainIsWatertight() { return mDomainIsWatertight; }
    bool solveDoubleSided() { return mSolveDoubleSided; }
    bool solveExterior() { return mSolveExterior; }
    const std::vector<Vector2i>& getBoundaryIndices() { return mIndices; }
    const std::vector<Vector2i>& getAbsorbingBoundaryIndices() { return mAbsorbingBoundaryIndices; }
    const std::vector<Vector2i>& getReflectingBoundaryIndices() { return mReflectingBoundaryIndices; }

    // getters for input domain
    const std::vector<Vector2>& getBoundaryPositions() { return mPositions; }
    const std::vector<Vector2>& getAbsorbingBoundaryPositions() { return mAbsorbingBoundaryPositions; }
    const std::vector<Vector2>& getReflectingBoundaryPositions() { return mReflectingBoundaryPositions; }
    const std::pair<Vector2, Vector2>& getBoundingBox() { return mBoundingBox; }
    std::shared_ptr<wosx::GPUPDE> getPDE() { return mPde; }
    std::shared_ptr<wosx::GPUGeometricQueries<2>> getGeometricQueries() { return mQueries; }

    // getters for inverted domain (needed for solving exterior problems)
    const wosx::KelvinTransform<T, 2>& getKelvinTransform() { return mKelvinTransform; }
    const std::vector<Vector2>& getInvertedAbsorbingBoundaryPositions() { return mInvertedAbsorbingBoundaryPositions; }
    const std::vector<Vector2>& getInvertedReflectingBoundaryPositions() { return mInvertedReflectingBoundaryPositions; }
    const std::pair<Vector2, Vector2>& getInvertedBoundingBox() { return mInvertedBoundingBox; }
    std::shared_ptr<wosx::GPUPDE> getPDEInvertedDomain() { return mPdeInvertedDomain; }
    std::shared_ptr<wosx::GPUGeometricQueries<2>> getGeometricQueriesInvertedDomain() { return mQueriesInvertedDomain; }

protected:
    // loads a boundary mesh from an OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // sets up the PDE
    void setupPDE();

    // partitions the boundary mesh into absorbing and reflecting parts
    void partitionBoundaryMesh();

    // sets up geometric queries for the absorbing and reflecting boundary
    void setupGeometricQueries(const std::vector<Vector2>& absorbingBoundaryPositions,
                               const std::vector<Vector2>& reflectingBoundaryPositions,
                               const std::pair<Vector2, Vector2>& boundingBox,
                               const std::vector<float>& minRobinCoeffValues,
                               const std::vector<float>& maxRobinCoeffValues,
                               bool areRobinConditionsPureNeumann,
                               std::shared_ptr<wosx::GPUGeometricQueries<2>>& queries);

    // applies a Kelvin transform to convert an exterior problem into an
    // equivalent interior problem with a modified PDE on the inverted domain
    void invertExteriorProblem();

    // members
    Image<IMAGE_CHANNELS> mSourceValue;
    Image<IMAGE_CHANNELS> mAbsorbingBoundaryValue;
    Image<IMAGE_CHANNELS> mReflectingBoundaryValue;
    Image<IMAGE_CHANNELS> mAbsorbingBoundaryNormalAlignedValue;
    Image<IMAGE_CHANNELS> mReflectingBoundaryNormalAlignedValue;
    Image<1> mIsReflectingBoundary;
    bool mSolveDoubleSided;
    bool mSolveExterior;
    bool mDomainIsWatertight;
    float mRobinCoeff;
    float mAbsorptionCoeff;

    std::vector<Vector2i> mIndices;
    std::vector<Vector2i> mAbsorbingBoundaryIndices;
    std::vector<Vector2i> mReflectingBoundaryIndices;

    std::vector<Vector2> mPositions;
    std::vector<Vector2> mAbsorbingBoundaryPositions;
    std::vector<Vector2> mReflectingBoundaryPositions;
    std::pair<Vector2, Vector2> mBoundingBox;
    std::shared_ptr<wosx::GPUPDE> mPde;
    std::vector<float> mMinRobinCoeffValues;
    std::vector<float> mMaxRobinCoeffValues;
    std::shared_ptr<wosx::GPUGeometricQueries<2>> mQueries;

    wosx::KelvinTransform<T, 2> mKelvinTransform;
    std::vector<Vector2> mInvertedAbsorbingBoundaryPositions;
    std::vector<Vector2> mInvertedReflectingBoundaryPositions;
    std::pair<Vector2, Vector2> mInvertedBoundingBox;
    std::shared_ptr<wosx::GPUPDE> mPdeInvertedDomain;
    std::vector<float> mMinRobinCoeffValuesInvertedDomain;
    std::vector<float> mMaxRobinCoeffValuesInvertedDomain;
    std::shared_ptr<wosx::GPUGeometricQueries<2>> mQueriesInvertedDomain;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T>
GPUModelProblemPDE<T>::GPUModelProblemPDE(float absorptionCoeff_, float robinCoeff_,
                                          const Vector2& gridMin_, const Vector2& gridMax_,
                                          const Image<IMAGE_CHANNELS>& sourceValue_,
                                          const Image<IMAGE_CHANNELS>& absorbingBoundaryValue_,
                                          const Image<IMAGE_CHANNELS>& reflectingBoundaryValue_,
                                          const Image<IMAGE_CHANNELS>& absorbingBoundaryNormalAlignedValue_,
                                          const Image<IMAGE_CHANNELS>& reflectingBoundaryNormalAlignedValue_):
mAbsorptionCoeff(absorptionCoeff_),
mRobinCoeff(robinCoeff_),
mSampler(std::make_shared<wosx::GPUSampler>()),
mSourceValue(nullptr),
mAbsorbingBoundaryValue(nullptr),
mReflectingBoundaryValue(nullptr),
mAbsorbingBoundaryNormalAlignedValue(nullptr),
mReflectingBoundaryNormalAlignedValue(nullptr)
{
    Vector2i sourceValueGridShape = Vector2i(sourceValue_.w, sourceValue_.h);
    mSourceValue = std::make_unique<wosx::GPUDenseGrid<float, CHANNELS, 2>>(
        mSampler, getTextureData(sourceValue_),
        sourceValueGridShape, gridMin_, gridMax_, false);
    Vector2i absorbingBoundaryValueGridShape = Vector2i(absorbingBoundaryValue_.w,
                                                        absorbingBoundaryValue_.h);
    mAbsorbingBoundaryValue = std::make_unique<wosx::GPUDenseGrid<float, CHANNELS, 2>>(
        mSampler, getTextureData(absorbingBoundaryValue_),
        absorbingBoundaryValueGridShape, gridMin_, gridMax_, false);
    Vector2i reflectingBoundaryValueGridShape = Vector2i(reflectingBoundaryValue_.w,
                                                         reflectingBoundaryValue_.h);
    mReflectingBoundaryValue = std::make_unique<wosx::GPUDenseGrid<float, CHANNELS, 2>>(
        mSampler, getTextureData(reflectingBoundaryValue_),
        reflectingBoundaryValueGridShape, gridMin_, gridMax_, false);
    if (absorbingBoundaryNormalAlignedValue_.h > 0 && absorbingBoundaryNormalAlignedValue_.w > 0) {
        mAbsorbingBoundaryNormalAlignedValue = std::make_unique<wosx::GPUDenseGrid<float, CHANNELS, 2>>(
            mSampler, getTextureData(absorbingBoundaryNormalAlignedValue_),
            absorbingBoundaryValueGridShape, gridMin_, gridMax_, false);
    }
    if (reflectingBoundaryNormalAlignedValue_.h > 0 && reflectingBoundaryNormalAlignedValue_.w > 0) {
        mReflectingBoundaryNormalAlignedValue = std::make_unique<wosx::GPUDenseGrid<float, CHANNELS, 2>>(
            mSampler, getTextureData(reflectingBoundaryNormalAlignedValue_),
            reflectingBoundaryValueGridShape, gridMin_, gridMax_, false);
    }
}

template <typename T>
typename GPUModelProblemPDE<T>::TextureData
GPUModelProblemPDE<T>::getTextureData(const Image<IMAGE_CHANNELS>& image) const
{
    // DenseGrid2D samples Slang Texture2D with localCoords.yx, while the CPU
    // model samples images through Image::get(uv), whose default flips Y.
    auto textureData = image.toEigen(true);
    if constexpr (CHANNELS == IMAGE_CHANNELS) {
        return textureData;

    } else {
        TextureData paddedTextureData = TextureData::Zero(image.h*image.w, CHANNELS);
        paddedTextureData.template leftCols<IMAGE_CHANNELS>() = textureData;
        return paddedTextureData;
    }
}

template <typename T>
void GPUModelProblemPDE<T>::allocate(wosx::GPUContext& context)
{
    mSampler->allocate(context, wosx::TextureFilteringMode::Linear,
                       wosx::TextureAddressingMode::ClampToEdge);
    mSourceValue->allocate(context);
    mAbsorbingBoundaryValue->allocate(context);
    mReflectingBoundaryValue->allocate(context);
    if (mAbsorbingBoundaryNormalAlignedValue) {
        mAbsorbingBoundaryNormalAlignedValue->allocate(context);
    }
    if (mReflectingBoundaryNormalAlignedValue) {
        mReflectingBoundaryNormalAlignedValue->allocate(context);
    }
}

template <typename T>
void GPUModelProblemPDE<T>::setResources(const wosx::ShaderCursor& cursor, bool printLogs) const
{
    int nResources = 4;
    if (mAbsorptionCoeff > 0.0f) {
        cursor["mAbsorptionCoeff"].setData(mAbsorptionCoeff);
        nResources++;
    }
    cursor["mRobinCoeff"].setData(mRobinCoeff);
    mSourceValue->setResources(cursor["mSourceValue"], printLogs);
    mAbsorbingBoundaryValue->setResources(cursor["mAbsorbingBoundaryValue"], printLogs);
    mReflectingBoundaryValue->setResources(cursor["mReflectingBoundaryValue"], printLogs);
    if (mAbsorbingBoundaryNormalAlignedValue) {
        mAbsorbingBoundaryNormalAlignedValue->setResources(
            cursor["mAbsorbingBoundaryNormalAlignedValue"], printLogs);
        nResources++;
    }
    if (mReflectingBoundaryNormalAlignedValue) {
        mReflectingBoundaryNormalAlignedValue->setResources(
            cursor["mReflectingBoundaryNormalAlignedValue"], printLogs);
        nResources++;
    }
    if (printLogs) {
        wosx::printReflectionInfo(cursor, nResources, getReflectionType());
    }
}

template <typename T>
std::string GPUModelProblemPDE<T>::getReflectionType() const
{
    return "ModelProblemPDE";
}

template <typename T>
wosx::GPUPDEType GPUModelProblemPDE<T>::getType() const
{
    return mAbsorptionCoeff > 0.0f ? wosx::GPUPDEType::ScreenedPoisson : wosx::GPUPDEType::Poisson;
}

template <typename T>
GPUModelProblem<T>::GPUModelProblem(const json& config, std::string directoryPath):
mKelvinTransform(Vector2(0.0f, 0.125f)), // ensure origin lies inside default domain for demo, a requirement for exterior problems
mPde(nullptr),
mQueries(nullptr),
mPdeInvertedDomain(nullptr),
mQueriesInvertedDomain(nullptr)
{
    // load config settings
    auto getFilePath = [config, directoryPath](const std::string& fileName) -> std::string {
        return directoryPath + "/" + getRequired<std::string>(config, fileName);
    };

    std::string geometryFile = getFilePath("geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    mSolveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    mSourceValue = Image<IMAGE_CHANNELS>(getFilePath("sourceValue"));
    mAbsorbingBoundaryValue = Image<IMAGE_CHANNELS>(getFilePath("absorbingBoundaryValue"));
    mReflectingBoundaryValue = Image<IMAGE_CHANNELS>(getFilePath("reflectingBoundaryValue"));
    if (mSolveDoubleSided) {
        mAbsorbingBoundaryNormalAlignedValue = Image<IMAGE_CHANNELS>(getFilePath("absorbingBoundaryNormalAlignedValue"));
        mReflectingBoundaryNormalAlignedValue = Image<IMAGE_CHANNELS>(getFilePath("reflectingBoundaryNormalAlignedValue"));
        if (mAbsorbingBoundaryNormalAlignedValue.h != mAbsorbingBoundaryValue.h ||
            mAbsorbingBoundaryNormalAlignedValue.w != mAbsorbingBoundaryValue.w ||
            mReflectingBoundaryNormalAlignedValue.h != mReflectingBoundaryValue.h ||
            mReflectingBoundaryNormalAlignedValue.w != mReflectingBoundaryValue.w) {
            std::cerr << "Error: normal-aligned value buffers must be the same shape as the value buffers" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    mIsReflectingBoundary = Image<1>(getFilePath("isReflectingBoundary"));
    mSolveExterior = getOptional<bool>(config, "solveExterior", false);
    mDomainIsWatertight = getOptional<bool>(config, "domainIsWatertight", false);
    mRobinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    mAbsorptionCoeff = mSolveExterior ? 0.0f : getOptional<float>(config, "absorptionCoeff", 0.0f); // kelvin transform requires absorption coefficient to be 0

    // load a boundary mesh from an OBJ file
    loadOBJ(geometryFile, normalize, flipOrientation);

    // setup the PDE
    setupPDE();

    // partition the boundary mesh into absorbing and reflecting boundary elements
    partitionBoundaryMesh();

    // specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
    // we use a constant value for all elements in this demo, but WoSX supports variable coefficients
    mMinRobinCoeffValues.resize(mReflectingBoundaryIndices.size(), std::fabs(mRobinCoeff));
    mMaxRobinCoeffValues.resize(mReflectingBoundaryIndices.size(), std::fabs(mRobinCoeff));

    // set up geometric queries for the absorbing and reflecting boundary
    bool areRobinConditionsPureNeumann = mRobinCoeff == 0.0f;
    setupGeometricQueries(mAbsorbingBoundaryPositions, mReflectingBoundaryPositions,
                          mBoundingBox, mMinRobinCoeffValues, mMaxRobinCoeffValues,
                          areRobinConditionsPureNeumann, mQueries);

    if (mSolveExterior) {
        // invert the exterior problem into an equivalent interior problem
        invertExteriorProblem();

        // set up geometric queries for the inverted absorbing and reflecting boundary
        setupGeometricQueries(mInvertedAbsorbingBoundaryPositions,
                              mInvertedReflectingBoundaryPositions,
                              mInvertedBoundingBox,
                              mMinRobinCoeffValuesInvertedDomain,
                              mMaxRobinCoeffValuesInvertedDomain,
                              areRobinConditionsPureNeumann,
                              mQueriesInvertedDomain);
    }
}

template <typename T>
void GPUModelProblem<T>::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    wosx::loadBoundaryMesh<2>(filename, mPositions, mIndices);
    if (normalize) wosx::normalize<2>(mPositions);
    if (flipOrientation) wosx::flipOrientation<2>(mIndices);
    mBoundingBox = wosx::computeBoundingBox<2>(mPositions, true, 1.0);
}

template <typename T>
void GPUModelProblem<T>::setupPDE()
{
    mPde = std::make_shared<GPUModelProblemPDE<T>>(mAbsorptionCoeff, mRobinCoeff,
                                                   mBoundingBox.first, mBoundingBox.second,
                                                   mSourceValue, mAbsorbingBoundaryValue,
                                                   mReflectingBoundaryValue,
                                                   mAbsorbingBoundaryNormalAlignedValue,
                                                   mReflectingBoundaryNormalAlignedValue);
}

template <typename T>
void GPUModelProblem<T>::partitionBoundaryMesh()
{
    std::function<bool(const Vector2&)> onReflectingBoundary = [this](const Vector2& x) {
        float maxLength = (this->mBoundingBox.second - this->mBoundingBox.first).maxCoeff();
        Vector2 uv = (x - this->mBoundingBox.first)/maxLength;
        return this->mSolveExterior ? this->mReflectingBoundaryValue.get(uv)[0] > 0 :
                                      this->mIsReflectingBoundary.get(uv)[0] > 0;
    };

    // use WoSX's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    wosx::partitionBoundaryMesh<2>(onReflectingBoundary, mPositions, mIndices,
                                   mAbsorbingBoundaryPositions, mAbsorbingBoundaryIndices,
                                   mReflectingBoundaryPositions, mReflectingBoundaryIndices);
}

template <typename T>
void GPUModelProblem<T>::setupGeometricQueries(const std::vector<Vector2>& absorbingBoundaryPositions,
                                               const std::vector<Vector2>& reflectingBoundaryPositions,
                                               const std::pair<Vector2, Vector2>& boundingBox,
                                               const std::vector<float>& minRobinCoeffValues,
                                               const std::vector<float>& maxRobinCoeffValues,
                                               bool areRobinConditionsPureNeumann,
                                               std::shared_ptr<wosx::GPUGeometricQueries<2>>& queries)
{
    // initialize GPU handler for absorbing boundary
    std::shared_ptr<wosx::GPUAbsorbingBoundaryHandler> absorbingBoundaryHandler = nullptr;
    if (absorbingBoundaryPositions.size() > 0) {
        absorbingBoundaryHandler = std::make_shared<wosx::GPUFcpwDirichletBoundaryHandler<2>>(
            absorbingBoundaryPositions, mAbsorbingBoundaryIndices);

    } else {
        absorbingBoundaryHandler = std::make_shared<wosx::GPUEmptyAbsorbingBoundaryHandler>();
    }

    // initialize GPU handler for reflecting boundary
    std::shared_ptr<wosx::GPUReflectingBoundaryHandler> reflectingBoundaryHandler = nullptr;
    if (reflectingBoundaryPositions.size() > 0) {
        std::function<bool(float, int)> ignoreCandidateSilhouette =
            wosx::getIgnoreCandidateSilhouetteCallback(mSolveDoubleSided);
        if (areRobinConditionsPureNeumann) {
            reflectingBoundaryHandler = std::make_shared<wosx::GPUFcpwNeumannBoundaryHandler<2>>(
                reflectingBoundaryPositions, mReflectingBoundaryIndices, ignoreCandidateSilhouette);

        } else {
            reflectingBoundaryHandler = std::make_shared<wosx::GPUFcpwRobinBoundaryHandler<2>>(
                reflectingBoundaryPositions, mReflectingBoundaryIndices, ignoreCandidateSilhouette,
                minRobinCoeffValues, maxRobinCoeffValues);
        }

    } else {
        reflectingBoundaryHandler = std::make_shared<wosx::GPUEmptyReflectingBoundaryHandler>();
    }

    // initialize GPU geometric queries
    queries = std::make_shared<wosx::GPUGeometricQueries<2>>(
        absorbingBoundaryHandler, reflectingBoundaryHandler,
        boundingBox.first, boundingBox.second, mDomainIsWatertight);
}

template <typename T>
void GPUModelProblem<T>::invertExteriorProblem()
{
    // invert the domain
    std::vector<Vector2> invertedPositions;
    mKelvinTransform.transformPoints(mPositions, invertedPositions);
    mKelvinTransform.transformPoints(mAbsorbingBoundaryPositions, mInvertedAbsorbingBoundaryPositions);
    mKelvinTransform.transformPoints(mReflectingBoundaryPositions, mInvertedReflectingBoundaryPositions);

    // compute the bounding box for the inverted domain
    mInvertedBoundingBox = wosx::computeBoundingBox<2>(invertedPositions, true, 1.0);

    // setup a modified PDE on the inverted domain
    mPdeInvertedDomain = std::make_shared<wosx::GPUKelvinPDE<T, 2>>(mPde, mKelvinTransform.getOrigin());

    // compute the modified Robin coefficients on the inverted domain
    if (mRobinCoeff != 0.0f) {
        // compute the modified Robin coefficients on the inverted domain
        std::vector<float> minRobinCoeffValues(mReflectingBoundaryIndices.size(), mRobinCoeff);
        std::vector<float> maxRobinCoeffValues(mReflectingBoundaryIndices.size(), mRobinCoeff);
        mKelvinTransform.computeRobinCoefficients(mInvertedReflectingBoundaryPositions,
                                                  mReflectingBoundaryIndices,
                                                  minRobinCoeffValues, maxRobinCoeffValues,
                                                  mMinRobinCoeffValuesInvertedDomain,
                                                  mMaxRobinCoeffValuesInvertedDomain);
    }
}
