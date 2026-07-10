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

#include <wosx/utils/sdf_grid_geometric_queries.h>
#include <fcpw/fcpw_gpu.h>

namespace wosx {

using fcpw::float2;
using fcpw::float3;
using namespace fcpw;

template <typename T, size_t CHANNELS, size_t DIM>
class GPUDenseGrid: public GPUShaderObject {
public:
    // constructors
    GPUDenseGrid(std::shared_ptr<GPUSampler> sampler_,
                 const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData_,
                 const Vectori<DIM>& gridShape_,
                 const Vector<DIM>& gridMin_,
                 const Vector<DIM>& gridMax_,
                 bool unorderedAccess_);
    GPUDenseGrid(std::shared_ptr<GPUSampler> sampler_,
                 std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback_,
                 const Vectori<DIM>& gridShape_,
                 const Vector<DIM>& gridMin_,
                 const Vector<DIM>& gridMax_,
                 bool unorderedAccess_);

    // methods to allocate and set GPU resources, and return reflection type
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    std::shared_ptr<GPUSampler> sampler;
    DenseGrid<T, CHANNELS, DIM> grid;
    GPUTexture<DIM> texture;
    bool unorderedAccess;
};

template <size_t DIM>
class GPUSdfGrid: public GPUShaderObject {
public:
    // constructors
    GPUSdfGrid(const Eigen::VectorXf& sdfData_,
               const Vectori<DIM>& gridShape_,
               const Vector<DIM>& gridMin_,
               const Vector<DIM>& gridMax_,
               bool unorderedAccess_);
    GPUSdfGrid(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback_,
               const Vectori<DIM>& gridShape_,
               const Vector<DIM>& gridMin_,
               const Vector<DIM>& gridMax_,
               bool unorderedAccess_);

    // methods to allocate and set GPU resources, and return reflection type
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // members
    SdfGrid<DIM> grid;
    GPUTexture<DIM> texture;
    GPUSampler sampler;
    bool unorderedAccess;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t CHANNELS, size_t DIM>
GPUDenseGrid<T, CHANNELS, DIM>::GPUDenseGrid(std::shared_ptr<GPUSampler> sampler_,
                                             const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData_,
                                             const Vectori<DIM>& gridShape_,
                                             const Vector<DIM>& gridMin_,
                                             const Vector<DIM>& gridMax_,
                                             bool unorderedAccess_):
sampler(sampler_),
grid(gridData_, gridShape_, gridMin_, gridMax_),
unorderedAccess(unorderedAccess_)
{
    // do nothing
}

template <typename T, size_t CHANNELS, size_t DIM>
GPUDenseGrid<T, CHANNELS, DIM>::GPUDenseGrid(std::shared_ptr<GPUSampler> sampler_,
                                             std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback_,
                                             const Vectori<DIM>& gridShape_,
                                             const Vector<DIM>& gridMin_,
                                             const Vector<DIM>& gridMax_,
                                             bool unorderedAccess_):
sampler(sampler_),
grid(gridDataCallback_, gridShape_, gridMin_, gridMax_),
unorderedAccess(unorderedAccess_)
{
    // do nothing
}

template <typename T, size_t CHANNELS, size_t DIM>
void GPUDenseGrid<T, CHANNELS, DIM>::allocate(GPUContext& context)
{
    // the texture width (u) axis maps to the fastest-varying (last) grid dimension:
    // the grid data is uploaded linearly in its row-major CPU layout, and the shaders
    // sample the texture with correspondingly reversed (.yx/.zyx) coordinate swizzles
    size_t width = grid.shape[DIM - 1];
    size_t height = grid.shape[DIM - 2];
    size_t depth = DIM == 3 ? grid.shape[0] : 1;
    size_t size = width*height*depth;
    if constexpr (CHANNELS == 1) {
        texture.template allocate<T, CHANNELS>(context.device, sampler->sampler,
                                               unorderedAccess, width, height, depth,
                                               grid.data.data(), size);

    } else {
        // Texture uploads require interleaved texels, while Eigen's default
        // column-major matrix layout stores each channel contiguously.
        Eigen::Matrix<T, Eigen::Dynamic, CHANNELS, Eigen::RowMajor> textureData = grid.data;
        texture.template allocate<T, CHANNELS>(context.device, sampler->sampler,
                                               unorderedAccess, width, height, depth,
                                               textureData.data(), size);
    }
}

template <typename T, size_t CHANNELS, size_t DIM>
void GPUDenseGrid<T, CHANNELS, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["data"].setBinding(texture.view);
    cursor["sampler"].setBinding(sampler->sampler);
    if (DIM == 2) {
        cursor["origin"].setData(float2{grid.origin[0], grid.origin[1]});
        cursor["extent"].setData(float2{grid.extent[0], grid.extent[1]});

    } else if (DIM == 3) {
        cursor["origin"].setData(float3{grid.origin[0], grid.origin[1], grid.origin[2]});
        cursor["extent"].setData(float3{grid.extent[0], grid.extent[1], grid.extent[2]});
    }

    if (printLogs) {
        printReflectionInfo(cursor, 4, getReflectionType());
    }
}

template <typename T, size_t CHANNELS, size_t DIM>
std::string GPUDenseGrid<T, CHANNELS, DIM>::getReflectionType() const
{
    std::string dataType = "";
    if (std::is_same<T, float>::value) {
        dataType = "Float"; // only float values are supported for now

    } else {
        std::cerr << "Error: Unsupported texture format for type T with "
                  << CHANNELS << " channel(s)" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (CHANNELS != 1) dataType += std::to_string(CHANNELS);
    return dataType + "DenseGrid" + std::to_string(DIM) + "D";
}

template <size_t DIM>
GPUSdfGrid<DIM>::GPUSdfGrid(const Eigen::VectorXf& sdfData_,
                            const Vectori<DIM>& gridShape_,
                            const Vector<DIM>& gridMin_,
                            const Vector<DIM>& gridMax_,
                            bool unorderedAccess_):
grid(sdfData_, gridShape_, gridMin_, gridMax_),
unorderedAccess(unorderedAccess_)
{
    // do nothing
}

template <size_t DIM>
GPUSdfGrid<DIM>::GPUSdfGrid(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback_,
                            const Vectori<DIM>& gridShape_,
                            const Vector<DIM>& gridMin_,
                            const Vector<DIM>& gridMax_,
                            bool unorderedAccess_):
grid(sdfDataCallback_, gridShape_, gridMin_, gridMax_),
unorderedAccess(unorderedAccess_)
{
    // do nothing
}

template <size_t DIM>
void GPUSdfGrid<DIM>::allocate(GPUContext& context)
{
    // the texture width (u) axis maps to the fastest-varying (last) grid dimension:
    // the grid data is uploaded linearly in its row-major CPU layout, and the shaders
    // sample the texture with correspondingly reversed (.yx/.zyx) coordinate swizzles
    size_t width = grid.shape[DIM - 1];
    size_t height = grid.shape[DIM - 2];
    size_t depth = DIM == 3 ? grid.shape[0] : 1;
    size_t size = width*height*depth;
    sampler.allocate(context, TextureFilteringMode::Linear, TextureAddressingMode::ClampToEdge);
    texture.template allocate<float, 1>(context.device, sampler.sampler,
                                        unorderedAccess, width, height, depth,
                                        grid.data.data(), size);
}

template <size_t DIM>
void GPUSdfGrid<DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    cursor["data"].setBinding(texture.view);
    cursor["sampler"].setBinding(sampler.sampler);
    if (DIM == 2) {
        cursor["origin"].setData(float2{grid.origin[0], grid.origin[1]});
        cursor["extent"].setData(float2{grid.extent[0], grid.extent[1]});
        cursor["cellSpacing"].setData(float2{grid.extent[0]/(float)grid.shape[0],
                                             grid.extent[1]/(float)grid.shape[1]});

    } else if (DIM == 3) {
        cursor["origin"].setData(float3{grid.origin[0], grid.origin[1], grid.origin[2]});
        cursor["extent"].setData(float3{grid.extent[0], grid.extent[1], grid.extent[2]});
        cursor["cellSpacing"].setData(float3{grid.extent[0]/(float)grid.shape[0],
                                             grid.extent[1]/(float)grid.shape[1],
                                             grid.extent[2]/(float)grid.shape[2]});
    }

    if (printLogs) {
        printReflectionInfo(cursor, 5, getReflectionType());
    }
}

template <size_t DIM>
std::string GPUSdfGrid<DIM>::getReflectionType() const
{
    return "SdfGrid" + std::to_string(DIM) + "D";
}

} // wosx
