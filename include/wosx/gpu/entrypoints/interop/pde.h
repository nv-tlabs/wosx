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

enum class GPUPDEType: int {
    Poisson = 1,
    ScreenedPoisson = 2
};

class GPUPDE: public GPUShaderObject {
public:
    // allocates GPU resources, and returns PDE type
    virtual void allocate(GPUContext& context) = 0;
    virtual GPUPDEType getType() const = 0;

    // checks if the PDE uses a Kelvin transform (e.g., for exterior problems)
    virtual bool usesKelvinTransform() const { return false; }
};

template <typename T, size_t DIM>
class GPUKelvinTransform: public GPUShaderObject {
public:
    // constructor
    GPUKelvinTransform(const Vector<DIM>& origin_);

    // updates the origin of the transform
    void updateOrigin(const Vector<DIM>& origin_);

    // sets GPU resources, and returns reflection type
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;

private:
    // member
    Vector<DIM> origin;
};

template <typename T, size_t DIM>
class GPUKelvinPDE: public GPUPDE {
public:
    // constructor
    GPUKelvinPDE(std::shared_ptr<GPUPDE> pdeExteriorDomain_,
                 const Vector<DIM>& origin_=Vector<DIM>::Zero());

    // updates the origin of the kelvin transform
    void updateOrigin(const Vector<DIM>& origin_);

    // allocates and sets GPU resources, and returns type info
    void allocate(GPUContext& context);
    void setResources(const ShaderCursor& cursor, bool printLogs) const;
    std::string getReflectionType() const;
    GPUPDEType getType() const;

    // checks if the PDE uses a Kelvin transform (e.g., for exterior problems)
    bool usesKelvinTransform() const { return true; }

private:
    // members
    std::shared_ptr<GPUPDE> pdeExteriorDomain;
    GPUKelvinTransform<T, DIM> kelvinTransform;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
GPUKelvinTransform<T, DIM>::GPUKelvinTransform(const Vector<DIM>& origin_)
{
    updateOrigin(origin_);
}

template <typename T, size_t DIM>
void GPUKelvinTransform<T, DIM>::updateOrigin(const Vector<DIM>& origin_)
{
    origin = origin_;
}

template <typename T, size_t DIM>
void GPUKelvinTransform<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    if (DIM == 2) {
        cursor["origin"].setData(float2{origin[0], origin[1]});

    } else if (DIM == 3) {
        cursor["origin"].setData(float3{origin[0], origin[1], origin[2]});
    }

    if (printLogs) {
        printReflectionInfo(cursor, 1, getReflectionType());
    }
}

template <typename T, size_t DIM>
std::string GPUKelvinTransform<T, DIM>::getReflectionType() const
{
    std::string arguments = getDataTypeName<T>() + ", " +
                            std::to_string(DIM);
    return "KelvinTransform<" + arguments + ">";
}

template <typename T, size_t DIM>
GPUKelvinPDE<T, DIM>::GPUKelvinPDE(std::shared_ptr<GPUPDE> pdeExteriorDomain_,
                                   const Vector<DIM>& origin_):
pdeExteriorDomain(pdeExteriorDomain_),
kelvinTransform(origin_)
{
    // do nothing
}

template <typename T, size_t DIM>
void GPUKelvinPDE<T, DIM>::updateOrigin(const Vector<DIM>& origin_)
{
    kelvinTransform.updateOrigin(origin_);
}

template <typename T, size_t DIM>
void GPUKelvinPDE<T, DIM>::allocate(GPUContext& context)
{
    pdeExteriorDomain->allocate(context);
}

template <typename T, size_t DIM>
void GPUKelvinPDE<T, DIM>::setResources(const ShaderCursor& cursor, bool printLogs) const
{
    pdeExteriorDomain->setResources(cursor["pdeExteriorDomain"], printLogs);
    kelvinTransform.setResources(cursor["kelvinTransform"], printLogs);
    if (printLogs) printReflectionInfo(cursor, 2, getReflectionType());
}

template <typename T, size_t DIM>
std::string GPUKelvinPDE<T, DIM>::getReflectionType() const
{
    std::string arguments = getDataTypeName<T>() + ", " +
                            pdeExteriorDomain->getReflectionType() + ", " +
                            std::to_string(DIM);
    return "KelvinPDE<" + arguments + ">";
}

template <typename T, size_t DIM>
GPUPDEType GPUKelvinPDE<T, DIM>::getType() const
{
    return GPUPDEType::Poisson;
}

} // wosx
