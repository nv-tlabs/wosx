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

// This file defines an interface for Partial Differential Equations (PDEs),
// specifically Poisson and screened Poisson equations, with Dirichlet, Neumann,
// and Robin boundary conditions. As part of the problem setup, users of WoSX
// should populate the callback functions defined by the PDE interface.

#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace wosx {

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <typename T, size_t DIM>
struct PDE {
    // constructor
    PDE();

    // members
    float absorptionCoeff; // must be positive or equal to zero
    bool isSourceConstant; // set to true if source term is constant
    bool areRobinConditionsPureNeumann; // set to false if Robin coefficients are non-zero anywhere
    bool areRobinCoeffsNonnegative; // set to false if Robin coefficients are negative anywhere

    // returns source term
    std::function<T(const Vector<DIM>&)> source;

    // returns Dirichlet boundary conditions
    std::function<T(const Vector<DIM>&, bool)> dirichlet;

    // returns Robin boundary conditions and coefficients
    std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> robin; // dual purposes for Neumann conditions when Robin coeff is zero
    std::function<float(const Vector<DIM>&, const Vector<DIM>&, bool)> robinCoeff;

    // checks if the PDE has reflecting boundary conditions (Neumann or Robin) at the given point
    std::function<bool(const Vector<DIM>&)> hasReflectingBoundaryConditions;

    // check if the PDE has a non-zero robin coefficient value at the given point
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff; // set automatically
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
PDE<T, DIM>::PDE():
absorptionCoeff(0.0f),
isSourceConstant(false),
areRobinConditionsPureNeumann(true),
areRobinCoeffsNonnegative(true),
source({}),
dirichlet({}),
robin({}),
robinCoeff({}),
hasReflectingBoundaryConditions({})
{
    hasNonZeroRobinCoeff = [this](const Vector<DIM>& x) {
        // the robin coefficient is zero everywhere for pure Neumann conditions,
        // so skip evaluating it
        if (this->areRobinConditionsPureNeumann) {
            return false;
        }

        if (this->robinCoeff) {
            Vector<DIM> n = Vector<DIM>::Zero();
            n(0) = 1.0f;

            return std::fabs(this->robinCoeff(x, n, true)) > 0.0f ||
                   std::fabs(this->robinCoeff(x, n, false)) > 0.0f;
        }

        return false;
    };
}

} // wosx
