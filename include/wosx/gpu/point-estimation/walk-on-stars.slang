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

// This file implements the Monte Carlo walk-on-stars algorithm for solving
// Poisson and screened Poisson equations with Dirichlet, Neumann and/or Robin
// boundary conditions. This algorithm is a strict generalization of walk-on-spheres,
// and hence can also be used to solve PDEs with pure Dirichlet boundary conditions.
// The estimated PDE solution, gradient, and other statistics can be queried
// from the SampleStatistics struct, which is stored in the SamplePoint struct.
// Each call to solve(...) improves the estimate of the PDE solution at the
// user-selected set of input points, enabling progressive evaluation.
//
// Resources:
// - Walk on Stars: Grid-Free Monte Carlo for PDEs with Neumann Conditions [2023]
// - Walkin’ Robin: Walk on Stars with Robin Boundary Conditions [2024]

implementing wosx;
__include point_estimation.common;

public struct WalkOnStars<T, PDE, GeometricQueries, let DIM : int>
    where T : IFloat
    where PDE : IPDE<T, DIM>
    where GeometricQueries : IGeometricQueries<DIM>
{
    private WalkSettings walkSettings;
    private GeometricQueries geometricQueries;
    private PDE pde;
    private typedef PDE::GreensFnBall GreensFn;
    private typedef GreensFn::SphereSampler SphereSampler;

    // solves the given PDE at the input point; NOTE: assumes the point does not
    // lie on the boundary when estimating the gradient
    public void solve(SamplePoint<DIM> samplePt,
                      uint nWalks, inout pcg32 rng,
                      inout SampleStatistics<T, DIM> statistics)
    {
        if (samplePt.estimationQuantity != EstimationQuantity::Skip)
        {
            if (samplePt.estimationQuantity == EstimationQuantity::SolutionAndGradient)
            {
                estimateSolutionAndGradient(samplePt, nWalks, rng, statistics);
            }
            else
            {
                estimateSolution(samplePt, nWalks, rng, statistics);
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // lower-level functions

    // check if the PDE has a non-zero robin coefficient value at the given point
    public bool checkForNonZeroRobinCoeff(vector<float, DIM> x)
    {
        vector<float, DIM> n = vector<float, DIM>(0.0);
        n[0] = 1.0;

        return abs(pde.robinCoeff(x, n, true)) > 0.0 || abs(pde.robinCoeff(x, n, false)) > 0.0;
    }

    // computes the contribution from the reflecting boundary at a particular point in the walk
    public void computeReflectingBoundaryContribution(float starRadius,
                                                      bool flipNormalOrientation,
                                                      inout pcg32 rng,
                                                      inout GreensFn greensFn,
                                                      inout WalkState<T, DIM> state)
    {
        if (!geometricQueries.isReflectingBoundaryEmpty() &&
            walkSettings.ignoreReflectingBoundaryContribution == 0)
        {
            // compute the non-zero reflecting boundary contribution inside the star-shaped region
            // (defined to be zero outside this region)
            vector<float, DIM> randNumsForBoundarySampling;
            for (int i = 0; i < DIM; i++)
            {
                randNumsForBoundarySampling[i] = rng.nextFloat();
            }

            BoundarySample<DIM> boundarySample;
            if (geometricQueries.sampleReflectingBoundary(state.currentPt, starRadius,
                                                          randNumsForBoundarySampling,
                                                          boundarySample))
            {
                vector<float, DIM> directionToSample = boundarySample.pt - state.currentPt;
                float distToSample = length(directionToSample);
                float alpha = state.onReflectingBoundary != 0 ? 2.0 : 1.0;
                bool estimateBoundaryNormalAligned = false;

                if (walkSettings.solveDoubleSided != 0)
                {
                    // normalize the direction to the sample, and flip the sample normal
                    // orientation if the geometry is front-facing; NOTE: using a precision
                    // parameter since unlike direction sampling, samples can lie on the same
                    // halfplane as the current walk location
                    directionToSample /= distToSample;
                    if (flipNormalOrientation)
                    {
                        estimateBoundaryNormalAligned = true;
                    }
                    else if (dot(directionToSample, boundarySample.normal) < -walkSettings.silhouettePrecision)
                    {
                        bool flipBoundarySampleNormal = true;
                        if (alpha > 1.0)
                        {
                            // on concave boundaries, we want to sample back-facing values
                            // on front-facing geometry below the hemisphere, so we avoid
                            // flipping the normal orientation in this case
                            flipBoundarySampleNormal = dot(directionToSample, state.currentNormal) <
                                                       -walkSettings.silhouettePrecision;
                        }

                        if (flipBoundarySampleNormal)
                        {
                            estimateBoundaryNormalAligned = true;
                        }
                    }
                }

                if (boundarySample.pdf > 0.0 && distToSample < starRadius)
                {
                    vector<float, DIM> boundarySampleNormal = boundarySample.normal * (estimateBoundaryNormalAligned ? -1.0 : 1.0);
                    if (!geometricQueries.intersectsWithReflectingBoundary(state.currentPt, boundarySample.pt,
                                                                           state.currentNormal, boundarySampleNormal,
                                                                           state.onReflectingBoundary != 0, true))
                    {
                        float r = max(greensFn.getRadiusClamp(), length(boundarySample.pt - state.currentPt));
                        float G = greensFn.evaluate(r);
                        bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided != 0 &&
                                                                estimateBoundaryNormalAligned;
                        T h = pde.robin(boundarySample.pt, boundarySample.normal, returnBoundaryNormalAlignedValue);

                        state.totalReflectingBoundaryContribution = state.totalReflectingBoundaryContribution +
                                                                    h.scale(state.throughput * alpha * G / boundarySample.pdf);
                    }
                }
            }
        }
    }

    // computes the source contribution at a particular point in the walk
    public void computeSourceContribution(vector<float, DIM> direction,
                                          float intersectionDist,
                                          inout pcg32 rng,
                                          inout GreensFn greensFn,
                                          inout WalkState<T, DIM> state)
    {
        if (walkSettings.ignoreSourceContribution == 0)
        {
            // compute the source contribution inside the star-shaped region;
            // define the source value to be zero outside this region
            float sourceRadius, sourcePdf;
            vector<float, DIM> sourcePt = greensFn.sampleVolume(direction, rng, sourceRadius, sourcePdf);
            if (sourceRadius <= intersectionDist && !geometricQueries.outsideBoundingDomain(sourcePt))
            {
                // NOTE: hemispherical sampling causes the alpha term to be cancelled out when
                // currentPt is on the reflecting boundary; in this case, the green's function
                // norm remains unchanged even though our domain is a hemisphere;
                // for double-sided problems in watertight domains, both the current pt
                // and source pt lie either inside or outside the domain by construction
                T sourceContribution = pde.source(sourcePt).scale(greensFn.norm());
                state.totalSourceContribution = state.totalSourceContribution + sourceContribution.scale(state.throughput);
            }
        }
    }

    // computes the throughput of a single walk step
    public float computeWalkStepThroughput(inout GreensFn greensFn,
                                           inout WalkState<T, DIM> state)
    {
        if (state.onReflectingBoundary != 0 && state.prevDistance > FLT_EPSILON)
        {
            float robinCoeff = 0.0;
            vector<float, DIM> normal = state.currentNormal;
            if (!pde.areRobinConditionsPureNeumann())
            {
                bool flipNormalOrientation = false;
                if (walkSettings.solveDoubleSided != 0)
                {
                    flipNormalOrientation = dot(state.prevDirection, state.currentNormal) < 0.0;
                    normal *= flipNormalOrientation ? -1.0 : 1.0;
                }

                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided != 0 &&
                                                        flipNormalOrientation;
                robinCoeff = pde.robinCoeff(state.currentPt, state.currentNormal,
                                            returnBoundaryNormalAlignedValue);
            }

            float reflectance = greensFn.reflectance(state.prevDistance, state.prevDirection,
                                                     normal, robinCoeff);
            float maxReflectance = pde.areRobinCoeffsNonnegative() ? 1.0 : 2.0;
            return clamp(reflectance, 0.0, maxReflectance);
        }

        return greensFn.directionSampledPoissonKernel(state.currentPt);
    }

    // applies russian roulette to a walk based on the current state
    public bool applyRussianRoulette(inout pcg32 rng,
                                     inout WalkState<T, DIM> state)
    {
        if (state.throughput < walkSettings.russianRouletteThreshold)
        {
            float survivalProb = state.throughput / walkSettings.russianRouletteThreshold;
            if (survivalProb < rng.nextFloat())
            {
                state.throughput = 0.0;
                return true;
            }

            state.throughput = walkSettings.russianRouletteThreshold;
        }

        return false;
    }

    // performs a single reflecting random walk starting at the input point
    public WalkCompletionCode walk(float distToAbsorbingBoundary,
                                   bool flipNormalOrientation,
                                   inout pcg32 rng,
                                   inout GreensFn greensFn,
                                   inout WalkState<T, DIM> state)
    {
        // recursively perform a random walk till it reaches the absorbing boundary
        bool firstStep = true;
        while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary)
        {
            // for problems with double-sided boundary conditions, flip the current
            // normal orientation if the geometry is front-facing
            if (!firstStep)
            {
                flipNormalOrientation = false;
            }

            if (walkSettings.solveDoubleSided != 0 && state.onReflectingBoundary != 0)
            {
                if (state.prevDistance > 0.0 && dot(state.prevDirection, state.currentNormal) < 0.0)
                {
                    state.currentNormal *= -1.0;
                    flipNormalOrientation = true;
                }
            }

            // compute the star radius
            float starRadius;
            if (walkSettings.stepsBeforeUsingMaximalSpheres <= state.walkLength)
            {
                starRadius = distToAbsorbingBoundary;
            }
            else if (state.onReflectingBoundary != 0 && checkForNonZeroRobinCoeff(state.currentPt))
            {
                // NOTE: reflectance, and hence sphere radius, is zero exactly on the boundary,
                // therefore we use a small epsilon for the sphere radius to ensure the walk continues
                starRadius = walkSettings.epsilonShellForReflectingBoundary;
            }
            else
            {
                // NOTE: using distToAbsorbingBoundary as the maximum radius for the star radius
                // query can result in a smaller than maximal star-shaped region: should ideally
                // use the distance to the closest visible point on the absorbing boundary
                starRadius = geometricQueries.computeStarRadiusForReflectingBoundary(
                    state.currentPt, walkSettings.epsilonShellForReflectingBoundary,
                    distToAbsorbingBoundary, walkSettings.silhouettePrecision,
                    flipNormalOrientation);

                // shrink the radius slightly for numerical robustness---using a conservative
                // distance does not impact correctness
                if (!geometricQueries.isReflectingBoundaryEmpty() &&
                    walkSettings.epsilonShellForReflectingBoundary <= distToAbsorbingBoundary)
                {
                    starRadius = max(RADIUS_SHRINK_PERCENTAGE * starRadius,
                                     walkSettings.epsilonShellForReflectingBoundary);
                }
            }

            // sample a direction uniformly
            vector<float, DIM> direction = SphereSampler.sampleUnitSphereUniform(rng);

            // perform hemispherical sampling if on the reflecting boundary, which cancels
            // the alpha term in our integral expression
            if (state.onReflectingBoundary != 0 && dot(state.currentNormal, direction) > 0.0)
            {
                direction *= -1.0;
            }

            // check if there is an intersection with the reflecting boundary along the ray:
            // currentPt + starRadius * direction
            IntersectionPoint<DIM> intersectionPt;
            bool intersectedReflectingBoundary = geometricQueries.intersectReflectingBoundary(
                state.currentPt, state.currentNormal, direction, starRadius,
                state.onReflectingBoundary != 0, intersectionPt);

            // check if there is no intersection with the reflecting boundary
            if (!intersectedReflectingBoundary)
            {
                // apply small offset to the current pt for numerical robustness if it on
                // the reflecting boundary---the same offset is applied during ray intersections
                vector<float, DIM> currentPt = state.onReflectingBoundary != 0 ?
                                               geometricQueries.offsetPointAlongDirection(state.currentPt, -state.currentNormal) :
                                               state.currentPt;

                // set intersectionPt to a point on the spherical arc of the ball
                intersectionPt.pt = currentPt + starRadius * direction;
                intersectionPt.dist = starRadius;
            }

            // update the ball center and radius
            greensFn.updateBall(state.currentPt, starRadius);

            // compute the contribution from the reflecting boundary
            computeReflectingBoundaryContribution(starRadius, flipNormalOrientation, rng, greensFn, state);

            // compute the source contribution
            computeSourceContribution(direction, intersectionPt.dist, rng, greensFn, state);

            // update walk position, normal and throughput
            state.prevDistance = intersectionPt.dist;
            state.prevDirection = direction;
            state.currentPt = intersectionPt.pt;
            state.currentNormal = intersectionPt.normal; // NOTE: stale unless intersectedReflectingBoundary is true
            state.onReflectingBoundary = intersectedReflectingBoundary ? 1 : 0;
            state.throughput *= computeWalkStepThroughput(greensFn, state);

            // check if the current pt lies outside the domain; for interior problems,
            // this tests for walks that escape due to numerical error
            if (state.onReflectingBoundary == 0 && geometricQueries.outsideBoundingDomain(state.currentPt))
            {
                return WalkCompletionCode::EscapedDomain;
            }

            // use russian roulette to decide whether to terminate the walk
            bool terminateWalk = applyRussianRoulette(rng, state);
            if (terminateWalk)
            {
                return WalkCompletionCode::TerminatedWithRussianRoulette;
            }

            // update the walk length and break if the max walk length is exceeded
            state.walkLength++;
            if (state.walkLength >= walkSettings.maxWalkLength)
            {
                return WalkCompletionCode::ExceededMaxWalkLength;
            }

            // compute the distance to the absorbing boundary
            distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(state.currentPt, false);
            firstStep = false;
        }

        return WalkCompletionCode::ReachedAbsorbingBoundary;
    }

    // returns the terminal contribution from the end of the walk
    public T getTerminalContribution(WalkCompletionCode code,
                                     inout WalkState<T, DIM> state)
    {
        if (code == WalkCompletionCode::ReachedAbsorbingBoundary &&
            walkSettings.ignoreAbsorbingBoundaryContribution == 0)
        {
            // project the walk position to the absorbing boundary and grab the known boundary value
            float signedDistance = 0.0;
            bool solveDoubleSided = walkSettings.solveDoubleSided != 0;
            geometricQueries.projectToAbsorbingBoundary(state.currentPt, state.currentNormal,
                                                        signedDistance, solveDoubleSided);
            bool returnBoundaryNormalAlignedValue = solveDoubleSided && signedDistance > 0.0;
            return pde.dirichlet(state.currentPt, returnBoundaryNormalAlignedValue);
        }

        // return 0 terminal contribution if ignoring absorbing boundary values,
        // or if walk is terminated with russian roulette or exceeds max walk length
        return T(0.0);
    }

    // estimates only the solution of the given PDE at the input point
    public void estimateSolution(SamplePoint<DIM> samplePt,
                                 uint nWalks, inout pcg32 rng,
                                 inout SampleStatistics<T, DIM> statistics)
    {
        // for problems with double-sided boundary conditions, initialize the direction
        // of approach for walks, and flip the current normal orientation if the geometry
        // is front-facing
        vector<float, DIM> currentNormal = samplePt.normal;
        vector<float, DIM> prevDirection = samplePt.normal;
        bool flipNormalOrientation = false;

        if (walkSettings.solveDoubleSided != 0 && samplePt.type == SampleType::OnReflectingBoundary)
        {
            if (samplePt.estimateBoundaryNormalAligned != 0)
            {
                currentNormal *= -1.0;
                prevDirection *= -1.0;
                flipNormalOrientation = true;
            }
        }

        // perform random walks
        for (uint w = 0; w < nWalks; w++)
        {
            // initialize the walk state
            WalkState<T, DIM> state;
            state.currentPt = samplePt.pt;
            state.currentNormal = currentNormal;
            state.prevDirection = prevDirection;
            state.prevDistance = FLT_MAX;
            state.throughput = 1.0;
            state.walkLength = 0;
            state.onReflectingBoundary = samplePt.type == SampleType::OnReflectingBoundary ? 1 : 0;

            // initialize the greens function
            GreensFn greensFn;
            greensFn.updateAbsorptionCoeff(pde.absorptionCoeff());

            // perform walk
            WalkCompletionCode code = walk(samplePt.distToAbsorbingBoundary, flipNormalOrientation,
                                           rng, greensFn, state);

            if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                code == WalkCompletionCode::ExceededMaxWalkLength)
            {
                // compute the walk contribution
                T terminalContribution = getTerminalContribution(code, state);
                T totalContribution = terminalContribution.scale(state.throughput) +
                                      state.totalReflectingBoundaryContribution +
                                      state.totalSourceContribution;

                // update statistics
                statistics.addSolutionEstimate(totalContribution);
                statistics.addWalkLength(state.walkLength);
            }
        }
    }

    // estimates the solution and gradient of the given PDE at the input point;
    // NOTE: assumes the point does not lie on the boundary; the directional derivative
    // can be accessed through statistics.getEstimatedDerivative()
    public void estimateSolutionAndGradient(SamplePoint<DIM> samplePt,
                                            uint nWalks, inout pcg32 rng,
                                            inout SampleStatistics<T, DIM> statistics)
    {
        // reduce nWalks by 2 if using antithetic sampling
        uint nAntitheticIters = 1;
        if (walkSettings.useGradientAntitheticVariates != 0)
        {
            nWalks = max(1, nWalks / 2);
            nAntitheticIters = 2;
        }

        // perform random walks
        vector<float, DIM> zero = vector<float, DIM>(0.0);
        for (uint w = 0; w < nWalks; w++)
        {
            // initialize temporary variables for antithetic sampling
            float sourceRadius = 0.0, sourcePdf = 0.0, boundaryPdf = 0.0;
            vector<float, DIM> sourcePt = zero, boundaryPt = zero;
            pcg32 savedRng;

            // compute control variates for the gradient estimate
            T boundaryGradientControlVariate = T(0.0);
            T sourceGradientControlVariate = T(0.0);
            if (walkSettings.useGradientControlVariates != 0)
            {
                boundaryGradientControlVariate = statistics.getEstimatedSolution();
                sourceGradientControlVariate = statistics.getMeanFirstSourceContribution();
            }

            for (uint antitheticIter = 0; antitheticIter < nAntitheticIters; antitheticIter++)
            {
                // initialize the walk state
                WalkState<T, DIM> state;
                state.currentPt = samplePt.pt;

                // initialize the greens function
                GreensFn greensFn;
                greensFn.updateAbsorptionCoeff(pde.absorptionCoeff());
                float boundaryDist = min(samplePt.distToAbsorbingBoundary,
                                         samplePt.distToReflectingBoundary);
                greensFn.updateBall(state.currentPt, RADIUS_SHRINK_PERCENTAGE * boundaryDist);

                // compute the source contribution inside the ball
                T firstSourceContribution = T(0.0);
                vector<float, DIM> sourceGradientDirection = zero;
                if (walkSettings.ignoreSourceContribution == 0)
                {
                    float greensFnNorm = greensFn.norm();
                    if (pde.isSourceConstant())
                    {
                        firstSourceContribution = pde.source(state.currentPt).scale(greensFnNorm);
                    }
                    else
                    {
                        if (antitheticIter == 0)
                        {
                            vector<float, DIM> sourceDirection = SphereSampler.sampleUnitSphereUniform(rng);
                            sourcePt = greensFn.sampleVolume(sourceDirection, rng, sourceRadius, sourcePdf);
                        }
                        else
                        {
                            vector<float, DIM> sourceDirection = sourcePt - state.currentPt;
                            sourcePt = state.currentPt - sourceDirection;
                        }

                        firstSourceContribution = pde.source(sourcePt).scale(greensFnNorm);
                        sourceGradientDirection = greensFn.gradient(sourceRadius, sourcePt) / (sourcePdf * greensFnNorm);
                    }

                    state.totalSourceContribution = state.totalSourceContribution + firstSourceContribution.scale(state.throughput);
                }

                // sample a point uniformly on the sphere; update the current position
                // of the walk, its throughput and record the boundary gradient direction
                if (antitheticIter == 0)
                {
                    vector<float, DIM> boundaryDirection = SphereSampler.sampleUnitSphereUniform(rng);
                    boundaryPt = greensFn.getBallCenter() + greensFn.getBallRadius() * boundaryDirection;
                    boundaryPdf = SphereSampler.pdfSampleSphereUniform(1.0);
                }
                else
                {
                    vector<float, DIM> boundaryDirection = boundaryPt - state.currentPt;
                    boundaryPt = state.currentPt - boundaryDirection;
                }

                state.prevDistance = greensFn.getBallRadius();
                state.prevDirection = (boundaryPt - state.currentPt) / greensFn.getBallRadius();
                state.currentPt = boundaryPt;
                state.throughput *= greensFn.poissonKernel() / boundaryPdf;
                vector<float, DIM> boundaryGradientDirection = greensFn.poissonKernelGradient(boundaryPt) / (boundaryPdf * state.throughput);

                // compute the distance to the absorbing boundary
                float distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(state.currentPt, false);

                // save the rng so the antithetic walk replays the same random sequence;
                // NOTE: copy the full generator (state and stream) rather than calling
                // seed(), which would re-derive a different state and reset the stream
                if (antitheticIter == 0)
                {
                    savedRng = rng;
                }
                else
                {
                    rng = savedRng;
                }

                // perform walk
                WalkCompletionCode code = walk(distToAbsorbingBoundary, false, rng, greensFn, state);

                if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                    code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                    code == WalkCompletionCode::ExceededMaxWalkLength)
                {
                    // compute the walk contribution
                    T terminalContribution = getTerminalContribution(code, state);
                    T totalContribution = terminalContribution.scale(state.throughput) +
                                          state.totalReflectingBoundaryContribution +
                                          state.totalSourceContribution;

                    // compute the gradient contribution
                    T boundaryGradientEstimate[DIM];
                    T sourceGradientEstimate[DIM];
                    T boundaryContribution = totalContribution - firstSourceContribution;

                    for (int i = 0; i < DIM; i++)
                    {
                        boundaryGradientEstimate[i] = (boundaryContribution - boundaryGradientControlVariate).scale(boundaryGradientDirection[i]);
                        sourceGradientEstimate[i] = (firstSourceContribution - sourceGradientControlVariate).scale(sourceGradientDirection[i]);
                    }

                    // update statistics
                    statistics.addSolutionEstimate(totalContribution);
                    statistics.addFirstSourceContribution(firstSourceContribution);
                    statistics.addGradientEstimate(boundaryGradientEstimate, sourceGradientEstimate);
                    statistics.addWalkLength(state.walkLength);
                }
            }
        }
    }
};
