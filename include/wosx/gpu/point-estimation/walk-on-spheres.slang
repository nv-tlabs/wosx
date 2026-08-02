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

// This file implements the Monte Carlo walk-on-spheres algorithm for solving
// Poisson and screened Poisson equations with Dirichlet boundary conditions.
// The estimated PDE solution, gradient, and other statistics can be queried
// from the SampleStatistics struct, which is stored in the SamplePoint struct.
// Each call to solve(...) improves the estimate of the PDE solution at the
// user-selected set of input points, enabling progressive evaluation.
//
// Resources:
// - Some Continuous Monte Carlo Methods for the Dirichlet Problem [1956]
// - Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains [2020]

implementing wosx;
__include point_estimation.common;

public struct WalkOnSpheres<T, PDE, GeometricQueries, let DIM : int>
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

    // computes the source contribution at a particular point in the walk
    public void computeSourceContribution(inout pcg32 rng,
                                          inout GreensFn greensFn,
                                          inout WalkState<T, DIM> state)
    {
        if (walkSettings.ignoreSourceContribution == 0)
        {
            // compute source contribution inside the sphere
            T sourceContribution;
            if (pde.isSourceConstant())
            {
                sourceContribution = pde.source(state.currentPt).scale(greensFn.norm());
            }
            else
            {
                float sourceRadius, sourcePdf;
                vector<float, DIM> sourcePt = greensFn.sampleVolume(rng, sourceRadius, sourcePdf);
                sourceContribution = pde.source(sourcePt).scale(greensFn.norm());
            }

            state.totalSourceContribution = state.totalSourceContribution + sourceContribution.scale(state.throughput);
        }
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

    // performs a single random walk starting at the input point
    public WalkCompletionCode walk(float distToAbsorbingBoundary,
                                   inout pcg32 rng,
                                   inout GreensFn greensFn,
                                   inout WalkState<T, DIM> state)
    {
        // recursively perform a random walk till it reaches the absorbing boundary
        while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary)
        {
            // update the ball center and radius
            greensFn.updateBall(state.currentPt, distToAbsorbingBoundary);

            // compute the source contribution
            computeSourceContribution(rng, greensFn, state);

            // sample a direction uniformly
            vector<float, DIM> direction = SphereSampler.sampleUnitSphereUniform(rng);

            // update walk position and throughput
            state.currentPt = state.currentPt + distToAbsorbingBoundary * direction;
            state.throughput *= greensFn.directionSampledPoissonKernel(state.currentPt);

            // check if the current pt lies outside the domain; for interior problems,
            // this tests for walks that escape due to numerical error
            if (geometricQueries.outsideBoundingDomain(state.currentPt))
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

            // compute the distance to the absorbing boundary; the walk stepped a distance
            // equal to the ball radius, so by the triangle inequality the new distance is
            // at most twice that radius---supplying this bound prunes the query traversal
            float distBound = RADIUS_EXPAND_PERCENTAGE * 2.0 * distToAbsorbingBoundary;
            distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(state.currentPt, false, distBound);
        }

        return WalkCompletionCode::ReachedAbsorbingBoundary;
    }

    // returns the terminal contribution from the end of the walk
    public T getTerminalContribution(WalkCompletionCode code,
                                     inout WalkState<T, DIM> state)
    {
        if ((code == WalkCompletionCode::ReachedAbsorbingBoundary ||
             code == WalkCompletionCode::ExceededMaxWalkLength) &&
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
        // or if walk is terminated with russian roulette
        return T(0.0);
    }

    // estimates only the solution of the given PDE at the input point
    public void estimateSolution(SamplePoint<DIM> samplePt,
                                 uint nWalks, inout pcg32 rng,
                                 inout SampleStatistics<T, DIM> statistics)
    {
        // perform random walks
        for (uint w = 0; w < nWalks; w++)
        {
            // initialize the walk state
            WalkState<T, DIM> state;
            state.currentPt = samplePt.pt;

            // initialize the greens function
            GreensFn greensFn;
            greensFn.updateAbsorptionCoeff(pde.absorptionCoeff());

            // perform walk
            WalkCompletionCode code = walk(samplePt.distToAbsorbingBoundary, rng, greensFn, state);

            if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                code == WalkCompletionCode::ExceededMaxWalkLength)
            {
                // compute the walk contribution
                T terminalContribution = getTerminalContribution(code, state);
                T totalContribution = terminalContribution.scale(state.throughput) +
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
                greensFn.updateBall(state.currentPt, RADIUS_SHRINK_PERCENTAGE * samplePt.distToAbsorbingBoundary);

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

                state.currentPt = boundaryPt;
                state.throughput *= greensFn.poissonKernel() / boundaryPdf;
                vector<float, DIM> boundaryGradientDirection = greensFn.poissonKernelGradient(boundaryPt) / (boundaryPdf * state.throughput);

                // compute the distance to the absorbing boundary; the first step moved the
                // ball radius away from the sample point, which bounds the new distance
                float distBound = RADIUS_EXPAND_PERCENTAGE * 2.0 * samplePt.distToAbsorbingBoundary;
                float distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(state.currentPt, false, distBound);

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
                WalkCompletionCode code = walk(distToAbsorbingBoundary, rng, greensFn, state);

                if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                    code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                    code == WalkCompletionCode::ExceededMaxWalkLength)
                {
                    // compute the walk contribution
                    T terminalContribution = getTerminalContribution(code, state);
                    T totalContribution = terminalContribution.scale(state.throughput) + state.totalSourceContribution;

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
