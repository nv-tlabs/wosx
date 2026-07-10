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

// This file provides various convenience structs and enums to specify settings
// and extract outputs from the algorithms implemented in WoSX. Specifically, in
// addition to the PDE and GeometricQueries interfaces, the user should populate, via
// their constructors, the WalkSettings and SamplePoint structs for the walk-on-spheres
// and walk-on-stars algorithms. For each SamplePoint where the PDE is to be solved,
// the SampleStatistics struct can be used to query the estimated solution, gradient,
// and other statistics.

implementing wosx;
__include core.pde;
__include core.geometric_queries;

internal static const float RADIUS_SHRINK_PERCENTAGE = 0.99;

public struct WalkSettings
{
    public float epsilonShellForAbsorbingBoundary;
    public float epsilonShellForReflectingBoundary;
    public float silhouettePrecision;
    public float russianRouletteThreshold;
    public uint maxWalkLength;
    public uint stepsBeforeUsingMaximalSpheres;
    public uint solveDoubleSided; // NOTE: this flag should be set to 1 if the domain is not watertight
    public uint useGradientControlVariates;
    public uint useGradientAntitheticVariates;
    public uint ignoreAbsorbingBoundaryContribution;
    public uint ignoreReflectingBoundaryContribution;
    public uint ignoreSourceContribution;

    // constructor
    public __init()
    {
        epsilonShellForAbsorbingBoundary = 1e-3;
        epsilonShellForReflectingBoundary = 1e-3;
        silhouettePrecision = 1e-3;
        russianRouletteThreshold = 0.0;
        maxWalkLength = 10000;
        stepsBeforeUsingMaximalSpheres = 10000;
        solveDoubleSided = 0;
        useGradientControlVariates = 1;
        useGradientAntitheticVariates = 1;
        ignoreAbsorbingBoundaryContribution = 0;
        ignoreReflectingBoundaryContribution = 0;
        ignoreSourceContribution = 0;
    }
};

public struct WalkState<T, let DIM : int>
    where T : IFloat
{
    public T totalReflectingBoundaryContribution;
    public T totalSourceContribution;
    public vector<float, DIM> currentPt;
    public vector<float, DIM> currentNormal;
    public vector<float, DIM> prevDirection;
    public float prevDistance;
    public float throughput;
    public uint walkLength;
    public uint onReflectingBoundary;

    // constructor
    public __init()
    {
        totalReflectingBoundaryContribution = T(0.0);
        totalSourceContribution = T(0.0);
        currentPt = vector<float, DIM>(0.0);
        currentNormal = vector<float, DIM>(0.0);
        prevDirection = vector<float, DIM>(0.0);
        prevDistance = 0.0;
        throughput = 1.0;
        walkLength = 0;
        onReflectingBoundary = 0;
    }
};

public enum WalkCompletionCode
{
    ReachedAbsorbingBoundary,
    TerminatedWithRussianRoulette,
    ExceededMaxWalkLength,
    EscapedDomain
}

public enum SampleType
{
    InDomain, // applies to both interior and exterior sample points for watertight domains
    OnAbsorbingBoundary,
    OnReflectingBoundary
}

public enum EstimationQuantity
{
    Solution,
    SolutionAndGradient,
    Skip
}

public struct SamplePoint<let DIM : int>
{
    public vector<float, DIM> pt;
    public vector<float, DIM> normal;
    public SampleType type;
    public EstimationQuantity estimationQuantity;
    public float pdf;
    public float distToAbsorbingBoundary;
    public float distToReflectingBoundary;
    public uint estimateBoundaryNormalAligned;

    // constructor
    public __init()
    {
        pt = vector<float, DIM>(0.0);
        normal = vector<float, DIM>(0.0);
        type = SampleType::InDomain;
        estimationQuantity = EstimationQuantity::Solution;
        pdf = 1.0;
        distToAbsorbingBoundary = 0.0;
        distToReflectingBoundary = 0.0;
        estimateBoundaryNormalAligned = 0;
    }
};

public struct SampleStatistics<T, let DIM : int>
    where T : IFloat
{
    private T solutionMean;
    private T gradientMean[DIM];
    private T totalFirstSourceContribution;
    private uint nSolutionEstimates;
    private uint nGradientEstimates;
    private uint totalWalkLength;
    private uint _padding;

    // constructor
    public __init()
    {
        reset();
    }

    // resets statistics
    [mutating]
    public void reset()
    {
        solutionMean = T(0.0);
        for (int i = 0; i < DIM; i++)
        {
            gradientMean[i] = T(0.0);
        }
        totalFirstSourceContribution = T(0.0);
        nSolutionEstimates = 0;
        nGradientEstimates = 0;
        totalWalkLength = 0;
    }

    // adds solution estimate to running sum
    [mutating]
    public void addSolutionEstimate(T estimate)
    {
        nSolutionEstimates += 1;
        update(estimate, solutionMean, nSolutionEstimates);
    }

    // adds gradient estimate to running sum
    [mutating]
    public void addGradientEstimate(T boundaryEstimate[DIM], T sourceEstimate[DIM])
    {
        nGradientEstimates += 1;
        for (int i = 0; i < DIM; i++)
        {
            update(boundaryEstimate[i] + sourceEstimate[i],
                   gradientMean[i], nGradientEstimates);
        }
    }

    // adds gradient estimate to running sum
    [mutating]
    public void addGradientEstimate(T estimate[DIM])
    {
        nGradientEstimates += 1;
        for (int i = 0; i < DIM; i++)
        {
            update(estimate[i], gradientMean[i], nGradientEstimates);
        }
    }

    // adds source contribution for the first step to running sum
    [mutating]
    public void addFirstSourceContribution(T contribution)
    {
        totalFirstSourceContribution = totalFirstSourceContribution + contribution;
    }

    // adds walk length to running sum
    [mutating]
    public void addWalkLength(uint walkLength)
    {
        totalWalkLength += walkLength;
    }

    // merges two statistics
    [mutating]
    public void merge(SampleStatistics<T, DIM> other)
    {
        if (other.nSolutionEstimates > 0)
        {
            uint N1 = nSolutionEstimates;
            uint N2 = other.nSolutionEstimates;
            uint N = N1 + N2;
            if (N1 == 0)
            {
                solutionMean = other.solutionMean;
            }
            else
            {
                T delta = other.solutionMean - solutionMean;
                solutionMean = solutionMean + delta.scale((float)N2 / N);
            }
            nSolutionEstimates = N;
        }

        if (other.nGradientEstimates > 0)
        {
            uint N1 = nGradientEstimates;
            uint N2 = other.nGradientEstimates;
            uint N = N1 + N2;
            for (int i = 0; i < DIM; i++)
            {
                if (N1 == 0)
                {
                    gradientMean[i] = other.gradientMean[i];
                }
                else
                {
                    T delta = other.gradientMean[i] - gradientMean[i];
                    gradientMean[i] = gradientMean[i] + delta.scale((float)N2 / N);
                }
            }
            nGradientEstimates = N;
        }

        totalFirstSourceContribution = totalFirstSourceContribution + other.totalFirstSourceContribution;
        totalWalkLength += other.totalWalkLength;
    }

    // returns estimated solution
    public T getEstimatedSolution()
    {
        return solutionMean;
    }

    // returns estimated gradient
    public void getEstimatedGradient(out T gradient[DIM])
    {
        for (int i = 0; i < DIM; i++)
        {
            gradient[i] = gradientMean[i];
        }
    }

    // returns mean source contribution for the first step
    public T getMeanFirstSourceContribution()
    {
        uint N = max(1, nSolutionEstimates);
        return totalFirstSourceContribution.scale(1.0 / N);
    }

    // returns estimated derivative
    public T getEstimatedDerivative(vector<float, DIM> direction)
    {
        T derivative = T(0.0);
        for (int i = 0; i < DIM; i++)
        {
            derivative = derivative + gradientMean[i].scale(direction[i]);
        }

        return derivative;
    }

    // returns number of solution estimates
    public uint getSolutionEstimateCount()
    {
        return nSolutionEstimates;
    }

    // returns number of gradient estimates
    public uint getGradientEstimateCount()
    {
        return nGradientEstimates;
    }

    // returns mean walk length
    public float getMeanWalkLength()
    {
        uint N = max(1, nSolutionEstimates);
        return (float)totalWalkLength / N;
    }

    // updates statistics
    private void update(T estimate, inout T mean, uint N)
    {
        T delta = estimate - mean;
        mean = mean + delta.scale(1.0 / N);
    }
};
