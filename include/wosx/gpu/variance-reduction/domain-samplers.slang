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

// This file defines a domain sampler for generating uniformly distributed sample points
// in a 2D or 3D domain. These sample points are required by the Boundary Value Caching (BVC)
// algorithm for reducing variance of the walk-on-spheres and walk-on-stars estimators for
// PDEs with non-zero source.

implementing wosx;
__include point_estimation.common;

public interface IDomainSampler<let DIM : int>
{
    // generates a uniformly distributed sample point inside the domain
    void generateSample(inout pcg32 rng,
                        inout SamplePoint<DIM> samplePt);
};

public struct EmptyDomainSampler<let DIM : int> : IDomainSampler<DIM>
{
    // generates a uniformly distributed sample point in the domain
    public void generateSample(inout pcg32 rng,
                               inout SamplePoint<DIM> samplePt)
    {
        // do nothing
    }
};

public interface ISolveRegion<let DIM : int>
{
    // returns region min
    vector<float, DIM> getRegionMin();

    // returns region max
    vector<float, DIM> getRegionMax();

    // returns region volume
    float getRegionVolume();

    // returns true if the input point is inside the solve region
    bool isInside(vector<float, DIM> x);
};

public struct UniformDomainSampler<SolveRegion, GeometricQueries, let DIM : int> : IDomainSampler<DIM>
    where SolveRegion : ISolveRegion<DIM>
    where GeometricQueries : IGeometricQueries<DIM>
{
    private SolveRegion solveRegion;
    private GeometricQueries geometricQueries;

    // bounds on the rejection sampling iteration budget: the lower bound
    // covers high-occupancy solve regions, while the upper bound prevents
    // spinning indefinitely on degenerate solve regions (e.g. zero volume)
    static const int minRejectionIterations = 64;
    static const int maxRejectionIterations = 65536;

    // generates a uniformly distributed sample point in the domain
    public void generateSample(inout pcg32 rng,
                               inout SamplePoint<DIM> samplePt)
    {
        // rejection sample a point inside the solve region; samples that fail to
        // land inside the region within the iteration budget are marked as skipped,
        // and should contribute zero to downstream estimates
        vector<float, DIM> regionExtent = solveRegion.getRegionMax() - solveRegion.getRegionMin();
        float boxVolume = 1.0;
        for (int i = 0; i < DIM; i++)
        {
            boxVolume *= regionExtent[i];
        }

        // each iteration accepts a candidate with probability regionVolume / boxVolume,
        // so a budget set to a generous multiple of the expected number of iterations
        // makes skips vanishingly rare for valid solve regions of any occupancy
        float expectedIterations = boxVolume / solveRegion.getRegionVolume();
        int nIterations = int(clamp(32.0 * expectedIterations,
                                    float(minRejectionIterations),
                                    float(maxRejectionIterations)));
        bool foundSample = false;

        for (int iter = 0; iter < nIterations; iter++)
        {
            vector<float, DIM> randNums;
            for (int i = 0; i < DIM; i++)
            {
                randNums[i] = rng.nextFloat();
            }

            vector<float, DIM> pt = solveRegion.getRegionMin() + regionExtent * randNums;
            samplePt.pt = pt;
            if (solveRegion.isInside(pt))
            {
                foundSample = true;
                break;
            }
        }

        samplePt.type = SampleType::InDomain;
        samplePt.estimationQuantity = foundSample ? EstimationQuantity::Solution :
                                                    EstimationQuantity::Skip;
        samplePt.pdf = 1.0 / solveRegion.getRegionVolume();
        samplePt.distToAbsorbingBoundary = geometricQueries.computeDistToAbsorbingBoundary(samplePt.pt, false);
        samplePt.distToReflectingBoundary = geometricQueries.computeDistToReflectingBoundary(samplePt.pt, false);
    }
};

public struct BoundingBoxSolveRegion<let DIM : int> : ISolveRegion<DIM>
{
    private vector<float, DIM> regionMin;
    private vector<float, DIM> regionMax;

    // returns region min
    public vector<float, DIM> getRegionMin()
    {
        return regionMin;
    }

    // returns region max
    public vector<float, DIM> getRegionMax()
    {
        return regionMax;
    }

    // returns region volume
    public float getRegionVolume()
    {
        float volume = 1.0;
        for (int i = 0; i < DIM; i++)
        {
            volume *= (regionMax[i] - regionMin[i]);
        }

        return volume;
    }

    // returns true if the input point is inside the solve region
    public bool isInside(vector<float, DIM> x)
    {
        for (int i = 0; i < DIM; i++)
        {
            if (x[i] < regionMin[i] || x[i] > regionMax[i])
            {
                return false;
            }
        }

        return true;
    }
};

public struct WatertightDomainSolveRegion<GeometricQueries, let DIM : int> : ISolveRegion<DIM>
    where GeometricQueries : IGeometricQueries<DIM>
{
    private GeometricQueries geometricQueries;
    private vector<float, DIM> regionMin;
    private vector<float, DIM> regionMax;
    private float regionVolume;

    // returns region min
    public vector<float, DIM> getRegionMin()
    {
        return regionMin;
    }

    // returns region max
    public vector<float, DIM> getRegionMax()
    {
        return regionMax;
    }

    // returns region volume
    public float getRegionVolume()
    {
        return regionVolume;
    }

    // returns true if the input point is inside the solve region
    public bool isInside(vector<float, DIM> x)
    {
        return geometricQueries.insideDomain(x);
    }
};
