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

// This file implements a dense sdf grid by extending the IAbsorbingBoundaryGeometricQueries interface.

implementing wosx;

public interface ISdfGrid<let DIM : int>
{
    // returns the grid origin
    vector<float, DIM> getOrigin();

    // returns the grid extent
    vector<float, DIM> getExtent();

    // returns the cell spacing
    vector<float, DIM> getCellSpacing();

    // returns data at the given point
    float operator()(vector<float, DIM> x);
};

public struct SdfGrid2D : ISdfGrid<2>
{
    private Texture2D<float> data;
    private SamplerState sampler;
    private float2 origin;
    private float2 extent;
    private float2 cellSpacing;

    // returns the grid origin
    public float2 getOrigin()
    {
        return origin;
    }

    // returns the grid extent
    public float2 getExtent()
    {
        return extent;
    }

    // returns the grid cell spacing
    public float2 getCellSpacing()
    {
        return cellSpacing;
    }

    // returns data at the given point
    public float operator()(float2 x)
    {
        float2 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.yx, 0);
    }
};

public struct SdfGrid3D : ISdfGrid<3>
{
    private Texture3D<float> data;
    private SamplerState sampler;
    private float3 origin;
    private float3 extent;
    private float3 cellSpacing;

    // returns the grid origin
    public float3 getOrigin()
    {
        return origin;
    }

    // returns the grid extent
    public float3 getExtent()
    {
        return extent;
    }

    // returns the grid cell spacing
    public float3 getCellSpacing()
    {
        return cellSpacing;
    }

    // returns data at the given point
    public float operator()(float3 x)
    {
        float3 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.zyx, 0);
    }
};

public struct SdfAbsorbingBoundaryGeometricQueries<SdfGrid, let DIM : int> :
                                                   IAbsorbingBoundaryGeometricQueries<DIM>
    where SdfGrid : ISdfGrid<DIM>
{
    private SdfGrid sdfGrid;

    // clamps the given point to the grid bounds
    private vector<float, DIM> clampInGrid(vector<float, DIM> x)
    {
        vector<float, DIM> xClamped = x;
        vector<float, DIM> origin = sdfGrid.getOrigin();
        vector<float, DIM> extent = sdfGrid.getExtent();
        for (int i = 0; i < DIM; i++)
        {
            float gridMin = origin[i] + 1e-6;
            float gridMax = origin[i] + extent[i] - 1e-6;
            xClamped[i] = clamp(xClamped[i], gridMin, gridMax);
        }

        return xClamped;
    }

    // computes the gradient of the sdf field at the given point
    private vector<float, DIM> computeGradient(vector<float, DIM> x)
    {
        // perform central differencing
        vector<float, DIM> gradient = vector<float, DIM>(0.0);
        vector<float, DIM> cellSpacing = sdfGrid.getCellSpacing();
        for (int i = 0; i < DIM; i++)
        {
            vector<float, DIM> offset = vector<float, DIM>(0.0);
            offset[i] = cellSpacing[i];
            vector<float, DIM> xMinus = clampInGrid(x - offset);
            vector<float, DIM> xPlus = clampInGrid(x + offset);
            float sdfMinus = sdfGrid(xMinus);
            float sdfPlus = sdfGrid(xPlus);
            // divide by the actual sample separation, which collapses to a one-sided
            // difference (spacing instead of 2*spacing) when clamping near a grid face
            float denom = xPlus[i] - xMinus[i];
            gradient[i] = denom > 0.0 ? (sdfPlus - sdfMinus) / denom : 0.0;
        }

        return gradient;
    }

    // computes the normal of the level set at the given point
    private vector<float, DIM> computeNormal(vector<float, DIM> x)
    {
        vector<float, DIM> dSdf = computeGradient(x);
        dSdf /= (length(dSdf) + 1e-6);

        return dSdf;
    }

    // projects the given point to the zero level set
    private void projectToZeroLevelSet(inout vector<float, DIM> x,
                                       out vector<float, DIM> normal,
                                       int maxIterations = 8,
                                       float epsilon = 1e-5)
    {
        // apply steepest descent to project x to the zero level set
        for (int i = 0; i < maxIterations; i++)
        {
            float sdf = sdfGrid(x);
            if (abs(sdf) < epsilon) break;
            vector<float, DIM> dSdf = computeGradient(x);
            vector<float, DIM> step = dSdf * sdf / (length(dSdf) + 1e-6);
            x = clampInGrid(x - step);
        }

        normal = computeNormal(x);
    }

    // intersects the zero level set with the given ray
    public bool intersectZeroLevelSet(vector<float, DIM> origin,
                                      vector<float, DIM> dir,
                                      float tMax,
                                      out IntersectionPoint<DIM> intersectionPt,
                                      int maxIterations = 128,
                                      float epsilon = 1e-5)
    {
        // first determine whether the ray intersects the grid bounds
        vector<float, DIM> bMin = sdfGrid.getOrigin();
        vector<float, DIM> bMax = sdfGrid.getOrigin() + sdfGrid.getExtent();
        vector<float, DIM> invDir = 1.0 / dir;
        vector<float, DIM> t0 = (bMin - origin) * invDir;
        vector<float, DIM> t1 = (bMax - origin) * invDir;
        vector<float, DIM> tNear = min(t0, t1);
        vector<float, DIM> tFar = max(t0, t1);
        float tEnter = 0.0;
        float tExit = tMax;
        for (int i = 0; i < DIM; i++)
        {
            tEnter = max(tEnter, tNear[i]);
            tExit = min(tExit, tFar[i]);
        }
        if (tEnter > tExit)
        {
            intersectionPt = IntersectionPoint<DIM>();
            return false; // ray misses the grid entirely
        }

        // sphere trace sdf field
        float tPrev = tEnter;
        float sdfPrev = 0.0;
        float t = tPrev;

        for (int i = 0; i < maxIterations && t <= tExit; i++)
        {
            vector<float, DIM> p = origin + dir * t;
            float sdf = sdfGrid(p);

            // check if the ray intersects the zero level set
            if (abs(sdf) < epsilon)
            {
                intersectionPt.pt = p;
                intersectionPt.normal = computeNormal(p);
                intersectionPt.dist = t;
                return true;
            }

            // perform a single secant refinement step if sign of sdf changes
            if (sdfPrev * sdf < 0.0)
            {
                float alpha = sdfPrev / (sdfPrev - sdf);
                float tHit = tPrev + alpha * (t - tPrev);
                vector<float, DIM> pHit = origin + dir * tHit;
                float sdfHit = sdfGrid(pHit);
                vector<float, DIM> dSdfHit = computeGradient(pHit);
                vector<float, DIM> step = dSdfHit * sdfHit / (length(dSdfHit) + 1e-6);
                pHit -= step;

                intersectionPt.pt = pHit;
                intersectionPt.normal = computeNormal(pHit);
                intersectionPt.dist = dot(pHit - origin, dir);
                return true;
            }

            // step forward
            tPrev = t;
            sdfPrev = sdf;
            t += abs(sdf);
        }

        intersectionPt = IntersectionPoint<DIM>();
        return false;
    }

    // computes the distance to the boundary
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        return computeSignedDistance ? sdfGrid(x) : abs(sdfGrid(x));
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        distance = computeSignedDistance ? sdfGrid(x) : abs(sdfGrid(x));
        projectToZeroLevelSet(x, normal);

        return true;
    }

    // intersects a ray with the boundary
    public bool intersectBoundary(vector<float, DIM> origin,
                                  vector<float, DIM> normal,
                                  vector<float, DIM> dir,
                                  float tMax,
                                  bool onBoundary,
                                  out IntersectionPoint<DIM> intersectionPt)
    {
        return intersectZeroLevelSet(origin, dir, tMax, intersectionPt);
    }
};
