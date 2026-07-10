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

// This file implements reflectance bounds for Robin boundary conditions.
// Users of WoSX need not interact with this file directly.

implementing wosx;
__include utils.reflectance_boundary_bvh.bvh;

public struct RobinLineSegmentBound : IReflectancePrimitiveBound
{
    // computes the squared star radius bound
    public static void computeSquaredStarRadiusBound(inout BoundingSphere s, float maxRobinCoeff,
                                                     float3 viewDirClosest, float closestPtDist,
                                                     float3 viewDirFarthest, float farthestPtDist,
                                                     float3 planePt, float3 planeNormal)
    {
        float3 planeClosestPt;
        float h = findClosestPointPlane(planePt, planeNormal, s.c, planeClosestPt);
        if (h > 0.0)
        {
            float cosUpperBound = abs(dot(viewDirClosest / closestPtDist, planeNormal));
            float cosLowerBound = abs(dot(viewDirFarthest / farthestPtDist, planeNormal));
            float cosLine = sqrt(h * maxRobinCoeff) / SQRT_2;
            float cosLineSegment = clamp(cosLine, cosLowerBound, cosUpperBound);
            float cosLineSegment2 = cosLineSegment * cosLineSegment;
            float lineSegmentRadius = (h / cosLineSegment) * exp(cosLineSegment2 / (h * maxRobinCoeff));
            float lineSegmentRadius2 = lineSegmentRadius * lineSegmentRadius;
            s.r2 = min(s.r2, lineSegmentRadius2);
        }
        else
        {
            // the query point lies on the segment's supporting line, where the radius bound
            // evaluates to 0/0: apply its limit as h -> 0, the distance to the segment itself
            s.r2 = min(s.r2, closestPtDist * closestPtDist);
        }
    }
};

public struct RobinTriangleBound : IReflectancePrimitiveBound
{
    // computes the squared star radius bound
    public static void computeSquaredStarRadiusBound(inout BoundingSphere s, float maxRobinCoeff,
                                                     float3 viewDirClosest, float closestPtDist,
                                                     float3 viewDirFarthest, float farthestPtDist,
                                                     float3 planePt, float3 planeNormal)
    {
        float3 planeClosestPt;
        float h = findClosestPointPlane(planePt, planeNormal, s.c, planeClosestPt);
        if (h > 0.0)
        {
            float cosUpperBound = abs(dot(viewDirClosest / closestPtDist, planeNormal));
            float cosLowerBound = abs(dot(viewDirFarthest / farthestPtDist, planeNormal));
            float maxCosForBound = sqrt(h * maxRobinCoeff);
            if (maxCosForBound >= cosLowerBound)
            {
                float cosPlane = maxCosForBound / SQRT_3;
                float cosTriangle = clamp(cosPlane, cosLowerBound, cosUpperBound);
                float cosTriangle2 = cosTriangle * cosTriangle;
                float triangleRadius = h * h * maxRobinCoeff / (cosTriangle * (h * maxRobinCoeff - cosTriangle2));
                float triangleRadius2 = triangleRadius * triangleRadius;
                s.r2 = min(s.r2, triangleRadius2);
            }
        }
        else
        {
            // the query point lies in the triangle's supporting plane, where the radius bound
            // evaluates to 0/0: apply its limit as h -> 0, the distance to the triangle itself
            s.r2 = min(s.r2, closestPtDist * closestPtDist);
        }
    }
};

public struct RobinBvhNodeBound2D : IReflectanceBvhNodeBound
{
    // computes the minimum squared star radius bound
    public static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                         float minRobinCoeff, float maxRobinCoeff,
                                                         float minCosTheta, float maxCosTheta)
    {
        float rBound = rMin * exp(minCosTheta / (maxRobinCoeff * rMax));
        return rBound * rBound;
    }

    // computes the maximum squared star radius bound
    public static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                         float minRobinCoeff, float maxRobinCoeff,
                                                         float minCosTheta, float maxCosTheta)
    {
        float rBound = rMax * exp(maxCosTheta / (minRobinCoeff * rMin));
        return rBound * rBound;
    }
};

public struct RobinBvhNodeBound3D : IReflectanceBvhNodeBound
{
    // computes the minimum squared star radius bound
    public static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                         float minRobinCoeff, float maxRobinCoeff,
                                                         float minCosTheta, float maxCosTheta)
    {
        if (rMax < minCosTheta / maxRobinCoeff)
        {
            return FLT_MAX;
        }

        float rBound = rMin / (1.0 - (minCosTheta / (maxRobinCoeff * rMax)));
        return rBound * rBound;
    }

    // computes the maximum squared star radius bound
    public static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                         float minRobinCoeff, float maxRobinCoeff,
                                                         float minCosTheta, float maxCosTheta)
    {
        if (rMin < maxCosTheta / minRobinCoeff)
        {
            return FLT_MAX;
        }

        float rBound = rMax / (1.0 - (maxCosTheta / (minRobinCoeff * rMin)));
        return rBound * rBound;
    }
};
