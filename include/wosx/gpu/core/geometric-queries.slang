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

// This file defines an interface for performing geometric queries on a domain boundary,
// such as computing distances to the boundary, projecting points to the boundary, and
// intersecting rays with the boundary. As part of the problem setup, users of WoSX
// should populate the callback functions defined by the IAbsorbingBoundaryGeometricQueries
// and IReflectingBoundaryGeometricQueries interfaces for the boundary representation used
// in their application.
//
// For surface meshes in 2D and 3D, the fcpw-geometric-queries.slang file provides
// implementations of these interfaces.

implementing wosx;
__include core.math_helpers;

public struct IntersectionPoint<let DIM : int>
{
    public vector<float, DIM> pt;
    public vector<float, DIM> normal;
    public float dist;

    // constructor
    public __init()
    {
        pt = vector<float, DIM>(0.0);
        normal = vector<float, DIM>(0.0);
        dist = FLT_MAX;
    }
};

public struct BoundarySample<let DIM : int>
{
    public vector<float, DIM> pt;
    public vector<float, DIM> normal;
    public float pdf;

    // constructor
    public __init()
    {
        pt = vector<float, DIM>(0.0);
        normal = vector<float, DIM>(0.0);
        pdf = 0.0;
    }
};

public interface IAbsorbingBoundaryGeometricQueries<let DIM : int>
{
    // computes the distance to the boundary
    float computeDistToBoundary(vector<float, DIM> x,
                                bool computeSignedDistance,
                                float maxDistance = FLT_MAX);

    // projects a point to the boundary
    bool projectToBoundary(inout vector<float, DIM> x,
                           out vector<float, DIM> normal,
                           out float distance,
                           bool computeSignedDistance);

    // intersects a ray with the boundary
    bool intersectBoundary(vector<float, DIM> origin,
                           vector<float, DIM> normal,
                           vector<float, DIM> dir,
                           float tMax,
                           bool onBoundary,
                           out IntersectionPoint<DIM> intersectionPt);
};

public interface IReflectingBoundaryGeometricQueries<let DIM : int>
{
    // computes the distance to the boundary
    float computeDistToBoundary(vector<float, DIM> x,
                                bool computeSignedDistance,
                                float maxDistance = FLT_MAX);

    // projects a point to the boundary
    bool projectToBoundary(inout vector<float, DIM> x,
                           out vector<float, DIM> normal,
                           out float distance,
                           bool computeSignedDistance);

    // intersects a ray with the boundary
    bool intersectBoundary(vector<float, DIM> origin,
                           vector<float, DIM> normal,
                           vector<float, DIM> dir,
                           float tMax,
                           bool onBoundary,
                           out IntersectionPoint<DIM> intersectionPt);

    // checks whether there is a line of sight between two points
    bool intersectsWithBoundary(vector<float, DIM> xi,
                                vector<float, DIM> xj,
                                vector<float, DIM> ni,
                                vector<float, DIM> nj,
                                bool offseti,
                                bool offsetj);

    // samples a point on the boundary
    bool sampleBoundary(vector<float, DIM> x,
                        float radius,
                        vector<float, DIM> randNums,
                        out BoundarySample<DIM> boundarySample);

    // computes the radius of a star-shaped region on the boundary
    float computeStarRadiusForBoundary(vector<float, DIM> x,
                                       float minRadius,
                                       float maxRadius,
                                       float silhouettePrecision,
                                       bool flipNormalOrientation);
};

public interface IGeometricQueries<let DIM : int>
{
    // computes the distance to the boundary
    float computeDistToAbsorbingBoundary(vector<float, DIM> x,
                                         bool computeSignedDistance,
                                         float maxDistance = FLT_MAX);
    float computeDistToReflectingBoundary(vector<float, DIM> x,
                                          bool computeSignedDistance,
                                          float maxDistance = FLT_MAX);
    float computeDistToBoundary(vector<float, DIM> x,
                                bool computeSignedDistance);

    // projects a point to the boundary
    bool projectToAbsorbingBoundary(inout vector<float, DIM> x,
                                    out vector<float, DIM> normal,
                                    out float distance,
                                    bool computeSignedDistance);
    bool projectToReflectingBoundary(inout vector<float, DIM> x,
                                     out vector<float, DIM> normal,
                                     out float distance,
                                     bool computeSignedDistance);
    bool projectToBoundary(inout vector<float, DIM> x,
                           out vector<float, DIM> normal,
                           out float distance,
                           bool computeSignedDistance);

    // offsets a point along a direction
    vector<float, DIM> offsetPointAlongDirection(vector<float, DIM> x,
                                                 vector<float, DIM> dir);

    // intersects a ray with the boundary
    bool intersectAbsorbingBoundary(vector<float, DIM> origin,
                                    vector<float, DIM> normal,
                                    vector<float, DIM> dir,
                                    float tMax,
                                    bool onAbsorbingBoundary,
                                    out IntersectionPoint<DIM> intersectionPt);
    bool intersectReflectingBoundary(vector<float, DIM> origin,
                                     vector<float, DIM> normal,
                                     vector<float, DIM> dir,
                                     float tMax,
                                     bool onReflectingBoundary,
                                     out IntersectionPoint<DIM> intersectionPt);
    bool intersectBoundary(vector<float, DIM> origin,
                           vector<float, DIM> normal,
                           vector<float, DIM> dir,
                           float tMax,
                           bool onAbsorbingBoundary,
                           bool onReflectingBoundary,
                           out IntersectionPoint<DIM> intersectionPt);

    // checks whether there is a line of sight between two points
    bool intersectsWithReflectingBoundary(vector<float, DIM> xi,
                                          vector<float, DIM> xj,
                                          vector<float, DIM> ni,
                                          vector<float, DIM> nj,
                                          bool offseti,
                                          bool offsetj);

    // samples a point on the reflecting boundary
    bool sampleReflectingBoundary(vector<float, DIM> x,
                                  float radius,
                                  vector<float, DIM> randNums,
                                  out BoundarySample<DIM> boundarySample);

    // computes the radius of a star-shaped region on the reflecting boundary
    float computeStarRadiusForReflectingBoundary(vector<float, DIM> x,
                                                 float minRadius,
                                                 float maxRadius,
                                                 float silhouettePrecision,
                                                 bool flipNormalOrientation);

    // checks if a point is inside the domain (assuming it is watertight)
    bool insideDomain(vector<float, DIM> x);

    // checks if a point is inside or outside the bounding domain
    bool insideBoundingDomain(vector<float, DIM> x);
    bool outsideBoundingDomain(vector<float, DIM> x);

    // checks if the absorbing and/or reflecting boundaries are empty
    bool isAbsorbingBoundaryEmpty();
    bool isReflectingBoundaryEmpty();
};

internal float3 toFloat3<let DIM : int>(vector<float, DIM> v)
{
    return float3(v[0], v[1], DIM == 2 ? 0.0 : v[2]);
}

internal vector<float, DIM> fromFloat3<let DIM : int>(float3 v)
{
    vector<float, DIM> result = vector<float, DIM>(0.0);
    for (int i = 0; i < DIM; i++)
    {
        result[i] = v[i];
    }

    return result;
}

internal vector<float, DIM> fromFloat2<let DIM : int>(float2 v)
{
    vector<float, DIM> result = vector<float, DIM>(0.0);
    for (int i = 0; i < DIM; i++)
    {
        result[i] = v[i];
    }

    return result;
}

internal float signedDistance(float3 x, float3 p, float3 n, float d)
{
    return (dot(x - p, n) > 0.0 ? 1.0 : -1.0) * d;
}

internal float3 offsetPointAlongDirectionImpl<let DIM : int>(vector<float, DIM> p,
                                                             vector<float, DIM> n)
{
    // source: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
    const float origin = 1.0 / 32.0;
    const float floatScale = 1.0 / 65536.0;
    const float intScale = 256.0;

    float3 result = float3(0.0);
    for (int i = 0; i < DIM; i++)
    {
        int nOffset = (int)(n[i] * intScale);
        float pOffset = asfloat(asint(p[i]) + (p[i] < 0.0 ? -nOffset : nOffset));
        result[i] = abs(p[i]) < origin ? p[i] + floatScale * n[i] : pOffset;
    }

    return result;
}

public struct EmptyAbsorbingBoundaryGeometricQueries<let DIM : int> : IAbsorbingBoundaryGeometricQueries<DIM>
{
    // computes the distance to the boundary
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        return FLT_MAX;
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        distance = 0.0;
        normal = vector<float, DIM>(0.0);
        return false;
    }

    // intersects a ray with the boundary
    public bool intersectBoundary(vector<float, DIM> origin,
                                  vector<float, DIM> normal,
                                  vector<float, DIM> dir,
                                  float tMax,
                                  bool onBoundary,
                                  out IntersectionPoint<DIM> intersectionPt)
    {
        intersectionPt = IntersectionPoint<DIM>();
        return false;
    }
};

public struct EmptyReflectingBoundaryGeometricQueries<let DIM : int> : IReflectingBoundaryGeometricQueries<DIM>
{
    // computes the distance to the boundary
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        return FLT_MAX;
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        distance = 0.0;
        normal = vector<float, DIM>(0.0);
        return false;
    }

    // intersects a ray with the boundary
    public bool intersectBoundary(vector<float, DIM> origin,
                                  vector<float, DIM> normal,
                                  vector<float, DIM> dir,
                                  float tMax,
                                  bool onBoundary,
                                  out IntersectionPoint<DIM> intersectionPt)
    {
        intersectionPt = IntersectionPoint<DIM>();
        return false;
    }

    // checks whether there is a line of sight between two points
    public bool intersectsWithBoundary(vector<float, DIM> xi,
                                       vector<float, DIM> xj,
                                       vector<float, DIM> ni,
                                       vector<float, DIM> nj,
                                       bool offseti,
                                       bool offsetj)
    {
        return false;
    }

    // samples a point on the boundary
    public bool sampleBoundary(vector<float, DIM> x,
                               float radius,
                               vector<float, DIM> randNums,
                               out BoundarySample<DIM> boundarySample)
    {
        boundarySample = BoundarySample<DIM>();
        return false;
    }

    // computes the radius of a star-shaped region on the boundary
    public float computeStarRadiusForBoundary(vector<float, DIM> x,
                                              float minRadius,
                                              float maxRadius,
                                              float silhouettePrecision,
                                              bool flipNormalOrientation)
    {
        return maxRadius;
    }
};

public struct GeometricQueries<AbsorbingBoundaryGeometricQueries,
                               ReflectingBoundaryGeometricQueries,
                               let DIM : int> : IGeometricQueries<DIM>
    where AbsorbingBoundaryGeometricQueries : IAbsorbingBoundaryGeometricQueries<DIM>
    where ReflectingBoundaryGeometricQueries : IReflectingBoundaryGeometricQueries<DIM>
{
    private AbsorbingBoundaryGeometricQueries absorbingBoundaryGeometricQueries;
    private ReflectingBoundaryGeometricQueries reflectingBoundaryGeometricQueries;
    private vector<float, DIM> domainMin;
    private vector<float, DIM> domainMax;
    private uint domainIsWatertight;
    private uint hasNonEmptyAbsorbingBoundary;
    private uint hasNonEmptyReflectingBoundary;

    // computes the distance to the boundary
    public float computeDistToAbsorbingBoundary(vector<float, DIM> x,
                                                bool computeSignedDistance,
                                                float maxDistance = FLT_MAX)
    {
        if (isAbsorbingBoundaryEmpty())
        {
            return length(min(domainMin - x, x - domainMax));
        }

        return absorbingBoundaryGeometricQueries.computeDistToBoundary(x, computeSignedDistance, maxDistance);
    }
    public float computeDistToReflectingBoundary(vector<float, DIM> x,
                                                 bool computeSignedDistance,
                                                 float maxDistance = FLT_MAX)
    {
        return reflectingBoundaryGeometricQueries.computeDistToBoundary(x, computeSignedDistance, maxDistance);
    }
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance)
    {
        float d1 = computeDistToAbsorbingBoundary(x, computeSignedDistance);
        float d2 = computeDistToReflectingBoundary(x, computeSignedDistance);

        return abs(d1) < abs(d2) ? d1 : d2;
    }

    // projects a point to the boundary
    public bool projectToAbsorbingBoundary(inout vector<float, DIM> x,
                                           out vector<float, DIM> normal,
                                           out float distance,
                                           bool computeSignedDistance)
    {
        return absorbingBoundaryGeometricQueries.projectToBoundary(x, normal, distance,
                                                                   computeSignedDistance);
    }
    public bool projectToReflectingBoundary(inout vector<float, DIM> x,
                                            out vector<float, DIM> normal,
                                            out float distance,
                                            bool computeSignedDistance)
    {
        return reflectingBoundaryGeometricQueries.projectToBoundary(x, normal, distance,
                                                                    computeSignedDistance);
    }
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        distance = FLT_MAX;
        bool didProject = false;
        vector<float, DIM> absorbingBoundaryPt = x;
        vector<float, DIM> reflectingBoundaryPt = x;
        vector<float, DIM> absorbingBoundaryNormal;
        vector<float, DIM> reflectingBoundaryNormal;
        float distanceToAbsorbingBoundary;
        float distanceToReflectingBoundary;

        if (projectToAbsorbingBoundary(absorbingBoundaryPt, absorbingBoundaryNormal,
                                       distanceToAbsorbingBoundary, computeSignedDistance))
        {
            x = absorbingBoundaryPt;
            normal = absorbingBoundaryNormal;
            distance = distanceToAbsorbingBoundary;
            didProject = true;
        }

        if (projectToReflectingBoundary(reflectingBoundaryPt, reflectingBoundaryNormal,
                                        distanceToReflectingBoundary, computeSignedDistance))
        {
            if (abs(distanceToReflectingBoundary) < abs(distance))
            {
                x = reflectingBoundaryPt;
                normal = reflectingBoundaryNormal;
                distance = distanceToReflectingBoundary;
            }

            didProject = true;
        }

        if (!didProject)
        {
            distance = 0.0;
        }

        return didProject;
    }

    // offsets a point along a direction
    public vector<float, DIM> offsetPointAlongDirection(vector<float, DIM> p,
                                                        vector<float, DIM> n)
    {
        return fromFloat3<DIM>(offsetPointAlongDirectionImpl<DIM>(p, n));
    }

    // intersects a ray with the boundary
    public bool intersectAbsorbingBoundary(vector<float, DIM> origin,
                                           vector<float, DIM> normal,
                                           vector<float, DIM> dir,
                                           float tMax,
                                           bool onAbsorbingBoundary,
                                           out IntersectionPoint<DIM> intersectionPt)
    {
        return absorbingBoundaryGeometricQueries.intersectBoundary(origin, normal, dir, tMax,
                                                                   onAbsorbingBoundary,
                                                                   intersectionPt);
    }
    public bool intersectReflectingBoundary(vector<float, DIM> origin,
                                            vector<float, DIM> normal,
                                            vector<float, DIM> dir,
                                            float tMax,
                                            bool onReflectingBoundary,
                                            out IntersectionPoint<DIM> intersectionPt)
    {
        return reflectingBoundaryGeometricQueries.intersectBoundary(origin, normal, dir, tMax,
                                                                    onReflectingBoundary,
                                                                    intersectionPt);
    }
    public bool intersectBoundary(vector<float, DIM> origin,
                                  vector<float, DIM> normal,
                                  vector<float, DIM> dir,
                                  float tMax,
                                  bool onAbsorbingBoundary,
                                  bool onReflectingBoundary,
                                  out IntersectionPoint<DIM> intersectionPt)
    {
        IntersectionPoint<DIM> absorbingBoundaryIntersectionPt;
        bool intersectedAbsorbingBoundary = intersectAbsorbingBoundary(
            origin, normal, dir, tMax, onAbsorbingBoundary, absorbingBoundaryIntersectionPt);

        IntersectionPoint<DIM> reflectingBoundaryIntersectionPt;
        bool intersectedReflectingBoundary = intersectReflectingBoundary(
            origin, normal, dir, tMax, onReflectingBoundary, reflectingBoundaryIntersectionPt);

        if (intersectedAbsorbingBoundary && intersectedReflectingBoundary)
        {
            if (absorbingBoundaryIntersectionPt.dist < reflectingBoundaryIntersectionPt.dist)
            {
                intersectionPt = absorbingBoundaryIntersectionPt;
            }
            else
            {
                intersectionPt = reflectingBoundaryIntersectionPt;
            }
        }
        else if (intersectedAbsorbingBoundary)
        {
            intersectionPt = absorbingBoundaryIntersectionPt;
        }
        else if (intersectedReflectingBoundary)
        {
            intersectionPt = reflectingBoundaryIntersectionPt;
        }

        return intersectedAbsorbingBoundary || intersectedReflectingBoundary;
    }

    // checks whether there is a line of sight between two points
    public bool intersectsWithReflectingBoundary(vector<float, DIM> xi,
                                                 vector<float, DIM> xj,
                                                 vector<float, DIM> ni,
                                                 vector<float, DIM> nj,
                                                 bool offseti,
                                                 bool offsetj)
    {
        return reflectingBoundaryGeometricQueries.intersectsWithBoundary(xi, xj, ni, nj, offseti, offsetj);
    }

    // samples a point on the reflecting boundary
    public bool sampleReflectingBoundary(vector<float, DIM> x,
                                         float radius,
                                         vector<float, DIM> randNums,
                                         out BoundarySample<DIM> boundarySample)
    {
        return reflectingBoundaryGeometricQueries.sampleBoundary(x, radius, randNums, boundarySample);
    }

    // computes the radius of a star-shaped region on the reflecting boundary
    public float computeStarRadiusForReflectingBoundary(vector<float, DIM> x,
                                                        float minRadius,
                                                        float maxRadius,
                                                        float silhouettePrecision,
                                                        bool flipNormalOrientation)
    {
        return reflectingBoundaryGeometricQueries.computeStarRadiusForBoundary(x, minRadius, maxRadius,
                                                                               silhouettePrecision,
                                                                               flipNormalOrientation);
    }

    // checks if a point is inside the domain (assuming it is watertight)
    public bool insideDomain(vector<float, DIM> x)
    {
        if (domainIsWatertight != 0)
        {
            return computeDistToBoundary(x, true) < 0.0;
        }

        return true;
    }

    // checks if a point is inside the bounding domain
    public bool insideBoundingDomain(vector<float, DIM> x)
    {
        return all(x >= domainMin) && all(x <= domainMax);
    }

    // checks if a point is outside the bounding domain
    public bool outsideBoundingDomain(vector<float, DIM> x)
    {
        return any(x < domainMin) || any(x > domainMax);
    }

    // checks if the absorbing boundary is empty
    public bool isAbsorbingBoundaryEmpty()
    {
        return hasNonEmptyAbsorbingBoundary == 0;
    }

    // checks if the reflecting boundary is empty
    public bool isReflectingBoundaryEmpty()
    {
        return hasNonEmptyReflectingBoundary == 0;
    }
};
