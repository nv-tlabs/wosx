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

// This file extends the 'LineSegment' and 'Triangle' primitive structures from
// the FCPW library to support reflectance-based boundary conditions. Users of
// WoSX need not interact with this file directly.

implementing wosx;
__include core.math_helpers;
import fcpw;

public interface IReflectancePrimitiveBound
{
    // computes the squared star radius bound
    static void computeSquaredStarRadiusBound(inout BoundingSphere s, float maxCoefficientValue,
                                              float3 viewDirClosest, float closestPtDist,
                                              float3 viewDirFarthest, float farthestPtDist,
                                              float3 planePt, float3 planeNormal);
};

public interface IReflectancePrimitive
{
    // computes the squared reflectance sphere radius
    void computeSquaredStarRadius(inout BoundingSphere s,
                                  bool flipNormalOrientation,
                                  float silhouettePrecision,
                                  bool performSilhouetteTests = true);

    // returns the number of adjacent faces
    uint getAdjacentFaceCount();

    // returns the center of an adjacent edge
    float3 getAdjacentEdgeCenter(uint faceIndex);

    // returns the normal of an adjacent face
    float3 getAdjacentFaceNormal(uint faceIndex);

    // checks if the primitive has an adjacent face
    bool checkHasAdjacentFace(uint faceIndex);

    // returns the minimum coefficient value for the primitive
    float getMinCoefficientValue();

    // returns the maximum coefficient value for the primitive
    float getMaxCoefficientValue();
};

internal float findClosestPointPlane(float3 p, float3 n, float3 x, out float3 pt)
{
    float t = dot(n, p - x) / dot(n, n);
    pt = x - t * n;

    return abs(t);
}

internal float findFarthestPointLineSegment(float3 pa, float3 pb, float3 x, out float3 pt)
{
    float da = dot(x - pa, x - pa);
    float db = dot(x - pb, x - pb);

    if (da > db)
    {
        pt = pa;
        return sqrt(da);
    }

    pt = pb;
    return sqrt(db);
}

public struct ReflectanceLineSegment<PrimitiveBound> : IPrimitive, IReflectancePrimitive
    where PrimitiveBound : IReflectancePrimitiveBound
{
    private float3 p[2];
    private float3 n[2];
    private int hasAdjacentFace[2];
    private float minCoefficientValue;
    private float maxCoefficientValue;
    private uint index;

    // returns the bounding box of the primitive
    public BoundingBox getBoundingBox()
    {
        float3 epsilon = float3(FLT_EPSILON, FLT_EPSILON, 0.0);
        return BoundingBox(min(p[0], p[1]) - epsilon, max(p[0], p[1]) + epsilon);
    }

    // returns the centroid of the primitive
    public float3 getCentroid()
    {
        return 0.5 * (p[0] + p[1]);
    }

    // returns the normal of the primitive
    public float3 getNormal()
    {
        float3 s = p[1] - p[0];
        float3 n = float3(s.y, -s.x, 0.0);

        return normalize(n);
    }

    // returns the surface area of the primitive
    public float getSurfaceArea()
    {
        return length(p[1] - p[0]);
    }

    // intersects primitive with ray
    // NOTE: specialized to 2D (z coordinate == 0)
    public bool intersect(Ray r, bool checkForOcclusion, inout Interaction i)
    {
        bool didIntersect = intersectLineSegment(p[0], p[1], r.o, r.d, r.tMax,
                                                 checkForOcclusion, i.p, i.n, i.uv, i.d);
        if (didIntersect)
        {
            i.index = index;
            return true;
        }

        return false;
    }

    // intersects primitive with sphere
    public bool intersect(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointLineSegment(p[0], p[1], s.c, i.p, i.uv[0]);
        if (d * d <= s.r2)
        {
            i.d = getSurfaceArea();
            i.index = index;
            return true;
        }

        return false;
    }

    // finds closest point on primitive from sphere center
    public bool findClosestPoint(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointLineSegment(p[0], p[1], s.c, i.p, i.uv[0]);
        if (d * d <= s.r2)
        {
            i.uv[1] = 0.0;
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // computes the squared reflectance sphere radius
    public void computeSquaredStarRadius(inout BoundingSphere s,
                                         bool flipNormalOrientation,
                                         float silhouettePrecision,
                                         bool performSilhouetteTests = true)
    {
        // find closest point on line segment
        float3 closestPt;
        float tLineSegment;
        float closestPtDist = findClosestPointLineSegment(p[0], p[1], s.c, closestPt, tLineSegment);
        float closestPtDist2 = closestPtDist * closestPtDist;
        if (closestPtDist2 > s.r2)
        {
            return;
        }

        if (maxCoefficientValue < FLT_MAX)
        {
            // perform silhouette tests for reflecting boundaries
            float3 n0 = getNormal();
            float3 viewDirClosest = s.c - closestPt;

            if (performSilhouetteTests)
            {
                if (hasAdjacentFace[0] >= 0)
                {
                    float3 viewDir = s.c - p[0];
                    float silhouettePtDist = length(viewDir);
                    bool isSilhouette = hasAdjacentFace[0] == 0;
                    if (!isSilhouette)
                    {
                        isSilhouette = isSilhouetteVertex(n0, n[0], viewDir, silhouettePtDist,
                                                          flipNormalOrientation, silhouettePrecision);
                    }

                    if (isSilhouette)
                    {
                        s.r2 = min(s.r2, silhouettePtDist * silhouettePtDist);
                    }
                }

                if (hasAdjacentFace[1] >= 0)
                {
                    float3 viewDir = s.c - p[1];
                    float silhouettePtDist = length(viewDir);
                    bool isSilhouette = hasAdjacentFace[1] == 0;
                    if (!isSilhouette)
                    {
                        isSilhouette = isSilhouetteVertex(n[1], n0, viewDir, silhouettePtDist,
                                                          flipNormalOrientation, silhouettePrecision);
                    }

                    if (isSilhouette)
                    {
                        s.r2 = min(s.r2, silhouettePtDist * silhouettePtDist);
                    }
                }
            }

            if (maxCoefficientValue > FLT_EPSILON)
            {
                // [Reflectance case]: shrink radius to ensure bounded reflectance
                if (closestPtDist < FLT_EPSILON)
                {
                    s.r2 = closestPtDist;
                }
                else if (s.r2 > closestPtDist2)
                {
                    float3 farthestPt;
                    float farthestPtDist = findFarthestPointLineSegment(p[0], p[1], s.c, farthestPt);
                    float3 viewDirFarthest = s.c - farthestPt;
                    PrimitiveBound.computeSquaredStarRadiusBound(s, maxCoefficientValue, viewDirClosest, closestPtDist,
                                                                 viewDirFarthest, farthestPtDist, p[0], n0);
                }
            }
            else
            {
                // [Perfectly reflecting case]: dist to closest visibility silhouette
            }
        }
        else
        {
            // [Perfectly absorbing case]: dist to closest point on line segment
            s.r2 = min(s.r2, closestPtDist2);
        }
    }

    // samples point on primitive and returns sampling pdf
    public float samplePoint(float2 randNums, out float2 uv, out float3 q, out float3 n)
    {
        float3 s = p[1] - p[0];
        float area = length(s);
        float u = randNums[0];
        uv = float2(u, 0.0);
        q = p[0] + u * s;
        n = float3(s[1], -s[0], 0.0) / area;

        return 1.0 / area;
    }

    // returns the number of adjacent faces
    public uint getAdjacentFaceCount()
    {
        return 2;
    }

    // returns the center of an adjacent edge
    public float3 getAdjacentEdgeCenter(uint faceIndex)
    {
        return p[faceIndex];
    }

    // returns the normal of an adjacent face
    public float3 getAdjacentFaceNormal(uint faceIndex)
    {
        return n[faceIndex];
    }

    // checks if the primitive has an adjacent face
    public bool checkHasAdjacentFace(uint faceIndex)
    {
        return hasAdjacentFace[faceIndex] != 0;
    }

    // returns the minimum coefficient value for the primitive
    public float getMinCoefficientValue()
    {
        return minCoefficientValue;
    }

    // returns the maximum coefficient value for the primitive
    public float getMaxCoefficientValue()
    {
        return maxCoefficientValue;
    }

    // returns the index of the primitive
    public uint getIndex()
    {
        return index;
    }
};

internal float findFarthestPointTriangle(float3 pa, float3 pb, float3 pc, float3 x, out float3 pt)
{
    float da = dot(x - pa, x - pa);
    float db = dot(x - pb, x - pb);
    float dc = dot(x - pc, x - pc);

    if (da >= db && da >= dc)
    {
        pt = pa;
        return sqrt(da);
    }
    else if (db >= da && db >= dc)
    {
        pt = pb;
        return sqrt(db);
    }

    pt = pc;
    return sqrt(dc);
}

public struct ReflectanceTriangle<PrimitiveBound> : IPrimitive, IReflectancePrimitive
    where PrimitiveBound : IReflectancePrimitiveBound
{
    private float3 p[3];
    private float3 n[3];
    private int hasAdjacentFace[3];
    private float minCoefficientValue;
    private float maxCoefficientValue;
    private uint index;

    // returns the bounding box of the primitive
    public BoundingBox getBoundingBox()
    {
        float3 epsilon = float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON);
        return BoundingBox(min(min(p[0], p[1]), p[2]) - epsilon, max(max(p[0], p[1]), p[2]) + epsilon);
    }

    // returns the centroid of the primitive
    public float3 getCentroid()
    {
        return (p[0] + p[1] + p[2]) / 3.0;
    }

    // returns the surface area of the primitive
    public float getSurfaceArea()
    {
        return 0.5 * length(cross(p[1] - p[0], p[2] - p[0]));
    }

    // returns the normal of the primitive
    public float3 getNormal()
    {
        float3 n = cross(p[1] - p[0], p[2] - p[0]);

        return normalize(n);
    }

    // intersects primitive with ray
    public bool intersect(Ray r, bool checkForOcclusion, inout Interaction i)
    {
        bool didIntersect = intersectTriangle(p[0], p[1], p[2], r.o, r.d, r.tMax,
                                              checkForOcclusion, i.p, i.n, i.uv, i.d);
        if (didIntersect)
        {
            i.index = index;
            return true;
        }

        return false;
    }

    // intersects primitive with sphere
    public bool intersect(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointTriangle(p[0], p[1], p[2], s.c, i.p, i.uv);
        if (d * d <= s.r2)
        {
            i.d = getSurfaceArea();
            i.index = index;
            return true;
        }

        return false;
    }

    // finds closest point on primitive from sphere center
    public bool findClosestPoint(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointTriangle(p[0], p[1], p[2], s.c, i.p, i.uv);
        if (d * d <= s.r2)
        {
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // computes the squared reflectance sphere radius
    public void computeSquaredStarRadius(inout BoundingSphere s,
                                         bool flipNormalOrientation,
                                         float silhouettePrecision,
                                         bool performSilhouetteTests = true)
    {
        // find closest point on triangle
        float3 closestPt;
        float2 tTriangle;
        float closestPtDist = findClosestPointTriangle(p[0], p[1], p[2], s.c, closestPt, tTriangle);
        float closestPtDist2 = closestPtDist * closestPtDist;
        if (closestPtDist2 > s.r2)
        {
            return;
        }

        if (maxCoefficientValue < FLT_MAX)
        {
            // perform silhouette tests for reflecting boundaries
            float3 n0 = getNormal();
            float3 viewDirClosest = s.c - closestPt;

            if (performSilhouetteTests)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (hasAdjacentFace[j] >= 0)
                    {
                        float3 pj = p[j];
                        float3 pk = p[(j + 1) % 3];
                        float3 pt;
                        float t;
                        float silhouettePtDist = findClosestPointLineSegment(pj, pk, s.c, pt, t);
                        float3 viewDir = s.c - pt;
                        bool isSilhouette = hasAdjacentFace[j] == 0;

                        if (!isSilhouette)
                        {
                            isSilhouette = isSilhouetteEdge(pj, pk, n[j], n0, viewDir, silhouettePtDist,
                                                            flipNormalOrientation, silhouettePrecision);
                        }

                        if (isSilhouette)
                        {
                            s.r2 = min(s.r2, silhouettePtDist * silhouettePtDist);
                        }
                    }
                }
            }

            if (maxCoefficientValue > FLT_EPSILON)
            {
                // [Reflectance case]: shrink radius to ensure bounded reflectance
                if (closestPtDist < FLT_EPSILON)
                {
                    s.r2 = closestPtDist;
                }
                else if (s.r2 > closestPtDist2)
                {
                    float3 farthestPt;
                    float farthestPtDist = findFarthestPointTriangle(p[0], p[1], p[2], s.c, farthestPt);
                    float3 viewDirFarthest = s.c - farthestPt;
                    PrimitiveBound.computeSquaredStarRadiusBound(s, maxCoefficientValue, viewDirClosest, closestPtDist,
                                                                 viewDirFarthest, farthestPtDist, p[0], n0);
                }
            }
            else
            {
                // [Perfectly reflecting case]: dist to closest visibility silhouette
            }
        }
        else
        {
            // [Perfectly absorbing case]: dist to closest point on triangle
            s.r2 = min(s.r2, closestPtDist2);
        }
    }

    // samples point on primitive and returns sampling pdf
    public float samplePoint(float2 randNums, out float2 uv, out float3 q, out float3 n)
    {
        n = cross(p[1] - p[0], p[2] - p[0]);
        float area = length(n);
        float u1 = sqrt(randNums[0]);
        float u2 = randNums[1];
        float u = 1.0 - u1;
        float v = u2 * u1;
        float w = 1.0 - u - v;
        uv = float2(u, v);
        q = p[0] * u + p[1] * v + p[2] * w;
        n /= area;

        return 2.0 / area;
    }

    // returns the number of adjacent faces
    public uint getAdjacentFaceCount()
    {
        return 3;
    }

    // returns the center of an adjacent edge
    public float3 getAdjacentEdgeCenter(uint faceIndex)
    {
        uint I = faceIndex;
        uint J = (faceIndex + 1) % 3;

        return 0.5 * (p[I] + p[J]);
    }

    // returns the normal of an adjacent face
    public float3 getAdjacentFaceNormal(uint faceIndex)
    {
        return n[faceIndex];
    }

    // checks if the primitive has an adjacent face
    public bool checkHasAdjacentFace(uint faceIndex)
    {
        return hasAdjacentFace[faceIndex] != 0;
    }

    // returns the minimum coefficient value for the primitive
    public float getMinCoefficientValue()
    {
        return minCoefficientValue;
    }

    // returns the maximum coefficient value for the primitive
    public float getMaxCoefficientValue()
    {
        return maxCoefficientValue;
    }

    // returns the index of the primitive
    public uint getIndex()
    {
        return index;
    }
};
