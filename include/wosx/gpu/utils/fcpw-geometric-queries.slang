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

// This file provides an implementation of the IAbsorbingBoundaryGeometricQueries
// and IReflectingBoundaryGeometricQueries interfaces using the FCPW library to
// perform accelerated geometric queries against a mesh.

implementing wosx;
__include core.geometric_queries;
__include core.distributions;
__include utils.reflectance_boundary_bvh.bvh;
__include utils.reflectance_boundary_bvh.robin_bounds;

public struct GreensFnBranchTraversalWeight : IBranchTraversalWeight
{
    private HarmonicGreensFnFreeSpace3D greensFn;

    // computes the traversal weight for a given squared distance
    public float compute(float r2)
    {
        float r = max(sqrt(r2), 1e-2); // NOTE: hardcoded tolerance for now
        return abs(greensFn.evaluate(r));
    }
};

public struct FcpwAbsorbingBoundaryGeometricQueries<AbsorbingBoundaryAggregate, let DIM : int> :
                                                    IAbsorbingBoundaryGeometricQueries<DIM>
    where AbsorbingBoundaryAggregate : IAggregate
{
    private AbsorbingBoundaryAggregate absorbingBoundaryAggregate;

    // computes the distance to the boundary
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = maxDistance < FLT_MAX ? maxDistance * maxDistance : FLT_MAX;
        Interaction interaction;
        bool found = absorbingBoundaryAggregate.findClosestPoint(sphere, interaction, computeSignedDistance);
        if (!found)
        {
            // NOTE: not accounting for signed distance here
            return maxDistance;
        }

        return computeSignedDistance ?
               signedDistance(sphere.c, interaction.p, interaction.n, interaction.d) :
               interaction.d;
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = FLT_MAX;
        Interaction interaction;
        bool found = absorbingBoundaryAggregate.findClosestPoint(sphere, interaction, computeSignedDistance);
        if (!found)
        {
            normal = vector<float, DIM>(0.0);
            distance = FLT_MAX;
            return false;
        }

        x = fromFloat3<DIM>(interaction.p);
        normal = fromFloat3<DIM>(interaction.n);
        distance = computeSignedDistance ?
                   signedDistance(sphere.c, interaction.p, interaction.n, interaction.d) :
                   interaction.d;
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
        Ray ray;
        ray.o = onBoundary ?
                offsetPointAlongDirectionImpl<DIM>(origin, -normal) :
                toFloat3<DIM>(origin);
        ray.d = toFloat3<DIM>(dir);
        ray.dInv = float3(1.0) / ray.d;
        ray.tMax = tMax;
        Interaction interaction;
        intersectionPt = IntersectionPoint<DIM>();
        bool hit = absorbingBoundaryAggregate.intersect(ray, false, interaction);
        if (!hit)
        {
            return false;
        }

        intersectionPt.pt = fromFloat3<DIM>(interaction.p);
        intersectionPt.normal = fromFloat3<DIM>(interaction.n);
        intersectionPt.dist = interaction.d;
        return true;
    }
};

public struct FcpwReflectingBoundaryGeometricQueries<ReflectingBoundaryAggregate, let DIM : int> :
                                                     IReflectingBoundaryGeometricQueries<DIM>
    where ReflectingBoundaryAggregate : IAggregate
{
    private ReflectingBoundaryAggregate reflectingBoundaryAggregate;
    private GreensFnBranchTraversalWeight branchTraversalWeight;

    // computes the distance to the boundary
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = maxDistance < FLT_MAX ? maxDistance * maxDistance : FLT_MAX;
        Interaction interaction;
        bool found = reflectingBoundaryAggregate.findClosestPoint(sphere, interaction, computeSignedDistance);
        if (!found)
        {
            // NOTE: not accounting for signed distance here
            return maxDistance;
        }

        return computeSignedDistance ?
               signedDistance(sphere.c, interaction.p, interaction.n, interaction.d) :
               interaction.d;
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = FLT_MAX;
        Interaction interaction;
        bool found = reflectingBoundaryAggregate.findClosestPoint(sphere, interaction, computeSignedDistance);
        if (!found)
        {
            normal = vector<float, DIM>(0.0);
            distance = FLT_MAX;
            return false;
        }

        x = fromFloat3<DIM>(interaction.p);
        normal = fromFloat3<DIM>(interaction.n);
        distance = computeSignedDistance ?
                   signedDistance(sphere.c, interaction.p, interaction.n, interaction.d) :
                   interaction.d;
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
        Ray ray;
        ray.o = onBoundary ?
                offsetPointAlongDirectionImpl<DIM>(origin, -normal) :
                toFloat3<DIM>(origin);
        ray.d = toFloat3<DIM>(dir);
        ray.dInv = float3(1.0) / ray.d;
        ray.tMax = tMax;
        Interaction interaction;
        intersectionPt = IntersectionPoint<DIM>();
        bool hit = reflectingBoundaryAggregate.intersect(ray, false, interaction);
        if (!hit)
        {
            return false;
        }

        intersectionPt.pt = fromFloat3<DIM>(interaction.p);
        intersectionPt.normal = fromFloat3<DIM>(interaction.n);
        intersectionPt.dist = interaction.d;
        return true;
    }

    // checks whether there is a line of sight between two points
    public bool intersectsWithBoundary(vector<float, DIM> xi,
                                       vector<float, DIM> xj,
                                       vector<float, DIM> ni,
                                       vector<float, DIM> nj,
                                       bool offseti,
                                       bool offsetj)
    {
        float3 pt1 = offseti ? offsetPointAlongDirectionImpl<DIM>(xi, -ni) : toFloat3<DIM>(xi);
        float3 pt2 = offsetj ? offsetPointAlongDirectionImpl<DIM>(xj, -nj) : toFloat3<DIM>(xj);
        float3 direction = pt2 - pt1;
        float dNorm = length(direction);
        direction /= dNorm;

        Ray ray;
        ray.o = pt1;
        ray.d = direction;
        ray.dInv = float3(1.0) / direction;
        ray.tMax = dNorm;
        Interaction interaction;
        return reflectingBoundaryAggregate.intersect(ray, true, interaction);
    }

    // samples a point on the boundary
    public bool sampleBoundary(vector<float, DIM> x,
                               float radius,
                               vector<float, DIM> randNums,
                               out BoundarySample<DIM> boundarySample)
    {
        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = radius * radius;
        Interaction interaction;
        boundarySample = BoundarySample<DIM>();
        bool hit = reflectingBoundaryAggregate.intersect<GreensFnBranchTraversalWeight>(
                    sphere, toFloat3<DIM>(randNums), branchTraversalWeight, interaction);
        if (!hit)
        {
            return false;
        }

        boundarySample.pt = fromFloat3<DIM>(interaction.p);
        boundarySample.normal = fromFloat3<DIM>(interaction.n);
        boundarySample.pdf = interaction.d;
        return true;
    }

    // computes the radius of a star-shaped region on the boundary
    public float computeStarRadiusForBoundary(vector<float, DIM> x,
                                              float minRadius,
                                              float maxRadius,
                                              float silhouettePrecision,
                                              bool flipNormalOrientation)
    {
        if (minRadius > maxRadius)
        {
            return maxRadius;
        }

        BoundingSphere sphere;
        sphere.c = toFloat3<DIM>(x);
        sphere.r2 = maxRadius < FLT_MAX ? maxRadius * maxRadius : FLT_MAX;
        bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
        if (flipNormalOrientation)
        {
            flipNormals = !flipNormals;
        }

        // NOTE: Unlike FCPW's Bvh, ReflectanceobinBvh::findClosestSilhouettePoint internally calls
        // ReflectanceBvh::computeSquaredStarRadius. This means that for reflectance-based boundaries,
        // the radius of a star-shaped region will typically be smaller than the distance to the
        // closest silhouette point on the reflecting boundary.
        Interaction interaction;
        bool found = reflectingBoundaryAggregate.findClosestSilhouettePoint(
                        sphere, flipNormals, minRadius * minRadius,
                        silhouettePrecision, interaction);
        return found ? max(interaction.d, minRadius) : max(maxRadius, minRadius);
    }
};
