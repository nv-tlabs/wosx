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

// This file implements the Boundary Value Caching algorithm for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points via sample caching and reuse.
//
// Resources:
// - Boundary Value Caching for Walk on Spheres [2023]

implementing wosx;
__include point_estimation.walk_on_stars;

public struct BVCEvaluationPoint<let DIM : int>
{
    public vector<float, DIM> pt;
    public vector<float, DIM> normal;
    public SampleType type;
#if WOSX_PROBLEM_DIMENSION == 2
    // Padding to match CPU layout: alignas(8) on float2 makes this struct 24B on CPU/CUDA,
    // but Vulkan scalar layout gives float2 only 4B alignment, making it 20B without padding.
    private uint _pad;
#endif

    // constructor
    public __init()
    {
        pt = vector<float, DIM>(0.0);
        normal = vector<float, DIM>(0.0);
        type = SampleType::InDomain;
    }
};

public struct BVCEvaluationOutputs<T, let DIM : int>
    where T : IFloat
{
    public T solution;
    public T gradient[DIM];
};

public struct BVCSampleStatistics<T, let DIM : int>
    where T : IFloat
{
    private T solutionMean;
    private T gradientMean[DIM];

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
    }

    // adds solution estimate to running sum
    [mutating]
    public void addSolutionEstimate(T estimate, uint N)
    {
        update(estimate, solutionMean, N);
    }

    // adds gradient estimate to running sum
    [mutating]
    public void addGradientEstimate(T estimate[DIM], uint N)
    {
        for (int i = 0; i < DIM; i++)
        {
            update(estimate[i], gradientMean[i], N);
        }
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

    // updates statistics
    private void update(T estimate, inout T mean, uint N)
    {
        T delta = estimate - mean;
        mean = mean + delta.scale(1.0 / N);
    }
};

public struct BVCEvaluationStatistics<T, let DIM : int>
    where T : IFloat
{
    public BVCSampleStatistics<T, DIM> absorbingBoundary;
    public BVCSampleStatistics<T, DIM> absorbingBoundaryNormalAligned;
    public BVCSampleStatistics<T, DIM> reflectingBoundary;
    public BVCSampleStatistics<T, DIM> reflectingBoundaryNormalAligned;
    public BVCSampleStatistics<T, DIM> source;

    // constructor
    public __init()
    {
        reset();
    }

    // resets statistics
    [mutating]
    public void reset()
    {
        absorbingBoundary.reset();
        absorbingBoundaryNormalAligned.reset();
        reflectingBoundary.reset();
        reflectingBoundaryNormalAligned.reset();
        source.reset();
    }

    // returns estimated solution
    public T getEstimatedSolution()
    {
        T solution = absorbingBoundary.getEstimatedSolution();
        solution = solution + absorbingBoundaryNormalAligned.getEstimatedSolution();
        solution = solution + reflectingBoundary.getEstimatedSolution();
        solution = solution + reflectingBoundaryNormalAligned.getEstimatedSolution();
        solution = solution + source.getEstimatedSolution();

        return solution;
    }

    // returns estimated gradient
    public void getEstimatedGradient(out T gradient[DIM])
    {
        T absorbingBoundaryGradient[DIM];
        T absorbingBoundaryNormalAlignedGradient[DIM];
        T reflectingBoundaryGradient[DIM];
        T reflectingBoundaryNormalAlignedGradient[DIM];
        T sourceGradient[DIM];
        absorbingBoundary.getEstimatedGradient(absorbingBoundaryGradient);
        absorbingBoundaryNormalAligned.getEstimatedGradient(absorbingBoundaryNormalAlignedGradient);
        reflectingBoundary.getEstimatedGradient(reflectingBoundaryGradient);
        reflectingBoundaryNormalAligned.getEstimatedGradient(reflectingBoundaryNormalAlignedGradient);
        source.getEstimatedGradient(sourceGradient);

        for (int i = 0; i < DIM; i++)
        {
            gradient[i] = absorbingBoundaryGradient[i];
            gradient[i] = gradient[i] + absorbingBoundaryNormalAlignedGradient[i];
            gradient[i] = gradient[i] + reflectingBoundaryGradient[i];
            gradient[i] = gradient[i] + reflectingBoundaryNormalAlignedGradient[i];
            gradient[i] = gradient[i] + sourceGradient[i];
        }
    }

    // returns evaluation outputs
    public BVCEvaluationOutputs<T, DIM> getOutputs()
    {
        BVCEvaluationOutputs<T, DIM> outputs;
        outputs.solution = getEstimatedSolution();
        getEstimatedGradient(outputs.gradient);

        return outputs;
    }
};

public struct BVCBoundaryData<T>
    where T : IFloat
{
    public T solution;
    public T normalDerivative;

    // constructor
    public __init()
    {
        solution = T(0.0);
        normalDerivative = T(0.0);
    }
};

public struct BoundaryValueCaching<T, PDE, let DIM : int>
    where T : IFloat
    where PDE : IPDE<T, DIM>
{
    private WalkSettings walkSettings;
    private PDE pde;
    private typedef PDE::GreensFnFreeSpace GreensFnFreeSpace;
    private typedef GreensFnFreeSpace::KernelRegularization KernelRegularization;

    // returns the boundary data for the provided sample point
    public BVCBoundaryData<T> getBoundaryData(SamplePoint<DIM> samplePt,
                                              T estimatedSolution,
                                              T estimatedNormalDerivative)
    {
        // initialize boundary data with the estimated solution and normal derivative
        BVCBoundaryData<T> boundaryData;
        boundaryData.solution = estimatedSolution;

        if (samplePt.type == SampleType::OnReflectingBoundary)
        {
            // set the normal derivative to the robin value for sample points on
            // the Neumann boundary; for points on the Robin boundary, the robin
            // data and coefficient are re-evaluated from the PDE during splatting
            if (walkSettings.ignoreReflectingBoundaryContribution == 0 &&
                pde.areRobinConditionsPureNeumann())
            {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided != 0 &&
                                                        samplePt.estimateBoundaryNormalAligned != 0;
                boundaryData.normalDerivative = pde.robin(samplePt.pt, samplePt.normal,
                                                          returnBoundaryNormalAlignedValue);
            }
        }
        else
        {
            boundaryData.normalDerivative = estimatedNormalDerivative;
        }

        return boundaryData;
    }

    // splats boundary data
    public void splatBoundaryData(SamplePoint<DIM> samplePt,
                                  BVCBoundaryData<T> boundaryData,
                                  BVCEvaluationPoint<DIM> evalPt,
                                  float radiusClamp,
                                  float kernelRegularization,
                                  uint nSamplesSplatted,
                                  inout BVCEvaluationStatistics<T, DIM> evalStatistics)
    {
        // initialize the greens function
        GreensFnFreeSpace greensFn;
        greensFn.updateAbsorptionCoeff(pde.absorptionCoeff());
        greensFn.updatePole(evalPt.pt);

        // compute the contribution of the boundary sample
        vector<float, DIM> pt = samplePt.pt;
        vector<float, DIM> n = samplePt.normal;
        T solution = boundaryData.solution;
        T normalDerivative = boundaryData.normalDerivative;
        float pdf = samplePt.pdf;
        SampleType sampleType = samplePt.type;
        bool estimateBoundaryNormalAligned = samplePt.estimateBoundaryNormalAligned != 0;

        // evaluate the robin data and coefficient for points on a Robin boundary
        T robin = T(0.0);
        float robinCoeff = 0.0;
        if (sampleType == SampleType::OnReflectingBoundary &&
            walkSettings.ignoreReflectingBoundaryContribution == 0 &&
            !pde.areRobinConditionsPureNeumann())
        {
            bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided != 0 &&
                                                    estimateBoundaryNormalAligned;
            robin = pde.robin(pt, n, returnBoundaryNormalAlignedValue);
            robinCoeff = pde.robinCoeff(pt, n, returnBoundaryNormalAlignedValue);
        }

        // evaluate the greens function and its gradients
        n = n * (estimateBoundaryNormalAligned ? -1.0 : 1.0);
        float r = max(radiusClamp, length(pt - greensFn.getPole()));
        float G = greensFn.evaluate(r);
        float P = greensFn.poissonKernel(r, pt, n);
        vector<float, DIM> dG = greensFn.gradient(r, pt);
        vector<float, DIM> dP = greensFn.poissonKernelGradient(r, pt, n);

        if (kernelRegularization > 0.0)
        {
            r /= kernelRegularization;
            G *= KernelRegularization.regularizationForGreensFn(r);
            P *= KernelRegularization.regularizationForPoissonKernel(r);
        }

        // compute the solution and gradient estimates
        T solutionEstimate;
        T gradientEstimate[DIM];
        float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                      evalPt.type == SampleType::OnReflectingBoundary ?
                      2.0 : 1.0;

        if (abs(robinCoeff) > 0.0)
        {
            solutionEstimate = robin.scale(alpha * G / pdf);
            solutionEstimate = solutionEstimate - solution.scale(alpha * (P + robinCoeff * G) / pdf);

            if (alpha > 1.0)
            {
                alpha = 0.0; // FUTURE: estimate gradient on the boundary
            }
            for (int i = 0; i < DIM; i++)
            {
                gradientEstimate[i] = robin.scale(alpha * dG[i] / pdf);
                gradientEstimate[i] = gradientEstimate[i] - solution.scale(alpha * (dP[i] + robinCoeff * dG[i]) / pdf);
            }
        }
        else
        {
            solutionEstimate = normalDerivative.scale(alpha * G / pdf);
            solutionEstimate = solutionEstimate - solution.scale(alpha * P / pdf);

            if (alpha > 1.0)
            {
                alpha = 0.0; // FUTURE: estimate gradient on the boundary
            }
            for (int i = 0; i < DIM; i++)
            {
                gradientEstimate[i] = normalDerivative.scale(alpha * dG[i] / pdf);
                gradientEstimate[i] = gradientEstimate[i] - solution.scale(alpha * dP[i] / pdf);
            }
        }

        // update statistics
        if (estimateBoundaryNormalAligned)
        {
            if (sampleType == SampleType::OnAbsorbingBoundary)
            {
                evalStatistics.absorbingBoundaryNormalAligned.addSolutionEstimate(solutionEstimate, nSamplesSplatted);
                evalStatistics.absorbingBoundaryNormalAligned.addGradientEstimate(gradientEstimate, nSamplesSplatted);
            }
            else if (sampleType == SampleType::OnReflectingBoundary)
            {
                evalStatistics.reflectingBoundaryNormalAligned.addSolutionEstimate(solutionEstimate, nSamplesSplatted);
                evalStatistics.reflectingBoundaryNormalAligned.addGradientEstimate(gradientEstimate, nSamplesSplatted);
            }
        }
        else
        {
            if (sampleType == SampleType::OnAbsorbingBoundary)
            {
                evalStatistics.absorbingBoundary.addSolutionEstimate(solutionEstimate, nSamplesSplatted);
                evalStatistics.absorbingBoundary.addGradientEstimate(gradientEstimate, nSamplesSplatted);
            }
            else if (sampleType == SampleType::OnReflectingBoundary)
            {
                evalStatistics.reflectingBoundary.addSolutionEstimate(solutionEstimate, nSamplesSplatted);
                evalStatistics.reflectingBoundary.addGradientEstimate(gradientEstimate, nSamplesSplatted);
            }
        }
    }

    // splats source data
    public void splatSourceData(SamplePoint<DIM> samplePt,
                                BVCEvaluationPoint<DIM> evalPt,
                                float radiusClamp,
                                float kernelRegularization,
                                uint nSamplesSplatted,
                                inout BVCEvaluationStatistics<T, DIM> evalStatistics)
    {
        // initialize the greens function
        GreensFnFreeSpace greensFn;
        greensFn.updateAbsorptionCoeff(pde.absorptionCoeff());
        greensFn.updatePole(evalPt.pt);

        // compute the contribution of the source sample; skipped
        // samples (e.g. rejected domain samples) contribute zero
        vector<float, DIM> pt = samplePt.pt;
        T source = samplePt.estimationQuantity == EstimationQuantity::Skip ?
                   T(0.0) : pde.source(pt);
        float pdf = samplePt.pdf;

        // evaluate the greens function and its gradient
        float r = max(radiusClamp, length(pt - greensFn.getPole()));
        float G = greensFn.evaluate(r);
        vector<float, DIM> dG = greensFn.gradient(r, pt);

        if (kernelRegularization > 0.0)
        {
            r /= kernelRegularization;
            G *= KernelRegularization.regularizationForGreensFn(r);
        }

        // compute the solution and gradient estimates
        float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                      evalPt.type == SampleType::OnReflectingBoundary ?
                      2.0 : 1.0;
        T solutionEstimate = source.scale(alpha * G / pdf);

        T gradientEstimate[DIM];
        if (alpha > 1.0)
        {
            alpha = 0.0; // FUTURE: estimate gradient on the boundary
        }
        for (int i = 0; i < DIM; i++)
        {
            gradientEstimate[i] = source.scale(alpha * dG[i] / pdf);
        }

        // update statistics
        evalStatistics.source.addSolutionEstimate(solutionEstimate, nSamplesSplatted);
        evalStatistics.source.addGradientEstimate(gradientEstimate, nSamplesSplatted);
    }
};
