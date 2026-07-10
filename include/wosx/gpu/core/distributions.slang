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

// This file defines Green's functions and their derivatives in free space
// and on ball domains for 2D and 3D Poisson and screened Poisson equations.
// These functions are used internally by the algorithms implemented in WoSX.

implementing wosx;
__include core.bessel;
__include core.sampling;

public interface IKernelRegularization<let DIM : int>
{
    // returns regularization for the Green's function
    static float regularizationForGreensFn(float r);

    // returns regularization for the Poisson Kernel
    static float regularizationForPoissonKernel(float r);
};

public struct KernelRegularization2D : IKernelRegularization<2>
{
    // returns regularization for the Green's function
    public static float regularizationForGreensFn(float r)
    {
        return 1.0;
    }

    // returns regularization for the Poisson Kernel
    public static float regularizationForPoissonKernel(float r)
    {
        // source: https://epubs.siam.org/doi/abs/10.1137/S0036142999362845
        return 1.0 - exp(-r * r);
    }
};

public struct KernelRegularization3D : IKernelRegularization<3>
{
    // returns regularization for the Green's function
    public static float regularizationForGreensFn(float r)
    {
        // source: https://arxiv.org/pdf/1508.00265.pdf
        return erf(r);
    }

    // returns regularization for the Poisson Kernel
    public static float regularizationForPoissonKernel(float r)
    {
        // source: https://arxiv.org/pdf/1508.00265.pdf
        return erf(r) - 2.0 * r * exp(-r * r) / sqrt(M_PI);
    }
};

public interface IGreensFnFreeSpace<let DIM : int>
{
    associatedtype KernelRegularization : IKernelRegularization<DIM>;

    // updates the absorption coefficient
    [mutating]
    void updateAbsorptionCoeff(float lambda_);

    // updates the pole
    [mutating]
    void updatePole(vector<float, DIM> x_, float rClamp_ = 1e-4);

    // returns the pole
    vector<float, DIM> getPole();

    // returns the radius clamp
    float getRadiusClamp();

    // evaluates the Green's function
    float evaluate(float r);

    // evaluates the gradient of the Green's function
    vector<float, DIM> gradient(float r, vector<float, DIM> y);

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    float poissonKernel(float r, vector<float, DIM> y, vector<float, DIM> n);

    // evaluates the gradient of the Poisson Kernel
    vector<float, DIM> poissonKernelGradient(float r, vector<float, DIM> y, vector<float, DIM> n);
};

public struct HarmonicGreensFnFreeSpace2D : IGreensFnFreeSpace<2>
{
    public typedef KernelRegularization2D KernelRegularization;
    private float2 x;
    private float rClamp;

    // constructor
    public __init()
    {
        updatePole(float2(0.0));
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        // do nothing
    }

    // updates the pole
    [mutating]
    public void updatePole(float2 x_, float rClamp_ = 1e-4)
    {
        x = x_;
        rClamp = rClamp_;
    }

    // returns the pole
    public float2 getPole()
    {
        return x;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        return -log(r) / (2.0 * M_PI);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float r, float2 y)
    {
        float2 xy = x - y;
        float r2 = r * r;

        return -xy / (2.0 * M_PI * r2);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float r, float2 y, float2 n)
    {
        float2 xy = x - y;
        float r2 = r * r;

        return dot(n, xy) / (2.0 * M_PI * r2);
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float r, float2 y, float2 n)
    {
        float2 xy = x - y;
        float r2 = r * r;

        return (n - 2.0 * (dot(n, xy) / r2) * xy) / (2.0 * M_PI * r2);
    }

    // evaluates the Green's function
    public float evaluate(float2 y)
    {
        float r = max(rClamp, length(y - x));
        return evaluate(r);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float2 y)
    {
        float r = max(rClamp, length(y - x));
        return gradient(r, y);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float2 y, float2 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernel(r, y, n);
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float2 y, float2 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernelGradient(r, y, n);
    }
};

public struct HarmonicGreensFnFreeSpace3D : IGreensFnFreeSpace<3>
{
    public typedef KernelRegularization3D KernelRegularization;
    private float3 x;
    private float rClamp;

    // constructor
    public __init()
    {
        updatePole(float3(0.0));
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        // do nothing
    }

    // updates the pole
    [mutating]
    public void updatePole(float3 x_, float rClamp_ = 1e-4)
    {
        x = x_;
        rClamp = rClamp_;
    }

    // returns the pole
    public float3 getPole()
    {
        return x;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        return 1.0 / (4.0 * M_PI * r);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float r, float3 y)
    {
        float3 xy = x - y;
        float r3 = r * r * r;

        return -xy / (4.0 * M_PI * r3);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float r, float3 y, float3 n)
    {
        float3 xy = x - y;
        float r3 = r * r * r;

        return dot(n, xy) / (4.0 * M_PI * r3);
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float r, float3 y, float3 n)
    {
        float3 xy = x - y;
        float r2 = r * r;
        float r3 = r * r * r;

        return (n - 3.0 * (dot(n, xy) / r2) * xy) / (4.0 * M_PI * r3);
    }

    // evaluates the Green's function
    public float evaluate(float3 y)
    {
        float r = max(rClamp, length(y - x));
        return evaluate(r);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float3 y)
    {
        float r = max(rClamp, length(y - x));
        return gradient(r, y);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float3 y, float3 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernel(r, y, n);
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float3 y, float3 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernelGradient(r, y, n);
    }
};

public struct YukawaGreensFnFreeSpace2D : IGreensFnFreeSpace<2>
{
    public typedef KernelRegularization2D KernelRegularization;
    private float2 x;
    private float rClamp;
    private float sqrtLambda;

    // constructor
    public __init()
    {
        updateAbsorptionCoeff(0.0);
        updatePole(float2(0.0));
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        sqrtLambda = sqrt(lambda_);
    }

    // updates the pole
    [mutating]
    public void updatePole(float2 x_, float rClamp_ = 1e-4)
    {
        x = x_;
        rClamp = rClamp_;
    }

    // returns the pole
    public float2 getPole()
    {
        return x;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        float mur = r * sqrtLambda;
        float K0mur = bessk0(mur);

        return K0mur / (2.0 * M_PI);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float r, float2 y)
    {
        float2 xy = x - y;
        float mur = r * sqrtLambda;
        float K1mur = bessk1(mur);
        float Qr = sqrtLambda * K1mur;

        return -xy * Qr / (2.0 * M_PI * r);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float r, float2 y, float2 n)
    {
        float2 xy = x - y;
        float mur = r * sqrtLambda;
        float K1mur = bessk1(mur);
        float Qr = sqrtLambda * K1mur;

        return dot(n, xy) * Qr / (2.0 * M_PI * r);
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float r, float2 y, float2 n)
    {
        float2 xy = x - y;
        float r2 = r * r;
        float mur = r * sqrtLambda;
        float K0mur = bessk0(mur);
        float K1mur = bessk1(mur);
        float K2mur = bessk(2, mur);
        float lambda = sqrtLambda * sqrtLambda;
        float Qr1 = sqrtLambda * K1mur;
        float Qr2 = lambda * (K0mur + K2mur) / 2.0;

        return (n * Qr1 - (dot(n, xy) / r2) * (Qr1 + r * Qr2) * xy) / (2.0 * M_PI * r);
    }

    // evaluates the Green's function
    public float evaluate(float2 y)
    {
        float r = max(rClamp, length(y - x));
        return evaluate(r);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float2 y)
    {
        float r = max(rClamp, length(y - x));
        return gradient(r, y);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float2 y, float2 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernel(r, y, n);
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float2 y, float2 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernelGradient(r, y, n);
    }
};

public struct YukawaGreensFnFreeSpace3D : IGreensFnFreeSpace<3>
{
    public typedef KernelRegularization3D KernelRegularization;
    private float3 x;
    private float rClamp;
    private float sqrtLambda;

    // constructor
    public __init()
    {
        updateAbsorptionCoeff(0.0);
        updatePole(float3(0.0));
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        sqrtLambda = sqrt(lambda_);
    }

    // updates the pole
    [mutating]
    public void updatePole(float3 x_, float rClamp_ = 1e-4)
    {
        x = x_;
        rClamp = rClamp_;
    }

    // returns the pole
    public float3 getPole()
    {
        return x;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);

        return expmur / (4.0 * M_PI * r);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float r, float3 y)
    {
        float3 xy = x - y;
        float r2 = r * r;
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float Qr = sqrtLambda * expmur * (1.0 + 1.0 / mur);

        return -xy * Qr / (4.0 * M_PI * r2);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float r, float3 y, float3 n)
    {
        float3 xy = x - y;
        float r2 = r * r;
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float Qr = sqrtLambda * expmur * (1.0 + 1.0 / mur);

        return dot(n, xy) * Qr / (4.0 * M_PI * r2);
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float r, float3 y, float3 n)
    {
        float3 xy = x - y;
        float r2 = r * r;
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float Qr1 = sqrtLambda * expmur * (1.0 + 1.0 / mur);
        float Qr2 = sqrtLambda * expmur / mur;

        return (n * Qr1 - (dot(n, xy) / r2) * (2.0 * Qr1 + Qr2) * xy) / (4.0 * M_PI * r2);
    }

    // evaluates the Green's function
    public float evaluate(float3 y)
    {
        float r = max(rClamp, length(y - x));
        return evaluate(r);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float3 y)
    {
        float r = max(rClamp, length(y - x));
        return gradient(r, y);
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel(float3 y, float3 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernel(r, y, n);
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float3 y, float3 n)
    {
        float r = max(rClamp, length(y - x));
        return poissonKernelGradient(r, y, n);
    }
};

public interface IGreensFnBall<let DIM : int>
{
    associatedtype SphereSampler : ISphereSampler<DIM>;

    // updates the absorption coefficient
    [mutating]
    void updateAbsorptionCoeff(float lambda_);

    // updates the ball center and radius
    [mutating]
    void updateBall(vector<float, DIM> c_, float R_, float rClamp_ = 1e-4);

    // returns the ball center
    vector<float, DIM> getBallCenter();

    // returns the ball radius
    float getBallRadius();

    // returns the radius clamp
    float getRadiusClamp();

    // samples a point inside the ball given the direction along which to sample the point
    vector<float, DIM> sampleVolume(vector<float, DIM> dir, inout pcg32 rng,
                                    out float r, out float pdf);

    // samples a point inside the ball
    vector<float, DIM> sampleVolume(inout pcg32 rng, out float r, out float pdf);

    // evaluates the Green's function
    float evaluate(float r);

    // evaluates the off-centered Green's function
    float evaluate(vector<float, DIM> x, vector<float, DIM> y);

    // evaluates the norm of the Green's function
    float norm();

    // evaluates the gradient norm of the Green's function
    float gradientNorm(float r);

    // evaluates the gradient of the Green's function
    vector<float, DIM> gradient(float r, vector<float, DIM> y);

    // samples a point on the surface of the ball
    vector<float, DIM> sampleSurface(inout pcg32 rng, out float pdf);

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    float poissonKernel();

    // evaluates the centered Poisson Kernel at a point y with normal n
    float poissonKernel(vector<float, DIM> y, vector<float, DIM> n);

    // directly evaluates the centered Poisson Kernel at a point y over the
    // direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
    float directionSampledPoissonKernel(vector<float, DIM> y);

    // computes the reflactance function for a Robin boundary condition
    float reflectance(float r, vector<float, DIM> dir, vector<float, DIM> n, float robinCoeff);

    // evaluates the gradient of the Poisson Kernel
    vector<float, DIM> poissonKernelGradient(vector<float, DIM> y);

    // returns the probability of a random walking reaching the boundary of the ball
    float potential();
};

internal vector<float, DIM> rejectionSampleGreensFn<let DIM : int, GreensFn : IGreensFnBall<DIM>>(GreensFn greensFn,
                                                                                                  vector<float, DIM> dir, float bound,
                                                                                                  inout pcg32 rng, out float r, out float pdf)
{
    vector<float, DIM> c = greensFn.getBallCenter();
    float R = greensFn.getBallRadius();
    float rClamp = greensFn.getRadiusClamp();
    float fnNorm = greensFn.norm();

    uint iter = 0;
    do
    {
        float u = rng.nextFloat();
        r = rng.nextFloat() * R;
        pdf = greensFn.evaluate(r) / fnNorm;
        float pdfRadius = pdf / greensFn.SphereSampler.pdfSampleSphereUniform(r);
        iter++;

        if (u < pdfRadius / bound)
        {
            break;
        }

    } while (iter < 1000);

    r = max(rClamp, r);
    if (r > R)
    {
        r = R / 2.0;
    }

    return c + r * dir;
}

public struct HarmonicGreensFnBall2D : IGreensFnBall<2>
{
    public typedef SphereSampler2D SphereSampler;
    private float2 c;
    private float R;
    private float rClamp;

    // constructor
    public __init()
    {
        updateBall(float2(0.0), 0.0);
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        // do nothing
    }

    // updates the ball center and radius
    [mutating]
    public void updateBall(float2 c_, float R_, float rClamp_ = 1e-4)
    {
        c = c_;
        R = R_;
        rClamp = rClamp_;
    }

    // returns the ball center
    public float2 getBallCenter()
    {
        return c;
    }

    // returns the ball radius
    public float getBallRadius()
    {
        return R;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // samples a point inside the ball given the direction along which to sample the point
    public float2 sampleVolume(float2 dir, inout pcg32 rng, out float r, out float pdf)
    {
        // sample radius r from pdf 4.0 * r * ln(R / r) / R^2
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();
        r = R * sqrt(u1 * u2);
        r = max(rClamp, r);
        if (r > R)
        {
            r = R / 2.0;
        }

        pdf = evaluate(r) / norm();
        return c + r * dir;
    }

    // samples a point inside the ball
    public float2 sampleVolume(inout pcg32 rng, out float r, out float pdf)
    {
        float2 dir = SphereSampler.sampleUnitSphereUniform(rng);

        return sampleVolume(dir, rng, r, pdf);
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        return log(R / r) / (2.0 * M_PI);
    }

    // evaluates the off-centered Green's function
    public float evaluate(float2 x, float2 y)
    {
        float r = max(rClamp, length(y - x));
        return (log(R * R - dot(x - c, y - c)) - log(R * r)) / (2.0 * M_PI);
    }

    // evaluates the norm of the Green's function
    public float norm()
    {
        return R * R / 4.0;
    }

    // evaluates the gradient norm of the Green's function
    public float gradientNorm(float r)
    {
        float r2 = r * r;
        return (1.0 / r2 - 1.0 / (R * R)) / (2.0 * M_PI);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float r, float2 y)
    {
        float2 d = y - c;
        return d * gradientNorm(r);
    }

    // samples a point on the surface of the ball
    public float2 sampleSurface(inout pcg32 rng, out float pdf)
    {
        float2 y = c + SphereSampler.sampleUnitSphereUniform(rng) * R;
        pdf = 1.0 / (2.0 * M_PI);

        return y;
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel()
    {
        return 1.0 / (2.0 * M_PI);
    }

    // evaluates the centered Poisson Kernel at a point y with normal n
    public float poissonKernel(float2 y, float2 n)
    {
        float2 xy = y - c;
        float xyNorm = length(xy);
        float r = max(rClamp, xyNorm);
        xy /= xyNorm;

        return dot(n, xy) / (2.0 * M_PI * r);
    }

    // directly evaluates the centered Poisson Kernel at a point y over the
    // direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
    public float directionSampledPoissonKernel(float2 y)
    {
        return 1.0;
    }

    // computes the reflactance function for Robin boundary conditions
    public float reflectance(float r, float2 dir, float2 n, float robinCoeff)
    {
        if (abs(robinCoeff) > 0.0)
        {
            float P = dot(n, dir) / r;
            float G = log(R / r);
            return 1.0 - robinCoeff * G / P;
        }

        return 1.0;
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float2 y)
    {
        float2 d = y - c;
        return 2.0 * d / (2.0 * M_PI * R * R);
    }

    // returns the probability of a random walking reaching the boundary of the ball
    public float potential()
    {
        return 2.0 * M_PI * poissonKernel();
    }
};

public struct HarmonicGreensFnBall3D : IGreensFnBall<3>
{
    public typedef SphereSampler3D SphereSampler;
    private float3 c;
    private float R;
    private float rClamp;

    // constructor
    public __init()
    {
        updateBall(float3(0.0), 0.0);
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        // do nothing
    }

    // updates the ball center and radius
    [mutating]
    public void updateBall(float3 c_, float R_, float rClamp_ = 1e-4)
    {
        c = c_;
        R = R_;
        rClamp = rClamp_;
    }

    // returns the ball center
    public float3 getBallCenter()
    {
        return c;
    }

    // returns the ball radius
    public float getBallRadius()
    {
        return R;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // samples a point inside the ball given the direction along which to sample the point
    public float3 sampleVolume(float3 dir, inout pcg32 rng, out float r, out float pdf)
    {
        // sample radius r from pdf 6.0f * r * (R - r) / R^3 using Ulrich's polar method
        float u1 = rng.nextFloat();
        float u2 = rng.nextFloat();
        float u3 = rng.nextFloat();
        float uMedian = max(min(u1, u2), min(max(u1, u2), u3));
        r = R * uMedian;
        r = max(rClamp, r);
        if (r > R)
        {
            r = R / 2.0;
        }

        pdf = evaluate(r) / norm();
        return c + r * dir;
    }

    // samples a point inside the ball
    public float3 sampleVolume(inout pcg32 rng, out float r, out float pdf)
    {
        float3 dir = SphereSampler.sampleUnitSphereUniform(rng);

        return sampleVolume(dir, rng, r, pdf);
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        return (1.0 / r - 1.0 / R) / (4.0 * M_PI);
    }

    // evaluates the off-centered Green's function
    public float evaluate(float3 x, float3 y)
    {
        float r = max(rClamp, length(y - x));
        return (1.0 / r - R / (R * R - dot(x - c, y - c))) / (4.0 * M_PI);
    }

    // evaluates the norm of the Green's function
    public float norm()
    {
        return R * R / 6.0;
    }

    // evaluates the gradient norm of the Green's function
    public float gradientNorm(float r)
    {
        float r3 = r * r * r;
        return (1.0 / r3 - 1.0 / (R * R * R)) / (4.0 * M_PI);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float r, float3 y)
    {
        float3 d = y - c;
        return d * gradientNorm(r);
    }

    // samples a point on the surface of the ball
    public float3 sampleSurface(inout pcg32 rng, out float pdf)
    {
        float3 y = c + SphereSampler.sampleUnitSphereUniform(rng) * R;
        pdf = 1.0 / (4.0 * M_PI);

        return y;
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel()
    {
        return 1.0 / (4.0 * M_PI);
    }

    // evaluates the centered Poisson Kernel at a point y with normal n
    public float poissonKernel(float3 y, float3 n)
    {
        float3 xy = y - c;
        float xyNorm = length(xy);
        float r = max(rClamp, xyNorm);
        float r2 = r * r;
        xy /= xyNorm;

        return dot(n, xy) / (4.0 * M_PI * r2);
    }

    // directly evaluates the centered Poisson Kernel at a point y over the
    // direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
    public float directionSampledPoissonKernel(float3 y)
    {
        return 1.0;
    }

    // computes the reflactance function for Robin boundary conditions
    public float reflectance(float r, float3 dir, float3 n, float robinCoeff)
    {
        if (abs(robinCoeff) > 0.0)
        {
            float r2 = r * r;
            float P = dot(n, dir) / r2;
            float G = 1.0 / r - 1.0 / R;
            return 1.0 - robinCoeff * G / P;
        }

        return 1.0;
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float3 y)
    {
        float3 d = y - c;
        return 3.0 * d / (4.0 * M_PI * R * R);
    }

    // returns the probability of a random walking reaching the boundary of the ball
    public float potential()
    {
        return 4.0 * M_PI * poissonKernel();
    }
};

public struct YukawaGreensFnBall2D : IGreensFnBall<2>
{
    public typedef SphereSampler2D SphereSampler;
    private float2 c;
    private float R;
    private float rClamp;
    private float sqrtLambda;
    private float K0muR;
    private float I0muR;
    private float K1muR;
    private float I1muR;

    // constructor
    public __init()
    {
        updateAbsorptionCoeff(0.0);
        updateBall(float2(0.0), 0.0);
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        sqrtLambda = sqrt(lambda_);
    }

    // updates the ball center and radius
    [mutating]
    public void updateBall(float2 c_, float R_, float rClamp_ = 1e-4)
    {
        c = c_;
        R = R_;
        rClamp = rClamp_;
        float muR = R * sqrtLambda;
        K0muR = bessk0(muR);
        I0muR = bessi0(muR);
        K1muR = bessk1(muR);
        I1muR = bessi1(muR);
    }

    // returns the ball center
    public float2 getBallCenter()
    {
        return c;
    }

    // returns the ball radius
    public float getBallRadius()
    {
        return R;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // samples a point inside the ball given the direction along which to sample the point
    public float2 sampleVolume(float2 dir, inout pcg32 rng, out float r, out float pdf)
    {
        // TODO: can probably do better
        // rejection sample radius r from pdf r * λ * (K_0(r√λ) * I_0(R√λ) - I_0(r√λ) * K_0(R√λ)) / (I_0(R√λ) - 1)
        float lambda = sqrtLambda * sqrtLambda;
        float bound = R <= lambda ?
                      max(max(2.2 / R, 2.2 / lambda), max(0.6 * sqrt(R), 0.6 * sqrtLambda)) :
                      max(min(2.2 / R, 2.2 / lambda), min(0.6 * sqrt(R), 0.6 * sqrtLambda));

        return rejectionSampleGreensFn<2, YukawaGreensFnBall2D>(this, dir, bound, rng, r, pdf);
    }

    // samples a point inside the ball
    public float2 sampleVolume(inout pcg32 rng, out float r, out float pdf)
    {
        float2 dir = SphereSampler.sampleUnitSphereUniform(rng);

        return sampleVolume(dir, rng, r, pdf);
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        float mur = r * sqrtLambda;
        float K0mur = bessk0(mur);
        float I0mur = bessi0(mur);

        return (K0mur - I0mur * K0muR / I0muR) / (2.0 * M_PI);
    }

    // evaluates the off-centered Green's function
    public float evaluate(float2 x, float2 y)
    {
        // NOTE: this is an approximation of the infinite series expression,
        // except when x coincides with c
        float r1 = max(rClamp, length(y - x));
        float r2 = (R * R - dot(x - c, y - c)) / R;
        float mur1 = r1 * sqrtLambda;
        float mur2 = r2 * sqrtLambda;
        float K0mur1 = bessk0(mur1);
        float K0mur2 = bessk0(mur2);
        float I0mur1 = bessi0(mur1);
        float I0mur2 = bessi0(mur2);
        float Q1 = K0mur1 - I0mur1 * K0muR / I0muR;
        float Q2 = K0mur2 - I0mur2 * K0muR / I0muR;

        return (Q1 - Q2) / (2.0 * M_PI);
    }

    // evaluates the norm of the Green's function
    public float norm()
    {
        float lambda = sqrtLambda * sqrtLambda;
        return (1.0 - 2.0 * M_PI * poissonKernel()) / lambda;
    }

    // evaluates the gradient norm of the Green's function
    public float gradientNorm(float r)
    {
        float mur = r * sqrtLambda;
        float K1mur = bessk1(mur);
        float I1mur = bessi1(mur);
        float Qr = sqrtLambda * (K1mur - I1mur * K1muR / I1muR);

        return Qr / (2.0 * M_PI * r);
    }

    // evaluates the gradient of the Green's function
    public float2 gradient(float r, float2 y)
    {
        float2 d = y - c;
        return d * gradientNorm(r);
    }

    // samples a point on the surface of the ball
    public float2 sampleSurface(inout pcg32 rng, out float pdf)
    {
        float2 y = c + SphereSampler.sampleUnitSphereUniform(rng) * R;
        pdf = 1.0 / (2.0 * M_PI);

        return y;
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel()
    {
        return 1.0 / (2.0 * M_PI * I0muR);
    }

    // evaluates the radial dampening factor associated with the centered Poisson Kernel
    private float poissonKernelDampeningFactor(float r)
    {
        float mur = r * sqrtLambda;
        float K1mur = bessk1(mur);
        float I1mur = bessi1(mur);
        float Q = K1mur + I1mur * K0muR / I0muR;

        return mur * Q;
    }

    // evaluates the centered Poisson Kernel at a point y with normal n
    public float poissonKernel(float2 y, float2 n)
    {
        float2 xy = y - c;
        float xyNorm = length(xy);
        float r = max(rClamp, xyNorm);
        xy /= xyNorm;

        return poissonKernelDampeningFactor(r) * dot(n, xy) / (2.0 * M_PI * r);
    }

    // directly evaluates the centered Poisson Kernel at a point y over the
    // direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
    public float directionSampledPoissonKernel(float2 y)
    {
        float r = max(rClamp, length(y - c));
        return poissonKernelDampeningFactor(r);
    }

    // computes the reflactance function for Robin boundary conditions
    public float reflectance(float r, float2 dir, float2 n, float robinCoeff)
    {
        float Q = poissonKernelDampeningFactor(r);
        if (abs(robinCoeff) > 0.0)
        {
            float P = dot(n, dir) / r;
            float mur = r * sqrtLambda;
            float K0mur = bessk0(mur);
            float I0mur = bessi0(mur);
            float G = K0mur - I0mur * K0muR / I0muR;

            return Q - robinCoeff * G / P;
        }

        return Q;
    }

    // evaluates the gradient of the Poisson Kernel
    public float2 poissonKernelGradient(float2 y)
    {
        float2 d = y - c;
        float QR = sqrtLambda / (R * I1muR);

        return d * QR / (2.0 * M_PI);
    }

    // returns the probability of a random walking reaching the boundary of the ball
    public float potential()
    {
        return 2.0 * M_PI * poissonKernel();
    }
};

public struct YukawaGreensFnBall3D : IGreensFnBall<3>
{
    public typedef SphereSampler3D SphereSampler;
    private float3 c;
    private float R;
    private float rClamp;
    private float sqrtLambda;
    private float muR;
    private float expmuR;
    private float sinhmuR;
    private float K32muR;
    private float I32muR;

    // constructor
    public __init()
    {
        updateAbsorptionCoeff(0.0);
        updateBall(float3(0.0), 0.0);
    }

    // updates the absorption coefficient
    [mutating]
    public void updateAbsorptionCoeff(float lambda_)
    {
        sqrtLambda = sqrt(lambda_);
    }

    // updates the ball center and radius
    [mutating]
    public void updateBall(float3 c_, float R_, float rClamp_ = 1e-4)
    {
        c = c_;
        R = R_;
        rClamp = rClamp_;
        muR = R * sqrtLambda;
        expmuR = exp(-muR);
        float exp2muR = expmuR * expmuR;
        float coshmuR = (1.0 + exp2muR) / (2.0 * expmuR);
        sinhmuR = (1.0 - exp2muR) / (2.0 * expmuR);
        K32muR = expmuR * (1.0 + 1.0 / muR);
        I32muR = coshmuR - sinhmuR / muR;
    }

    // returns the ball center
    public float3 getBallCenter()
    {
        return c;
    }

    // returns the ball radius
    public float getBallRadius()
    {
        return R;
    }

    // returns the radius clamp
    public float getRadiusClamp()
    {
        return rClamp;
    }

    // samples a point inside the ball given the direction along which to sample the point
    public float3 sampleVolume(float3 dir, inout pcg32 rng, out float r, out float pdf)
    {
        // TODO: can probably do better
        // rejection sample radius r from pdf r * λ * sinh((R - r)√λ) / (sinh(R√λ) - R√λ)
        float lambda = sqrtLambda * sqrtLambda;
        float bound = R <= lambda ?
                      max(max(2.0 / R, 2.0 / lambda), max(0.5 * sqrt(R), 0.5 * sqrtLambda)) :
                      max(min(2.0 / R, 2.0 / lambda), min(0.5 * sqrt(R), 0.5 * sqrtLambda));

        return rejectionSampleGreensFn<3, YukawaGreensFnBall3D>(this, dir, bound, rng, r, pdf);
    }

    // samples a point inside the ball
    public float3 sampleVolume(inout pcg32 rng, out float r, out float pdf)
    {
        float3 dir = SphereSampler.sampleUnitSphereUniform(rng);

        return sampleVolume(dir, rng, r, pdf);
    }

    // evaluates the Green's function
    public float evaluate(float r)
    {
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float sinhmur = (1.0 - expmur * expmur) / (2.0 * expmur);

        return (expmur - expmuR * sinhmur / sinhmuR) / (4.0 * M_PI * r);
    }

    // evaluates the off-centered Green's function
    public float evaluate(float3 x, float3 y)
    {
        // NOTE: this is an approximation of the infinite series expression,
        // except when x coincides with c
        float r1 = max(rClamp, length(y - x));
        float r2 = (R * R - dot(x - c, y - c)) / R;
        float mur1 = r1 * sqrtLambda;
        float mur2 = r2 * sqrtLambda;
        float expmur1 = exp(-mur1);
        float expmur2 = exp(-mur2);
        float sinhmur1 = (1.0 - expmur1 * expmur1) / (2.0 * expmur1);
        float sinhmur2 = (1.0 - expmur2 * expmur2) / (2.0 * expmur2);
        float Q1 = (expmur1 - expmuR * sinhmur1 / sinhmuR) / r1;
        float Q2 = (expmur2 - expmuR * sinhmur2 / sinhmuR) / r2;

        return (Q1 - Q2) / (4.0 * M_PI);
    }

    // evaluates the norm of the Green's function
    public float norm()
    {
        float lambda = sqrtLambda * sqrtLambda;
        return (1.0 - 4.0 * M_PI * poissonKernel()) / lambda;
    }

    // evaluates the gradient norm of the Green's function
    public float gradientNorm(float r)
    {
        float r2 = r * r;
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float exp2mur = expmur * expmur;
        float coshmur = (1.0 + exp2mur) / (2.0 * expmur);
        float sinhmur = (1.0 - exp2mur) / (2.0 * expmur);
        float K32mur = expmur * (1.0 + 1.0 / mur);
        float I32mur = coshmur - sinhmur / mur;
        float Qr = sqrtLambda * (K32mur - I32mur * K32muR / I32muR);

        return Qr / (4.0 * M_PI * r2);
    }

    // evaluates the gradient of the Green's function
    public float3 gradient(float r, float3 y)
    {
        float3 d = y - c;
        return d * gradientNorm(r);
    }

    // samples a point on the surface of the ball
    public float3 sampleSurface(inout pcg32 rng, out float pdf)
    {
        float3 y = c + SphereSampler.sampleUnitSphereUniform(rng) * R;
        pdf = 1.0 / (4.0 * M_PI);

        return y;
    }

    // evaluates the Poisson Kernel (normal derivative of the Green's function)
    public float poissonKernel()
    {
        return muR / (4.0 * M_PI * sinhmuR);
    }

    // evaluates the radial dampening factor associated with the centered Poisson Kernel
    private float poissonKernelDampeningFactor(float r)
    {
        float mur = r * sqrtLambda;
        float expmur = exp(-mur);
        float exp2mur = expmur * expmur;
        float coshmur = (1.0 + exp2mur) / (2.0 * expmur);
        float sinhmur = (1.0 - exp2mur) / (2.0 * expmur);
        float K32mur = expmur * (1.0 + 1.0 / mur);
        float I32mur = coshmur - sinhmur / mur;
        float Q = K32mur + I32mur * expmuR / sinhmuR;

        return mur * Q;
    }

    // evaluates the centered Poisson Kernel at a point y with normal n
    public float poissonKernel(float3 y, float3 n)
    {
        float3 xy = y - c;
        float xyNorm = length(xy);
        float r = max(rClamp, xyNorm);
        float r2 = r * r;
        xy /= xyNorm;

        return poissonKernelDampeningFactor(r) * dot(n, xy) / (4.0 * M_PI * r2);
    }

    // directly evaluates the centered Poisson Kernel at a point y over the
    // direction sampling pdf (i.e., HarmonicGreensFnFreeSpace::poissonKernel(y, n))
    public float directionSampledPoissonKernel(float3 y)
    {
        float r = max(rClamp, length(y - c));
        return poissonKernelDampeningFactor(r);
    }

    // computes the reflactance function for Robin boundary conditions
    public float reflectance(float r, float3 dir, float3 n, float robinCoeff)
    {
        float Q = poissonKernelDampeningFactor(r);
        if (abs(robinCoeff) > 0.0)
        {
            float r2 = r * r;
            float P = dot(n, dir) / r2;
            float mur = r * sqrtLambda;
            float expmur = exp(-mur);
            float sinhmur = (1.0 - expmur * expmur) / (2.0 * expmur);
            float G = (expmur - expmuR * sinhmur / sinhmuR) / r;

            return Q - robinCoeff * G / P;
        }

        return Q;
    }

    // evaluates the gradient of the Poisson Kernel
    public float3 poissonKernelGradient(float3 y)
    {
        float3 d = y - c;
        float lambda = sqrtLambda * sqrtLambda;
        float QR = lambda / I32muR;

        return d * QR / (4.0 * M_PI);
    }

    // returns the probability of a random walking reaching the boundary of the ball
    public float potential()
    {
        return 4.0 * M_PI * poissonKernel();
    }
};
