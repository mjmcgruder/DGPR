/* DGPR: Discontinuous Galerkin Performance Research                          */
/* Copyright (C) 2023  Miles McGruder                                         */
/*                                                                            */
/* This program is free software: you can redistribute it and/or modify       */
/* it under the terms of the GNU General Public License as published by       */
/* the Free Software Foundation, either version 3 of the License, or          */
/* (at your option) any later version.                                        */
/*                                                                            */
/* This program is distributed in the hope that it will be useful,            */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of             */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              */
/* GNU General Public License for more details.                               */
/*                                                                            */
/* You should have received a copy of the GNU General Public License          */
/* along with this program.  If not, see <https://www.gnu.org/licenses/>.     */

#pragma once

#include <cinttypes>

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

#include "solve_cuda.cu"
#include "time.cpp"

struct cu_square
{
  __device__ float operator()(float x) const
  {
    return x * x;
  }
};

void residual_norm(u32 solarr_size, float* residual, u64 tstep)
{
  cu_square square;
  thrust::device_ptr<float> dptr_residual(residual);

  float sumsq =
  thrust::transform_reduce(dptr_residual, dptr_residual + solarr_size, square,
                           (float)0., thrust::plus<float>());

  float Rnorm = sqrt((1. / ((float)solarr_size)) * sumsq);
  printf("%" PRIu64 " : %.17e\n", tstep, Rnorm);
}

__global__ void tvdRK3_cuda_acc1(u32 solarr_size, float dt, float* U, float* U1,
                                 float* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U1[i] = U[i] + dt * f[i];
  }
}

__global__ void tvdRK3_cuda_acc2(u32 solarr_size, float dt, float* U, float* U1,
                                 float* U2, float* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U2[i] = 0.75 * U[i] + 0.25 * U1[i] + 0.25 * dt * f[i];
  }
}

__global__ void tvdRK3_cuda_acc3(u32 solarr_size, float dt, float* U, float* U1,
                                 float* U2, float* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U[i] = (1. / 3.) * U[i] + (2. / 3.) * U2[i] + (2. / 3.) * dt * f[i];
  }
}

__global__ void RK4_cuda_acc1(u32 solarr_size, float dt, float* U, float* f1,
                              float* U1)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U1[i] = U[i] + dt * f1[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc2(u32 solarr_size, float dt, float* U, float* f2,
                              float* U2)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U2[i] = U[i] + dt * f2[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc3(u32 solarr_size, float dt, float* U, float* f3,
                              float* U3)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U3[i] = U[i] + dt * f3[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc4(u32 solarr_size, float dt, float* U, float* f1,
                              float* f2, float* f3, float* f4)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U[i] += (1.f / 6.f) * (f1[i] + 2.f * f2[i] + 2.f * f3[i] + f4[i]) * dt;
  }
}

void tvdRK3_cuda(u64 tstep, float dt, cuda_device_geometry h_cugeom,
                 float* state, parameters params, cuda_residual_workspace wsp)
{
  float* residual = wsp.aux[0];
  float* U1       = wsp.aux[1];
  float* U2       = wsp.aux[2];
  float* f        = wsp.aux[3];

  cuda_residual(state, h_cugeom, params, wsp, residual, f);
  tvdRK3_cuda_acc1<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(
  wsp.solarr_size, dt, state, U1, f);
  cudaDeviceSynchronize();

  residual_norm(wsp.solarr_size, residual, tstep);

  cuda_residual(U1, h_cugeom, params, wsp, residual, f);
  tvdRK3_cuda_acc2<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(
  wsp.solarr_size, dt, state, U1, U2, f);
  cudaDeviceSynchronize();

  cuda_residual(U2, h_cugeom, params, wsp, residual, f);
  tvdRK3_cuda_acc3<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(
  wsp.solarr_size, dt, state, U1, U2, f);
  cudaDeviceSynchronize();
}

void RK4_cuda(u64 tstep, float dt, cuda_device_geometry h_cugeom,
              float* state, parameters params, cuda_residual_workspace wsp)
{
  float* residual = wsp.aux[0];
  float* U1       = wsp.aux[1];
  float* U2       = wsp.aux[2];
  float* U3       = wsp.aux[3];
  float* f1       = wsp.aux[4];
  float* f2       = wsp.aux[5];
  float* f3       = wsp.aux[6];
  float* f4       = wsp.aux[7];

  cuda_residual(state, h_cugeom, params, wsp, residual, f1);
  RK4_cuda_acc1<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(wsp.solarr_size, dt,
                                                          state, f1, U1);
  cudaDeviceSynchronize();

  residual_norm(wsp.solarr_size, residual, tstep);

  cuda_residual(U1, h_cugeom, params, wsp, residual, f2);
  RK4_cuda_acc2<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(wsp.solarr_size, dt,
                                                          state, f2, U2);
  cudaDeviceSynchronize();

  cuda_residual(U2, h_cugeom, params, wsp, residual, f3);
  RK4_cuda_acc3<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(wsp.solarr_size, dt,
                                                          state, f3, U3);
  cudaDeviceSynchronize();

  cuda_residual(U3, h_cugeom, params, wsp, residual, f4);
  RK4_cuda_acc4<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(
  wsp.solarr_size, dt, state, f1, f2, f3, f4);
  cudaDeviceSynchronize();
}
