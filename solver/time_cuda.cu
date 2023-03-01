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

#include "solve_cuda2.cu"
#include "time.cpp"

struct cu_square
{
  __device__ real operator()(real x) const
  {
    return x * x;
  }
};

void residual_norm(u32 solarr_size, real* residual, u64 tstep)
{
  cu_square square;
  thrust::device_ptr<real> dptr_residual(residual);

  real sumsq =
  thrust::transform_reduce(dptr_residual, dptr_residual + solarr_size, square,
                           (real)0., thrust::plus<real>());

  real Rnorm = sqrt((1. / ((real)solarr_size)) * sumsq);
  printf("%" PRIu64 " : %.17e\n", tstep, Rnorm);
}

__global__ void tvdRK3_cuda_fe(u32 solarr_size, real dt, real* U, real* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U[i] = U[i] + dt * f[i];
  }
}

__global__ void tvdRK3_cuda_acc1(u32 solarr_size, real dt, real* U, real* U1,
                                 real* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U1[i] = U[i] + dt * f[i];
  }
}

__global__ void tvdRK3_cuda_acc2(u32 solarr_size, real dt, real* U, real* U1,
                                 real* U2, real* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U2[i] = 0.75 * U[i] + 0.25 * U1[i] + 0.25 * dt * f[i];
  }
}

__global__ void tvdRK3_cuda_acc3(u32 solarr_size, real dt, real* U, real* U1,
                                 real* U2, real* f)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U[i] = (1. / 3.) * U[i] + (2. / 3.) * U2[i] + (2. / 3.) * dt * f[i];
  }
}

__global__ void RK4_cuda_acc1(u32 solarr_size, real dt, real* U, real* f1,
                              real* U1)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U1[i] = U[i] + dt * f1[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc2(u32 solarr_size, real dt, real* U, real* f2,
                              real* U2)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U2[i] = U[i] + dt * f2[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc3(u32 solarr_size, real dt, real* U, real* f3,
                              real* U3)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U3[i] = U[i] + dt * f3[i] * 0.5f;
  }
}

__global__ void RK4_cuda_acc4(u32 solarr_size, real dt, real* U, real* f1,
                              real* f2, real* f3, real* f4)
{
  u32 i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < solarr_size)
  {
    U[i] += (1.f / 6.f) * (f1[i] + 2.f * f2[i] + 2.f * f3[i] + f4[i]) * dt;
  }
}

void tvdRK3_cuda(u64 tstep, real dt, custore store, cuworkspace wsp,
                 parameters params, real* U)
{
  real* residual = wsp.aux[0];
  real* U1       = wsp.aux[1];
  real* U2       = wsp.aux[2];
  real* f        = wsp.aux[3];

  // cuda_residual(store, params, U, residual, f);
  // tvdRK3_cuda_acc1<<<idiv_ceil(store.solarr_size, 256), 256>>>(
  // store.solarr_size, dt, U, U1, f);
  // cudaDeviceSynchronize();

  cuda_residual(store, params, U, residual, f);
  tvdRK3_cuda_fe<<<idiv_ceil(store.solarr_size, 256), 256>>>(
  store.solarr_size, dt, U, f);
  cudaDeviceSynchronize();

  residual_norm(store.solarr_size, residual, tstep);

  // cuda_residual(store, params, U1, residual, f);
  // tvdRK3_cuda_acc2<<<idiv_ceil(store.solarr_size, 256), 256>>>(
  // store.solarr_size, dt, U, U1, U2, f);
  // cudaDeviceSynchronize();

  // cuda_residual(store, params, U2, residual, f);
  // tvdRK3_cuda_acc3<<<idiv_ceil(store.solarr_size, 256), 256>>>(
  // store.solarr_size, dt, U, U1, U2, f);
  // cudaDeviceSynchronize();
}

void RK4_cuda(u64 tstep, real dt, custore store, cuworkspace wsp,
              parameters params, real* U)
{
  real* residual = wsp.aux[0];
  real* U1       = wsp.aux[1];
  real* U2       = wsp.aux[2];
  real* U3       = wsp.aux[3];
  real* f1       = wsp.aux[4];
  real* f2       = wsp.aux[5];
  real* f3       = wsp.aux[6];
  real* f4       = wsp.aux[7];

  cuda_residual(store, params, U, residual, f1);
  RK4_cuda_acc1<<<idiv_ceil(store.solarr_size, 256), 256>>>(store.solarr_size,
                                                            dt, U, f1, U1);
  cudaDeviceSynchronize();

  residual_norm(store.solarr_size, residual, tstep);

  cuda_residual(store, params, U1, residual, f2);
  RK4_cuda_acc2<<<idiv_ceil(store.solarr_size, 256), 256>>>(store.solarr_size,
                                                            dt, U, f2, U2);
  cudaDeviceSynchronize();

  cuda_residual(store, params, U2, residual, f3);
  RK4_cuda_acc3<<<idiv_ceil(store.solarr_size, 256), 256>>>(store.solarr_size,
                                                            dt, U, f3, U3);
  cudaDeviceSynchronize();

  cuda_residual(store, params, U3, residual, f4);
  RK4_cuda_acc4<<<idiv_ceil(store.solarr_size, 256), 256>>>(
  store.solarr_size, dt, U, f1, f2, f3, f4);
  cudaDeviceSynchronize();
}
