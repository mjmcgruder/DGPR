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

#include <helper_types.cpp>

// naive matrix multiplication
__global__ void cuda_gemm_0(u32 m, u32 k, u32 n, float* A, float* B, float* C)
{
  u32 rC = blockDim.y * blockIdx.y + threadIdx.y;
  u32 cC = blockDim.x * blockIdx.x + threadIdx.x;

  if (rC > m || cC > n)
    return;

  float accC = 0.f;
  for (u32 i = 0; i < k; ++i)
  {
    accC += A[k * rC + i] * B[n * i + cC];
  }

  C[n * rC + cC] = accC;
}

#define TILE_SIZE 16

// reasonable matrix multiplication
__global__ void cuda_gemm_1(u32 m, u32 k, u32 n, float* A, float* B, float* C)
{
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  u32 rT = threadIdx.y;
  u32 cT = threadIdx.x;
  u32 rC = TILE_SIZE * blockIdx.y + rT;
  u32 cC = TILE_SIZE * blockIdx.x + cT;

  float accC = 0.;

  for (u32 bk = 0; bk < (k + TILE_SIZE - 1) / TILE_SIZE; ++bk)
  {
    if ((rC) < m && (TILE_SIZE * bk + cT) < k)
      tileA[rT][cT] = A[k * (rC) + (TILE_SIZE * bk + cT)];
    else
      tileA[rT][cT] = 0.;

    if ((TILE_SIZE * bk + rT) < k && (cC) < n)
      tileB[rT][cT] = B[n * (TILE_SIZE * bk + rT) + (cC)];
    else
      tileB[rT][cT] = 0.;

    __syncthreads();

    for (uint i = 0; i < TILE_SIZE; ++i)
      accC += tileA[rT][i] * tileB[i][cT];

    __syncthreads();
  }

  if (rC < m && cC < n)
    C[n * rC + cC] = accC;
}
