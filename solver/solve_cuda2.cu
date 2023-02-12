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

#include "helper_types.cpp"
#include "geometry_shared.cpp"

/* struct storing pre-computes and workspace values on the device side ------ */

struct custore
{
  u32 nelem       = 0;
  u32 nbfp        = 0;
  u32 nvolqp      = 0;
  u32 solarr_size = 0;

  float* vgrad_;  // [direction [elements [test funcs [qpoints]]]]

  __forceinline__ float index_state(float* U, u32 bi, u32 ei, u32 ri)
  {
    return U[nbfp * (nelem * ri + ei) + bi];
  }

  __forceinline__ float vgrad(u32 qi, u32 ti, u32 ei, u32 di)
  {
    return vgrad_[nvolqp * (nbfp * (nelem * di + ei) + ti) + qi];
  }
};

__host__ custore custore_make(shared_geometry* geom, simstate* U, float* d_U)
{
  custore store;
  core_geometry* core = &(geom->core);

  /* initialize state */

  array<float, 3> shuffle_U({0, 2, 1}, U->U);
  cudaMemcpy(d_U, shuffle_U.data, shuffle_U.len * sizeof(float),
             cudaMemcpyHostToDevice);

  /* initialize pre-computes */

  store.solarr_size = U->size();
  store.nelem       = core->nelem;
  store.nbfp        = core->refp.nbf3d;
  store.nvolqp      = core->refp.vqrule.n;

  // volume basis function gradient evaluations
  array<float, 4> shuffle_vgrad({1, 2, 0, 3}, geom->core.vgrad_);
  cudaMalloc(&store.vgrad_, shuffle_vgrad.len * sizeof(float));
  cudaMemcpy(store.vgrad_, shuffle_vgrad.data, 
             shuffle_vgrad.len * sizeof(float), cudaMemcpyHostToDevice);

  return store;
}

__host__ void custore_free(custore* store)
{
  cudaFree(store->vgrad_);
}

/* residual and related kernels --------------------------------------------- */

__global__ void cuda_evaluate_volume_gradients(custore store, float* U)
{

}

#define TILE_SIZE 16

__global__ void cuda_gemm(u32 m, u32 k, u32 n, float* A, float* B, float* C)
{
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  u32 rT = threadIdx.y;
  u32 cT = threadIdx.x;
  u32 rC = TILE_SIZE * blockIdx.y + rT;
  u32 cC = TILE_SIZE * blockIdx.x + cT;

  float accC = 0.f;

  for (u32 bk = 0; bk < (k + TILE_SIZE - 1) / TILE_SIZE; ++bk)
  {
    if ((rC) < m && (TILE_SIZE * bk + cT) < k)
      tileA[rT][cT] = A[k * (rC) + (TILE_SIZE * bk + cT)];
    else
      tileA[rT][cT] = 0.f;

    if ((TILE_SIZE * bk + rT) < k && (cC) < n)
      tileB[rT][cT] = B[n * (TILE_SIZE * bk + rT) + (cC)];
    else
      tileB[rT][cT] = 0.f;

    __syncthreads();

    for (u32 i = 0; i < TILE_SIZE; ++i)
      accC += tileA[rT][i] * tileB[i][cT];

    __syncthreads();
  }

  if (rC < m && cC < n)
    C[n * rC + cC] = accC;
}

__host__ void cuda_residual(custore store, float* d_R)
{
  cudaMemset(d_R, 0, store.solarr_size);
}
