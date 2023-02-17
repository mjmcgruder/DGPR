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
  static constexpr u32 rank = 5;

  u32 nelem       = 0;
  u32 nbfp        = 0;
  u32 nvolqp      = 0;
  u32 solarr_size = 0;

  /* pre-computes */

  real* veval_ = nullptr;  // [bfuncs [qpoints]]
  real* vgrad_ = nullptr;  // [elems [bfuncs [direction [qpoints]]]]

  /* workspace */

  real* state_veval_ = nullptr;
  real* state_vgrad_ = nullptr;

  /* indexing */

  __forceinline__ __device__ real& index_state(real* U, u32 bi, u32 ei, u32 ri)
  {
    return U[nbfp * (rank * ei + ri) + bi];
  }

  __forceinline__ __device__ real& veval(u32 qi, u32 bi)
  {
    return veval_[nvolqp * bi + qi];
  }

  __forceinline__ __device__ real& vgrad(u32 qi, u32 bi, u32 ei, u32 di)
  {
    return vgrad_[nvolqp * (3 * (nbfp * ei + bi) + di) + qi];
  }

  __forceinline__ __device__ real& state_veval(u32 qi, u32 ri, u32 ei)
  {
    return state_veval_[nvolqp * (rank * ei + ri) + qi];
  }

  __forceinline__ __device__ real& state_vgrad(u32 qi, u32 di, u32 ri, u32 ei)
  {
    return state_vgrad_[nvolqp * (3 * (rank * ei + ri) + di) + qi];
  }

  /* sizing */

  __host__ u32 size_state_veval() const
  {
    return nvolqp * rank * nelem;
  }

  __host__ u32 size_state_vgrad() const
  {
    return nvolqp * 3 * rank * nelem;
  }
};

__host__ custore custore_make(shared_geometry* geom, simstate* U, real* d_U)
{
  custore store;
  core_geometry* core = &(geom->core);

  /* initialize state */

  cudaMemcpy(d_U, U->U.data, U->U.len * sizeof(real), cudaMemcpyHostToDevice);

  /* initialize pre-computes */

  store.solarr_size = U->size();
  store.nelem       = core->nelem;
  store.nbfp        = core->refp.nbf3d;
  store.nvolqp      = core->refp.vqrule.n;

  array<real, 2> shuffle_veval({1, 0}, core->refp.veval);
  cudaMalloc(&store.veval_, shuffle_veval.len * sizeof(real));
  cudaMemcpy(store.veval_, shuffle_veval.data, shuffle_veval.len * sizeof(real),
             cudaMemcpyHostToDevice);

  array<real, 4> shuffle_vgrad({2, 0, 1, 3}, geom->core.vgrad_);
  cudaMalloc(&store.vgrad_, shuffle_vgrad.len * sizeof(real));
  cudaMemcpy(store.vgrad_, shuffle_vgrad.data, 
             shuffle_vgrad.len * sizeof(real), cudaMemcpyHostToDevice);

  /* allocate (and initialize?) workspace */

  cudaMalloc(&store.state_veval_, store.size_state_veval() * sizeof(real));
  cudaMalloc(&store.state_vgrad_, store.size_state_vgrad() * sizeof(real));

  return store;
}

__host__ void custore_free(custore* store)
{
  cudaFree(store->veval_);
  cudaFree(store->vgrad_);
  cudaFree(store->state_veval_);
  cudaFree(store->state_vgrad_);
}

/* residual and related kernels --------------------------------------------- */

__global__ void cuda_evaluate_volume_states(real* U, custore store)
{
  u32 ei = blockIdx.x;

  real tmp = 0.;
  for (u32 i = threadIdx.x; i < store.nvolqp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nvolqp;    
    u32 qi = i - (ri * store.nvolqp);

    for (u32 bi = 0; bi < store.nbfp; ++bi)
    {
      tmp += store.index_state(U, bi, ei, ri) * store.veval(qi, bi);
    }

    store.state_veval(qi, ri, ei) = tmp;
  }
}

__global__ void cuda_evaluate_interior_face_states(real* U, custore store)
{
}

#define TILE_SIZE 16

__global__ void cuda_gemm(u32 m, u32 k, u32 n, real* A, real* B, real* C)
{
  __shared__ real tileA[TILE_SIZE][TILE_SIZE];
  __shared__ real tileB[TILE_SIZE][TILE_SIZE];

  u32 rT = threadIdx.y;
  u32 cT = threadIdx.x;
  u32 rC = TILE_SIZE * blockIdx.y + rT;
  u32 cC = TILE_SIZE * blockIdx.x + cT;

  real accC = 0.f;

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

__host__ void cuda_residual(custore store, real* d_R)
{
  cudaMemset(d_R, 0, store.solarr_size);
}
