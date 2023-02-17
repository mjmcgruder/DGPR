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
  u32 nfacqp      = 0;
  u32 niface      = 0;
  u32 nbface      = 0;
  u32 solarr_size = 0;

  /* pre-computes */

  real* veval_ = nullptr;  // [bfuncs [qpoints]]
  real* vgrad_ = nullptr;  // [elems [bfuncs [direction [qpoints]]]]
  real* feval_ = nullptr;  // [local face [bfuncs [qpoints]]]
  real* fgrad_ = nullptr;  // [elem [local face [bfuncs [direction [qpoints]]]]]

  s32* iflist = nullptr;  // [glo face, elem L, elem R, face L, face R]
  s32* bflist = nullptr;  // [glo face, elem L, bound type, face L] 

  /* workspace */

  real* state_veval_  = nullptr;
  real* state_vgrad_  = nullptr;
  real* state_ifeval_ = nullptr;
  real* state_ifgrad_ = nullptr;
  real* state_bfeval_ = nullptr;
  real* state_bfgrad_ = nullptr;

  /* indexing */

  __forceinline__ __device__ real& index_state(real* U, u32 bi, u32 ri, u32 ei)
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

  __forceinline__ __device__ real& feval(u32 qi, u32 bi, u32 fi)
  {
    return feval_[nfacqp * (nbfp * fi + bi) + qi];
  }

  __forceinline__ __device__ real& fgrad(u32 qi, u32 di, u32 bi, u32 fi, u32 ei)
  {
    return fgrad_[nfacqp * (3 * (nbfp * (6 * ei + fi) + bi) + di) + qi];
  }

  __forceinline__ __device__ real& state_veval(u32 qi, u32 ri, u32 ei)
  {
    return state_veval_[nvolqp * (rank * ei + ri) + qi];
  }

  __forceinline__ __device__ real& state_vgrad(u32 qi, u32 di, u32 ri, u32 ei)
  {
    return state_vgrad_[nvolqp * (3 * (rank * ei + ri) + di) + qi];
  }

  __forceinline__ __device__ real& state_ifeval(u32 qi, u32 ri, u32 si, u32 fi)
  {
    return state_ifeval_[nfacqp * (rank * (2 * fi + si) + ri) + qi];
  }

  __forceinline__ __device__ real& state_ifgrad(u32 qi, u32 di, u32 ri, u32 si,
                                                u32 fi)
  {
    return state_ifgrad_[nfacqp * (3 * (rank * (2 * fi + si) + ri) + di) + qi];
  }

  __forceinline__ __device__ real& state_bfeval(u32 qi, u32 ri, u32 fi)
  {
    return state_bfeval_[nfacqp * (rank * fi + ri) + qi];
  }

  __forceinline__ __device__ real& state_bfgrad(u32 qi, u32 di, u32 ri, u32 fi)
  {
    return state_bfgrad_[nfacqp * (3 * (rank * fi + ri) + di) + qi];
  }

  /* sizing */

  __host__ u32 size_state_veval() const
  {
    return nvolqp * rank * nelem;
  }

  __host__ u32 size_state_vgrad() const
  {
    return size_state_veval() * 3;
  }

  __host__ u32 size_state_ifeval() const
  {
    return 2 * nfacqp * rank * niface;
  }

  __host__ u32 size_state_ifgrad() const
  {
    return size_state_ifeval() * 3;
  }

  __host__ u32 size_state_bfeval() const
  {
    return nfacqp * rank * nbface;
  }

  __host__ u32 size_state_bfgrad() const
  {
    return size_state_bfeval() * 3;
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
  store.nfacqp      = core->refp.fqrule.n;
  store.niface      = geom->num_interior_faces();
  store.nbface      = geom->num_boundary_faces();

  array<real, 2> shuffle_veval({1, 0}, core->refp.veval);
  cudaMalloc(&store.veval_, shuffle_veval.len * sizeof(real));
  cudaMemcpy(store.veval_, shuffle_veval.data, shuffle_veval.len * sizeof(real),
             cudaMemcpyHostToDevice);

  array<real, 4> shuffle_vgrad({2, 0, 1, 3}, geom->core.vgrad_);
  cudaMalloc(&store.vgrad_, shuffle_vgrad.len * sizeof(real));
  cudaMemcpy(store.vgrad_, shuffle_vgrad.data, 
             shuffle_vgrad.len * sizeof(real), cudaMemcpyHostToDevice);

  array<real, 3> shuffle_feval({1, 0, 2}, geom->core.refp.feval);
  cudaMalloc(&store.feval_, shuffle_feval.len * sizeof(real));
  cudaMemcpy(store.feval_, shuffle_feval.data, shuffle_feval.len * sizeof(real),
             cudaMemcpyHostToDevice);

  array<real, 5> shuffle_fgrad({2, 0, 1, 3, 4}, geom->core.fgrad_);
  cudaMalloc(&store.fgrad_, shuffle_fgrad.len * sizeof(real));
  cudaMemcpy(store.fgrad_, shuffle_fgrad.data, shuffle_fgrad.len * sizeof(real),
             cudaMemcpyHostToDevice);

  array<s32> iflist(geom->interior_face_list.len);
  for (u32 i = 0; i < iflist.len; ++i) 
    iflist[i] = (s32)geom->interior_face_list[i];
  cudaMalloc(&store.iflist, iflist.len * sizeof(s32));
  cudaMemcpy(store.iflist, iflist.data, iflist.len * sizeof(s32),
             cudaMemcpyHostToDevice);

  array<s32> bflist(geom->boundary_face_list.len);
  for (u32 i = 0; i < bflist.len; ++i)
    bflist[i] = (s32)geom->boundary_face_list[i];
  cudaMalloc(&store.bflist, bflist.len * sizeof(s32));
  cudaMemcpy(store.bflist, bflist.data, bflist.len * sizeof(s32),
             cudaMemcpyHostToDevice);

  /* allocate (and initialize?) workspace */

  cudaMalloc(&store.state_veval_, store.size_state_veval() * sizeof(real));
  cudaMalloc(&store.state_vgrad_, store.size_state_vgrad() * sizeof(real));
  cudaMalloc(&store.state_ifeval_, store.size_state_ifeval() * sizeof(real));
  cudaMalloc(&store.state_ifgrad_, store.size_state_ifgrad() * sizeof(real));
  cudaMalloc(&store.state_bfeval_, store.size_state_bfeval() * sizeof(real));
  cudaMalloc(&store.state_bfgrad_, store.size_state_bfgrad() * sizeof(real));

  return store;
}

__host__ void custore_free(custore* store)
{
  cudaFree(store->veval_);
  cudaFree(store->vgrad_);
  cudaFree(store->feval_);
  cudaFree(store->fgrad_);
  cudaFree(store->iflist);
  cudaFree(store->bflist);
  cudaFree(store->state_veval_);
  cudaFree(store->state_vgrad_);
  cudaFree(store->state_ifeval_);
  cudaFree(store->state_ifgrad_);
  cudaFree(store->state_bfeval_);
  cudaFree(store->state_bfgrad_);
}

/* residual and related kernels --------------------------------------------- */

__global__ void cuda_evaluate_volume_states(real* U, custore store)
{
  u32 ei = blockIdx.x;

  for (u32 i = threadIdx.x; i < store.nvolqp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nvolqp;    
    u32 qi = i - (ri * store.nvolqp);

    real tmp_st = 0., tmp_dx = 0., tmp_dy = 0., tmp_dz = 0.;
    for (u32 bi = 0; bi < store.nbfp; ++bi)
    {
      real state = store.index_state(U, bi, ri, ei);
      tmp_st += state * store.veval(qi, bi);
      tmp_dx += state * store.vgrad(qi, bi, ei, 0);
      tmp_dy += state * store.vgrad(qi, bi, ei, 1);
      tmp_dz += state * store.vgrad(qi, bi, ei, 2);
    }

    store.state_veval(qi, ri, ei)    = tmp_st;
    store.state_vgrad(qi, 0, ri, ei) = tmp_dx;
    store.state_vgrad(qi, 1, ri, ei) = tmp_dy;
    store.state_vgrad(qi, 2, ri, ei) = tmp_dz;
  }
}

__global__ void cuda_evaluate_interior_face_states(real* U, custore store)
{
  u32 fi  = blockIdx.x;
  u32 eL  = (u32)store.iflist[5 * fi + 1];
  u32 eR  = (u32)store.iflist[5 * fi + 2];
  u32 lfL = (u32)store.iflist[5 * fi + 3];
  u32 lfR = (u32)store.iflist[5 * fi + 4];

  for (u32 i = threadIdx.x; i < store.nfacqp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nfacqp;
    u32 qi = i - (ri * store.nfacqp);

    real tmp_stL = 0., tmp_dxL = 0., tmp_dyL = 0., tmp_dzL = 0.;
    real tmp_stR = 0., tmp_dxR = 0., tmp_dyR = 0., tmp_dzR = 0.;
    for (u32 bi = 0; bi < store.nbfp; ++bi)
    {
      real stateL = store.index_state(U, bi, ri, eL);
      real stateR = store.index_state(U, bi, ri, eR);

      tmp_stL += stateL * store.feval(qi, bi, lfL);
      tmp_dxL += stateL * store.fgrad(qi, 0, bi, lfL, eL);
      tmp_dyL += stateL * store.fgrad(qi, 1, bi, lfL, eL);
      tmp_dzL += stateL * store.fgrad(qi, 2, bi, lfL, eL);

      tmp_stR += stateR * store.feval(qi, bi, lfR);
      tmp_dxR += stateR * store.fgrad(qi, 0, bi, lfR, eR);
      tmp_dyR += stateR * store.fgrad(qi, 1, bi, lfR, eR);
      tmp_dzR += stateR * store.fgrad(qi, 2, bi, lfR, eR);
    }

    store.state_ifeval(qi, ri, 0, fi)    = tmp_stL;
    store.state_ifgrad(qi, 0, ri, 0, fi) = tmp_dxL;
    store.state_ifgrad(qi, 1, ri, 0, fi) = tmp_dyL;
    store.state_ifgrad(qi, 2, ri, 0, fi) = tmp_dzL;
    store.state_ifeval(qi, ri, 1, fi)    = tmp_stR;
    store.state_ifgrad(qi, 0, ri, 1, fi) = tmp_dxR;
    store.state_ifgrad(qi, 1, ri, 1, fi) = tmp_dyR;
    store.state_ifgrad(qi, 2, ri, 1, fi) = tmp_dzR;
  }
}

__global__ void cuda_evaluate_boundary_face_states(real* U, custore store)
{
  u32 fi  = blockIdx.x;
  u32 eL  = (u32)store.bflist[4 * fi + 1];
  u32 lfL = (u32)store.bflist[4 * fi + 3];
  
  for (u32 i = threadIdx.x; i < store.nfacqp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nfacqp;
    u32 qi = i - (ri * store.nfacqp);

    real tmp_stL = 0., tmp_dxL = 0., tmp_dyL = 0., tmp_dzL = 0.;
    for (u32 bi = 0; bi < store.nbfp; ++bi)
    {
      real stateL = store.index_state(U, bi, ri, eL);

      tmp_stL += stateL * store.feval(qi, bi, lfL);
      tmp_dxL += stateL * store.fgrad(qi, 0, bi, lfL, eL);
      tmp_dyL += stateL * store.fgrad(qi, 1, bi, lfL, eL);
      tmp_dzL += stateL * store.fgrad(qi, 2, bi, lfL, eL);
    }

    store.state_bfeval(qi, ri, fi)    = tmp_stL;
    store.state_bfgrad(qi, 0, ri, fi) = tmp_dxL;
    store.state_bfgrad(qi, 1, ri, fi) = tmp_dyL;
    store.state_bfgrad(qi, 2, ri, fi) = tmp_dzL;
  }
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
