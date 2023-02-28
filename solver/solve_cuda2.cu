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
#include "solver_params.cpp"
#include "flux.cpp"
#include "bound_states.cpp"

/* struct storing pre-computes and workspace values on the device side ------ */

struct custore
{
  static constexpr u32 rank = 5;

  /* counts */

  u32 nelem       = 0;
  u32 nbfp        = 0;
  u32 nvolqp      = 0;
  u32 nfacqp      = 0;
  u32 niface      = 0;
  u32 nbface      = 0;
  u32 solarr_size = 0;

  /* pre-computes */

  real* Minv_   = nullptr;

  real* veval_  = nullptr;  // [bfuncs [qpoints]]
  real* vgrad_  = nullptr;  // [elems [bfuncs [direction [qpoints]]]]
  real* feval_  = nullptr;  // [local face [bfuncs [qpoints]]]
  real* fgrad_  = nullptr;  // [elem [local face [bfuncs [direction [qpoints]]]]]

  real* vquadw_ = nullptr;  // [qpoints]
  real* fquadw_ = nullptr;  // [qpoints]
  real* vJ_     = nullptr;  // [elem [qpoints]]
  real* fJ_     = nullptr;  // [face (global) [qpoint]]

  real* n_     = nullptr;  // [face num [ dim [qpoints]]]
  real* h_int_ = nullptr;  // [side [interior face]]
  real* h_bnd_ = nullptr;  // [boundary face]

  s32* iflist = nullptr;  // [glo face, elem L, elem R, face L, face R]
  s32* bflist = nullptr;  // [glo face, elem L, bound type, face L] 

  /* workspace */

  real* state_veval_  = nullptr;
  real* state_vgrad_  = nullptr;
  real* state_ifeval_ = nullptr;
  real* state_ifgrad_ = nullptr;
  real* state_bfeval_ = nullptr;
  real* state_bfgrad_ = nullptr;

  real* flux_vol_ = nullptr;
  real* flux_int_ = nullptr;
  real* flux_bnd_ = nullptr;

  real* integral_face_ = nullptr;

  /* indexing */

  __forceinline__ __device__ real& index_state(real* U, u32 bi, u32 ri, u32 ei)
  {
    return U[nbfp * (rank * ei + ri) + bi];
  }

  __forceinline__ __device__ real* Minv(u32 ei)
  {
    return Minv_ + ((nbfp * (nbfp + 1)) / 2) * ei;
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

  __forceinline__ __device__ real& n(u32 qi, u32 di, u32 gfi)
  {
    return n_[nfacqp * (3 * gfi + di) + qi];
  }

  __forceinline__ __device__ real& h_int(u32 ifi, u32 si)
  {
    return h_int_[niface * si + ifi];
  }

  __forceinline__ __device__ real& h_bnd(u32 bfi)
  {
    return h_bnd_[bfi];
  }

  __forceinline__ __device__ real& vquadw(u32 qi)
  {
    return vquadw_[qi];
  }

  __forceinline__ __device__ real& fquadw(u32 qi)
  {
    return fquadw_[qi];
  }

  __forceinline__ __device__ real& vJ(u32 qi, u32 ei)
  {
    return vJ_[nvolqp * (ei) + qi];
  }

  __forceinline__ __device__ real& fJ(u32 qi, u32 gfi)
  {
    return fJ_[nfacqp * gfi + qi];
  }

  __forceinline__ __device__ real& integral_face(u32 ti, u32 ri, u32 lfi,
                                                 u32 ei)
  {
    return integral_face_[nbfp * (rank * (6 * ei + lfi) + ri) + ti];
  }

  // Uses "vector iterator" vi

  // vectors are stored in this order: Fx Fy Fz Qx Qy Qz
  __forceinline__ __device__  real& flux_vol(u32 qi, u32 vi, u32 ri, u32 ei)
  {
    return flux_vol_[nvolqp * (6 * (rank * ei + ri) + vi) + qi];
  }

  // vectors stored in this order: F dcQLx dcQLy dcQLz dcQRx dcQRy dcQRz Qh
  __forceinline__ __device__ real& flux_int(u32 qi, u32 vi, u32 ri, u32 fi)
  {
    return flux_int_[nfacqp * (8 * (rank * fi + ri) + vi) + qi];
  }

  // vectors stored in this order: F dcQLx dcQLy dcQLz Qh
  __forceinline__ __device__ real& flux_bnd(u32 qi, u32 vi, u32 ri, u32 fi)
  {
    return flux_bnd_[nfacqp * (5 * (rank * fi + ri) + vi) + qi];
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

  __host__ u32 size_flux_vol() const
  {
    return nvolqp * rank * 6 * nelem;
  }

  __host__ u32 size_flux_int() const
  {
    return nfacqp * rank * 8 * niface;
  }

  __host__ u32 size_flux_bnd() const
  {
    return nfacqp * rank * 5 * nbface;
  }

  __host__ u32 size_integral_face() const
  {
    return nelem * 6 * rank * nbfp;
  }
};

__host__ custore custore_make(shared_geometry* geom, simstate* U, real* d_U)
{
  custore store;
  core_geometry* core = &(geom->core);

  /* initialize state */

  cudaMemcpy(d_U, U->U.data, U->U.len * sizeof(real), cudaMemcpyHostToDevice);

  /* initialize counts */

  store.solarr_size = U->size();
  store.nelem       = core->nelem;
  store.nbfp        = core->refp.nbf3d;
  store.nvolqp      = core->refp.vqrule.n;
  store.nfacqp      = core->refp.fqrule.n;
  store.niface      = geom->num_interior_faces();
  store.nbface      = geom->num_boundary_faces();

  /* initialize pre-computes */

  cudaMalloc(&store.Minv_, core->Minv_.len * sizeof(real));
  cudaMemcpy(store.Minv_, core->Minv_.data, core->Minv_.len * sizeof(real),
             cudaMemcpyHostToDevice);

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

  array<real, 3> shuffle_n({1, 0, 2}, geom->n_);
  cudaMalloc(&store.n_, shuffle_n.len * sizeof(real));
  cudaMemcpy(store.n_, shuffle_n.data, shuffle_n.len * sizeof(real),
             cudaMemcpyHostToDevice);

  array<real, 2> shuffle_h_int({1, 0}, geom->interior_h);
  cudaMalloc(&store.h_int_, shuffle_h_int.len * sizeof(real));
  cudaMemcpy(store.h_int_, shuffle_h_int.data, shuffle_h_int.len * sizeof(real),
             cudaMemcpyHostToDevice);

  cudaMalloc(&store.h_bnd_, geom->boundary_h.len * sizeof(real));
  cudaMemcpy(store.h_bnd_, geom->boundary_h.data,
             geom->boundary_h.len * sizeof(real), cudaMemcpyHostToDevice);

  cudaMalloc(&store.vquadw_, core->refp.vqrule.w.len * sizeof(real));
  cudaMemcpy(store.vquadw_, core->refp.vqrule.w.data,
             core->refp.vqrule.w.len * sizeof(real), cudaMemcpyHostToDevice);

  cudaMalloc(&store.fquadw_, core->refp.fqrule.w.len * sizeof(real));
  cudaMemcpy(store.fquadw_, core->refp.fqrule.w.data,
             core->refp.fqrule.w.len * sizeof(real), cudaMemcpyHostToDevice);

  cudaMalloc(&store.vJ_, core->vJ_.len * sizeof(real));
  cudaMemcpy(store.vJ_, core->vJ_.data, core->vJ_.len * sizeof(real),
             cudaMemcpyHostToDevice);

  cudaMalloc(&store.fJ_, geom->fJ_.len * sizeof(real));
  cudaMemcpy(store.fJ_, geom->fJ_.data, geom->fJ_.len * sizeof(real),
             cudaMemcpyHostToDevice);

  /* allocate (and initialize?) workspace */

  cudaMalloc(&store.state_veval_,   store.size_state_veval()   * sizeof(real));
  cudaMalloc(&store.state_vgrad_,   store.size_state_vgrad()   * sizeof(real));
  cudaMalloc(&store.state_ifeval_,  store.size_state_ifeval()  * sizeof(real));
  cudaMalloc(&store.state_ifgrad_,  store.size_state_ifgrad()  * sizeof(real));
  cudaMalloc(&store.state_bfeval_,  store.size_state_bfeval()  * sizeof(real));
  cudaMalloc(&store.state_bfgrad_,  store.size_state_bfgrad()  * sizeof(real));
  cudaMalloc(&store.flux_vol_,      store.size_flux_vol()      * sizeof(real));
  cudaMalloc(&store.flux_int_,      store.size_flux_int()      * sizeof(real));
  cudaMalloc(&store.flux_bnd_,      store.size_flux_bnd()      * sizeof(real));
  cudaMalloc(&store.integral_face_, store.size_integral_face() * sizeof(real));

  return store;
}

__host__ void custore_free(custore* store)
{
  cudaFree(store->Minv_);
  cudaFree(store->veval_);
  cudaFree(store->vgrad_);
  cudaFree(store->feval_);
  cudaFree(store->fgrad_);
  cudaFree(store->vquadw_);
  cudaFree(store->fquadw_);
  cudaFree(store->vJ_);
  cudaFree(store->fJ_);
  cudaFree(store->n_);
  cudaFree(store->h_int_);
  cudaFree(store->h_bnd_);
  cudaFree(store->iflist);
  cudaFree(store->bflist);
  cudaFree(store->state_veval_);
  cudaFree(store->state_vgrad_);
  cudaFree(store->state_ifeval_);
  cudaFree(store->state_ifgrad_);
  cudaFree(store->state_bfeval_);
  cudaFree(store->state_bfgrad_);
  cudaFree(store->flux_vol_);
  cudaFree(store->flux_int_);
  cudaFree(store->flux_bnd_);
  cudaFree(store->integral_face_);
}

struct cuworkspace
{
  u32 naux;
  real** aux;
};

cuworkspace cuworkspace_make(custore* store, u32 naux)
{
  cuworkspace wsp;

  wsp.naux = naux;
  wsp.aux = new real*[naux];

  for (u32 i = 0; i < naux; ++i)
  {
    cudaMalloc(&wsp.aux[i], store->solarr_size * sizeof(real));
  }

  return wsp;
}

void cuworkspace_free(cuworkspace* wsp)
{
  for(u32 i = 0; i < wsp->naux; ++i)
  {
    cudaFree(wsp->aux[i]);
  }

  delete[] wsp->aux;
}

/* residual and related kernels --------------------------------------------- */

__global__ void cuda_evaluate_volume_states(custore store, real* U)
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

__global__ void cuda_evaluate_interior_face_states(custore store, real* U)
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

__global__ void cuda_evaluate_boundary_face_states(custore store, real* U)
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

__global__ void cuda_evaluate_interior_flux(custore store, parameters params)
{
  u32 tid         = blockDim.x * blockIdx.x + threadIdx.x;
  u32 stride      = blockDim.x * gridDim.x;
  u32 num_glob_qp = store.nelem * store.nvolqp;

  for (u32 i = tid; i < num_glob_qp; i += stride)
  {
    u32 ei = i / store.nvolqp; 
    u32 qi = i - (ei * store.nvolqp);

    real S[5], Sx[5], Sy[5], Sz[5];

    S[0] = store.state_veval(qi, 0, ei);
    S[1] = store.state_veval(qi, 1, ei);
    S[2] = store.state_veval(qi, 2, ei);
    S[3] = store.state_veval(qi, 3, ei);
    S[4] = store.state_veval(qi, 4, ei);

    Sx[0] = store.state_vgrad(qi, 0, 0, ei);
    Sx[1] = store.state_vgrad(qi, 0, 1, ei);
    Sx[2] = store.state_vgrad(qi, 0, 2, ei);
    Sx[3] = store.state_vgrad(qi, 0, 3, ei);
    Sx[4] = store.state_vgrad(qi, 0, 4, ei);

    Sy[0] = store.state_vgrad(qi, 1, 0, ei);
    Sy[1] = store.state_vgrad(qi, 1, 1, ei);
    Sy[2] = store.state_vgrad(qi, 1, 2, ei);
    Sy[3] = store.state_vgrad(qi, 1, 3, ei);
    Sy[4] = store.state_vgrad(qi, 1, 4, ei);

    Sz[0] = store.state_vgrad(qi, 2, 0, ei);
    Sz[1] = store.state_vgrad(qi, 2, 1, ei);
    Sz[2] = store.state_vgrad(qi, 2, 2, ei);
    Sz[3] = store.state_vgrad(qi, 2, 3, ei);
    Sz[4] = store.state_vgrad(qi, 2, 4, ei);

    real Fx[5], Fy[5], Fz[5];
    real Qx[5], Qy[5], Qz[5];
    analytical_flux(S, params.gamma, Fx, Fy, Fz);
    A(S, Sx, Sy, Sz, params.mu, Qx, Qy, Qz);

    store.flux_vol(qi, 0, 0, ei) = Fx[0];
    store.flux_vol(qi, 0, 1, ei) = Fx[1];
    store.flux_vol(qi, 0, 2, ei) = Fx[2];
    store.flux_vol(qi, 0, 3, ei) = Fx[3];
    store.flux_vol(qi, 0, 4, ei) = Fx[4];

    store.flux_vol(qi, 1, 0, ei) = Fy[0];
    store.flux_vol(qi, 1, 1, ei) = Fy[1];
    store.flux_vol(qi, 1, 2, ei) = Fy[2];
    store.flux_vol(qi, 1, 3, ei) = Fy[3];
    store.flux_vol(qi, 1, 4, ei) = Fy[4];

    store.flux_vol(qi, 2, 0, ei) = Fz[0];
    store.flux_vol(qi, 2, 1, ei) = Fz[1];
    store.flux_vol(qi, 2, 2, ei) = Fz[2];
    store.flux_vol(qi, 2, 3, ei) = Fz[3];
    store.flux_vol(qi, 2, 4, ei) = Fz[4];

    store.flux_vol(qi, 3, 0, ei) = Qx[0];
    store.flux_vol(qi, 3, 1, ei) = Qx[1];
    store.flux_vol(qi, 3, 2, ei) = Qx[2];
    store.flux_vol(qi, 3, 3, ei) = Qx[3];
    store.flux_vol(qi, 3, 4, ei) = Qx[4];

    store.flux_vol(qi, 4, 0, ei) = Qy[0];
    store.flux_vol(qi, 4, 1, ei) = Qy[1];
    store.flux_vol(qi, 4, 2, ei) = Qy[2];
    store.flux_vol(qi, 4, 3, ei) = Qy[3];
    store.flux_vol(qi, 4, 4, ei) = Qy[4];

    store.flux_vol(qi, 5, 0, ei) = Qz[0];
    store.flux_vol(qi, 5, 1, ei) = Qz[1];
    store.flux_vol(qi, 5, 2, ei) = Qz[2];
    store.flux_vol(qi, 5, 3, ei) = Qz[3];
    store.flux_vol(qi, 5, 4, ei) = Qz[4];
  }
}

__global__ void cuda_evaluate_interior_face_flux(custore store,
                                                 parameters params)
{
  u32 tid         = blockDim.x * blockIdx.x + threadIdx.x;
  u32 num_glob_qp = store.niface * store.nfacqp;
  u32 stride      = gridDim.x * blockDim.x;

  for (u32 i = tid; i < num_glob_qp; i += stride)
  {
    u32 fi  = i / store.nfacqp;
    u32 qi  = i - (fi * store.nfacqp);
    u32 gfi = (u32)store.iflist[5 * fi + 0];

    real SL[5], SR[5];
    real SLx[5], SLy[5], SLz[5], SRx[5], SRy[5], SRz[5];

    real n[3];
    n[0] = store.n(qi, 0, gfi);
    n[1] = store.n(qi, 1, gfi);
    n[2] = store.n(qi, 2, gfi);

    SL[0] = store.state_ifeval(qi, 0, 0, fi); 
    SL[1] = store.state_ifeval(qi, 1, 0, fi); 
    SL[2] = store.state_ifeval(qi, 2, 0, fi); 
    SL[3] = store.state_ifeval(qi, 3, 0, fi); 
    SL[4] = store.state_ifeval(qi, 4, 0, fi); 

    SR[0] = store.state_ifeval(qi, 0, 1, fi); 
    SR[1] = store.state_ifeval(qi, 1, 1, fi); 
    SR[2] = store.state_ifeval(qi, 2, 1, fi); 
    SR[3] = store.state_ifeval(qi, 3, 1, fi); 
    SR[4] = store.state_ifeval(qi, 4, 1, fi); 

    SLx[0] = store.state_ifgrad(qi, 0, 0, 0, fi);
    SLx[1] = store.state_ifgrad(qi, 0, 1, 0, fi);
    SLx[2] = store.state_ifgrad(qi, 0, 2, 0, fi);
    SLx[3] = store.state_ifgrad(qi, 0, 3, 0, fi);
    SLx[4] = store.state_ifgrad(qi, 0, 4, 0, fi);

    SLy[0] = store.state_ifgrad(qi, 1, 0, 0, fi);
    SLy[1] = store.state_ifgrad(qi, 1, 1, 0, fi);
    SLy[2] = store.state_ifgrad(qi, 1, 2, 0, fi);
    SLy[3] = store.state_ifgrad(qi, 1, 3, 0, fi);
    SLy[4] = store.state_ifgrad(qi, 1, 4, 0, fi);
    
    SLz[0] = store.state_ifgrad(qi, 2, 0, 0, fi);
    SLz[1] = store.state_ifgrad(qi, 2, 1, 0, fi);
    SLz[2] = store.state_ifgrad(qi, 2, 2, 0, fi);
    SLz[3] = store.state_ifgrad(qi, 2, 3, 0, fi);
    SLz[4] = store.state_ifgrad(qi, 2, 4, 0, fi);

    SRx[0] = store.state_ifgrad(qi, 0, 0, 1, fi);
    SRx[1] = store.state_ifgrad(qi, 0, 1, 1, fi);
    SRx[2] = store.state_ifgrad(qi, 0, 2, 1, fi);
    SRx[3] = store.state_ifgrad(qi, 0, 3, 1, fi);
    SRx[4] = store.state_ifgrad(qi, 0, 4, 1, fi);

    SRy[0] = store.state_ifgrad(qi, 1, 0, 1, fi);
    SRy[1] = store.state_ifgrad(qi, 1, 1, 1, fi);
    SRy[2] = store.state_ifgrad(qi, 1, 2, 1, fi);
    SRy[3] = store.state_ifgrad(qi, 1, 3, 1, fi);
    SRy[4] = store.state_ifgrad(qi, 1, 4, 1, fi);
    
    SRz[0] = store.state_ifgrad(qi, 2, 0, 1, fi);
    SRz[1] = store.state_ifgrad(qi, 2, 1, 1, fi);
    SRz[2] = store.state_ifgrad(qi, 2, 2, 1, fi);
    SRz[3] = store.state_ifgrad(qi, 2, 3, 1, fi);
    SRz[4] = store.state_ifgrad(qi, 2, 4, 1, fi);

    /* inviscid flux term */

    real F[5];
    real dcQLx[5], dcQLy[5], dcQLz[5];
    real dcQRx[5], dcQRy[5], dcQRz[5];
    real Qh[5];

    {
      real smax;
      roe(SL, SR, n, params.gamma, F, &smax);
    }

    /* dual consistency term */

    {
      real Sh[5];
      for (u32 ri = 0; ri < 5; ++ri)
        Sh[ri] = 0.5 * (SL[ri] + SR[ri]);

      real dL[5], dR[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dL[ri] = SL[ri] - Sh[ri];
        dR[ri] = SR[ri] - Sh[ri];
      }

      real dLnx[5], dLny[5], dLnz[5], dRnx[5], dRny[5], dRnz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dLnx[ri] = dL[ri] * n[0];
        dLny[ri] = dL[ri] * n[1];
        dLnz[ri] = dL[ri] * n[2];

        dRnx[ri] = dR[ri] * -n[0];
        dRny[ri] = dR[ri] * -n[1];
        dRnz[ri] = dR[ri] * -n[2];
      }

      A(SL, dLnx, dLny, dLnz, params.mu, dcQLx, dcQLy, dcQLz);
      A(SR, dRnx, dRny, dRnz, params.mu, dcQRx, dcQRy, dcQRz);
    }

    /* viscous flux term */

    {
      real dLx[5] = {}, dLy[5] = {}, dLz[5] = {};
      real dRx[5] = {}, dRy[5] = {}, dRz[5] = {};

      real diffL[5], diffR[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        diffL[ri] = SL[ri] - SR[ri];
        diffR[ri] = SR[ri] - SL[ri];
      }

      real DLNx[5], DLNy[5], DLNz[5], DRNx[5], DRNy[5], DRNz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        DLNx[ri] = diffL[ri] * n[0];
        DLNy[ri] = diffL[ri] * n[1];
        DLNz[ri] = diffL[ri] * n[2];

        DRNx[ri] = diffR[ri] * -n[0];
        DRNy[ri] = diffR[ri] * -n[1];
        DRNz[ri] = diffR[ri] * -n[2];
      }

      real QLx[5], QLy[5], QLz[5], QRx[5], QRy[5], QRz[5];
      A(SL, DLNx, DLNy, DLNz, params.mu, QLx, QLy, QLz);
      A(SR, DRNx, DRNy, DRNz, params.mu, QRx, QRy, QRz);

      real hL = 1.f / store.h_int(fi, 0);
      real hR = 1.f / store.h_int(fi, 1);
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dLx[ri] = hL * QLx[ri];
        dLy[ri] = hL * QLy[ri];
        dLz[ri] = hL * QLz[ri];
        dRx[ri] = hR * QRx[ri];
        dRy[ri] = hR * QRy[ri];
        dRz[ri] = hR * QRz[ri];
      }

      A(SL, SLx, SLy, SLz, params.mu, QLx, QLy, QLz);
      A(SR, SRx, SRy, SRz, params.mu, QRx, QRy, QRz);

      real QHx[5], QHy[5], QHz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        QHx[ri] =
        0.5 * (QLx[ri] + QRx[ri]) - params.eta * 0.5 * (dLx[ri] + dRx[ri]);
        QHy[ri] =
        0.5 * (QLy[ri] + QRy[ri]) - params.eta * 0.5 * (dLy[ri] + dRy[ri]);
        QHz[ri] =
        0.5 * (QLz[ri] + QRz[ri]) - params.eta * 0.5 * (dLz[ri] + dRz[ri]);
      }

      for (u32 ri = 0; ri < 5; ++ri)
        Qh[ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
    }

    store.flux_int(qi, 0, 0, fi) = F[0];
    store.flux_int(qi, 0, 1, fi) = F[1];
    store.flux_int(qi, 0, 2, fi) = F[2];
    store.flux_int(qi, 0, 3, fi) = F[3];
    store.flux_int(qi, 0, 4, fi) = F[4];

    store.flux_int(qi, 1, 0, fi) = dcQLx[0];
    store.flux_int(qi, 1, 1, fi) = dcQLx[1];
    store.flux_int(qi, 1, 2, fi) = dcQLx[2];
    store.flux_int(qi, 1, 3, fi) = dcQLx[3];
    store.flux_int(qi, 1, 4, fi) = dcQLx[4];

    store.flux_int(qi, 2, 0, fi) = dcQLy[0];
    store.flux_int(qi, 2, 1, fi) = dcQLy[1];
    store.flux_int(qi, 2, 2, fi) = dcQLy[2];
    store.flux_int(qi, 2, 3, fi) = dcQLy[3];
    store.flux_int(qi, 2, 4, fi) = dcQLy[4];

    store.flux_int(qi, 3, 0, fi) = dcQLz[0];
    store.flux_int(qi, 3, 1, fi) = dcQLz[1];
    store.flux_int(qi, 3, 2, fi) = dcQLz[2];
    store.flux_int(qi, 3, 3, fi) = dcQLz[3];
    store.flux_int(qi, 3, 4, fi) = dcQLz[4];

    store.flux_int(qi, 4, 0, fi) = dcQRx[0];
    store.flux_int(qi, 4, 1, fi) = dcQRx[1];
    store.flux_int(qi, 4, 2, fi) = dcQRx[2];
    store.flux_int(qi, 4, 3, fi) = dcQRx[3];
    store.flux_int(qi, 4, 4, fi) = dcQRx[4];

    store.flux_int(qi, 5, 0, fi) = dcQRy[0];
    store.flux_int(qi, 5, 1, fi) = dcQRy[1];
    store.flux_int(qi, 5, 2, fi) = dcQRy[2];
    store.flux_int(qi, 5, 3, fi) = dcQRy[3];
    store.flux_int(qi, 5, 4, fi) = dcQRy[4];

    store.flux_int(qi, 6, 0, fi) = dcQRz[0];
    store.flux_int(qi, 6, 1, fi) = dcQRz[1];
    store.flux_int(qi, 6, 2, fi) = dcQRz[2];
    store.flux_int(qi, 6, 3, fi) = dcQRz[3];
    store.flux_int(qi, 6, 4, fi) = dcQRz[4];

    store.flux_int(qi, 7, 0, fi) = Qh[0];
    store.flux_int(qi, 7, 1, fi) = Qh[1];
    store.flux_int(qi, 7, 2, fi) = Qh[2];
    store.flux_int(qi, 7, 3, fi) = Qh[3];
    store.flux_int(qi, 7, 4, fi) = Qh[4];
  }
}

__global__ void cuda_evaluate_boundary_face_flux(custore store,
                                                 parameters params)
{
  u32 tid         = blockDim.x * blockIdx.x + threadIdx.x;
  u32 num_glob_qp = store.nbface * store.nfacqp;
  u32 stride      = gridDim.x * blockDim.x;

  for (u32 i = tid; i < num_glob_qp; i += stride)
  {
    u32 fi         = i / store.nfacqp;
    u32 qi         = i - (fi * store.nfacqp);
    s32 gfi        = store.bflist[4 * fi + 0];
    s32 bound_type = store.bflist[4 * fi + 2];

    real SL[5], Sb[5];
    real SLx[5], SLy[5], SLz[5];

    real n[3];
    n[0] = store.n(qi, 0, gfi);
    n[1] = store.n(qi, 1, gfi);
    n[2] = store.n(qi, 2, gfi);

    SL[0] = store.state_bfeval(qi, 0, fi);
    SL[1] = store.state_bfeval(qi, 1, fi);
    SL[2] = store.state_bfeval(qi, 2, fi);
    SL[3] = store.state_bfeval(qi, 3, fi);
    SL[4] = store.state_bfeval(qi, 4, fi);

    SLx[0] = store.state_bfgrad(qi, 0, 0, fi);
    SLx[1] = store.state_bfgrad(qi, 0, 1, fi);
    SLx[2] = store.state_bfgrad(qi, 0, 2, fi);
    SLx[3] = store.state_bfgrad(qi, 0, 3, fi);
    SLx[4] = store.state_bfgrad(qi, 0, 4, fi);

    SLy[0] = store.state_bfgrad(qi, 1, 0, fi);
    SLy[1] = store.state_bfgrad(qi, 1, 1, fi);
    SLy[2] = store.state_bfgrad(qi, 1, 2, fi);
    SLy[3] = store.state_bfgrad(qi, 1, 3, fi);
    SLy[4] = store.state_bfgrad(qi, 1, 4, fi);

    SLz[0] = store.state_bfgrad(qi, 2, 0, fi);
    SLz[1] = store.state_bfgrad(qi, 2, 1, fi);
    SLz[2] = store.state_bfgrad(qi, 2, 2, fi);
    SLz[3] = store.state_bfgrad(qi, 2, 3, fi);
    SLz[4] = store.state_bfgrad(qi, 2, 4, fi);

    switch (bound_type)
    {
      case freestream:
      {
        for (u32 i = 0; i < 5; ++i) Sb[i] = params.Ufr[i];
      }
      break;
      case inviscid_wall:
      {
        inviscid_wall_state(SL, n, params.gamma, Sb);
      }
      break;
      case no_slip_wall:
      {
        no_slip_wall_state(SL, Sb);
      }
      break;
      case subsonic_inflow: 
      {
        real gm1 = params.gamma - 1.;
        real u   = params.Ufr[1] / params.Ufr[0];
        real v   = params.Ufr[2] / params.Ufr[0];
        real w   = params.Ufr[3] / params.Ufr[0];
        real M   = sqrt(u * u + v * v + w * w) / params.c;
        real Tt  = params.T * (1. + 0.5 * gm1 * M * M);
        real Pt  = params.P * pow(Tt / params.T, params.gamma / gm1);
        subsonic_inflow_state(SL, n, params.gamma, params.R, Pt, Tt, Sb);
      }
      break;
      case subsonic_outflow:
      {
        subsonic_outflow_state(SL, n, params.gamma, params.P, Sb);
      }
      break;
    };

    /* inviscid flux term */

    real F[5];
    {
      real Fx[5], Fy[5], Fz[5];
      analytical_flux(Sb, params.gamma, Fx, Fy, Fz);

      F[0] = Fx[0] * n[0] + Fy[0] * n[1] + Fz[0] * n[2];
      F[1] = Fx[1] * n[0] + Fy[1] * n[1] + Fz[1] * n[2];
      F[2] = Fx[2] * n[0] + Fy[2] * n[1] + Fz[2] * n[2];
      F[3] = Fx[3] * n[0] + Fy[3] * n[1] + Fz[3] * n[2];
      F[4] = Fx[4] * n[0] + Fy[4] * n[1] + Fz[4] * n[2];
    }

    /* dual consistency term */

    real dcQLx[5], dcQLy[5], dcQLz[5];
    {
      real Sh[5];
      for (u32 i = 0; i < 5; ++i)
        Sh[i] = Sb[i];

      real diffL[5];
      for (u32 i = 0; i < 5; ++i)
        diffL[i] = SL[i] - Sh[i];

      real DLNx[5], DLNy[5], DLNz[5];
      for (u32 i = 0; i < 5; ++i)
      {
        DLNx[i] = diffL[i] * n[0];
        DLNy[i] = diffL[i] * n[1];
        DLNz[i] = diffL[i] * n[2];
      }

      A(SL, DLNx, DLNy, DLNz, params.mu, dcQLx, dcQLy, dcQLz);
    }

    /* viscous flux term */

    real Qh[5];
    {
      real dLx[5] = {}, dLy[5] = {}, dLz[5] = {};

      real diffL[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        diffL[ri] = SL[ri] - Sb[ri];
      }

      real DLNx[5], DLNy[5], DLNz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        DLNx[ri] = diffL[ri] * n[0];
        DLNy[ri] = diffL[ri] * n[1];
        DLNz[ri] = diffL[ri] * n[2];
      }

      real QLx[5], QLy[5], QLz[5];
      A(SL, DLNx, DLNy, DLNz, params.mu, QLx, QLy, QLz);

      real hL = 1. / store.h_bnd(fi);
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dLx[ri] = hL * QLx[ri];
        dLy[ri] = hL * QLy[ri];
        dLz[ri] = hL * QLz[ri];
      }

      A(SL, SLx, SLy, SLz, params.mu, QLx, QLy, QLz);

      real QHx[5], QHy[5], QHz[5];
      for (u64 ri = 0; ri < 5; ++ri)
      {
        QHx[ri] = QLx[ri] - params.eta * dLx[ri];
        QHy[ri] = QLy[ri] - params.eta * dLy[ri];
        QHz[ri] = QLz[ri] - params.eta * dLz[ri];
      }

      for (u64 ri = 0; ri < 5; ++ri)
        Qh[ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
    }

    store.flux_bnd(qi, 0, 0, fi) = F[0];
    store.flux_bnd(qi, 0, 1, fi) = F[1];
    store.flux_bnd(qi, 0, 2, fi) = F[2];
    store.flux_bnd(qi, 0, 3, fi) = F[3];
    store.flux_bnd(qi, 0, 4, fi) = F[4];

    store.flux_bnd(qi, 1, 0, fi) = dcQLx[0];
    store.flux_bnd(qi, 1, 1, fi) = dcQLx[1];
    store.flux_bnd(qi, 1, 2, fi) = dcQLx[2];
    store.flux_bnd(qi, 1, 3, fi) = dcQLx[3];
    store.flux_bnd(qi, 1, 4, fi) = dcQLx[4];

    store.flux_bnd(qi, 2, 0, fi) = dcQLy[0];
    store.flux_bnd(qi, 2, 1, fi) = dcQLy[1];
    store.flux_bnd(qi, 2, 2, fi) = dcQLy[2];
    store.flux_bnd(qi, 2, 3, fi) = dcQLy[3];
    store.flux_bnd(qi, 2, 4, fi) = dcQLy[4];

    store.flux_bnd(qi, 3, 0, fi) = dcQLz[0];
    store.flux_bnd(qi, 3, 1, fi) = dcQLz[1];
    store.flux_bnd(qi, 3, 2, fi) = dcQLz[2];
    store.flux_bnd(qi, 3, 3, fi) = dcQLz[3];
    store.flux_bnd(qi, 3, 4, fi) = dcQLz[4];

    store.flux_bnd(qi, 4, 0, fi) = Qh[0];
    store.flux_bnd(qi, 4, 1, fi) = Qh[1];
    store.flux_bnd(qi, 4, 2, fi) = Qh[2];
    store.flux_bnd(qi, 4, 3, fi) = Qh[3];
    store.flux_bnd(qi, 4, 4, fi) = Qh[4];
  }
}

__global__ void cuda_evaluate_interior_residual(custore store, real* R)
{
  u32 ei = blockIdx.x;

  for (u32 i = threadIdx.x; i < store.nbfp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nbfp;
    u32 ti = i - (ri * store.nbfp);

    real residual = 0.;
    for (u32 qi = 0; qi < store.nvolqp; ++qi)
    {
      real qw = store.vquadw(qi);
      real J  = store.vJ(qi, ei);

      real tgrad0 = store.vgrad(qi, ti, ei, 0);
      real tgrad1 = store.vgrad(qi, ti, ei, 1);
      real tgrad2 = store.vgrad(qi, ti, ei, 2);

      real Fx = store.flux_vol(qi, 0, ri, ei);
      real Fy = store.flux_vol(qi, 1, ri, ei);
      real Fz = store.flux_vol(qi, 2, ri, ei);
      real Qx = store.flux_vol(qi, 3, ri, ei);
      real Qy = store.flux_vol(qi, 4, ri, ei);
      real Qz = store.flux_vol(qi, 5, ri, ei);

      residual +=
      (tgrad0 * (Qx - Fx) + tgrad1 * (Qy - Fy) + tgrad2 * (Qz - Fz)) * (J * qw);
    }

    store.index_state(R, ti, ri, ei) += residual;
  }
}

__global__ void cuda_evaluate_interior_face_residuals(custore store)
{
  u32 fi = blockIdx.x;

  for (u32 i = threadIdx.x; i < store.nbfp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nbfp;
    u32 ti = i - (ri * store.nbfp);

    s32 gfi = store.iflist[5 * fi + 0];
    s32 eL  = store.iflist[5 * fi + 1];
    s32 eR  = store.iflist[5 * fi + 2];
    s32 lfL = store.iflist[5 * fi + 3];
    s32 lfR = store.iflist[5 * fi + 4];

    real residualL = 0.;
    real residualR = 0.;
    for (u32 qi = 0; qi < store.nfacqp; ++qi)
    {
      real J   = store.fJ(qi, gfi);
      real qw  = store.fquadw(qi);
      real tfL = store.feval(qi, ti, lfL);
      real tfR = store.feval(qi, ti, lfR);

      real tgLx = store.fgrad(qi, 0, ti, lfL, eL);
      real tgLy = store.fgrad(qi, 1, ti, lfL, eL);
      real tgLz = store.fgrad(qi, 2, ti, lfL, eL);

      real tgRx = store.fgrad(qi, 0, ti, lfR, eR);
      real tgRy = store.fgrad(qi, 1, ti, lfR, eR);
      real tgRz = store.fgrad(qi, 2, ti, lfR, eR);

      real F     = store.flux_int(qi, 0, ri, fi);
      real dcQLx = store.flux_int(qi, 1, ri, fi);
      real dcQLy = store.flux_int(qi, 2, ri, fi);
      real dcQLz = store.flux_int(qi, 3, ri, fi);
      real dcQRx = store.flux_int(qi, 4, ri, fi);
      real dcQRy = store.flux_int(qi, 5, ri, fi);
      real dcQRz = store.flux_int(qi, 6, ri, fi);
      real Qh    = store.flux_int(qi, 7, ri, fi);

      residualL +=
      (+(tfL * F) - (tgLx * dcQLx + tgLy * dcQLy + tgLz * dcQLz) - (tfL * Qh)) *
      (J * qw);
      residualR +=
      (-(tfR * F) + (tgRx * dcQRx + tgRy * dcQRy + tgRz * dcQRz) + (tfR * Qh)) *
      (J * qw);
    }

    store.integral_face(ti, ri, lfL, eL) = residualL;
    store.integral_face(ti, ri, lfR, eR) = residualR;
  }
}

__global__ void cuda_evaluate_boundary_face_residuals(custore store)
{
  u32 fi = blockIdx.x;

  for (u32 i = threadIdx.x; i < store.nbfp * store.rank; i += blockDim.x)
  {
    u32 ri = i / store.nbfp;
    u32 ti = i - (ri * store.nbfp);

    s32 gfi = store.bflist[4 * fi + 0];
    s32 eL  = store.bflist[4 * fi + 1];
    s32 lfL = store.bflist[4 * fi + 3];

    real residualL = 0.;
    for (u32 qi = 0; qi < store.nfacqp; ++qi)
    {
      real J   = store.fJ(qi, gfi);
      real qw  = store.fquadw(qi);
      real tfL = store.feval(qi, ti, lfL);

      real tgLx = store.fgrad(qi, 0, ti, lfL, eL);
      real tgLy = store.fgrad(qi, 1, ti, lfL, eL);
      real tgLz = store.fgrad(qi, 2, ti, lfL, eL);

      real F     = store.flux_bnd(qi, 0, ri, fi);
      real dcQLx = store.flux_bnd(qi, 1, ri, fi);
      real dcQLy = store.flux_bnd(qi, 2, ri, fi);
      real dcQLz = store.flux_bnd(qi, 3, ri, fi);
      real Qh    = store.flux_bnd(qi, 4, ri, fi);

      residualL +=
      (+(tfL * F) - (tgLx * dcQLx + tgLy * dcQLy + tgLz * dcQLz) - (tfL * Qh)) *
      (J * qw);
    }

    store.integral_face(ti, ri, lfL, eL) = residualL;
  }
}

__global__ void cuda_accumulate_face_residuals(custore store, real* R)
{
  u32 tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (u32 i = tid; i < store.nelem * store.rank * store.nbfp;
       i += gridDim.x * blockDim.x)
  {
    u32 ei = i / (store.rank * store.nbfp);
    u32 ri = (i - (ei * store.rank * store.nbfp)) / store.nbfp;
    u32 ti = i - (ei * store.rank * store.nbfp) - (ri * store.nbfp);

    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 0, ei);
    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 1, ei);
    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 2, ei);
    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 3, ei);
    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 4, ei);
    store.index_state(R, ti, ri, ei) += store.integral_face(ti, ri, 5, ei);
  }
}

__global__ void cuda_residual_mass_multiply(custore store, real* R, real* f)
{
  u32 ei = blockIdx.x;

  real* M = store.Minv(ei);

  for (u32 i = threadIdx.x; i < store.rank * store.nbfp; i += blockDim.x)
  {
    u32 ri = i / store.nbfp;
    u32 bi = i - (ri * store.nbfp);

    real tmp = 0.;
    for (u32 ti = 0; ti < store.nbfp; ++ti)
    {
      u32 iMu = ((bi + 1) + ((2 * store.nbfp - (ti + 1)) * (ti) / 2)) - 1;
      u32 iMl = ((ti + 1) + ((2 * store.nbfp - (bi + 1)) * (bi) / 2)) - 1;

      u32 iM = (bi >= ti) * iMu + (bi < ti) * iMl;

      tmp += -1. * M[iM] * store.index_state(R, ti, ri, ei);
    }
    store.index_state(f, bi, ri, ei) = tmp;
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

/* utilities ---------------------------------------------------------------- */

__host__ u32 idiv_ceil(u32 a, u32 b)
{
  return (a + b - 1) / b;
}

__host__ void cuda_residual(custore store, parameters params, real* d_U,
                            real* d_R, real* d_f)
{
  cudaMemset(d_R, 0, store.solarr_size);
  cudaDeviceSynchronize();
  
  cudaMemset(store.integral_face_, 0, store.size_integral_face());
  cudaDeviceSynchronize();

  cuda_evaluate_volume_states<<<store.nelem, 1>>>(store, d_U);
  cudaDeviceSynchronize();
  cuda_evaluate_interior_flux<<<1, 1>>>(store, params);
  cudaDeviceSynchronize();
  cuda_evaluate_interior_residual<<<store.nelem, 1>>>(store, d_R);
  cudaDeviceSynchronize();

  cuda_evaluate_interior_face_states<<<store.niface, 1>>>(store, d_U);
  cudaDeviceSynchronize();
  cuda_evaluate_interior_face_flux<<<1, 1>>>(store, params);
  cudaDeviceSynchronize();
  cuda_evaluate_interior_face_residuals<<<store.niface, 1>>>(store);
  cudaDeviceSynchronize();

  if (store.nbface > 0)
  {
    cuda_evaluate_boundary_face_states<<<store.nbface, 1>>>(store, d_U);
    cudaDeviceSynchronize();
    cuda_evaluate_boundary_face_flux<<<1, 1>>>(store, params);
    cudaDeviceSynchronize();
    cuda_evaluate_boundary_face_residuals<<<store.nbface, 1>>>(store);
    cudaDeviceSynchronize();
  }

  cuda_accumulate_face_residuals<<<1, 1>>>(store, d_R);
  cudaDeviceSynchronize();

  cuda_residual_mass_multiply<<<store.nelem, 1>>>(store, d_R, d_f);
  cudaDeviceSynchronize();
}
