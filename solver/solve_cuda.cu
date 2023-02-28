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

#include <cstdint>

#include "helper_types.cpp"
#include "flux.cpp"
#include "bound_states.cpp"
#include "boundaries.cpp"
#include "solver_params.cpp"
#include "geometry_shared.cpp"

/* indexing macros */

#define NENTRYMINV ((refp_nbf3d * (refp_nbf3d + 1)) / 2)

#define REFP_VEVAL(qn, bf)     \
  (refp_veval[vqrule_n * (bf) + (qn)])
// #define REFP_VEVAL(qn, bf)     \
//   (refp_veval[refp_nbf3d * (qn) + (bf)])
#define REFP_FEVAL(lf, qn, bf) \
  (refp_feval[refp_nbf3d * (fqrule_n * (lf) + (qn)) + (bf)])
// #define REFP_FEVAL(lf, qn, bf) \
//   (refp_feval[fqrule_n * (refp_nbf3d * (lf) + (bf)) + (qn)])
#define MINV(e)                \
  (geo_Minv + NENTRYMINV * (e))
#define VJ(e, qn)              \
  (geo_vJ[(e) * vqrule_n + (qn)])
// #define VGRAD(e, qn, bf, dim)       \
//   (geo_vgrad [vqrule_n * (3 * (refp_nbf3d * (e) + (bf)) + (dim)) + (qn)])
#define VGRAD(e, qn, bf, dim)       \
  (geo_vgrad[3 * (refp_nbf3d * (vqrule_n * (e) + (qn)) + (bf)) + (dim)])
#define FGRAD(e, f, qn, bf)    \
  (geo_fgrad + (refp_nbf3d * (fqrule_n * (6 * (e) + (f)) + (qn)) + (bf)) * 3)
#define FJ(f, qn)              \
  (geo_fJ[fqrule_n * (f) + (qn)])
#define N(f, qn)               \
  (geo_n + 3 * (fqrule_n * (f) + (qn)))
#define STATE_INDX(e, rank, bf) \
  (refp_nbf3d * (5 * (e) + (rank)) + (bf))
#define U(e, rank, bf)         \
  (state[STATE_INDX((e), (rank), (bf))])
// #define U(e, rank, bf)         \
//   (state[refp_nbf3d * (5 * (e) + (rank)) + (bf)])
#define R(e, rank, bf)         \
  (residual[STATE_INDX((e), (rank), (bf))])
// #define R(e, rank, bf)         \
//   (residual[refp_nbf3d * (5 * (e) + (rank)) + (bf)])
#define F(e, rank, bf)         \
  (f[STATE_INDX((e), (rank), (bf))])
// #define F(e, rank, bf)         \
//   (f[refp_nbf3d * (5 * (e) + (rank)) + (bf)])
#define FACERESID(e, lf, rank, tf) \
  (face_integrals[refp_nbf3d * (5 * (6 * (e) + (lf)) + (rank)) + (tf)])

void shuffle_state(simstate& cpu_state, shared_geometry& geom, float* state)
{
  u32 nelem      = geom.core.nelem;
  u32 refp_nbf3d = geom.core.refp.nbf3d;

  for (u64 ei = 0; ei < geom.core.nelem; ++ei)
    for (u64 ri = 0; ri < 5; ++ri)
      for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
        U(ei, ri, bi) = cpu_state(ei, ri, bi);
}

void unshuffle_state(float* state, shared_geometry& geom, simstate& cpu_state)
{
  u32 nelem      = geom.core.nelem;
  u32 refp_nbf3d = geom.core.refp.nbf3d;

  for (u64 ei = 0; ei < geom.core.nelem; ++ei)
    for (u64 ri = 0; ri < 5; ++ri)
      for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
        cpu_state(ei, ri, bi) = U(ei, ri, bi);
}

void shuffle_refp_veval(shared_geometry& geom, float* refp_veval)
{
  u32 vqrule_n = geom.core.refp.vqrule.n;
  u32 refp_nbf3d = geom.core.refp.nbf3d;

  for(u64 qi = 0; qi < vqrule_n; ++qi)
    for (u32 bi = 0; bi < refp_nbf3d; ++bi)
      REFP_VEVAL(qi, bi) = geom.core.refp.veval_at(bi, qi);
}

void shuffle_refp_feval(shared_geometry& geom, float* refp_feval)
{
  u32 fqrule_n   = geom.core.refp.fqrule.n;
  u32 refp_nbf3d = geom.core.refp.nbf3d;

  for (u64 lfi = 0; lfi < 6; ++lfi)
    for (u64 bi = 0; bi < refp_nbf3d; ++bi)
      for (u64 qi = 0; qi < fqrule_n; ++qi)
        REFP_FEVAL(lfi, qi, bi) = geom.core.refp.feval_at(lfi, bi, qi);
}

void shuffle_vgrad(shared_geometry& geom, float* geo_vgrad)
{
  u32 nelem      = geom.core.nelem;
  u32 refp_nbf3d = geom.core.refp.nbf3d;
  u32 vqrule_n = geom.core.refp.vqrule.n;

  for (u64 ei = 0; ei < nelem; ++ei)
  {
    for (u64 qi = 0; qi < vqrule_n; ++qi)
    {
      for (u64 bi = 0; bi < refp_nbf3d; ++bi)
      {
        float* grad = geom.core.vgrad(ei, bi, qi);
        for (u64 di = 0; di < 3; ++di)
        {
          VGRAD(ei, qi, bi, 0) = grad[0];
          VGRAD(ei, qi, bi, 1) = grad[1];
          VGRAD(ei, qi, bi, 2) = grad[2];
        }
      }
    }
  }
}

struct cuda_device_geometry
{
  u32 p;
  u32 q;
  u32 nelem;
  u32 nface;
  u32 niface;
  u32 nbface;
  u32 vqrule_n;
  u32 fqrule_n;
  u32 refp_nbf2d;
  u32 refp_nbf3d;

  float* vqrule_w;
  float* fqrule_w;

  float* refp_veval;
  float* refp_feval;

  float* geo_Minv;
  float* geo_vJ;
  float* geo_vgrad;
  float* geo_fgrad;
  float* geo_fJ;
  float* geo_n;
  s32*   geo_ifl;
  s32*   geo_bfl;
  float* geo_ih;
  float* geo_bh;

  cuda_device_geometry(shared_geometry& geomsh);
};

cuda_device_geometry::cuda_device_geometry(shared_geometry& geomsh)
{
  p          = geomsh.core.p;
  q          = geomsh.core.q;
  nelem      = geomsh.core.nelem;
  nface      = geomsh.nface();
  niface     = geomsh.num_interior_faces();
  nbface     = geomsh.num_boundary_faces();
  vqrule_n   = geomsh.core.refp.vqrule.n;
  fqrule_n   = geomsh.core.refp.fqrule.n;
  refp_nbf2d = geomsh.core.refp.nbf2d;
  refp_nbf3d = geomsh.core.refp.nbf3d;

  array<s32> ifl_32(geomsh.interior_face_list.len);
  array<s32> bfl_32(geomsh.boundary_face_list.len);
  for (u64 i = 0; i < geomsh.interior_face_list.len; ++i)
    ifl_32[i] = (s32)geomsh.interior_face_list[i];
  for (u64 i = 0; i < geomsh.boundary_face_list.len; ++i)
    bfl_32[i] = (s32)geomsh.boundary_face_list[i];

  cudaMalloc(&vqrule_w, vqrule_n * sizeof(*vqrule_w));
  cudaMalloc(&fqrule_w, fqrule_n * sizeof(*fqrule_w));

  cudaMalloc(&refp_veval, refp_nbf3d * vqrule_n     * sizeof(*refp_veval));
  cudaMalloc(&refp_feval, refp_nbf3d * fqrule_n * 6 * sizeof(*refp_feval));

  cudaMalloc(&geo_Minv , geomsh.core.Minv_size()          * sizeof(*geo_Minv));
  cudaMalloc(&geo_vJ   , geomsh.core.vJ_size()            * sizeof(*geo_vJ));
  cudaMalloc(&geo_vgrad, geomsh.core.vgrad_size()         * sizeof(*geo_vgrad));
  cudaMalloc(&geo_fgrad, geomsh.core.fgrad_size()         * sizeof(*geo_fgrad));
  cudaMalloc(&geo_fJ   , geomsh.fJ_size()                 * sizeof(*geo_fJ));
  cudaMalloc(&geo_n    , geomsh.n_size()                  * sizeof(*geo_n));
  cudaMalloc(&geo_ifl  , geomsh.interior_face_list_size() * sizeof(*geo_ifl));
  cudaMalloc(&geo_bfl  , geomsh.boundary_face_list_size() * sizeof(*geo_bfl));
  cudaMalloc(&geo_ih, geomsh.interior_h_size() * sizeof(*geo_ih));
  cudaMalloc(&geo_bh, geomsh.boundary_h_size() * sizeof(*geo_bh));

  cudaMemcpy(vqrule_w, geomsh.core.refp.vqrule.w.data,
             vqrule_n * sizeof(*vqrule_w),
             cudaMemcpyHostToDevice);
  cudaMemcpy(fqrule_w, geomsh.core.refp.fqrule.w.data,
             fqrule_n * sizeof(*fqrule_w),
             cudaMemcpyHostToDevice);

  {
    array<float> refp_veval_shuffle(geomsh.core.refp.veval.len);
    shuffle_refp_veval(geomsh, refp_veval_shuffle.data);
    cudaMemcpy(refp_veval, refp_veval_shuffle.data,
               geomsh.core.refp.veval.len * sizeof(*refp_veval),
               cudaMemcpyHostToDevice);
  }

  {
    array<float> refp_feval_shuffle(geomsh.core.refp.feval.len);
    shuffle_refp_feval(geomsh, refp_feval_shuffle.data);
    cudaMemcpy(refp_feval, refp_feval_shuffle.data,
               geomsh.core.refp.feval.len * sizeof(*refp_feval),
               cudaMemcpyHostToDevice);
  }

  cudaMemcpy(geo_Minv, geomsh.core.Minv_.data,
             geomsh.core.Minv_size() * sizeof(*geo_Minv),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_vJ, geomsh.core.vJ_.data,
             geomsh.core.vJ_size() * sizeof(*geo_vJ),
             cudaMemcpyHostToDevice);
  {
    array<float> vgrad_shuffle(geomsh.core.vgrad_size());
    shuffle_vgrad(geomsh, vgrad_shuffle.data);
    cudaMemcpy(geo_vgrad, vgrad_shuffle.data,
               geomsh.core.vgrad_size() * sizeof(*geo_vgrad),
               cudaMemcpyHostToDevice);
  }
  cudaMemcpy(geo_fgrad, geomsh.core.fgrad_.data,
             geomsh.core.fgrad_size() * sizeof(*geo_fgrad),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_fJ, geomsh.fJ_.data,
             geomsh.fJ_size() * sizeof(*geo_fJ),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_n, geomsh.n_.data,
             geomsh.n_size() * sizeof(*geo_n),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_ifl, ifl_32.data,
             ifl_32.len * sizeof(*geo_ifl),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_bfl, bfl_32.data,
             bfl_32.len * sizeof(*geo_bfl),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_ih, geomsh.interior_h.data,
             geomsh.interior_h_size() * sizeof(*geo_ih),
             cudaMemcpyHostToDevice);
  cudaMemcpy(geo_bh, geomsh.boundary_h.data,
             geomsh.boundary_h_size() * sizeof(*geo_bh),
             cudaMemcpyHostToDevice);
}

void free_cuda_device_geometry(cuda_device_geometry* cugeom)
{
  cudaFree(cugeom->vqrule_w);
  cudaFree(cugeom->fqrule_w);
  cudaFree(cugeom->refp_veval);
  cudaFree(cugeom->refp_feval);
  cudaFree(cugeom->geo_Minv);
  cudaFree(cugeom->geo_vJ);
  cudaFree(cugeom->geo_n);
  cudaFree(cugeom->geo_ifl);
  cudaFree(cugeom->geo_bfl);
  cudaFree(cugeom->geo_ih);
  cudaFree(cugeom->geo_bh);
}

struct cuda_residual_workspace
{
  // face residual accumulation
  u32 face_integrals_size;
  float* face_integrals;

  // br2
  u32 br2_workspace_size;
  float* RL;
  float* RR;
  float* DL;
  float* DR;

  // time stepping scheme states
  u32 naux;
  u32 solarr_size;
  float** aux;

  cuda_residual_workspace(cuda_device_geometry* cugeom, u32 naux_state);
};

cuda_residual_workspace::cuda_residual_workspace(cuda_device_geometry* cugeom,
                                                 u32 naux_state)
{
  solarr_size         = cugeom->nelem * 5 * cugeom->refp_nbf3d;
  face_integrals_size = cugeom->nelem * cugeom->refp_nbf3d * 6 * 5;
  br2_workspace_size  = cugeom->nface * 5 * 3 * cugeom->refp_nbf3d;

  cudaMalloc(&face_integrals, face_integrals_size * sizeof(*face_integrals));

  cudaMalloc(&DL, br2_workspace_size * sizeof(*DL));
  cudaMalloc(&DR, br2_workspace_size * sizeof(*DR));
  cudaMalloc(&RL, br2_workspace_size * sizeof(*RL));
  cudaMalloc(&RR, br2_workspace_size * sizeof(*RR));

  naux = naux_state;
  aux  = new float*[naux];
  for (u32 i = 0; i < naux; ++i)
    cudaMalloc(&aux[i], solarr_size * sizeof(float));
}

void free_cuda_residual_workspace(cuda_residual_workspace* wsp)
{
  cudaFree(wsp->face_integrals);
  cudaFree(wsp->RL);
  cudaFree(wsp->RR);
  cudaFree(wsp->DL);
  cudaFree(wsp->DR);
  for (u32 i = 0; i < wsp->naux; ++i)
  {
    cudaFree(wsp->aux[i]);
  }
  delete[] wsp->aux;
}
__global__ void cuda_zero_array(u32 n, float* a)
{
  u32 tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < n)
    a[tid] = 0.f;
}

__global__ void cuda_interior_residual(cuda_device_geometry cugeom, float mu,
                                       float gamma, float* state,
                                       float* residual)
{
  extern __shared__ u32 shared[];

  u32 nelem         = cugeom.nelem;
  u32 refp_nbf3d    = cugeom.refp_nbf3d;
  u32 vqrule_n      = cugeom.vqrule_n;
  float* vqrule_w   = cugeom.vqrule_w;
  float* refp_veval = cugeom.refp_veval;
  float* geo_vJ     = cugeom.geo_vJ;
  float* geo_vgrad  = cugeom.geo_vgrad;

  // shared state: [rank [basis function]]
  // F:            [dim [rank]]
  // Q:            [dim [rank]]

  u32 shared_start    = 0;

  float* shared_state = (float*)shared + shared_start;
  shared_start += 5 * refp_nbf3d;
  float* F = (float*)shared + shared_start;
  shared_start += vqrule_n * 15;

  float3* Fflt3 = (float3*)F;

  u32 ei = blockIdx.x;

  /* load state (all threads in bock) */

  for (u32 i = threadIdx.x; i < 5 * refp_nbf3d; i += blockDim.x)
  {
    u32 ri = i / refp_nbf3d;
    u32 bi = i - refp_nbf3d * ri;
    shared_state[refp_nbf3d * ri + bi] = U(ei, ri, bi);
  }
  __syncthreads();

  /* compute integrand at each quad point (using shared state) */

  if (threadIdx.x < vqrule_n)
  {
    u32 qi = threadIdx.x;
    float S[5] = {}, Sx[5] = {}, Sy[5] = {}, Sz[5] = {};
    for (u32 bi = 0; bi < refp_nbf3d; ++bi)
    {
      float bfnc  = REFP_VEVAL(qi, bi); // TODO: load into shared?
      // float* grad = VGRAD(ei, qi, bi);  // TODO: load into shared?

      // float3 gradvec = *((float3*)grad);
      float grad0 = VGRAD(ei, qi, bi, 0);
      float grad1 = VGRAD(ei, qi, bi, 1);
      float grad2 = VGRAD(ei, qi, bi, 2);

      float U0 = shared_state[refp_nbf3d * 0 + bi];
      float U1 = shared_state[refp_nbf3d * 1 + bi];
      float U2 = shared_state[refp_nbf3d * 2 + bi];
      float U3 = shared_state[refp_nbf3d * 3 + bi];
      float U4 = shared_state[refp_nbf3d * 4 + bi];

      // float U0 = U(ei, 0, bi);
      // float U1 = U(ei, 1, bi);
      // float U2 = U(ei, 2, bi);
      // float U3 = U(ei, 3, bi);
      // float U4 = U(ei, 4, bi);

      S[0] += U0 * bfnc;
      S[1] += U1 * bfnc;
      S[2] += U2 * bfnc;
      S[3] += U3 * bfnc;
      S[4] += U4 * bfnc;

      Sx[0] += U0 * grad0;
      Sx[1] += U1 * grad0;
      Sx[2] += U2 * grad0;
      Sx[3] += U3 * grad0;
      Sx[4] += U4 * grad0;

      Sy[0] += U0 * grad1;
      Sy[1] += U1 * grad1;
      Sy[2] += U2 * grad1;
      Sy[3] += U3 * grad1;
      Sy[4] += U4 * grad1;

      Sz[0] += U0 * grad2;
      Sz[1] += U1 * grad2;
      Sz[2] += U2 * grad2;
      Sz[3] += U3 * grad2;
      Sz[4] += U4 * grad2;
    }

    float Floc[15];
    analytical_flux(S, gamma, Floc + 0, Floc + 5, Floc + 10);

    float Qloc[15];
    A(S, Sx, Sy, Sz, mu, Qloc + 0, Qloc + 5, Qloc + 10);

    for (u32 ri = 0; ri < 5; ++ri)
    {
      float3 flux;
      flux.x = Qloc[5 * 0 + ri] - Floc[5 * 0 + ri];
      flux.y = Qloc[5 * 1 + ri] - Floc[5 * 1 + ri];
      flux.z = Qloc[5 * 2 + ri] - Floc[5 * 2 + ri];

      Fflt3[vqrule_n * ri + qi] = flux;

      // F[15 * qi + 5 * 0 + ri] = Qloc[5 * 0 + ri] - Floc[5 * 0 + ri];
      // F[15 * qi + 5 * 1 + ri] = Qloc[5 * 1 + ri] - Floc[5 * 1 + ri];
      // F[15 * qi + 5 * 2 + ri] = Qloc[5 * 2 + ri] - Floc[5 * 2 + ri];
    }
  }
  __syncthreads();

  /* accumulate residual into shared mem (all threads in block) */

  for (u32 i = threadIdx.x; i < refp_nbf3d; i += blockDim.x)
  {
    u32 ti                             = i;
    float local_resid[5] = {};
    for (u32 qi = 0; qi < vqrule_n; ++qi)
    {
      float qw     = vqrule_w[qi];
      float J      = VJ(ei, qi);
      float tgrad0 = VGRAD(ei, qi, ti, 0);
      float tgrad1 = VGRAD(ei, qi, ti, 1);
      float tgrad2 = VGRAD(ei, qi, ti, 2);

      // float* Fx = F + 15 * qi + 0;
      // float* Fy = F + 15 * qi + 5;
      // float* Fz = F + 15 * qi + 10;

      for (u32 ri = 0; ri < 5; ++ri)
      {
        float3 flux = Fflt3[vqrule_n * ri + qi];
        // local_resid[ri] +=
        // (tgrad[0] * Fx[ri] + tgrad[1] * Fy[ri] + tgrad[2] * Fz[ri]) * (J * qw);
        local_resid[ri] +=
        (tgrad0 * flux.x + tgrad1 * flux.y + tgrad2 * flux.z) * (J * qw);
      }
    }
    for (u32 ri = 0; ri < 5; ++ri)
    {
      R(ei, ri, ti) += local_resid[ri];
    }
  }
  __syncthreads();
}

__global__ void cuda_source(cuda_device_geometry cugeom, parameters params,
                            float* residual)
{

  u32 refp_nbf3d    = cugeom.refp_nbf3d;
  u32 vqrule_n      = cugeom.vqrule_n;

  u32 shared_start    = 0;

  u32 ei    = blockIdx.x;
  u32 nelem = cugeom.nelem;
  if (ei < nelem)
  {
    float* refp_veval = cugeom.refp_veval;
    float* geo_vJ     = cugeom.geo_vJ;

    for (u32 i = threadIdx.x; i < refp_nbf3d; i += blockDim.x)
    {
      u32 ti = i;
      float local_resid[5] = {};
      for (u32 qi = 0; qi < vqrule_n; ++qi)
      {
        float qw = cugeom.vqrule_w[qi];
        float J  = VJ(ei, qi);
        float tfnc = REFP_VEVAL(qi, ti);
        for (u32 ri = 0; ri < 5; ++ri)
        {
          local_resid[ri] -= tfnc * params.source[ri] * J * qw;
        }
      }
      for (u32 ri = 0; ri < 5; ++ri)
      {
        R(ei, ri, ti) += local_resid[ri];
      }
    }
  }
}

__device__ void cuda_spmm(u32 n, u32 p, float* M, float* v, float c, float* r,
                          u32 irow)
{
  if (irow < n)
  {
    for (u32 ivec = 0; ivec < p; ++ivec)
    {
      float tmp = 0.f;
      for (u32 icol = 0; icol < n; ++icol)
      {
        u32 iMu = ((icol + 1) + ((2 * n - (irow + 1)) * (irow) / 2)) - 1;
        u32 iMl = ((irow + 1) + ((2 * n - (icol + 1)) * (icol) / 2)) - 1;

        u32 iM = (icol >= irow) * iMu + (icol < irow) * iMl;

        tmp += M[iM] * c * v[n * ivec + icol];
      }
      r[n * ivec + irow] = tmp;
    }
  }
}

// note that SL should alreay be zero initialized before calling this function
__device__ void cuda_compute_face_state(s32 lf, u32 qi, u32 nelem,
                                        u32 refp_nbf3d, u32 fqrule_n,
                                        float* shared_state, float* refp_feval,
                                        float* S)
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float bf = REFP_FEVAL(lf, qi, bi);

    S[0] += shared_state[refp_nbf3d * 0 + bi] * bf;
    S[1] += shared_state[refp_nbf3d * 1 + bi] * bf;
    S[2] += shared_state[refp_nbf3d * 2 + bi] * bf;
    S[3] += shared_state[refp_nbf3d * 3 + bi] * bf;
    S[4] += shared_state[refp_nbf3d * 4 + bi] * bf;
  }
}

// note that SL should alreay be zero initialized before calling this function
__device__ void cuda_compute_face_state_global(u32 ei, s32 lf, u32 qi,
                                               u32 nelem, u32 refp_nbf3d,
                                               u32 fqrule_n, float* state,
                                               float* refp_feval, float* S)
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float bf = REFP_FEVAL(lf, qi, bi);

    S[0] += U(ei, 0, bi) * bf;
    S[1] += U(ei, 1, bi) * bf;
    S[2] += U(ei, 2, bi) * bf;
    S[3] += U(ei, 3, bi) * bf;
    S[4] += U(ei, 4, bi) * bf;
  }
}

// note you should zero before calling this function
__device__ void cuda_compute_face_grad(u32 ei, u32 lf, u32 qi, u32 nelem,
                                       u32 refp_nbf3d, u32 fqrule_n,
                                       float* shared_state, float* geo_fgrad,
                                       float Sx[5], float Sy[5], float Sz[5])
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float* grad = FGRAD(ei, lf, qi, bi);

    float U0 = shared_state[refp_nbf3d * 0 + bi];
    float U1 = shared_state[refp_nbf3d * 1 + bi];
    float U2 = shared_state[refp_nbf3d * 2 + bi];
    float U3 = shared_state[refp_nbf3d * 3 + bi];
    float U4 = shared_state[refp_nbf3d * 4 + bi];

    Sx[0] += U0 * grad[0];
    Sx[1] += U1 * grad[0];
    Sx[2] += U2 * grad[0];
    Sx[3] += U3 * grad[0];
    Sx[4] += U4 * grad[0];

    Sy[0] += U0 * grad[1];
    Sy[1] += U1 * grad[1];
    Sy[2] += U2 * grad[1];
    Sy[3] += U3 * grad[1];
    Sy[4] += U4 * grad[1];

    Sz[0] += U0 * grad[2];
    Sz[1] += U1 * grad[2];
    Sz[2] += U2 * grad[2];
    Sz[3] += U3 * grad[2];
    Sz[4] += U4 * grad[2];
  }
}

// note you should zero before calling this function
__device__ void cuda_compute_face_grad_global(u32 ei, u32 lf, u32 qi, u32 nelem,
                                              u32 refp_nbf3d, u32 fqrule_n,
                                              float* state, float* geo_fgrad,
                                              float Sx[5], float Sy[5],
                                              float Sz[5])
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float* grad = FGRAD(ei, lf, qi, bi);

    float U0 = U(ei, 0, bi);
    float U1 = U(ei, 1, bi);
    float U2 = U(ei, 2, bi);
    float U3 = U(ei, 3, bi);
    float U4 = U(ei, 4, bi);

    Sx[0] += U0 * grad[0];
    Sx[1] += U1 * grad[0];
    Sx[2] += U2 * grad[0];
    Sx[3] += U3 * grad[0];
    Sx[4] += U4 * grad[0];

    Sy[0] += U0 * grad[1];
    Sy[1] += U1 * grad[1];
    Sy[2] += U2 * grad[1];
    Sy[3] += U3 * grad[1];
    Sy[4] += U4 * grad[1];

    Sz[0] += U0 * grad[2];
    Sz[1] += U1 * grad[2];
    Sz[2] += U2 * grad[2];
    Sz[3] += U3 * grad[2];
    Sz[4] += U4 * grad[2];
  }
}

__device__ void cuda_compute_face_all_global(u32 ei, u32 lf, u32 qi, u32 nelem,
                                             u32 refp_nbf3d, u32 fqrule_n,
                                             float* state, float* refp_feval,
                                             float* geo_fgrad, float S[5],
                                             float Sx[5], float Sy[5],
                                             float Sz[5])
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float bf    = REFP_FEVAL(lf, qi, bi);
    float* grad = FGRAD(ei, lf, qi, bi);

    float U0 = U(ei, 0, bi);
    float U1 = U(ei, 1, bi);
    float U2 = U(ei, 2, bi);
    float U3 = U(ei, 3, bi);
    float U4 = U(ei, 4, bi);

    S[0] += U0 * bf;
    S[1] += U1 * bf;
    S[2] += U2 * bf;
    S[3] += U3 * bf;
    S[4] += U4 * bf;

    Sx[0] += U0 * grad[0];
    Sx[1] += U1 * grad[0];
    Sx[2] += U2 * grad[0];
    Sx[3] += U3 * grad[0];
    Sx[4] += U4 * grad[0];

    Sy[0] += U0 * grad[1];
    Sy[1] += U1 * grad[1];
    Sy[2] += U2 * grad[1];
    Sy[3] += U3 * grad[1];
    Sy[4] += U4 * grad[1];

    Sz[0] += U0 * grad[2];
    Sz[1] += U1 * grad[2];
    Sz[2] += U2 * grad[2];
    Sz[3] += U3 * grad[2];
    Sz[4] += U4 * grad[2];
  }
}

__device__ void cuda_compute_face_all(u32 ei, u32 lf, u32 qi, u32 nelem,
                                      u32 refp_nbf3d, u32 fqrule_n,
                                      float* shared_state, float* refp_feval,
                                      float* geo_fgrad, float S[5], float Sx[5],
                                      float Sy[5], float Sz[5])
{
  for (u32 bi = 0; bi < refp_nbf3d; ++bi)
  {
    float bf    = REFP_FEVAL(lf, qi, bi);
    float* grad = FGRAD(ei, lf, qi, bi);

    float U0 = shared_state[refp_nbf3d * 0 + bi];
    float U1 = shared_state[refp_nbf3d * 1 + bi];
    float U2 = shared_state[refp_nbf3d * 2 + bi];
    float U3 = shared_state[refp_nbf3d * 3 + bi];
    float U4 = shared_state[refp_nbf3d * 4 + bi];

    S[0] += U0 * bf;
    S[1] += U1 * bf;
    S[2] += U2 * bf;
    S[3] += U3 * bf;
    S[4] += U4 * bf;

    Sx[0] += U0 * grad[0];
    Sx[1] += U1 * grad[0];
    Sx[2] += U2 * grad[0];
    Sx[3] += U3 * grad[0];
    Sx[4] += U4 * grad[0];

    Sy[0] += U0 * grad[1];
    Sy[1] += U1 * grad[1];
    Sy[2] += U2 * grad[1];
    Sy[3] += U3 * grad[1];
    Sy[4] += U4 * grad[1];

    Sz[0] += U0 * grad[2];
    Sz[1] += U1 * grad[2];
    Sz[2] += U2 * grad[2];
    Sz[3] += U3 * grad[2];
    Sz[4] += U4 * grad[2];
  }
}

__global__ void cuda_interior_face_residual(cuda_device_geometry cugeom,
                                            parameters params,
                                            cuda_residual_workspace wsp,
                                            float* state)
{
  extern __shared__ u32 shared[];

  u32 ifi = blockIdx.x;

  u32 nelem             = cugeom.nelem;
  u32 refp_nbf3d        = cugeom.refp_nbf3d;
  u32 fqrule_n          = cugeom.fqrule_n;
  float* refp_feval     = cugeom.refp_feval;
  float* fqrule_w       = cugeom.fqrule_w;
  // float* geo_Minv       = cugeom.geo_Minv;
  float* geo_fgrad      = cugeom.geo_fgrad;
  s32* geo_ifl          = cugeom.geo_ifl;
  float* geo_n          = cugeom.geo_n;
  float* geo_fJ         = cugeom.geo_fJ;
  float* face_integrals = wsp.face_integrals;

  u32 shared_start = 0;

  float* F      = (float*)shared + shared_start;
  shared_start += fqrule_n * 5;
  float* Qh     = (float*)shared + shared_start;
  shared_start += fqrule_n * 5;
  float* dcQL   = (float*)shared + shared_start;
  shared_start += fqrule_n * 15;
  float* dcQR   = (float*)shared + shared_start;
  shared_start += fqrule_n * 15;

  s32 fi  = geo_ifl[5 * ifi + 0];
  s32 eL  = geo_ifl[5 * ifi + 1];
  s32 eR  = geo_ifl[5 * ifi + 2];
  s32 lfL = geo_ifl[5 * ifi + 3];
  s32 lfR = geo_ifl[5 * ifi + 4];

  // float mu    = params.mu;
  // float eta   = params.eta;
  // float gamma = params.gamma;

  /* find fluxes from all terms */

  for (u32 qi = threadIdx.x; qi < fqrule_n; qi += blockDim.x)
  {
    float SL[5] = {}, SR[5] = {};
    float SLx[5] = {}, SLy[5] = {}, SLz[5] = {};
    float SRx[5] = {}, SRy[5] = {}, SRz[5] = {};

    cuda_compute_face_all_global(eL, lfL, qi, nelem, refp_nbf3d, fqrule_n,
                                 state, refp_feval, geo_fgrad, SL, SLx, SLy,
                                 SLz);
    cuda_compute_face_all_global(eR, lfR, qi, nelem, refp_nbf3d, fqrule_n,
                                 state, refp_feval, geo_fgrad, SR, SRx, SRy,
                                 SRz);

    float* n = N(fi, qi);

    /* inviscid flux term */

    {
      float smax;
      roe(SL, SR, n, params.gamma, F + 5 * qi, &smax);
    }

    /* dual consistency term */

    {
      float Sh[5];
      for (u32 ri = 0; ri < 5; ++ri)
        Sh[ri] = 0.5 * (SL[ri] + SR[ri]);

      float dL[5], dR[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dL[ri] = SL[ri] - Sh[ri];
        dR[ri] = SR[ri] - Sh[ri];
      }

      float dLnx[5], dLny[5], dLnz[5], dRnx[5], dRny[5], dRnz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dLnx[ri] = dL[ri] * n[0];
        dLny[ri] = dL[ri] * n[1];
        dLnz[ri] = dL[ri] * n[2];

        dRnx[ri] = dR[ri] * -n[0];
        dRny[ri] = dR[ri] * -n[1];
        dRnz[ri] = dR[ri] * -n[2];
      }

      float* dcQLx = dcQL + qi * 15 + 0;
      float* dcQLy = dcQL + qi * 15 + 5;
      float* dcQLz = dcQL + qi * 15 + 10;
      float* dcQRx = dcQR + qi * 15 + 0;
      float* dcQRy = dcQR + qi * 15 + 5;
      float* dcQRz = dcQR + qi * 15 + 10;

      A(SL, dLnx, dLny, dLnz, params.mu, dcQLx, dcQLy, dcQLz);
      A(SR, dRnx, dRny, dRnz, params.mu, dcQRx, dcQRy, dcQRz);
    }

    /* viscous flux term */

    {
      float dLx[5] = {}, dLy[5] = {}, dLz[5] = {};
      float dRx[5] = {}, dRy[5] = {}, dRz[5] = {};

      float diffL[5], diffR[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        diffL[ri] = SL[ri] - SR[ri];
        diffR[ri] = SR[ri] - SL[ri];
      }

      float DLNx[5], DLNy[5], DLNz[5], DRNx[5], DRNy[5], DRNz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        DLNx[ri] = diffL[ri] * n[0];
        DLNy[ri] = diffL[ri] * n[1];
        DLNz[ri] = diffL[ri] * n[2];

        DRNx[ri] = diffR[ri] * -n[0];
        DRNy[ri] = diffR[ri] * -n[1];
        DRNz[ri] = diffR[ri] * -n[2];
      }

      float QLx[5], QLy[5], QLz[5], QRx[5], QRy[5], QRz[5];
      A(SL, DLNx, DLNy, DLNz, params.mu, QLx, QLy, QLz);
      A(SR, DRNx, DRNy, DRNz, params.mu, QRx, QRy, QRz);

      float hL = 1.f / cugeom.geo_ih[2 * ifi + 0];
      float hR = 1.f / cugeom.geo_ih[2 * ifi + 1];
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

      float QHx[5], QHy[5], QHz[5];
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
        Qh[5 * qi + ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
    }
  }
  __syncthreads();

  /* residual accumulation */

  for (u32 ti = threadIdx.x; ti < refp_nbf3d; ti += blockDim.x)
  {
    float local_residL[5] = {};
    float local_residR[5] = {};
    for (u32 qi = 0; qi < fqrule_n; ++qi)
    {
      float J    = FJ(fi, qi);
      float qw   = fqrule_w[qi];
      float tfL  = REFP_FEVAL(lfL, qi, ti);
      float tfR  = REFP_FEVAL(lfR, qi, ti);
      float* tgL = FGRAD(eL, lfL, qi, ti);
      float* tgR = FGRAD(eR, lfR, qi, ti);

      float* dcQLx = dcQL + qi * 15 + 0;
      float* dcQLy = dcQL + qi * 15 + 5;
      float* dcQLz = dcQL + qi * 15 + 10;
      float* dcQRx = dcQR + qi * 15 + 0;
      float* dcQRy = dcQR + qi * 15 + 5;
      float* dcQRz = dcQR + qi * 15 + 10;

      for (u32 ri = 0; ri < 5; ++ri)
      {
       local_residL[ri] +=
       (+(tfL * F[5 * qi + ri]) -
        (tgL[0] * dcQLx[ri] + tgL[1] * dcQLy[ri] + tgL[2] * dcQLz[ri]) -
        (tfL * Qh[5 * qi + ri])) *
       (J * qw);
       local_residR[ri] +=
       (-(tfR * F[5 * qi + ri]) +
        (tgR[0] * dcQRx[ri] + tgR[1] * dcQRy[ri] + tgR[2] * dcQRz[ri]) +
        (tfR * Qh[5 * qi + ri])) *
       (J * qw);
      }
    }
    for (u32 ri = 0; ri < 5; ++ri)
    {
      FACERESID(eL, lfL, ri, ti) += local_residL[ri];
      FACERESID(eR, lfR, ri, ti) += local_residR[ri];
    }
  }
  __syncthreads();
}

__device__ void cuda_compute_bound_state(s32 bt, parameters params, float SL[5],
                                         float n[3], float gamma, float Pt,
                                         float Tt, float* Sb)
{
   switch (bt)
   {
     case freestream:
       for (u32 i = 0; i < 5; ++i) Sb[i] = params.Ufr[i];
     break;
     case inviscid_wall:
       inviscid_wall_state(SL, n, gamma, Sb);
     break;
     case no_slip_wall:
       no_slip_wall_state(SL, Sb);
     break;
     case subsonic_inflow:
       subsonic_inflow_state(SL, n, gamma, params.R, Pt, Tt, Sb);
     break;
     case subsonic_outflow:
       subsonic_outflow_state(SL, n, gamma, params.P, Sb);
     break;
   };
}

__global__ void cuda_boundary_face_residual(cuda_device_geometry cugeom,
                                            parameters params,
                                            cuda_residual_workspace wsp,
                                            float* state)
{
  extern __shared__ u32 shared[];

  u32 bfi = blockIdx.x;

  u32 nelem             = cugeom.nelem;
  u32 refp_nbf3d        = cugeom.refp_nbf3d;
  u32 fqrule_n          = cugeom.fqrule_n;
  float* fqrule_w       = cugeom.fqrule_w;
  float* refp_feval     = cugeom.refp_feval;
  float* geo_Minv       = cugeom.geo_Minv;
  float* geo_fgrad      = cugeom.geo_fgrad;
  s32* geo_bfl          = cugeom.geo_bfl;
  float* geo_n          = cugeom.geo_n;
  float* geo_fJ         = cugeom.geo_fJ;
  float* face_integrals = wsp.face_integrals;

  float gamma = params.gamma;
  float Tt, Pt;
  {
    float gm1 = gamma - 1.;
    float u   = params.Ufr[1] / params.Ufr[0];
    float v   = params.Ufr[2] / params.Ufr[0];
    float w   = params.Ufr[3] / params.Ufr[0];
    float M   = sqrt(u * u + v * v + w * w) / params.c;
    Tt        = params.T * (1. + 0.5 * gm1 * M * M);
    Pt        = params.P * pow(Tt / params.T, gamma / gm1);
  }

  u32 shared_start = 0;
  float* shared_stateL = (float*)shared + shared_start;
  shared_start += refp_nbf3d * 5;
  float* RL            = (float*)shared + shared_start;
  shared_start += refp_nbf3d * 15;
  float* DL            = (float*)shared + shared_start;
  shared_start += refp_nbf3d * 15;
  float* F             = (float*)shared + shared_start;
  shared_start += fqrule_n * 5;
  float* Qh            = (float*)shared + shared_start;
  shared_start += fqrule_n * 5;
  float* dcQL          = (float*)shared + shared_start;
  shared_start += fqrule_n * 15;
  float* shared_residL = (float*)shared + shared_start;
  shared_start += refp_nbf3d * 5;

  s32 fi  = geo_bfl[4 * bfi + 0];
  s32 eL  = geo_bfl[4 * bfi + 1];
  s32 bt  = geo_bfl[4 * bfi + 2];
  s32 lfL = geo_bfl[4 * bfi + 3];

  u32 dst   = 5 * refp_nbf3d;
  float eta = params.eta;
  float mu  = params.mu;

  /* load state into shared memory (also zero br2 stuff) (all threads) */

  for (u32 i = threadIdx.x; i < refp_nbf3d * 5; i += blockDim.x)
  {
    u32 ri = i / refp_nbf3d;
    u32 bi = i - refp_nbf3d * ri;
    shared_stateL[refp_nbf3d * ri + bi] = U(eL, ri, bi);
  }
  __syncthreads();

  /* find fluxes from all terms */

  if (threadIdx.x < fqrule_n)
  {
    u32 qi = threadIdx.x;

    float SL[5] = {};
    cuda_compute_face_state(lfL, qi, nelem, refp_nbf3d, fqrule_n, shared_stateL,
                            refp_feval, SL);

    float Sb[5] = {};
    float* n    = N(fi, qi);
    cuda_compute_bound_state(bt, params, SL, n, gamma, Pt, Tt, Sb);

    /* inviscid flux term */

    {
      float Fx[5], Fy[5], Fz[5];
      analytical_flux(Sb, gamma, Fx, Fy, Fz);

      F[5 * qi + 0] = Fx[0] * n[0] + Fy[0] * n[1] + Fz[0] * n[2];
      F[5 * qi + 1] = Fx[1] * n[0] + Fy[1] * n[1] + Fz[1] * n[2];
      F[5 * qi + 2] = Fx[2] * n[0] + Fy[2] * n[1] + Fz[2] * n[2];
      F[5 * qi + 3] = Fx[3] * n[0] + Fy[3] * n[1] + Fz[3] * n[2];
      F[5 * qi + 4] = Fx[4] * n[0] + Fy[4] * n[1] + Fz[4] * n[2];
    }

    /* dual consistency term */

    // float dcQLx[5], dcQLy[5], dcQLz[5];
    {
      float Sh[5];
      for (u32 i = 0; i < 5; ++i)
        Sh[i] = Sb[i];

      float diffL[5];
      for (u32 i = 0; i < 5; ++i)
        diffL[i] = SL[i] - Sh[i];

      float DLNx[5], DLNy[5], DLNz[5];
      for (u32 i = 0; i < 5; ++i)
      {
        DLNx[i] = diffL[i] * n[0];
        DLNy[i] = diffL[i] * n[1];
        DLNz[i] = diffL[i] * n[2];
      }

      float* dcQLx = dcQL + qi * 15 + 0;
      float* dcQLy = dcQL + qi * 15 + 5;
      float* dcQLz = dcQL + qi * 15 + 10;

      A(SL, DLNx, DLNy, DLNz, mu, dcQLx, dcQLy, dcQLz);
    }

    /* viscous flux term */

    {
      float dLx[5] = {}, dLy[5] = {}, dLz[5] = {};

      float diffL[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        diffL[ri] = SL[ri] - Sb[ri];
      }

      float DLNx[5], DLNy[5], DLNz[5];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        DLNx[ri] = diffL[ri] * n[0];
        DLNy[ri] = diffL[ri] * n[1];
        DLNz[ri] = diffL[ri] * n[2];
      }

      float QLx[5], QLy[5], QLz[5];
      A(SL, DLNx, DLNy, DLNz, mu, QLx, QLy, QLz);

      float hL = 1.f / cugeom.geo_bh[bfi];
      for (u32 ri = 0; ri < 5; ++ri)
      {
        dLx[ri] = hL * QLx[ri];
        dLy[ri] = hL * QLy[ri];
        dLz[ri] = hL * QLz[ri];
      }

      float SLx[5] = {}, SLy[5] = {}, SLz[5] = {};
      cuda_compute_face_grad(eL, lfL, qi, nelem, refp_nbf3d, fqrule_n,
                             shared_stateL, geo_fgrad, SLx, SLy, SLz);

      A(SL, SLx, SLy, SLz, mu, QLx, QLy, QLz);

      float QHx[5], QHy[5], QHz[5];
      for (u64 ri = 0; ri < 5; ++ri)
      {
        QHx[ri] = QLx[ri] - eta * dLx[ri];
        QHy[ri] = QLy[ri] - eta * dLy[ri];
        QHz[ri] = QLz[ri] - eta * dLz[ri];
      }

      for (u64 ri = 0; ri < 5; ++ri)
        Qh[5 * qi + ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
    }
  }
  __syncthreads();

  /* residual accumulation */

  for (u32 i = threadIdx.x; i < refp_nbf3d * 5; i += blockDim.x)
  {
    u32 ri                              = i / refp_nbf3d;
    u32 ti                              = i - refp_nbf3d * ri;
    shared_residL[refp_nbf3d * ri + ti] = 0.f;
    for (u32 qi = 0; qi < fqrule_n; ++qi)
    {
      float J    = FJ(fi, qi);
      float qw   = fqrule_w[qi];
      float tfL  = REFP_FEVAL(lfL, qi, ti);
      float* tgL = FGRAD(eL, lfL, qi, ti);

      float* dcQLx = dcQL + qi * 15 + 0;
      float* dcQLy = dcQL + qi * 15 + 5;
      float* dcQLz = dcQL + qi * 15 + 10;

      shared_residL[refp_nbf3d * ri + ti] +=
      ((tfL * F[5 * qi + ri]) -
       (tgL[0] * dcQLx[ri] + tgL[1] * dcQLy[ri] + tgL[2] * dcQLz[ri]) -
       (tfL * Qh[5 * qi + ri])) *
      (J * qw);
    }
  }
  __syncthreads();

  /* residual write back to global mem */

  for (u32 i = threadIdx.x; i < refp_nbf3d * 5; i += blockDim.x)
  {
    u32 ri = i / refp_nbf3d;
    u32 ti = i - refp_nbf3d * ri;
    FACERESID(eL, lfL, ri, ti) = shared_residL[refp_nbf3d * ri + ti];
  }
  __syncthreads();
}

__global__ void cuda_accumulate_face_residuals(u32 nelem, u32 refp_nbf3d,
                                               float* face_integrals,
                                               float* residual)
{
  u32 ei = blockIdx.x;
  u32 ti = threadIdx.x;

  if (ei < nelem && ti < refp_nbf3d)
  {
    for (u32 lfi = 0; lfi < 6; ++lfi)
    {
      R(ei, 0, ti) += FACERESID(ei, lfi, 0, ti);
      R(ei, 1, ti) += FACERESID(ei, lfi, 1, ti);
      R(ei, 2, ti) += FACERESID(ei, lfi, 2, ti);
      R(ei, 3, ti) += FACERESID(ei, lfi, 3, ti);
      R(ei, 4, ti) += FACERESID(ei, lfi, 4, ti);
    }
  }
}

__global__ void cuda_update_state(u32 nelem, u32 refp_nbf3d, float* geo_Minv,
                                  float* residual, float* f)
{
  u32 ei   = blockIdx.x;
  u32 irow = threadIdx.x;

  if (ei < nelem)
  {
    cuda_spmm(refp_nbf3d, 5, MINV(ei), &R(ei, 0, 0), -1., &F(ei, 0, 0), irow);
  }
}

__host__ u32 idiv_ceil(u32 a, u32 b)
{
  return (a + b - 1) / b;
}

__host__ u32 round_up_32(u32 a)
{
  return (((a - 1) / 32) + 1) * 32;
}


__host__ void cuda_residual(float* state, cuda_device_geometry h_cugeom,
                            parameters params, cuda_residual_workspace wsp,
                            float* residual, float* f, shared_geometry& geom,
                            float* dbg_resid)
{
  cuda_zero_array<<<idiv_ceil(wsp.solarr_size, 256), 256>>>(
  wsp.solarr_size, residual);
  cudaDeviceSynchronize();

  cuda_zero_array<<<idiv_ceil(wsp.face_integrals_size, 256), 256>>>(
  wsp.face_integrals_size, wsp.face_integrals);
  cudaDeviceSynchronize();

  u32 interior_block_size = round_up_32(h_cugeom.vqrule_n);
  cuda_interior_residual<<<h_cugeom.nelem, interior_block_size,
                           ((1 * 5 * h_cugeom.refp_nbf3d) +
                            (1 * 15 * h_cugeom.vqrule_n)) *
                           sizeof(float)>>>(h_cugeom, params.mu, params.gamma,
                                            state, residual);
  cudaDeviceSynchronize();

  cuda_source<<<h_cugeom.nelem, round_up_32(h_cugeom.vqrule_n)>>>(
  h_cugeom, params, residual);
  cudaDeviceSynchronize();

  u32 interior_face_block_size = round_up_32(h_cugeom.fqrule_n);
  cuda_interior_face_residual<<<h_cugeom.niface, interior_face_block_size,
                                (2 * (h_cugeom.fqrule_n * 5) +
                                 2 * (h_cugeom.fqrule_n * 15)) *
                                sizeof(float)>>>(h_cugeom, params, wsp, state);
  cudaDeviceSynchronize();

  u32 boundary_face_block_size = round_up_32(h_cugeom.fqrule_n);
  cuda_boundary_face_residual<<<
  h_cugeom.nbface, boundary_face_block_size,
  (2 * h_cugeom.refp_nbf3d * 5 + 2 * h_cugeom.refp_nbf3d * 15 +
   2 * h_cugeom.fqrule_n * 5 + h_cugeom.fqrule_n * 15) *
  sizeof(float)>>>(h_cugeom, params, wsp, state);
  cudaDeviceSynchronize();

  cuda_accumulate_face_residuals<<<h_cugeom.nelem,
                                   round_up_32(h_cugeom.refp_nbf3d)>>>(
  h_cugeom.nelem, h_cugeom.refp_nbf3d, wsp.face_integrals, residual);
  cudaDeviceSynchronize();

  cuda_update_state<<<h_cugeom.nelem, round_up_32(h_cugeom.refp_nbf3d)>>>(
  h_cugeom.nelem, h_cugeom.refp_nbf3d, h_cugeom.geo_Minv, residual, f);
  cudaDeviceSynchronize();

  {
    array<real> h_R_shuffle(wsp.solarr_size);
    cudaMemcpy(h_R_shuffle.data, f, wsp.solarr_size * sizeof(real),
               cudaMemcpyDeviceToHost);

    simstate h_R(geom.core);
    unshuffle_state(h_R_shuffle.data, geom, h_R);

    for (u32 i = 0; i < h_R.size(); ++i)
    {
      dbg_resid[i] = h_R[i];
    }
  }
}

#undef NENTRYMINV
#undef REFP_VEVAL
#undef REFP_FEVAL
#undef MINV
#undef VJ
#undef VGRAD
#undef FGRAD
#undef FJ
#undef N
#undef U
#undef R
#undef F
#undef FACERESID
