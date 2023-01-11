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

#include "bound_states.cpp"
#include "flux.cpp"
#include "geometry_distributed.cpp"
#include "solver_params.cpp"

// residuals -------------------------------------------------------------------

void eval_state_face(simstate& U, u64 ei, u64 lf, u64 qi,
                     distributed_geometry& geom, real* S)
{
  S[0] = 0.;
  S[1] = 0.;
  S[2] = 0.;
  S[3] = 0.;
  S[4] = 0.;
  for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
  {
    real beval = geom.core.refp.feval_at(lf, bi, qi);
    S[0] += U(ei, 0, bi) * beval;
    S[1] += U(ei, 1, bi) * beval;
    S[2] += U(ei, 2, bi) * beval;
    S[3] += U(ei, 3, bi) * beval;
    S[4] += U(ei, 4, bi) * beval;
  }
}

void eval_grad_face(simstate& U, u64 ei, u64 lf, u64 qi,
                    distributed_geometry& geom, real* Sx, real* Sy, real* Sz)
{
  for (u64 ri = 0; ri < 5; ++ri)
  {
    Sx[ri] = 0.;
    Sy[ri] = 0.;
    Sz[ri] = 0.;
  }
  for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
  {
    real* grad = geom.core.fgrad(ei, lf, bi, qi);
    for (u64 ri = 0; ri < 5; ++ri)
    {
      Sx[ri] += U(ei, ri, bi) * grad[0];
      Sy[ri] += U(ei, ri, bi) * grad[1];
      Sz[ri] += U(ei, ri, bi) * grad[2];
    }
  }
}

struct residual_workspace
{
  array<real> RL;
  array<real> RR;
  array<real> DL;
  array<real> DR;

  // Evaluations for each state of each rank on each quadrature point in the
  // domain. Storage order:
  //   rank
  //     face
  //       quad point
  array<real> ULst;
  array<real> URst;
  array<real> Ubst;

  array<real> ULgrad;
  array<real> URgrad;

  residual_workspace(distributed_geometry& geom);
};

residual_workspace::residual_workspace(distributed_geometry& geom) :
RL(5 * 3 * geom.core.refp.nbf3d),
RR(5 * 3 * geom.core.refp.nbf3d),
DL(5 * 3 * geom.core.refp.nbf3d),
DR(5 * 3 * geom.core.refp.nbf3d),
ULst(geom.core.refp.fqrule.n * geom.nface() * 5),
URst(geom.core.refp.fqrule.n * geom.nface() * 5),
Ubst(geom.core.refp.fqrule.n * geom.nface() * 5),
ULgrad(geom.core.refp.fqrule.n * geom.nface() * 5 * 3),
URgrad(geom.core.refp.fqrule.n * geom.nface() * 5 * 3)
{}

void to_state_store(distributed_geometry& geom, u64 fi, u64 qi, real* state,
                    array<real>& state_store)
{
  for (u64 ri = 0; ri < 5; ++ri)
  {
    state_store[geom.core.refp.fqrule.n * geom.nface() * ri +
                geom.core.refp.fqrule.n * fi + qi] = state[ri];
  }
}

void from_state_store(distributed_geometry& geom, u64 fi, u64 qi,
                      array<real>& state_store, real* state)
{
  for (u64 ri = 0; ri < 5; ++ri)
  {
    state[ri] = state_store[geom.core.refp.fqrule.n * geom.nface() * ri +
                            geom.core.refp.fqrule.n * fi + qi];
  }
}

void to_grad_store(distributed_geometry& geom, u64 fi, u64 qi, real* Ux,
                   real* Uy, real* Uz, array<real>& grad_store)
{
  for (u64 ri = 0; ri < 5; ++ri)
  {
    grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 0 + ri) +
               geom.core.refp.fqrule.n * fi + qi] = Ux[ri];
    grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 1 + ri) +
               geom.core.refp.fqrule.n * fi + qi] = Uy[ri];
    grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 2 + ri) +
               geom.core.refp.fqrule.n * fi + qi] = Uz[ri];
  }
}

void from_grad_store(distributed_geometry& geom, u64 fi, u64 qi,
                     array<real>& grad_store, real* Ux, real* Uy, real* Uz)
{
  for (u64 ri = 0; ri < 5; ++ri)
  {
    Ux[ri] = grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 0 + ri) +
                        geom.core.refp.fqrule.n * fi + qi];
    Uy[ri] = grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 1 + ri) +
                        geom.core.refp.fqrule.n * fi + qi];
    Uz[ri] = grad_store[geom.core.refp.fqrule.n * geom.nface() * (5 * 2 + ri) +
                        geom.core.refp.fqrule.n * fi + qi];
  }
}

void residual(simstate& U, distributed_geometry& geom, parameters& sim_params,
              residual_workspace& wsp, simstate& R, simstate& f)
{
  real mu      = sim_params.mu;
  real gamma   = sim_params.gamma;
  real gcnst   = sim_params.R;
  real c       = sim_params.c;
  real T       = sim_params.T;
  real P       = sim_params.P;
  real* Uref   = sim_params.Ufr;
  real* source = sim_params.source;
  real eta     = sim_params.eta;  // BR2 stability constant

  real gm1  = gamma - 1.;
  u64 nelem = geom.core.nx * geom.core.ny * geom.core.nz;
  u64 nbfnc = geom.core.refp.nbf3d;

  real M, Tt, Pt;
  {
    real u = Uref[1] / Uref[0];
    real v = Uref[2] / Uref[0];
    real w = Uref[3] / Uref[0];
    M      = sqrt(u * u + v * v + w * w) / c;
    Tt     = T * (1. + 0.5 * gm1 * M * M);
    Pt     = P * pow(Tt / T, gamma / gm1);
  }

  u64 br2vsize = 5 * 3 * geom.core.refp.nbf3d;

  // zero the residual (you'll tack on to this as you evaluate each term)
  memset(&R[0], 0, R.size() * sizeof(R[0]));

  /* pre-calculate state on each quadrature point of each face */

  for (u64 ifi = 0; ifi < geom.num_interior_faces(); ++ifi)
  {
    s64 fi         = geom.interior_face_list[5 * ifi + 0];
    s64 eL         = geom.interior_face_list[5 * ifi + 1];
    s64 eR         = geom.interior_face_list[5 * ifi + 2];
    s64 lfL        = geom.interior_face_list[5 * ifi + 3];
    s64 lfR        = geom.interior_face_list[5 * ifi + 4];
    bool proc_edge = (eR < 0) ? true : false;

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real SL[5], SLx[5], SLy[5], SLz[5];
      real SR[5], SRx[5], SRy[5], SRz[5];

      eval_state_face(U, eL, lfL, qi, geom, SL);
      to_state_store(geom, fi, qi, SL, wsp.ULst);

      eval_grad_face(U, eL, lfL, qi, geom, SLx, SLy, SLz);
      to_grad_store(geom, fi, qi, SLx, SLy, SLz, wsp.ULgrad);

      if (!proc_edge)
      {
        eval_state_face(U, eR, lfR, qi, geom, SR);
        to_state_store(geom, fi, qi, SR, wsp.URst);

        eval_grad_face(U, eR, lfR, qi, geom, SRx, SRy, SRz);
        to_grad_store(geom, fi, qi, SRx, SRy, SRz, wsp.URgrad);
      }
    }
  }

  for (u64 bfi = 0; bfi < geom.num_boundary_faces(); ++bfi)
  {
    s64 fi  = geom.boundary_face_list[4 * bfi + 0];
    s64 eL  = geom.boundary_face_list[4 * bfi + 1];
    s64 bt  = geom.boundary_face_list[4 * bfi + 2];
    s64 lfL = geom.boundary_face_list[4 * bfi + 3];

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real SL[5], SLx[5], SLy[5], SLz[5], SB[5];

      real* n = geom.n(fi, qi);

      eval_state_face(U, eL, lfL, qi, geom, SL);
      to_state_store(geom, fi, qi, SL, wsp.ULst);

      eval_grad_face(U, eL, lfL, qi, geom, SLx, SLy, SLz);
      to_grad_store(geom, fi, qi, SLx, SLy, SLz, wsp.ULgrad);

      // clang-format off
      switch (bt)
      {
        case freestream:
          for (u64 i = 0; i < 5; ++i) SB[i] = Uref[i];
        break;
        case inviscid_wall:
          inviscid_wall_state(SL, n, gamma, SB);
        break;
        case no_slip_wall:
          no_slip_wall_state(SL, SB);
        break;
        case subsonic_inflow:
          subsonic_inflow_state(SL, n, gamma, gcnst, Pt, Tt, SB);
        break;
        case subsonic_outflow:
          subsonic_outflow_state(SL, n, gamma, P, SB);
        break;
      };
      // clang-format on
      to_state_store(geom, fi, qi, SB, wsp.Ubst);
    }
  }

  /* initiate communication of boundary face states */

  MPI_Request state_requests[6];
  MPI_Request grad_requests[6];
  for (u64 bi = 0; bi < 6; ++bi)
  {
    if (geom.core.bounds[bi] >= 0)
    {
      MPI_Issend(wsp.ULst + 0, 5, geom.tfer_dtypes[bi],
                 (int)geom.core.bounds[bi], bi, geom.comm, &state_requests[bi]);
      MPI_Issend(wsp.ULgrad + 0, 5 * 3, geom.tfer_dtypes[bi],
                 (int)geom.core.bounds[bi], bi + 6, geom.comm,
                 &grad_requests[bi]);
    }
  }

  /* interior residual contribution */

  for (u64 ei = 0; ei < nelem; ++ei)
  {
    for (u64 qi = 0; qi < geom.core.refp.vqrule.n; ++qi)
    {
      real Fx[5], Fy[5], Fz[5];
      real Qx[5], Qy[5], Qz[5];
      real S[5]  = {};
      real Sx[5] = {}, Sy[5] = {}, Sz[5] = {};

      real quad_w = geom.core.refp.vqrule.wgt(qi);
      real Jdet   = geom.core.vJ(ei, qi);

      for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
      {
        real beval = geom.core.refp.veval_at(bi, qi);
        real* grad = geom.core.vgrad(ei, bi, qi);
        real Ur0   = U(ei, 0, bi);
        real Ur1   = U(ei, 1, bi);
        real Ur2   = U(ei, 2, bi);
        real Ur3   = U(ei, 3, bi);
        real Ur4   = U(ei, 4, bi);

        S[0] += Ur0 * beval;
        S[1] += Ur1 * beval;
        S[2] += Ur2 * beval;
        S[3] += Ur3 * beval;
        S[4] += Ur4 * beval;

        Sx[0] += Ur0 * grad[0];
        Sx[1] += Ur1 * grad[0];
        Sx[2] += Ur2 * grad[0];
        Sx[3] += Ur3 * grad[0];
        Sx[4] += Ur4 * grad[0];

        Sy[0] += Ur0 * grad[1];
        Sy[1] += Ur1 * grad[1];
        Sy[2] += Ur2 * grad[1];
        Sy[3] += Ur3 * grad[1];
        Sy[4] += Ur4 * grad[1];

        Sz[0] += Ur0 * grad[2];
        Sz[1] += Ur1 * grad[2];
        Sz[2] += Ur2 * grad[2];
        Sz[3] += Ur3 * grad[2];
        Sz[4] += Ur4 * grad[2];
      }

      analytical_flux(S, gamma, Fx, Fy, Fz);

      A(S, Sx, Sy, Sz, mu, Qx, Qy, Qz);

      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          real* tfnc_grad = geom.core.vgrad(ei, ti, qi);

          R(ei, ri, ti) += ((tfnc_grad[0] * Qx[ri] + tfnc_grad[1] * Qy[ri] +
                             tfnc_grad[2] * Qz[ri]) -
                            (tfnc_grad[0] * Fx[ri] + tfnc_grad[1] * Fy[ri] +
                             tfnc_grad[2] * Fz[ri])) *
                           Jdet * quad_w;
        }
      }
    }
  }

  /* source term */

  for (u64 ei = 0; ei < nelem; ++ei)
  {
    for (u64 qi = 0; qi < geom.core.refp.vqrule.n; ++qi)
    {
      real Jdet   = geom.core.vJ(ei, qi);
      real quad_w = geom.core.refp.vqrule.wgt(qi);
      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          R(ei, ri, ti) -=
          geom.core.refp.veval_at(ti, qi) * source[ri] * Jdet * quad_w;
        }
      }
    }
  }

  /* receive send boundary information from other procs */

  for (u64 bi = 0; bi < 6; ++bi)
  {
    if (geom.core.bounds[bi] >= 0)
    {
      MPI_Recv(wsp.URst + 0, 5, geom.tfer_dtypes[bi], (int)geom.core.bounds[bi],
               opposite_local_face[bi], geom.comm, MPI_STATUS_IGNORE);
      MPI_Recv(wsp.URgrad + 0, 5 * 3, geom.tfer_dtypes[bi],
               (int)geom.core.bounds[bi], opposite_local_face[bi] + 6,
               geom.comm, MPI_STATUS_IGNORE);
    }
  }

  /* ensure communication of boundary face info has finished */
  // TODO: for some reason processor scaling is trash with this at the end of
  // the function???

  for (u64 bi = 0; bi < 6; ++bi)
  {
    if (geom.core.bounds[bi] >= 0)
    {
      MPI_Wait(&state_requests[bi], MPI_STATUS_IGNORE);
      MPI_Wait(&grad_requests[bi], MPI_STATUS_IGNORE);
    }
  }

  /* ------------------ */
  /* boundary residuals */
  /* ------------------ */

  for (u64 ifi = 0; ifi < geom.num_interior_faces(); ++ifi)
  {
    s64 fi         = geom.interior_face_list[5 * ifi + 0];
    s64 eL         = geom.interior_face_list[5 * ifi + 1];
    s64 eR         = geom.interior_face_list[5 * ifi + 2];
    s64 lfL        = geom.interior_face_list[5 * ifi + 3];
    s64 lfR        = geom.interior_face_list[5 * ifi + 4];
    bool proc_edge = (eR < 0) ? true : false;

    /* pre-calculate BR2 stabilization term */

    memset(&wsp.RL[0], 0, br2vsize * sizeof(real));
    memset(&wsp.RR[0], 0, br2vsize * sizeof(real));
    memset(&wsp.DL[0], 0, br2vsize * sizeof(real));
    memset(&wsp.DR[0], 0, br2vsize * sizeof(real));

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real SL[5], SR[5];
      real diffL[5], diffR[5];
      real DLNx[5], DLNy[5], DLNz[5];
      real DRNx[5], DRNy[5], DRNz[5];
      real QLx[5], QLy[5], QLz[5];
      real QRx[5], QRy[5], QRz[5];

      u64 dimos      = geom.core.refp.nbf3d * 5;
      real* n        = geom.n(fi, qi);
      real surf_elem = geom.fJ(fi, qi);
      real quad_w    = geom.core.refp.fqrule.wgt(qi);

      from_state_store(geom, fi, qi, wsp.ULst, SL);
      from_state_store(geom, fi, qi, wsp.URst, SR);

      for (u64 i = 0; i < 5; ++i)
      {
        diffL[i] = SL[i] - SR[i];
        diffR[i] = SR[i] - SL[i];
      }

      for (u64 i = 0; i < 5; ++i)
      {
        DLNx[i] = diffL[i] * n[0];
        DLNy[i] = diffL[i] * n[1];
        DLNz[i] = diffL[i] * n[2];

        DRNx[i] = diffR[i] * -n[0];
        DRNy[i] = diffR[i] * -n[1];
        DRNz[i] = diffR[i] * -n[2];
      }

      A(SL, DLNx, DLNy, DLNz, mu, QLx, QLy, QLz);
      A(SR, DRNx, DRNy, DRNz, mu, QRx, QRy, QRz);

      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          real tfncL = geom.core.refp.feval_at(lfL, ti, qi);
          real tfncR = geom.core.refp.feval_at(lfR, ti, qi);

          u64 tstos = geom.core.refp.nbf3d * ri + ti;

          wsp.RL[dimos * 0 + tstos] +=
          (0.5 * tfncL * QLx[ri]) * surf_elem * quad_w;
          wsp.RL[dimos * 1 + tstos] +=
          (0.5 * tfncL * QLy[ri]) * surf_elem * quad_w;
          wsp.RL[dimos * 2 + tstos] +=
          (0.5 * tfncL * QLz[ri]) * surf_elem * quad_w;

          wsp.RR[dimos * 0 + tstos] +=
          (0.5 * tfncR * QRx[ri]) * surf_elem * quad_w;
          wsp.RR[dimos * 1 + tstos] +=
          (0.5 * tfncR * QRy[ri]) * surf_elem * quad_w;
          wsp.RR[dimos * 2 + tstos] +=
          (0.5 * tfncR * QRz[ri]) * surf_elem * quad_w;
        }
      }
    }

    real* MinvL = geom.core.Minv(eL);
    real* MinvR;
    if (eR < 0)
      MinvR = geom.Minv_aux((u64)(-(eR + 1)));
    else
      MinvR = geom.core.Minv(eR);

    spmm(geom.core.refp.nbf3d, 15, MinvL, wsp.RL.data, 1., wsp.DL.data);
    spmm(geom.core.refp.nbf3d, 15, MinvR, wsp.RR.data, 1., wsp.DR.data);

    /* execute actual terms now */

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real UL[5], UR[5];
      real SLx[5], SLy[5], SLz[5];
      real SRx[5], SRy[5], SRz[5];

      real* n        = geom.n(fi, qi);
      real surf_elem = geom.fJ(fi, qi);
      real quad_w    = geom.core.refp.fqrule.wgt(qi);

      from_state_store(geom, fi, qi, wsp.ULst, UL);
      from_state_store(geom, fi, qi, wsp.URst, UR);
      from_grad_store(geom, fi, qi, wsp.ULgrad, SLx, SLy, SLz);
      from_grad_store(geom, fi, qi, wsp.URgrad, SRx, SRy, SRz);

      /* inviscid boundary term */

      real F[5];
      {
        real smax;
        roe(UL, UR, n, gamma, F, &smax);
      }

      /* dual consistency term */

      real dcQLx[5] = {}, dcQLy[5] = {}, dcQLz[5] = {};
      real dcQRx[5] = {}, dcQRy[5] = {}, dcQRz[5] = {};
      {
        real Uh[5];
        for (u64 i = 0; i < 5; ++i)
          Uh[i] = 0.5 * (UL[i] + UR[i]);

        real diffL[5];
        for (u64 i = 0; i < 5; ++i)
          diffL[i] = UL[i] - Uh[i];

        real DLNx[5], DLNy[5], DLNz[5];
        for (u64 i = 0; i < 5; ++i)
        {
          DLNx[i] = diffL[i] * n[0];
          DLNy[i] = diffL[i] * n[1];
          DLNz[i] = diffL[i] * n[2];
        }

        A(UL, DLNx, DLNy, DLNz, mu, dcQLx, dcQLy, dcQLz);

        if (!proc_edge)
        {
          real diffR[5];
          for (u64 i = 0; i < 5; ++i)
            diffR[i] = UR[i] - Uh[i];

          real DRNx[5], DRNy[5], DRNz[5];
          for (u64 i = 0; i < 5; ++i)
          {
            DRNx[i] = diffR[i] * -n[0];
            DRNy[i] = diffR[i] * -n[1];
            DRNz[i] = diffR[i] * -n[2];
          }

          A(UR, DRNx, DRNy, DRNz, mu, dcQRx, dcQRy, dcQRz);
        }
      }

      /* viscous flux term */

      real QH[5] = {};
      {
        real QLx[5], QLy[5], QLz[5];
        real QRx[5], QRy[5], QRz[5];
        real QHx[5], QHy[5], QHz[5];
        real dLx[5] = {}, dLy[5] = {}, dLz[5] = {};
        real dRx[5] = {}, dRy[5] = {}, dRz[5] = {};
        u64 dimos = geom.core.refp.nbf3d * 5;

        for (u64 ri = 0; ri < 5; ++ri)
        {
          for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
          {
            u64 tstos = geom.core.refp.nbf3d * ri + bi;

            real phiL = geom.core.refp.feval_at(lfL, bi, qi);
            real phiR = geom.core.refp.feval_at(lfR, bi, qi);

            dLx[ri] += phiL * wsp.DL[dimos * 0 + tstos];
            dLy[ri] += phiL * wsp.DL[dimos * 1 + tstos];
            dLz[ri] += phiL * wsp.DL[dimos * 2 + tstos];

            dRx[ri] += phiR * wsp.DR[dimos * 0 + tstos];
            dRy[ri] += phiR * wsp.DR[dimos * 1 + tstos];
            dRz[ri] += phiR * wsp.DR[dimos * 2 + tstos];
          }
        }

        A(UL, SLx, SLy, SLz, mu, QLx, QLy, QLz);
        A(UR, SRx, SRy, SRz, mu, QRx, QRy, QRz);

        for (u64 ri = 0; ri < 5; ++ri)
        {
          QHx[ri] = 0.5 * (QLx[ri] + QRx[ri]) - eta * 0.5 * (dLx[ri] + dRx[ri]);
          QHy[ri] = 0.5 * (QLy[ri] + QRy[ri]) - eta * 0.5 * (dLy[ri] + dRy[ri]);
          QHz[ri] = 0.5 * (QLz[ri] + QRz[ri]) - eta * 0.5 * (dLz[ri] + dRz[ri]);
        }

        for (u64 ri = 0; ri < 5; ++ri)
          QH[ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
      }

      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          real tfncL     = geom.core.refp.feval_at(lfL, ti, qi);
          real* tf_gradL = geom.core.fgrad(eL, lfL, ti, qi);

          R(eL, ri, ti) += ((tfncL * F[ri]) -
                            (tf_gradL[0] * dcQLx[ri] + tf_gradL[1] * dcQLy[ri] +
                             tf_gradL[2] * dcQLz[ri]) -
                            (tfncL * QH[ri])) *
                           surf_elem * quad_w;
        }
      }

      if (!proc_edge)
      {
        for (u64 ri = 0; ri < 5; ++ri)
        {
          for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
          {
            real tfncR     = geom.core.refp.feval_at(lfR, ti, qi);
            real* tf_gradR = geom.core.fgrad(eR, lfR, ti, qi);

            R(eR, ri, ti) +=
            (-(tfncR * F[ri]) -
             (tf_gradR[0] * dcQRx[ri] + tf_gradR[1] * dcQRy[ri] +
              tf_gradR[2] * dcQRz[ri]) +
             (tfncR * QH[ri])) *
            surf_elem * quad_w;
          }
        }
      }
    }
  }

  for (u64 bfi = 0; bfi < geom.num_boundary_faces(); ++bfi)
  {
    s64 fi  = geom.boundary_face_list[4 * bfi + 0];
    s64 eL  = geom.boundary_face_list[4 * bfi + 1];
    s64 bt  = geom.boundary_face_list[4 * bfi + 2];
    s64 lfL = geom.boundary_face_list[4 * bfi + 3];

    /* pre-calculate BR2 stabilization term */

    memset(&wsp.RL[0], 0, br2vsize * sizeof(real));
    memset(&wsp.RR[0], 0, br2vsize * sizeof(real));
    memset(&wsp.DL[0], 0, br2vsize * sizeof(real));
    memset(&wsp.DR[0], 0, br2vsize * sizeof(real));

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real SL[5], Sb[5];
      real diffL[5];
      real DLNx[5], DLNy[5], DLNz[5];
      real QLx[5], QLy[5], QLz[5];

      real* n        = geom.n(fi, qi);
      real surf_elem = geom.fJ(fi, qi);
      real quad_w    = geom.core.refp.fqrule.wgt(qi);
      u64 dimos      = geom.core.refp.nbf3d * 5;

      from_state_store(geom, fi, qi, wsp.ULst, SL);
      from_state_store(geom, fi, qi, wsp.Ubst, Sb);

      for (u64 i = 0; i < 5; ++i)
        diffL[i] = SL[i] - Sb[i];

      for (u64 i = 0; i < 5; ++i)
      {
        DLNx[i] = diffL[i] * n[0];
        DLNy[i] = diffL[i] * n[1];
        DLNz[i] = diffL[i] * n[2];
      }

      A(Sb, DLNx, DLNy, DLNz, mu, QLx, QLy, QLz);

      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          real tfncL = geom.core.refp.feval_at(lfL, ti, qi);

          u64 tstos = geom.core.refp.nbf3d * ri + ti;

          wsp.RL[dimos * 0 + tstos] += (tfncL * QLx[ri]) * surf_elem * quad_w;
          wsp.RL[dimos * 1 + tstos] += (tfncL * QLy[ri]) * surf_elem * quad_w;
          wsp.RL[dimos * 2 + tstos] += (tfncL * QLz[ri]) * surf_elem * quad_w;
        }
      }
    }

    for (u64 entry = 0; entry < 15; ++entry)
    {
      symmstore_mvmul(geom.core.refp.nbf3d, geom.core.Minv(eL),
                      wsp.RL + geom.core.refp.nbf3d * entry, 1.,
                      wsp.DL + geom.core.refp.nbf3d * entry);
    }

    /* actual residual terms now */

    for (u64 qi = 0; qi < geom.core.refp.fqrule.n; ++qi)
    {
      real UL[5], Ub[5];
      real SLx[5], SLy[5], SLz[5];

      real* n        = geom.n(fi, qi);
      real surf_elem = geom.fJ(fi, qi);
      real quad_w    = geom.core.refp.fqrule.wgt(qi);

      from_state_store(geom, fi, qi, wsp.ULst, UL);
      from_state_store(geom, fi, qi, wsp.Ubst, Ub);
      from_grad_store(geom, fi, qi, wsp.ULgrad, SLx, SLy, SLz);

      /* inviscid boundary term */

      real F[5] = {};
      {
        real Fx[5], Fy[5], Fz[5];
        analytical_flux(Ub, gamma, Fx, Fy, Fz);

        F[0] = Fx[0] * n[0] + Fy[0] * n[1] + Fz[0] * n[2];
        F[1] = Fx[1] * n[0] + Fy[1] * n[1] + Fz[1] * n[2];
        F[2] = Fx[2] * n[0] + Fy[2] * n[1] + Fz[2] * n[2];
        F[3] = Fx[3] * n[0] + Fy[3] * n[1] + Fz[3] * n[2];
        F[4] = Fx[4] * n[0] + Fy[4] * n[1] + Fz[4] * n[2];
      }

      /* dual consistency term */

      real dcQLx[5], dcQLy[5], dcQLz[5];
      {
        real Uh[5];
        for (u64 i = 0; i < 5; ++i)
          Uh[i] = Ub[i];

        real diffL[5];
        for (u64 i = 0; i < 5; ++i)
          diffL[i] = UL[i] - Uh[i];

        real DLNx[5], DLNy[5], DLNz[5];
        for (u64 i = 0; i < 5; ++i)
        {
          DLNx[i] = diffL[i] * n[0];
          DLNy[i] = diffL[i] * n[1];
          DLNz[i] = diffL[i] * n[2];
        }

        A(UL, DLNx, DLNy, DLNz, mu, dcQLx, dcQLy, dcQLz);
      }

      /* viscous flux term */

      real QH[5] = {};
      {
        real QLx[5], QLy[5], QLz[5];
        real QHx[5] = {}, QHy[5] = {}, QHz[5] = {};
        real dLx[5] = {}, dLy[5] = {}, dLz[5] = {};
        u64 dimos = geom.core.refp.nbf3d * 5;

        for (u64 ri = 0; ri < 5; ++ri)
        {
          for (u64 bi = 0; bi < geom.core.refp.nbf3d; ++bi)
          {
            u64 tstos = geom.core.refp.nbf3d * ri + bi;

            real phiL = geom.core.refp.feval_at(lfL, bi, qi);

            dLx[ri] += phiL * wsp.DL[dimos * 0 + tstos];
            dLy[ri] += phiL * wsp.DL[dimos * 1 + tstos];
            dLz[ri] += phiL * wsp.DL[dimos * 2 + tstos];
          }
        }

        A(UL, SLx, SLy, SLz, mu, QLx, QLy, QLz);

        for (u64 ri = 0; ri < 5; ++ri)
        {
          QHx[ri] = QLx[ri] - eta * dLx[ri];
          QHy[ri] = QLy[ri] - eta * dLy[ri];
          QHz[ri] = QLz[ri] - eta * dLz[ri];
        }

        for (u64 ri = 0; ri < 5; ++ri)
          QH[ri] = QHx[ri] * n[0] + QHy[ri] * n[1] + QHz[ri] * n[2];
      }

      for (u64 ri = 0; ri < 5; ++ri)
      {
        for (u64 ti = 0; ti < geom.core.refp.nbf3d; ++ti)
        {
          real tfncL     = geom.core.refp.feval_at(lfL, ti, qi);
          real* tf_gradL = geom.core.fgrad(eL, lfL, ti, qi);

          R(eL, ri, ti) += ((tfncL * F[ri]) -
                            (tf_gradL[0] * dcQLx[ri] + tf_gradL[1] * dcQLy[ri] +
                             tf_gradL[2] * dcQLz[ri]) -
                            (tfncL * QH[ri])) *
                           surf_elem * quad_w;
        }
      }
    }
  }

  /* ------------------- */
  /* include mass matrix */
  /* ------------------- */

  for (u64 ei = 0; ei < nelem; ++ei)
  {
    spmm(geom.core.refp.nbf3d, 5, geom.core.Minv(ei), &R(ei, 0, 0), -1,
         &f(ei, 0, 0));
  }

  MPI_Barrier(geom.comm);
}
