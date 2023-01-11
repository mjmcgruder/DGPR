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

#include <cmath>

#include "compilation_config.cpp"
#include "error.cpp"

DECLSPEC void analytical_flux(real* U, real gamma, real* Fx, real* Fy,
                              real* Fz)
{
  real rho = U[0];
  real u   = U[1] / rho;
  real v   = U[2] / rho;
  real w   = U[3] / rho;
  real p   = (gamma - 1.) * (U[4] - 0.5 * rho * (u * u + v * v + w * w));
  real H   = (U[4] + p) / rho;
  Fx[0]    = rho * u;
  Fx[1]    = rho * u * u + p;
  Fx[2]    = rho * u * v;
  Fx[3]    = rho * u * w;
  Fx[4]    = rho * u * H;
  Fy[0]    = rho * v;
  Fy[1]    = rho * u * v;
  Fy[2]    = rho * v * v + p;
  Fy[3]    = rho * v * w;
  Fy[4]    = rho * v * H;
  Fz[0]    = rho * w;
  Fz[1]    = rho * u * w;
  Fz[2]    = rho * v * w;
  Fz[3]    = rho * w * w + p;
  Fz[4]    = rho * w * H;
}

// 2020 523 notes locations:
// 135 - euler equations
// 148 - roe flux
// Toro bible notes locations:
// 124 - 3d euler equations

DECLSPEC void roe(real* Ul, real* Ur, real* n, real g, real* f, real* smax)
{
  real gm1 = g - 1.0;
  // left primitive quantities
  real rl = Ul[0];
  real ul = Ul[1] / rl;
  real vl = Ul[2] / rl;
  real wl = Ul[3] / rl;
  real El = Ul[4] / rl;
  real pl = gm1 * (Ul[4] - 0.5 * rl * (ul * ul + vl * vl + wl * wl));
  real Hl = El + (pl / rl);
  // right primitive quantities
  real rr = Ur[0];
  real ur = Ur[1] / rr;
  real vr = Ur[2] / rr;
  real wr = Ur[3] / rr;
  real Er = Ur[4] / rr;
  real pr = gm1 * (Ur[4] - 0.5 * rr * (ur * ur + vr * vr + wr * wr));
  real Hr = Er + (pr / rr);
  // expensive(?) pre-computes
  real sqrt_rl = sqrt(rl);
  real sqrt_rr = sqrt(rr);
  // roe state
  real state_denom = sqrt_rl + sqrt_rr;
  real u           = (sqrt_rl * ul + sqrt_rr * ur) / state_denom;
  real v           = (sqrt_rl * vl + sqrt_rr * vr) / state_denom;
  real w           = (sqrt_rl * wl + sqrt_rr * wr) / state_denom;
  real H           = (sqrt_rl * Hl + sqrt_rr * Hr) / state_denom;
  // r - l state differences
  real dr  = rr - rl;
  real dxm = Ur[1] - Ul[1];
  real dym = Ur[2] - Ul[2];
  real dzm = Ur[3] - Ul[3];
  real drE = Ur[4] - Ul[4];
  // (abs)eigenvalues (with entropy fix) (you can probably kill the branch here)
  real qsq = u * u + v * v + w * w;
  real c   = sqrt(gm1 * (H - 0.5 * qsq));
  real un  = u * n[0] + v * n[1] + w * n[2];  // roe state bound normal speed
  real l1  = un + c;
  real l2  = un - c;
  real l3  = un;
  real al1 = fabs(l1);
  real al2 = fabs(l2);
  real al3 = fabs(l3);
  real eps = 0.05 * c;
  if (al1 < eps)
  {
    al1 = fabs((eps * eps + l1 * l1) / (2.0 * eps));
  }
  if (al2 < eps)
  {
    al2 = fabs((eps * eps + l2 * l2) / (2.0 * eps));
  }
  if (al3 < eps)
  {
    al3 = fabs((eps * eps + l3 * l3) / (2.0 * eps));
  }
  // itermediates
  real s1 = 0.5 * (al1 + al2);
  real s2 = 0.5 * (al1 - al2);
  real G1 = gm1 * (0.5 * qsq * dr - (u * dxm + v * dym + w * dzm) + drE);
  real G2 = -un * dr + (dxm * n[0] + dym * n[1] + dzm * n[2]);
  real C1 = (G1 / (c * c)) * (s1 - al3) + (G2 / c) * s2;
  real C2 = (G1 / c) * s2 + (s1 - al3) * G2;
  // assign fluxes (you can remove some multiplication here)
  f[0] = 0.5 * ((Ul[1] * n[0] + Ul[2] * n[1] + Ul[3] * n[2]) +
                (Ur[1] * n[0] + Ur[2] * n[1] + Ur[3] * n[2])) -
         0.5 * (al3 * dr + C1);
  f[1] = 0.5 * (((rl * ul * ul + pl) * n[0] + (rl * ul * vl) * n[1] +
                 (rl * ul * wl) * n[2]) +
                ((rr * ur * ur + pr) * n[0] + (rr * ur * vr) * n[1] +
                 (rr * ur * wr) * n[2])) -
         0.5 * (al3 * dxm + C1 * u + C2 * n[0]);
  f[2] = 0.5 * (((rl * ul * vl) * n[0] + (rl * vl * vl + pl) * n[1] +
                 (rl * vl * wl) * n[2]) +
                ((rr * ur * vr) * n[0] + (rr * vr * vr + pr) * n[1] +
                 (rr * vr * wr) * n[2])) -
         0.5 * (al3 * dym + C1 * v + C2 * n[1]);
  f[3] = 0.5 * (((rl * ul * wl) * n[0] + (rl * vl * wl) * n[1] +
                 (rl * wl * wl + pl) * n[2]) +
                ((rr * ur * wr) * n[0] + (rr * vr * wr) * n[1] +
                 (rr * wr * wr + pr) * n[2])) -
         0.5 * (al3 * dzm + C1 * w + C2 * n[2]);
  f[4] =
  0.5 *
  (((rl * ul * Hl) * n[0] + (rl * vl * Hl) * n[1] + (rl * wl * Hl) * n[2]) +
   ((rr * ur * Hr) * n[0] + (rr * vr * Hr) * n[1] + (rr * wr * Hr) * n[2])) -
  0.5 * (al3 * drE + C1 * H + C2 * un);
  // assign max wave speed (get more clever?)
  *smax = al1;
  if (al2 > *smax)
  {
    *smax = al2;
  }
  if (al3 > *smax)
  {
    *smax = al3;
  }
}

DECLSPEC void A(real* s, real* sx, real* sy, real* sz, real mu, real* Qx,
                real* Qy, real* Qz)
{
  real lam = -(2. / 3.) * mu;

  real rho = s[0];
  real u   = s[1] / rho;
  real v   = s[2] / rho;
  real w   = s[3] / rho;

  real ux = (sx[1] - sx[0] * u) / rho;
  real uy = (sy[1] - sy[0] * u) / rho;
  real uz = (sz[1] - sz[0] * u) / rho;
  real vx = (sx[2] - sx[0] * v) / rho;
  real vy = (sy[2] - sy[0] * v) / rho;
  real vz = (sz[2] - sz[0] * v) / rho;
  real wx = (sx[3] - sx[0] * w) / rho;
  real wy = (sy[3] - sy[0] * w) / rho;
  real wz = (sz[3] - sz[0] * w) / rho;

  real dv = ux + vy + wz;

  real txx = lam * dv + 2. * mu * ux;
  real tyy = lam * dv + 2. * mu * vy;
  real tzz = lam * dv + 2. * mu * wz;
  real txy = mu * (vx + uy);
  real txz = mu * (uz + wx);
  real tyz = mu * (wy + vz);

  Qx[0] = 0.;
  Qx[1] = txx;
  Qx[2] = txy;
  Qx[3] = txz;
  Qx[4] = u * txx + v * txy + w * txz;

  Qy[0] = 0.;
  Qy[1] = txy;
  Qy[2] = tyy;
  Qy[3] = tyz;
  Qy[4] = u * txy + v * tyy + w * tyz;

  Qz[0] = 0.;
  Qz[1] = txz;
  Qz[2] = tyz;
  Qz[3] = tzz;
  Qz[4] = u * txz + v * tyz + w * tzz;
}
