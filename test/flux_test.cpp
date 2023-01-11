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

#include <mpitest.h>

#include "flux.cpp"

struct roe_3d
{
  // constants
  real g    = 1.4;
  real Minf = 2.2;
  real c    = 1.0;

  // angles
  real phl = 5.0 * ((real)M_PI / 180.0);   // left velocity vector
  real thl = 85.0 * ((real)M_PI / 180.0);  //
  real phr = 3.0 * ((real)M_PI / 180.0);   // right velocity vector
  real thr = 87.0 * ((real)M_PI / 180.0);  //
  real phn = 2.0 * ((real)M_PI / 180.0);   // boundary normal vector
  real thn = 83.0 * ((real)M_PI / 180.0);  //

  // left primitives
  real Ml = Minf + 0.01;
  real rl = 1.02;
  real ul = Ml * cosf(phl) * sinf(thl);
  real vl = Ml * sinf(phl) * sinf(thl);
  real wl = Ml * cosf(thl);

  // right primitives
  real Mr = Minf - 0.02;
  real rr = 1.04;
  real ur = Mr * cosf(phr) * sinf(thr);
  real vr = Mr * sinf(phr) * sinf(thr);
  real wr = Mr * cosf(thr);

  // energy is set such that speed of sound is as listed above and the velocity
  // is simply the mach number of the flow as above

  // left state
  real U0l = rl;
  real U1l = rl * ul;
  real U2l = rl * vl;
  real U3l = rl * wl;
  real U4l = (rl * c * c) / (g * (g - 1.0)) + 0.5 * rl * Ml * Ml;
  real pl  = (g - 1.0) * (U4l - 0.5 * rl * (ul * ul + vl * vl + wl * wl));
  real Hl  = (U4l / U0l) + (pl / rl);

  // right state
  real U0r = rr;
  real U1r = rr * ur;
  real U2r = rr * vr;
  real U3r = rr * wr;
  real U4r = (rr * c * c) / (g * (g - 1.0)) + 0.5 * rr * Mr * Mr;

  // final arrays
  real Ul[5] = {U0l, U1l, U2l, U3l, U4l};
  real Ur[5] = {U0r, U1r, U2r, U3r, U4r};
  real n[3]  = {cosf(phn) * sinf(thn), sinf(phn) * sinf(thn), cosf(thn)};
  real nn[3] = {-n[0], -n[1], -n[2]};
  real smax;

  // left (upwind) analytical fluxes
  real f0 = Ul[1] * n[0] + Ul[2] * n[1] + Ul[3] * n[2];
  real f1 =
  (rl * ul * ul + pl) * n[0] + (rl * ul * vl) * n[1] + (rl * ul * wl) * n[2];
  real f2 =
  (rl * ul * vl) * n[0] + (rl * vl * vl + pl) * n[1] + (rl * vl * wl) * n[2];
  real f3 =
  (rl * ul * wl) * n[0] + (rl * vl * wl) * n[1] + (rl * wl * wl + pl) * n[2];
  real f4 =
  ((rl * ul * Hl) * n[0] + (rl * vl * Hl) * n[1] + (rl * wl * Hl) * n[2]);
};

TEST(roe_3d_analytical, 1)
{
  roe_3d data;
  real f[5];

  roe(data.Ul, data.Ul, data.n, data.g, f, &data.smax);

  ASSERT_DOUBLE_EQ(f[0], data.f0, 5);
  ASSERT_DOUBLE_EQ(f[1], data.f1, 5);
  ASSERT_DOUBLE_EQ(f[2], data.f2, 5);
  ASSERT_DOUBLE_EQ(f[3], data.f3, 5);
  ASSERT_DOUBLE_EQ(f[4], data.f4, 5);
}

TEST(roe_3d_flip, 1)
{
  roe_3d data;
  real ff[5];  // flux forward
  real fr[5];  // flux reverse

  roe(data.Ul, data.Ur, data.n, data.g, ff, &data.smax);
  roe(data.Ur, data.Ul, data.nn, data.g, fr, &data.smax);

  ASSERT_DOUBLE_EQ(ff[0], -fr[0], 5);
  ASSERT_DOUBLE_EQ(ff[1], -fr[1], 5);
  ASSERT_DOUBLE_EQ(ff[2], -fr[2], 5);
  ASSERT_DOUBLE_EQ(ff[3], -fr[3], 5);
  ASSERT_DOUBLE_EQ(ff[4], -fr[4], 5);
}

TEST(roe_3d_supersonic_upwind, 1)
{
  roe_3d data;
  real f[5];

  roe(data.Ul, data.Ur, data.n, data.g, f, &data.smax);

  ASSERT_DOUBLE_EQ(f[0], data.f0, 5);
  ASSERT_DOUBLE_EQ(f[1], data.f1, 5);
  ASSERT_DOUBLE_EQ(f[2], data.f2, 5);
  ASSERT_DOUBLE_EQ(f[3], data.f3, 5);
  ASSERT_DOUBLE_EQ(f[4], data.f4, 5);
}
