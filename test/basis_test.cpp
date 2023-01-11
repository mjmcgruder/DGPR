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

#include <mpitest.h>

#include "basis.cpp"

struct lagrange_funcs
{
  // some points to check
  real pnt0 = 0.223;
  real pnt1 = 0.437;
  real pnt2 = 0.945;
  v2 pnt2d  = {pnt0, pnt1};
  v3 pnt3d  = {pnt0, pnt1, pnt2};

  // some hard coded 1d functions
  real phi1d_p0_0(real x)
  {
    return 1.0;
  }
  real phi1d_p1_0(real x)
  {
    return 1.0 - x;
  }
  real phi1d_p1_1(real x)
  {
    return x;
  }
  real phi1d_p2_0(real x)
  {
    return 2.0 * x * x - 3.0 * x + 1.0;
  }
  real phi1d_p2_1(real x)
  {
    return -4.0 * x * x + 4.0 * x;
  }
  real phi1d_p2_2(real x)
  {
    return 2.0 * x * x - x;
  }
  real dphi1d_p0_0(real x)
  {
    return 0.0;
  }
  real dphi1d_p1_0(real x)
  {
    return -1.0;
  }
  real dphi1d_p1_1(real x)
  {
    return 1.0;
  }
  real dphi1d_p2_0(real x)
  {
    return 4.0 * x - 3.0;
  }
  real dphi1d_p2_1(real x)
  {
    return -8.0 * x + 4.0;
  }
  real dphi1d_p2_2(real x)
  {
    return 4.0 * x - 1.0;
  }

  // some hard coded 3d functions
  real phi3d_p2_0(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_0(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_1(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_0(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_2(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_0(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_3(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_1(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_4(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_1(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_5(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_1(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_6(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_2(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_7(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_2(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_8(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_2(y) * phi1d_p2_0(z);
  }
  real phi3d_p2_9(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_0(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_10(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_0(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_11(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_0(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_12(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_1(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_13(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_1(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_14(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_1(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_15(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_2(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_16(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_2(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_17(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_2(y) * phi1d_p2_1(z);
  }
  real phi3d_p2_18(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_0(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_19(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_0(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_20(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_0(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_21(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_1(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_22(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_1(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_23(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_1(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_24(real x, real y, real z)
  {
    return phi1d_p2_0(x) * phi1d_p2_2(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_25(real x, real y, real z)
  {
    return phi1d_p2_1(x) * phi1d_p2_2(y) * phi1d_p2_2(z);
  }
  real phi3d_p2_26(real x, real y, real z)
  {
    return phi1d_p2_2(x) * phi1d_p2_2(y) * phi1d_p2_2(z);
  }
};

TEST(lagrange_funcs_1d_p0, 1)
{
  lagrange_funcs data;

  // lagrange property (doesn't much apply here really...)
  EXPECT_DOUBLE_EQ(1., lagrange1d(0, 0, 0.), 10);

  // point test
  EXPECT_DOUBLE_EQ(1., lagrange1d(0, 0, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(1., lagrange1d(0, 0, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(1., lagrange1d(0, 0, data.pnt2), 10);
}

TEST(lagrange_funcs_1d_p1, 1)
{
  lagrange_funcs data;

  // lagrange property
  EXPECT_DOUBLE_EQ(1., lagrange1d(0, 1, 0.0), 10);
  EXPECT_DOUBLE_EQ(0., lagrange1d(0, 1, 1.0), 10);
  EXPECT_DOUBLE_EQ(0., lagrange1d(1, 1, 0.0), 10);
  EXPECT_DOUBLE_EQ(1., lagrange1d(1, 1, 1.0), 10);

  // point test
  EXPECT_DOUBLE_EQ(data.phi1d_p1_0(data.pnt0), lagrange1d(0, 1, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p1_0(data.pnt1), lagrange1d(0, 1, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p1_0(data.pnt2), lagrange1d(0, 1, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p1_1(data.pnt0), lagrange1d(1, 1, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p1_1(data.pnt1), lagrange1d(1, 1, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p1_1(data.pnt2), lagrange1d(1, 1, data.pnt2), 10);
}

TEST(lagrange_funcs_1d_p2, 1)
{
  lagrange_funcs data;

  // lagrange property
  EXPECT_DOUBLE_EQ(1.0, lagrange1d(0, 2, 0.0), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(0, 2, 0.5), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(0, 2, 1.0), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(1, 2, 0.0), 10);
  EXPECT_DOUBLE_EQ(1.0, lagrange1d(1, 2, 0.5), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(1, 2, 1.0), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(2, 2, 0.0), 10);
  EXPECT_DOUBLE_EQ(0.0, lagrange1d(2, 2, 0.5), 10);
  EXPECT_DOUBLE_EQ(1.0, lagrange1d(2, 2, 1.0), 10);

  // test the points
  EXPECT_DOUBLE_EQ(data.phi1d_p2_0(data.pnt0), lagrange1d(0, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_0(data.pnt1), lagrange1d(0, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_0(data.pnt2), lagrange1d(0, 2, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_1(data.pnt0), lagrange1d(1, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_1(data.pnt1), lagrange1d(1, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_1(data.pnt2), lagrange1d(1, 2, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_2(data.pnt0), lagrange1d(2, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_2(data.pnt1), lagrange1d(2, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(data.phi1d_p2_2(data.pnt2), lagrange1d(2, 2, data.pnt2), 10);
}

TEST(lagrange_funcs_1d_p0_deriv, 1)
{
  lagrange_funcs data;

  // point test
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p0_0(data.pnt0), lagrange1d_deriv(0, 0, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p0_0(data.pnt1), lagrange1d_deriv(0, 0, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p0_0(data.pnt2), lagrange1d_deriv(0, 0, data.pnt2), 10);
}

TEST(lagrange_funcs_1d_p1_deriv, 1)
{
  lagrange_funcs data;

  // point test
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_0(data.pnt0), lagrange1d_deriv(0, 1, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_0(data.pnt1), lagrange1d_deriv(0, 1, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_0(data.pnt2), lagrange1d_deriv(0, 1, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_1(data.pnt0), lagrange1d_deriv(1, 1, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_1(data.pnt1), lagrange1d_deriv(1, 1, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p1_1(data.pnt2), lagrange1d_deriv(1, 1, data.pnt2), 10);
}

TEST(lagrange_funcs_1d_p2_deriv, 1)
{
  lagrange_funcs data;

  // point test
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_0(data.pnt0), lagrange1d_deriv(0, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_0(data.pnt1), lagrange1d_deriv(0, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_0(data.pnt2), lagrange1d_deriv(0, 2, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_1(data.pnt0), lagrange1d_deriv(1, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_1(data.pnt1), lagrange1d_deriv(1, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_1(data.pnt2), lagrange1d_deriv(1, 2, data.pnt2), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_2(data.pnt0), lagrange1d_deriv(2, 2, data.pnt0), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_2(data.pnt1), lagrange1d_deriv(2, 2, data.pnt1), 10);
  EXPECT_DOUBLE_EQ(
  data.dphi1d_p2_2(data.pnt2), lagrange1d_deriv(2, 2, data.pnt2), 10);
}

TEST(lagrange_funcs_3d_p2, 1)
{
  lagrange_funcs data;

  // point test
  EXPECT_DOUBLE_EQ(data.phi3d_p2_0(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(0, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_1(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(1, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_2(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(2, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_3(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(3, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_4(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(4, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_5(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(5, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_6(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(6, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_7(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(7, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_8(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(8, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_9(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(9, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_10(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(10, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_11(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(11, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_12(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(12, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_13(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(13, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_14(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(14, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_15(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(15, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_16(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(16, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_17(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(17, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_18(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(18, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_19(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(19, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_20(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(20, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_21(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(21, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_22(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(22, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_23(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(23, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_24(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(24, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_25(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(25, 2, data.pnt3d),
                   25);
  EXPECT_DOUBLE_EQ(data.phi3d_p2_26(data.pnt0, data.pnt1, data.pnt2),
                   lagrange3d(26, 2, data.pnt3d),
                   25);
}

struct pre_computes
{
  void chkeval(basis& pn)
  {
    for (u64 iq = 0; iq < pn.vqrule.n; ++iq)
    {
      real acc = 0.0;
      for (u64 ib = 0; ib < pn.nbf3d; ++ib)
      {
        acc += pn.veval_at(ib, iq);
      }
      EXPECT_DOUBLE_EQ(1., acc, 30);
    }
  }

  void chkfeval(basis& pn)
  {
    for (u64 lfi = 0; lfi < 6; ++lfi)
    {
      for (u64 iq = 0; iq < pn.fqrule.n; ++iq)
      {
        real acc = 0.0;
        for (u64 ib = 0; ib < pn.nbf3d; ++ib)
        {
          acc += pn.feval_at(lfi, ib, iq);
        }
        EXPECT_DOUBLE_EQ(1., acc, 10);
      }
    }
  }

  void chkgrad(basis& pn)
  {
    for (u64 iq = 0; iq < pn.vqrule.n; ++iq)
    {
      real accx = 0.0;
      real accy = 0.0;
      real accz = 0.0;
      for (u64 ib = 0; ib < pn.nbf3d; ++ib)
      {
        accx += pn.vgrad_at(ib, iq, component::x);
        accy += pn.vgrad_at(ib, iq, component::y);
        accz += pn.vgrad_at(ib, iq, component::z);
      }
      EXPECT_DOUBLE_EQ(0., accx, 10, 5e-14);
      EXPECT_DOUBLE_EQ(0., accy, 10, 5e-14);
      EXPECT_DOUBLE_EQ(0., accz, 10, 5e-14);
    }
  }
};

TEST(basis_pre_computes_p0, 1)
{
  pre_computes data;

  basis p0{0, 0};
  data.chkeval(p0);
  data.chkfeval(p0);
  data.chkgrad(p0);
}

TEST(basis_pre_computes_p2, 1)
{
  pre_computes data;

  basis p1{2, 2};
  data.chkeval(p1);
  data.chkfeval(p1);
  data.chkgrad(p1);
}

TEST(basis_pre_computes_p7, 1)
{
  pre_computes data;

  basis p7{7, 7};
  data.chkeval(p7);
  data.chkfeval(p7);
  data.chkgrad(p7);
}
