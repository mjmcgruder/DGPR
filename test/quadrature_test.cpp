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

#include "quadrature.cpp"

// 1d test functions

real ord2_1d(real x)
{
  return 1.2785 * x * x + 4.382 * x + 0.3854;
}

real ord5_1d(real x)
{
  return 2.394 * x * x * x * x * x + 8.371 * x * x * x + 0.23444 * x + 3.223;
}

real ord16_1d(real x)
{
  return 5.76594 * pow(x, 16.0) + 3.7694 * pow(x, 9.0) + 1.7674 * pow(x, 2.0) +
         4.11854;
}

real ord32_1d(real x)
{
  return 7.584 * pow(x, 32.0) + 2.69 * pow(x, 17.0) + 10.9473;
}

// 2d test functions

real ord2_2d(real x, real y)
{
  return 1.2785 * x * x + 4.1854 * y * y + 9.43 * x * y + 4.382 * x +
         2.333 * y + 0.3854;
}

real ord16_2d(real x, real y)
{
  return 5.76594 * pow(x, 16.0) - 0.5542 * pow(y, 16.0) +
         4.2833 * pow(x, 10.0) * pow(y, 8.0) +
         3.7694 * pow(x, 9.0) * pow(y, 3.0) + 1.7674 * pow(x, 2.0) + 4.11854;
}

real ord32_2d(real x, real y)
{
  return 10.3874 * pow(x, 32.0) * pow(y, 32.0) +
         3.452 * pow(x, 4.0) * pow(y, 20.0) + 3.6654;
}

// 3d test functions

real ord2_3d(real x, real y, real z)
{
  return 5.2038 * pow(x, 2.0) * pow(y, 2.0) * pow(z, 2.0) - 2.3385 * y * z +
         1.224;
}
real ord16_3d(real x, real y, real z)
{
  return 5.76594 * pow(x, 16.0) - 0.5542 * pow(y, 16.0) * pow(z, 16.0) +
         4.2833 * pow(x, 10.0) * pow(y, 8.0) * pow(z, 6.0) +
         3.7694 * pow(x, 9.0) * pow(y, 3.0) +
         1.7674 * pow(x, 2.0) * pow(z, 2.0) + 4.11854;
}

// 1d integrator

real int1d(quad rule, real (*func)(real))
{
  real integral = 0.0;
  for (u64 i = 0; i < rule.nx; ++i)
  {
    // ensure weight accessors are equivalent
    real w = rule.w[i];
    EXPECT_EQ(w, rule.wgt(i));
    EXPECT_EQ(w, rule.wgt(i, 0, 0));
    // ensure position accessors are equivalent
    real x = rule.x[i];
    EXPECT_EQ(x, rule.pos(i)[0]);
    EXPECT_EQ(x, rule.pos(i, 0, 0)[0]);
    EXPECT_EQ(x, rule.pos(i, component::x));
    EXPECT_EQ(x, rule.pos(i, 0, 0, component::x));
    // ok do some quadrature now...
    integral += w * func(x);
  }
  return integral;
}

// 2d integrator

real int2d(quad rule, real (*func)(real, real))
{
  real integral = 0.0;
  for (u64 j = 0; j < rule.ny; ++j)
  {
    for (u64 i = 0; i < rule.nx; ++i)
    {
      u64 nn = j * rule.nx + i;
      // ensure weight accessors are equivalent
      real w = rule.w[nn];
      EXPECT_EQ(w, rule.wgt(nn));
      EXPECT_EQ(w, rule.wgt(i, j, 0));
      // ensure position accessors are equivalent
      real x = rule.x[2 * nn];
      real y = rule.x[2 * nn + 1];
      EXPECT_EQ(x, rule.pos(nn)[0]);
      EXPECT_EQ(x, rule.pos(i, j, 0)[0]);
      EXPECT_EQ(x, rule.pos(nn, component::x));
      EXPECT_EQ(x, rule.pos(i, j, 0, component::x));
      EXPECT_EQ(y, rule.pos(nn)[1]);
      EXPECT_EQ(y, rule.pos(i, j, 0)[1]);
      EXPECT_EQ(y, rule.pos(nn, component::y));
      EXPECT_EQ(y, rule.pos(i, j, 0, component::y));
      // ok do some quadrature now...
      integral += w * func(x, y);
    }
  }
  return integral;
}

// 3d integrator

real int3d(quad rule, real (*func)(real, real, real))
{
  real integral = 0.0;
  for (u64 k = 0; k < rule.nz; ++k)
  {
    for (u64 j = 0; j < rule.ny; ++j)
    {
      for (u64 i = 0; i < rule.nx; ++i)
      {
        u64 nn = rule.nx * rule.ny * k + rule.nx * j + i;
        // ensure weight accessors are equivalent
        real w = rule.w[nn];
        EXPECT_EQ(w, rule.wgt(nn));
        EXPECT_EQ(w, rule.wgt(i, j, k));
        // ensure position accessors are equivalent
        real x = rule.x[3 * nn];
        real y = rule.x[3 * nn + 1];
        real z = rule.x[3 * nn + 2];
        EXPECT_EQ(x, rule.pos(nn)[0]);
        EXPECT_EQ(x, rule.pos(i, j, k)[0]);
        EXPECT_EQ(x, rule.pos(nn, component::x));
        EXPECT_EQ(x, rule.pos(i, j, k, component::x));
        EXPECT_EQ(y, rule.pos(nn)[1]);
        EXPECT_EQ(y, rule.pos(i, j, k)[1]);
        EXPECT_EQ(y, rule.pos(nn, component::y));
        EXPECT_EQ(y, rule.pos(i, j, k, component::y));
        EXPECT_EQ(z, rule.pos(nn)[2]);
        EXPECT_EQ(z, rule.pos(i, j, k)[2]);
        EXPECT_EQ(z, rule.pos(nn, component::z));
        EXPECT_EQ(z, rule.pos(i, j, k, component::z));
        // ok do some quadrature now...
        integral += w * func(x, y, z);
      }
    }
  }
  return integral;
}

TEST(gauss_legendre_1d, 1)
{
  quad rule;

  rule = gauss_legendre_1d(2, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(1.62313333333333, int1d(rule, ord2_1d), 20);

  rule = gauss_legendre_1d(2, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(3.00256666666667, int1d(rule, ord2_1d), 20);

  rule = gauss_legendre_1d(5, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(6.44600000000000, int1d(rule, ord5_1d), 20);

  rule = gauss_legendre_1d(5, 0.68834, 1.0);
  EXPECT_DOUBLE_EQ(3.04565071481875, int1d(rule, ord5_1d), 20);

  rule = gauss_legendre_1d(16, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(10.0936925490196, int1d(rule, ord16_1d), 20);

  rule = gauss_legendre_1d(16, -0.22954, 1.0);
  EXPECT_DOUBLE_EQ(6.37628085636999, int1d(rule, ord16_1d), 20);

  rule = gauss_legendre_1d(32, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(22.3542363636364, int1d(rule, ord32_1d), 20);

  rule = gauss_legendre_1d(32, 0.78452, 1.0);
  EXPECT_DOUBLE_EQ(2.73621651233878, int1d(rule, ord32_1d), 20);
}

TEST(gauss_legendre_2d, 1)
{
  quad rule;

  rule = gauss_legendre_2d(2, 2, -1.0, 1.0, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(8.82680000000000, int2d(rule, ord2_2d), 20);

  rule = gauss_legendre_2d(2, 2, 0.0, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(7.92170000000000, int2d(rule, ord2_2d), 20);

  rule = gauss_legendre_2d(16, 16, -1.0, 1.0, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(20.2300477243018, int2d(rule, ord16_2d), 20);

  rule = gauss_legendre_2d(16, 16, 0.0, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(5.15174693107546, int2d(rule, ord16_2d), 20);

  rule = gauss_legendre_2d(32, 32, -1.0, 1.0, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(14.8312586645678, int2d(rule, ord32_2d), 50);

  rule = gauss_legendre_2d(32, 32, 0.0, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(3.70781466614194, int2d(rule, ord32_2d), 6);
}

TEST(gauss_legendre_3d, 1)
{
  quad rule;

  rule = gauss_legendre_3d(2, 2, 2, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(11.3338666666667, int3d(rule, ord2_3d), 20);

  rule = gauss_legendre_3d(2, 2, 2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(0.832108333333333, int3d(rule, ord2_3d), 20);

  rule = gauss_legendre_3d(16, 16, 16, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(37.2668310398099, int3d(rule, ord16_3d), 16);

  rule = gauss_legendre_3d(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  EXPECT_DOUBLE_EQ(4.75258887997623, int3d(rule, ord16_3d), 20);
}
