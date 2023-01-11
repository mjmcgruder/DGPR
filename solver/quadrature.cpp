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

#include <cfloat>
#include <cmath>

#include "constants.cpp"
#include "helper_types.cpp"
#include "data_structures.cpp"

// Stores a single one, two, or three dimensional quadrature rule.
// Data access can be handled manually or by the accessors, memory management is
// automated by special members.
// Nodes in the "x" position array have the following ordering:
//   z positions
//     y positions
//       x positions
//         [x (y) (z) component]  | y and z components are optional
// The "w" weight array takes the same node ordering
//   z positions
//     y positions
//       x positions
//         [weight at this node]
struct quad
{
  u64 dim;  // dimension (1, 2, or 3)
  u64 n;    // total node count
  u64 nx;   // x node count
  u64 ny;   // y node count
  u64 nz;   // z node count
  array<real> x;  // positions
  array<real> w;  // weights

  // constructors

  quad();
  quad(u64 nx_);
  quad(u64 nx_, u64 ny_);
  quad(u64 nx_, u64 ny_, u64 nz_);

  // conventient position access

  v2 pos2d(u64 index);
  v3 pos3d(u64 index);
  real* pos(u64 index);
  real* pos(u64 ix, u64 iy, u64 iz);
  real& pos(u64 index, component comp);
  real& pos(u64 ix, u64 iy, u64 iz, component comp);
  real& wgt(u64 index);
  real& wgt(u64 ix, u64 iy, u64 iz);
};

// quadrature rule generators --------------------------------------------------

// Returns a 1d quadrature rule in arrays "x" (positions) and "w" (weights)
// given the beginning and end of the quadrature interval "is" and "ie"
// respectively, and the node cound "n."
// This function uses Newton's method to find the roots of legendre polynomials
// and thus the node locations for the rule. Chebyshev nodes are used for the
// initial guess. The Legendre polynomials and their derivatives are evaluated
// through recurrence relations that can probably still be found on Wolfram
// Mathworld.
// The function operates in double precision since it is seeking to eliminate
// any roundoff error that could creep into a single precision quadrature rule
// (if the code happens to be configured for single precision).
// I have also observed Newton's method struggling to converge near machine
// precision for high order rules in a single precision version of this
// function.
static void gauss_legendre_1d_dbl(
double is, double ie, u64 n, double* x, double* w)
{
  // quick out if very low order
  if (n < 2)
  {
    x[0] = is + (ie - is) * (0.0 + 1.0) / 2.0;
    w[0] = 1.0 / (1.0 / (ie - is));
    return;
  }
  if (n == 2)
  {
    x[0] = is + (ie - is) * ((-1.0 / sqrt(3.0) + 1.0) / 2.0);
    x[1] = is + (ie - is) * ((+1.0 / sqrt(3.0) + 1.0) / 2.0);
    w[0] = 1.0 / (2.0 / (ie - is));
    w[1] = 1.0 / (2.0 / (ie - is));
    return;
  }

  // chebyshev node initial guess
  u64 i, j;
  for (j = 0; j < n; ++j)
  {
    x[j] = cos(((2.0 * (double)(n - j) - 1.0) / (2.0 * (double)n)) * PI);
  }

  // newton iteration initialization
  double* p       = new double[n * (n + 1)];  // polynomial evaluation history
  double* dp      = new double[n];            // polynomial derivative evals
  double* x_old   = new double[n];            // old node positions (for newton)
  double max_diff = HUGE_VAL;
  double diff;
  for (j = 0; j < n; ++j)
  {
    p[j] = 1.0;
  }
  dp[0] = 0.0;
  dp[1] = 1.0;

  // newton iterations
  while (max_diff > DBL_EPSILON)
  {
    // init second polynomial manually (recurrence required previous two orders)
    for (j = 0; j < n; ++j)
    {
      p[j + n] = x[j];
    }
    // eval polynomial at each x using recurrence relations
    for (i = 2; i <= n; ++i)
    {  // iter through orders
      for (j = 0; j < n; ++j)
      {  // iter over nodes
        p[j + i * n] =
        ((2.0 * (double)(i - 1) + 1.0) * x[j] * p[j + (i - 1) * n] -
         (double)(i - 1) * p[j + (i - 2) * n]) /
        (double)i;
      }
    }
    // eval derivative at each x
    for (j = 0; j < n; ++j)
    {
      dp[j] =
      (-(double)n * x[j] * p[j + n * n] + (double)n * p[j + (n - 1) * n]) /
      (1.0 - x[j] * x[j]);
    }
    // store old stuff, newton step, and calculate (max) error
    max_diff = 0.0;
    for (j = 0; j < n; ++j)
    {
      x_old[j] = x[j];
      x[j] -= p[j + n * n] / dp[j];
      if ((diff = fabs(x[j] - x_old[j])) > max_diff)
      {
        max_diff = diff;
      }
    }
  }

  // weight calculation and interval change
  for (j = 0; j < n; ++j)
  {
    w[j] = ((ie - is) / ((1.0 - x[j] * x[j]) * dp[j] * dp[j]));
    x[j] = (is + (ie - is) * ((x[j] + 1.0) / 2.0));
  }

  // bye bye
  delete[] p;
  delete[] dp;
  delete[] x_old;
}

// The functions below use the double precision 1D gauss legendre quadrature
// rule evaluation above to generate 1D, 2D, and 3D rules in the desired
// precision (whatever "real" is set to).
// The multi-dimensional rules use tensor products of 1D rules.

// Computes quadrature nodes and weights for a 1D Gauss-Legendre quadrature rule
// and returns the result in a "quad" struct. The returned quadrature rule will
// contain the minimum number of required nodes to to integrate a polynomial of
// order "order" or less to near machine precision over the interval
// "is" (interval start) to "ie" (interval end).
// The inverval is specified in double precision to minimize round off error in
// the generation process even if simulating in single precision.
quad gauss_legendre_1d(int order, double is, double ie)
{
  u64 n = (u64)ceil(((double)order + 1.0) / 2.0);
  quad rule(n);
  // calculate single precision rule using doubles
  double* x = new double[n];
  double* w = new double[n];
  gauss_legendre_1d_dbl(is, ie, n, x, w);
  // fill rule and cast
  for (u64 i = 0; i < n; ++i)
  {
    rule.x[i] = (real)x[i];
    rule.w[i] = (real)w[i];
  }

  delete[] x;
  delete[] w;
  return rule;
}

// Computes quadrature nodes and weights for a 2D Gauss-Legendre quadrature
// rule over the interval "isx" to "iex" in x and likewise in y. A polynomial of
// "orderx" in x and "ordery" in y or less will be integrated to near machine
// precision. The rule is generated in tensor product fashion, two 1D rules are
// combined by multiplying the weights for each combination of 1D position to
// form the 2D weight and storing each position in different dimensions.
quad gauss_legendre_2d(
int orderx, int ordery, double isx, double iex, double isy, double iey)
{
  u64 nx = (u64)ceil(((double)orderx + 1.0) / 2.0);
  u64 ny = (u64)ceil(((double)ordery + 1.0) / 2.0);
  quad rule(nx, ny);

  double* xx = new double[nx];
  double* wx = new double[nx];
  double* xy = new double[ny];
  double* wy = new double[ny];
  gauss_legendre_1d_dbl(isx, iex, nx, xx, wx);
  gauss_legendre_1d_dbl(isy, iey, ny, xy, wy);

  u64 nn;
  for (u64 j = 0; j < ny; ++j)
  {
    for (u64 i = 0; i < nx; ++i)
    {
      nn                 = nx * j + i;
      rule.x[2 * nn + 0] = (real)xx[i];
      rule.x[2 * nn + 1] = (real)xy[j];
      rule.w[nn]         = (real)(wx[i] * wy[j]);
    }
  }

  delete[] xx;
  delete[] wx;
  delete[] xy;
  delete[] wy;
  return rule;
}

// Same as above but gives a 3D quadrature rule.
quad gauss_legendre_3d(int orderx,
                       int ordery,
                       int orderz,
                       double isx,
                       double iex,
                       double isy,
                       double iey,
                       double isz,
                       double iez)
{
  u64 nx = (u64)ceil(((double)orderx + 1.0) / 2.0);
  u64 ny = (u64)ceil(((double)ordery + 1.0) / 2.0);
  u64 nz = (u64)ceil(((double)orderz + 1.0) / 2.0);
  quad rule(nx, ny, nz);

  double* xx = new double[nx];
  double* wx = new double[nx];
  double* xy = new double[ny];
  double* wy = new double[ny];
  double* xz = new double[nz];
  double* wz = new double[nz];
  gauss_legendre_1d_dbl(isx, iex, nx, xx, wx);
  gauss_legendre_1d_dbl(isy, iey, ny, xy, wy);
  gauss_legendre_1d_dbl(isz, iex, nz, xz, wz);

  u64 nn;
  for (u64 k = 0; k < nz; ++k)
  {
    for (u64 j = 0; j < ny; ++j)
    {
      for (u64 i = 0; i < nx; ++i)
      {
        nn                 = nx * ny * k + nx * j + i;
        rule.x[3 * nn + 0] = (real)xx[i];
        rule.x[3 * nn + 1] = (real)xy[j];
        rule.x[3 * nn + 2] = (real)xz[k];
        rule.w[nn]         = (real)(wx[i] * wy[j] * wz[k]);
      }
    }
  }

  delete[] xx;
  delete[] wx;
  delete[] xy;
  delete[] wy;
  delete[] xz;
  delete[] wz;
  return rule;
}

// quad implementation ---------------------------------------------------------

// conventient position access
v2 quad::pos2d(u64 index)
{
  return v2(x[dim * index + 0], x[dim * index + 1]);
}
v3 quad::pos3d(u64 index)
{
  return v3(x[dim * index + 0], x[dim * index + 1], x[dim * index + 2]);
}
real* quad::pos(u64 index)
{
  return x + dim * index;
}
real* quad::pos(u64 ix, u64 iy, u64 iz)
{
  return x + (dim * (nx * ny * iz + nx * iy + ix));
}
real& quad::pos(u64 index, component comp)
{
  return x[dim * index + (u64)comp];
}
real& quad::pos(u64 ix, u64 iy, u64 iz, component comp)
{
  return x[dim * (nx * ny * iz + nx * iy + ix) + (u64)comp];
}
real& quad::wgt(u64 index)
{
  return w[index];
}
real& quad::wgt(u64 ix, u64 iy, u64 iz)
{
  return w[nx * ny * iz + nx * iy + ix];
}

// construct default
quad::quad() : dim(0), n(0), nx(0), ny(0), nz(0), x(), w()
{}

// construct 1d
quad::quad(u64 nx_) :
dim(1), n(nx_), nx(nx_), ny(1), nz(1), x(nx), w(nx)
{}

// construct 2d
quad::quad(u64 nx_, u64 ny_) :
dim(2),
n(nx_ * ny_),
nx(nx_),
ny(ny_),
nz(1),
x(dim * n),
w(n)
{}

// construct 3d
quad::quad(u64 nx_, u64 ny_, u64 nz_) :
dim(3),
n(nx_ * ny_ * nz_),
nx(nx_),
ny(ny_),
nz(nz_),
x(dim * n),
w(n)
{}
