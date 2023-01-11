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

#include "io.cpp"

struct solver_inputs
{
  u64 sol_order = 1;
  u64 geo_order = 2;

  u64 nx            = 6;
  u64 ny            = 4;
  u64 nz            = 4;
  real lx           = 2.;
  real ly           = 1.;
  real lz           = 1.;
  u64 npx           = 1;
  u64 npy           = 1;
  u64 npz           = 1;
  real stretch      = 0.;
  real bmp_height   = 0.1;
  real bmp_center   = 1.0;
  real bmp_variance = 0.05;
  btype bnd_mx      = subsonic_inflow;
  btype bnd_px      = subsonic_outflow;
  btype bnd_my      = inviscid_wall;
  btype bnd_py      = inviscid_wall;
  btype bnd_mz      = periodic;
  btype bnd_pz      = periodic;

  real gamma = 1.4;   // this group of variables are the free parameters in the
  real gcnst = 287.;  // simulation reference state
  real mu    = 1e-3;
  real c     = 1.0;
  real rho   = 1.0;
  real u     = 0.1;
  real v     = 0.0;
  real w     = 0.0;

  char ofile_prefix[512] = "output";
  char ifile_prefix[512] = "";

  real dt             = 1e-4;
  u64 niter           = 1000;
  u64 chkint          = 1000;
  u64 mean_start      = (u64)1 << 63;
  time_scheme scheme  = tvdRK3;
  real eta            = 6;
  real source[5]      = {};
  bool init_turbulent = false;
};

solver_inputs parse_solver_inputs(int argc, char** argv)
{
  solver_inputs inputs;

  const int solver_optc              = 38;
  option solver_optlist[solver_optc] = {
  // simulation params
  mkopt("p", "solution order", &inputs.sol_order),
  mkopt("q", "geometry order", &inputs.geo_order),
  mkopt("dt", "time step", &inputs.dt),
  mkopt("niter", "iteration count", &inputs.niter),
  mkopt("chkint", "checkpoint interval", &inputs.chkint),
  mkopt("scheme", "time stepping scheme", &inputs.scheme),
  mkopt("eta", "BR2 stabilization parameter", &inputs.eta),
  mkopt("turb", "initialize turbulent instead of uniform",
        &inputs.init_turbulent),
  mkopt("mean_start", "iteration at which to start averaging",
        &inputs.mean_start),
  // reference params
  mkopt("gamma", "ratio of specific heats", &inputs.gamma),
  mkopt("mu", "dynamic viscosity", &inputs.mu),
  mkopt("c", "speed of sound", &inputs.c),
  mkopt("rho", "referenec density", &inputs.rho),
  mkopt("u", "reference x velocity", &inputs.u),
  mkopt("v", "reference y velocity", &inputs.v),
  mkopt("w", "reference z velocity", &inputs.w),
  mkopt("fx", "force per unit volume in x", &inputs.source[1]),
  // bump parameters
  mkopt("nx", "num elems in x", &inputs.nx),
  mkopt("ny", "num elems in y", &inputs.ny),
  mkopt("nz", "num elems in z", &inputs.nz),
  mkopt("lx", "domain length in x", &inputs.lx),
  mkopt("ly", "domain length in y", &inputs.ly),
  mkopt("lz", "domain length in z", &inputs.lz),
  mkopt("a", "wall normal stretching parameter (0, 1)", &inputs.stretch),
  mkopt("bmx", "-x boundary type", &inputs.bnd_mx),
  mkopt("bpx", "+x boundary type", &inputs.bnd_px),
  mkopt("bmy", "-y boundary type", &inputs.bnd_my),
  mkopt("bpy", "+y boundary type", &inputs.bnd_py),
  mkopt("bmz", "-z boundary type", &inputs.bnd_mz),
  mkopt("bpz", "+z boundary type", &inputs.bnd_pz),
  mkopt("bmp_h", "bump height", &inputs.bmp_height),
  mkopt("bmp_c", "bump center in x", &inputs.bmp_center),
  mkopt("bmp_v", "bump variance", &inputs.bmp_variance),
  // threading parameters
  mkopt("npx", "processor count in x", &inputs.npx),
  mkopt("npy", "processor count in y", &inputs.npy),
  mkopt("npz", "processor count in z", &inputs.npz),
  // i/o parameters
  mkopt("ofile", "output file prefix", inputs.ofile_prefix),
  mkopt("ifile", "input file prefix", inputs.ifile_prefix),
  };

  if (optparse(argc, argv, solver_optc, solver_optlist))
    errout("solver argument parsing failed!");

  return inputs;
}
