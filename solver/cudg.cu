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

#include <cstdint>
#include <cstdio>

#include "solver_init.cpp"

#include "time_cuda.cu"

int main(int argc, char** argv)
{
  if (sizeof(real) != sizeof(float))
    errout("set \"-DSINGLE_PRECISION\" for gpu code");

  // parse input
  bool help = false;
  solver_inputs inputs = parse_solver_inputs(argc, argv, help);
  if (help) return 0;

  // make reference state
  parameters sim_params = make_reference_state(inputs);

  /* initialize solution */

  u64 tstep = 0;
  shared_geometry geom;
  map<simstate> outputs;

  if (strcmp(inputs.ifile_prefix, "") == 0)
  {
    btype bounds[6] = {inputs.bnd_mx, inputs.bnd_px, inputs.bnd_my,
                       inputs.bnd_py, inputs.bnd_mz, inputs.bnd_pz};

    geom = bump_shared(inputs.sol_order, inputs.geo_order, inputs.nx, inputs.ny,
                       inputs.nz, inputs.lx, inputs.ly, inputs.lz,
                       inputs.stretch, inputs.bmp_height, inputs.bmp_center,
                       inputs.bmp_variance, bounds);

    simstate& U = outputs.add("state", simstate(geom.core));

    if (inputs.init_turbulent)
      turbulent(geom.core, sim_params, inputs.lx, U);
    else
      uniform(sim_params, U);
  }
  else
  {
    char ifile[1024];
    snprintf(ifile, 1024, "%s.dg", inputs.ifile_prefix);

    core_geometry core_geom;
    read_state(ifile, tstep, core_geom, outputs);

    geom = shared_geometry(core_geom);
    geom.precompute();

    tstep += 1;  // starting on the step after the one that was read
  }

  /* data prep */

  u32 naux = 0;
  void (*step)(u64 tstep, real dt, custore store, cuworkspace wsp,
               parameters params, real * U);

  switch (inputs.scheme)
  {
    case tvdRK3:
      step = tvdRK3_cuda;
      naux = 4;
      break;
    case RK4:
      step = RK4_cuda;
      naux = 8;
      break;
    default:
      errout("requested time scheme not implemented!");
      break;
  }

  simstate& U = outputs.get("state");

  real* d_state;
  cudaMalloc(&d_state, U.size() * sizeof(real));

  custore store         = custore_make(&geom, &U, d_state);
  cuworkspace workspace = cuworkspace_make(&store, naux);

  /* time stepping */

  char ofile[1024];
  u64 tfinal = tstep + inputs.niter;
  for (; tstep < tfinal; ++tstep)
  {
    // checkpoint
    if ((tstep + 1) % inputs.chkint == 0)
    {
      cudaMemcpy(U.U.data, d_state, U.size() * sizeof(real),
                 cudaMemcpyDeviceToHost);

      sprintf(ofile, "%s%" PRIu64 ".dg", inputs.ofile_prefix, tstep);
      write_state(ofile, tstep, geom.core, outputs);
    }

    // step in time
    step(tstep, inputs.dt, store, workspace, sim_params, d_state);
  }

  cudaMemcpy(U.U.data, d_state, U.size() * sizeof(real),
             cudaMemcpyDeviceToHost);

  sprintf(ofile, "%s.dg", inputs.ofile_prefix);
  write_state(ofile, tstep - 1, geom.core, outputs);

  /* clean up */

  cudaFree(d_state);
  custore_free(&store);
  cuworkspace_free(&workspace);

  return 0;
}
