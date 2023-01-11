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

#include <cinttypes>
#include <cmath>
#include <cstdio>

#include "io_distributed.cpp"
#include "solver_init.cpp"
#include "time_distributed.cpp"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* argument parsing */

  solver_inputs inputs = parse_solver_inputs(argc, argv);

  /* --------------------------------- */
  /* state and geometry initialization */
  /* --------------------------------- */

  // generate simulation reference state
  parameters sim_params = make_reference_state(inputs);

  distributed_geometry geom;
  map<simstate> outputs;

  // start from scratch if no restart file, otherwise load restart
  u64 tstep = 0;
  if (strcmp(inputs.ifile_prefix, "") == 0)
  {
    btype bounds[6] = {inputs.bnd_mx, inputs.bnd_px, inputs.bnd_my,
                       inputs.bnd_py, inputs.bnd_mz, inputs.bnd_pz};

    geom = bump_distributed(
    inputs.sol_order, inputs.geo_order, inputs.nx, inputs.ny, inputs.nz,
    inputs.lx, inputs.ly, inputs.lz, inputs.stretch, inputs.bmp_height,
    inputs.bmp_center, inputs.bmp_variance, bounds, MPI_COMM_WORLD, size,
    inputs.npx, inputs.npy, inputs.npz);

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

    read_state_distributed(ifile, MPI_COMM_WORLD, size, inputs.npx, inputs.npy,
                           inputs.npz, tstep, geom, outputs);
    tstep += 1;  // starting on the step after the one that was read
    geom.precompute();
  }

  /* --------------------- */
  /* integrating over time */
  /* --------------------- */

  u64 naux = 0;
  void (*step)(u64 tstep, real dt, simstate & U, simstate * Uaux,
               distributed_geometry & geom, parameters & sim_params,
               residual_workspace & workspace);

  switch (inputs.scheme)
  {
    case tvdRK3:
      step = tvdRK3_distributed;
      naux = 4;
      break;
    case RK4:
      step = RK4_distributed;
      naux = 8;
      break;
    default:
      errout("requested time scheme not implemented!");
      break;
  }

  simstate& U    = outputs.get("state");
  simstate* Uaux = new simstate[naux];
  for (u64 i = 0; i < naux; ++i)
    Uaux[i] = simstate(geom.core);

  residual_workspace workspace(geom);

  char ofile[1024];
  u64 tfinal = tstep + inputs.niter;
  for (; tstep < tfinal; ++tstep)
  {
    // checkpoint
    if ((tstep + 1) % inputs.chkint == 0)
    {
      sprintf(ofile, "%s%" PRIu64 ".dg", inputs.ofile_prefix, tstep);
      write_state_distributed(ofile, tstep, geom, outputs);
    }

    // step in time
    step(tstep, inputs.dt, U, Uaux, geom, sim_params, workspace);

    // (re-)set mean state if it's time to start averaging
    if (tstep == inputs.mean_start)
      outputs.add("mean", simstate(geom.core));

    // handle averaging
    if (tstep >= inputs.mean_start)
    {
      simstate& M = outputs.get("mean");
      update_mean(U, M, tstep, inputs.mean_start);
    }
  }

  sprintf(ofile, "%s.dg", inputs.ofile_prefix);
  write_state_distributed(ofile, tstep - 1, geom, outputs);

  delete[] Uaux;
  MPI_Finalize();
  return 0;
}
