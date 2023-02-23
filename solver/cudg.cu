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

// #include "solve_cuda.cu"
#include "solve_cuda2.cu"

int main(int argc, char** argv)
{
  if (sizeof(real) != sizeof(float))
    errout("set \"-DSINGLE_PRECISION\" for gpu code");

  // parse input
  solver_inputs inputs = parse_solver_inputs(argc, argv);

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



  simstate& U = outputs.get("state");

  float *d_U, *d_R, *d_f;  // [rank [elem [bfuncs]]]
  cudaMalloc(&d_U, U.size() * sizeof(real));
  cudaMalloc(&d_R, U.size() * sizeof(real));
  cudaMalloc(&d_f, U.size() * sizeof(real));

  custore store = custore_make(&geom, &U, d_U);

  cuda_residual(store, sim_params, d_U, d_R, d_f);

  cudaFree(d_U);
  cudaFree(d_R);
  cudaFree(d_f);
  custore_free(&store);



  /* data prep */

  // u32 naux = 0;
  // void (*step)(u64 tstep, float dt, cuda_device_geometry h_cugeom, float* state,
  //              parameters params, cuda_residual_workspace wsp);

  // switch (inputs.scheme)
  // {
  //   case tvdRK3:
  //     step = tvdRK3_cuda;
  //     naux = 4;
  //     break;
  //   case RK4:
  //     step = RK4_cuda;
  //     naux = 8;
  //     break;
  //   default:
  //     errout("requested time scheme not implemented!");
  //     break;
  // }

  // cuda_device_geometry cugeom(geom);
  // cuda_residual_workspace workspace(&cugeom, naux);

  // simstate& U = outputs.get("state");
  // float* state;
  // {
  //   array<float> U_rearranged(U.U.len);

  //   shuffle_state(U, geom, U_rearranged.data);

  //   cudaMalloc(&state, U.size() * sizeof(*state));
  //   cudaMemcpy(state, U_rearranged.data, U.size() * sizeof(*state),
  //              cudaMemcpyHostToDevice);
  // }

  // /* time stepping */

  // char ofile[1024];
  // u64 tfinal = tstep + inputs.niter;
  // for (; tstep < tfinal; ++tstep)
  // {
  //   // checkpoint
  //   if ((tstep + 1) % inputs.chkint == 0)
  //   {
  //     array<float> U_rearranged(U.U.len);

  //     cudaMemcpy(U_rearranged.data, state, U.size() * sizeof(*state),
  //                cudaMemcpyDeviceToHost);

  //     unshuffle_state(U_rearranged.data, geom, U);

  //     sprintf(ofile, "%s%" PRIu64 ".dg", inputs.ofile_prefix, tstep);
  //     write_state(ofile, tstep, geom.core, outputs);
  //   }

  //   // step in time
  //   step(tstep, inputs.dt, cugeom, state, sim_params, workspace);
  // }

  // {
  //   array<float> U_rearranged(U.U.len);
  //   cudaMemcpy(U_rearranged.data, state, U.size() * sizeof(*state),
  //              cudaMemcpyDeviceToHost);

  //   unshuffle_state(U_rearranged.data, geom, U);
  // }

  // sprintf(ofile, "%s.dg", inputs.ofile_prefix);
  // write_state(ofile, tstep - 1, geom.core, outputs);

  /* clean up */

  // cudaFree(state);

  // free_cuda_device_geometry(&cugeom);
  // free_cuda_residual_workspace(&workspace);

  return 0;
}
