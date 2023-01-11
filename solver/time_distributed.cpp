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

#include "geometry_distributed.cpp"
#include "solve.cpp"
#include "time.cpp"

void residual_norm(simstate& R, u64 tstep, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  u64 Rsize_local = R.size(), Rsize = 0;
  real Rsum_local = 0., Rsum = 0.;
  for (u64 i = 0; i < Rsize_local; ++i)
    Rsum_local += R[i] * R[i];

  MPI_Allreduce(&Rsum_local, &Rsum, 1, MPI_REAL, MPI_SUM, comm);
  MPI_Allreduce(&Rsize_local, &Rsize, 1, MPI_UINT64_T, MPI_SUM, comm);

  real Rnorm = sqrt((1. / ((real)Rsize)) * Rsum);

  if (rank == 0)
  {
    printf("%" PRIu64 " : %.17e\n", tstep, Rnorm);
    if (std::isnan(Rnorm))
      errout("looks like things went unstable!");
  }
}

void RK4_distributed(u64 tstep, real dt, simstate& U, simstate* Uaux,
                     distributed_geometry& geom, parameters& sim_params,
                     residual_workspace& workspace)
{
  simstate& R  = Uaux[0];
  simstate& U1 = Uaux[1];
  simstate& U2 = Uaux[2];
  simstate& U3 = Uaux[3];
  simstate& f1 = Uaux[4];
  simstate& f2 = Uaux[5];
  simstate& f3 = Uaux[6];
  simstate& f4 = Uaux[7];

  residual(U, geom, sim_params, workspace, R, f1);

  residual_norm(R, tstep, geom.comm);

  for (u64 i = 0; i < U.size(); ++i)
    U1[i] = U[i] + dt * f1[i] * 0.5;
  residual(U1, geom, sim_params, workspace, R, f2);

  for (u64 i = 0; i < U.size(); ++i)
    U2[i] = U[i] + dt * f2[i] * 0.5;
  residual(U2, geom, sim_params, workspace, R, f3);

  for (u64 i = 0; i < U.size(); ++i)
    U3[i] = U[i] + dt * f3[i];
  residual(U3, geom, sim_params, workspace, R, f4);

  for (u64 i = 0; i < U.size(); ++i)
    U[i] = U[i] + (1. / 6.) * (f1[i] + 2. * f2[i] + 2. * f3[i] + f4[i]) * dt;
}

void tvdRK3_distributed(u64 tstep, real dt, simstate& U, simstate* Uaux,
                        distributed_geometry& geom, parameters& sim_params,
                        residual_workspace& workspace)
{
  simstate& R  = Uaux[0];
  simstate& U1 = Uaux[1];
  simstate& U2 = Uaux[2];
  simstate& f  = Uaux[3];

  residual(U, geom, sim_params, workspace, R, f);

  residual_norm(R, tstep, geom.comm);

  for (u64 i = 0; i < U.size(); ++i)
    U1[i] = U[i] + dt * f[i];
  residual(U1, geom, sim_params, workspace, R, f);

  for (u64 i = 0; i < U.size(); ++i)
    U2[i] = 0.75 * U[i] + 0.25 * U1[i] + 0.25 * dt * f[i];
  residual(U2, geom, sim_params, workspace, R, f);

  for (u64 i = 0; i < U.size(); ++i)
    U[i] = (1. / 3.) * U[i] + (2. / 3.) * U2[i] + (2. / 3.) * dt * f[i];
}
