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
#include "io.cpp"

void write_state_distributed(const char* fname, u64 time_step,
                             distributed_geometry& geom_local,
                             map<simstate>& outputs_local)
{
  core_geometry geom_global;
  map<simstate> outputs_global;

  gather(geom_local, outputs_local, geom_global, outputs_global);

  if (geom_local.proc == 0)
    write_state(fname, time_step, geom_global, outputs_global);

  MPI_Barrier(geom_local.comm);
}

void read_state_distributed(const char* fname, MPI_Comm comm, u64 np, u64 npx,
                            u64 npy, u64 npz, u64& time_step,
                            distributed_geometry& geom_local,
                            map<simstate>& outputs_local)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  core_geometry geom_global;
  map<simstate> outputs_global;
  if (rank == 0)
    read_state(fname, time_step, geom_global, outputs_global);

  MPI_Bcast(&time_step, 1, MPI_UINT64_T, 0, comm);

  scatter(geom_global, outputs_global, comm, np, npx, npy, npz, geom_local,
          outputs_local);
}
