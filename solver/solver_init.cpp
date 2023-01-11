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

#include "geometry_common.cpp"
#include "solver_params.cpp"

void uniform(parameters& sim_params, simstate& U)
{
  for (u64 ei = 0; ei < U.nelem; ++ei)
    for (u64 ri = 0; ri < U.rank; ++ri)
      for (u64 bi = 0; bi < U.nbfnc; ++bi)
        U(ei, ri, bi) = sim_params.Ufr[ri];
}

void turbulent(core_geometry& geom, parameters& sim_params, real l, simstate& U)
{
  real twopi = 2. * PI;
  real offst = 0.3;

  real rho = sim_params.Ufr[0];
  real u   = sim_params.Ufr[1] / rho;
  real v   = sim_params.Ufr[2] / rho;
  real w   = sim_params.Ufr[3] / rho;

  real vmag = sqrt(u * u + v * v + w * w);
  real amp  = vmag / 10.;

  for (u64 ei = 0; ei < U.nelem; ++ei)
  {
    for (u64 bi = 0; bi < U.nbfnc; ++bi)
    {
      v3 ref = lagrange_node3d(bi, geom.p);
      v3 glo = geom.ref2glo(ei, ref);

      U(ei, 0, bi) = sim_params.Ufr[0];
      U(ei, 4, bi) = sim_params.Ufr[4];

      real vel[3] = {sim_params.Ufr[1] / rho, sim_params.Ufr[2] / rho,
                     sim_params.Ufr[3] / rho};

      for (int i = 0; i < 3; ++i)
      {
        vel[i] += amp * (sin(twopi * ((glo.x - i * offst) / (0.5 * l))) *
                         sin(twopi * ((glo.y - i * offst) / (0.5 * l))) *
                         sin(twopi * ((glo.z - i * offst) / (0.5 * l))));
        vel[i] += amp * (sin(twopi * ((glo.x - i * offst) / (0.25 * l))) *
                         sin(twopi * ((glo.y - i * offst) / (0.25 * l))) *
                         sin(twopi * ((glo.z - i * offst) / (0.25 * l))));
        vel[i] += amp * (sin(twopi * ((glo.x - i * offst) / (0.125 * l))) *
                         sin(twopi * ((glo.y - i * offst) / (0.125 * l))) *
                         sin(twopi * ((glo.z - i * offst) / (0.125 * l))));
        vel[i] += amp * (sin(twopi * ((glo.x - i * offst) / (0.0625 * l))) *
                         sin(twopi * ((glo.y - i * offst) / (0.0625 * l))) *
                         sin(twopi * ((glo.z - i * offst) / (0.0625 * l))));
      }

      U(ei, 1, bi) = vel[0] * rho;
      U(ei, 2, bi) = vel[1] * rho;
      U(ei, 3, bi) = vel[2] * rho;
    }
  }
}
