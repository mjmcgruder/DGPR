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

#include <cstring>

#include "geometry_common.cpp"

enum struct output_type
{
  mach,
  rho,
  u,
  v,
  w,
};

map<output_type> output_types;
output_type& _DUMMY_OUTPUT1 = output_types.add("mach", output_type::mach);
output_type& _DUMMY_OUTPUT2 = output_types.add("rho", output_type::rho);
output_type& _DUMMY_OUTPUT3 = output_types.add("u", output_type::u);
output_type& _DUMMY_OUTPUT4 = output_types.add("v", output_type::v);
output_type& _DUMMY_OUTPUT5 = output_types.add("w", output_type::w);

void update_mean(simstate& state, simstate& mean, u64 time_step, u64 mean_start)
{
  if (state.size() != mean.size())
    errout("Mean field size does not match state field size!");

  real old_time_step_range = (real)(time_step - mean_start);
  real new_time_step_range = (real)(time_step + 1 - mean_start);

  for (u64 i = 0; i < state.size(); ++i)
    mean[i] = (mean[i] * old_time_step_range + state[i]) / new_time_step_range;
}
