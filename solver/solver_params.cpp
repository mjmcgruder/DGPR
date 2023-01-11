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

#include "compilation_config.cpp"
#include "input.cpp"

struct parameters
{
  // fluid parameters
  real gamma;  // ratio of specific heats
  real R;      // specific gas constant
  real mu;     // dynamic viscosity
  // reference parameters
  real c;          // speed of sound
  real T;          // static temperature
  real P;          // static pressure
  real Ufr[5];     // freestream state
  real source[5];  // constant source term values
  // simulation tuning
  real eta;  // BR2 stabilization parameter
};

// Generate a reference state from the input with energy set such that the
// requested speed of sound is matched.
parameters make_reference_state(solver_inputs& inputs)
{
  parameters sim_params;

  real u = inputs.u;
  real v = inputs.v;
  real w = inputs.w;

  real P = inputs.c * inputs.c * (inputs.rho / inputs.gamma);

  sim_params.gamma  = inputs.gamma;
  sim_params.R      = inputs.gcnst;
  sim_params.mu     = inputs.mu;
  sim_params.c      = inputs.c;
  sim_params.T      = P / (inputs.rho * inputs.gcnst);
  sim_params.P      = P;
  sim_params.eta    = inputs.eta;
  sim_params.Ufr[0] = inputs.rho;
  sim_params.Ufr[1] = inputs.rho * inputs.u;
  sim_params.Ufr[2] = inputs.rho * inputs.v;
  sim_params.Ufr[3] = inputs.rho * inputs.w;
  sim_params.Ufr[4] =
  (P / (inputs.gamma - 1.)) + 0.5 * inputs.rho * (u * u + v * v + w * w);
  sim_params.source[0] = inputs.source[0];
  sim_params.source[1] = inputs.source[1];
  sim_params.source[2] = inputs.source[2];
  sim_params.source[3] = inputs.source[3];
  sim_params.source[4] = inputs.source[4];

  return sim_params;
}
