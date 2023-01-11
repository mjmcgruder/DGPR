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

#include <cmath>

#include "helper_types.cpp"
#include "error.cpp"

DECLSPEC void inviscid_wall_state(real* UL, real* n, real gamma, real* Ub)
{
  // evaluate wall tangential velocity
  real up  = UL[1] / UL[0];  // "+" (interior) velocities
  real vp  = UL[2] / UL[0];
  real wp  = UL[3] / UL[0];
  real vdn = up * n[0] + vp * n[1] + wp * n[2];
  real ub  = up - vdn * n[0];  // tangential wallboundary velocity
  real vb  = vp - vdn * n[1];
  real wb  = wp - vdn * n[2];

  Ub[0] = UL[0];
  Ub[1] = UL[0] * ub;
  Ub[2] = UL[0] * vb;
  Ub[3] = UL[0] * wb;
  Ub[4] = UL[4];
}

DECLSPEC void no_slip_wall_state(real* UL, real* Ub)
{
  Ub[0] = UL[0];
  Ub[1] = 0.;
  Ub[2] = 0.;
  Ub[3] = 0.;
  Ub[4] = UL[4];
}

DECLSPEC void subsonic_inflow_state(
real* UL, real* n, real gamma, real R, real Pt, real Tt, real* Ub)
{
  real gm1 = gamma - 1.;

  real up   = UL[1] / UL[0];
  real vp   = UL[2] / UL[0];
  real wp   = UL[3] / UL[0];
  real Vsqp = up * up + vp * vp + wp * wp;
  real pp   = gm1 * (UL[4] - 0.5 * UL[0] * Vsqp);
  real unp  = up * n[0] + vp * n[1] + wp * n[2];
  real cp   = sqrt(gamma * pp / UL[0]);
  real Jp   = unp + ((2. * cp) / gm1);
  real a    = gamma * R * Tt - 0.5 * gm1 * Jp * Jp;
  real b    = (4. * gamma * R * Tt * -1.) / gm1;
  real c    = (4. * gamma * R * Tt) / (gm1 * gm1) - Jp * Jp;
  real sol0 = (-b + sqrt(b * b - 4. * a * c)) / (2. * a);
  real sol1 = (-b - sqrt(b * b - 4. * a * c)) / (2. * a);
  real Mb   = 0.;

  if (sol0 >= 0. && sol1 >= 0.)
    if (sol0 < sol1)
      Mb = sol0;
    else
      Mb = sol1;
  else if (sol0 >= 0.)
    Mb = sol0;
  else if (sol1 >= 0.)
    Mb = sol1;
#ifndef __CUDACC__
  else
    errout("looks like you have a negative inflow bound Mach number");
#endif

  real Tb   = Tt / (1. + 0.5 * gm1 * Mb * Mb);
  real Pb   = Pt * pow(Tb / Tt, gamma / gm1);
  real rhob = Pb / (R * Tb);
  real cb   = sqrt((gamma * Pb) / rhob);
  real ub   = Mb * cb * -n[0];
  real vb   = Mb * cb * -n[1];
  real wb   = Mb * cb * -n[2];
  real rEb  = (Pb / gm1) + 0.5 * rhob * (ub * ub + vb * vb + wb * wb);

  Ub[0] = rhob;
  Ub[1] = rhob * ub;
  Ub[2] = rhob * vb;
  Ub[3] = rhob * wb;
  Ub[4] = rEb;
}

DECLSPEC void subsonic_outflow_state(real* UL, real* n, real gamma, real P,
                                     real* Ub)
{
  real gm1  = gamma - 1.;
  real rhop = UL[0];
  real up   = UL[1] / rhop;
  real vp   = UL[2] / rhop;
  real wp   = UL[3] / rhop;
  real Vsqp = up * up + vp * vp + wp * wp;
  real pp   = gm1 * (UL[4] - 0.5 * rhop * Vsqp);
  real Sp   = pp / pow(rhop, gamma);
  real rhob = pow(P / Sp, 1. / gamma);
  real cb   = sqrt((gamma * P) / rhob);
  real unp  = up * n[0] + vp * n[1] + wp * n[2];
  real cp   = sqrt(gamma * pp / rhop);
  real Jp   = unp + ((2. * cp) / gm1);
  real unb  = Jp - ((2. * cb) / (gamma - 1.));

  real vpdn = (up * n[0] + vp * n[1] + wp * n[2]);
  real ub   = up - vpdn * n[0] + unb * n[0];
  real vb   = vp - vpdn * n[1] + unb * n[1];
  real wb   = wp - vpdn * n[2] + unb * n[2];
  real rEb  = (P / (gamma - 1.)) + 0.5 * rhob * (ub * ub + vb * vb + wb * wb);

  Ub[0] = rhob;
  Ub[1] = rhob * ub;
  Ub[2] = rhob * vb;
  Ub[3] = rhob * wb;
  Ub[4] = rEb;
}
