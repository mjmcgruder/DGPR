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

#include <cinttypes>

#include "geometry_common.cpp"

enum time_scheme
{
  RK4,
  tvdRK3,
};

map<time_scheme> multistage_schemes;
time_scheme& _DUMMY_SCHEME1 = multistage_schemes.add("tvdRK3", tvdRK3);
time_scheme& _DUMMY_SCHEME2 = multistage_schemes.add("RK4", RK4);
