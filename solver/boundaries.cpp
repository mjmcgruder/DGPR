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

#include "data_structures.cpp"

// This enum represents boundary types available in the simulation, well,
// hopefully they're actually implemented. This enum specificallly assigns
// negative numbers to the various boundary types because boundary types in the
// face list must be stored as negative numbers, element indices are stored as
// positive numbers (see the "face_list" member of "geometry" for more details).
enum btype : s64
{
  interior         = -1,
  periodic         = -2,
  freestream       = -3,
  inviscid_wall    = -4,
  subsonic_inflow  = -5,
  subsonic_outflow = -6,
  no_slip_wall     = -7,
};

map<btype> boundary_types;
// btype& _DUMMY_BTYPE1 = boundary_types.add("interior", interior);
btype& _DUMMY_BTYPE2 = boundary_types.add("periodic", periodic);
btype& _DUMMY_BTYPE3 = boundary_types.add("freestream", freestream);
btype& _DUMMY_BTYPE4 = boundary_types.add("inviscid_wall", inviscid_wall);
btype& _DUMMY_BTYPE5 = boundary_types.add("subsonic_inflow", subsonic_inflow);
btype& _DUMMY_BTYPE6 = boundary_types.add("subsonic_outflow", subsonic_outflow);
btype& _DUMMY_BTYPE7 = boundary_types.add("no_slip_wall", no_slip_wall);
