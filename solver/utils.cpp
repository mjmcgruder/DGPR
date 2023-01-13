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

#include "helper_types.cpp"

template<typename int_type>
void split2_cartesian(int_type i, int_type nx, int_type* ix, int_type* iy)
{
  *iy = i / nx;  // truncation intentional
  *ix = i - (nx * (*iy));
}

template<typename int_type>
void split3_cartesian(int_type i, int_type nx, int_type ny, int_type* ix,
                      int_type* iy, int_type* iz)
{
  *iz = i / (nx * ny);               // truncation intentional
  *iy = (i - nx * ny * (*iz)) / nx;  // truncation intentional
  *ix = i - (nx * ny * (*iz)) - (nx * (*iy));
}

// Linearizes a "ndim" dimensional index "indx" with dimension sizes "dims"
// where the dimension at index 0 iterates fastest and the dimension at index
// ndim - 1 iterates slowest.
template<typename int_type>
int_type linear_index(int_type ndim, int_type* dims, int_type* indx)
{
  int_type lindx = 0;

  for (size_t di = 0; di < ndim; ++di) 
  {
    int_type acc = indx[di];
    for (size_t dj = 0; dj < di; ++dj)
    {
      acc *= dims[dj];
    }
    lindx += acc;
  }

  return lindx;
}
