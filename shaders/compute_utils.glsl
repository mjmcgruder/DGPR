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

void split3(const in uint i, const in uint N, 
            out uint ix, out uint iy, out uint iz)
{
  iz = i / (N * N);           // truncation intentional
  iy = (i - N * N * iz) / N;  // truncation intentional
  ix = i - (N * N * iz) - (N * iy);
}

float lagrange1d(const in uint i, const in uint p, const in float x)
{
  float xj, eval = 1.;
  const float xi = float(i) / float(p);
  for (uint j = 0; j < p + 1; ++j)
  {
    if (i != j)
    {
      xj = float(j) / float(p);
      eval *= (x - xj) / (xi - xj);
    }
  }
  return eval;
}

float lagrange3d(const in uint bi, const in uint p, const in vec3 pos)
{
  uint bix, biy, biz;
  split3(bi, p + 1, bix, biy, biz);

  return lagrange1d(bix, p, pos.x) * lagrange1d(biy, p, pos.y) *
         lagrange1d(biz, p, pos.z);
}
