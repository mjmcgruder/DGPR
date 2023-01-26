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

#version 450

/* generally useful functions ----------------------------------------------- */

#if defined(BASIS)

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

#endif

/* BASIS -------------------------------------------------------------------- */

#ifdef BASIS

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer resolution       { uint N; };
layout(std430, set = 0, binding = 1) buffer polynomial_order { uint p; };
layout(std430, set = 0, binding = 2) buffer basis_functions  { float bf[]; };

void main()
{
  uint Np1    = N + 1;
  uint Np1p3  = Np1 * Np1 * Np1;
  uint pp1    = p + 1;
  uint nbf3d  = pp1 * pp1 * pp1;
  uint bf_len = Np1p3 * nbf3d;
  uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;

  float ref_unit = 1. / float(N);

  for (uint bf_pos = gl_GlobalInvocationID.x; bf_pos < bf_len; bf_pos += stride)
  {
    uint bi       = bf_pos / Np1p3;  // truncation intentional
    uint node_num = bf_pos - (bi * Np1p3);

    uint nx, ny, nz;
    split3(node_num, Np1, nx, ny, nz);
    
    vec3 ref_loc = ref_unit * vec3(float(nx), float(ny), float(nz));

    bf[bf_pos] = lagrange3d(bi, p, ref_loc); 
  }
}

#endif

/* GEMM --------------------------------------------------------------------- */

#ifdef GEMM

#define TILE_SIZE 16

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE) in;

layout(std430, binding = 0) buffer dims0 { uint m; };
layout(std430, binding = 1) buffer dims1 { uint k; };
layout(std430, binding = 2) buffer dims2 { uint n; };
layout(std430, binding = 3) buffer arrayA { float A[]; };
layout(std430, binding = 4) buffer arrayB { float B[]; };
layout(std430, binding = 5) buffer arrayC { float C[]; };

shared float tileA[TILE_SIZE * TILE_SIZE];
shared float tileB[TILE_SIZE * TILE_SIZE];

void main()
{
  uint rT = gl_LocalInvocationID.y;
  uint cT = gl_LocalInvocationID.x;
  uint rC = TILE_SIZE * gl_WorkGroupID.y + rT;
  uint cC = TILE_SIZE * gl_WorkGroupID.x + cT;

  float accC = 0.;

  for (uint bk = 0; bk < (k + TILE_SIZE - 1) / TILE_SIZE; ++bk)
  {
    if ((rC) < m && (TILE_SIZE * bk + cT) < k)
      tileA[TILE_SIZE * rT + cT] = A[k * (rC) + (TILE_SIZE * bk + cT)];
    else
      tileA[TILE_SIZE * rT + cT] = 0.;

    if ((TILE_SIZE * bk + rT) < k && (cC) < n)
      tileB[TILE_SIZE * rT + cT] = B[n * (TILE_SIZE * bk + rT) + (cC)];
    else
      tileB[TILE_SIZE * rT + cT] = 0.;

    memoryBarrierShared();
    barrier();

    for (uint i = 0; i < TILE_SIZE; ++i)
      accC += tileA[TILE_SIZE * rT + i] * tileB[TILE_SIZE * i + cT];

    memoryBarrierShared();  // necessary?
    barrier();
  }

  if (rC < m && cC < n)
    C[n * rC + cC] = accC;
}

#endif
