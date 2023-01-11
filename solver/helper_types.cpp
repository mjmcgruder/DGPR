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

#include <cstdint>

#include "compilation_config.cpp"

/* define fixed width integer shorthand */

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// Scoped enum to specify an array component offset in a cute way.
//   global space (x, y, z)
//   reference space (xi, eta, zeta)
//   surface integral parameters (s, t)
enum struct component : u64
{
  x    = 0,  // global space
  y    = 1,
  z    = 2,
  xi   = 0,  // reference space
  eta  = 1,
  zeta = 2,
  s    = 0,  // parametrized (for surface integrals)
  t    = 1
};

/* small vector types */

struct v2
{
  real x;
  real y;

  v2();
  v2(real x_, real y_);

  v2 operator+(v2 rhs);
  v2 operator-(v2 rhs);
  v2 operator*(v2 rhs);
  v2 operator/(v2 rhs);
  v2 operator/(real rhs);
};

v2 operator+(real sca, v2 vec);
v2 operator+(v2 vec, real sca);
v2 operator-(real sca, v2 vec);
v2 operator-(v2 vec, real sca);
v2 operator*(real sca, v2 vec);
v2 operator*(v2 vec, real sca);

struct v3
{
  real x;
  real y;
  real z;

  v3();
  v3(real x_, real y_, real z_);

  v3 operator+(v3 rhs);
  v3 operator-(v3 rhs);
  v3 operator*(v3 rhs);
  v3 operator/(v3 rhs);
  v3 operator/(real rhs);
};

v3 operator+(real sca, v3 vec);
v3 operator+(v3 vec, real sca);
v3 operator-(real sca, v3 vec);
v3 operator-(v3 vec, real sca);
v3 operator*(real sca, v3 vec);
v3 operator*(v3 vec, real sca);

// v2 implementation -----------------------------------------------------------

v2::v2() : x(0.), y(0.)
{}

v2::v2(real x_, real y_) : x(x_), y(y_)
{}

v2 v2::operator+(v2 rhs)
{
  return v2(x + rhs.x, y + rhs.y);
}

v2 v2::operator-(v2 rhs)
{
  return v2(x - rhs.x, y - rhs.y);
}

v2 v2::operator*(v2 rhs)
{
  return v2(x * rhs.x, y * rhs.y);
}

v2 v2::operator/(v2 rhs)
{
  return v2(x / rhs.x, y / rhs.y);
}

v2 v2::operator/(real rhs)
{
  return v2(x / rhs, y / rhs);
}

v2 operator+(real sca, v2 vec)
{
  return v2(vec.x + sca, vec.y + sca);
}

v2 operator+(v2 vec, real sca)
{
  return v2(vec.x + sca, vec.y + sca);
}

v2 operator-(real sca, v2 vec)
{
  return v2(vec.x - sca, vec.y - sca);
}

v2 operator-(v2 vec, real sca)
{
  return v2(vec.x - sca, vec.y - sca);
}

v2 operator*(real sca, v2 vec)
{
  return v2(vec.x * sca, vec.y * sca);
}

v2 operator*(v2 vec, real sca)
{
  return v2(vec.x * sca, vec.y * sca);
}

// v3 implementation -----------------------------------------------------------

v3::v3() : x(0.), y(0.), z(0.)
{}

v3::v3(real x_, real y_, real z_) : x(x_), y(y_), z(z_)
{}

v3 v3::operator+(v3 rhs)
{
  return v3(x + rhs.x, y + rhs.y, z + rhs.z);
}

v3 v3::operator-(v3 rhs)
{
  return v3(x - rhs.x, y - rhs.y, z - rhs.z);
}

v3 v3::operator*(v3 rhs)
{
  return v3(x * rhs.x, y * rhs.y, z * rhs.z);
}

v3 v3::operator/(v3 rhs)
{
  return v3(x / rhs.x, y / rhs.y, z / rhs.z);
}

v3 v3::operator/(real rhs)
{
  return v3(x / rhs, y / rhs, z / rhs);
}

v3 operator+(real sca, v3 vec)
{
  return v3(vec.x + sca, vec.y + sca, vec.z + sca);
}

v3 operator+(v3 vec, real sca)
{
  return v3(vec.x + sca, vec.y + sca, vec.z + sca);
}

v3 operator-(real sca, v3 vec)
{
  return v3(vec.x - sca, vec.y - sca, vec.z - sca);
}

v3 operator-(v3 vec, real sca)
{
  return v3(vec.x - sca, vec.y - sca, vec.z - sca);
}

v3 operator*(real sca, v3 vec)
{
  return v3(vec.x * sca, vec.y * sca, vec.z * sca);
}

v3 operator*(v3 vec, real sca)
{
  return v3(vec.x * sca, vec.y * sca, vec.z * sca);
}
