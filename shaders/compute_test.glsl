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

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer buffer_a { float a[]; };
layout(std430, set = 0, binding = 1) buffer buffer_b { float b[]; };
layout(std430, set = 0, binding = 2) buffer buffer_c { float c[]; };
layout(std430, set = 0, binding = 3) buffer length   { uint len; };

void main()
{
  uint i = gl_GlobalInvocationID.x;

  if (i >= len)
    return;

  c[i] = a[i] * b[i];  
}
