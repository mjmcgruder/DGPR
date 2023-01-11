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

#include <cfloat>

/* set floating point type */

#ifndef SINGLE_PRECISION
#define SINGLE_PRECISION 0
#endif

#if SINGLE_PRECISION
typedef float real;
#define REAL_MAX FLT_MAX
#else
typedef double real;
#define REAL_MAX DBL_MAX
#endif

#ifdef __CUDACC__
#define DECLSPEC __device__
#else
#define DECLSPEC
#endif
