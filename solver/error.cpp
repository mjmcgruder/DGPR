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

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "compilation_config.cpp"

// Standard error "handling" macro. This program is just  CFD, if there's an
// error you're already dead so we'll just make the operating system earn it's
// keep and fall on our sword with a cute error message. A better way to do
// things would be to actually return a stack trace along with the cute message
// but... that's hard so maybe later, debuggers for the trace will suffice for
// now.
// The input to this macro is just like printf, the first argument of the
// variadic macro needs to be the format spec or weird things will probably
// happen.
#define errout(...) \
  fall_on_your_sword(__FILE__, __LINE__, __func__, __VA_ARGS__)

#define ERR_MSG_SIZE 4096

inline void fall_on_your_sword(const char* file,
                               int line,
                               const char* func,
                               ...)
{
  char error_message[ERR_MSG_SIZE];
  va_list args;
  va_start(args, func);
  vsnprintf(error_message, ERR_MSG_SIZE, va_arg(args, char*), args);
  va_end(args);
  printf("AN ERROR OCCURRED!!! (in %s at line %d of %s):\n", func, line, file);
  printf("%s\n\n", error_message);
  std::exit(1);
}
