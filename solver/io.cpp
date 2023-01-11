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

#include "output.cpp"
#include "time.cpp"

#define iochk(f, num)                  \
  {                                    \
    if ((f) != (num))                  \
    {                                  \
      errout("i/o operation failed!"); \
    }                                  \
  }

// everything will be converted to double precision for writing
void write_state(const char* fname, u64 time_step, core_geometry& geom,
                 map<simstate>& outputs)
{
  FILE* fstr = fopen(fname, "wb");
  if (!fstr)
    errout("failed to open file \"%s\" for state write!", fname);

  /* metadata */

  // time step
  iochk(fwrite(&time_step, sizeof(time_step), 1, fstr), 1);

  // dimensions
  iochk(fwrite(&geom.nx, sizeof(geom.nx), 1, fstr), 1);
  iochk(fwrite(&geom.ny, sizeof(geom.ny), 1, fstr), 1);
  iochk(fwrite(&geom.nz, sizeof(geom.nz), 1, fstr), 1);

  // orders
  iochk(fwrite(&geom.p, sizeof(geom.p), 1, fstr), 1);
  iochk(fwrite(&geom.q, sizeof(geom.q), 1, fstr), 1);

  // boundary types
  for (u64 i = 0; i < 6; ++i)
    iochk(fwrite(&geom.bounds[i], sizeof(geom.bounds[0]), 1, fstr), 1);

  // output count
  u64 noutput = 0;
  for (auto oi = outputs.begin(); oi != outputs.end(); ++oi)
    ++noutput;
  iochk(fwrite(&noutput, sizeof(noutput), 1, fstr), 1);

  /* geometry nodes */

  for (u64 i = 0; i < geom.node_size(); ++i)
  {
    double data = (double)geom.node_[i];
    iochk(fwrite(&data, sizeof(double), 1, fstr), 1);
  }

  /* state info */

  for (auto oi = outputs.begin(); oi != outputs.end(); ++oi)
  {
    string key      = oi.key();
    simstate output = oi.val();

    u64 key_len = key.len();
    iochk(fwrite(&key_len, sizeof(key_len), 1, fstr), 1);
    iochk(fwrite(key.c_str(), sizeof(char), key_len, fstr), key_len);
    for (u64 di = 0; di < output.size(); ++di)
    {
      double data = (double)output[di];
      iochk(fwrite(&data, sizeof(double), 1, fstr), 1);
    }
  }

  fclose(fstr);
}

void read_state(const char* fname, u64& time_step, core_geometry& geom,
                map<simstate>& outputs)
{
  u64 nx, ny, nz;
  u64 p, q;
  s64 bounds[6];

  FILE* fstr = fopen(fname, "rb");
  if (!fstr)
    errout("failed to open file \"%s\" for state read!", fname);

  /* metadata */

  // time step
  iochk(fread(&time_step, sizeof(time_step), 1, fstr), 1);

  // dimensions
  iochk(fread(&nx, sizeof(nx), 1, fstr), 1);
  iochk(fread(&ny, sizeof(ny), 1, fstr), 1);
  iochk(fread(&nz, sizeof(nz), 1, fstr), 1);

  // orders
  iochk(fread(&p, sizeof(p), 1, fstr), 1);
  iochk(fread(&q, sizeof(q), 1, fstr), 1);

  // boundary types
  for (u64 i = 0; i < 6; ++i)
    iochk(fread(&bounds[i], sizeof(bounds[0]), 1, fstr), 1);

  // output count
  u64 noutput;
  iochk(fread(&noutput, sizeof(noutput), 1, fstr), 1);

  /* construct solution geometry and state (only need metadata for sizing) */

  geom = core_geometry(p, q, nx, ny, nz, bounds);

  /* read geometry nodes */

  for (u64 i = 0; i < geom.node_size(); ++i)
  {
    double data;
    iochk(fread(&data, sizeof(double), 1, fstr), 1);
    geom.node_[i] = (real)data;
  }

  /* read each output */

  for (u64 oi = 0; oi < noutput; ++oi)
  {
    u64 key_len;
    char key_arr[1024];
    iochk(fread(&key_len, sizeof(key_len), 1, fstr), 1);
    iochk(fread(key_arr, sizeof(char), key_len, fstr), key_len);
    key_arr[key_len] = '\0';

    simstate& output = outputs.add(string(key_arr), simstate(geom));

    for (u64 i = 0; i < output.size(); ++i)
    {
      double data;
      iochk(fread(&data, sizeof(double), 1, fstr), 1);
      output[i] = data;
    }
  }

  fclose(fstr);
}

/*
 *  command line input
 */

enum optype
{
  ot_float,
  ot_double,
  ot_u32,
  ot_u64,
  ot_s32,
  ot_s64,
  ot_string,
  ot_bool,
  ot_btype,
  ot_rndtype,
  ot_time_scheme,
};

// argument parser will take an array of these options and convert the string it
// finds into the appropriate type based on the stored type info
// note the strings pointed to here need to be null termianted
struct option
{
  const char* name;  // options have only a single name (no long / short)
  const char* description;
  optype type;
  void* data;
};

// for now let's say all options are specified with a single dash
// also for now let's say all arguments are specified in the next argument
// everything else is just... ignored
int optparse(int argc, char** argv, int optc, option* opts)
{
  int head = 1;  // set at position of next arg to be read
  // print help and exit if requested
  if (argc > 1 && strcmp(argv[head], "-h") == 0)
  {
    printf("Options:\n");
    for (u32 oi = 0; oi < optc; ++oi)
    {
      printf("  %10s: %s\n", opts[oi].name, opts[oi].description);
    }
    return 1;
  }
  // otherwise parse
  while (head < argc)
  {
    if (argv[head][0] == '-')
    {
      // search for the argument in the arg list (stupidly)
      u32 oi;
      for (oi = 0; oi < optc; ++oi)
      {
        if (strcmp(opts[oi].name, argv[head] + 1) == 0)
        {
          // break out if the option is found
          optype type = opts[oi].type;
          if ((head + 1 >= argc) && type != ot_bool)
          {
            printf("no argument provided for \"%s\" at end of argument list\n",
                   argv[head] + 1);
            ++head;
            break;
          }

          char* end;
          switch (type)
          {
            case ot_float:
              *((float*)opts[oi].data) = strtof(argv[head + 1], &end);
              break;
            case ot_double:
              *((double*)opts[oi].data) = strtod(argv[head + 1], &end);
              break;
            case ot_u32:
              *((u32*)opts[oi].data) = (u32)strtoul(argv[head + 1], &end, 10);
              break;
            case ot_u64:
              *((u64*)opts[oi].data) = (u64)strtoul(argv[head + 1], &end, 10);
              break;
            case ot_s32:
              *((u32*)opts[oi].data) = (s32)atol(argv[head + 1]);
              break;
            case ot_s64:
              *((u64*)opts[oi].data) = (s64)atol(argv[head + 1]);
              break;
            case ot_string: strcpy((char*)opts[oi].data, argv[head + 1]); break;
            case ot_bool:
              *((bool*)opts[oi].data) = true;
              --head;
              break;
            case ot_btype: {
              btype* bound_type = boundary_types.get_strict(argv[head + 1]);

              if (bound_type)
                *((btype*)opts[oi].data) = *bound_type;
              else
                printf("unknown boundary type \"%s\"\n", argv[head + 1]);
            }
            break;
            case ot_rndtype: {
              output_type* output = output_types.get_strict(argv[head + 1]);

              if (output)
                *((output_type*)opts[oi].data) = *output;
              else
                printf("render type \"%s\" not found\n", argv[head + 1]);
            }
            break;
            case ot_time_scheme: {
              time_scheme* sch = multistage_schemes.get_strict(argv[head + 1]);

              if (sch)
                *((time_scheme*)opts[oi].data) = *sch;
              else
                printf("unknown time scheme \"%s\"\n", argv[head + 1]);
            }
            break;
          }
          head += 2;
          break;
        }
      }
      if (oi == optc)
      {
        printf("failed to find option number %d : %s\n", head, argv[head] + 1);
        ++head;
      }
    }
    else
    {
      // simply check the next argument
      printf("ignoring non-option argument %s\n", argv[head]);
      ++head;
    }
  }
  return 0;
}

option mkopt_prefix(const char* name, const char* description, void* data)
{
  option opt{};
  opt.name        = name;
  opt.description = description;
  opt.data        = data;
  return opt;
}

option mkopt(const char* name, const char* description, u32* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_u32;
  return opt;
}

option mkopt(const char* name, const char* description, u64* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_u64;
  return opt;
}

option mkopt(const char* name, const char* description, s32* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_s32;
  return opt;
}

option mkopt(const char* name, const char* description, s64* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_s64;
  return opt;
}

option mkopt(const char* name, const char* description, float* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_float;
  return opt;
}

option mkopt(const char* name, const char* description, double* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_double;
  return opt;
}

option mkopt(const char* name, const char* description, char* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_string;
  return opt;
}

option mkopt(const char* name, const char* description, bool* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_bool;
  return opt;
}

option mkopt(const char* name, const char* description, btype* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_btype;
  return opt;
}

option mkopt(const char* name, const char* description, output_type* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_rndtype;
  return opt;
}

option mkopt(const char* name, const char* description, time_scheme* data)
{
  option opt = mkopt_prefix(name, description, data);
  opt.type   = ot_time_scheme;
  return opt;
}
