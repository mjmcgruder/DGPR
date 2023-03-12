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

#include "init.cpp"
#include "io.cpp"
#include "rendering.cpp"
#include "gameplay.cpp"

int main(int argc, char** argv)
{
  /* input parsing */

  real gamma             = 1.4;
  char ifile_prefix[256] = "output";
  bool print_vkfeatures  = false;

  const int optc       = 4;
  option optlist[optc] = {
  mkopt("N", "render resolution parameter", &resolution),
  mkopt("ifile", "input file prefix", ifile_prefix),
  mkopt("render", "output to render", &render_output),
  mkopt("features", "prints vulkan implementation features", &print_vkfeatures),
  };

  bool help = false;
  if (optparse(argc, argv, optc, optlist, help))
    return 1;
  if (help) return 0;

  /* read input file */

  char ifile[512];
  snprintf(ifile, 512, "%s.dg", ifile_prefix);

  u64 time_step = 0;
  core_geometry geom;
  read_state(ifile, time_step, geom, rendering_outputs);

  /* render */

  render_loop(geom, time_step, gamma, print_vkfeatures);

  /* exit */

  printf("\n");
  clean_global_resources();
  return 0;
}
