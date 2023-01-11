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

#include <cctype>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "vrtxgen.cpp"

#define MAX_COMMAND_SIZE 1024

// parsing ---------------------------------------------------------------------

// Copies the next word from the "source" buffer to the "word" buffer subject to
// maximum word size "max_word_size". This function returns a pointer to the
// location in the original source buffer just after the read word or where
// reading was forced to stop because the word buffer would overflow.
const char* next_word(const char* source, char* word, int max_word_size)
{
  int si = 0, wi = 0;  // source iterator, word iterator

  // clear leading whitespace
  while (isspace(source[si]))
    ++si;

  // copy next (whitespace delimited) word to word buffer if not overflowing
  while (!isspace(source[si]) && source[si] != '\0')
  {
    if (wi < max_word_size - 1)
    {
      word[wi] = source[si];
    }
    else
    {
      word[wi] = '\0';
      return source + si;
    }
    ++si;
    ++wi;
  }

  word[wi] = '\0';
  return source + si;
}

// options ---------------------------------------------------------------------

void execute_lims(const char* options)
{
  print_limits = true;
}

void execute_tstp(const char* options)
{
  print_time_step = true;
}

void execute_output(const char* options)
{
  char output_name[128];
  next_word(options, output_name, 128);

  output_type* output = output_types.get_strict(output_name);

  if (output)
  {
    render_output   = *output;
    recompute_state = true;
  }
  else
  {
    printf("unknown output: %s\n", output_name);
  }
}

void help_output()
{
  printf("Available outputs:\n");

  for (auto i = output_types.begin(); i != output_types.end(); ++i)
    printf("  %s\n", i.key().c_str());
}

void execute_state(const char* options)
{
  char state[128];
  next_word(options, state, 128);

  simstate* new_state = rendering_outputs.get_strict(state);
  if (new_state != nullptr)
  {
    render_state    = new_state;
    recompute_state = true;
  }
  else
    printf("State %s not available for rendering!\n", state);
}

void help_state()
{
  printf("Available states to render:\n");
  for (auto i = rendering_outputs.begin(); i != rendering_outputs.end(); ++i)
    printf("  %s\n", i.key().c_str());
}

void execute_lprobe(const char* options)
{
  v3 point_1;
  v3 point_2;
  u64 nsamp = 0;

  char buff[128];

  options   = next_word(options, buff, 128);
  point_1.x = atof(buff);
  options   = next_word(options, buff, 128);
  point_1.y = atof(buff);
  options   = next_word(options, buff, 128);
  point_1.z = atof(buff);

  options   = next_word(options, buff, 128);
  point_2.x = atof(buff);
  options   = next_word(options, buff, 128);
  point_2.y = atof(buff);
  options   = next_word(options, buff, 128);
  point_2.z = atof(buff);

  options = next_word(options, buff, 128);
  nsamp   = (u64)strtoul(buff, nullptr, 10);

  line_probe_point_1     = point_1;
  line_probe_point_2     = point_2;
  line_probe_num_samples = nsamp;
  execute_line_probe     = true;
}

void execute_cmap(const char* options)
{
  char cmap_string[256];
  next_word(options, cmap_string, 256);
  colormap* candidate_colormap = colormaps.get_strict(cmap_string);
  if (candidate_colormap)
  {
    global_colormap = candidate_colormap;
    recompute_state = true;
  }
  else
  {
    printf("Colormap \"%s\" not recognized\n", cmap_string);
  }
}

void help_cmap()
{
  printf("Available color maps:\n");
  for (auto i = colormaps.begin(); i != colormaps.end(); ++i)
    printf("  %s\n", i.key().c_str());
}

struct render_option
{
  const char* name;
  const char* usage;
  void (*execute)(const char* options);
  void (*help)();
};

const int num_render_options                     = 6;
render_option render_options[num_render_options] = {
{"lims", "", execute_lims, nullptr},
{"output", "output <output>", execute_output, help_output},
{"tstp", "", execute_tstp, nullptr},
{"state", "state <state name to render>", execute_state, help_state},
{"lprobe",
 "lprobe <x1> <y1> <z1> <x2> <y2> <z2> <num samples>",
 execute_lprobe,
 nullptr},
{"cmap", "cmap <color map name>", execute_cmap, help_cmap},
};

// command line ----------------------------------------------------------------

bool cli()
{
  struct timeval timeout;
  fd_set stdin_descriptor_set;
  char command[MAX_COMMAND_SIZE];

  timeout.tv_sec  = 0;
  timeout.tv_usec = 100;

  FD_ZERO(&stdin_descriptor_set);
  FD_SET(STDIN_FILENO, &stdin_descriptor_set);

  if (select(1, &stdin_descriptor_set, nullptr, nullptr, &timeout))
  {
    int command_len = read(STDIN_FILENO, command, MAX_COMMAND_SIZE - 1);
    if (command[command_len - 1] == '\n')
      --command_len;
    command[command_len] = '\0';

    // parse command

    char exec[MAX_COMMAND_SIZE];
    const char* options = next_word(command, exec, MAX_COMMAND_SIZE);

    // handle commands

    if (exec[0] == '\0')
      return true;

    if (strcmp(exec, "help") == 0)
    {
      char help_command[MAX_COMMAND_SIZE];
      next_word(options, help_command, MAX_COMMAND_SIZE);

      // possibly print a particular help message
      bool found_option = false;
      for (int oi = 0; oi < num_render_options; ++oi)
      {
        if (render_options[oi].help != nullptr &&
            strcmp(help_command, render_options[oi].name) == 0)
        {
          render_options[oi].help();
          found_option = true;
          break;
        }
      }

      // if no particular message requested, print the generic help
      if (!found_option)
      {
        for (int oi = 0; oi < num_render_options; ++oi)
        {
          char helpmsg[2048];
          snprintf(helpmsg, 2048, "%16s -- %s\n", render_options[oi].name,
                   render_options[oi].usage);
          int helpmsg_len = strlen(helpmsg);
          write(STDOUT_FILENO, helpmsg, helpmsg_len);
        }
      }
      return true;
    }

    bool found_option = false;
    for (int oi = 0; oi < num_render_options; ++oi)
    {
      if (strcmp(exec, render_options[oi].name) == 0)
      {
        render_options[oi].execute(options);
        found_option = true;
        break;
      }
    }

    if (!found_option)
    {
      char errmsg[2048];
      snprintf(errmsg, 2048, "unknown command: \"%s\"\n", exec);
      int errmsg_len = strlen(errmsg);
      write(STDOUT_FILENO, errmsg, errmsg_len);
    }

    return true;
  }
  return false;
}
