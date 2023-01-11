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

#include "state.cpp"

void frame_buffer_resized_callback(GLFWwindow* this_window,
                                   int width,
                                   int height)
{
  frame_buffer_resized = true;
}

void scroll_callback(GLFWwindow* win, double xoffset, double yoffset)
{
  double old_cam_pos = cam_pos;
  cam_pos -= 0.1 * yoffset;
  if (cam_pos < 0.)
    cam_pos = old_cam_pos;
}

void cursor_position_callback(GLFWwindow* win, double xpos, double ypos)
{
  last_cursor_xpos = cursor_xpos;
  last_cursor_ypos = cursor_ypos;
  cursor_xpos      = xpos;
  cursor_ypos      = ypos;
  mouse_dx         = cursor_xpos - last_cursor_xpos;
  mouse_dy         = cursor_ypos - last_cursor_ypos;
}

void mouse_button_callback(GLFWwindow* win, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    mouse_left_pressed = true;
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    mouse_left_pressed = false;
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    mouse_right_pressed = true;
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    mouse_right_pressed = false;
}

void key_callback(GLFWwindow* win, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_A && action == GLFW_PRESS)
    ubo_modifier = true;
  if (key == GLFW_KEY_A && action == GLFW_RELEASE)
    ubo_modifier = false;

  if (key == GLFW_KEY_M && action == GLFW_PRESS)
    mesh_display_toggle_on = !mesh_display_toggle_on;

  if (key == GLFW_KEY_E && action == GLFW_PRESS)
    elem_display_toggle_on = !elem_display_toggle_on;
  if (key == GLFW_KEY_S && action == GLFW_PRESS)
    slice_display_toggle_on = !slice_display_toggle_on;
  if (key == GLFW_KEY_P && action == GLFW_PRESS)
    preview_display_toggle_on = !preview_display_toggle_on;

  if (key == GLFW_KEY_R && (mods & GLFW_MOD_SHIFT) && action == GLFW_PRESS)
  {
    if (resolution > 1)
    {
      --resolution;
      recompute_state = true;
    }
  }
  else if (key == GLFW_KEY_R && action == GLFW_PRESS)
  {
    ++resolution;
    recompute_state = true;
  }
}
