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

layout(set = 0, binding = 0) uniform scene_buffer {
  mat4 model;
  mat4 view;
  mat4 proj;
  bvec4 render_mesh;
  bvec4 slicing;
} scene_ubo;

layout(set = 1, binding = 0) uniform object_buffer {
  mat4 model;
  mat4 view;
} object_ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_ref;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 out_ref;

void main() {
  gl_Position = scene_ubo.proj   * 
                (scene_ubo.view  * object_ubo.view)  * 
                (scene_ubo.model * object_ubo.model) * vec4(in_position, 1.0);
  frag_color  = in_color;
  out_ref     = in_ref;
}
