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

layout(set = 0, binding = 0) uniform uniform_buffer_object {
  mat4 model;
  mat4 view;
  mat4 proj;
  bvec4 render_mesh;
  bvec4 slicing;
} ubo;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 ref;

layout(location = 0) out vec4 out_color;

void main() {

  bool xlim = ref.x < 1e-2 || ref.x > (1. - 1e-2);
  bool ylim = ref.y < 1e-2 || ref.y > (1. - 1e-2);
  bool zlim = ref.z < 1e-2 || ref.z > (1. - 1e-2);
  if (ubo.render_mesh.x && ubo.slicing.x && (xlim || ylim || zlim))
  {
    out_color = vec4(vec3(1., 0., 0.), 1.);
  }
  else if (ubo.render_mesh.x && ((xlim && ylim) || (xlim && zlim) || 
                                 (ylim && xlim) || (ylim && zlim) || 
                                 (zlim && xlim) || (zlim && ylim)))
  {
    out_color = vec4(vec3(1., 0., 0.), 1.);
  }
  else
  {
    out_color = vec4(frag_color, 1.);
  }
}
