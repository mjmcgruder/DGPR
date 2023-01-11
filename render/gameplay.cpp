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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

#include "buffers.cpp"
#include "init.cpp"
#include "pipeline.cpp"
#include "cli.cpp"
#include "vrtxgen.cpp"

void move_scene(scene_transform& mvp)
{
  float screen_width  = swap_chain_extent.width;
  float screen_height = swap_chain_extent.height;
  mvp.proj = glm::perspective(glm::radians(45.f), screen_width / screen_height,
                              0.1f, 100.f);
  mvp.proj[1][1] *= -1.;  // makes things right-handed

  glm::vec3 center = glm::vec3(0.f, 0.f, 0.f);
  glm::vec3 eye    = glm::vec3(cam_pos, 0.f, 0.f);
  mvp.view         = glm::lookAt(eye, center, glm::vec3(0.f, 1.f, 0.f));
}

void move_object(glm::mat4& model, const scene_transform& scene)
{
  glm::mat4 model_inv = glm::inverse(scene.model * model);
  glm::mat4 view_inv  = glm::inverse(scene.view);

  glm::vec3 center = glm::vec3(0.f, 0.f, 0.f);
  glm::vec3 eye    = glm::vec3(cam_pos, 0.f, 0.f);

  // mouse motion vector in screen space
  glm::vec4 m_s(-mouse_dx, mouse_dy, 0.f, 0.f);
  float mouse_speed = glm::length(m_s);

  // no rotation if mouse didn't move
  if (mouse_speed > 0.f)
  {
    // find mouse vector in global space
    glm::vec4 m_g = view_inv * m_s;

    if (mouse_right_pressed)
    {
      glm::vec4 t_m = model_inv * m_g;
      model = glm::translate(model, -0.01f * glm::vec3(t_m));
    }

    // apply rotation to model
    if (mouse_left_pressed)
    {
      // find axis of rotation in global space
      glm::vec3 f   = normalize(center - eye);
      glm::vec3 a_g = cross(f, glm::vec3(m_g));

      // find axis of rotation in model space
      glm::vec4 a_m = model_inv * glm::vec4(a_g[0], a_g[1], a_g[2], 0.f);
      glm::vec3 current_translation = glm::vec3(model[3]);
      model =
      glm::rotate(model, mouse_speed * glm::radians(1.f), glm::vec3(a_m));
    }
  }
}

void game(list<entity>& scene_list, list<entity>& ui_list,
          vrtxgen_metadata& metadata, core_geometry& geom, real gamma,
          u64 time_step, uniform_set<scene_transform>& scene_ubo,
          u32 swap_chain_image_indx, graphics_pipeline& scene_pipeline,
          graphics_pipeline& ui_pipeline)
{
  object_transform identity_transform;
  bool recompute = false;
  v3 pp(0., 0., 0.), pn(0., 0., 1.);

  // Handle actions that require re-generating the render geometry.
  //   Since an entity's presence in the scene is determined by its toggle, 
  //   clearing the scene list causes all entities to be automatically
  //   re-computed with updated settings below.

  if (recompute_state)
  {
    recompute = true;

    metadata.compute_metadata_cpu(resolution, render_output, *render_state,
                                  geom, gamma);

    metadata.compute_metadata_gpu(resolution, geom);

    recompute_state = false;
  }

  // update reference mvp

  move_scene(scene_ubo.host_data);

  if (!ubo_modifier)
  {
    glm::mat4 model_update(1.f);
    move_object(model_update, scene_ubo.host_data);
    scene_ubo.host_data.model = scene_ubo.host_data.model * model_update;
  }

  scene_ubo.host_data.render_mesh = glm::bvec4(mesh_display_toggle_on);

  if (entity* axis = ui_list.get_strict("axis"))
  {
    axis->ubo.host_data.model = glm::inverse(scene_ubo.host_data.model);
    axis->ubo.host_data.view  = glm::inverse(scene_ubo.host_data.view);
    axis->ubo.host_data.view[3][0] -= 10.f;
    axis->ubo.host_data.view[3][1] -= 3.f;
    axis->ubo.host_data.view[3][2] += 3.f;

    glm::mat3 Ms(scene_ubo.host_data.model);
    glm::mat3 Vs(scene_ubo.host_data.view);

    glm::mat3 model_rotation = glm::inverse(Ms) * Vs * Ms;  // wow math works

    for (u64 c = 0; c < 3; ++c)
      for (u64 r = 0; r < 3; ++r)
        axis->ubo.host_data.model[c][r] = model_rotation[c][r];

    axis->ubo.map_to(swap_chain_image_indx);
  }

  // add / remove from scene ---------------------------------------------

  // element surfaces

  if (recompute)
  {
    if (entity* elems = scene_list.get_strict("elems"))
    {
      generate_elem_surface_vertices(*elems, swap_chain_image_indx, geom,
                                     *render_state, resolution, render_output,
                                     metadata, gamma, global_colormap);
    }
  }

  entity* elems_check = scene_list.get_strict("elems");
  if (elems_check == nullptr && elem_display_toggle_on)
  {
    entity& elems =
    scene_list.add("elems", entity(&scene_pipeline.object_layout));

    generate_elem_surface_vertices(elems, swap_chain_image_indx, geom,
                                   *render_state, resolution, render_output,
                                   metadata, gamma, global_colormap);
  }
  else if (elems_check != nullptr && !elem_display_toggle_on)
  {
    scene_list.remove("elems");
  }

  if (entity* elems = scene_list.get_strict("elems"))
  {
    elems->ubo.map_to(swap_chain_image_indx);
    elems->vertices.propagate_references(swap_chain_image_indx);
    elems->indices.propagate_references(swap_chain_image_indx);
  }

  // preview plane
  
  entity* preview_check = scene_list.get_strict("preview");
  if (preview_check == nullptr && preview_display_toggle_on)
  {
    vertex slice_preview_vertices[] = {
    {{0.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    };

    u32 slice_preview_indices[] = {0, 1, 3, 1, 2, 3};

    entity& preview =
    scene_list.add("preview", entity(&scene_pipeline.object_layout));

    preview.update_geometry(swap_chain_image_indx, slice_preview_vertices, 4,
                            slice_preview_indices, 6);
  }
  else if (preview_check != nullptr && !preview_display_toggle_on)
  {
    scene_list.remove("preview");
  }

  if (entity* preview = scene_list.get_strict("preview"))
  {
    if (ubo_modifier)
      move_object(preview->ubo.host_data.model, scene_ubo.host_data);

    glm::vec3 pp_glm(preview->ubo.host_data.model[3]);
    glm::vec3 pn_glm =
    preview->ubo.host_data.model * glm::vec4(0.f, 0.f, 1.f, 0.f);

    pp = v3(pp_glm[0], pp_glm[1], pp_glm[2]);
    pn = v3(pn_glm[0], pn_glm[1], pn_glm[2]);

    preview->ubo.map_to(swap_chain_image_indx);
    preview->vertices.propagate_references(swap_chain_image_indx);
    preview->indices.propagate_references(swap_chain_image_indx);
  }

  // slice

  if (recompute)
  {
    if (entity* slice = scene_list.get_strict("slice"))
    {
      generate_slice_vertices(*slice, swap_chain_image_indx, geom,
                              *render_state, resolution, pp, pn, render_output,
                              metadata, gamma, global_colormap);
    }
  }

  entity* slice_check = scene_list.get_strict("slice");
  if (slice_check == nullptr && slice_display_toggle_on)
  {
    entity& slice = scene_list.add(
    "slice",
    entity(&scene_pipeline.object_layout));

    generate_slice_vertices(slice, swap_chain_image_indx, geom, *render_state,
                            resolution, pp, pn, render_output, metadata, gamma,
                            global_colormap);
  }
  else if (slice_check != nullptr && !slice_display_toggle_on)
  {
    scene_ubo.host_data.slicing = glm::bvec4(false);
    scene_list.remove("slice");
  }

  if (entity* slice = scene_list.get_strict("slice"))
  {
    slice->ubo.map_to(swap_chain_image_indx);
    slice->vertices.propagate_references(swap_chain_image_indx);
    slice->indices.propagate_references(swap_chain_image_indx);
    scene_ubo.host_data.slicing = glm::bvec4(true);
  }

  mouse_dx = 0.;
  mouse_dy = 0.;

  scene_ubo.map_to(swap_chain_image_indx);

  // non-rendering tasks -------------------------------------------------------

  if (print_limits)
  {
    printf("output limits:\n");
    printf("max: %f\n", metadata.global_output_max);
    printf("min: %f\n", metadata.global_output_min);
    print_limits = false;
  }

  if (print_time_step)
  {
    printf("%" PRIu64 "\n", time_step);
    print_time_step = false;
  }

  if (execute_line_probe)
  {
    line_probe(geom, *render_state, line_probe_point_1, line_probe_point_2,
               line_probe_num_samples, resolution, render_output, metadata,
               gamma);
    execute_line_probe = false;
  }
}
