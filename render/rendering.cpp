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

#include "cli.cpp"
#include "vrtxgen.cpp"
#include "init.cpp"
#include "data_structures.cpp"
#include "pipeline.cpp"
#include "state.cpp"
#include "buffers.cpp"
#include "pipeline.cpp"
#include "gameplay.cpp"

void record_command_buffer(uniform<scene_transform>& scene_ubo,
                           list<entity>& scene_list, list<entity>& ui_list,
                           graphics_pipeline& scene_pipeline,
                           graphics_pipeline& ui_pipeline, u32 swap_chain_image,
                           VkCommandBuffer* command_buffer)
{
  const u32 nclear_values = 2;
  VkClearValue clear_values[nclear_values]{};
  clear_values[0].color        = {{0.0f, 0.0f, 0.0f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};
  VkDeviceSize offsets[]       = {0};

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pInheritanceInfo = nullptr;

  VK_CHECK(vkBeginCommandBuffer(*command_buffer, &begin_info),
           "failed to start a command buffer!");

  /* scene pass */

  VkRenderPassBeginInfo scene_pass_info{};
  scene_pass_info.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  scene_pass_info.renderPass  = scene_pipeline.render_pass;
  scene_pass_info.framebuffer = scene_pipeline.framebuffers[swap_chain_image];
  scene_pass_info.renderArea.offset = {0, 0};
  scene_pass_info.renderArea.extent = swap_chain_extent;
  scene_pass_info.clearValueCount   = nclear_values;
  scene_pass_info.pClearValues      = clear_values;

  vkCmdBeginRenderPass(*command_buffer, &scene_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    scene_pipeline.pipeline);

  for (auto i = scene_list.begin(); i != scene_list.end(); ++i)
  {
    entity& obj = i.val();

    vkCmdBindVertexBuffers(*command_buffer, 0, 1, &obj.vertices.buffer,
                           offsets);

    vkCmdBindIndexBuffer(*command_buffer, obj.indices.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            scene_pipeline.layout, 0, 1, &scene_ubo.dset.dset,
                            0, nullptr);

    vkCmdBindDescriptorSets(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            scene_pipeline.layout, 1, 1, &obj.ubo.dset.dset, 0,
                            nullptr);

    vkCmdDrawIndexed(*command_buffer, (u32)obj.indices.nelems, 1, 0, 0, 0);
  }

  vkCmdEndRenderPass(*command_buffer);

  /* ui pass */

  VkRenderPassBeginInfo ui_pass_info{};
  ui_pass_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  ui_pass_info.renderPass        = ui_pipeline.render_pass;
  ui_pass_info.framebuffer       = ui_pipeline.framebuffers[swap_chain_image];
  ui_pass_info.renderArea.offset = {0, 0};
  ui_pass_info.renderArea.extent = swap_chain_extent;
  ui_pass_info.clearValueCount   = nclear_values;
  ui_pass_info.pClearValues      = clear_values;

  vkCmdBeginRenderPass(*command_buffer, &ui_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    ui_pipeline.pipeline);

  for (auto i = ui_list.begin(); i != ui_list.end(); ++i)
  {
    entity& obj = i.val();

    vkCmdBindVertexBuffers(*command_buffer, 0, 1, &obj.vertices.buffer,
                           offsets);

    vkCmdBindIndexBuffer(*command_buffer, obj.indices.buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            ui_pipeline.layout, 0, 1, &scene_ubo.dset.dset, 0,
                            nullptr);

    vkCmdBindDescriptorSets(*command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            ui_pipeline.layout, 1, 1, &obj.ubo.dset.dset, 0,
                            nullptr);

    vkCmdDrawIndexed(*command_buffer, (u32)obj.indices.nelems, 1, 0, 0, 0);
  }

  vkCmdEndRenderPass(*command_buffer);

  //

  VK_CHECK(vkEndCommandBuffer(*command_buffer),
           "failed to end command buffer!");
}

template<typename T>
T clamp(T v, T min, T max) {
  if (v < min) {
    return min;
  } else if (max < v) {
    return max;
  } else {
    return v;
  }
}

void make_swap_chain() {

  VkSurfaceCapabilitiesKHR capabilities;
  VkSurfaceFormatKHR surf_format;
  VkPresentModeKHR present_mode;
  VkExtent2D extent;
  u32 image_count;

  /*
   * determine swap chain settings ---------------------------------------------
   */

  {
    u32 nformats;
    VkSurfaceFormatKHR* formats;
    u32 npresent_modes;
    VkPresentModeKHR* present_modes;

    // find surface capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                              &capabilities);

    // find surface formats
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &nformats,
                                         nullptr);
    formats = new VkSurfaceFormatKHR[nformats];
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &nformats,
                                         formats);

    // find present modes
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                              &npresent_modes, nullptr);
    present_modes = new VkPresentModeKHR[npresent_modes];
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface,
                                              &npresent_modes, present_modes);

    // ensure format and present mode support
    if (nformats == 0) { VKTERMINATE("no swap chain format support!"); }
    if (npresent_modes == 0) {
      VKTERMINATE("no swap chain present mode support!");
    }

    // choose surface format
    u32 sfi;
    for (sfi = 0; sfi < nformats; ++sfi) {
      if (formats[sfi].format == VK_FORMAT_B8G8R8A8_SRGB &&
          formats[sfi].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        surf_format = formats[sfi];
        break;
      }
    }
    if (sfi == nformats) { surf_format = formats[0]; }

    // choose present mode
    u32 pmi;
    for (pmi = 0; pmi < npresent_modes; ++pmi) {
      if (present_modes[pmi] == VK_PRESENT_MODE_FIFO_KHR) {
        present_mode = present_modes[pmi];
        break;
      }
    }
    if (pmi == npresent_modes) { present_mode = present_modes[0]; }

    // choose extent (the complexity here handles resolution scaling)
    if (capabilities.currentExtent.width != UINT32_MAX) {
      extent = capabilities.currentExtent;
    } else {
      int win_width, win_height;
      glfwGetFramebufferSize(window, &win_width, &win_height);
      VkExtent2D actual_extent = {(u32)win_width, (u32)win_height};

      actual_extent.width =
      clamp(actual_extent.width, capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
      actual_extent.height =
      clamp(actual_extent.height, capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

      extent = actual_extent;
    }

    // these'll be useful later
    swap_chain_image_format = surf_format.format;
    swap_chain_extent       = extent;

    // choose the number of swap chain images
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        image_count > capabilities.maxImageCount) {
      image_count = capabilities.maxImageCount;
    }

    delete[] formats;
    delete[] present_modes;
  }

  /*
   * create the swap chain -----------------------------------------------------
   */

  {
    u32 queue_fam_index_list[num_queue] = {queue_family_indices.graphics,
                                           queue_family_indices.presentation};
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface          = surface;
    create_info.minImageCount    = image_count;
    create_info.imageFormat      = surf_format.format;
    create_info.imageColorSpace  = surf_format.colorSpace;
    create_info.imageExtent      = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (queue_family_indices.graphics != queue_family_indices.presentation) {
      create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices   = queue_fam_index_list;
    } else {
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    create_info.preTransform   = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode    = present_mode;
    create_info.clipped        = VK_TRUE;
    create_info.oldSwapchain   = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(device, &create_info, nullptr, &swap_chain),
             "swap chain creation failed!");
  }

  /*
   * find swap chain images ----------------------------------------------------
   */

  {
    vkGetSwapchainImagesKHR(device, swap_chain, &num_swap_chain_images,
                            nullptr);
    swap_chain_images = new VkImage[num_swap_chain_images];
    vkGetSwapchainImagesKHR(device, swap_chain, &num_swap_chain_images,
                            swap_chain_images);
  }

  /*
   * make swap chain image views -----------------------------------------------
   */

  {
    swap_chain_image_views = new VkImageView[num_swap_chain_images];
    for (u64 i = 0; i < num_swap_chain_images; ++i) {
      VkImageViewCreateInfo create_info{};
      create_info.sType        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      create_info.image        = swap_chain_images[i];
      create_info.viewType     = VK_IMAGE_VIEW_TYPE_2D;
      create_info.format       = swap_chain_image_format;
      create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      create_info.subresourceRange.baseMipLevel   = 0;
      create_info.subresourceRange.levelCount     = 1;
      create_info.subresourceRange.baseArrayLayer = 0;
      create_info.subresourceRange.layerCount     = 1;

      VK_CHECK(vkCreateImageView(device, &create_info, nullptr,
                                 &swap_chain_image_views[i]),
               "image view #%d creation failed!", (int)i);
    }
  }
}

void make_swap_chain_dependencies(graphics_pipeline& scene_pipeline,
                                  graphics_pipeline& ui_pipeline)
{
  make_swap_chain();
  make_depth_resources();
  scene_pipeline.update_swap_chain_dependencies(VK_ATTACHMENT_LOAD_OP_CLEAR);
  ui_pipeline.update_swap_chain_dependencies(VK_ATTACHMENT_LOAD_OP_LOAD);
}

void remake_swap_chain(graphics_pipeline& scene_pipeline,
                       graphics_pipeline& ui_pipeline)
{
  // don't do anything if minimized
  int width = 0, height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  while (width == 0 || height == 0)
  {
    glfwGetFramebufferSize(window, &width, &height);
    glfwWaitEvents();
  }
  vkDeviceWaitIdle(device);

  clean_swap_chain();

  make_swap_chain_dependencies(scene_pipeline, ui_pipeline);
}

void render_loop(core_geometry& geom, u64 time_step, real gamma,
                 bool print_vkfeatures)
{
  /*
   * initialization ------------------------------------------------------------
   */

  vkinit(print_vkfeatures);

  list<entity> scene_list;
  list<entity> ui_list;

  graphics_pipeline scene_pipeline;
  graphics_pipeline ui_pipeline;

  make_swap_chain_dependencies(scene_pipeline, ui_pipeline);

  uniform<scene_transform> scene_ubo(&scene_pipeline.scene_layout);

  vrtxgen_metadata metadata;

  render_state = rendering_outputs.get_strict("state");

  {
    vertex vertices[] = {
    {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{0.f, 1.f, 0.f}, {0.f, 1.f, 0.f}, {0.5f, 0.5f, 0.5f}},
    {{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}, {0.5f, 0.5f, 0.5f}},
    };
    u32 indices[] = {0, 2, 1, 1, 2, 3, 0, 3, 2, 0, 1, 3};

    entity& axis = ui_list.add("axis", entity(&ui_pipeline.object_layout));
    axis.update_geometry(vertices, 4, indices, 12);
  }

  /*
   * render synchronization objects --------------------------------------------
   */

  VkCommandBuffer command_buffer        = VK_NULL_HANDLE;
  VkFence render_in_progress            = VK_NULL_HANDLE;
  VkSemaphore image_available_semaphore = VK_NULL_HANDLE;
  VkSemaphore render_finished_semaphore = VK_NULL_HANDLE;

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &render_in_progress),
           "render fence creation failed!");

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr,
                             &image_available_semaphore),
           "image available semaphore creation failed!");
  VK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr,
                             &render_finished_semaphore),
           "render finished semaphone creatino failed!");

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = graphics_command_pool;
  alloc_info.level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = (uint32_t)1;
  VK_CHECK(vkAllocateCommandBuffers(device, &alloc_info, &command_buffer),
           "render command buffer creation failed!");

  /*
   * render loop -------------------------------------------------------------
   */

  VkPipelineStageFlags wait_stages[] = {
  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  write(STDOUT_FILENO, "$> ", 3);
  while (!glfwWindowShouldClose(window))
  {
    u32 swap_chain_image_indx;

    // process input

    glfwPollEvents();
    if (cli()) write(STDOUT_FILENO, "$> ", 3);

    // check for window resize

    VkSurfaceCapabilitiesKHR surface_capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface,
                                                       &surface_capabilities),
             "failed to acquire surfce capabilies to check for resizing!");

    if (surface_capabilities.currentExtent.width != swap_chain_extent.width ||
        surface_capabilities.currentExtent.height != swap_chain_extent.height)
    {
      remake_swap_chain(scene_pipeline, ui_pipeline);
      continue;
    }

    // generate geometry

    game(scene_list, ui_list, metadata, geom, gamma, time_step, scene_ubo,
         scene_pipeline, ui_pipeline);

    // reset / determine frame assets

    vkResetFences(device, 1, &render_in_progress);
    vkResetCommandBuffer(command_buffer, 0);

    VK_CHECK_SUBOPTIMAL(
    vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX,
                          image_available_semaphore, VK_NULL_HANDLE,
                          &swap_chain_image_indx),
    "failed to determine the swap chain image for this frame!");

    // record render command buffer

    record_command_buffer(scene_ubo, scene_list, ui_list, scene_pipeline,
                          ui_pipeline, swap_chain_image_indx, &command_buffer);

    // submit command buffer

    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &image_available_semaphore;
    si.pWaitDstStageMask    = wait_stages;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &command_buffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &render_finished_semaphore;
    VK_CHECK(vkQueueSubmit(graphics_queue, 1, &si, render_in_progress),
             "submission to draw command buffer failed!");

    // present

    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &render_finished_semaphore;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swap_chain;
    pi.pImageIndices      = &swap_chain_image_indx;
    pi.pResults           = nullptr;
    VK_CHECK_SUBOPTIMAL(vkQueuePresentKHR(present_queue, &pi),
                        "presentation failed!");

    // ensure rendering is finished before continuing to the next frame
    //   (this is essential in this program because you might be re-generating
    //   geometry every frame you can't afford to use several times the memory
    //   for large entities to avoid data races with the cpu and gpu are out
    //   of sync)
    vkWaitForFences(device, 1, &render_in_progress, VK_TRUE, UINT64_MAX);
  }

  vkDeviceWaitIdle(device);

  vkDestroyFence(device, render_in_progress, nullptr);
  vkDestroySemaphore(device, render_finished_semaphore, nullptr);
  vkDestroySemaphore(device, image_available_semaphore, nullptr);
  vkFreeCommandBuffers(device, graphics_command_pool, (uint32_t)1,
                       &command_buffer);
}
