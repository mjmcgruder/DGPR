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

// reads a file and returns a pointer to the contents
// NOTE!!! don't forget to free the memory for the returned file buffer!
// TODO: you should probably check if these commands actually succeed
char* read_binary(const char* address, u32& size)
{
  FILE* fID = fopen(address, "rb");
  if (fID == nullptr)
  { VKTERMINATE("reading of shader file %s failed!", address); }
  // find the buffer size
  size = 0;
  while (fgetc(fID) != EOF) { ++size; }
  rewind(fID);
  // make and fill the buffer
  char* buff = new char[size];
  if (fread(buff, sizeof(char), size, fID) != size)
  { VKTERMINATE("failed to correctly read shader file %s", address); }
  fclose(fID);
  return buff;
}

VkFormat find_supported_format(const VkFormat* candidates, u32 ncandidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features)
{
  VkFormat format;
  for (u32 i = 0; i < ncandidates; ++i)
  {
    format = candidates[i];
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features)
    {
      return format;
    }
    else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
             (props.optimalTilingFeatures & features) == features)
    {
      return format;
    }
  }
  VKTERMINATE("failed to find a format for depth buffering!");
  return format;
}

VkFormat find_depth_format()
{
  const u32 ncandidates            = 3;
  VkFormat candidates[ncandidates] = {VK_FORMAT_D32_SFLOAT,
                                      VK_FORMAT_D32_SFLOAT_S8_UINT,
                                      VK_FORMAT_D24_UNORM_S8_UINT};
  VkFormat depth_format =
  find_supported_format(candidates, ncandidates, VK_IMAGE_TILING_OPTIMAL,
                        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  return depth_format;
}

VkShaderModule make_shader_module(char* code, u32 size)
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = size;
  create_info.pCode    = (uint32_t*)code;
  VkShaderModule shader_module;
  VK_CHECK(vkCreateShaderModule(device, &create_info, nullptr, &shader_module),
           "shader module creation failed!");
  return shader_module;
}

void make_render_pass(VkAttachmentLoadOp color_load_op,
                      VkRenderPass& render_pass)
{
  VkAttachmentDescription color_attachment{};
  color_attachment.format         = swap_chain_image_format;
  color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp         = color_load_op;
  color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  if(color_load_op == VK_ATTACHMENT_LOAD_OP_LOAD)
    color_attachment.initialLayout  = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  else
    color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;

  color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depth_attachment{};
  depth_attachment.format         = find_depth_format();
  depth_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  if (color_load_op == VK_ATTACHMENT_LOAD_OP_LOAD)
    depth_attachment.initialLayout =
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  else
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  depth_attachment.finalLayout =
  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref{};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout =
  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount    = 1;
  subpass.pColorAttachments       = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;

  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;

  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  const u32 nattachments                            = 2;
  VkAttachmentDescription attachments[nattachments] = {color_attachment,
                                                       depth_attachment};
  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = nattachments;
  render_pass_info.pAttachments    = attachments;
  render_pass_info.subpassCount    = 1;
  render_pass_info.pSubpasses      = &subpass;
  render_pass_info.dependencyCount = 1;
  render_pass_info.pDependencies   = &dependency;

  VK_CHECK(vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass),
           "render pass creation failed!");
}

void make_graphics_pipeline(
VkRenderPass render_pass,
array<VkDescriptorSetLayout>& ubo_descriptor_set_layout,
VkPipelineLayout& pipeline_layout, VkShaderModule vert_shader_module,
VkShaderModule frag_shader_module, VkPipeline& graphics_pipeline)
{
  // shader stages
  VkPipelineShaderStageCreateInfo vert_create_info{};
  vert_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_create_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vert_create_info.module = vert_shader_module;
  vert_create_info.pName  = "main";
  VkPipelineShaderStageCreateInfo frag_create_info{};
  frag_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_create_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_create_info.module = frag_shader_module;
  frag_create_info.pName  = "main";
  VkPipelineShaderStageCreateInfo shader_stage_create_info[] = {
  vert_create_info, frag_create_info};

  // vertex input
  VkVertexInputAttributeDescription attribute_descriptions[3];
  vertex::attribute_descriptions(attribute_descriptions);
  VkVertexInputBindingDescription binding_description =
  vertex::binding_description();
  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_info.vertexBindingDescriptionCount   = 1;
  vertex_input_info.pVertexBindingDescriptions      = &binding_description;
  vertex_input_info.vertexAttributeDescriptionCount = 3;
  vertex_input_info.pVertexAttributeDescriptions    = attribute_descriptions;

  // input assembly
  VkPipelineInputAssemblyStateCreateInfo input_assembly_info{};
  input_assembly_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly_info.primitiveRestartEnable = VK_FALSE;

  // viewport (and scissor)
  VkViewport viewport{};
  viewport.x        = 0.f;
  viewport.y        = 0.f;
  viewport.width    = (float)swap_chain_extent.width;
  viewport.height   = (float)swap_chain_extent.height;
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swap_chain_extent;
  VkPipelineViewportStateCreateInfo viewport_state_info{};
  viewport_state_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state_info.viewportCount = 1;
  viewport_state_info.pViewports    = &viewport;
  viewport_state_info.scissorCount  = 1;
  viewport_state_info.pScissors     = &scissor;

  // rasterizer
  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth               = 1.f;
  rasterizer.cullMode                = VK_CULL_MODE_NONE;
  rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.f;
  rasterizer.depthBiasClamp          = 0.f;
  rasterizer.depthBiasSlopeFactor    = 0.f;

  // multisampling
  VkPipelineMultisampleStateCreateInfo multisample_info{};
  multisample_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_info.sampleShadingEnable   = VK_FALSE;
  multisample_info.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  multisample_info.minSampleShading      = 1.f;
  multisample_info.pSampleMask           = nullptr;
  multisample_info.alphaToCoverageEnable = VK_FALSE;
  multisample_info.alphaToOneEnable      = VK_FALSE;

  // blending
  VkPipelineColorBlendAttachmentState color_blend_attachment_info{};
  color_blend_attachment_info.colorWriteMask =
  VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment_info.blendEnable         = VK_FALSE;
  color_blend_attachment_info.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment_info.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment_info.colorBlendOp        = VK_BLEND_OP_ADD;
  color_blend_attachment_info.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment_info.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment_info.alphaBlendOp        = VK_BLEND_OP_ADD;
  VkPipelineColorBlendStateCreateInfo color_blending_info{};
  color_blending_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending_info.logicOpEnable     = VK_FALSE;
  color_blending_info.logicOp           = VK_LOGIC_OP_COPY;
  color_blending_info.attachmentCount   = 1;
  color_blending_info.pAttachments      = &color_blend_attachment_info;
  color_blending_info.blendConstants[0] = 0.f;
  color_blending_info.blendConstants[1] = 0.f;
  color_blending_info.blendConstants[2] = 0.f;
  color_blending_info.blendConstants[3] = 0.f;

  // dynamic state
  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                     VK_DYNAMIC_STATE_LINE_WIDTH};
  VkPipelineDynamicStateCreateInfo dynamic_state_info{};
  dynamic_state_info.sType =
  VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state_info.dynamicStateCount = 2;
  dynamic_state_info.pDynamicStates    = dynamic_states;

  // pipeline layout (for uniforms)
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount         = 2;
  pipeline_layout_info.pSetLayouts            = &ubo_descriptor_set_layout[0];
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges    = nullptr;
  VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                  &pipeline_layout),
           "pipeline layout creation failed!");

  // depth testing
  VkPipelineDepthStencilStateCreateInfo depth_stencil{};
  depth_stencil.sType =
  VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depth_stencil.depthTestEnable       = VK_TRUE;
  depth_stencil.depthWriteEnable      = VK_TRUE;
  depth_stencil.depthCompareOp        = VK_COMPARE_OP_LESS;
  depth_stencil.depthBoundsTestEnable = VK_FALSE;
  depth_stencil.minDepthBounds        = 0.0f;
  depth_stencil.maxDepthBounds        = 1.0f;
  depth_stencil.stencilTestEnable     = VK_FALSE;
  depth_stencil.front                 = {};
  depth_stencil.back                  = {};

  // finally... the graphics pipeline
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType      = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages    = shader_stage_create_info;
  pipeline_info.pVertexInputState   = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly_info;
  pipeline_info.pViewportState      = &viewport_state_info;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState   = &multisample_info;
  pipeline_info.pDepthStencilState  = &depth_stencil;
  pipeline_info.pColorBlendState    = &color_blending_info;
  pipeline_info.pDynamicState       = nullptr;
  pipeline_info.layout              = pipeline_layout;
  pipeline_info.renderPass          = render_pass;
  pipeline_info.subpass             = 0;
  pipeline_info.basePipelineHandle  = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex   = -1;
  VK_CHECK(
  vkCreateGraphicsPipelines(
  device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline),
  "graphics pipeline creation failed!");
}

/*
 * descriptor_set_layout -------------------------------------------------------
 */

struct descriptor_set_layout
{
  VkDescriptorSetLayout layout;
  array<VkDescriptorSetLayoutBinding> layout_bindings;

  // --

  descriptor_set_layout();
  descriptor_set_layout(u32 nbinding, VkDescriptorType descriptor_type,
                        VkShaderStageFlags stage_flags);

  descriptor_set_layout(descriptor_set_layout& oth) = delete;
  descriptor_set_layout& operator=(descriptor_set_layout& oth) = delete;

  descriptor_set_layout(descriptor_set_layout&& oth) noexcept;
  descriptor_set_layout& operator=(descriptor_set_layout&& oth) noexcept;

  ~descriptor_set_layout();
  
  // --

  void clean();
};

descriptor_set_layout::descriptor_set_layout(u32 nbinding,
                                             VkDescriptorType descriptor_type,
                                             VkShaderStageFlags stage_flags)
{
  layout_bindings = array<VkDescriptorSetLayoutBinding>(nbinding);

  for (u32 bi = 0; bi < nbinding; ++bi)
  {
    layout_bindings[bi].binding            = bi;
    layout_bindings[bi].descriptorType     = descriptor_type;
    layout_bindings[bi].descriptorCount    = 1;
    layout_bindings[bi].stageFlags         = stage_flags;
    layout_bindings[bi].pImmutableSamplers = nullptr;
  }

  VkDescriptorSetLayoutCreateInfo ci{};
  ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = nbinding;
  ci.pBindings    = &layout_bindings[0];
  VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout),
           "descriptor set layout creation failed!");
}

descriptor_set_layout::descriptor_set_layout() :
layout(VK_NULL_HANDLE), layout_bindings{}
{}

descriptor_set_layout::descriptor_set_layout(
descriptor_set_layout&& oth) noexcept :
layout(oth.layout), layout_bindings(std::move(oth.layout_bindings))
{
  oth.layout = VK_NULL_HANDLE;
}

descriptor_set_layout& descriptor_set_layout::operator=(
descriptor_set_layout&& oth) noexcept
{
  clean();

  layout          = oth.layout;
  layout_bindings = std::move(oth.layout_bindings);

  oth.layout = VK_NULL_HANDLE;

  return *this;
}

void descriptor_set_layout::clean()
{
  vkDestroyDescriptorSetLayout(device, layout, nullptr);
}

descriptor_set_layout::~descriptor_set_layout()
{
  clean();
}

/*
 * descriptor_set --------------------------------------------------------------
 */

struct descriptor_set
{
  descriptor_set_layout* layout;

  VkDescriptorPool dpool;
  VkDescriptorSet dset;

  // --

  descriptor_set();
  descriptor_set(descriptor_set_layout* _layout);

  descriptor_set(const descriptor_set& oth)            = delete;
  descriptor_set& operator=(const descriptor_set& oth) = delete;

  descriptor_set(descriptor_set&& oth) noexcept;
  descriptor_set& operator=(descriptor_set&& oth) noexcept;

  ~descriptor_set();

  // --

  void clean();
};

descriptor_set::descriptor_set(descriptor_set_layout* _layout) :
layout(_layout), dpool(VK_NULL_HANDLE), dset(VK_NULL_HANDLE)
{
  /* make the descriptor pool */

  VkDescriptorPoolSize pool_size{};
  pool_size.descriptorCount = layout->layout_bindings.len;
  pool_size.type            = layout->layout_bindings[0].descriptorType;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes    = &pool_size;
  pool_info.maxSets       = 1;

  VK_CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &dpool),
           "descriptor pool creation failed!");

  /* allocte the descriptor sets */

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = dpool;
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts        = &(layout->layout);

  VK_CHECK(vkAllocateDescriptorSets(device, &alloc_info, &dset),
           "descriptor set allocation failed!");
}

descriptor_set::descriptor_set() :
layout(nullptr),
dpool(VK_NULL_HANDLE),
dset(VK_NULL_HANDLE)
{}

void descriptor_set::clean()
{
  vkDestroyDescriptorPool(device, dpool, nullptr);
}

descriptor_set::descriptor_set(descriptor_set&& oth) noexcept :
layout(std::move(oth.layout)),
dpool(std::move(oth.dpool)),
dset(std::move(oth.dset))
{
  oth.layout = nullptr;
  oth.dpool  = VK_NULL_HANDLE;
  oth.dset   = VK_NULL_HANDLE;
}

descriptor_set& descriptor_set::operator=(descriptor_set&& oth) noexcept
{
  clean();

  layout = std::move(oth.layout);
  dpool  = std::move(oth.dpool);
  dset   = std::move(oth.dset);

  oth.layout = VK_NULL_HANDLE;
  oth.dpool  = VK_NULL_HANDLE;
  oth.dset   = VK_NULL_HANDLE;

  return *this;
}

descriptor_set::~descriptor_set()
{
  clean();
}

/*
 * graphics_pipeline -----------------------------------------------------------
 */

struct graphics_pipeline
{
  VkPipeline pipeline;
  VkRenderPass render_pass;

  VkShaderModule vert_shader_module;
  VkShaderModule frag_shader_module;

  array<VkFramebuffer> framebuffers;

  VkPipelineLayout layout;

  descriptor_set_layout scene_layout;
  descriptor_set_layout object_layout;

  // ---

  graphics_pipeline(string vertex_shader, string fragment_shader);

  graphics_pipeline(graphics_pipeline& oth)            = delete;
  graphics_pipeline& operator=(graphics_pipeline& oth) = delete;

  graphics_pipeline(graphics_pipeline&& oth) noexcept = delete;
  graphics_pipeline& operator=(graphics_pipeline&& oth) noexcept = delete;

  ~graphics_pipeline();

  // ---

  void clean_swapchain_dependencies();
  void update_swap_chain_dependencies(VkAttachmentLoadOp color_load_op);
};

graphics_pipeline::graphics_pipeline(string vertex_shader,
                                     string fragment_shader) :
pipeline(VK_NULL_HANDLE),
render_pass(VK_NULL_HANDLE),
vert_shader_module(VK_NULL_HANDLE),
frag_shader_module(VK_NULL_HANDLE),
framebuffers(),
layout(VK_NULL_HANDLE),
scene_layout(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
object_layout(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
{
  u32 szvert, szfrag;
  char* vert_shader_code = read_binary(vertex_shader.c_str(), szvert);
  char* frag_shader_code = read_binary(fragment_shader.c_str(), szfrag);

  vert_shader_module = make_shader_module(vert_shader_code, szvert);
  frag_shader_module = make_shader_module(frag_shader_code, szfrag);

  delete[] vert_shader_code;
  delete[] frag_shader_code;
}

graphics_pipeline::~graphics_pipeline()
{
  clean_swapchain_dependencies();
  vkDestroyShaderModule(device, vert_shader_module, nullptr);
  vkDestroyShaderModule(device, frag_shader_module, nullptr);
}

void graphics_pipeline::clean_swapchain_dependencies()
{
  for (u64 i = 0; i < framebuffers.len; ++i)
  {
    vkDestroyFramebuffer(device, framebuffers[i], nullptr);
  }
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, layout, nullptr);
  vkDestroyRenderPass(device, render_pass, nullptr);
}

void graphics_pipeline::update_swap_chain_dependencies(
VkAttachmentLoadOp color_load_op)
{
  clean_swapchain_dependencies();

  array<VkDescriptorSetLayout> layouts(2);
  layouts[0] = scene_layout.layout;
  layouts[1] = object_layout.layout;
  make_render_pass(color_load_op, render_pass);
  make_graphics_pipeline(render_pass, layouts, layout, vert_shader_module,
                         frag_shader_module, pipeline);

  framebuffers = array<VkFramebuffer>(num_swap_chain_images);
  for (u64 i = 0; i < framebuffers.len; ++i)
  {
    VkImageView attachments[] = {swap_chain_image_views[i], depth_image_view};

    VkFramebufferCreateInfo cinf{};
    cinf.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    cinf.renderPass      = render_pass;
    cinf.attachmentCount = 2;
    cinf.pAttachments    = attachments;
    cinf.width           = swap_chain_extent.width;
    cinf.height          = swap_chain_extent.height;
    cinf.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(device, &cinf, nullptr, &framebuffers[i]),
             "frame buffer creation failed!");
  }
}

/*
 * compute_pipeline ------------------------------------------------------------
 */

struct compute_pipeline
{
  VkPipeline pipeline;
  VkPipelineLayout pipeline_layout;
  VkCommandBuffer command_buffer;

  descriptor_set_layout dset_layout;
  descriptor_set dset;

  // ---

  compute_pipeline(string shader, u64 nargs);

  compute_pipeline(const compute_pipeline& oth) = delete;
  compute_pipeline& operator=(const compute_pipeline& oth) = delete;

  compute_pipeline(compute_pipeline&& oth) = delete;
  compute_pipeline& operator=(compute_pipeline&& oth) = delete;

  ~compute_pipeline();

  // ---

  void run(u32 gcx, u32 gcy, u32 gcz, u32 swap_chain_image = 0);
};

compute_pipeline::compute_pipeline(string shader, u64 nargs)
{
  /*
   * descriptor set creation ---------------------------------------------------
   */

  dset_layout = descriptor_set_layout(nargs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_SHADER_STAGE_COMPUTE_BIT);

  dset = descriptor_set(&dset_layout);

  /*
   * shader module creation ----------------------------------------------------
   */

  u32 shader_size;
  char* shader_code = read_binary(shader.c_str(), shader_size);

  VkShaderModule shader_module =
  make_shader_module(shader_code, shader_size);

  VkPipelineShaderStageCreateInfo shader_ci{};
  shader_ci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_ci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_ci.module = shader_module;
  shader_ci.pName  = "main";

  /*
   * pipeline layout creation --------------------------------------------------
   */

  VkPipelineLayoutCreateInfo pipeline_layout_ci{};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.setLayoutCount = 1;
  pipeline_layout_ci.pSetLayouts    = &dset_layout.layout;

  VK_CHECK(vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr,
                                  &pipeline_layout),
           "compute pipeline layout creation failed!");

  /*
   * pipeline creation ---------------------------------------------------------
   */

  VkComputePipelineCreateInfo pipeline_ci{};
  pipeline_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_ci.stage  = shader_ci;
  pipeline_ci.layout = pipeline_layout;

  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci,
                                    nullptr, &pipeline),
           "compute pipeline creation failed for %s!", shader.c_str());

  delete[] shader_code;
  vkDestroyShaderModule(device, shader_module, nullptr);
}

compute_pipeline::~compute_pipeline()
{
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
}

void compute_pipeline::run(u32 gcx, u32 gcy, u32 gcz, u32 swap_chain_image)
{
  /*
   * command buffer record -----------------------------------------------------
   */

  VkCommandBufferAllocateInfo command_buffer_ai{};
  command_buffer_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_ai.commandPool = compute_command_pool;
  command_buffer_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_ai.commandBufferCount = 1;
  VK_CHECK(
  vkAllocateCommandBuffers(device, &command_buffer_ai, &command_buffer),
  "compute command buffer allocation failed!");

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(command_buffer, &bi),
           "failed to begin compute command buffer!");

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline_layout, 0, 1, &dset.dset, 0, nullptr);

  vkCmdDispatch(command_buffer, gcx, gcy, gcz);

  VK_CHECK(vkEndCommandBuffer(command_buffer),
           "failed to end compute command buffer!");

  /*
   * command buffer run --------------------------------------------------------
   */

  VkSubmitInfo submit_info{};
  submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers    = &command_buffer;

  VkFence fence;
  VkFenceCreateInfo fence_ci{};
  fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device, &fence_ci, nullptr, &fence),
           "compute fence creation failed!");

  VK_CHECK(vkQueueSubmit(compute_queue, 1, &submit_info, fence),
           "compute queue submission failed!");

  VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
           "failure at compute wait for fences!");

  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, compute_command_pool, 1, &command_buffer);
}
