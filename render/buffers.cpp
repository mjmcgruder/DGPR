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
#include <utility>

#include "pipeline.cpp"
#include "state.cpp"

struct scene_transform
{
  alignas(16) glm::mat4 model        = glm::mat4(1.f);
  alignas(16) glm::mat4 view         = glm::mat4(1.f);
  alignas(16) glm::mat4 proj         = glm::mat4(1.f);
  alignas(16) glm::bvec4 render_mesh = glm::bvec4(false);
  alignas(16) glm::bvec4 slicing     = glm::bvec4(false);
};

struct object_transform
{
  alignas(16) glm::mat4 model = glm::mat4(1.f);
  alignas(16) glm::mat4 view  = glm::mat4(1.f);
};

u32 find_memory_type(u32 type_filter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
  for (u32 i = 0; i < mem_properties.memoryTypeCount; ++i)
  {
    if ((type_filter & (1 << i)) &&
        ((mem_properties.memoryTypes[i].propertyFlags & properties) ==
         properties))
    {
      return i;
    }
  }
  VKTERMINATE("failed to find a suitable memory type!");
  return 1;
}

void make_image(u32 width, u32 height, VkFormat format, VkImageTiling tiling,
                VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                VkImage& image, VkDeviceMemory& image_memory)
{
  VkImageCreateInfo cinf{};
  cinf.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  cinf.imageType     = VK_IMAGE_TYPE_2D;
  cinf.extent.width  = width;
  cinf.extent.height = height;
  cinf.extent.depth  = 1;
  cinf.mipLevels     = 1;
  cinf.arrayLayers   = 1;
  cinf.format        = format;
  cinf.tiling        = tiling;
  cinf.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  cinf.usage         = usage;
  cinf.samples       = VK_SAMPLE_COUNT_1_BIT;
  cinf.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateImage(device, &cinf, nullptr, &image),
           "image creation failed!");

  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(device, image, &mem_req);

  VkMemoryAllocateInfo ainf{};
  ainf.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ainf.allocationSize  = mem_req.size;
  ainf.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, properties);
  VK_CHECK(vkAllocateMemory(device, &ainf, nullptr, &image_memory),
           "failed to allocate image memory!");

  vkBindImageMemory(device, image, image_memory, 0);
}

VkImageView make_image_view(VkImage image, VkFormat format,
                            VkImageAspectFlags aspect_flags)
{
  VkImageView image_view;

  VkImageViewCreateInfo cinf{};
  cinf.sType                         = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  cinf.image                         = image;
  cinf.viewType                      = VK_IMAGE_VIEW_TYPE_2D;
  cinf.format                        = format;
  cinf.subresourceRange.aspectMask   = aspect_flags;
  cinf.subresourceRange.baseMipLevel = 0;
  cinf.subresourceRange.levelCount   = 1;
  cinf.subresourceRange.baseArrayLayer = 0;
  cinf.subresourceRange.layerCount     = 1;
  VK_CHECK(vkCreateImageView(device, &cinf, nullptr, &image_view),
           "failed to create image view!");

  return image_view;
}

bool has_stencil_component(VkFormat format)
{
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void make_depth_resources()
{
  VkFormat depth_format = find_depth_format();

  make_image(
  swap_chain_extent.width, swap_chain_extent.height, depth_format,
  VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depth_image, depth_image_memory);

  depth_image_view =
  make_image_view(depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags properties, VkBuffer& buffer,
                 VkDeviceMemory& buffer_memory)
{

  VkBufferCreateInfo buffer_create_info{};
  buffer_create_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size        = size;
  buffer_create_info.usage       = usage;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer),
           "buffer creation failed!");

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
  find_memory_type(mem_requirements.memoryTypeBits, properties);
  VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &buffer_memory),
           "failed to allocate buffer memory!");

  vkBindBufferMemory(device, buffer, buffer_memory, 0);
}

void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size)
{
  // allocating the command buffer
  VkCommandBuffer command_buffer;
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = transfer_command_pool;
  alloc_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

  // record command buffer

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size      = size;
  vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

  vkEndCommandBuffer(command_buffer);

  // execute command buffer to transfer data

  VkSubmitInfo submit_info{};
  submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers    = &command_buffer;
  vkQueueSubmit(transfer_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(transfer_queue);

  vkFreeCommandBuffers(device, transfer_command_pool, 1, &command_buffer);
}

/* ---------- */
/* buffer_set --------------------------------------------------------------- */
/* ---------- */

template<typename T>
struct buffer_set
{
  enum memory_residency : u32
  {
    memloc_host   = 0,
    memloc_device = 1,
  };

  VkMemoryPropertyFlags property_flags[2] = {
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};

  // ---

  array<VkBuffer> buffer;
  array<VkDeviceMemory> memory;
  array<u32> nelems;

  descriptor_set* descset;
  u32 binding;

  VkBufferUsageFlags usage_flags;
  memory_residency memory_locale;

  u64 last_updated_image;
  array<bool> up_to_date;

  // ---

  buffer_set();
  buffer_set(descriptor_set* _descset, u32 _binding,
             VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
             memory_residency locale  = memloc_device);

  buffer_set(buffer_set& oth)            = delete;
  buffer_set& operator=(buffer_set& oth) = delete;

  buffer_set(buffer_set&& oth) noexcept;
  buffer_set& operator=(buffer_set&& oth) noexcept;

  ~buffer_set();

  // ---

  void clear_buffer(u64 swap_chain_image);
  void clear();

  void update(T* input_data, u64 count, u64 swap_chain_image = 0);
  void retrieve(T* output, u64 swap_chain_image = 0);

  void update_descriptor_set_reference(u32 swap_chain_image);
  void propagate_references(u64 swap_chain_image);
};

template<typename T>
buffer_set<T>::buffer_set(descriptor_set* _descset, u32 _binding,
                          VkBufferUsageFlags usage, memory_residency locale)
{
  buffer     = array<VkBuffer>(num_swap_chain_images);
  memory     = array<VkDeviceMemory>(num_swap_chain_images);
  nelems     = array<u32>(num_swap_chain_images);
  up_to_date = array<bool>(num_swap_chain_images);
  descset    = _descset;
  binding    = _binding;

  usage_flags   = usage;
  memory_locale = locale;

  last_updated_image = 0;
}

template<typename T>
void buffer_set<T>::update(T* input_data, u64 count, u64 swap_chain_image)
{
  VkDeviceSize buffer_size = count * sizeof(*input_data);

  bool buffer_outdated = false;
  if (count != nelems[swap_chain_image])
    buffer_outdated = true;

  switch (memory_locale)
  {
    case memloc_device: 
    {
      if (buffer_outdated)
      {
        clear_buffer(swap_chain_image);

        make_buffer(buffer_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | usage_flags,
                    property_flags[memory_locale], buffer[swap_chain_image],
                    memory[swap_chain_image]);
        nelems[swap_chain_image] = count;
      }

      VkBuffer staging_buffer              = VK_NULL_HANDLE;
      VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;

      make_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging_buffer, staging_buffer_memory);

      void* data;
      vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
      memcpy(data, input_data, (size_t)buffer_size);
      vkUnmapMemory(device, staging_buffer_memory);

      copy_buffer(staging_buffer, buffer[swap_chain_image], buffer_size);

      vkDestroyBuffer(device, staging_buffer, nullptr);
      vkFreeMemory(device, staging_buffer_memory, nullptr);
    }
    break;
    case memloc_host: 
    {
      if (buffer_outdated)
      {
        clear_buffer(swap_chain_image);

        make_buffer(buffer_size, usage_flags, property_flags[memory_locale],
                    buffer[swap_chain_image], memory[swap_chain_image]);
        nelems[swap_chain_image] = count;
      }

      void* data;
      vkMapMemory(device, memory[swap_chain_image], 0, buffer_size, 0, &data);
      memcpy(data, input_data, buffer_size);
      vkUnmapMemory(device, memory[swap_chain_image]);
    }
    break;
  }

  if (descset)
  {
    update_descriptor_set_reference(swap_chain_image);
  }

  last_updated_image = swap_chain_image;
  if (buffer_outdated)
  {
    for (u64 i = 0; i < num_swap_chain_images; ++i)
      up_to_date[i] = false;
  }
  up_to_date[swap_chain_image] = true;
}

template<typename T>
void buffer_set<T>::retrieve(T* output, u64 swap_chain_image)
{
  u64 buffer_size = nelems[swap_chain_image] * sizeof(*output);

  VkBuffer staging_buffer              = VK_NULL_HANDLE;
  VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;

  make_buffer(
  buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  staging_buffer, staging_buffer_memory);

  copy_buffer(buffer[swap_chain_image], staging_buffer, buffer_size);

  void* data;
  vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
  memcpy(output, data, (size_t)buffer_size);
  vkUnmapMemory(device, staging_buffer_memory);

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
}

template<typename T>
void buffer_set<T>::update_descriptor_set_reference(u32 swap_chain_image)
{
  VkDeviceSize buffer_size = nelems[swap_chain_image] * sizeof(T);

  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = buffer[swap_chain_image];
  buffer_info.offset = 0;
  buffer_info.range  = buffer_size;

  VkWriteDescriptorSet descriptor_write{};
  descriptor_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptor_write.dstSet          = descset->descriptor_sets[swap_chain_image];
  descriptor_write.dstBinding      = binding;
  descriptor_write.dstArrayElement = 0;
  descriptor_write.descriptorType =
  descset->layout->layout_bindings[binding].descriptorType;
  descriptor_write.descriptorCount  = 1;
  descriptor_write.pBufferInfo      = &buffer_info;
  descriptor_write.pImageInfo       = nullptr;
  descriptor_write.pTexelBufferView = nullptr;

  vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, nullptr);
}

template<typename T>
void buffer_set<T>::propagate_references(u64 swap_chain_image)
{
  if (!up_to_date[swap_chain_image])
  {
    if (buffer[swap_chain_image] != VK_NULL_HANDLE)
    {
      clear_buffer(swap_chain_image);
    }

    buffer[swap_chain_image] = buffer[last_updated_image];
    memory[swap_chain_image] = memory[last_updated_image];
    nelems[swap_chain_image] = nelems[last_updated_image];

    if (descset)
    {
      update_descriptor_set_reference(swap_chain_image);
    }

    up_to_date[swap_chain_image] = true;
  }
}

template<typename T>
void buffer_set<T>::clear_buffer(u64 swap_chain_image)
{
  VkBuffer current_buffer = buffer[swap_chain_image];
  vkDestroyBuffer(device, buffer[swap_chain_image], nullptr);
  vkFreeMemory(device, memory[swap_chain_image], nullptr);
  for (u64 i = 0; i < buffer.len; ++i)
  {
    if (buffer[i] == current_buffer)
    {
      buffer[i] = VK_NULL_HANDLE;
      memory[i] = VK_NULL_HANDLE;
    }
  }
}

template<typename T>
void buffer_set<T>::clear()
{
  for (u64 i = 0; i < buffer.len; ++i)
    clear_buffer(i);
}

template<typename T>
buffer_set<T>::~buffer_set()
{
  clear();
}

template<typename T>
buffer_set<T>::buffer_set() :
buffer{},
memory{},
nelems{},
descset(nullptr),
binding(0),
usage_flags(0),
memory_locale(memloc_device),
last_updated_image(0),
up_to_date()
{}

template<typename T>
buffer_set<T>::buffer_set(buffer_set&& oth) noexcept :
buffer(std::move(oth.buffer)),
memory(std::move(oth.memory)),
nelems(std::move(oth.nelems)),
descset(oth.descset),
binding(oth.binding),
usage_flags(oth.usage_flags),
memory_locale(oth.memory_locale),
last_updated_image(oth.last_updated_image),
up_to_date(std::move(oth.up_to_date))
{
  oth.descset            = nullptr;
  oth.binding            = 0;
  oth.usage_flags        = 0;
  oth.memory_locale      = memloc_device;
  oth.last_updated_image = 0;
}

template<typename T>
buffer_set<T>& buffer_set<T>::operator=(buffer_set&& oth) noexcept
{
  clear();

  buffer             = std::move(oth.buffer);
  memory             = std::move(oth.memory);
  nelems             = std::move(oth.nelems);
  descset            = oth.descset;
  binding            = oth.binding;
  usage_flags        = oth.usage_flags;
  memory_locale      = oth.memory_locale;
  last_updated_image = oth.last_updated_image;
  up_to_date         = std::move(oth.up_to_date);

  oth.descset            = nullptr;
  oth.binding            = 0;
  oth.usage_flags        = 0;
  oth.memory_locale      = memloc_device;
  oth.last_updated_image = 0;

  return *this;
}

/* ----------- */
/* uniform set -------------------------------------------------------------- */
/* ----------- */

template<typename T>
struct uniform_set
{
  T host_data;

  descriptor_set dset;
  buffer_set<T> buffers;

  // --

  uniform_set();

  uniform_set(uniform_set& oth) = delete;
  uniform_set& operator=(uniform_set& oth) = delete;

  uniform_set(uniform_set&& oth) noexcept;
  uniform_set& operator=(uniform_set&& oth) noexcept;

  uniform_set(descriptor_set_layout* dset_layout);

  // --

  void map_to(u32 swap_chain_image);
};

template<typename T>
uniform_set<T>::uniform_set(descriptor_set_layout* dset_layout) :
host_data{},
dset{dset_layout},
buffers(&dset, 0, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        buffer_set<T>::memloc_device)
{}

template<typename T>
void uniform_set<T>::map_to(u32 image)
{
  buffers.update(&host_data, 1, image);
}

template<typename T>
uniform_set<T>::uniform_set() :
host_data{},
dset{},
buffers{}
{}

template<typename T>
uniform_set<T>::uniform_set(uniform_set&& oth) noexcept :
host_data(std::move(oth.host_data)),
dset(std::move(oth.dset)), 
buffers(std::move(oth.buffers))
{
  buffers.descset = &dset; 
}

template<typename T>
uniform_set<T>& uniform_set<T>::operator=(uniform_set&& oth) noexcept
{
  host_data = std::move(oth.host_data);
  dset      = std::move(oth.dset);
  buffers   = std::move(oth.buffers);

  buffers.descset = &dset; 

  return *this;
}

/* --------------------- */
/* entity implementation ---------------------------------------------------- */
/* --------------------- */

struct entity
{
  buffer_set<vertex> vertices;
  buffer_set<u32> indices;
  uniform_set<object_transform> ubo;

  descriptor_set vertex_dset;
  descriptor_set index_dset;

  // ---

  entity();

  entity(entity& oth) = delete;
  entity& operator=(entity& oth) = delete;

  entity(entity&& oth) noexcept;
  entity& operator=(entity&& oth) noexcept;

  entity(descriptor_set_layout* uniform_layout,
         descriptor_set_layout* vertex_layout,
         descriptor_set_layout* index_layout);

  // ---

  void update_uniform(u64 swap_chain_image);
  void update_geometry(u64 swap_chain_image, vertex* vertex_data,
                       u64 vertex_len, u32* index_data, u64 index_len);
};

entity::entity() : vertices{}, indices{}, ubo{}, vertex_dset{}, index_dset{}
{}

entity::entity(entity&& oth) noexcept :
vertices(std::move(oth.vertices)),
indices(std::move(oth.indices)),
ubo(std::move(oth.ubo)),
vertex_dset(std::move(oth.vertex_dset)),
index_dset(std::move(oth.index_dset))
{
  if (vertices.descset)
    vertices.descset = &vertex_dset;

  if (indices.descset)
    indices.descset = &index_dset;
}

entity& entity::operator=(entity&& oth) noexcept
{
  vertices    = std::move(oth.vertices);
  indices     = std::move(oth.indices);
  ubo         = std::move(oth.ubo);
  vertex_dset = std::move(oth.vertex_dset);
  index_dset  = std::move(oth.index_dset);

  if (vertices.descset)
    vertices.descset = &vertex_dset;

  if (indices.descset)
    indices.descset = &index_dset;

  return *this;
}

entity::entity(descriptor_set_layout* uniform_layout,
               descriptor_set_layout* vertex_layout = nullptr,
               descriptor_set_layout* index_layout  = nullptr)
{
  descriptor_set* pvertex_dset = nullptr;
  descriptor_set* pindex_dset  = nullptr;

  if (vertex_layout)
  {
    vertex_dset  = descriptor_set(vertex_layout);
    pvertex_dset = &vertex_dset;
  }

  if (index_layout)
  {
    index_dset  = descriptor_set(index_layout);
    pindex_dset = &index_dset;
  }

  vertices = buffer_set<vertex>(
  pvertex_dset, 0,
  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
  buffer_set<vertex>::memloc_device);

  indices = buffer_set<u32>(
  pindex_dset, 0,
  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
  buffer_set<u32>::memloc_device);

  ubo = uniform_set<object_transform>{uniform_layout};
}

void entity::update_uniform(u64 swap_chain_image)
{
  ubo.map_to(swap_chain_image);
}

void entity::update_geometry(u64 swap_chain_image, vertex* vertex_data,
                             u64 vertex_len, u32* index_data, u64 index_len)
{
  vertices.update(vertex_data, vertex_len, swap_chain_image);
  indices.update(index_data, index_len, swap_chain_image);
}
