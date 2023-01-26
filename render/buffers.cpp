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

/* ------- */
/* dbuffer ------------------------------------------------------------------ */
/* ------- */

template<typename T>
struct dbuffer
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

  VkBuffer buffer;
  VkDeviceMemory memory;
  u32 nelems;

  descriptor_set* descset;
  u32 binding;

  VkBufferUsageFlags usage_flags;
  memory_residency memory_locale;

  // ---

  dbuffer();
  dbuffer(descriptor_set* _descset, u32 _binding,
          VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          memory_residency locale  = memloc_device);

  dbuffer(dbuffer& oth) = delete;
  dbuffer& operator=(dbuffer& oth) = delete;

  dbuffer(dbuffer&& oth) noexcept;
  dbuffer& operator=(dbuffer&& oth) noexcept;

  ~dbuffer();

  // ---

  void allocate(u64 count);
  void send(T* input_data, u64 count);
  void retrieve(T* output);
  void update(T* input_data, u64 count);
  void bind(descriptor_set* _descset, u32 _binding);

  void update_dset();

  void clear();
};

template<typename T>
void dbuffer<T>::update_dset()
{
  VkDeviceSize buffer_size = nelems * sizeof(T);

  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = buffer;
  buffer_info.offset = 0;
  buffer_info.range  = buffer_size;

  VkWriteDescriptorSet descriptor_write{};
  descriptor_write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptor_write.dstSet          = descset->dset;
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
dbuffer<T>::dbuffer(descriptor_set* _descset, u32 _binding,
                    VkBufferUsageFlags usage, memory_residency locale)
{
  buffer  = VK_NULL_HANDLE;
  memory  = VK_NULL_HANDLE;
  nelems  = 0;
  descset = _descset;
  binding = _binding;

  usage_flags   = usage;
  memory_locale = locale;
}

template<typename T>
void dbuffer<T>::bind(descriptor_set* _descset, u32 _binding)
{
  descset = _descset;
  binding = _binding;
  update_dset();
}

template<typename T>
void dbuffer<T>::allocate(u64 count)
{
  VkDeviceSize buffer_size = count * sizeof(T);

  bool buffer_outdated = false;
  if (count != nelems)
    buffer_outdated = true;

  switch (memory_locale)
  {
    case memloc_device: {
      if (buffer_outdated)
      {
        clear();

        make_buffer(buffer_size,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | usage_flags,
                    property_flags[memory_locale], buffer, memory);
        nelems = count;
      }
    }
    break;
    case memloc_host: {
      if (buffer_outdated)
      {
        clear();

        make_buffer(buffer_size, usage_flags, property_flags[memory_locale],
                    buffer, memory);
        nelems = count;
      }
    }
    break;
  }

  if (descset && buffer_outdated)
  {
    update_dset();
  }
}

template<typename T>
void dbuffer<T>::send(T* input_data, u64 count)
{
  VkDeviceSize buffer_size = count * sizeof(*input_data);

  switch (memory_locale)
  {
    case memloc_device: {
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

      copy_buffer(staging_buffer, buffer, buffer_size);

      vkDestroyBuffer(device, staging_buffer, nullptr);
      vkFreeMemory(device, staging_buffer_memory, nullptr);
    }
    break;
    case memloc_host: {
      void* data;
      vkMapMemory(device, memory, 0, buffer_size, 0, &data);
      memcpy(data, input_data, buffer_size);
      vkUnmapMemory(device, memory);
    }
    break;
  }
}

template<typename T>
void dbuffer<T>::retrieve(T* output)
{
  u64 buffer_size = nelems * sizeof(*output);

  VkBuffer staging_buffer              = VK_NULL_HANDLE;
  VkDeviceMemory staging_buffer_memory = VK_NULL_HANDLE;

  make_buffer(
  buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  staging_buffer, staging_buffer_memory);

  copy_buffer(buffer, staging_buffer, buffer_size);

  void* data;
  vkMapMemory(device, staging_buffer_memory, 0, buffer_size, 0, &data);
  memcpy(output, data, (size_t)buffer_size);
  vkUnmapMemory(device, staging_buffer_memory);

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_buffer_memory, nullptr);
}

template<typename T>
void dbuffer<T>::update(T* input_data, u64 count)
{
  allocate(count);
  send(input_data, count);
}

template<typename T>
void dbuffer<T>::clear()
{
  vkDestroyBuffer(device, buffer, nullptr);
  vkFreeMemory(device, memory, nullptr);
}

template<typename T>
dbuffer<T>::~dbuffer()
{
  clear();
}

template<typename T>
dbuffer<T>::dbuffer() :
buffer(VK_NULL_HANDLE),
memory(VK_NULL_HANDLE),
nelems(0),
descset(nullptr),
binding(0),
usage_flags(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
memory_locale(memloc_device)
{}

template<typename T>
dbuffer<T>::dbuffer(dbuffer&& oth) noexcept :
buffer(std::move(oth.buffer)),
memory(std::move(oth.memory)),
nelems(std::move(oth.nelems)),
descset(std::move(oth.descset)),
binding(std::move(oth.binding)),
usage_flags(std::move(oth.usage_flags)),
memory_locale(std::move(oth.memory_locale))
{
  oth.buffer        = VK_NULL_HANDLE;
  oth.memory        = VK_NULL_HANDLE;
  oth.nelems        = 0;
  oth.descset       = nullptr;
  oth.binding       = 0;
  oth.usage_flags   = 0;
  oth.memory_locale = memloc_device;
}

template<typename T>
dbuffer<T>& dbuffer<T>::operator=(dbuffer&& oth) noexcept
{
  clear();

  buffer             = std::move(oth.buffer);
  memory             = std::move(oth.memory);
  nelems             = std::move(oth.nelems);
  descset            = std::move(oth.descset);
  binding            = std::move(oth.binding);
  usage_flags        = std::move(oth.usage_flags);
  memory_locale      = std::move(oth.memory_locale);

  oth.buffer        = VK_NULL_HANDLE;
  oth.memory        = VK_NULL_HANDLE;
  oth.nelems        = 0;
  oth.descset       = nullptr;
  oth.binding       = 0;
  oth.usage_flags   = 0;
  oth.memory_locale = memloc_device;

  return *this;
}

/* ------- */
/* uniform ------------------------------------------------------------------ */
/* ------- */

template<typename T>
struct uniform
{
  T host_data;

  descriptor_set dset;
  dbuffer<T> buffers;

  // --

  uniform();

  uniform(uniform& oth) = delete;
  uniform& operator=(uniform& oth) = delete;

  uniform(uniform&& oth) noexcept;
  uniform& operator=(uniform&& oth) noexcept;

  uniform(descriptor_set_layout* dset_layout);

  // --

  void map();
};

template<typename T>
uniform<T>::uniform(descriptor_set_layout* dset_layout) :
host_data{},
dset{dset_layout},
buffers(&dset, 0, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, dbuffer<T>::memloc_device)
{}

template<typename T>
void uniform<T>::map()
{
  buffers.update(&host_data, 1);
}

template<typename T>
uniform<T>::uniform() :
host_data{},
dset{},
buffers{}
{}

template<typename T>
uniform<T>::uniform(uniform&& oth) noexcept :
host_data(std::move(oth.host_data)),
dset(std::move(oth.dset)), 
buffers(std::move(oth.buffers))
{
  buffers.descset = &dset; 
}

template<typename T>
uniform<T>& uniform<T>::operator=(uniform&& oth) noexcept
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
  dbuffer<vertex> vertices;
  dbuffer<u32> indices;
  uniform<object_transform> ubo;

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

  void update_uniform();
  void update_geometry(vertex* vertex_data, u64 vertex_len, u32* index_data,
                       u64 index_len);
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

  vertices = dbuffer<vertex>(
  pvertex_dset, 0,
  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
  dbuffer<vertex>::memloc_device);

  indices = dbuffer<u32>(
  pindex_dset, 0,
  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
  dbuffer<u32>::memloc_device);

  ubo = uniform<object_transform>{uniform_layout};
}

void entity::update_uniform()
{
  ubo.map();
}

void entity::update_geometry(vertex* vertex_data, u64 vertex_len,
                             u32* index_data, u64 index_len)
{
  vertices.update(vertex_data, vertex_len);
  indices.update(index_data, index_len);
}
