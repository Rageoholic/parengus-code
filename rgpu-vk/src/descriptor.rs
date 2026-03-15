//! Descriptor set layout, pool, and set wrappers.
//!
//! [`DescriptorSetLayout`] describes the binding slots within a
//! descriptor set. [`DescriptorPool`] allocates descriptor sets from a
//! fixed-size pool. [`DescriptorSet`] is a typed handle to an allocated
//! set; its lifetime is managed by its parent pool.

use std::sync::Arc;

use ash::vk;

use crate::buffer::BufferHandle;
use crate::device::Device;
use crate::image::Texture;
use crate::sampler::Sampler;

// ---------------------------------------------------------------------------
// DescriptorBindingDesc
// ---------------------------------------------------------------------------

/// Describes a single binding within a descriptor set layout.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorBindingDesc {
    /// Binding slot index used by the shader.
    pub binding: u32,
    /// Type of descriptor at this binding.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors in this binding (array length).
    pub count: u32,
    /// Shader stages that can access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

impl From<DescriptorBindingDesc> for vk::DescriptorSetLayoutBinding<'static> {
    #[inline]
    fn from(b: DescriptorBindingDesc) -> Self {
        vk::DescriptorSetLayoutBinding::default()
            .binding(b.binding)
            .descriptor_type(b.descriptor_type)
            .descriptor_count(b.count)
            .stage_flags(b.stage_flags)
    }
}

// ---------------------------------------------------------------------------
// DescriptorSetLayout
// ---------------------------------------------------------------------------

/// An owned wrapper around a `VkDescriptorSetLayout`.
pub struct DescriptorSetLayout {
    parent: Arc<Device>,
    handle: vk::DescriptorSetLayout,
}

impl std::fmt::Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSetLayout")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl DescriptorSetLayout {
    /// Create a descriptor set layout from a slice of binding
    /// descriptions.
    pub fn new(
        device: &Arc<Device>,
        bindings: &[DescriptorBindingDesc],
        name: Option<&str>,
    ) -> Result<Self, vk::Result> {
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> =
            bindings.iter().copied().map(Into::into).collect();
        let create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        // SAFETY: create_info references valid binding descriptions
        // for the duration of this call.
        let handle =
            unsafe { device.create_raw_descriptor_set_layout(&create_info) }?;
        // SAFETY: handle is a valid descriptor set layout from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!(
                "Failed to name descriptor set layout {:?}: {e}",
                handle
            );
        }
        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    #[inline]
    pub fn raw_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.handle
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        tracing::debug!("Dropping descriptor set layout {:?}", self.handle);
        // SAFETY: handle was created from parent and is being
        // destroyed during teardown. No descriptor pool that used
        // this layout may still be alive.
        unsafe { self.parent.destroy_raw_descriptor_set_layout(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// DescriptorPool
// ---------------------------------------------------------------------------

/// An owned wrapper around a `VkDescriptorPool`.
///
/// Allocates [`DescriptorSet`] handles. Each set holds an
/// [`Arc`] back to its parent pool, so the pool is kept alive for
/// at least as long as any set allocated from it. The pool is freed
/// when the last [`Arc`] referencing it is dropped.
///
/// `VkDescriptorPool` requires external synchronization for all
/// allocation and free operations; this type is `!Sync` so that a
/// shared `&DescriptorPool` cannot be obtained across threads.
pub struct DescriptorPool {
    parent: Arc<Device>,
    handle: vk::DescriptorPool,
    /// `vkAllocateDescriptorSets` and `vkFreeDescriptorSets` require
    /// external synchronization of the pool; `!Sync` prevents sharing
    /// `&DescriptorPool` across threads.
    _not_sync: crate::marker::PhantomUnsync,
}

impl std::fmt::Debug for DescriptorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorPool")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl DescriptorPool {
    /// Create a descriptor pool.
    ///
    /// `max_sets` is the total number of descriptor sets that may be
    /// allocated from this pool. `pool_sizes` specifies the capacity
    /// per descriptor type.
    pub fn new(
        device: &Arc<Device>,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
        name: Option<&str>,
    ) -> Result<Self, vk::Result> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        // SAFETY: create_info is valid and references only stack data.
        let handle =
            unsafe { device.create_raw_descriptor_pool(&create_info) }?;
        // SAFETY: handle is a valid descriptor pool from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name descriptor pool {:?}: {e}", handle);
        }
        Ok(Self {
            parent: Arc::clone(device),
            handle,
            _not_sync: crate::marker::PhantomUnsync::default(),
        })
    }

    /// Allocate one descriptor set per provided layout.
    ///
    /// # Safety
    /// The caller must ensure that this pool outlives all descriptor
    /// sets allocated from it. Descriptor sets become invalid when
    /// their pool is reset or destroyed.
    pub unsafe fn allocate_sets(
        &self,
        layouts: &[&DescriptorSetLayout],
    ) -> Result<Vec<DescriptorSet>, vk::Result> {
        let raw_layouts: Vec<vk::DescriptorSetLayout> = layouts
            .iter()
            .map(|l| l.raw_descriptor_set_layout())
            .collect();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(&raw_layouts);
        // SAFETY: alloc_info references a valid pool and valid
        // layouts, all created from self.parent.
        let raw_sets =
            unsafe { self.parent.allocate_raw_descriptor_sets(&alloc_info) }?;
        Ok(raw_sets
            .into_iter()
            .map(|handle| DescriptorSet { handle })
            .collect())
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        tracing::debug!("Dropping descriptor pool {:?}", self.handle);
        // SAFETY: handle was created from parent and is being
        // destroyed during teardown. All in-flight GPU work
        // referencing descriptor sets from this pool must be
        // complete before drop.
        unsafe { self.parent.destroy_raw_descriptor_pool(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// DescriptorSet
// ---------------------------------------------------------------------------

/// A descriptor set allocated from a [`DescriptorPool`].
///
/// The pool must outlive all descriptor sets allocated from it;
/// this contract is documented on [`DescriptorPool::allocate_sets`].
pub struct DescriptorSet {
    handle: vk::DescriptorSet,
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSet")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl DescriptorSet {
    #[inline]
    pub fn raw_descriptor_set(&self) -> vk::DescriptorSet {
        self.handle
    }

    /// Assign a debug name to this descriptor set.
    ///
    /// The name is visible in validation layer output and GPU
    /// debuggers. Naming is best-effort; failures are logged and
    /// ignored.
    pub fn set_name(&self, device: &Arc<Device>, name: Option<&str>) {
        // SAFETY: handle is a valid descriptor set from device.
        let name_result =
            unsafe { device.set_object_name_str(self.handle, name) };
        if let Err(e) = name_result {
            tracing::warn!(
                "Failed to name descriptor set {:?}: {e}",
                self.handle
            );
        }
    }

    /// Update this descriptor set's binding with a combined image sampler.
    ///
    /// # Safety
    /// - `image_view` must be a valid `VkImageView` created from `device`.
    /// - `sampler` must be a valid `VkSampler` created from `device`.
    /// - `image_layout` must be the layout the image will be in when
    ///   shaders access it (typically `SHADER_READ_ONLY_OPTIMAL`).
    /// - Both handles must remain valid for as long as this descriptor
    ///   set is bound in any submitted command buffer.
    pub unsafe fn write_combined_image_sampler(
        &self,
        device: &Arc<Device>,
        binding: u32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .sampler(sampler)
            .image_layout(image_layout);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));
        // SAFETY: Caller guarantees device, image_view, sampler, and
        // image_layout validity.
        unsafe {
            device.update_raw_descriptor_sets(std::slice::from_ref(&write), &[])
        }
    }

    /// Update this descriptor set's binding with a texture and sampler.
    ///
    /// The image layout is assumed to be `SHADER_READ_ONLY_OPTIMAL`,
    /// which is the layout left by [`Texture::record_copy_from`].
    ///
    /// # Safety
    /// - `texture` and `sampler` must remain alive for as long as this
    ///   descriptor set is bound in any submitted command buffer.
    pub unsafe fn write_texture_sampler(
        &self,
        device: &Arc<Device>,
        binding: u32,
        texture: &Texture,
        sampler: &Sampler,
    ) {
        // SAFETY: caller guarantees texture and sampler outlive the
        // descriptor set binding.
        unsafe {
            self.write_combined_image_sampler(
                device,
                binding,
                texture.raw_image_view(),
                sampler.raw_sampler(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
        }
    }

    /// Update this descriptor set's binding with a uniform buffer.
    ///
    /// # Safety
    /// - `buffer` must be a valid buffer created from `device` with
    ///   `UNIFORM_BUFFER` usage.
    /// - `range` must not exceed the buffer's size.
    /// - The buffer must remain valid for as long as this descriptor
    ///   set is bound in any submitted command buffer.
    pub unsafe fn write_uniform_buffer<B: BufferHandle>(
        &self,
        device: &Arc<Device>,
        binding: u32,
        buffer: &B,
        range: vk::DeviceSize,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.raw_buffer())
            .offset(0)
            .range(range);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.handle)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        // SAFETY: Caller guarantees device, buffer, and range
        // validity.
        unsafe {
            device.update_raw_descriptor_sets(std::slice::from_ref(&write), &[])
        }
    }
}
