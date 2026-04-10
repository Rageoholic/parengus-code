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
    /// Per-binding descriptor indexing flags (e.g.
    /// `PARTIALLY_BOUND_EXT`). Leave empty (`DescriptorBindingFlags
    /// ::empty()`) when not using descriptor indexing.
    pub binding_flags: vk::DescriptorBindingFlags,
}

// ---------------------------------------------------------------------------
// DescriptorUpdateBuilder
// ---------------------------------------------------------------------------

/// Builder that accumulates descriptor writes and associated info
/// vectors so a single `vkUpdateDescriptorSets` call can be made.
///
/// Usage model:
/// 1. Create with `DescriptorUpdateBuilder::with_capacity(expected_writes, expected_total_infos)`.
/// 2. Call `push_write(...)` to declare a `VkWriteDescriptorSet` and get
///    back an index.
/// 3. Call `push_image_info`, `push_buffer_info`, or `push_texel_buffer_view`
///    to append infos for that write by index.
/// 4. Call `apply(device)` to perform a single `device.update_raw_descriptor_sets`.
pub struct DescriptorUpdateBuilder {
    image_infos: Vec<vk::DescriptorImageInfo>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    texel_buffer_views: Vec<vk::BufferView>,
    pending: Vec<PendingWrite>,
}

struct PendingWrite {
    dst_set: vk::DescriptorSet,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_type: vk::DescriptorType,

    image_start: Option<usize>,
    image_count: u32,

    buffer_start: Option<usize>,
    buffer_count: u32,

    texel_view_start: Option<usize>,
    texel_view_count: u32,
}

impl DescriptorUpdateBuilder {
    /// Create a builder with pre-allocated capacity.
    ///
    /// `expected_writes` is the anticipated number of `WriteDescriptorSet`
    /// entries. `expected_total_infos` is a soft hint for the total number
    /// of `DescriptorImageInfo` / `DescriptorBufferInfo` elements across
    /// all writes.
    pub fn with_capacity(
        expected_writes: usize,
        expected_total_infos: usize,
    ) -> Self {
        Self {
            image_infos: Vec::with_capacity(expected_total_infos),
            buffer_infos: Vec::with_capacity(expected_total_infos),
            texel_buffer_views: Vec::with_capacity(expected_total_infos),
            pending: Vec::with_capacity(expected_writes),
        }
    }

    /// Declare a write descriptor and return its index. Use the index to
    /// append infos for that write.
    #[inline]
    fn push_write(
        &mut self,
        dst_set: vk::DescriptorSet,
        dst_binding: u32,
        descriptor_type: vk::DescriptorType,
        dst_array_element: u32,
    ) -> usize {
        let pw = PendingWrite {
            dst_set,
            dst_binding,
            dst_array_element,
            descriptor_type,
            image_start: None,
            image_count: 0,
            buffer_start: None,
            buffer_count: 0,
            texel_view_start: None,
            texel_view_count: 0,
        };
        self.pending.push(pw);
        self.pending.len() - 1
    }

    /// Append an image info to the write at `write_index`.
    #[inline]
    fn push_image_info(
        &mut self,
        write_index: usize,
        info: vk::DescriptorImageInfo,
    ) {
        let start = self.image_infos.len();
        self.image_infos.push(info);
        let pw = &mut self.pending[write_index];
        if pw.image_start.is_none() {
            pw.image_start = Some(start);
            pw.image_count = 1;
        } else {
            pw.image_count += 1;
        }
    }

    /// Append a buffer info to the write at `write_index`.
    #[inline]
    fn push_buffer_info(
        &mut self,
        write_index: usize,
        info: vk::DescriptorBufferInfo,
    ) {
        let start = self.buffer_infos.len();
        self.buffer_infos.push(info);
        let pw = &mut self.pending[write_index];
        if pw.buffer_start.is_none() {
            pw.buffer_start = Some(start);
            pw.buffer_count = 1;
        } else {
            pw.buffer_count += 1;
        }
    }

    /// Append a texel buffer view to the write at `write_index`.
    #[inline]
    fn push_texel_buffer_view(
        &mut self,
        write_index: usize,
        view: vk::BufferView,
    ) {
        let start = self.texel_buffer_views.len();
        self.texel_buffer_views.push(view);
        let pw = &mut self.pending[write_index];
        if pw.texel_view_start.is_none() {
            pw.texel_view_start = Some(start);
            pw.texel_view_count = 1;
        } else {
            pw.texel_view_count += 1;
        }
    }

    /// Append a `Sampler` as a sampler-only descriptor to the write at
    /// `write_index`.
    #[inline]
    pub fn push_sampler(&mut self, write_index: usize, sampler: &Sampler) {
        self.push_image_info(write_index, sampler.descriptor_image_info());
    }

    /// Append a `Texture`'s default view to the write at `write_index` with
    /// the given `image_layout`.
    #[inline]
    pub fn push_texture(
        &mut self,
        write_index: usize,
        texture: &Texture,
        image_layout: vk::ImageLayout,
    ) {
        self.push_image_info(
            write_index,
            texture.descriptor_image_info(image_layout),
        );
    }

    /// Append a buffer descriptor for `buffer` to the write at `write_index`.
    #[inline]
    pub fn push_buffer<B: BufferHandle>(
        &mut self,
        write_index: usize,
        buffer: &B,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let info = buffer.descriptor_buffer_info(offset, range);
        self.push_buffer_info(write_index, info);
    }

    /// Declare a write and append the provided image info in one call.
    #[inline]
    pub fn push_write_image_info(
        &mut self,
        dst_set: vk::DescriptorSet,
        dst_binding: u32,
        descriptor_type: vk::DescriptorType,
        dst_array_element: u32,
        info: vk::DescriptorImageInfo,
    ) -> usize {
        let idx = self.push_write(
            dst_set,
            dst_binding,
            descriptor_type,
            dst_array_element,
        );
        self.push_image_info(idx, info);
        idx
    }

    /// Declare a write and append the provided buffer info in one call.
    #[inline]
    pub fn push_write_buffer_info(
        &mut self,
        dst_set: vk::DescriptorSet,
        dst_binding: u32,
        descriptor_type: vk::DescriptorType,
        dst_array_element: u32,
        info: vk::DescriptorBufferInfo,
    ) -> usize {
        let idx = self.push_write(
            dst_set,
            dst_binding,
            descriptor_type,
            dst_array_element,
        );
        self.push_buffer_info(idx, info);
        idx
    }

    /// Declare a write and append the provided texel buffer view in one call.
    #[inline]
    pub fn push_write_texel_buffer_view(
        &mut self,
        dst_set: vk::DescriptorSet,
        dst_binding: u32,
        descriptor_type: vk::DescriptorType,
        dst_array_element: u32,
        view: vk::BufferView,
    ) -> usize {
        let idx = self.push_write(
            dst_set,
            dst_binding,
            descriptor_type,
            dst_array_element,
        );
        self.push_texel_buffer_view(idx, view);
        idx
    }

    /// Apply all accumulated writes as a single `vkUpdateDescriptorSets`
    /// call against `device`.
    ///
    /// # Safety
    /// All handles and infos must be valid and remain alive for the
    /// duration described by the Vulkan spec (this mirrors
    /// `Device::update_raw_descriptor_sets` safety requirements).
    pub unsafe fn apply(&mut self, device: &Arc<Device>) {
        // Build the WriteDescriptorSet list referencing the owned info
        // arrays. Slices borrow from `self` and remain valid for the
        // duration of this call.
        let mut writes: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(self.pending.len());
        for pw in &self.pending {
            let mut w = vk::WriteDescriptorSet::default()
                .dst_set(pw.dst_set)
                .dst_binding(pw.dst_binding)
                .dst_array_element(pw.dst_array_element)
                .descriptor_type(pw.descriptor_type);

            if pw.image_count > 0 {
                let start = pw
                    .image_start
                    .expect("image_start set when image_count > 0");
                let slice =
                    &self.image_infos[start..start + pw.image_count as usize];
                w = w.image_info(slice);
            } else if pw.buffer_count > 0 {
                let start = pw
                    .buffer_start
                    .expect("buffer_start set when buffer_count > 0");
                let slice =
                    &self.buffer_infos[start..start + pw.buffer_count as usize];
                w = w.buffer_info(slice);
            } else if pw.texel_view_count > 0 {
                let start = pw
                    .texel_view_start
                    .expect("texel_view_start set when texel_view_count > 0");
                let slice = &self.texel_buffer_views
                    [start..start + pw.texel_view_count as usize];
                w = w.texel_buffer_view(slice);
            }

            writes.push(w);
        }

        // SAFETY: caller ensures handles and infos remain valid as
        // required by Vulkan. Forward to device helper.
        unsafe { device.update_raw_descriptor_sets(&writes, &[]) };
    }
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
        let mut create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        // If any binding requests per-binding flags (e.g.
        // PARTIALLY_BOUND_EXT), chain a binding flags create-info.
        // The flags vec must be one entry per binding in the same
        // order, with empty flags for bindings that don't need them.
        let any_flags = bindings.iter().any(|b| !b.binding_flags.is_empty());
        let binding_flags_vec: Vec<vk::DescriptorBindingFlags>;
        let mut binding_flags_info;
        if any_flags {
            binding_flags_vec =
                bindings.iter().map(|b| b.binding_flags).collect();
            binding_flags_info =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&binding_flags_vec);
            create_info = create_info.push_next(&mut binding_flags_info);
        }
        // If any binding uses UPDATE_AFTER_BIND, the layout itself must
        // declare UPDATE_AFTER_BIND_POOL_BIT so the pool can allocate
        // sets from it (VkDescriptorSetLayoutCreateInfo spec rule).
        let any_update_after_bind = bindings.iter().any(|b| {
            b.binding_flags
                .contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND)
        });
        if any_update_after_bind {
            create_info = create_info.flags(
                vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
            );
        }
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
        update_after_bind: bool,
        name: Option<&str>,
    ) -> Result<Self, vk::Result> {
        let mut pool_flags = vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        if update_after_bind {
            pool_flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
        }
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(pool_flags);
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
}
