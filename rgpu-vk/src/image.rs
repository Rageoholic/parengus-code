//! GPU image types: [`DeviceLocalImage`] and [`ImageView`].
//!
//! [`DeviceLocalImage`] wraps a `VkImage` backed by `GpuOnly` memory.
//! Populate it from a [`HostVisibleBuffer`] using
//! [`upload_from_host_visible`](DeviceLocalImage::upload_from_host_visible),
//! which records the necessary layout transitions and copy into a provided
//! command buffer and optionally waits for completion.
//!
//! [`ImageView`] wraps a `VkImageView` tied to a [`DeviceLocalImage`].

use std::sync::Arc;

use ash::vk;
use gpu_allocator::{AllocationError, vulkan::Allocation};
use thiserror::Error;

use crate::buffer::HostVisibleBuffer;
use crate::command::CommandBufferHandle;
use crate::device::{Device, MemoryUsage};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CreateImageError {
    #[error("Vulkan error creating image: {0}")]
    CreateImage(vk::Result),

    #[error("GPU allocator error allocating memory: {0}")]
    AllocateMemory(AllocationError),

    #[error("Vulkan error binding image memory: {0}")]
    BindMemory(vk::Result),
}

#[derive(Debug, Error)]
pub enum UploadImageError {
    #[error("Vulkan error resetting upload command buffer: {0}")]
    ResetCommandBuffer(vk::Result),

    #[error("Vulkan error beginning upload command buffer: {0}")]
    BeginCommandBuffer(vk::Result),

    #[error("Vulkan error ending upload command buffer: {0}")]
    EndCommandBuffer(vk::Result),

    #[error("Queue submit failed: {0}")]
    QueueSubmit(vk::Result),

    #[error("Device wait idle failed after upload submit: {0}")]
    WaitIdle(vk::Result),
}

#[derive(Debug, Error)]
pub enum CreateImageViewError {
    #[error("Vulkan error creating image view: {0}")]
    Vulkan(vk::Result),
}

// ---------------------------------------------------------------------------
// AllocatedImage — private inner state
// ---------------------------------------------------------------------------

struct AllocatedImage {
    parent: Arc<Device>,
    handle: vk::Image,
    allocation: Option<Allocation>,
    extent: vk::Extent3D,
    format: vk::Format,
}

impl std::fmt::Debug for AllocatedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocatedImage")
            .field("handle", &self.handle)
            .field("extent", &self.extent)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl AllocatedImage {
    fn new(
        device: &Arc<Device>,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateImageError> {
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        // SAFETY: create_info is fully initialised and has no borrowed data.
        let handle = unsafe { device.create_raw_image(&create_info) }
            .map_err(CreateImageError::CreateImage)?;

        // SAFETY: handle is a valid image created from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name image {:?}: {e}", handle);
        }

        // SAFETY: handle is a valid image created from this device.
        let reqs =
            unsafe { device.get_raw_image_memory_requirements(handle) };
        let allocation_name = name.unwrap_or("image");
        let allocation = device
            .allocate_memory(allocation_name, reqs, MemoryUsage::GpuOnly, false)
            .map_err(|e| {
                // SAFETY: handle was created from this device and is not
                // bound to memory yet.
                unsafe { device.destroy_raw_image(handle) };
                CreateImageError::AllocateMemory(e)
            })?;

        // SAFETY: handle and allocation memory are valid and belong to this
        // device.
        let bind_result = unsafe {
            device.bind_raw_image_memory(
                handle,
                allocation.memory(),
                allocation.offset(),
            )
        };
        if let Err(e) = bind_result {
            let _ = device.free_memory(allocation);
            // SAFETY: handle is valid and owned by this scope.
            unsafe { device.destroy_raw_image(handle) };
            return Err(CreateImageError::BindMemory(e));
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
            allocation: Some(allocation),
            extent,
            format,
        })
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        tracing::debug!("Dropping image {:?}", self.handle);
        // SAFETY: handle was created from parent and is owned by this wrapper.
        unsafe { self.parent.destroy_raw_image(self.handle) };

        if let Some(allocation) = self.allocation.take()
            && let Err(e) = self.parent.free_memory(allocation)
        {
            tracing::error!("Failed to free GPU image allocation: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceLocalImage
// ---------------------------------------------------------------------------

/// A GPU-only image backed by `GpuOnly` memory.
///
/// Created with `OPTIMAL` tiling. Populate from a [`HostVisibleBuffer`]
/// using [`upload_from_host_visible`](Self::upload_from_host_visible).
#[derive(Debug)]
pub struct DeviceLocalImage {
    inner: AllocatedImage,
}

impl DeviceLocalImage {
    /// Create a 2-D device-local image.
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateImageError> {
        let extent = vk::Extent3D { width, height, depth: 1 };
        Ok(Self {
            inner: AllocatedImage::new(device, extent, format, usage, name)?,
        })
    }

    pub fn raw_image(&self) -> vk::Image {
        self.inner.handle
    }

    pub fn extent(&self) -> vk::Extent3D {
        self.inner.extent
    }

    pub fn format(&self) -> vk::Format {
        self.inner.format
    }

    pub fn parent(&self) -> &Arc<Device> {
        &self.inner.parent
    }

    /// Upload pixel data from a staging buffer into this image using a
    /// one-time submission.
    ///
    /// Records the full sequence into `command_buffer`:
    /// - barrier: `UNDEFINED` → `TRANSFER_DST_OPTIMAL`
    /// - `vkCmdCopyBufferToImage`
    /// - barrier: `TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`
    ///
    /// Then submits and, when `fence` is `None`, calls `wait_idle`.
    ///
    /// # Safety
    /// - `command_buffer` must be externally synchronized and valid for
    ///   reset/begin/record/end on the current thread.
    /// - `src` must be created with `TRANSFER_SRC` usage and contain exactly
    ///   `width * height * bytes_per_pixel` bytes of packed pixel data.
    /// - `self` must be created with at least `TRANSFER_DST | SAMPLED` usage.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution has completed.
    /// - If `fence` is `Some`, the caller must wait/reset it before reusing
    ///   the resources referenced by the copy.
    pub unsafe fn upload_from_host_visible(
        &self,
        command_buffer: &mut impl CommandBufferHandle,
        src: &HostVisibleBuffer,
        fence: Option<vk::Fence>,
    ) -> Result<(), UploadImageError> {
        let raw_cmd = command_buffer.raw_command_buffer();
        let device = &self.inner.parent;
        let image = self.inner.handle;
        let extent = self.inner.extent;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: caller guarantees command buffer is not pending.
        unsafe {
            device.reset_raw_command_buffer(
                raw_cmd,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(UploadImageError::ResetCommandBuffer)?;

        // SAFETY: caller guarantees command buffer is in initial state.
        unsafe { device.begin_raw_command_buffer(raw_cmd, &begin_info) }
            .map_err(UploadImageError::BeginCommandBuffer)?;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // Barrier: UNDEFINED → TRANSFER_DST_OPTIMAL
        let to_transfer = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(subresource_range);
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&to_transfer));
        // SAFETY: recording state; image handle is valid.
        unsafe { device.cmd_pipeline_barrier2(raw_cmd, &dep_info) };

        // Copy buffer → image
        let subresource_layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource_layers)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent);
        // SAFETY: recording state; src buffer and image are valid and
        // in the correct layouts.
        unsafe {
            device.cmd_copy_buffer_to_image(
                raw_cmd,
                src.raw_buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            )
        };

        // Barrier: TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL
        let to_shader = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(subresource_range);
        let dep_info_shader = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&to_shader));
        // SAFETY: recording state; image handle is valid.
        unsafe { device.cmd_pipeline_barrier2(raw_cmd, &dep_info_shader) };

        // SAFETY: caller guarantees recording state.
        unsafe { device.end_raw_command_buffer(raw_cmd) }
            .map_err(UploadImageError::EndCommandBuffer)?;

        let cmd_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(raw_cmd);
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_info));
        let submit_fence = fence.unwrap_or(vk::Fence::null());

        // SAFETY: command buffer is executable and references valid resources.
        unsafe {
            device.graphics_present_queue_submit2_raw_fence(
                std::slice::from_ref(&submit),
                submit_fence,
            )
        }
        .map_err(UploadImageError::QueueSubmit)?;

        if fence.is_none() {
            device.wait_idle().map_err(UploadImageError::WaitIdle)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImageView
// ---------------------------------------------------------------------------

/// An owned `VkImageView` for a [`DeviceLocalImage`].
pub struct ImageView {
    parent: Arc<Device>,
    handle: vk::ImageView,
}

impl std::fmt::Debug for ImageView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageView")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl ImageView {
    /// Create a 2-D colour image view for `image`.
    pub fn new(
        device: &Arc<Device>,
        image: &DeviceLocalImage,
        name: Option<&str>,
    ) -> Result<Self, CreateImageViewError> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image.raw_image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format())
            .subresource_range(subresource_range);

        // SAFETY: create_info references a valid image from the same device.
        let handle = unsafe { device.create_raw_image_view(&create_info) }
            .map_err(CreateImageViewError::Vulkan)?;

        // SAFETY: handle is a valid image view from this device.
        let name_result =
            unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name image view {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_image_view(&self) -> vk::ImageView {
        self.handle
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        tracing::debug!("Dropping image view {:?}", self.handle);
        // SAFETY: handle was created from parent and is owned by this
        // wrapper. No GPU work may still reference it.
        unsafe { self.parent.destroy_raw_image_view(self.handle) };
    }
}
