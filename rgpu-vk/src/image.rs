//! GPU image types.
//!
//! The public-facing types are [`Texture`] (colour images uploaded
//! from a staging buffer) and [`DepthImage`] (depth render targets).
//! [`DeviceLocalImage`] and [`ImageView`] are `pub(crate)` building
//! blocks.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::{AllocationError, vulkan::Allocation};
use thiserror::Error;

use crate::buffer::HostVisibleBuffer;
use crate::command::ResettableCommandBuffer;
use crate::device::{Device, MemoryUsage};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub(crate) enum CreateImageError {
    #[error("Vulkan error creating image: {0}")]
    CreateImage(vk::Result),

    #[error("GPU allocator error allocating memory: {0}")]
    AllocateMemory(AllocationError),

    #[error("Vulkan error binding image memory: {0}")]
    BindMemory(vk::Result),
}

#[derive(Debug, Error)]
pub(crate) enum CreateImageViewError {
    #[error("Vulkan error creating image view: {0}")]
    Vulkan(vk::Result),
}

/// Error returned by [`Texture::record_copy_from`].
#[derive(Debug, Error)]
pub enum RecordCopyFromError {
    #[error(
        "Staging buffer ({actual} bytes) too small; \
         image requires {required} bytes"
    )]
    StagingTooSmall { required: u64, actual: u64 },

    #[error("Cannot determine texel size for format {0:?}")]
    UnknownFormat(vk::Format),
}

/// Error returned by [`Texture::new`].
#[derive(Debug, Error)]
pub enum CreateTextureError {
    #[error("Vulkan error creating image: {0}")]
    CreateImage(vk::Result),

    #[error("GPU allocator error allocating memory: {0}")]
    AllocateMemory(AllocationError),

    #[error("Vulkan error binding image memory: {0}")]
    BindMemory(vk::Result),

    #[error("Vulkan error creating image view: {0}")]
    CreateView(vk::Result),
}

impl From<CreateImageError> for CreateTextureError {
    fn from(e: CreateImageError) -> Self {
        match e {
            CreateImageError::CreateImage(r) => Self::CreateImage(r),
            CreateImageError::AllocateMemory(e) => Self::AllocateMemory(e),
            CreateImageError::BindMemory(r) => Self::BindMemory(r),
        }
    }
}

impl From<CreateImageViewError> for CreateTextureError {
    fn from(e: CreateImageViewError) -> Self {
        match e {
            CreateImageViewError::Vulkan(r) => Self::CreateView(r),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns bytes per texel for packed (non-compressed, non-planar)
/// formats with an unambiguous single-aspect buffer layout, or
/// `None` for block-compressed formats, planar YCbCr formats,
/// and combined depth-stencil formats where the size is
/// aspect-dependent.
fn format_texel_size(format: vk::Format) -> Option<u32> {
    match format {
        // 1 byte
        vk::Format::R4G4_UNORM_PACK8
        | vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_USCALED
        | vk::Format::R8_SSCALED
        | vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8_SRGB
        | vk::Format::S8_UINT
        | vk::Format::A8_UNORM_KHR => Some(1),

        // 2 bytes
        vk::Format::R4G4B4A4_UNORM_PACK16
        | vk::Format::B4G4R4A4_UNORM_PACK16
        | vk::Format::A4R4G4B4_UNORM_PACK16
        | vk::Format::A4B4G4R4_UNORM_PACK16
        | vk::Format::R5G6B5_UNORM_PACK16
        | vk::Format::B5G6R5_UNORM_PACK16
        | vk::Format::R5G5B5A1_UNORM_PACK16
        | vk::Format::B5G5R5A1_UNORM_PACK16
        | vk::Format::A1R5G5B5_UNORM_PACK16
        | vk::Format::A1B5G5R5_UNORM_PACK16_KHR
        | vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_USCALED
        | vk::Format::R8G8_SSCALED
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_SRGB
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_USCALED
        | vk::Format::R16_SSCALED
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R10X6_UNORM_PACK16
        | vk::Format::R12X4_UNORM_PACK16
        | vk::Format::D16_UNORM => Some(2),

        // 3 bytes
        vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_USCALED
        | vk::Format::R8G8B8_SSCALED
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8_SRGB
        | vk::Format::B8G8R8_UNORM
        | vk::Format::B8G8R8_SNORM
        | vk::Format::B8G8R8_USCALED
        | vk::Format::B8G8R8_SSCALED
        | vk::Format::B8G8R8_UINT
        | vk::Format::B8G8R8_SINT
        | vk::Format::B8G8R8_SRGB => Some(3),

        // 4 bytes
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_USCALED
        | vk::Format::R8G8B8A8_SSCALED
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_USCALED
        | vk::Format::B8G8R8A8_SSCALED
        | vk::Format::B8G8R8A8_UINT
        | vk::Format::B8G8R8A8_SINT
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SNORM_PACK32
        | vk::Format::A8B8G8R8_USCALED_PACK32
        | vk::Format::A8B8G8R8_SSCALED_PACK32
        | vk::Format::A8B8G8R8_UINT_PACK32
        | vk::Format::A8B8G8R8_SINT_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32
        | vk::Format::A2R10G10B10_UNORM_PACK32
        | vk::Format::A2R10G10B10_SNORM_PACK32
        | vk::Format::A2R10G10B10_USCALED_PACK32
        | vk::Format::A2R10G10B10_SSCALED_PACK32
        | vk::Format::A2R10G10B10_UINT_PACK32
        | vk::Format::A2R10G10B10_SINT_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2B10G10R10_SNORM_PACK32
        | vk::Format::A2B10G10R10_USCALED_PACK32
        | vk::Format::A2B10G10R10_SSCALED_PACK32
        | vk::Format::A2B10G10R10_UINT_PACK32
        | vk::Format::A2B10G10R10_SINT_PACK32
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_USCALED
        | vk::Format::R16G16_SSCALED
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16_S10_5_NV
        | vk::Format::R10X6G10X6_UNORM_2PACK16
        | vk::Format::R12X4G12X4_UNORM_2PACK16
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32_SFLOAT
        | vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::X8_D24_UNORM_PACK32
        | vk::Format::D32_SFLOAT => Some(4),

        // 6 bytes
        vk::Format::R16G16B16_UNORM
        | vk::Format::R16G16B16_SNORM
        | vk::Format::R16G16B16_USCALED
        | vk::Format::R16G16B16_SSCALED
        | vk::Format::R16G16B16_UINT
        | vk::Format::R16G16B16_SINT
        | vk::Format::R16G16B16_SFLOAT => Some(6),

        // 8 bytes
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_USCALED
        | vk::Format::R16G16B16A16_SSCALED
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16
        | vk::Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R64_UINT
        | vk::Format::R64_SINT
        | vk::Format::R64_SFLOAT => Some(8),

        // 12 bytes
        vk::Format::R32G32B32_UINT
        | vk::Format::R32G32B32_SINT
        | vk::Format::R32G32B32_SFLOAT => Some(12),

        // 16 bytes
        vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R64G64_UINT
        | vk::Format::R64G64_SINT
        | vk::Format::R64G64_SFLOAT => Some(16),

        // 24 bytes
        vk::Format::R64G64B64_UINT
        | vk::Format::R64G64B64_SINT
        | vk::Format::R64G64B64_SFLOAT => Some(24),

        // 32 bytes
        vk::Format::R64G64B64A64_UINT
        | vk::Format::R64G64B64A64_SINT
        | vk::Format::R64G64B64A64_SFLOAT => Some(32),

        // Block-compressed, planar YCbCr, and combined depth-stencil
        // formats have no single well-defined bytes-per-texel value.
        _ => None,
    }
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
    mip_levels: u32,
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
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateImageError> {
        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(samples)
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
        let reqs = unsafe { device.get_raw_image_memory_requirements(handle) };
        let allocation_name = name.unwrap_or("image");
        let allocation = device
            .allocate_memory(allocation_name, reqs, MemoryUsage::GpuOnly, false)
            .map_err(|e| {
                // SAFETY: handle was created from this device and is not
                // bound to memory yet.
                unsafe { device.destroy_raw_image(handle) };
                CreateImageError::AllocateMemory(e)
            })?;

        // SAFETY: handle and allocation memory are valid and belong to
        // this device.
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
            mip_levels,
        })
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        tracing::debug!("Dropping image {:?}", self.handle);
        // SAFETY: handle was created from parent and is owned by this
        // wrapper.
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
pub(crate) struct DeviceLocalImage {
    inner: AllocatedImage,
}

impl DeviceLocalImage {
    /// Create a 2-D device-local image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateImageError> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        Ok(Self {
            inner: AllocatedImage::new(
                device, extent, format, mip_levels, usage, samples, name,
            )?,
        })
    }

    #[inline]
    pub(crate) fn raw_image(&self) -> vk::Image {
        self.inner.handle
    }

    #[inline]
    fn whole_image_barrier<'a>(&'a self) -> vk::ImageMemoryBarrier<'a> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(self.inner.mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        crate::memory::image_barrier()
            .image(self.inner.handle)
            .subresource_range(subresource_range)
    }

    #[inline]
    fn whole_image_barrier2<'a>(&'a self) -> vk::ImageMemoryBarrier2<'a> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(self.inner.mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        crate::memory::image_barrier2()
            .image(self.inner.handle)
            .subresource_range(subresource_range)
    }

    /// Build a `vk::BufferImageCopy` for the given mip level and buffer
    /// offset. This encodes the subresource, offset, and extent used when
    /// copying from a linear staging buffer into the image.
    #[inline]
    fn buffer_image_copy_for_mip(
        &self,
        buffer_offset: vk::DeviceSize,
        mip_level: u32,
        mip_extent: vk::Extent3D,
    ) -> vk::BufferImageCopy {
        let subresource_layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(1);
        vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource_layers)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(mip_extent)
    }

    #[inline]
    pub(crate) fn format(&self) -> vk::Format {
        self.inner.format
    }

    /// Record the commands to upload pixel data from a staging buffer to
    /// this image in `command_buffer` using VK 1.0 core barriers.
    ///
    /// Records:
    /// - barrier: `UNDEFINED` → `TRANSFER_DST_OPTIMAL`
    /// - `vkCmdCopyBufferToImage`
    /// - barrier: `TRANSFER_DST_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL`
    ///
    /// The caller is responsible for begin/end/submit and any CPU/GPU
    /// synchronization.
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - `src` must be created with `TRANSFER_SRC` usage and contain
    ///   exactly `width * height * bytes_per_pixel` bytes of packed
    ///   pixel data.
    /// - `self` must be created with at least `TRANSFER_DST | SAMPLED`
    ///   usage.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the submitted commands has completed.
    pub(crate) unsafe fn record_upload_from(
        &self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
    ) -> Result<(), RecordCopyFromError> {
        debug_assert_eq!(
            command_buffer.state(),
            crate::command::CommandBufferState::Recording
        );
        let texel_size = format_texel_size(self.inner.format)
            .ok_or(RecordCopyFromError::UnknownFormat(self.inner.format))?;
        let extent = self.inner.extent;
        let required = u64::from(extent.width)
            * u64::from(extent.height)
            * u64::from(extent.depth)
            * u64::from(texel_size);
        let actual = src.size();
        if actual < required {
            return Err(RecordCopyFromError::StagingTooSmall {
                required,
                actual,
            });
        }

        let raw_cmd = command_buffer.raw();
        let device = &self.inner.parent;
        let image = self.inner.handle;

        // NOTE: layout transitions are performed by the caller; this
        // function only records the buffer→image copy.

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
        // SAFETY: caller guarantees recording state; src buffer and
        // image are valid and in the correct layouts.
        unsafe {
            device.cmd_copy_buffer_to_image(
                raw_cmd,
                src.raw_buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            )
        };

        // NOTE: caller must perform any required layout transition
        // after the copy (e.g. TRANSFER_DST_OPTIMAL →
        // SHADER_READ_ONLY_OPTIMAL).

        Ok(())
    }

    /// Record a buffer-to-image copy for one mip level.
    ///
    /// Infallible — no size validation is performed (required because
    /// `format_texel_size` returns `None` for block-compressed
    /// formats).
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - `src` must be created with `TRANSFER_SRC` usage.
    /// - The image must be in `TRANSFER_DST_OPTIMAL` layout.
    /// - `buffer_offset` + the byte size of `mip_extent` at this
    ///   format must not exceed `src.size()`.
    /// - The caller must ensure `src` and `self` remain alive until
    ///   GPU execution of the submitted commands has completed.
    pub(crate) unsafe fn record_upload_mip_from(
        &self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
        buffer_offset: vk::DeviceSize,
        mip_level: u32,
        mip_extent: vk::Extent3D,
    ) {
        debug_assert_eq!(
            command_buffer.state(),
            crate::command::CommandBufferState::Recording
        );
        let raw_cmd = command_buffer.raw();
        let device = &self.inner.parent;
        let image = self.inner.handle;

        let subresource_layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(1);
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(subresource_layers)
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(mip_extent);
        // SAFETY: caller guarantees recording state; src buffer and
        // image are valid and in the correct layouts.
        unsafe {
            device.cmd_copy_buffer_to_image(
                raw_cmd,
                src.raw_buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            )
        };
    }
}

// ---------------------------------------------------------------------------
// ImageView
// ---------------------------------------------------------------------------

/// An owned `VkImageView` for a [`DeviceLocalImage`].
pub(crate) struct ImageView {
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
    pub(crate) fn new(
        device: &Arc<Device>,
        image: &DeviceLocalImage,
        mip_levels: u32,
        name: Option<&str>,
    ) -> Result<Self, CreateImageViewError> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image.raw_image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format())
            .subresource_range(subresource_range);

        // SAFETY: create_info references a valid image from the same
        // device.
        let handle = unsafe { device.create_raw_image_view(&create_info) }
            .map_err(CreateImageViewError::Vulkan)?;

        // SAFETY: handle is a valid image view from this device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name image view {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    #[inline]
    pub(crate) fn raw_image_view(&self) -> vk::ImageView {
        self.handle
    }

    /// Create a 2-D depth image view for `image`.
    pub(crate) fn new_depth(
        device: &Arc<Device>,
        image: &DeviceLocalImage,
        name: Option<&str>,
    ) -> Result<Self, CreateImageViewError> {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image.raw_image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format())
            .subresource_range(subresource_range);

        // SAFETY: create_info references a valid image from the same
        // device.
        let handle = unsafe { device.create_raw_image_view(&create_info) }
            .map_err(CreateImageViewError::Vulkan)?;

        // SAFETY: handle is a valid image view from this device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name image view {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
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

// ---------------------------------------------------------------------------
// Texture
// ---------------------------------------------------------------------------

/// A GPU-only 2-D texture: a [`DeviceLocalImage`] bundled with its
/// default 2-D colour [`ImageView`].
///
/// `Texture` is the public-facing image type. [`DeviceLocalImage`] and
/// [`ImageView`] are `pub(crate)` implementation details.
///
/// # Drop order
///
/// `view` is declared before `image` so Rust drops the view first,
/// satisfying the Vulkan requirement that all image views must be
/// destroyed before the image they reference.
#[derive(Debug)]
pub struct Texture {
    // IMPORTANT: `view` must be declared before `image`.
    // Rust drops fields in declaration order; `vkDestroyImageView`
    // must be called before `vkDestroyImage`.
    view: ImageView,
    image: DeviceLocalImage,
}

impl Texture {
    /// Create a 2-D device-local texture (image + default view).
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateTextureError> {
        let image = DeviceLocalImage::new(
            device,
            width,
            height,
            format,
            mip_levels,
            usage,
            vk::SampleCountFlags::TYPE_1,
            name,
        )?;
        let view = ImageView::new(device, &image, mip_levels, name)?;
        Ok(Self { view, image })
    }

    /// Record upload commands using VK 1.0 core barriers.
    ///
    /// Records a layout transition to `TRANSFER_DST_OPTIMAL`, copies the
    /// buffer, then transitions to `SHADER_READ_ONLY_OPTIMAL`. The caller
    /// is responsible for begin/end/submit and CPU/GPU synchronization.
    ///
    /// Returns [`RecordCopyFromError::UnknownFormat`] if the texel size
    /// cannot be determined, or [`RecordCopyFromError::StagingTooSmall`]
    /// if `src` is smaller than the image requires.
    ///
    /// # Safety
    ///
    /// - `command_buffer` must be in the recording state.
    /// - `src` must be created with `TRANSFER_SRC` usage and contain
    ///   exactly `width * height * bytes_per_pixel` bytes of packed
    ///   pixel data.
    /// - `self` must have been created with at least
    ///   `TRANSFER_DST | SAMPLED` usage.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the submitted commands has completed.
    /// - Image must be in the layout TRANSFER_DST_OPTIMAL
    pub unsafe fn record_copy_from(
        &self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
    ) -> Result<(), RecordCopyFromError> {
        // SAFETY: caller upholds the same preconditions.
        unsafe { self.image.record_upload_from(command_buffer, src) }
    }

    /// Record a buffer-to-image copy for a single mip level.
    ///
    /// Infallible — no size validation. See
    /// [`DeviceLocalImage::record_upload_mip_from`] for full safety
    /// requirements.
    ///
    /// # Safety
    /// Same as [`record_copy_from`](Self::record_copy_from), except
    /// `buffer_offset`, `mip_level`, and `mip_extent` select which
    /// mip and which region of the staging buffer to use.
    pub unsafe fn record_copy_mip_from(
        &self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
        buffer_offset: vk::DeviceSize,
        mip_level: u32,
        mip_extent: vk::Extent3D,
    ) {
        // SAFETY: caller upholds the same preconditions.
        unsafe {
            self.image.record_upload_mip_from(
                command_buffer,
                src,
                buffer_offset,
                mip_level,
                mip_extent,
            )
        }
    }

    /// Build a `vk::BufferImageCopy` describing the copy region for the
    /// given mip level. Useful when accumulating regions to pass to a
    /// single `vkCmdCopyBufferToImage` call.
    pub fn buffer_image_copy_for_mip(
        &self,
        buffer_offset: vk::DeviceSize,
        mip_level: u32,
        mip_extent: vk::Extent3D,
    ) -> vk::BufferImageCopy {
        self.image.buffer_image_copy_for_mip(
            buffer_offset,
            mip_level,
            mip_extent,
        )
    }

    /// Helper that returns an `ImageMemoryBarrier` pre-filled with
    /// this texture's `VkImage` and a subresource range that covers
    /// the whole image (colour aspect, mip 0, array layer 0, 1
    /// level/layer).
    #[inline]
    pub fn whole_image_barrier<'a>(&'a self) -> vk::ImageMemoryBarrier<'a> {
        self.image.whole_image_barrier()
    }

    /// Helper that returns an `ImageMemoryBarrier2` pre-filled with
    /// this texture's `VkImage` and a subresource range that covers
    /// the whole image. Use with `vkCmdPipelineBarrier2`.
    #[inline]
    pub fn whole_image_barrier2<'a>(&'a self) -> vk::ImageMemoryBarrier2<'a> {
        self.image.whole_image_barrier2()
    }

    /// Returns the extent of the underlying image.
    #[inline]
    pub fn extent(&self) -> vk::Extent3D {
        self.image.inner.extent
    }

    /// Returns the raw `VkImageView` handle for use in descriptor
    /// writes.
    #[inline]
    pub fn raw_image_view(&self) -> vk::ImageView {
        self.view.raw_image_view()
    }

    pub fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }
}

// ---------------------------------------------------------------------------
// DepthImage
// ---------------------------------------------------------------------------

/// A GPU-only 2-D depth render target: a [`DeviceLocalImage`] with a
/// depth [`ImageView`].
///
/// Unlike [`Texture`], a `DepthImage` is never uploaded to from a
/// staging buffer. Layout transitions are performed per-frame using
/// `old_layout = UNDEFINED` so the driver can discard old contents;
/// this is safe because `LOAD_OP_CLEAR` is always used.
///
/// # Drop order
///
/// `view` is declared before `image` so Rust drops the view first,
/// satisfying the Vulkan requirement that all image views must be
/// destroyed before the image they reference.
#[derive(Debug)]
pub struct DepthImage {
    // IMPORTANT: `view` must be declared before `image`.
    view: ImageView,
    image: DeviceLocalImage,
}

impl DepthImage {
    /// Create a 2-D device-local depth image (image + depth view).
    ///
    /// `format` must be a depth-only format (e.g.
    /// `vk::Format::D32_SFLOAT`). The image is created with
    /// `DEPTH_STENCIL_ATTACHMENT` usage.
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateTextureError> {
        let image = DeviceLocalImage::new(
            device,
            width,
            height,
            format,
            1,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            samples,
            name,
        )?;
        let view = ImageView::new_depth(device, &image, name)?;
        Ok(Self { view, image })
    }

    /// Returns the raw `VkImage` handle for use in barriers.
    #[inline]
    pub fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    /// Returns the raw `VkImageView` handle for use as a depth
    /// attachment.
    #[inline]
    pub fn raw_image_view(&self) -> vk::ImageView {
        self.view.raw_image_view()
    }
}

// ---------------------------------------------------------------------------
// MsaaImage
// ---------------------------------------------------------------------------

/// A GPU-only 2-D MSAA colour render target: a multisampled
/// [`DeviceLocalImage`] with a colour [`ImageView`].
///
/// Created with `COLOR_ATTACHMENT | TRANSIENT_ATTACHMENT` usage.
/// `TRANSIENT_ATTACHMENT` hints to the driver that the contents need
/// not be written to main memory — only the resolved single-sample
/// result is kept.
///
/// # Drop order
///
/// `view` is declared before `image` so Rust drops the view first,
/// satisfying the Vulkan requirement that all image views must be
/// destroyed before the image they reference.
#[derive(Debug)]
pub struct MsaaImage {
    // IMPORTANT: `view` must be declared before `image`.
    view: ImageView,
    image: DeviceLocalImage,
}

impl MsaaImage {
    /// Create a 2-D MSAA colour image (image + colour view).
    ///
    /// `format` should match the swapchain surface format.
    /// `samples` must be greater than `TYPE_1`.
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateTextureError> {
        let image = DeviceLocalImage::new(
            device,
            width,
            height,
            format,
            1,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            samples,
            name,
        )?;
        let view = ImageView::new(device, &image, 1, name)?;
        Ok(Self { view, image })
    }

    /// Returns the raw `VkImage` handle.
    #[inline]
    pub fn raw_image(&self) -> vk::Image {
        self.image.raw_image()
    }

    /// Returns the raw `VkImageView` handle for use as a colour
    /// attachment.
    #[inline]
    pub fn raw_image_view(&self) -> vk::ImageView {
        self.view.raw_image_view()
    }
}
