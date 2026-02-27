//! GPU buffer types and the [`BufferHandle`] trait.
//!
//! Two concrete buffer wrappers are provided:
//!
//! - [`HostVisibleBuffer`] — CPU-writable (`CpuToGpu`) memory, suitable
//!   for staging or small per-frame uploads. Write data with
//!   [`write_pod`](HostVisibleBuffer::write_pod).
//! - [`DeviceLocalBuffer`] — GPU-only memory, highest bandwidth.
//!   Populate via a one-time copy submission using
//!   [`upload_from_host_visible`](DeviceLocalBuffer::upload_from_host_visible).
//!
//! Both types own their allocation and destroy it on drop.
//! [`BufferHandle`] is a thin trait for passing either type (or raw
//! `vk::Buffer` references) to command recording helpers.

use std::sync::Arc;

use ash::vk;
use bytemuck::Pod;
use gpu_allocator::{AllocationError, MemoryLocation, vulkan::Allocation};
use thiserror::Error;

use crate::command::CommandBufferHandle;
use crate::device::Device;

/// Trait for types that expose a raw `VkBuffer` handle.
///
/// Implemented by [`HostVisibleBuffer`] and [`DeviceLocalBuffer`].
/// Blanket impls cover `&T` and `&mut T`, so both owned wrappers and
/// borrows of them satisfy the bound. Allows recording helpers (e.g.
/// `bind_vertex_buffer`) to be generic over concrete buffer types.
pub trait BufferHandle {
    fn raw_buffer(&self) -> vk::Buffer;
}

impl<T> BufferHandle for &T
where
    T: BufferHandle + ?Sized,
{
    fn raw_buffer(&self) -> vk::Buffer {
        (*self).raw_buffer()
    }
}

#[derive(Debug, Error)]
pub enum CreateBufferError {
    #[error("Vulkan error creating buffer: {0}")]
    CreateBuffer(vk::Result),

    #[error("GPU allocator error allocating memory: {0}")]
    AllocateMemory(AllocationError),

    #[error("Vulkan error binding buffer memory: {0}")]
    BindMemory(vk::Result),
}

#[derive(Debug, Error)]
pub enum WriteBufferError {
    #[error(
        "Data size ({data_bytes} bytes) exceeds buffer size ({buffer_bytes} bytes)"
    )]
    DataTooLarge {
        data_bytes: usize,
        buffer_bytes: vk::DeviceSize,
    },

    #[error("Vulkan error flushing mapped memory: {0}")]
    FlushMemory(vk::Result),

    #[error("Allocation is not host-mapped")]
    NotMapped,
}

#[derive(Debug, Error)]
pub enum UploadBufferError {
    #[error(
        "Source buffer ({src_bytes} bytes) exceeds destination buffer \
         ({dst_bytes} bytes)"
    )]
    SourceTooLarge {
        src_bytes: vk::DeviceSize,
        dst_bytes: vk::DeviceSize,
    },

    #[error(
        "Copy region out of bounds: src(size={src_size}, offset={src_offset}, \
         copy={copy_size}), dst(size={dst_size}, offset={dst_offset}, \
         copy={copy_size})"
    )]
    RegionOutOfBounds {
        src_size: vk::DeviceSize,
        src_offset: vk::DeviceSize,
        dst_size: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        copy_size: vk::DeviceSize,
    },

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

struct AllocatedBuffer {
    parent: Arc<Device>,
    handle: vk::Buffer,
    allocation: Option<Allocation>,
    size: vk::DeviceSize,
}

impl std::fmt::Debug for AllocatedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocatedBuffer")
            .field("handle", &self.handle)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl AllocatedBuffer {
    fn new(
        device: &Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
        location: MemoryLocation,
    ) -> Result<Self, CreateBufferError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: create_info is fully initialised and has no borrowed data.
        let handle = unsafe { device.create_raw_buffer(&create_info) }
            .map_err(CreateBufferError::CreateBuffer)?;

        // SAFETY: handle is a valid buffer created from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name buffer {:?}: {e}", handle);
        }

        // SAFETY: handle is a valid buffer created from this device.
        let reqs = unsafe { device.get_raw_buffer_memory_requirements(handle) };
        let allocation_name = name.unwrap_or("buffer");
        let allocation = device
            .allocate_memory(allocation_name, reqs, location, true)
            .map_err(|e| {
                // SAFETY: handle was created from this device and is not bound
                // to memory yet.
                unsafe { device.destroy_raw_buffer(handle) };
                CreateBufferError::AllocateMemory(e)
            })?;

        // SAFETY: handle and allocation memory are valid and belong to this
        // device.
        let bind_result = unsafe {
            device.bind_raw_buffer_memory(
                handle,
                allocation.memory(),
                allocation.offset(),
            )
        };
        if let Err(e) = bind_result {
            let _ = device.free_memory(allocation);
            // SAFETY: handle is valid and owned by this scope.
            unsafe {
                device.destroy_raw_buffer(handle);
            }
            return Err(CreateBufferError::BindMemory(e));
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
            allocation: Some(allocation),
            size,
        })
    }

    fn raw_buffer(&self) -> vk::Buffer {
        self.handle
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        tracing::debug!("Dropping buffer {:?}", self.handle);
        // SAFETY: handle was created from parent and is owned by this wrapper.
        unsafe {
            self.parent.destroy_raw_buffer(self.handle);
        }

        if let Some(allocation) = self.allocation.take()
            && let Err(e) = self.parent.free_memory(allocation)
        {
            tracing::error!("Failed to free GPU allocation: {e}");
        }
    }
}

/// A CPU-writable GPU buffer backed by `CpuToGpu` memory.
///
/// Suitable for staging uploads or small per-frame data. Write data
/// with [`write_pod`](Self::write_pod), which copies bytes into the
/// mapped region and flushes non-coherent memory ranges as needed.
#[derive(Debug)]
pub struct HostVisibleBuffer {
    inner: AllocatedBuffer,
}

impl HostVisibleBuffer {
    pub fn new(
        device: &Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateBufferError> {
        Ok(Self {
            inner: AllocatedBuffer::new(
                device,
                size,
                usage,
                name,
                MemoryLocation::CpuToGpu,
            )?,
        })
    }

    pub fn write_pod<T: Pod>(
        &mut self,
        data: &[T],
    ) -> Result<(), WriteBufferError> {
        let bytes = bytemuck::cast_slice(data);
        if bytes.len() as vk::DeviceSize > self.inner.size {
            return Err(WriteBufferError::DataTooLarge {
                data_bytes: bytes.len(),
                buffer_bytes: self.inner.size,
            });
        }

        let allocation = self
            .inner
            .allocation
            .as_ref()
            .expect("allocation is only None during drop");
        let mapped_ptr =
            allocation.mapped_ptr().ok_or(WriteBufferError::NotMapped)?;

        // SAFETY: mapped_ptr points to CPU-visible allocation memory and
        // bytes.len() has been bounds-checked against buffer size above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped_ptr.as_ptr().cast::<u8>(),
                bytes.len(),
            );
        }

        // HOST_COHERENT memory is always visible to the GPU after the
        // CPU write; no explicit flush is needed.
        let is_coherent = allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        if !is_coherent && !bytes.is_empty() {
            let atom = self.inner.parent.non_coherent_atom_size();
            // Both invariants guaranteed by Device::allocate_memory.
            debug_assert_eq!(allocation.offset() % atom, 0);
            debug_assert_eq!(allocation.size() % atom, 0);
            // Round written bytes up to the atom boundary. Fits
            // within allocation.size() since bytes.len() <=
            // self.inner.size <= allocation.size().
            let flush_size =
                (bytes.len() as vk::DeviceSize).div_ceil(atom) * atom;
            let flush_range = vk::MappedMemoryRange::default()
                // SAFETY: allocation was returned by gpu-allocator
                // for this device and remains live while self is
                // alive.
                .memory(unsafe { allocation.memory() })
                .offset(allocation.offset())
                .size(flush_size);
            // SAFETY: flush_range references a valid mapped memory
            // allocation from this device.
            unsafe {
                self.inner.parent.flush_raw_mapped_memory_ranges(
                    std::slice::from_ref(&flush_range),
                )
            }
            .map_err(WriteBufferError::FlushMemory)?;
        }

        Ok(())
    }

    pub fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.inner.size()
    }

    pub fn parent(&self) -> &Arc<Device> {
        self.inner.parent()
    }
}

impl BufferHandle for HostVisibleBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }
}

/// A GPU-only buffer backed by `GpuOnly` memory.
///
/// Provides the highest memory bandwidth but cannot be written by the
/// CPU directly. Populate from a [`HostVisibleBuffer`] using
/// [`upload_from_host_visible`](Self::upload_from_host_visible).
#[derive(Debug)]
pub struct DeviceLocalBuffer {
    inner: AllocatedBuffer,
}

impl DeviceLocalBuffer {
    pub fn new(
        device: &Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateBufferError> {
        Ok(Self {
            inner: AllocatedBuffer::new(
                device,
                size,
                usage,
                name,
                MemoryLocation::GpuOnly,
            )?,
        })
    }

    pub fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.inner.size()
    }

    pub fn parent(&self) -> &Arc<Device> {
        self.inner.parent()
    }

    /// Upload data from a host-visible source buffer into this device-local
    /// buffer using a one-time copy submission.
    ///
    /// # Safety
    /// - `command_buffer` must be externally synchronized and valid for
    ///   reset/begin/record/end on the current thread.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the submitted copy has completed.
    /// - If `fence` is `Some`, the caller must ensure it is unsignaled and
    ///   later wait/reset it before reusing resources referenced by the copy.
    pub unsafe fn upload_from_host_visible(
        &mut self,
        command_buffer: &mut impl CommandBufferHandle,
        src: &HostVisibleBuffer,
        fence: Option<vk::Fence>,
    ) -> Result<(), UploadBufferError> {
        let copy_size = src.size();
        if copy_size > self.size() {
            return Err(UploadBufferError::SourceTooLarge {
                src_bytes: copy_size,
                dst_bytes: self.size(),
            });
        }

        // SAFETY: forwards to region helper with full-buffer offsets and size.
        unsafe {
            self.upload_from_host_visible_region(
                command_buffer,
                src,
                0,
                0,
                copy_size,
                fence,
            )
        }
    }

    /// Upload a byte range from a host-visible source buffer into this
    /// device-local buffer.
    ///
    /// # Safety
    /// - `command_buffer` must be externally synchronized and valid for
    ///   reset/begin/record/end on the current thread.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the submitted copy has completed.
    /// - If `fence` is `Some`, the caller must ensure it is unsignaled and
    ///   later wait/reset it before reusing resources referenced by the copy.
    pub unsafe fn upload_from_host_visible_region(
        &mut self,
        command_buffer: &mut impl CommandBufferHandle,
        src: &HostVisibleBuffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        copy_size: vk::DeviceSize,
        fence: Option<vk::Fence>,
    ) -> Result<(), UploadBufferError> {
        if src_offset.saturating_add(copy_size) > src.size()
            || dst_offset.saturating_add(copy_size) > self.size()
        {
            return Err(UploadBufferError::RegionOutOfBounds {
                src_size: src.size(),
                src_offset,
                dst_size: self.size(),
                dst_offset,
                copy_size,
            });
        }

        let raw_command_buffer = command_buffer.raw_command_buffer();
        let begin_info = vk::CommandBufferBeginInfo::default();

        // SAFETY: caller guarantees command buffer is not pending.
        unsafe {
            self.parent().reset_raw_command_buffer(
                raw_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .map_err(UploadBufferError::ResetCommandBuffer)?;

        // SAFETY: caller guarantees command buffer is in initial state.
        unsafe {
            self.parent()
                .begin_raw_command_buffer(raw_command_buffer, &begin_info)
        }
        .map_err(UploadBufferError::BeginCommandBuffer)?;

        let copy_region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(copy_size);
        // SAFETY: buffers and region are valid and in-bounds.
        unsafe {
            self.parent().cmd_copy_buffer(
                raw_command_buffer,
                src.raw_buffer(),
                self.raw_buffer(),
                std::slice::from_ref(&copy_region),
            )
        };

        // SAFETY: caller guarantees command buffer is in recording state.
        unsafe { self.parent().end_raw_command_buffer(raw_command_buffer) }
            .map_err(UploadBufferError::EndCommandBuffer)?;

        let upload_cmd_info = vk::CommandBufferSubmitInfo::default()
            .command_buffer(raw_command_buffer);
        let upload_submit = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&upload_cmd_info));
        let submit_fence = fence.unwrap_or(vk::Fence::null());

        // SAFETY: command buffer is executable and references valid resources.
        unsafe {
            self.parent().graphics_present_queue_submit2(
                std::slice::from_ref(&upload_submit),
                submit_fence,
            )
        }
        .map_err(UploadBufferError::QueueSubmit)?;

        if fence.is_none() {
            self.parent()
                .wait_idle()
                .map_err(UploadBufferError::WaitIdle)?;
        }

        Ok(())
    }
}

impl BufferHandle for DeviceLocalBuffer {
    fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }
}
