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

use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use bytemuck::Pod;
use gpu_allocator::{AllocationError, vulkan::Allocation};
use thiserror::Error;

use crate::command::ResettableCommandBuffer;
use crate::device::{Device, MemoryUsage};

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
    #[inline]
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
        memory_usage: MemoryUsage,
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
            .allocate_memory(allocation_name, reqs, memory_usage, true)
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

    #[inline]
    fn raw_buffer(&self) -> vk::Buffer {
        self.handle
    }

    #[inline]
    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    #[inline]
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
    ptr: NonNull<u8>,
}

impl HostVisibleBuffer {
    pub fn new(
        device: &Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Result<Self, CreateBufferError> {
        let inner = AllocatedBuffer::new(
            device,
            size,
            usage,
            name,
            MemoryUsage::CpuToGpu,
        )?;
        Ok(Self {
            ptr: inner
                .allocation
                .as_ref()
                .expect("Allocation can only be null during drop")
                .mapped_ptr()
                .expect(
                    "Allocation was created to be mapped and yet it was not",
                )
                .cast(),
            inner,
        })
    }

    #[inline]
    pub fn write_pod<T: Pod>(
        &mut self,
        data: &[T],
    ) -> Result<(), WriteBufferError> {
        self.write_pod_iter_exact(data.iter().copied())
    }

    #[inline]
    pub fn write_pod_iter_exact<T: Pod, I>(
        &mut self,
        iter: I,
    ) -> Result<(), WriteBufferError>
    where
        I: Iterator<Item = T> + ExactSizeIterator,
    {
        let size = iter.len() * size_of::<T>();
        self.check_size_for_write(size)?;
        self.write_pod_iter(iter)
    }

    fn write_pod_iter<T: Pod, I>(
        &mut self,
        iter: I,
    ) -> Result<(), WriteBufferError>
    where
        I: Iterator<Item = T>,
    {
        let allocation = self
            .inner
            .allocation
            .as_ref()
            .expect("allocation is only None during drop");
        let mapped_ptr = self.ptr;
        let dst = mapped_ptr.as_ptr().cast::<u8>();
        let cap = self.inner.size() as usize;
        let mut written: usize = 0;

        for item in iter {
            let bytes = bytemuck::bytes_of(&item);
            let len = bytes.len();
            if written + len > cap {
                return Err(WriteBufferError::DataTooLarge {
                    data_bytes: written + len,
                    buffer_bytes: self.inner.size(),
                });
            }
            // SAFETY: dst is valid for writes within allocation size;
            // bounds checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    dst.add(written),
                    len,
                );
            }
            written += len;
        }

        // Flush non-coherent ranges if needed (same logic as write_u8_iter)
        let is_coherent = allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        if !is_coherent && written > 0 {
            let atom = self.inner.parent.non_coherent_atom_size();
            debug_assert_eq!(allocation.offset() % atom, 0);
            debug_assert_eq!(allocation.size() % atom, 0);
            let flush_size = (written as vk::DeviceSize).div_ceil(atom) * atom;
            let flush_range = vk::MappedMemoryRange::default()
                // SAFETY: Allocation is currently live
                .memory(unsafe { allocation.memory() })
                .offset(allocation.offset())
                .size(flush_size);
            // SAFETY: Allocation is currently live. flush range is aligned to
            // non_coherent_atom_size (guaranteed by the allocator), offset +
            // size is within the buffer
            unsafe {
                self.inner.parent.flush_raw_mapped_memory_ranges(
                    std::slice::from_ref(&flush_range),
                )
            }
            .map_err(WriteBufferError::FlushMemory)?;
        }

        Ok(())
    }

    #[inline]
    pub fn write_u8_iter_exact<I>(
        &mut self,
        iter: I,
    ) -> Result<(), WriteBufferError>
    where
        I: Iterator<Item = u8> + ExactSizeIterator,
    {
        self.check_size_for_write(iter.len())?;
        self.write_u8_iter(iter)
    }

    #[inline]
    fn write_u8_iter<I>(&mut self, iter: I) -> Result<(), WriteBufferError>
    where
        I: Iterator<Item = u8>,
    {
        // Delegate to the canonical POD writer with `u8` as the POD type.
        self.write_pod_iter(iter)
    }

    #[inline]
    pub fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }

    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.inner.size()
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        self.inner.parent()
    }

    #[inline]
    fn check_size_for_write(
        &self,
        size: usize,
    ) -> Result<(), WriteBufferError> {
        if self.size() >= size as u64 {
            Ok(())
        } else {
            Err(WriteBufferError::DataTooLarge {
                data_bytes: size,
                buffer_bytes: self.size(),
            })
        }
    }
}

impl BufferHandle for HostVisibleBuffer {
    #[inline]
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
                MemoryUsage::GpuOnly,
            )?,
        })
    }

    #[inline]
    pub fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }

    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.inner.size()
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        self.inner.parent()
    }

    /// Record an upload of the entire source buffer into this device-local
    /// buffer. Returns [`UploadBufferError::SourceTooLarge`] if `src` is larger
    /// than `self`.
    ///
    /// The caller is responsible for begin/end/submit and any CPU/GPU
    /// synchronization.
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the recorded copy has completed.
    /// - `src` must be created with `TRANSFER_SRC` usage and `self` with
    ///   `TRANSFER_DST` usage.
    pub unsafe fn record_copy_from(
        &mut self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
    ) -> Result<(), UploadBufferError> {
        debug_assert_eq!(
            command_buffer.state(),
            crate::command::CommandBufferState::Recording
        );
        let copy_size = src.size();
        if copy_size > self.size() {
            return Err(UploadBufferError::SourceTooLarge {
                src_bytes: copy_size,
                dst_bytes: self.size(),
            });
        }
        // SAFETY: preconditions carry through; offset 0 is in-bounds.
        unsafe {
            self.record_upload_region_from(command_buffer, src, 0, 0, copy_size)
        }
    }

    /// Record an upload of a byte range from the source buffer into this
    /// device-local buffer. Returns [`UploadBufferError::RegionOutOfBounds`] if
    /// any region extends past the end of its buffer.
    ///
    /// The caller is responsible for begin/end/submit and any CPU/GPU
    /// synchronization.
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - The caller must ensure `src` and `self` remain alive until GPU
    ///   execution of the recorded copy has completed.
    /// - `src` must be created with `TRANSFER_SRC` usage and `self` with
    ///   `TRANSFER_DST` usage.
    pub unsafe fn record_upload_region_from(
        &mut self,
        command_buffer: &mut ResettableCommandBuffer,
        src: &HostVisibleBuffer,
        src_offset: vk::DeviceSize,
        dst_offset: vk::DeviceSize,
        copy_size: vk::DeviceSize,
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

        let raw_cmd = command_buffer.raw();
        let copy_region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(copy_size);
        // SAFETY: caller guarantees recording state; buffers and
        // region are valid and in-bounds.
        unsafe {
            self.parent().cmd_copy_buffer(
                raw_cmd,
                src.raw_buffer(),
                self.raw_buffer(),
                std::slice::from_ref(&copy_region),
            )
        };

        Ok(())
    }
}

impl BufferHandle for DeviceLocalBuffer {
    #[inline]
    fn raw_buffer(&self) -> vk::Buffer {
        self.inner.raw_buffer()
    }
}
