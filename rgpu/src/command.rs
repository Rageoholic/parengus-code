use std::{
    marker::PhantomData,
    sync::{Arc, mpsc},
};

use ash::vk;
use thiserror::Error;

use crate::buffer::BufferHandle;
use crate::device::{Device, DynamicRenderingError};

pub trait CommandBufferHandle {
    fn raw_command_buffer(&self) -> vk::CommandBuffer;
}

impl<T> CommandBufferHandle for &T
where
    T: CommandBufferHandle + ?Sized,
{
    fn raw_command_buffer(&self) -> vk::CommandBuffer {
        (*self).raw_command_buffer()
    }
}

impl<T> CommandBufferHandle for &mut T
where
    T: CommandBufferHandle + ?Sized,
{
    fn raw_command_buffer(&self) -> vk::CommandBuffer {
        (**self).raw_command_buffer()
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CreateCommandPoolError {
    #[error("Vulkan error creating command pool: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum AllocateCommandBufferError {
    #[error("Vulkan error allocating command buffer: {0}")]
    Vulkan(vk::Result),
}

// ---------------------------------------------------------------------------
// CommandPoolShared — private inner state co-owned by pool and its buffers
// ---------------------------------------------------------------------------

/// Shared ownership of the raw Vulkan pool handle.
///
/// Held via `Arc` by both [`ResettableCommandPool`] and every
/// [`ResettableCommandBuffer`] allocated from it. The Vulkan pool is not
/// destroyed until all of those `Arc` clones are dropped, which prevents a
/// command buffer from holding a handle into a destroyed pool.
struct CommandPoolShared {
    parent: Arc<Device>,
    pool: vk::CommandPool,
}

impl Drop for CommandPoolShared {
    fn drop(&mut self) {
        tracing::debug!("Dropping command pool {:?}", self.pool);
        // SAFETY: pool was created from parent and is being destroyed. This
        // runs only when both ResettableCommandPool and every
        // ResettableCommandBuffer allocated from it have been dropped.
        // vkDestroyCommandPool implicitly frees all allocated command buffers.
        unsafe { self.parent.destroy_raw_command_pool(self.pool) };
    }
}

// ---------------------------------------------------------------------------
// ResettableCommandPool
// ---------------------------------------------------------------------------

/// An owned command pool that allocates individually-resettable
/// command buffers.
///
/// The pool is created with `RESET_COMMAND_BUFFER`, allowing each allocated
/// command buffer to be reset individually via
/// [`ResettableCommandBuffer::reset`].
///
/// `ResettableCommandPool` is `!Sync`: it cannot be shared across threads.
/// The Vulkan spec requires external synchronization for pool-level operations
/// (`vkAllocateCommandBuffers`); by being `!Sync` this is guaranteed
/// structurally rather than with a mutex. If cross-thread sharing is needed,
/// synchronize at a higher level.
///
/// The underlying Vulkan pool is not destroyed until both this wrapper and
/// every [`ResettableCommandBuffer`] allocated from it are dropped.
pub struct ResettableCommandPool {
    shared: Arc<CommandPoolShared>,
    /// Cloned into each newly allocated [`ResettableCommandBuffer`] so that
    /// dropping a buffer sends its handle back for recycling.
    sender: mpsc::Sender<vk::CommandBuffer>,
    /// Receives handles returned by dropped [`ResettableCommandBuffer`]s.
    /// Only drained by `allocate_command_buffer` on the pool-owning thread.
    /// `Receiver` is `!Sync`, making `ResettableCommandPool` structurally
    /// `!Sync` regardless of the `PhantomData` below.
    receiver: mpsc::Receiver<vk::CommandBuffer>,
    /// Explicit `!Sync` marker documenting the design intent. Redundant with
    /// `Receiver` but kept for clarity.
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for ResettableCommandPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResettableCommandPool")
            .field("pool", &self.shared.pool)
            .finish_non_exhaustive()
    }
}

impl ResettableCommandPool {
    /// Create a resettable command pool for the given queue family.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils` when
    /// the extension is available. Naming failures are logged as warnings and
    /// do not cause the call to fail.
    pub fn new(
        device: &Arc<Device>,
        queue_family: u32,
        name: Option<&str>,
    ) -> Result<Self, CreateCommandPoolError> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: create_info uses a valid queue family index for this device.
        let pool = unsafe { device.create_raw_command_pool(&create_info) }
            .map_err(CreateCommandPoolError::Vulkan)?;

        // SAFETY: pool is a valid command pool created from device.
        let name_result = unsafe { device.set_object_name_str(pool, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name command pool {:?}: {e}", pool);
        }

        let (sender, receiver) = mpsc::channel();

        Ok(Self {
            shared: Arc::new(CommandPoolShared {
                parent: Arc::clone(device),
                pool,
            }),
            sender,
            receiver,
            _not_sync: PhantomData,
        })
    }

    /// Allocate a single primary command buffer from this pool.
    ///
    /// All handles that were returned to the pool's channel (by previously
    /// dropped [`ResettableCommandBuffer`]s) are drained. One is recycled for
    /// the caller; any surplus handles are freed via `vkFreeCommandBuffers` to
    /// return their memory to the pool's allocator and bound peak usage. If no
    /// returned handles are available a new buffer is allocated from Vulkan.
    ///
    /// In all cases the returned buffer may not be in the initial state and
    /// **must be reset before recording**.
    ///
    /// The returned buffer holds a clone of the pool's shared inner `Arc`,
    /// so the underlying Vulkan pool is kept alive until both this pool and
    /// all its buffers are dropped.
    pub fn allocate_command_buffer(
        &self,
    ) -> Result<ResettableCommandBuffer, AllocateCommandBufferError> {
        // Drain all returned handles. Recycle one; free the rest to return
        // their memory to the pool's allocator and prevent runaway growth.
        let mut returned: Vec<vk::CommandBuffer> =
            std::iter::from_fn(|| self.receiver.try_recv().ok()).collect();

        let handle = if let Some(recycled) = returned.pop() {
            if !returned.is_empty() {
                // SAFETY: All handles in `returned` were allocated from
                // self.shared.pool. The drop→send contract requires callers
                // not to drop a ResettableCommandBuffer while its GPU work is
                // still executing, so every handle here is idle. External
                // synchronization on the pool is guaranteed by
                // ResettableCommandPool being !Sync — only the owning thread
                // can reach this call site.
                unsafe {
                    self.shared
                        .parent
                        .free_raw_command_buffers(self.shared.pool, &returned)
                };
            }
            recycled
        } else {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.shared.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            // SAFETY: allocate_info references a valid pool created from
            // parent. ResettableCommandPool is !Sync so no concurrent pool
            // access is possible.
            unsafe {
                self.shared
                    .parent
                    .allocate_raw_command_buffers(&allocate_info)
            }
            .map(|mut bufs| {
                debug_assert_eq!(bufs.len(), 1);
                bufs.remove(0)
            })
            .map_err(AllocateCommandBufferError::Vulkan)?
        };

        Ok(ResettableCommandBuffer {
            _pool: Arc::clone(&self.shared),
            parent: Arc::clone(&self.shared.parent),
            handle,
            return_sender: self.sender.clone(),
        })
    }

    pub fn raw_command_pool(&self) -> vk::CommandPool {
        self.shared.pool
    }

    pub fn parent(&self) -> &Arc<Device> {
        &self.shared.parent
    }
}

// ---------------------------------------------------------------------------
// ResettableCommandBuffer
// ---------------------------------------------------------------------------

/// A primary command buffer allocated from a [`ResettableCommandPool`].
///
/// All recording operations (`reset`, `begin`, `end`) are `unsafe` — the
/// caller is responsible for correct Vulkan state sequencing.
///
/// On drop, the raw handle is sent back to the pool's return channel for
/// recycling. If the pool has already been dropped the send is silently
/// discarded; `vkDestroyCommandPool` handles cleanup via [`CommandPoolShared`].
pub struct ResettableCommandBuffer {
    /// Keeps the pool alive until this buffer is dropped.
    _pool: Arc<CommandPoolShared>,
    parent: Arc<Device>,
    handle: vk::CommandBuffer,
    /// Returns the handle to the pool's channel on drop.
    return_sender: mpsc::Sender<vk::CommandBuffer>,
}

impl Drop for ResettableCommandBuffer {
    fn drop(&mut self) {
        // Send the handle back for recycling. If the receiver (pool) has been
        // dropped the error is intentionally ignored — the handle will be freed
        // implicitly when CommandPoolShared (and its
        // vkDestroyCommandPool) runs.
        let _ = self.return_sender.send(self.handle);
    }
}

impl std::fmt::Debug for ResettableCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResettableCommandBuffer")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl ResettableCommandBuffer {
    /// Reset this buffer to the initial state.
    ///
    /// # Safety
    /// The buffer must not be pending execution on the GPU.
    pub unsafe fn reset(&mut self) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees the buffer is not pending.
        unsafe {
            self.parent.reset_raw_command_buffer(
                self.handle,
                vk::CommandBufferResetFlags::empty(),
            )
        }
    }

    /// Begin recording.
    ///
    /// # Safety
    /// The buffer must be in the initial state (freshly allocated or reset).
    pub unsafe fn begin(&mut self) -> Result<(), vk::Result> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: Caller guarantees the buffer is in the initial state.
        unsafe {
            self.parent
                .begin_raw_command_buffer(self.handle, &begin_info)
        }
    }

    /// End recording.
    ///
    /// # Safety
    /// The buffer must be in the recording state.
    pub unsafe fn end(&mut self) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees the buffer is in the recording state.
        unsafe { self.parent.end_raw_command_buffer(self.handle) }
    }

    /// Record a pipeline barrier using the synchronization2 API.
    ///
    /// # Safety
    /// The buffer must be in the recording state. All handles in
    /// `dependency_info` must be valid and consistent with current state.
    pub unsafe fn pipeline_barrier2(
        &mut self,
        dependency_info: &vk::DependencyInfo<'_>,
    ) {
        // SAFETY: Caller guarantees recording state and
        // dependency_info validity.
        unsafe {
            self.parent
                .cmd_pipeline_barrier2(self.handle, dependency_info)
        }
    }

    /// Begin a dynamic render pass.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `rendering_info` and all
    /// objects it references must be valid for the duration of the render pass.
    /// All images must already be in the layouts declared in `rendering_info`.
    pub unsafe fn begin_rendering(
        &mut self,
        rendering_info: &vk::RenderingInfo<'_>,
    ) -> Result<(), DynamicRenderingError> {
        // SAFETY: Caller guarantees recording state and
        // rendering_info validity.
        unsafe {
            self.parent
                .cmd_begin_raw_rendering(self.handle, rendering_info)
        }
    }

    /// End the current dynamic render pass.
    ///
    /// # Safety
    /// The buffer must be inside a render pass begun with
    /// [`begin_rendering`](Self::begin_rendering).
    pub unsafe fn end_rendering(
        &mut self,
    ) -> Result<(), DynamicRenderingError> {
        // SAFETY: Caller guarantees active render pass state.
        unsafe { self.parent.cmd_end_raw_rendering(self.handle) }
    }

    /// Bind a graphics pipeline for subsequent draw commands.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `pipeline` must be a valid
    /// graphics pipeline created from the same device as this buffer.
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: vk::Pipeline) {
        // SAFETY: Caller guarantees recording state and pipeline validity.
        unsafe {
            self.parent
                .cmd_bind_graphics_pipeline(self.handle, pipeline)
        }
    }

    /// Bind vertex buffers for subsequent draw commands.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `buffers` and `offsets`
    /// must have equal length. All buffers must be valid handles created from
    /// the same device as this command buffer.
    pub unsafe fn bind_raw_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        // SAFETY: Caller guarantees recording state and buffer validity.
        unsafe {
            self.parent.cmd_bind_vertex_buffers(
                self.handle,
                first_binding,
                buffers,
                offsets,
            )
        }
    }

    /// Bind vertex buffers for subsequent draw commands.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `buffers` and `offsets`
    /// must have equal length. All buffers must be valid handles created from
    /// the same device as this command buffer.
    pub unsafe fn bind_vertex_buffers<B>(
        &mut self,
        first_binding: u32,
        buffers: &[B],
        offsets: &[vk::DeviceSize],
    ) where
        B: BufferHandle,
    {
        let raw_buffers: Vec<vk::Buffer> =
            buffers.iter().map(|b| b.raw_buffer()).collect();
        // SAFETY: Caller guarantees recording state and buffer validity.
        unsafe {
            self.bind_raw_vertex_buffers(first_binding, &raw_buffers, offsets)
        }
    }

    /// Bind heterogeneous vertex buffers for subsequent draw commands.
    ///
    /// This overload accepts mixed wrapper types through trait objects.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `buffers` and `offsets`
    /// must have equal length. All buffers must be valid handles created from
    /// the same device as this command buffer.
    pub unsafe fn bind_heterogenous_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[&dyn BufferHandle],
        offsets: &[vk::DeviceSize],
    ) {
        let raw_buffers: Vec<vk::Buffer> =
            buffers.iter().map(|b| b.raw_buffer()).collect();
        // SAFETY: Caller guarantees recording state and buffer validity.
        unsafe {
            self.bind_raw_vertex_buffers(first_binding, &raw_buffers, offsets)
        }
    }

    /// Bind a single vertex buffer for subsequent draw commands.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `buffer` must be a valid
    /// handle created from the same device as this command buffer.
    pub unsafe fn bind_vertex_buffer<B>(
        &mut self,
        binding: u32,
        buffer: B,
        offset: vk::DeviceSize,
    ) where
        B: BufferHandle,
    {
        let buffers = [buffer];
        let offsets = [offset];
        // SAFETY: Caller guarantees recording state and buffer validity.
        unsafe { self.bind_vertex_buffers(binding, &buffers, &offsets) }
    }

    /// Record a buffer-to-buffer copy.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `src_buffer` and
    /// `dst_buffer` must be valid handles created from the same device as
    /// this command buffer. Regions must be valid and in-bounds.
    pub unsafe fn copy_buffer(
        &mut self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        // SAFETY: Caller guarantees recording state and copy validity.
        unsafe {
            self.parent.cmd_copy_buffer(
                self.handle,
                src_buffer,
                dst_buffer,
                regions,
            )
        }
    }

    /// Set the viewport dynamically.
    ///
    /// # Safety
    /// The buffer must be in the recording state with a pipeline bound that
    /// declares `VK_DYNAMIC_STATE_VIEWPORT`.
    pub unsafe fn set_viewport(&mut self, viewports: &[vk::Viewport]) {
        // SAFETY: Caller guarantees recording state and dynamic
        // viewport pipeline.
        unsafe { self.parent.cmd_set_viewport(self.handle, viewports) }
    }

    /// Set the scissor rectangle dynamically.
    ///
    /// # Safety
    /// The buffer must be in the recording state with a pipeline bound that
    /// declares `VK_DYNAMIC_STATE_SCISSOR`.
    pub unsafe fn set_scissor(&mut self, scissors: &[vk::Rect2D]) {
        // SAFETY: Caller guarantees recording state and dynamic
        // scissor pipeline.
        unsafe { self.parent.cmd_set_scissor(self.handle, scissors) }
    }

    /// Record a non-indexed draw call.
    ///
    /// # Safety
    /// The buffer must be in the recording state inside an active render pass,
    /// with a compatible graphics pipeline bound and all required dynamic
    /// state set.
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        // SAFETY: Caller guarantees render pass and pipeline state validity.
        unsafe {
            self.parent.cmd_draw(
                self.handle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        }
    }

    /// Bind an index buffer for subsequent indexed draw commands.
    ///
    /// # Safety
    /// The buffer must be in the recording state. `buffer` must be a
    /// valid index buffer created from the same device as this command
    /// buffer, with `INDEX_BUFFER` usage.
    pub unsafe fn bind_index_buffer<B>(
        &mut self,
        buffer: B,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) where
        B: BufferHandle,
    {
        // SAFETY: Caller guarantees recording state and buffer validity.
        unsafe {
            self.parent.cmd_bind_index_buffer(
                self.handle,
                buffer.raw_buffer(),
                offset,
                index_type,
            )
        }
    }

    /// Record an indexed draw call.
    ///
    /// # Safety
    /// The buffer must be in the recording state inside an active render
    /// pass, with a compatible graphics pipeline bound, all required
    /// dynamic state set, and a valid index buffer bound.
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // SAFETY: Caller guarantees render pass, pipeline, and
        // index buffer state validity.
        unsafe {
            self.parent.cmd_draw_indexed(
                self.handle,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
    }

    pub fn raw_command_buffer(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl CommandBufferHandle for ResettableCommandBuffer {
    fn raw_command_buffer(&self) -> vk::CommandBuffer {
        self.handle
    }
}

// ---------------------------------------------------------------------------
// Auto-trait assertions
// ---------------------------------------------------------------------------

// Verified at compile time: both types are Send.
// ResettableCommandPool: Send + !Sync (Receiver/Sender/PhantomData<Cell<()>>)
// ResettableCommandBuffer: Send + !Sync (Sender<T>: !Sync)
#[allow(dead_code)]
trait AssertSend: Send {}
impl AssertSend for ResettableCommandPool {}
impl AssertSend for ResettableCommandBuffer {}
