//! Command pool and command buffer wrappers.
//!
//! ## Pool variants
//!
//! | Type | Pool flags | Buffer lifetime |
//! |---|---|---|
//! | [`ResettableCommandPool<Q>`] | `RESET_COMMAND_BUFFER` | Reset individually |
//! | [`TransientCommandPool<Q>`] | `TRANSIENT` | Freed on pool reset |
//!
//! ## Queue capability markers
//!
//! All pool and buffer types carry a queue-capability marker type
//! parameter `Q`:
//!
//! | Marker | Capabilities |
//! |---|---|
//! | [`Graphics`] | graphics + compute + transfer |
//! | [`Compute`] | compute + transfer |
//! | [`Transfer`] | transfer only |
//!
//! Recording methods on [`Recorder`] are gated by [`SupportsGraphics`],
//! [`SupportsCompute`], and [`SupportsTransfer`] trait bounds as
//! appropriate. Copy and barrier operations require [`SupportsTransfer`];
//! debug-label commands are ungated.
//!
//! ## Recording API
//!
//! Both buffer types implement [`Recordable<Q>`], which
//! exposes [`begin_recording`](Recordable::begin_recording).
//! That method returns a [`Recorder`], a RAII guard that owns the
//! recording session. **`Recorder` panics if dropped without calling
//! [`end_recording`](Recorder::end_recording).**
//!
//! ```ignore
//! let mut rec = cmd.begin_recording();
//! unsafe { rec.draw(3, 1, 0, 0) };
//! rec.end_recording();
//! ```
//!
//! ## Extensibility
//!
//! [`Recordable<Q>`], [`SupportsGraphics`], and
//! [`SupportsCompute`] are **not sealed**. External crates may implement
//! them to integrate custom pool or synchronisation strategies anywhere
//! a standard pool or recorder is accepted. See `rgpu-vk/README.md` for
//! the full design policy.

use std::{marker::PhantomData, sync::Arc};

use ash::vk;
use thiserror::Error;

use crate::buffer::BufferHandle;
use crate::descriptor::DescriptorSet;
use crate::device::{
    Device, QueueFamily, Submittable, SupportsGraphics, SupportsTransfer,
};
use crate::pipeline::PipelineLayout;

// ---------------------------------------------------------------------------
// State tracking
// ---------------------------------------------------------------------------

/// The recording state of a command buffer.
///
/// Transitions:
/// - `Reset` → `Recording` via
///   [`Recordable::begin_recording`]
/// - `Recording` → `Recorded` via
///   [`Recorder::end_recording`]
/// - Any → `Reset` via [`ResettableCommandBuffer::reset`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferState {
    /// Initial / post-reset state. The buffer is ready to begin recording.
    Reset,
    /// The buffer is between `begin_recording()` and `end_recording()`.
    /// Recording commands may be issued.
    Recording,
    /// Recording has ended. The buffer is executable or pending on the GPU.
    Recorded,
}

/// Error returned when a state-transition method is called from the
/// wrong [`CommandBufferState`].
#[derive(Debug, Error)]
#[error("command buffer in {actual:?} state, expected {expected:?}")]
pub struct CommandBufferStateError {
    pub expected: CommandBufferState,
    pub actual: CommandBufferState,
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

// ---------------------------------------------------------------------------
// Debug-utils label helpers
// ---------------------------------------------------------------------------

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum DebugLabelType {
    GraphicsPass,
    ComputePass,
    GraphicsSubpass,
    ComputeSubpass,
    ImageCopy,
    BufferCopy,
    ImageUpload,
    BufferUpload,
    ImageDownload,
    BufferDownload,
    Other,
}
type DebugUtilColor = [f32; 4];

// NOTE: I haven't tested these colors on light screens. I probably
// don't actually care but if someone else thinks the values are off,
// feel free to tweak the saturation and I'll check if it works.
const COLOR_GREEN: DebugUtilColor = [0.0, 1.0, 0.0, 1.0];
const COLOR_BLUE: DebugUtilColor = [0.0, 0.0, 1.0, 1.0];
const COLOR_CYAN: DebugUtilColor = [0.0, 1.0, 1.0, 1.0];
const COLOR_MAGENTA: DebugUtilColor = [0.9, 0.0, 0.9, 1.0];
pub const COLOR_YELLOW_DARK: DebugUtilColor = [0.502, 0.400, 0.000, 1.0];
pub const COLOR_YELLOW_MEDIUM: DebugUtilColor = [0.800, 0.722, 0.000, 1.0];
pub const COLOR_YELLOW_LIGHT: DebugUtilColor = [1.000, 0.878, 0.200, 1.0];
const COLOR_GREY: DebugUtilColor = [0.7, 0.7, 0.7, 1.0];

impl DebugLabelType {
    fn to_color(self) -> [f32; 4] {
        match self {
            DebugLabelType::GraphicsPass => COLOR_GREEN,
            DebugLabelType::ComputePass => COLOR_CYAN,
            DebugLabelType::GraphicsSubpass => COLOR_BLUE,
            DebugLabelType::ComputeSubpass => COLOR_MAGENTA,
            DebugLabelType::Other => COLOR_GREY,
            // Yellow labels are for data-movement operations.
            // Darkness signals CPU involvement: GPU→GPU copies are fine,
            // uploads require CPU sync beforehand, downloads block the CPU
            // afterward (rare — mostly screenshots).
            DebugLabelType::ImageUpload => COLOR_YELLOW_MEDIUM,
            DebugLabelType::BufferUpload => COLOR_YELLOW_MEDIUM,
            DebugLabelType::ImageCopy => COLOR_YELLOW_LIGHT,
            DebugLabelType::BufferCopy => COLOR_YELLOW_LIGHT,
            DebugLabelType::ImageDownload => COLOR_YELLOW_DARK,
            DebugLabelType::BufferDownload => COLOR_YELLOW_DARK,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DebugLabel<'a> {
    pub name: &'a str,
    pub ty: DebugLabelType,
}

#[derive(Clone, Copy, Debug)]
pub struct LazyDebugLabelReturn<NameRef: AsRef<str>> {
    pub name_ref: NameRef,
    pub ty: DebugLabelType,
}

// ---------------------------------------------------------------------------
// Recordable trait
// ---------------------------------------------------------------------------

/// Primary trait for command buffers that can be recorded.
///
/// Implementors call `vkBeginCommandBuffer` in `begin_recording` and
/// update their state in `on_end_recording`. Both methods are provided
/// so that state-tracking logic lives in the buffer type, not in
/// [`Recorder`].
///
/// # Contract for implementors
///
/// - `begin_recording` **must** call `vkBeginCommandBuffer` (or an
///   equivalent) before constructing and returning the `Recorder`.
/// - `on_end_recording` is called by [`Recorder::end_recording`] after
///   `vkEndCommandBuffer` succeeds. Use it to transition internal state
///   (e.g. to [`CommandBufferState::Recorded`]), or leave it as a no-op
///   if state tracking is not needed.
/// - `raw` and `parent` are used by [`Recorder`] to issue Vulkan
///   commands; they must return the buffer's underlying handle and
///   parent device respectively.
///
/// # Extensibility
///
/// This trait is **not sealed**. Downstream crates may implement it for
/// custom pool or synchronisation strategies and pass their buffers
/// anywhere a standard pool is accepted.
pub trait Recordable<Q> {
    /// Called by [`Recorder::end_recording`] after `vkEndCommandBuffer`
    /// succeeds. Update internal state here (or leave as no-op).
    fn on_end_recording(&mut self);

    /// Validate preconditions if desired, call `vkBeginCommandBuffer`,
    /// and return a [`Recorder`].
    ///
    /// State tracking and precondition checking are the implementor's
    /// responsibility; [`Recorder`] itself does not inspect state.
    fn begin_recording(&mut self) -> Recorder<'_, Q, Self>
    where
        Self: Sized;

    /// The raw `vk::CommandBuffer` handle for this buffer.
    ///
    /// Used by [`Recorder`] to issue Vulkan commands. Must remain valid
    /// for the lifetime of the borrow.
    fn raw(&self) -> vk::CommandBuffer;

    /// The parent [`Device`] this buffer was allocated from.
    ///
    /// Used by [`Recorder`] to dispatch Vulkan commands. Must remain
    /// valid for the lifetime of the borrow.
    fn parent(&self) -> &Arc<Device>;
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// RAII guard for an active command buffer recording session.
///
/// Obtained from [`Recordable::begin_recording`]. All
/// recording commands are issued through this type.
///
/// # Drop behaviour
///
/// **`Recorder` panics if dropped without calling
/// [`end_recording`](Self::end_recording).** `end_recording` consumes
/// the recorder and calls `std::mem::forget` on it so the destructor
/// is not invoked.
///
/// # Capability bounds
///
/// Methods are gated by the queue capability marker `Q`:
/// - `Q: SupportsTransfer`: barrier and copy operations.
/// - `Q: SupportsGraphics`: render-pass, draw, and pipeline-bind
///   commands.
/// - `Q: SupportsCompute`: reserved for future compute dispatch.
/// - No bound: debug-label commands (independent of queue capability).
pub struct Recorder<'a, Q, B: Recordable<Q>> {
    buffer: &'a mut B,
    _queue: PhantomData<Q>,
}

impl<Q, B: Recordable<Q>> Drop for Recorder<'_, Q, B> {
    fn drop(&mut self) {
        panic!(
            "Recorder dropped without calling end_recording(). \
             Call end_recording() to finish recording."
        );
    }
}

impl<'a, Q, B: Recordable<Q>> Recorder<'a, Q, B> {
    // Internal constructor — only buffer impls create Recorders.
    pub(crate) fn new(buffer: &'a mut B) -> Self {
        Self {
            buffer,
            _queue: PhantomData,
        }
    }

    /// Finish recording, call `vkEndCommandBuffer`, and notify the
    /// originating buffer via
    /// [`on_end_recording`](Recordable::on_end_recording).
    ///
    /// Uses `std::mem::forget` to prevent the panicking `Drop` impl
    /// from running.
    pub fn end_recording(self) {
        // SAFETY: begin_recording guarantees vkBeginCommandBuffer was
        // called, so the buffer is in the recording state.
        unsafe {
            self.buffer
                .parent()
                .end_raw_command_buffer(self.buffer.raw())
        }
        .expect("vkEndCommandBuffer failed");
        self.buffer.on_end_recording();
        // Bypass the panicking Drop.
        std::mem::forget(self);
    }

    // -----------------------------------------------------------------------
    // Ungated commands — valid on all queue types
    // -----------------------------------------------------------------------

    /// Record a pipeline barrier using the synchronization2 API.
    ///
    /// # Safety
    /// `dependency_info` must be valid and consistent with current
    /// resource state.
    pub unsafe fn pipeline_barrier2(
        &mut self,
        dependency_info: &vk::DependencyInfo<'_>,
    ) where
        Q: SupportsTransfer,
    {
        // SAFETY: Forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_pipeline_barrier2(self.buffer.raw(), dependency_info)
        }
    }

    /// Record a synchronization2 pipeline barrier from individual
    /// barrier slices.
    ///
    /// # Safety
    /// All barrier objects must be valid and consistent with current
    /// resource state.
    pub unsafe fn pipeline_barrier2_by_barriers(
        &mut self,
        memory_barriers: &[vk::MemoryBarrier2<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier2<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrier2<'_>],
    ) where
        Q: SupportsTransfer,
    {
        let dep_info = vk::DependencyInfo::default()
            .memory_barriers(memory_barriers)
            .buffer_memory_barriers(buffer_memory_barriers)
            .image_memory_barriers(image_memory_barriers);
        // SAFETY: forwarded to caller.
        unsafe { self.pipeline_barrier2(&dep_info) };
    }

    /// Record a synchronization2 pipeline barrier from a pre-built
    /// `vk::DependencyInfo`.
    ///
    /// # Safety
    /// `dependency_info` must be valid for the current command buffer
    /// state.
    pub unsafe fn pipeline_barrier2_by_dep_info(
        &mut self,
        dependency_info: &vk::DependencyInfo<'_>,
    ) where
        Q: SupportsTransfer,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_pipeline_barrier2(self.buffer.raw(), dependency_info)
        }
    }

    /// Record an old-style pipeline barrier (`vkCmdPipelineBarrier`).
    ///
    /// # Safety
    /// All handles and image layouts must be valid and consistent with
    /// current resource state.
    pub unsafe fn pipeline_barrier(
        &mut self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) where
        Q: SupportsTransfer,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_pipeline_barrier(
                self.buffer.raw(),
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            )
        }
    }

    /// Record a buffer-to-buffer copy.
    ///
    /// # Safety
    /// `src_buffer` and `dst_buffer` must be valid handles from the
    /// same device. Regions must be valid and in-bounds.
    pub unsafe fn copy_buffer(
        &mut self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) where
        Q: SupportsTransfer,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_copy_buffer(
                self.buffer.raw(),
                src_buffer,
                dst_buffer,
                regions,
            )
        }
    }

    /// Record a buffer-to-image copy.
    ///
    /// # Safety
    /// `src_buffer` and `dst_image` must be valid handles from the
    /// same device. The image must be in `dst_image_layout` for the
    /// duration of the copy. Regions must be valid and in-bounds.
    pub unsafe fn copy_buffer_to_image(
        &mut self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) where
        Q: SupportsTransfer,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_copy_buffer_to_image(
                self.buffer.raw(),
                src_buffer,
                dst_image,
                dst_image_layout,
                regions,
            )
        }
    }

    /// Begin a debug label region on this command buffer.
    ///
    /// No-op when `VK_EXT_debug_utils` is not enabled.
    ///
    /// # Safety
    /// Every call must be matched by a corresponding
    /// [`end_debug_label`](Self::end_debug_label) before the buffer is
    /// submitted.
    pub unsafe fn begin_debug_label(&mut self, label: DebugLabel) {
        use std::ffi::CString;
        // SAFETY: CString::new only fails on interior NUL bytes;
        // a label from &str has none.
        let c_label = CString::new(label.name).unwrap_or_default();
        // SAFETY: handle is valid and recording. c_label is valid UTF-8.
        unsafe {
            self.buffer.parent().cmd_begin_debug_label_cstr(
                self.buffer.raw(),
                Some(&c_label),
                label.ty.to_color(),
            )
        }
    }

    /// Lazily begin a debug label region, constructing the string only
    /// if `VK_EXT_debug_utils` is enabled.
    ///
    /// # Safety
    /// Every call must be matched by a corresponding
    /// [`end_debug_label`](Self::end_debug_label) before the buffer is
    /// submitted.
    pub unsafe fn begin_debug_label_lazy<LabelFn, StringRef>(
        &mut self,
        f: LabelFn,
    ) where
        LabelFn: FnOnce() -> LazyDebugLabelReturn<StringRef>,
        StringRef: AsRef<str>,
    {
        if self.buffer.parent().debug_utils_enabled() {
            let label = f();
            // SAFETY: valid UTF-8 (from AsRef<str>); recording state
            // guaranteed by the caller's context.
            unsafe {
                self.begin_debug_label(DebugLabel {
                    name: label.name_ref.as_ref(),
                    ty: label.ty,
                })
            };
        }
    }

    /// End the most recently begun debug label region.
    ///
    /// No-op when `VK_EXT_debug_utils` is not enabled.
    ///
    /// # Safety
    /// A matching [`begin_debug_label`](Self::begin_debug_label) must
    /// have been recorded previously.
    pub unsafe fn end_debug_label(&mut self) {
        // SAFETY: forwarded to caller.
        unsafe { self.buffer.parent().end_cmd_debug_label(self.buffer.raw()) }
    }

    // -----------------------------------------------------------------------
    // Graphics-only commands
    // -----------------------------------------------------------------------

    /// Begin a dynamic render pass.
    ///
    /// # Safety
    /// `rendering_info` and all objects it references must be valid.
    /// All images must already be in the layouts declared in
    /// `rendering_info`.
    pub unsafe fn begin_rendering(
        &mut self,
        rendering_info: &vk::RenderingInfo<'_>,
    ) where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_begin_raw_rendering(self.buffer.raw(), rendering_info)
        }
    }

    /// End the current dynamic render pass.
    ///
    /// # Safety
    /// The buffer must be inside a render pass begun with
    /// [`begin_rendering`](Self::begin_rendering).
    pub unsafe fn end_rendering(&mut self)
    where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_end_raw_rendering(self.buffer.raw())
        }
    }

    /// Begin a render pass.
    ///
    /// # Safety
    /// All objects referenced by `render_pass_begin` must be valid and
    /// derived from the same device as this buffer.
    pub unsafe fn begin_render_pass(
        &mut self,
        render_pass_begin: &vk::RenderPassBeginInfo<'_>,
        contents: vk::SubpassContents,
    ) where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_begin_render_pass(
                self.buffer.raw(),
                render_pass_begin,
                contents,
            )
        }
    }

    /// End the current render pass.
    ///
    /// # Safety
    /// The buffer must be inside a render pass begun with
    /// [`begin_render_pass`](Self::begin_render_pass).
    pub unsafe fn end_render_pass(&mut self)
    where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe { self.buffer.parent().cmd_end_render_pass(self.buffer.raw()) }
    }

    /// Bind a graphics pipeline for subsequent draw commands.
    ///
    /// # Safety
    /// `pipeline` must be a valid graphics pipeline created from the
    /// same device as this buffer.
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: vk::Pipeline)
    where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_bind_graphics_pipeline(self.buffer.raw(), pipeline)
        }
    }

    /// Bind vertex buffers for subsequent draw commands.
    ///
    /// # Safety
    /// `buffers` and `offsets` must have equal length. All buffers must
    /// be valid handles from the same device.
    pub unsafe fn bind_raw_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_bind_vertex_buffers(
                self.buffer.raw(),
                first_binding,
                buffers,
                offsets,
            )
        }
    }

    /// Bind typed vertex buffers for subsequent draw commands.
    ///
    /// # Safety
    /// `buffers` and `offsets` must have equal length. All buffers must
    /// be valid handles from the same device.
    pub unsafe fn bind_vertex_buffers<Buf>(
        &mut self,
        first_binding: u32,
        buffers: &[Buf],
        offsets: &[vk::DeviceSize],
    ) where
        Q: SupportsGraphics,
        Buf: BufferHandle,
    {
        let raw_buffers: Vec<vk::Buffer> =
            buffers.iter().map(|b| b.raw_buffer()).collect();
        // SAFETY: forwarded to caller.
        unsafe {
            self.bind_raw_vertex_buffers(first_binding, &raw_buffers, offsets)
        }
    }

    /// Bind heterogeneous vertex buffers through trait objects.
    ///
    /// # Safety
    /// `buffers` and `offsets` must have equal length. All buffers must
    /// be valid handles from the same device.
    pub unsafe fn bind_heterogenous_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[&dyn BufferHandle],
        offsets: &[vk::DeviceSize],
    ) where
        Q: SupportsGraphics,
    {
        let raw_buffers: Vec<vk::Buffer> =
            buffers.iter().map(|b| b.raw_buffer()).collect();
        // SAFETY: forwarded to caller.
        unsafe {
            self.bind_raw_vertex_buffers(first_binding, &raw_buffers, offsets)
        }
    }

    /// Bind a single vertex buffer.
    ///
    /// # Safety
    /// `buffer` must be a valid handle from the same device.
    pub unsafe fn bind_vertex_buffer<Buf>(
        &mut self,
        binding: u32,
        buffer: Buf,
        offset: vk::DeviceSize,
    ) where
        Q: SupportsGraphics,
        Buf: BufferHandle,
    {
        let buffers = [buffer];
        let offsets = [offset];
        // SAFETY: forwarded to caller.
        unsafe { self.bind_vertex_buffers(binding, &buffers, &offsets) }
    }

    /// Bind an index buffer for subsequent indexed draw commands.
    ///
    /// # Safety
    /// `buffer` must be a valid index buffer with `INDEX_BUFFER` usage
    /// from the same device.
    pub unsafe fn bind_index_buffer<Buf>(
        &mut self,
        buffer: Buf,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) where
        Q: SupportsGraphics,
        Buf: BufferHandle,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_bind_index_buffer(
                self.buffer.raw(),
                buffer.raw_buffer(),
                offset,
                index_type,
            )
        }
    }

    /// Set the viewport dynamically.
    ///
    /// # Safety
    /// A pipeline with `VK_DYNAMIC_STATE_VIEWPORT` must be bound.
    pub unsafe fn set_viewport(&mut self, viewports: &[vk::Viewport])
    where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_set_viewport(self.buffer.raw(), viewports)
        }
    }

    /// Set the scissor rectangle dynamically.
    ///
    /// # Safety
    /// A pipeline with `VK_DYNAMIC_STATE_SCISSOR` must be bound.
    pub unsafe fn set_scissor(&mut self, scissors: &[vk::Rect2D])
    where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer
                .parent()
                .cmd_set_scissor(self.buffer.raw(), scissors)
        }
    }

    /// Record a non-indexed draw call.
    ///
    /// # Safety
    /// The buffer must be inside an active render pass with a compatible
    /// graphics pipeline bound and all required dynamic state set.
    pub unsafe fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_draw(
                self.buffer.raw(),
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            )
        }
    }

    /// Record an indexed draw call.
    ///
    /// # Safety
    /// The buffer must be inside an active render pass with a compatible
    /// graphics pipeline bound, all required dynamic state set, and a
    /// valid index buffer bound.
    pub unsafe fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) where
        Q: SupportsGraphics,
    {
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_draw_indexed(
                self.buffer.raw(),
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
    }

    /// Bind descriptor sets for subsequent draw or dispatch commands.
    ///
    /// # Safety
    /// `layout` must be compatible with the pipeline that will be used.
    /// All sets must be valid and allocated from a pool on the same
    /// device.
    pub unsafe fn bind_descriptor_sets(
        &self,
        layout: &PipelineLayout,
        first_set: u32,
        descriptor_sets: &[&DescriptorSet],
    ) where
        Q: SupportsGraphics,
    {
        let raw_sets: Vec<vk::DescriptorSet> = descriptor_sets
            .iter()
            .map(|s| s.raw_descriptor_set())
            .collect();
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_bind_descriptor_sets(
                self.buffer.raw(),
                layout.raw_pipeline_layout(),
                first_set,
                &raw_sets,
                &[],
            )
        }
    }

    /// Upload push-constant data into the command buffer.
    ///
    /// # Safety
    /// - `layout` must be compatible with the bound pipeline.
    /// - `stage_flags` and `offset` must match a push constant range
    ///   declared in `layout`.
    /// - The byte length of `values` must not exceed the range size.
    pub unsafe fn push_constants<T: bytemuck::Pod>(
        &mut self,
        layout: &PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        values: &[T],
    ) where
        Q: SupportsGraphics,
    {
        // bytemuck operates on CPU-side stack data, not mapped GPU
        // memory, so no aliasing issue arises.
        let bytes = bytemuck::cast_slice(values);
        // SAFETY: forwarded to caller.
        unsafe {
            self.buffer.parent().cmd_push_constants(
                self.buffer.raw(),
                layout.raw_pipeline_layout(),
                stage_flags,
                offset,
                bytes,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// ResettableCommandPool
// ---------------------------------------------------------------------------

/// A command pool that allocates individually-resettable command buffers.
///
/// Created with `RESET_COMMAND_BUFFER`, allowing each allocated buffer to
/// be reset individually via [`ResettableCommandBuffer::reset`].
///
/// `Q` is the queue capability marker (`Graphics`, `Compute`, or
/// `Transfer`). Pools and buffers carry this marker so that recording
/// methods can be statically gated.
///
/// The pool is `!Sync`: Vulkan requires external synchronization for
/// pool-level operations.
pub struct ResettableCommandPool<Q> {
    parent: Arc<Device>,
    pool: vk::CommandPool,
    _not_sync: crate::marker::PhantomUnsync,
    _queue: PhantomData<Q>,
}

impl<Q> std::fmt::Debug for ResettableCommandPool<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResettableCommandPool")
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl<Q> Drop for ResettableCommandPool<Q> {
    fn drop(&mut self) {
        tracing::debug!("Dropping command pool {:?}", self.pool);
        // SAFETY: pool was created from parent and is no longer in use.
        unsafe { self.parent.destroy_raw_command_pool(self.pool) };
    }
}

impl<Q: QueueFamily> ResettableCommandPool<Q> {
    /// Create a resettable command pool for queue capability `Q`.
    ///
    /// The queue family index is resolved automatically from `device`
    /// via [`QueueFamily`]. `name` is an optional debug label applied
    /// via `VK_EXT_debug_utils`. Naming failures are logged and do not
    /// cause the call to fail.
    pub fn new(
        device: &Arc<Device>,
        name: Option<&str>,
    ) -> Result<Self, CreateCommandPoolError> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(Q::family(device))
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: create_info uses a valid queue family index.
        let pool = unsafe { device.create_raw_command_pool(&create_info) }
            .map_err(CreateCommandPoolError::Vulkan)?;

        // SAFETY: pool is a valid command pool created from device.
        let name_result = unsafe { device.set_object_name_str(pool, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name command pool {:?}: {e}", pool);
        }

        Ok(Self {
            parent: Arc::clone(device),
            pool,
            _not_sync: Default::default(),
            _queue: PhantomData,
        })
    }

    /// Allocate a single primary command buffer from this pool.
    ///
    /// The returned buffer is in the
    /// [`Reset`](CommandBufferState::Reset) state and must be freed
    /// via [`ResettableCommandPool::free`] when no longer needed.
    pub fn allocate_command_buffer(
        &mut self,
    ) -> Result<ResettableCommandBuffer<Q>, AllocateCommandBufferError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: allocate_info references a valid pool from parent.
        // !Sync prevents concurrent pool access.
        let handle =
            unsafe { self.parent.allocate_raw_command_buffers(&allocate_info) }
                .map(|mut bufs| {
                    debug_assert_eq!(bufs.len(), 1);
                    bufs.remove(0)
                })
                .map_err(AllocateCommandBufferError::Vulkan)?;

        Ok(ResettableCommandBuffer {
            parent: Arc::clone(&self.parent),
            handle,
            state: CommandBufferState::Reset,
            _queue: PhantomData,
        })
    }

    /// Free a single command buffer allocated from this pool.
    ///
    /// # Safety
    /// - `buf` must have been allocated from this pool.
    /// - `buf` must not be pending execution on the GPU.
    pub unsafe fn free(&mut self, buf: ResettableCommandBuffer<Q>) {
        // SAFETY: buf was allocated from self.pool; caller guarantees
        // it is not pending.
        unsafe {
            self.parent
                .free_raw_command_buffers(self.pool, &[buf.handle])
        };
        std::mem::forget(buf);
    }

    #[inline]
    pub fn raw_command_pool(&self) -> vk::CommandPool {
        self.pool
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

// ---------------------------------------------------------------------------
// ResettableCommandBuffer
// ---------------------------------------------------------------------------

/// A primary command buffer allocated from a [`ResettableCommandPool<Q>`].
///
/// [`CommandBufferState`] is tracked at runtime. `reset()` is `unsafe`
/// because the GPU-not-pending guarantee cannot be checked at runtime.
///
/// Must be freed via [`ResettableCommandPool::free`] before the pool
/// is dropped. Dropping without freeing leaks the Vulkan handle.
pub struct ResettableCommandBuffer<Q> {
    parent: Arc<Device>,
    handle: vk::CommandBuffer,
    state: CommandBufferState,
    _queue: PhantomData<Q>,
}

impl<Q> Drop for ResettableCommandBuffer<Q> {
    fn drop(&mut self) {
        tracing::warn!(
            "ResettableCommandBuffer {:?} dropped without being \
             freed — Vulkan handle leaked",
            self.handle
        );
    }
}

impl<Q> std::fmt::Debug for ResettableCommandBuffer<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResettableCommandBuffer")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl<Q> ResettableCommandBuffer<Q> {
    /// Reset this buffer to the [`Reset`](CommandBufferState::Reset)
    /// state.
    ///
    /// # Safety
    /// The buffer must not be pending execution on the GPU.
    pub unsafe fn reset(&mut self) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees the buffer is not pending.
        let result = unsafe {
            self.parent.reset_raw_command_buffer(
                self.handle,
                vk::CommandBufferResetFlags::empty(),
            )
        };
        if result.is_ok() {
            self.state = CommandBufferState::Reset;
        }
        result
    }

    /// Returns the current [`CommandBufferState`].
    #[inline]
    pub fn state(&self) -> CommandBufferState {
        self.state
    }

    #[inline]
    pub fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl<Q> Recordable<Q> for ResettableCommandBuffer<Q> {
    fn on_end_recording(&mut self) {
        self.state = CommandBufferState::Recorded;
    }

    /// Begin recording.
    ///
    /// # Panics
    /// Panics if the buffer is not in the
    /// [`Reset`](CommandBufferState::Reset) state.
    fn begin_recording(&mut self) -> Recorder<'_, Q, Self> {
        assert_eq!(
            self.state,
            CommandBufferState::Reset,
            "ResettableCommandBuffer::begin_recording called in \
             {:?} state; reset() first",
            self.state
        );
        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: state == Reset guarantees the buffer is in Vulkan's
        // initial state, which is the precondition for
        // vkBeginCommandBuffer.
        unsafe {
            self.parent
                .begin_raw_command_buffer(self.handle, &begin_info)
        }
        .expect("vkBeginCommandBuffer failed");
        self.state = CommandBufferState::Recording;
        Recorder::new(self)
    }

    fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }

    fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl<Q> Submittable<Q> for ResettableCommandBuffer<Q> {
    fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }
}

// ---------------------------------------------------------------------------
// TransientCommandPool
// ---------------------------------------------------------------------------

/// A command pool for short-lived, one-shot command buffers.
///
/// Created with `TRANSIENT_BIT` only (no `RESET_COMMAND_BUFFER`).
/// All buffers allocated from this pool are freed atomically when the
/// pool is reset via [`TransientCommandPool::reset`]; there is no
/// per-buffer reset.
///
/// Allocation is `unsafe` because the caller must guarantee that no
/// [`TransientCommandBuffer`] previously allocated from this pool is
/// still live (in use or referenced) when the pool is reset.
///
/// `Q` is the queue capability marker (`Graphics`, `Compute`, or
/// `Transfer`).
pub struct TransientCommandPool<Q> {
    parent: Arc<Device>,
    pool: vk::CommandPool,
    _not_sync: crate::marker::PhantomUnsync,
    _queue: PhantomData<Q>,
}

impl<Q> std::fmt::Debug for TransientCommandPool<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransientCommandPool")
            .field("pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl<Q> Drop for TransientCommandPool<Q> {
    fn drop(&mut self) {
        tracing::debug!("Dropping command pool {:?}", self.pool);
        // SAFETY: pool was created from parent and is no longer in use.
        unsafe { self.parent.destroy_raw_command_pool(self.pool) };
    }
}

impl<Q: QueueFamily> TransientCommandPool<Q> {
    /// Create a transient command pool for queue capability `Q`.
    ///
    /// The queue family index is resolved automatically from `device`
    /// via [`QueueFamily`]. `name` is an optional debug label. Naming
    /// failures are logged and do not cause the call to fail.
    pub fn new(
        device: &Arc<Device>,
        name: Option<&str>,
    ) -> Result<Self, CreateCommandPoolError> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(Q::family(device))
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        // SAFETY: create_info uses a valid queue family index.
        let pool = unsafe { device.create_raw_command_pool(&create_info) }
            .map_err(CreateCommandPoolError::Vulkan)?;

        // SAFETY: pool is a valid command pool created from device.
        let name_result = unsafe { device.set_object_name_str(pool, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name command pool {:?}: {e}", pool);
        }

        Ok(Self {
            parent: Arc::clone(device),
            pool,
            _not_sync: Default::default(),
            _queue: PhantomData,
        })
    }

    /// Allocate a single primary command buffer.
    ///
    /// The returned buffer is in the [`Reset`](CommandBufferState::Reset)
    /// state and ready to begin recording.
    ///
    /// # Safety
    /// The caller must ensure that every previously allocated
    /// [`TransientCommandBuffer`] from this pool is no longer live
    /// (GPU work complete, handle not referenced) before calling
    /// [`reset`](Self::reset). Otherwise the pool reset will implicitly
    /// free in-flight buffers, causing undefined behaviour.
    pub unsafe fn allocate_command_buffer(
        &mut self,
    ) -> Result<TransientCommandBuffer<Q>, AllocateCommandBufferError> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: allocate_info references a valid pool from parent.
        // !Sync prevents concurrent pool access.
        let handle =
            unsafe { self.parent.allocate_raw_command_buffers(&allocate_info) }
                .map(|mut bufs: Vec<vk::CommandBuffer>| {
                    debug_assert_eq!(bufs.len(), 1);
                    bufs.remove(0)
                })
                .map_err(AllocateCommandBufferError::Vulkan)?;

        Ok(TransientCommandBuffer {
            parent: Arc::clone(&self.parent),
            handle,
            state: CommandBufferState::Reset,
            _queue: PhantomData,
        })
    }

    /// Reset the pool, implicitly freeing all allocated buffers.
    ///
    /// # Safety
    /// All [`TransientCommandBuffer`]s previously allocated from this
    /// pool must be idle (GPU work complete) and must not be used after
    /// this call.
    pub unsafe fn reset(&self) -> Result<(), vk::Result> {
        // SAFETY: caller guarantees all buffers are idle.
        unsafe {
            self.parent.reset_raw_command_pool(
                self.pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )
        }
    }

    #[inline]
    pub fn raw_command_pool(&self) -> vk::CommandPool {
        self.pool
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

// ---------------------------------------------------------------------------
// TransientCommandBuffer
// ---------------------------------------------------------------------------

/// A primary command buffer allocated from a [`TransientCommandPool<Q>`].
///
/// Unlike [`ResettableCommandBuffer`], transient buffers do not hold an
/// `Arc` back to the pool and have no return channel. Their lifetime is
/// managed by the pool: calling [`TransientCommandPool::reset`] frees all
/// buffers at once. There is no per-buffer reset.
///
/// `Q` is the queue capability marker.
pub struct TransientCommandBuffer<Q> {
    parent: Arc<Device>,
    handle: vk::CommandBuffer,
    state: CommandBufferState,
    _queue: PhantomData<Q>,
}

impl<Q> std::fmt::Debug for TransientCommandBuffer<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransientCommandBuffer")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl<Q> TransientCommandBuffer<Q> {
    /// Returns the current [`CommandBufferState`].
    #[inline]
    pub fn state(&self) -> CommandBufferState {
        self.state
    }

    #[inline]
    pub fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl<Q> Recordable<Q> for TransientCommandBuffer<Q> {
    fn on_end_recording(&mut self) {
        self.state = CommandBufferState::Recorded;
    }

    /// Begin recording.
    ///
    /// # Panics
    /// Panics if the buffer is not in the
    /// [`Reset`](CommandBufferState::Reset) state.
    fn begin_recording(&mut self) -> Recorder<'_, Q, Self> {
        assert_eq!(
            self.state,
            CommandBufferState::Reset,
            "TransientCommandBuffer::begin_recording called in \
             {:?} state",
            self.state
        );
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: state == Reset guarantees the buffer is in Vulkan's
        // initial state, which is the precondition for
        // vkBeginCommandBuffer.
        unsafe {
            self.parent
                .begin_raw_command_buffer(self.handle, &begin_info)
        }
        .expect("vkBeginCommandBuffer failed");
        self.state = CommandBufferState::Recording;
        Recorder::new(self)
    }

    fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }

    fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl<Q> Submittable<Q> for TransientCommandBuffer<Q> {
    fn raw(&self) -> vk::CommandBuffer {
        self.handle
    }
}

// ---------------------------------------------------------------------------
// Auto-trait assertions
// ---------------------------------------------------------------------------

// ResettableCommandPool<Q>: Send + !Sync
//   (PhantomUnsync makes it !Sync)
// ResettableCommandBuffer<Q>: Send
//   (Arc<Device> is Send; no non-Send fields)
// TransientCommandPool<Q>: Send + !Sync
//   (PhantomUnsync makes it !Sync)
// TransientCommandBuffer<Q>: Send
//   (no non-Send fields; Arc<Device> is Send)
#[allow(dead_code)]
trait AssertSend: Send {}
impl<Q: Send> AssertSend for ResettableCommandPool<Q> {}
impl<Q: Send> AssertSend for ResettableCommandBuffer<Q> {}
impl<Q: Send> AssertSend for TransientCommandPool<Q> {}
impl<Q: Send> AssertSend for TransientCommandBuffer<Q> {}
