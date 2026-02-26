#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::{
    cell::Cell,
    fs::{self, File},
    mem::size_of,
    sync::Arc,
};

use bytemuck::{Pod, Zeroable};
use clap::Parser;
use rgpu::{
    ash::vk::{self, CullModeFlags, FrontFace},
    buffer::{DeviceLocalBuffer, HostVisibleBuffer},
    command::{ResettableCommandBuffer, ResettableCommandPool},
    device::{Device, DeviceConfig, QueueMode},
    instance::{Instance, InstanceExtensions},
    pipeline::{
        DynamicPipeline, DynamicPipelineDesc, VertexAttributeDesc,
        VertexBindingDesc,
    },
    shader::{ShaderModule, ShaderStage},
    surface::Surface,
    swapchain::Swapchain,
    sync::{Fence, Semaphore},
};
use tracing_subscriber::{
    Layer, filter::LevelFilter, layer::SubscriberExt, util::SubscriberInitExt,
};
use vek::{Vec2, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ControlFlow,
    window::{Window as WinitWindow, WindowAttributes},
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: Vec2<f32>,
    color: Vec3<f32>,
}

const RECT_VERTICES: [Vertex; 4] = [
    Vertex {
        position: Vec2::new(-0.5, -0.5),
        color: Vec3::new(0.0, 0.0, 1.0),
    },
    Vertex {
        position: Vec2::new(0.5, 0.5),
        color: Vec3::new(1.0, 1.0, 1.0),
    },
    Vertex {
        position: Vec2::new(0.5, -0.5),
        color: Vec3::new(0.0, 1.0, 1.0),
    },
    Vertex {
        position: Vec2::new(-0.5, 0.5),
        color: Vec3::new(1.0, 0.0, 1.0),
    },
];

const RECT_INDICES: [u16; 6] = [0, 1, 2, 0, 3, 1];

#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, clap::ValueEnum,
)]
/// Log verbosity level for tracing output.
enum TracingLogLevel {
    /// Disable tracing logs.
    Off,
    /// Very detailed execution logs.
    Trace,
    /// Informational runtime events.
    Info,
    /// Debug-level diagnostics.
    Debug,
    /// Warnings and errors only.
    Warn,
    #[default]
    /// Errors only.
    Error,
}

impl From<TracingLogLevel> for LevelFilter {
    fn from(value: TracingLogLevel) -> Self {
        use tracing_subscriber::filter::LevelFilter;
        match value {
            TracingLogLevel::Off => LevelFilter::OFF,
            TracingLogLevel::Trace => LevelFilter::TRACE,
            TracingLogLevel::Info => LevelFilter::INFO,
            TracingLogLevel::Debug => LevelFilter::DEBUG,
            TracingLogLevel::Warn => LevelFilter::WARN,
            TracingLogLevel::Error => LevelFilter::ERROR,
        }
    }
}

#[derive(clap::Parser, Debug)]
#[command(about = "Sample Vulkan app", long_about = None)]
struct CliArgs {
    /// Tracing verbosity for stdout/file logging.
    #[arg(short, long, default_value = "error")]
    tracing_log_level: TracingLogLevel,

    /// Vulkan validation/debug callback severity threshold.
    #[arg(short, long)]
    graphics_debug_level: Option<CliVulkanLogLevel>,

    /// Queue-family selection strategy for graphics/present.
    #[arg(long, default_value = "auto")]
    queue_mode: CliQueueMode,

    /// Load debug-info shader binary (`shader.debug.spv`) for RenderDoc.
    #[arg(long)]
    shader_debug_info: bool,

    /// Disable ANSI color codes in stdout log output.
    #[arg(long)]
    no_color: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
/// Queue-family selection policy.
enum CliQueueMode {
    #[default]
    /// Pick the best available mode automatically.
    Auto,
    /// Prefer one family for graphics and present when possible.
    Unified,
    /// Use a single queue/family path.
    Single,
}

impl From<CliQueueMode> for QueueMode {
    fn from(value: CliQueueMode) -> Self {
        match value {
            CliQueueMode::Auto => QueueMode::Auto,
            CliQueueMode::Unified => QueueMode::Unified,
            CliQueueMode::Single => QueueMode::Single,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
/// Vulkan validation/debug callback severity threshold.
enum CliVulkanLogLevel {
    /// Include verbose diagnostics.
    Verbose,
    /// Include informational messages and above.
    Info,
    /// Include warnings and errors.
    Warning,
    /// Include errors only.
    Error,
}

impl From<CliVulkanLogLevel> for rgpu::log::VulkanLogLevel {
    fn from(value: CliVulkanLogLevel) -> Self {
        match value {
            CliVulkanLogLevel::Verbose => rgpu::log::VulkanLogLevel::Verbose,
            CliVulkanLogLevel::Info => rgpu::log::VulkanLogLevel::Info,
            CliVulkanLogLevel::Warning => rgpu::log::VulkanLogLevel::Warning,
            CliVulkanLogLevel::Error => rgpu::log::VulkanLogLevel::Error,
        }
    }
}

fn main() -> eyre::Result<()> {
    let app_dirs = directories::ProjectDirs::from("", "parengus", "samp-app")
        .ok_or_else(|| {
        eyre::eyre!("Failed to determine application directories")
    })?;

    let self_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| {
            eyre::eyre!(
                "Failed to take the parent directory of the current executable"
            )
        })?
        .to_owned();

    let log_dir = app_dirs
        .runtime_dir()
        .unwrap_or_else(|| app_dirs.data_dir())
        .to_owned();

    let cli_args = CliArgs::parse();

    if cli_args.tracing_log_level != TracingLogLevel::Off {
        fs::create_dir_all(&log_dir)?;

        let mut log_file_path = log_dir.clone();
        log_file_path.push("log-file");
        log_file_path.set_extension("txt");
        let log_file = File::create(&log_file_path)?;
        let file_log = tracing_subscriber::fmt::layer()
            .with_writer(log_file)
            .with_ansi(false);

        let stdout_log = tracing_subscriber::fmt::layer().pretty().with_ansi(
            !cli_args.no_color
                && supports_color::on(supports_color::Stream::Stdout).is_some(),
        );

        tracing_subscriber::registry()
            .with(
                stdout_log
                    .with_filter(LevelFilter::from(cli_args.tracing_log_level))
                    .and_then(file_log),
            )
            .init();

        tracing::debug!("log_file_path: {}", log_file_path.display());
        tracing::debug!("cli_args: {:#?}", cli_args);
    }

    let event_loop = winit::event_loop::EventLoop::builder().build()?;

    // SAFETY: Instance::new loads the Vulkan library via
    // ash::Entry::load (libloading). The loaded Entry must outlive
    // all Vulkan objects derived from it; wrapping the Instance in
    // an Arc ensures it stays alive at least as long as any derived
    // object does.
    let instance = Arc::new(unsafe {
        rgpu::instance::Instance::new(
            "samp-app",
            cli_args.graphics_debug_level.map(Into::into),
            Some(&event_loop),
            InstanceExtensions { surface: true },
        )
    }?);

    let device_config = DeviceConfig {
        swapchain: true,
        dynamic_rendering: true,
        queue_mode: cli_args.queue_mode.into(),
    };

    let mut app = AppRunner(Some(App::Initializing(InitializingState {
        instance,
        device_config,
        self_dir: self_dir.to_owned(),
        shader_debug_info: cli_args.shader_debug_info,
    })));

    tracing::trace!("Entering main event loop");
    Ok(event_loop.run_app(&mut app)?)
}

#[derive(Debug)]
struct AppRunner(Option<App>);

#[derive(Debug)]
enum App {
    Running(RunningState),
    Initializing(InitializingState),
    Suspended(SuspendedState),
    Exiting(ExitingState),
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame-in-flight synchronization primitives and command buffer.
#[derive(Debug)]
struct FrameSync {
    image_available: Semaphore,
    /// Signaled by the queue submit when rendering of this frame is complete.
    /// Waited on by `vkQueuePresentKHR`. Lives here (not keyed by
    /// swapchain image) so it is not destroyed when the swapchain is
    /// recreated mid-flight.
    render_finished: Semaphore,
    in_flight_fence: Fence,
    command_buffer: ResettableCommandBuffer,
    /// Keeps the swapchain alive until this frame slot's fence signals.
    ///
    /// Set to the current swapchain after each submit and cleared at the
    /// start of the next use of this slot (after `wait_and_reset`). This
    /// prevents a retired swapchain from being destroyed while the
    /// presentation engine may still hold references to its images.
    retained_swapchain: Option<Arc<Swapchain<WinitWindow>>>,
    /// Keeps the pipeline alive until this frame slot's fence signals.
    ///
    /// Set to the current pipeline after each submit and cleared at the
    /// start of the next use of this slot. This prevents a retired pipeline
    /// from being destroyed while in-flight command buffers still reference it.
    retained_pipeline: Option<Arc<DynamicPipeline>>,
}

impl FrameSync {
    /// Release all retained resource references for this frame slot.
    ///
    /// # Safety
    /// The GPU must not be accessing any of the retained resources. The caller
    /// must ensure either this slot's fence has signaled, or `vkDeviceWaitIdle`
    /// has returned successfully, before calling this function.
    unsafe fn release_retained(&mut self) {
        self.retained_swapchain = None;
        self.retained_pipeline = None;
    }
}

/// Ensures [`Device::wait_idle`] is called before [`RunningState`]'s Vulkan
/// resources are destroyed.
///
/// Stored as the **first** field of `RunningState` so that Rust's field drop
/// order guarantees `wait_idle` fires before the swapchain, frames,
/// semaphores, and fences are freed.
///
/// The two exit paths are:
///
/// - **Drop** (`exit_from_running`, etc.): the guard is dropped with the rest
///   of `RunningState`; `Drop` calls `wait_idle` automatically.
/// - **Destructure** (`suspended()` transition): `RunningState` is
///   destructured to move fields into `SuspendedState`; `wait_idle` is called
///   explicitly first (for error handling), and the guard drops at end of
///   scope, calling `wait_idle` a second time — which is a harmless no-op on
///   an already-idle device.
struct RunningStateTransitionGuard {
    device: Arc<Device>,
}

impl RunningStateTransitionGuard {
    /// # Safety
    /// This guard **must** be dropped rather than forgotten. Dropping it calls
    /// `vkDeviceWaitIdle`, ensuring the GPU is idle before the caller's Vulkan
    /// resources are freed. Forgetting it (via `mem::forget`, `ManuallyDrop`,
    /// etc.) skips that wait and may allow resources to be destroyed while
    /// the GPU is still accessing them.
    unsafe fn new(device: Arc<Device>) -> Self {
        Self { device }
    }
}

impl Drop for RunningStateTransitionGuard {
    fn drop(&mut self) {
        if let Err(e) = self.device.wait_idle() {
            tracing::error!(
                "Error waiting for device idle on RunningState drop: {e}"
            );
        }
    }
}

#[derive(Debug)]
struct InitializingState {
    instance: Arc<Instance>,
    device_config: DeviceConfig,
    self_dir: std::path::PathBuf,
    shader_debug_info: bool,
}
#[derive(Debug)]
struct DebugCounters {
    swapchain: Cell<u64>,
    pipeline: Cell<u64>,
}

impl DebugCounters {
    fn new() -> Self {
        Self {
            swapchain: Cell::new(0),
            pipeline: Cell::new(0),
        }
    }

    fn next_swapchain(&self) -> u64 {
        let n = self.swapchain.get() + 1;
        self.swapchain.set(n);
        n
    }

    fn next_pipeline(&self) -> u64 {
        let n = self.pipeline.get() + 1;
        self.pipeline.set(n);
        n
    }
}

struct RunningState {
    /// First field: drops first, calling `wait_idle` before all other
    /// resources.
    _idle_guard: RunningStateTransitionGuard,
    win: Arc<WinitWindow>,
    device: Arc<Device>,
    _surface: Arc<Surface<WinitWindow>>,
    // `None` means the window/surface is currently zero-sized. We stay in
    // Running and recreate on the next non-zero resize/scale event.
    swapchain: Option<Arc<Swapchain<WinitWindow>>>,
    shader: ShaderModule,
    pipeline: Arc<DynamicPipeline>,
    vertex_buffer: Arc<DeviceLocalBuffer>,
    index_buffer: Arc<DeviceLocalBuffer>,
    pipeline_color_format: vk::Format,
    command_pool: ResettableCommandPool,
    frames: Vec<FrameSync>,
    current_frame: usize,
    debug_counters: DebugCounters,
}

impl std::fmt::Debug for RunningState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningState")
            .field("win", &self.win)
            .field("device", &self.device)
            .field("_surface", &self._surface)
            .field("swapchain", &self.swapchain)
            .field("shader", &self.shader)
            .field("pipeline", &self.pipeline)
            .field("vertex_buffer", &self.vertex_buffer)
            .field("index_buffer", &self.index_buffer)
            .field("pipeline_color_format", &self.pipeline_color_format)
            .field("command_pool", &self.command_pool)
            .field("frames", &self.frames)
            .field("current_frame", &self.current_frame)
            .field("debug_counters", &self.debug_counters)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct SuspendedState {
    win: Arc<WinitWindow>,
    device: Arc<Device>,
    shader: ShaderModule,
    pipeline: Arc<DynamicPipeline>,
    vertex_buffer: Arc<DeviceLocalBuffer>,
    index_buffer: Arc<DeviceLocalBuffer>,
    pipeline_color_format: vk::Format,
    command_pool: ResettableCommandPool,
    frames: Vec<FrameSync>,
    debug_counters: DebugCounters,
}
#[derive(Debug)]
struct ExitingState {}

impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.0.is_some());
        if let Some(initializing_state) = self.take_initializing() {
            let _span = tracing::debug_span!(
                "state_transition",
                from = "Initializing",
                to = "Running"
            )
            .entered();
            event_loop.set_control_flow(ControlFlow::Poll);
            match Self::initializing_to_running(initializing_state, event_loop)
            {
                Ok(running_state) => {
                    self.set_running(running_state);
                }
                Err(e) => {
                    tracing::error!("Error during initialization: {:#}", e);
                    self.transition_to_exiting("Initializing", event_loop);
                }
            }
        } else if let Some(suspended_state) = self.take_suspended() {
            let _span = tracing::debug_span!(
                "state_transition",
                from = "Suspended",
                to = "Running"
            )
            .entered();
            event_loop.set_control_flow(ControlFlow::Poll);
            match Self::suspended_to_running(suspended_state) {
                Ok(running_state) => {
                    self.set_running(running_state);
                }
                Err(e) => {
                    tracing::error!("Error during resume: {:#}", e);
                    self.transition_to_exiting("Suspended", event_loop);
                }
            }
        } else if self.is_exiting() {
            tracing::warn!("resumed() called while in Exiting state");
        }
    }
    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.0.is_some());
        if let Some(running_state) = self.take_running() {
            let _span = tracing::debug_span!(
                "state_transition",
                from = "Running",
                to = "Suspended"
            )
            .entered();
            event_loop.set_control_flow(ControlFlow::Wait);
            let RunningState {
                _idle_guard,
                win,
                device,
                _surface: _,
                swapchain: _,
                shader,
                pipeline,
                vertex_buffer,
                index_buffer,
                pipeline_color_format,
                command_pool,
                mut frames,
                current_frame: _,
                debug_counters,
            } = running_state;

            if let Err(e) = device.wait_idle() {
                tracing::error!(
                    "Error while waiting for device idle during suspend: {}",
                    e
                );
                self.transition_to_exiting("Running", event_loop);
                return;
            }

            // GPU is idle; eagerly release all retained references so resources
            // dropped via field destructuring above are destroyed now.
            for frame in &mut frames {
                // SAFETY: wait_idle() succeeded above; no GPU work is in
                // flight.
                unsafe { frame.release_retained() };
            }

            self.set_suspended(SuspendedState {
                win,
                device,
                shader,
                pipeline,
                vertex_buffer,
                index_buffer,
                pipeline_color_format,
                command_pool,
                frames,
                debug_counters,
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        window_event: winit::event::WindowEvent,
    ) {
        assert!(self.0.is_some());
        if !self.is_running_window(window_id) {
            return;
        }

        if matches!(&window_event, WindowEvent::CloseRequested) {
            tracing::trace!("Close window request received for window");
            self.exit_from_running(event_loop);
            return;
        }

        match &window_event {
            WindowEvent::Resized(_)
            | WindowEvent::ScaleFactorChanged { .. } => {
                let desired_extent = {
                    if let Some(running_state) = self.as_running()
                        && let Some(extent) = Self::desired_extent_for_event(
                            &running_state.win,
                            &window_event,
                        )
                    {
                        extent
                    } else {
                        return;
                    }
                };

                let should_keep_running = {
                    let running_state = match self.as_running_mut() {
                        Some(running_state) => running_state,
                        None => return,
                    };

                    Self::recreate_swapchain_if_needed(
                        running_state,
                        desired_extent,
                    )
                };

                if !should_keep_running {
                    self.exit_from_running(event_loop);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        let outcome = match self.as_running_mut() {
            Some(state) => Self::draw_frame(state),
            None => return,
        };

        match outcome {
            DrawFrameOutcome::Success => {}
            DrawFrameOutcome::SwapchainOutOfDate => {
                if let Some(running_state) = self.as_running_mut() {
                    let size = running_state.win.inner_size();
                    let extent = vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    };
                    if !Self::recreate_swapchain_if_needed(
                        running_state,
                        extent,
                    ) {
                        self.exit_from_running(event_loop);
                    }
                }
            }
            DrawFrameOutcome::Fatal(msg) => {
                tracing::error!("Fatal frame error: {msg}");
                self.exit_from_running(event_loop);
            }
        }
    }
}

enum DrawFrameOutcome {
    Success,
    SwapchainOutOfDate,
    Fatal(String),
}

impl AppRunner {
    fn draw_frame(state: &mut RunningState) -> DrawFrameOutcome {
        // Skip if the window is currently zero-sized (no swapchain).
        if state.swapchain.is_none() {
            return DrawFrameOutcome::Success;
        }

        let frame_idx = state.current_frame;

        // Wait until this frame slot's previous GPU work is done, then
        // reset the fence for reuse. This also guarantees image_available
        // is unsignaled and the command buffer is no longer in use.
        //
        // SAFETY: fence was submitted with GPU work for this frame slot;
        // the wait ensures all that work has completed.
        if let Err(e) = unsafe {
            state.frames[frame_idx]
                .in_flight_fence
                .wait_and_reset(u64::MAX)
        } {
            return DrawFrameOutcome::Fatal(format!(
                "Fence wait/reset failed: {e}"
            ));
        }

        // GPU work for this slot is done; release retained references.
        // This allows retired resources to be destroyed once all slots
        // have cleared.
        // SAFETY: wait_and_reset() succeeded above; this slot's GPU work
        // is complete.
        unsafe { state.frames[frame_idx].release_retained() };

        // SAFETY: is_none() is checked at the top of this function.
        let sc = state
            .swapchain
            .as_ref()
            .expect("swapchain present: checked above");
        let swapchain_raw = sc.raw_swapchain();
        let image_available =
            state.frames[frame_idx].image_available.raw_semaphore();

        // Acquire the next presentable image.
        //
        // SAFETY: image_available is unsignaled (fence wait above ensures the
        // previous acquire+submit cycle for this slot has completed).
        let acquire_result = unsafe {
            sc.acquire_next_image(u64::MAX, image_available, vk::Fence::null())
        };

        let (image_index, suboptimal) = match acquire_result {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return DrawFrameOutcome::SwapchainOutOfDate;
            }
            Err(e) => {
                return DrawFrameOutcome::Fatal(format!("Acquire failed: {e}"));
            }
        };

        let i = image_index as usize;
        let extent = sc.extent();
        let image = sc.images()[i];
        let image_view = sc
            .image_views()
            .expect("image views were created with the swapchain")[i];
        // Clone the retained reference now while sc is borrowed; sc's
        // borrow ends here.
        let retained = Arc::clone(sc);
        let render_finished =
            state.frames[frame_idx].render_finished.raw_semaphore();

        let fence = state.frames[frame_idx].in_flight_fence.raw_fence();
        let pipeline_handle = state.pipeline.raw_pipeline();
        let frame_cmd = &mut state.frames[frame_idx].command_buffer;

        // Reset and re-record the command buffer for this frame slot.
        //
        // SAFETY: fence wait above guarantees the buffer is not
        // pending on the GPU.
        if let Err(e) = unsafe { frame_cmd.reset() } {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer reset failed: {e}"
            ));
        }
        // SAFETY: buffer was just reset to the initial state.
        if let Err(e) = unsafe { frame_cmd.begin() } {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer begin failed: {e}"
            ));
        }

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // Transition: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        let to_color = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(subresource_range);
        let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&to_color));
        // SAFETY: recording state; image is a valid swapchain image.
        unsafe { frame_cmd.pipeline_barrier2(&dep_info) };

        // Begin dynamic rendering with a clear.
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        // SAFETY: recording; image in COLOR_ATTACHMENT_OPTIMAL;
        // image_view valid.
        if let Err(e) = unsafe { frame_cmd.begin_rendering(&rendering_info) } {
            return DrawFrameOutcome::Fatal(format!(
                "begin_rendering failed: {e}"
            ));
        }

        // Bind pipeline, set dynamic viewport/scissor, draw.
        // SAFETY: inside a dynamic render pass with a compatible
        // color attachment.
        unsafe { frame_cmd.bind_graphics_pipeline(pipeline_handle) };
        // SAFETY: inside render pass recording; buffer is valid and bound to
        // host-visible memory for the app lifetime.
        unsafe { frame_cmd.bind_vertex_buffer(0, &*state.vertex_buffer, 0) };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        // SAFETY: pipeline declares VK_DYNAMIC_STATE_VIEWPORT.
        unsafe { frame_cmd.set_viewport(std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        // SAFETY: pipeline declares VK_DYNAMIC_STATE_SCISSOR.
        unsafe { frame_cmd.set_scissor(std::slice::from_ref(&scissor)) };

        // SAFETY: inside render pass recording; buffer is valid.
        unsafe {
            frame_cmd.bind_index_buffer(
                &*state.index_buffer,
                0,
                vk::IndexType::UINT16,
            )
        };

        // Draw a rectangle using the index buffer.
        // SAFETY: all required dynamic state has been set;
        // render pass is active; index buffer is bound.
        unsafe {
            frame_cmd.draw_indexed(RECT_INDICES.len() as u32, 1, 0, 0, 0)
        };

        // SAFETY: inside a dynamic render pass.
        if let Err(e) = unsafe { frame_cmd.end_rendering() } {
            return DrawFrameOutcome::Fatal(format!(
                "end_rendering failed: {e}"
            ));
        }

        // Transition: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
        let to_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(subresource_range);
        let dep_info_present = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&to_present));
        // SAFETY: recording; image is in COLOR_ATTACHMENT_OPTIMAL.
        unsafe { frame_cmd.pipeline_barrier2(&dep_info_present) };

        // SAFETY: recording state.
        if let Err(e) = unsafe { frame_cmd.end() } {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer end failed: {e}"
            ));
        }

        let cmd_handle = frame_cmd.raw_command_buffer();
        // frame_cmd borrow ends here; subsequent accesses are on
        // different fields.

        // Submit — wait on image_available, signal render_finished,
        // signal fence when done.
        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let cmd_submit_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd_handle);
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(std::slice::from_ref(&wait_info))
            .command_buffer_infos(std::slice::from_ref(&cmd_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_info));
        // SAFETY: image_available is signaled by acquire;
        // render_finished is unsignaled; fence is unsignaled
        // (just reset above); cmd is in the executable state.
        if let Err(e) = unsafe {
            state.device.graphics_present_queue_submit2(
                std::slice::from_ref(&submit),
                fence,
            )
        } {
            return DrawFrameOutcome::Fatal(format!(
                "Queue submit failed: {e}"
            ));
        }

        // Retain references until this slot's fence signals, preventing the
        // presentation engine / GPU from accessing destroyed resources.
        state.frames[frame_idx].retained_swapchain = Some(retained);
        state.frames[frame_idx].retained_pipeline =
            Some(Arc::clone(&state.pipeline));

        // Present — wait on render_finished.
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&render_finished))
            .swapchains(std::slice::from_ref(&swapchain_raw))
            .image_indices(std::slice::from_ref(&image_index));
        // SAFETY: render_finished is signaled by submit; image is in
        // PRESENT_SRC_KHR.
        let present_result =
            unsafe { state.device.queue_present(&present_info) };

        state.current_frame = (state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        match present_result {
            Ok(false) if !suboptimal => DrawFrameOutcome::Success,
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                DrawFrameOutcome::SwapchainOutOfDate
            }
            Err(e) => DrawFrameOutcome::Fatal(format!("Present failed: {e}")),
        }
    }
}

fn build_pipeline<F>(
    device: &Arc<Device>,
    shader: &ShaderModule,
    color_format: vk::Format,
    name: Option<F>,
) -> eyre::Result<DynamicPipeline>
where
    F: FnOnce() -> String,
{
    let vert = shader.entry_point("vert_main", ShaderStage::Vertex)?;
    let frag = shader.entry_point("frag_main", ShaderStage::Fragment)?;
    let vertex_bindings = [VertexBindingDesc {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let vertex_attributes = [
        VertexAttributeDesc {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, position) as u32,
        },
        VertexAttributeDesc {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, color) as u32,
        },
    ];
    Ok(DynamicPipeline::new(
        device,
        &DynamicPipelineDesc {
            stages: &[vert, frag],
            color_attachment_formats: &[color_format],
            vertex_bindings: &vertex_bindings,
            vertex_attributes: &vertex_attributes,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::COUNTER_CLOCKWISE,
            ..Default::default()
        },
        name,
    )?)
}

impl AppRunner {
    fn initializing_to_running(
        state: InitializingState,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> eyre::Result<RunningState> {
        let win = Arc::new(event_loop.create_window(
            WindowAttributes::default().with_inner_size(LogicalSize {
                width: 1600,
                height: 900,
            }),
        )?);

        // SAFETY: Surface must be destroyed only after all swapchains
        // derived from it are destroyed and no GPU work accesses their
        // images. Swapchain holds Arc<Surface>, so the surface outlives
        // the swapchain. Frame slots hold Arc<Swapchain> via
        // retained_swapchain, which is cleared only after
        // device.wait_idle() in the suspended() handler — so the
        // surface is not freed while GPU work is pending.
        let surface = Arc::new(unsafe {
            Surface::new(&state.instance, Arc::clone(&win))
        }?);

        let device = Arc::new(Device::create_compatible(
            &state.instance,
            &surface,
            state.device_config,
        )?);

        let vertex_buffer_size =
            (RECT_VERTICES.len() * size_of::<Vertex>()) as vk::DeviceSize;
        let mut staging_vertex_buffer = HostVisibleBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("rect staging vertex buffer"),
        )?;
        staging_vertex_buffer.write_pod(&RECT_VERTICES)?;

        let mut vertex_buffer = DeviceLocalBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            Some("rect vertex buffer"),
        )?;

        let index_buffer_size =
            (RECT_INDICES.len() * size_of::<u16>()) as vk::DeviceSize;
        let mut staging_index_buffer = HostVisibleBuffer::new(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("rect staging index buffer"),
        )?;
        staging_index_buffer.write_pod(&RECT_INDICES)?;

        let mut index_buffer = DeviceLocalBuffer::new(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            Some("rect index buffer"),
        )?;

        let upload_command_pool = ResettableCommandPool::new(
            &device,
            device.graphics_present_queue_family(),
            Some("upload command pool"),
        )?;
        let mut upload_command_buffer =
            upload_command_pool.allocate_command_buffer()?;
        // SAFETY: upload_command_pool is used on this thread only during
        // initialization, and `None` fence path waits for device idle before
        // returning, so source/destination buffers outlive GPU access.
        unsafe {
            vertex_buffer.upload_from_host_visible(
                &mut upload_command_buffer,
                &staging_vertex_buffer,
                None,
            )
        }?;
        // SAFETY: Same contract as vertex buffer upload above.
        unsafe {
            index_buffer.upload_from_host_visible(
                &mut upload_command_buffer,
                &staging_index_buffer,
                None,
            )
        }?;
        let vertex_buffer = Arc::new(vertex_buffer);
        let index_buffer = Arc::new(index_buffer);

        let win_size = win.inner_size();
        let debug_counters = DebugCounters::new();
        let swapchain = if win_size.width == 0 || win_size.height == 0 {
            tracing::trace!(
                "Skipping initial swapchain create because window \
                 extent is zero: {}x{}",
                win_size.width,
                win_size.height
            );
            None
        } else {
            let sc_idx = debug_counters.next_swapchain();
            let requested_extent = vk::Extent2D {
                width: win_size.width,
                height: win_size.height,
            };
            let swapchain_create_span = tracing::trace_span!(
                "initial_swapchain_create",
                requested_width = requested_extent.width,
                requested_height = requested_extent.height
            )
            .entered();
            let swapchain = Swapchain::new(
                &device,
                &surface,
                requested_extent,
                true,
                None,
                Some(move || format!("Swapchain {sc_idx}")),
            )?;
            drop(swapchain_create_span);
            Some(Arc::new(swapchain))
        };

        let shader_dir = state.self_dir.join("shaders");
        let shader_default_path = shader_dir.join("shader.spv");
        let shader_debug_path = shader_dir.join("shader.debug.spv");
        let shader_path = if state.shader_debug_info {
            if shader_debug_path.exists() {
                shader_debug_path
            } else {
                tracing::warn!(
                    path = %shader_debug_path.display(),
                    "Shader debug info requested but debug shader was not \
                     found; falling back to non-debug shader"
                );
                shader_default_path.clone()
            }
        } else {
            shader_default_path.clone()
        };
        let shader = {
            let _span = tracing::trace_span!(
                "shader_load",
                path = ?shader_path
            )
            .entered();
            let shader_bytes = std::fs::read(&shader_path).map_err(|e| {
                eyre::eyre!(
                    "Error loading shader {}: {e}",
                    shader_path.display()
                )
            })?;
            ShaderModule::new(&device, &shader_bytes, Some("shader"))?
        };

        let pipeline_color_format = swapchain
            .as_ref()
            .map(|sc| sc.format())
            .unwrap_or(vk::Format::B8G8R8A8_UNORM);
        let pipeline = {
            let _span = tracing::trace_span!(
                "pipeline_create",
                color_format = ?pipeline_color_format,
            )
            .entered();
            Arc::new(build_pipeline(
                &device,
                &shader,
                pipeline_color_format,
                Some({
                    let idx = debug_counters.next_pipeline();
                    move || format!("main pipeline {idx}")
                }),
            )?)
        };

        let command_pool = ResettableCommandPool::new(
            &device,
            device.graphics_present_queue_family(),
            Some("graphics command pool"),
        )?;

        // Start each fence signaled so the first frame's
        // wait_and_reset returns immediately.
        // Each frame slot owns its command buffer so the CPU can encode
        // frame N while the GPU executes frame N-1 without contention.
        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| -> eyre::Result<FrameSync> {
                Ok(FrameSync {
                    image_available: Semaphore::new(
                        &device,
                        Some(&format!("image available [{i}]")),
                    )?,
                    render_finished: Semaphore::new(
                        &device,
                        Some(&format!("render finished [{i}]")),
                    )?,
                    in_flight_fence: Fence::new(
                        &device,
                        true,
                        Some(&format!("in flight [{i}]")),
                    )?,
                    command_buffer: command_pool.allocate_command_buffer()?,
                    retained_swapchain: None,
                    retained_pipeline: None,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        // SAFETY: guard will call wait_idle in Drop. Must not be forgotten.
        let idle_guard =
            unsafe { RunningStateTransitionGuard::new(Arc::clone(&device)) };
        Ok(RunningState {
            _idle_guard: idle_guard,
            win,
            device,
            _surface: surface,
            swapchain,
            shader,
            pipeline,
            vertex_buffer,
            index_buffer,
            pipeline_color_format,
            command_pool,
            frames,
            current_frame: 0,
            debug_counters,
        })
    }

    fn suspended_to_running(
        state: SuspendedState,
    ) -> eyre::Result<RunningState> {
        // SAFETY: Same contract as initializing_to_running: the surface
        // outlives all derived swapchains via Arc<Surface>, and frame
        // slots' retained_swapchain Arcs are cleared only after
        // device.wait_idle() in the next suspended() call.
        let surface = Arc::new(unsafe {
            Surface::new(state.device.parent(), Arc::clone(&state.win))
        }?);

        let win_size = state.win.inner_size();
        let debug_counters = state.debug_counters;
        let swapchain = if win_size.width == 0 || win_size.height == 0 {
            tracing::trace!(
                "Skipping swapchain create on resume because window \
                 extent is zero: {}x{}",
                win_size.width,
                win_size.height
            );
            None
        } else {
            let sc_idx = debug_counters.next_swapchain();
            Some(Arc::new(Swapchain::new(
                &state.device,
                &surface,
                vk::Extent2D {
                    width: win_size.width,
                    height: win_size.height,
                },
                true,
                Some(state.pipeline_color_format),
                Some(move || format!("Swapchain {sc_idx}")),
            )?))
        };

        // Reuse the existing pipeline when the swapchain honored the preferred
        // format. Recreate it only if the surface forced a different format.
        let swapchain_format = swapchain.as_ref().map(|sc| sc.format());
        let (pipeline, pipeline_color_format) =
            if let Some(new_format) = swapchain_format {
                if new_format == state.pipeline_color_format {
                    (state.pipeline, state.pipeline_color_format)
                } else {
                    tracing::debug!(
                        "Swapchain format changed on resume \
                     ({:?} -> {:?}); recreating pipeline",
                        state.pipeline_color_format,
                        new_format,
                    );
                    let pipeline = {
                        let _span = tracing::trace_span!(
                            "pipeline_create",
                            color_format = ?new_format,
                        )
                        .entered();
                        Arc::new(build_pipeline(
                            &state.device,
                            &state.shader,
                            new_format,
                            Some({
                                let idx = debug_counters.next_pipeline();
                                move || format!("main pipeline {idx}")
                            }),
                        )?)
                    };
                    (pipeline, new_format)
                }
            } else {
                (state.pipeline, state.pipeline_color_format)
            };

        // SAFETY: guard will call wait_idle in Drop. Must not be forgotten.
        let idle_guard = unsafe {
            RunningStateTransitionGuard::new(Arc::clone(&state.device))
        };
        Ok(RunningState {
            _idle_guard: idle_guard,
            win: state.win,
            device: state.device,
            _surface: surface,
            swapchain,
            shader: state.shader,
            pipeline,
            vertex_buffer: state.vertex_buffer,
            index_buffer: state.index_buffer,
            pipeline_color_format,
            command_pool: state.command_pool,
            frames: state.frames,
            current_frame: 0,
            debug_counters,
        })
    }

    fn recreate_swapchain_if_needed(
        running_state: &mut RunningState,
        desired_extent: rgpu::ash::vk::Extent2D,
    ) -> bool {
        if desired_extent.width == 0 || desired_extent.height == 0 {
            let _span = tracing::trace_span!(
                "swapchain_teardown",
                width = desired_extent.width,
                height = desired_extent.height,
            )
            .entered();
            if let Err(e) = running_state.device.wait_idle() {
                tracing::error!(
                    "Error while waiting for device idle on zero extent: {}",
                    e
                );
                return false;
            }
            for frame in &mut running_state.frames {
                // SAFETY: wait_idle() succeeded above; no GPU work is in
                // flight.
                unsafe { frame.release_retained() };
            }
            running_state.swapchain = None;
            return true;
        }

        if let Some(existing_swapchain) = running_state.swapchain.as_ref()
            && existing_swapchain.extent() == desired_extent
        {
            tracing::trace!(
                "Skipping swapchain recreate because extent is \
                 unchanged: {}x{}",
                desired_extent.width,
                desired_extent.height
            );
            return true;
        }

        let _span = tracing::trace_span!(
            "swapchain_recreate",
            width = desired_extent.width,
            height = desired_extent.height,
        )
        .entered();

        let sc_idx = running_state.debug_counters.next_swapchain();
        match Swapchain::new_with_old(
            &running_state.device,
            &running_state._surface,
            desired_extent,
            running_state.swapchain.as_ref().map(|sc| sc.as_ref()),
            true,
            Some(running_state.pipeline_color_format),
            Some(move || format!("Swapchain {sc_idx}")),
        ) {
            Ok(swapchain) => {
                tracing::trace!(
                    "Swapchain recreation succeeded for extent: {}x{}",
                    desired_extent.width,
                    desired_extent.height
                );
                let new_format = swapchain.format();
                running_state.swapchain = Some(Arc::new(swapchain));

                if new_format != running_state.pipeline_color_format {
                    tracing::debug!(
                        "Swapchain format changed on resize \
                         ({:?} -> {:?}); recreating pipeline",
                        running_state.pipeline_color_format,
                        new_format,
                    );
                    let _span = tracing::trace_span!(
                        "pipeline_create",
                        color_format = ?new_format,
                    )
                    .entered();
                    match build_pipeline(
                        &running_state.device,
                        &running_state.shader,
                        new_format,
                        Some({
                            let idx =
                                running_state.debug_counters.next_pipeline();
                            move || format!("main pipeline {idx}")
                        }),
                    ) {
                        Ok(pipeline) => {
                            // Arc::clone in draw_frame's retained_pipeline
                            // keeps the old pipeline alive until each
                            // in-flight slot's fence signals.
                            running_state.pipeline = Arc::new(pipeline);
                            running_state.pipeline_color_format = new_format;
                        }
                        Err(e) => {
                            tracing::error!(
                                "Error recreating pipeline after \
                                 format change: {}",
                                e
                            );
                            return false;
                        }
                    }
                }

                true
            }
            Err(e) => {
                tracing::error!("Error while recreating swapchain: {}", e);
                false
            }
        }
    }

    fn desired_extent_for_event(
        win: &WinitWindow,
        window_event: &WindowEvent,
    ) -> Option<rgpu::ash::vk::Extent2D> {
        match window_event {
            WindowEvent::Resized(size) => Some(rgpu::ash::vk::Extent2D {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = win.inner_size();
                Some(rgpu::ash::vk::Extent2D {
                    width: size.width,
                    height: size.height,
                })
            }
            _ => None,
        }
    }
}

#[allow(dead_code, reason = "these functions exist for API completeness")]
impl AppRunner {
    fn transition_to_exiting(
        &mut self,
        from_state: &'static str,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        let _span = tracing::debug_span!(
            "state_transition",
            from = from_state,
            to = "Exiting"
        )
        .entered();
        self.set_exiting(ExitingState {});
        event_loop.exit();
    }

    fn exit_from_running(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        if let Some(running_state) = self.take_running() {
            // _idle_guard drops first (first field), calling wait_idle before
            // the swapchain, frames, semaphores, and fences are freed.
            drop(running_state);
            self.transition_to_exiting("Running", event_loop);
        } else {
            tracing::warn!(
                "Requested Running -> Exiting transition \
                 while not in Running state"
            );
            event_loop.exit();
        }
    }

    fn is_running_window(&self, window_id: winit::window::WindowId) -> bool {
        if let Some(running_state) = self.as_running()
            && window_id == running_state.win.id()
        {
            true
        } else {
            false
        }
    }

    fn is_initializing(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Initializing(_)))
    }

    fn take_initializing(&mut self) -> Option<InitializingState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Initializing(_))) {
            match self.0.take() {
                Some(App::Initializing(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_initializing(&self) -> Option<&InitializingState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Initializing(s)) => Some(s),
            _ => None,
        }
    }

    fn as_initializing_mut(&mut self) -> Option<&mut InitializingState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Initializing(s)) => Some(s),
            _ => None,
        }
    }

    fn set_initializing(&mut self, state: InitializingState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Initializing(state));
    }

    fn is_running(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Running(_)))
    }

    fn take_running(&mut self) -> Option<RunningState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Running(_))) {
            match self.0.take() {
                Some(App::Running(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_running(&self) -> Option<&RunningState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Running(s)) => Some(s),
            _ => None,
        }
    }

    fn as_running_mut(&mut self) -> Option<&mut RunningState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Running(s)) => Some(s),
            _ => None,
        }
    }

    fn set_running(&mut self, state: RunningState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Running(state));
    }

    fn is_suspended(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Suspended(_)))
    }

    fn take_suspended(&mut self) -> Option<SuspendedState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Suspended(_))) {
            match self.0.take() {
                Some(App::Suspended(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_suspended(&self) -> Option<&SuspendedState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Suspended(s)) => Some(s),
            _ => None,
        }
    }

    fn as_suspended_mut(&mut self) -> Option<&mut SuspendedState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Suspended(s)) => Some(s),
            _ => None,
        }
    }

    fn set_suspended(&mut self, state: SuspendedState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Suspended(state));
    }

    fn is_exiting(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Exiting(_)))
    }

    fn take_exiting(&mut self) -> Option<ExitingState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Exiting(_))) {
            match self.0.take() {
                Some(App::Exiting(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_exiting(&self) -> Option<&ExitingState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Exiting(s)) => Some(s),
            _ => None,
        }
    }

    fn as_exiting_mut(&mut self) -> Option<&mut ExitingState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Exiting(s)) => Some(s),
            _ => None,
        }
    }

    fn set_exiting(&mut self, state: ExitingState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Exiting(state));
    }
}
