#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

//! Classic Vulkan 1.0 sample app.
//!
//! Renders the same scene as `samp-app` (three overlapping textured
//! quads with depth testing) using only Vulkan 1.0 core APIs:
//!
//! - `VkRenderPass` / `VkFramebuffer` instead of dynamic rendering
//! - `vkCmdPipelineBarrier` instead of sync2 `vkCmdPipelineBarrier2`
//! - `vkQueueSubmit` / `VkSubmitInfo` instead of sync2
//!   `vkQueueSubmit2`
//!
//! The render pass uses `initial_layout = UNDEFINED` for both
//! attachments and `final_layout = PRESENT_SRC_KHR` for the color
//! attachment, eliminating the need for explicit pre-render and
//! pre-present image barriers. The subpass dependency handles all
//! stage/access synchronisation.

use std::{
    cell::Cell,
    f32::consts::PI as PI32,
    fs::{self, File},
    mem::size_of,
    sync::Arc,
    time::Instant,
};

use asset_pipeline::AssetMap;
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use rgpu_vk::{
    ash::vk,
    buffer::{DeviceLocalBuffer, HostVisibleBuffer},
    command::{ResettableCommandBuffer, ResettableCommandPool},
    descriptor::{
        DescriptorBindingDesc, DescriptorPool, DescriptorSet,
        DescriptorSetLayout,
    },
    device::{Device, DeviceConfig, QueueConfig},
    image::{DepthImage, MsaaImage, Texture},
    instance::{Instance, InstanceConfig},
    pipeline::{
        CullModeFlags, FrontFace, PipelineLayout, PipelineLayoutDesc,
        RenderPassPipeline, RenderPassPipelineDesc, VertexAttributeDesc,
        VertexBindingDesc, VertexInputRate,
    },
    renderpass::{RenderPass, RenderPassDesc},
    sampler::Sampler,
    shader::{ShaderModule, ShaderStage},
    surface::Surface,
    swapchain::Swapchain,
    sync::{Fence, Semaphore},
};
use tracing_subscriber::{
    Layer,
    filter::{LevelFilter, Targets},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};
use vek::{Mat4, Vec2, Vec3};
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
    position: Vec3<f32>,
    tex_coord: Vec2<f32>,
}

/// Uniform buffer object — combined view-projection matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Ubo {
    view_proj: Mat4<f32>,
}

/// Push-constant block — model matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    model: Mat4<f32>,
}

/// Three overlapping 0.8×0.8 quads at increasing Z heights.
const SCENE_VERTICES: [Vertex; 12] = [
    // Nearest (z = 0.7), centre at (−0.25, 0)
    Vertex {
        position: Vec3::new(-0.65, -0.4, 0.7),
        tex_coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        position: Vec3::new(0.15, 0.4, 0.7),
        tex_coord: Vec2::new(1.0, 1.0),
    },
    Vertex {
        position: Vec3::new(0.15, -0.4, 0.7),
        tex_coord: Vec2::new(1.0, 0.0),
    },
    Vertex {
        position: Vec3::new(-0.65, 0.4, 0.7),
        tex_coord: Vec2::new(0.0, 1.0),
    },
    // Middle (z = 0.35), centre at (0.25, 0)
    Vertex {
        position: Vec3::new(-0.15, -0.4, 0.35),
        tex_coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        position: Vec3::new(0.65, 0.4, 0.35),
        tex_coord: Vec2::new(1.0, 1.0),
    },
    Vertex {
        position: Vec3::new(0.65, -0.4, 0.35),
        tex_coord: Vec2::new(1.0, 0.0),
    },
    Vertex {
        position: Vec3::new(-0.15, 0.4, 0.35),
        tex_coord: Vec2::new(0.0, 1.0),
    },
    // Farthest (z = 0.0), centre at (0, 0)
    Vertex {
        position: Vec3::new(-0.4, -0.4, 0.0),
        tex_coord: Vec2::new(0.0, 0.0),
    },
    Vertex {
        position: Vec3::new(0.4, 0.4, 0.0),
        tex_coord: Vec2::new(1.0, 1.0),
    },
    Vertex {
        position: Vec3::new(0.4, -0.4, 0.0),
        tex_coord: Vec2::new(1.0, 0.0),
    },
    Vertex {
        position: Vec3::new(-0.4, 0.4, 0.0),
        tex_coord: Vec2::new(0.0, 1.0),
    },
];

// ---------------------------------------------------------------------------
// Matrix math — right-handed, +Z up
// ---------------------------------------------------------------------------

#[rustfmt::skip]
fn look_at_rh(
    eye: Vec3<f32>,
    center: Vec3<f32>,
    world_up: Vec3<f32>,
) -> Mat4<f32> {
    let f = (center - eye).normalized();
    let r = f.cross(world_up).normalized();
    let u = r.cross(f);
    Mat4::from_col_arrays(
        [
        [r.x,         u.x,         -f.x,        0.0,],
        [r.y,         u.y,         -f.y,        0.0,],
        [r.z,         u.z,         -f.z,        0.0,],
        [-r.dot(eye), -u.dot(eye), f.dot(eye),  1.0,]],
    )
}

fn perspective_rh_zo(
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> Mat4<f32> {
    let f = 1.0 / (fov_y * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    // Negate Y to compensate for Vulkan's Y-down NDC convention.
    // Without VK_KHR_maintenance1's negative-height viewport trick,
    // Vulkan maps NDC Y+ to the bottom of the screen. Negating Y in
    // the projection restores the expected Y-up orientation.
    Mat4::from_col_arrays([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, -1.0],
        [0.0, 0.0, b, 0.0],
    ])
}

const SCENE_INDICES: [u16; 18] = [
    0, 2, 1, 0, 1, 3, // nearest
    4, 6, 5, 4, 5, 7, // middle
    8, 10, 9, 8, 9, 11, // farthest
];

#[rustfmt::skip]
#[derive(
    Debug, PartialEq, Eq, PartialOrd, Ord,
    Clone, Copy, Default, clap::ValueEnum,
)]
enum TracingLogLevel {
    Off,
    Trace,
    Info,
    Debug,
    Warn,
    #[default]
    Error,
}

impl From<TracingLogLevel> for LevelFilter {
    fn from(value: TracingLogLevel) -> Self {
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

/// Anti-aliasing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AntiAliasing {
    Off,
    Msaa2,
    Msaa4,
    Msaa8,
}

impl AntiAliasing {
    fn sample_count(self) -> vk::SampleCountFlags {
        match self {
            Self::Off => vk::SampleCountFlags::TYPE_1,
            Self::Msaa2 => vk::SampleCountFlags::TYPE_2,
            Self::Msaa4 => vk::SampleCountFlags::TYPE_4,
            Self::Msaa8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

#[derive(clap::Parser, Debug)]
#[command(
    about = "Sample Vulkan app (classic VK 1.0 APIs)",
    long_about = None
)]
struct CliArgs {
    #[arg(long, default_value = "error")]
    tracing_log_level: TracingLogLevel,

    #[arg(long)]
    graphics_debug_level: Option<CliVulkanLogLevel>,

    #[arg(long)]
    rgpu_log_level: Option<TracingLogLevel>,

    #[arg(long, default_value = "true")]
    dedicated_transfer: bool,

    #[arg(long, default_value = "true")]
    dedicated_compute: bool,

    #[arg(long, default_value = "true")]
    parallel: bool,

    #[arg(long)]
    queue_config_strict: bool,

    /// Load debug-info shader binary (`shader.debug.spv`) for
    /// RenderDoc.
    #[arg(long)]
    shader_debug_info: bool,

    #[arg(long)]
    no_color: bool,

    /// Anti-aliasing mode.
    #[arg(long, default_value = "msaa4")]
    aa: AntiAliasing,

    /// Error if the device does not support the requested AA mode.
    #[arg(long)]
    aa_strict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliVulkanLogLevel {
    Trace,
    Info,
    Warn,
    Error,
}

impl From<CliVulkanLogLevel> for rgpu_vk::instance::VulkanLogLevel {
    fn from(value: CliVulkanLogLevel) -> Self {
        match value {
            CliVulkanLogLevel::Trace => {
                rgpu_vk::instance::VulkanLogLevel::Verbose
            }
            CliVulkanLogLevel::Info => rgpu_vk::instance::VulkanLogLevel::Info,
            CliVulkanLogLevel::Warn => {
                rgpu_vk::instance::VulkanLogLevel::Warning
            }
            CliVulkanLogLevel::Error => {
                rgpu_vk::instance::VulkanLogLevel::Error
            }
        }
    }
}

fn subscriber_filter(
    tracing_level: TracingLogLevel,
    rgpu_level: Option<TracingLogLevel>,
    gfx_level: Option<CliVulkanLogLevel>,
) -> Targets {
    let base = LevelFilter::from(tracing_level);
    let rgpu_filter = rgpu_level.map(LevelFilter::from).unwrap_or(base);
    let mut targets = Targets::new().with_default(base);
    if rgpu_filter != base {
        targets = targets.with_target("rgpu_vk", rgpu_filter);
    }
    if let Some(gfx) = gfx_level {
        let gfx_filter = match gfx {
            CliVulkanLogLevel::Trace => LevelFilter::TRACE,
            CliVulkanLogLevel::Info => LevelFilter::INFO,
            CliVulkanLogLevel::Warn => LevelFilter::WARN,
            CliVulkanLogLevel::Error => LevelFilter::ERROR,
        };
        if gfx_filter > rgpu_filter {
            targets = targets
                .with_target("rgpu_vk::instance::debug_utils", gfx_filter);
        }
    }
    targets
}

fn main() -> eyre::Result<()> {
    let app_dirs =
        directories::ProjectDirs::from("", "parengus", "samp-app-noext")
            .ok_or_else(|| {
                eyre::eyre!("Failed to determine application directories")
            })?;

    let self_dir = std::env::current_exe()?
        .parent()
        .ok_or_else(|| {
            eyre::eyre!(
                "Failed to take the parent directory of the \
                 current executable"
            )
        })?
        .to_owned();

    let log_dir = app_dirs
        .runtime_dir()
        .unwrap_or_else(|| app_dirs.data_dir())
        .to_owned();

    let cli_args = CliArgs::parse();

    let needs_subscriber = cli_args.tracing_log_level != TracingLogLevel::Off
        || cli_args.graphics_debug_level.is_some()
        || cli_args
            .rgpu_log_level
            .is_some_and(|l| l != TracingLogLevel::Off);

    if needs_subscriber {
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

        let filter = subscriber_filter(
            cli_args.tracing_log_level,
            cli_args.rgpu_log_level,
            cli_args.graphics_debug_level,
        );
        tracing_subscriber::registry()
            .with(stdout_log.with_filter(filter.clone()))
            .with(file_log.with_filter(filter))
            .init();

        tracing::debug!("log_file_path: {}", log_file_path.display());
        tracing::debug!("cli_args: {:#?}", cli_args);
    }

    let event_loop = winit::event_loop::EventLoop::builder().build()?;

    // SAFETY: Instance::new loads the Vulkan library. The Entry must
    // outlive all derived objects; Arc<Instance> ensures this.
    let instance = Arc::new(unsafe {
        rgpu_vk::instance::Instance::new(
            "samp-app-noext",
            cli_args.graphics_debug_level.map(Into::into),
            Some(&event_loop),
            InstanceConfig { surface: true },
        )
    }?);

    let device_config = DeviceConfig {
        swapchain: true,
        dynamic_rendering: false,
        synchronization2: false,
        maintenance1: false,
        shader_non_semantic_info: true,
        queue_config: QueueConfig {
            dedicated_transfer: cli_args.dedicated_transfer,
            dedicated_compute: cli_args.dedicated_compute,
            parallel: cli_args.parallel,
        },
        queue_config_strict: cli_args.queue_config_strict,
        min_sample_count: cli_args.aa.sample_count(),
        min_sample_count_strict: cli_args.aa_strict,
    };
    let requested_sample_count = cli_args.aa.sample_count();

    let mut app = AppRunner(Some(App::Initializing(InitializingState {
        instance,
        device_config,
        self_dir: self_dir.to_owned(),
        shader_debug_info: cli_args.shader_debug_info,
        requested_sample_count,
    })));

    tracing::trace!("Entering main event loop");
    Ok(event_loop.run_app(&mut app)?)
}

#[derive(Debug)]
struct AppRunner(Option<App>);

#[derive(Debug)]
enum App {
    Running(Box<RunningState>),
    Initializing(InitializingState),
    Suspended(Box<SuspendedState>),
    Exiting(ExitingState),
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const DEPTH_FORMAT_CANDIDATES: &[vk::Format] = &[
    vk::Format::D32_SFLOAT,
    vk::Format::D32_SFLOAT_S8_UINT,
    vk::Format::D24_UNORM_S8_UINT,
];

/// Per-frame-in-flight synchronization primitives and command buffer.
#[derive(Debug)]
struct FrameSync {
    image_available: Semaphore,
    in_flight_fence: Fence,
    command_buffer: ResettableCommandBuffer,
}

struct RunningStateTransitionGuard {
    device: Arc<Device>,
}

impl RunningStateTransitionGuard {
    /// # Safety
    /// This guard must be dropped (not forgotten) to ensure
    /// `vkDeviceWaitIdle` is called before resources are freed.
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
    requested_sample_count: vk::SampleCountFlags,
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
    // Swapchain is not Arc-wrapped so we can call create_framebuffers
    // (&mut self). `None` means the window is currently zero-sized.
    swapchain: Option<Swapchain<WinitWindow>>,
    /// One render-finished semaphore per swapchain image.
    /// `None` iff `swapchain` is `None`.
    render_finished_semaphores: Option<Vec<Semaphore>>,
    depth_format: vk::Format,
    sample_count: vk::SampleCountFlags,
    shader: ShaderModule,
    render_pass: RenderPass,
    pipeline: RenderPassPipeline,
    vertex_buffer: DeviceLocalBuffer,
    index_buffer: DeviceLocalBuffer,
    pipeline_color_format: vk::Format,
    command_pool: ResettableCommandPool,
    frames: Vec<FrameSync>,
    current_frame: usize,
    debug_counters: DebugCounters,
    camera_set_layout: Arc<DescriptorSetLayout>,
    material_set_layout: Arc<DescriptorSetLayout>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_pool: DescriptorPool,
    ubo_buffers: Vec<HostVisibleBuffer>,
    camera_descriptor_sets: Vec<DescriptorSet>,
    material_descriptor_set: DescriptorSet,
    start_time: Instant,
    asset_map: AssetMap,
    texture: Texture,
    sampler: Sampler,
}

impl std::fmt::Debug for RunningState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningState")
            .field("win", &self.win)
            .field("device", &self.device)
            .field("_surface", &self._surface)
            .field("pipeline_color_format", &self.pipeline_color_format)
            .field("current_frame", &self.current_frame)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct SuspendedState {
    win: Arc<WinitWindow>,
    device: Arc<Device>,
    shader: ShaderModule,
    render_pass: RenderPass,
    pipeline: RenderPassPipeline,
    vertex_buffer: DeviceLocalBuffer,
    index_buffer: DeviceLocalBuffer,
    pipeline_color_format: vk::Format,
    sample_count: vk::SampleCountFlags,
    command_pool: ResettableCommandPool,
    frames: Vec<FrameSync>,
    debug_counters: DebugCounters,
    camera_set_layout: Arc<DescriptorSetLayout>,
    material_set_layout: Arc<DescriptorSetLayout>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_pool: DescriptorPool,
    ubo_buffers: Vec<HostVisibleBuffer>,
    camera_descriptor_sets: Vec<DescriptorSet>,
    material_descriptor_set: DescriptorSet,
    start_time: Instant,
    asset_map: AssetMap,
    texture: Texture,
    sampler: Sampler,
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
                render_finished_semaphores: _,
                depth_format: _,
                sample_count,
                shader,
                render_pass,
                pipeline,
                vertex_buffer,
                index_buffer,
                pipeline_color_format,
                command_pool,
                frames,
                current_frame: _,
                debug_counters,
                camera_set_layout,
                material_set_layout,
                pipeline_layout,
                descriptor_pool,
                ubo_buffers,
                camera_descriptor_sets,
                material_descriptor_set,
                start_time,
                asset_map,
                texture,
                sampler,
            } = running_state;

            if let Err(e) = device.wait_idle() {
                tracing::error!(
                    "Error while waiting for device idle during \
                     suspend: {}",
                    e
                );
                self.transition_to_exiting("Running", event_loop);
                return;
            }

            self.set_suspended(SuspendedState {
                win,
                device,
                shader,
                render_pass,
                pipeline,
                vertex_buffer,
                index_buffer,
                pipeline_color_format,
                sample_count,
                command_pool,
                frames,
                debug_counters,
                camera_set_layout,
                material_set_layout,
                pipeline_layout,
                descriptor_pool,
                ubo_buffers,
                camera_descriptor_sets,
                material_descriptor_set,
                start_time,
                asset_map,
                texture,
                sampler,
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
    fn make_render_finished_semaphores(
        device: &Arc<Device>,
        image_count: usize,
    ) -> eyre::Result<Vec<Semaphore>> {
        (0..image_count)
            .map(|i| {
                Semaphore::new(
                    device,
                    Some(&format!("render finished img[{i}]")),
                )
                .map_err(eyre::Report::from)
            })
            .collect()
    }

    /// Resolve the effective MSAA sample count: highest supported count
    /// ≤ `requested` (both colour and depth framebuffer limits apply).
    /// Falls back through TYPE_8 → TYPE_4 → TYPE_2 → TYPE_1.
    fn resolve_sample_count(
        device: &Arc<Device>,
        requested: vk::SampleCountFlags,
    ) -> vk::SampleCountFlags {
        if requested == vk::SampleCountFlags::TYPE_1 {
            return vk::SampleCountFlags::TYPE_1;
        }
        let props = device.properties();
        let supported = props.limits.framebuffer_color_sample_counts
            & props.limits.framebuffer_depth_sample_counts;
        for count in [
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
            vk::SampleCountFlags::TYPE_1,
        ] {
            if count.as_raw() <= requested.as_raw() && supported.contains(count)
            {
                if count != requested {
                    tracing::warn!(
                        "Requested MSAA sample count {:?} not \
                         supported; using {:?}",
                        requested,
                        count,
                    );
                }
                return count;
            }
        }
        vk::SampleCountFlags::TYPE_1
    }

    /// Create one depth image per swapchain image (indexed by image
    /// index, not by frame-in-flight).
    fn make_depth_images(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        sample_count: vk::SampleCountFlags,
        count: usize,
    ) -> eyre::Result<Vec<DepthImage>> {
        (0..count)
            .map(|i| {
                DepthImage::new(
                    device,
                    extent.width,
                    extent.height,
                    depth_format,
                    sample_count,
                    Some(&format!("depth [{i}]")),
                )
                .map_err(eyre::Report::from)
            })
            .collect()
    }

    /// Create one MSAA colour image per swapchain image.
    fn make_msaa_images(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        format: vk::Format,
        sample_count: vk::SampleCountFlags,
        count: usize,
    ) -> eyre::Result<Vec<MsaaImage>> {
        (0..count)
            .map(|i| {
                MsaaImage::new(
                    device,
                    extent.width,
                    extent.height,
                    format,
                    sample_count,
                    Some(&format!("msaa [{i}]")),
                )
                .map_err(eyre::Report::from)
            })
            .collect()
    }

    fn draw_frame(state: &mut RunningState) -> DrawFrameOutcome {
        if state.swapchain.is_none() {
            return DrawFrameOutcome::Success;
        }

        let frame_idx = state.current_frame;
        let frame_objs = &mut state.frames[frame_idx];

        if frame_objs.in_flight_fence.is_submitted() {
            // SAFETY: fence was submitted with GPU work for this slot;
            // wait ensures all that work has completed.
            let fence_result =
                unsafe { frame_objs.in_flight_fence.wait_and_reset(u64::MAX) };
            if let Err(e) = fence_result {
                return DrawFrameOutcome::Fatal(format!(
                    "Fence wait/reset failed: {e}"
                ));
            }
        }

        let sc = state
            .swapchain
            .as_ref()
            .expect("swapchain present: checked above");
        let swapchain_raw = sc.raw_swapchain();
        let image_available = frame_objs.image_available.raw_semaphore();

        // SAFETY: image_available is unsignaled (fence wait above).
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

        // Framebuffers and depth images are indexed by swapchain image.
        let framebuffer = sc
            .framebuffers()
            .expect("framebuffers created with swapchain")[i];
        let render_finished = state
            .render_finished_semaphores
            .as_ref()
            .expect("render_finished_semaphores present with swapchain")[i]
            .raw_semaphore();

        let elapsed = state.start_time.elapsed().as_secs_f32();
        let aspect = extent.width as f32 / extent.height as f32;

        let view =
            look_at_rh(Vec3::new(2.0, -2.0, 1.5), Vec3::zero(), Vec3::unit_z());
        let ubo = Ubo {
            view_proj: perspective_rh_zo(PI32 / 3.0, aspect, 0.1, 100.0) * view,
        };
        if let Err(e) =
            state.ubo_buffers[frame_idx].write_pod(std::slice::from_ref(&ubo))
        {
            return DrawFrameOutcome::Fatal(format!("UBO write failed: {e}"));
        }

        let model = Mat4::<f32>::rotation_z(elapsed * PI32 * 2.0 / 5.0);
        let push = PushConstants { model };

        let pipeline_handle = state.pipeline.raw_pipeline();
        let frame_cmd = &mut state.frames[frame_idx].command_buffer;

        // SAFETY: fence wait guarantees buffer is not pending on GPU.
        if let Err(e) = unsafe { frame_cmd.reset() } {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer reset failed: {e}"
            ));
        }
        if let Err(e) = frame_cmd.begin() {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer begin failed: {e}"
            ));
        }

        // No explicit image barriers needed here.
        //
        // Color: initial_layout = UNDEFINED (render pass discards
        // previous contents; compatible with LOAD_OP_CLEAR). The
        // render pass end automatically transitions to PRESENT_SRC_KHR
        // via final_layout.
        //
        // Depth: initial_layout = UNDEFINED (discards; compatible with
        // LOAD_OP_CLEAR). The subpass dependency covers the required
        // stage/access synchronisation for both attachments.

        let color_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = [color_clear, depth_clear];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(state.render_pass.raw_render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        // SAFETY: recording; render pass and framebuffer are valid and
        // compatible; clear values match attachment count.
        unsafe {
            frame_cmd.begin_render_pass(
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            )
        };

        // SAFETY: inside render pass; pipeline is compatible with
        // the active render pass.
        unsafe { frame_cmd.bind_graphics_pipeline(pipeline_handle) };

        // SAFETY: layout is compatible with the bound pipeline;
        // descriptor set is valid and its buffer remains alive for
        // this frame's GPU work.
        unsafe {
            frame_cmd.bind_descriptor_sets(
                &state.pipeline_layout,
                0,
                &[
                    &state.camera_descriptor_sets[frame_idx],
                    &state.material_descriptor_set,
                ],
            )
        };

        // SAFETY: layout compatible; VERTEX stage; offset 0 matches
        // the declared range; push sized within 128-byte guarantee.
        unsafe {
            frame_cmd.push_constants(
                &state.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_ref(&push),
            )
        };

        // SAFETY: inside render pass; buffer is valid.
        unsafe { frame_cmd.bind_vertex_buffer(0, &state.vertex_buffer, 0) };

        // Standard Vulkan 1.0 viewport: Y points down in NDC, no
        // VK_KHR_maintenance1 needed.
        let h = extent.height as f32;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: h,
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

        // SAFETY: inside render pass; buffer is valid.
        unsafe {
            frame_cmd.bind_index_buffer(
                &state.index_buffer,
                0,
                vk::IndexType::UINT16,
            )
        };

        // SAFETY: all dynamic state set; render pass active; index
        // buffer bound.
        unsafe {
            frame_cmd.draw_indexed(SCENE_INDICES.len() as u32, 1, 0, 0, 0)
        };

        // SAFETY: inside a render pass.
        unsafe { frame_cmd.end_render_pass() };

        // No post-render barrier needed: the render pass final_layout
        // = PRESENT_SRC_KHR transitions the color image automatically.

        if let Err(e) = frame_cmd.end() {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer end failed: {e}"
            ));
        }

        let cmd_handle = frame_cmd.raw_command_buffer();

        // Old-style submit: wait on image_available at
        // COLOR_ATTACHMENT_OUTPUT, signal render_finished.
        let wait_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&image_available))
            .wait_dst_stage_mask(std::slice::from_ref(&wait_stage))
            .command_buffers(std::slice::from_ref(&cmd_handle))
            .signal_semaphores(std::slice::from_ref(&render_finished));

        let queue_index = if state.device.queue_config().parallel {
            frame_idx
        } else {
            0
        };

        // SAFETY: image_available signaled by acquire; render_finished
        // unsignaled; fence just reset; cmd in executable state.
        if let Err(e) = unsafe {
            state.device.graphics_present_queue_submit(
                std::slice::from_ref(&submit),
                Some(&mut state.frames[frame_idx].in_flight_fence),
                queue_index,
            )
        } {
            return DrawFrameOutcome::Fatal(format!(
                "Queue submit failed: {e}"
            ));
        }

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&render_finished))
            .swapchains(std::slice::from_ref(&swapchain_raw))
            .image_indices(std::slice::from_ref(&image_index));
        // SAFETY: render_finished signaled by submit; image in
        // PRESENT_SRC_KHR (via render pass final_layout transition).
        let present_result =
            unsafe { state.device.queue_present(&present_info, queue_index) };

        state.current_frame = (state.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        match present_result {
            Ok(false) if !suboptimal => DrawFrameOutcome::Success,
            Ok(_)
            | Err(rgpu_vk::device::QueuePresentError::Vulkan(
                vk::Result::ERROR_OUT_OF_DATE_KHR,
            )) => DrawFrameOutcome::SwapchainOutOfDate,
            Err(e) => DrawFrameOutcome::Fatal(format!("Present failed: {e}")),
        }
    }
}

fn build_render_pass(
    device: &Arc<Device>,
    color_format: vk::Format,
    depth_format: vk::Format,
    sample_count: vk::SampleCountFlags,
) -> eyre::Result<RenderPass> {
    Ok(RenderPass::new(
        device,
        &RenderPassDesc {
            color_format,
            depth_format,
            sample_count,
        },
    )?)
}

fn build_pipeline<F>(
    device: &Arc<Device>,
    shader: &ShaderModule,
    render_pass: &RenderPass,
    layout: Option<Arc<PipelineLayout>>,
    sample_count: vk::SampleCountFlags,
    name: Option<F>,
) -> eyre::Result<RenderPassPipeline>
where
    F: FnOnce() -> String,
{
    let vert = shader.entry_point("vert_main", ShaderStage::Vertex)?;
    let frag = shader.entry_point("frag_main", ShaderStage::Fragment)?;
    let vertex_bindings = [VertexBindingDesc {
        binding: 0,
        stride: size_of::<Vertex>() as u32,
        input_rate: VertexInputRate::VERTEX,
    }];
    let vertex_attributes = [
        VertexAttributeDesc {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, position) as u32,
        },
        VertexAttributeDesc {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, tex_coord) as u32,
        },
    ];
    Ok(RenderPassPipeline::new(
        device,
        &RenderPassPipelineDesc {
            stages: &[vert, frag],
            render_pass: render_pass.raw_render_pass(),
            subpass: 0,
            vertex_bindings: &vertex_bindings,
            vertex_attributes: &vertex_attributes,
            layout,
            cull_mode: CullModeFlags::BACK,
            // Negating Y in the projection matrix (col 1 = -f) cancels
            // Vulkan's Y-down framebuffer convention, so world-space CCW
            // winding remains CCW in screen space.
            front_face: FrontFace::COUNTER_CLOCKWISE,
            depth_test: true,
            depth_write: true,
            sample_count,
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
        let win = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("samp-app-noext")
                    .with_visible(false)
                    .with_inner_size(LogicalSize {
                        width: 1600u32,
                        height: 900u32,
                    }),
            )?,
        );

        // SAFETY: Surface must be destroyed only after all derived
        // swapchains are destroyed and no GPU work accesses them.
        // Swapchain holds Arc<Surface> so the surface outlives it.
        // wait_idle is always called before swapchain replacement and
        // on RunningState drop.
        let surface = Arc::new(unsafe {
            Surface::new(&state.instance, Arc::clone(&win))
        }?);

        let device = Arc::new(Device::create_compatible(
            &state.instance,
            &surface,
            state.device_config,
        )?);

        let sample_count =
            Self::resolve_sample_count(&device, state.requested_sample_count);

        let vertex_buffer_size =
            (SCENE_VERTICES.len() * size_of::<Vertex>()) as vk::DeviceSize;
        let mut staging_vertex_buffer = HostVisibleBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("rect staging vertex buffer"),
        )?;
        staging_vertex_buffer.write_pod(&SCENE_VERTICES)?;

        let mut vertex_buffer = DeviceLocalBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            Some("rect vertex buffer"),
        )?;

        let index_buffer_size =
            (SCENE_INDICES.len() * size_of::<u16>()) as vk::DeviceSize;
        let mut staging_index_buffer = HostVisibleBuffer::new(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("rect staging index buffer"),
        )?;
        staging_index_buffer.write_pod(&SCENE_INDICES)?;

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
        let mut upload_cmd = upload_command_pool.allocate_command_buffer()?;
        upload_cmd.begin()?;

        let win_size = win.inner_size();
        let debug_counters = DebugCounters::new();

        let depth_format = device
            .find_depth_format(DEPTH_FORMAT_CANDIDATES)
            .ok_or_else(|| eyre::eyre!("No supported depth format found"))?;

        let asset_map_path =
            state.self_dir.join("assets").join("asset_map.toml");
        let asset_map: AssetMap = toml::from_str(
            &std::fs::read_to_string(&asset_map_path).map_err(|e| {
                eyre::eyre!("Failed to read {}: {e}", asset_map_path.display())
            })?,
        )?;
        tracing::debug!(asset_map = ?asset_map.map, "Loaded asset map");

        let assets_dir = state.self_dir.join("assets");
        let shader_name = if state.shader_debug_info {
            "shader-debug"
        } else {
            "shader"
        };
        let shader_filename = asset_map
            .map
            .get(shader_name)
            .ok_or_else(|| eyre::eyre!("asset '{shader_name}' not in map"))?;
        let shader_path = assets_dir.join(shader_filename);
        let shader_path = if state.shader_debug_info && !shader_path.exists() {
            tracing::warn!(
                path = %shader_path.display(),
                "Shader debug info requested but debug shader was \
                 not found; falling back to non-debug shader"
            );
            let fallback = asset_map
                .map
                .get("shader")
                .ok_or_else(|| eyre::eyre!("asset 'shader' not in map"))?;
            assets_dir.join(fallback)
        } else {
            shader_path
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

        let camera_set_layout = Arc::new(DescriptorSetLayout::new(
            &device,
            &[DescriptorBindingDesc {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
            }],
            Some("camera set layout"),
        )?);
        let material_set_layout = Arc::new(DescriptorSetLayout::new(
            &device,
            &[DescriptorBindingDesc {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
            }],
            Some("material set layout"),
        )?);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            &device,
            &PipelineLayoutDesc {
                set_layouts: &[&camera_set_layout, &material_set_layout],
                push_constant_ranges: &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: size_of::<PushConstants>() as u32,
                }],
            },
        )?);

        // Pipeline color format is determined from the swapchain (or a
        // sensible default when the window is zero-sized at startup).
        let pipeline_color_format = if win_size.width > 0 && win_size.height > 0
        {
            // Peek at the preferred format without keeping the
            // swapchain — the real swapchain is created below.
            vk::Format::B8G8R8A8_SRGB
        } else {
            vk::Format::B8G8R8A8_SRGB
        };

        let render_pass = {
            let _span = tracing::trace_span!("render_pass_create").entered();
            build_render_pass(
                &device,
                pipeline_color_format,
                depth_format,
                sample_count,
            )?
        };

        let pipeline = {
            let _span = tracing::trace_span!(
                "pipeline_create",
                color_format = ?pipeline_color_format,
            )
            .entered();
            build_pipeline(
                &device,
                &shader,
                &render_pass,
                Some(Arc::clone(&pipeline_layout)),
                sample_count,
                Some({
                    let idx = debug_counters.next_pipeline();
                    move || format!("main pipeline {idx}")
                }),
            )?
        };

        let command_pool = ResettableCommandPool::new(
            &device,
            device.graphics_present_queue_family(),
            Some("graphics command pool"),
        )?;

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| -> eyre::Result<FrameSync> {
                Ok(FrameSync {
                    image_available: Semaphore::new(
                        &device,
                        Some(&format!("image available [{i}]")),
                    )?,
                    in_flight_fence: Fence::new(
                        &device,
                        false,
                        Some(&format!("in flight [{i}]")),
                    )?,
                    command_buffer: command_pool.allocate_command_buffer()?,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let ubo_size = size_of::<Ubo>() as vk::DeviceSize;
        let ubo_buffers = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                HostVisibleBuffer::new(
                    &device,
                    ubo_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    Some(&format!("ubo [{i}]")),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        // One camera set per frame + one material set = FIF + 1 total.
        let descriptor_pool = DescriptorPool::new(
            &device,
            (MAX_FRAMES_IN_FLIGHT + 1) as u32,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
            ],
            Some("descriptor pool"),
        )?;

        let camera_layouts: Vec<&DescriptorSetLayout> = (0
            ..MAX_FRAMES_IN_FLIGHT)
            .map(|_| camera_set_layout.as_ref())
            .collect();
        // SAFETY: descriptor_pool outlives camera_descriptor_sets and
        // material_descriptor_set (all stored in the same state struct,
        // dropped in declaration order — sets before pool).
        let camera_descriptor_sets =
            unsafe { descriptor_pool.allocate_sets(&camera_layouts) }?;
        for (i, (set, buf)) in camera_descriptor_sets
            .iter()
            .zip(ubo_buffers.iter())
            .enumerate()
        {
            set.set_name(&device, Some(&format!("camera set {i}")));
            // SAFETY: buf is a valid UNIFORM_BUFFER that remains alive
            // for the lifetime of the descriptor set.
            unsafe { set.write_uniform_buffer(&device, 0, buf, ubo_size) };
        }
        // SAFETY: same guarantee as camera_descriptor_sets above.
        let mut material_sets = unsafe {
            descriptor_pool.allocate_sets(&[material_set_layout.as_ref()])
        }?;
        let material_descriptor_set = material_sets
            .pop()
            .expect("allocated exactly one material set");
        material_descriptor_set.set_name(&device, Some("material set"));

        let tex_filename = asset_map
            .map
            .get("statue-tex")
            .ok_or_else(|| eyre::eyre!("asset 'statue-tex' not in map"))?;
        let tex_path = assets_dir.join(tex_filename);
        let tex_img = image::open(&tex_path)
            .map_err(|e| {
                eyre::eyre!(
                    "Failed to open texture {}: {e}",
                    tex_path.display()
                )
            })?
            .into_rgba8();
        let (tex_width, tex_height) = tex_img.dimensions();
        let tex_bytes = tex_img.into_raw();
        let tex_staging_size = tex_bytes.len() as vk::DeviceSize;

        let mut tex_staging = HostVisibleBuffer::new(
            &device,
            tex_staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("statue-tex staging"),
        )?;
        tex_staging.write_pod(tex_bytes.as_slice())?;

        let texture = rgpu_vk::image::Texture::new(
            &device,
            tex_width,
            tex_height,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            Some("statue-tex"),
        )?;

        // Record all uploads into one command buffer, submit once, then
        // wait for completion before dropping the staging buffers.
        //
        // SAFETY: upload_cmd is in the recording state; all buffers
        // remain alive until wait_idle completes.
        unsafe {
            vertex_buffer
                .record_copy_from(&mut upload_cmd, &staging_vertex_buffer)
        }?;
        // SAFETY: same as vertex buffer copy above.
        unsafe {
            index_buffer
                .record_copy_from(&mut upload_cmd, &staging_index_buffer)
        }?;
        // SAFETY: same as buffer copies above; image in recording
        // state; uses VK 1.0 core barriers (no sync2 required).
        unsafe { texture.record_copy_from(&mut upload_cmd, &tex_staging) }?;

        upload_cmd.end()?;
        let cmd_handle = upload_cmd.raw_command_buffer();
        let upload_submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd_handle));
        // SAFETY: command buffer is executable; referenced resources
        // remain alive until wait_idle below.
        unsafe {
            device.graphics_present_queue_submit(
                std::slice::from_ref(&upload_submit),
                None,
                0,
            )
        }?;
        device.wait_idle()?;

        let sampler = rgpu_vk::sampler::Sampler::new(
            &device,
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            Some("statue-tex sampler"),
        )?;

        // SAFETY: texture and sampler live in RunningState /
        // SuspendedState and outlive any command buffer referencing
        // this descriptor set.
        unsafe {
            material_descriptor_set
                .write_texture_sampler(&device, 0, &texture, &sampler)
        };

        // Create the initial swapchain (if the window is non-zero).
        let (swapchain, render_finished_semaphores) = if win_size.width == 0
            || win_size.height == 0
        {
            tracing::trace!(
                "Skipping initial swapchain create because window \
                     extent is zero: {}x{}",
                win_size.width,
                win_size.height
            );
            (None, None)
        } else {
            let sc_idx = debug_counters.next_swapchain();
            let requested_extent = vk::Extent2D {
                width: win_size.width,
                height: win_size.height,
            };
            let _span = tracing::trace_span!(
                "initial_swapchain_create",
                requested_width = requested_extent.width,
                requested_height = requested_extent.height
            )
            .entered();
            let mut sc = Swapchain::new(
                &device,
                &surface,
                requested_extent,
                true,
                None,
                Some(move || format!("Swapchain {sc_idx}")),
            )?;
            let image_count = sc.images().len();
            let depths = Self::make_depth_images(
                &device,
                sc.extent(),
                depth_format,
                sample_count,
                image_count,
            )?;
            let msaa = if sample_count != vk::SampleCountFlags::TYPE_1 {
                Some(Self::make_msaa_images(
                    &device,
                    sc.extent(),
                    sc.format(),
                    sample_count,
                    image_count,
                )?)
            } else {
                None
            };
            // SAFETY: image collections match swapchain image count;
            // swapchain takes ownership.
            unsafe {
                sc.create_framebuffers(
                    render_pass.raw_render_pass(),
                    Some(depths),
                    msaa,
                )
            }?;
            let sems =
                Self::make_render_finished_semaphores(&device, image_count)?;
            (Some(sc), Some(sems))
        };

        // SAFETY: guard will call wait_idle in Drop. Must not be
        // forgotten.
        let idle_guard =
            unsafe { RunningStateTransitionGuard::new(Arc::clone(&device)) };
        win.set_visible(true);
        Ok(RunningState {
            _idle_guard: idle_guard,
            win,
            device,
            _surface: surface,
            swapchain,
            render_finished_semaphores,
            depth_format,
            sample_count,
            shader,
            render_pass,
            pipeline,
            vertex_buffer,
            index_buffer,
            pipeline_color_format,
            command_pool,
            frames,
            current_frame: 0,
            debug_counters,
            camera_set_layout,
            material_set_layout,
            pipeline_layout,
            descriptor_pool,
            ubo_buffers,
            camera_descriptor_sets,
            material_descriptor_set,
            start_time: Instant::now(),
            asset_map,
            texture,
            sampler,
        })
    }

    fn suspended_to_running(
        state: SuspendedState,
    ) -> eyre::Result<RunningState> {
        // SAFETY: The surface outlives all derived swapchains via Arc.
        let surface = Arc::new(unsafe {
            Surface::new(state.device.parent(), Arc::clone(&state.win))
        }?);

        let win_size = state.win.inner_size();
        let debug_counters = state.debug_counters;
        let depth_format = state
            .device
            .find_depth_format(DEPTH_FORMAT_CANDIDATES)
            .ok_or_else(|| eyre::eyre!("No supported depth format found"))?;

        let (mut render_pass, mut pipeline, mut pipeline_color_format) =
            if win_size.width > 0 && win_size.height > 0 {
                // Peek at the swapchain format by creating a temporary
                // swapchain; if it matches, reuse existing pass/pipeline.
                // We don't actually create the full swapchain here —
                // that happens below. Reuse the stored pipeline_color_format
                // as the hint; the driver will honor it if supported.
                (
                    state.render_pass,
                    state.pipeline,
                    state.pipeline_color_format,
                )
            } else {
                (
                    state.render_pass,
                    state.pipeline,
                    state.pipeline_color_format,
                )
            };

        let (swapchain, render_finished_semaphores) = if win_size.width == 0
            || win_size.height == 0
        {
            tracing::trace!(
                "Skipping swapchain create on resume because window \
                     extent is zero: {}x{}",
                win_size.width,
                win_size.height
            );
            (None, None)
        } else {
            let sc_idx = debug_counters.next_swapchain();
            let mut sc = Swapchain::new(
                &state.device,
                &surface,
                vk::Extent2D {
                    width: win_size.width,
                    height: win_size.height,
                },
                true,
                Some(state.pipeline_color_format),
                Some(move || format!("Swapchain {sc_idx}")),
            )?;

            let new_format = sc.format();
            if new_format != pipeline_color_format {
                tracing::debug!(
                    "Swapchain format changed on resume \
                         ({:?} -> {:?}); recreating render pass \
                         and pipeline",
                    pipeline_color_format,
                    new_format,
                );
                render_pass = build_render_pass(
                    &state.device,
                    new_format,
                    depth_format,
                    state.sample_count,
                )?;
                pipeline = build_pipeline(
                    &state.device,
                    &state.shader,
                    &render_pass,
                    Some(Arc::clone(&state.pipeline_layout)),
                    state.sample_count,
                    Some({
                        let idx = debug_counters.next_pipeline();
                        move || format!("main pipeline {idx}")
                    }),
                )?;
                pipeline_color_format = new_format;
            }

            let image_count = sc.images().len();
            let depths = Self::make_depth_images(
                &state.device,
                sc.extent(),
                depth_format,
                state.sample_count,
                image_count,
            )?;
            let msaa = if state.sample_count != vk::SampleCountFlags::TYPE_1 {
                Some(Self::make_msaa_images(
                    &state.device,
                    sc.extent(),
                    sc.format(),
                    state.sample_count,
                    image_count,
                )?)
            } else {
                None
            };
            // SAFETY: image collections match swapchain image count;
            // swapchain takes ownership.
            unsafe {
                sc.create_framebuffers(
                    render_pass.raw_render_pass(),
                    Some(depths),
                    msaa,
                )
            }?;
            let sems = Self::make_render_finished_semaphores(
                &state.device,
                image_count,
            )?;
            (Some(sc), Some(sems))
        };

        // SAFETY: guard will call wait_idle in Drop. Must not be
        // forgotten.
        let idle_guard = unsafe {
            RunningStateTransitionGuard::new(Arc::clone(&state.device))
        };
        Ok(RunningState {
            _idle_guard: idle_guard,
            win: state.win,
            device: state.device,
            _surface: surface,
            swapchain,
            render_finished_semaphores,
            depth_format,
            sample_count: state.sample_count,
            shader: state.shader,
            render_pass,
            pipeline,
            vertex_buffer: state.vertex_buffer,
            index_buffer: state.index_buffer,
            pipeline_color_format,
            command_pool: state.command_pool,
            frames: state.frames,
            current_frame: 0,
            debug_counters,
            camera_set_layout: state.camera_set_layout,
            material_set_layout: state.material_set_layout,
            pipeline_layout: state.pipeline_layout,
            descriptor_pool: state.descriptor_pool,
            ubo_buffers: state.ubo_buffers,
            camera_descriptor_sets: state.camera_descriptor_sets,
            material_descriptor_set: state.material_descriptor_set,
            start_time: state.start_time,
            asset_map: state.asset_map,
            texture: state.texture,
            sampler: state.sampler,
        })
    }

    fn recreate_swapchain_if_needed(
        state: &mut RunningState,
        desired_extent: vk::Extent2D,
    ) -> bool {
        if desired_extent.width == 0 || desired_extent.height == 0 {
            let _span = tracing::trace_span!(
                "swapchain_teardown",
                width = desired_extent.width,
                height = desired_extent.height,
            )
            .entered();
            if let Err(e) = state.device.wait_idle() {
                tracing::error!(
                    "Error while waiting for device idle on zero \
                     extent: {}",
                    e
                );
                return false;
            }
            state.swapchain = None;
            state.render_finished_semaphores = None;
            return true;
        }

        if let Some(existing_sc) = state.swapchain.as_ref()
            && existing_sc.extent() == desired_extent
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

        // Wait for all GPU work to complete before dropping the old
        // swapchain. Image views (and framebuffers in noext) must not
        // be destroyed while in-flight commands still reference them.
        if let Err(e) = state.device.wait_idle() {
            tracing::error!(
                "Error waiting for device idle before swapchain \
                 recreation: {}",
                e
            );
            return false;
        }

        state.swapchain.take();

        let sc_idx = state.debug_counters.next_swapchain();
        let sc_result = Swapchain::new(
            &state.device,
            &state._surface,
            desired_extent,
            true,
            Some(state.pipeline_color_format),
            Some(move || format!("Swapchain {sc_idx}")),
        );

        let mut sc = match sc_result {
            Ok(sc) => sc,
            Err(e) => {
                tracing::error!("Error while recreating swapchain: {}", e);
                return false;
            }
        };

        tracing::trace!(
            "Swapchain recreation succeeded for extent: {}x{}",
            desired_extent.width,
            desired_extent.height
        );

        let new_format = sc.format();
        let new_extent = sc.extent();
        let image_count = sc.images().len();

        // Recreate render pass and pipeline if format changed.
        if new_format != state.pipeline_color_format {
            tracing::debug!(
                "Swapchain format changed on resize \
                 ({:?} -> {:?}); recreating render pass and pipeline",
                state.pipeline_color_format,
                new_format,
            );
            let new_rp = match build_render_pass(
                &state.device,
                new_format,
                state.depth_format,
                state.sample_count,
            ) {
                Ok(rp) => rp,
                Err(e) => {
                    tracing::error!(
                        "Error recreating render pass after format \
                         change: {}",
                        e
                    );
                    return false;
                }
            };
            let new_pipeline = match build_pipeline(
                &state.device,
                &state.shader,
                &new_rp,
                Some(Arc::clone(&state.pipeline_layout)),
                state.sample_count,
                Some({
                    let idx = state.debug_counters.next_pipeline();
                    move || format!("main pipeline {idx}")
                }),
            ) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(
                        "Error recreating pipeline after format \
                         change: {}",
                        e
                    );
                    return false;
                }
            };
            state.render_pass = new_rp;
            state.pipeline = new_pipeline;
            state.pipeline_color_format = new_format;
        }

        // Create depth images per swapchain image.
        let depths = match Self::make_depth_images(
            &state.device,
            new_extent,
            state.depth_format,
            state.sample_count,
            image_count,
        ) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("Error recreating depth images: {}", e);
                return false;
            }
        };

        let msaa = if state.sample_count != vk::SampleCountFlags::TYPE_1 {
            match Self::make_msaa_images(
                &state.device,
                new_extent,
                new_format,
                state.sample_count,
                image_count,
            ) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::error!("Error recreating MSAA images: {}", e);
                    return false;
                }
            }
        } else {
            None
        };

        // SAFETY: image collections match swapchain image count;
        // swapchain takes ownership.
        if let Err(e) = unsafe {
            sc.create_framebuffers(
                state.render_pass.raw_render_pass(),
                Some(depths),
                msaa,
            )
        } {
            tracing::error!("Error creating framebuffers: {}", e);
            return false;
        }

        let sems = match Self::make_render_finished_semaphores(
            &state.device,
            image_count,
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    "Error recreating render-finished semaphores: {}",
                    e
                );
                return false;
            }
        };

        state.render_finished_semaphores = Some(sems);
        state.swapchain = Some(sc);
        true
    }

    fn desired_extent_for_event(
        win: &WinitWindow,
        window_event: &WindowEvent,
    ) -> Option<vk::Extent2D> {
        match window_event {
            WindowEvent::Resized(size) => Some(vk::Extent2D {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = win.inner_size();
                Some(vk::Extent2D {
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
        if self.take_running().is_some() {
            // _idle_guard drops first (first field), calling wait_idle
            // before all other resources are freed.
            self.transition_to_exiting("Running", event_loop);
        }
    }

    fn take_initializing(&mut self) -> Option<InitializingState> {
        match self.0.take()? {
            App::Initializing(s) => Some(s),
            other => {
                self.0 = Some(other);
                None
            }
        }
    }

    fn take_running(&mut self) -> Option<RunningState> {
        match self.0.take()? {
            App::Running(s) => Some(*s),
            other => {
                self.0 = Some(other);
                None
            }
        }
    }

    fn take_suspended(&mut self) -> Option<SuspendedState> {
        match self.0.take()? {
            App::Suspended(s) => Some(*s),
            other => {
                self.0 = Some(other);
                None
            }
        }
    }

    fn as_running(&self) -> Option<&RunningState> {
        match self.0.as_ref()? {
            App::Running(s) => Some(s),
            _ => None,
        }
    }

    fn as_running_mut(&mut self) -> Option<&mut RunningState> {
        match self.0.as_mut()? {
            App::Running(s) => Some(s),
            _ => None,
        }
    }

    fn set_running(&mut self, state: RunningState) {
        self.0 = Some(App::Running(Box::new(state)));
    }

    fn set_suspended(&mut self, state: SuspendedState) {
        self.0 = Some(App::Suspended(Box::new(state)));
    }

    fn set_exiting(&mut self, state: ExitingState) {
        self.0 = Some(App::Exiting(state));
    }

    fn is_running_window(&self, window_id: winit::window::WindowId) -> bool {
        self.as_running().is_some_and(|s| s.win.id() == window_id)
    }

    fn is_exiting(&self) -> bool {
        matches!(self.0, Some(App::Exiting(_)))
    }
}
