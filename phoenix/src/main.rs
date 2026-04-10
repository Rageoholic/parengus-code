#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::{
    cell::Cell, collections::HashMap, f32::consts::PI as PI32, fs,
    mem::size_of, sync::Arc, time::Instant,
};

use asset_loader::{AssetMap, MeshAsset, TexAsset};
use asset_shared::{
    ColorSpace, TexFormat, TexRole, TextureId, mesh_id, shader_id,
};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use parengus_tracing::{TracingLogLevel, init_default};
use rgpu_vk::{
    ash::vk::{self, BufferImageCopy},
    buffer::{BufferHandle, DeviceLocalBuffer, HostVisibleBuffer},
    command::{
        self, DebugLabel, DebugLabelType, LazyDebugLabelReturn, Recordable,
        ResettableCommandBuffer, ResettableCommandPool, TransientCommandPool,
    },
    descriptor::{
        DescriptorBindingDesc, DescriptorPool, DescriptorSet,
        DescriptorSetLayout,
    },
    device::{Device, DeviceConfig, Graphics, QueueConfig, Transfer},
    image::{DepthImage, MsaaImage, Texture},
    instance::{Instance, InstanceConfig},
    memory::image_barrier2,
    pipeline::{
        CullModeFlags, DynamicPipeline, DynamicPipelineDesc, FrontFace,
        PipelineLayout, PipelineLayoutDesc, VertexAttributeDesc,
        VertexBindingDesc, VertexInputRate,
    },
    sampler::Sampler,
    shader::{ShaderModule, ShaderStage},
    surface::Surface,
    swapchain::Swapchain,
    sync::{Fence, Semaphore},
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

/// Camera uniform buffer object — combined view-projection matrix.
///
/// view_proj = projection × view; baked on the CPU so the shader
/// needs only one matrix multiply per vertex. Updated every frame
/// (aspect ratio can change on resize; view is fixed but cheap to
/// recompute). Carried in the camera descriptor set (set 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Ubo {
    view_proj: Mat4<f32>,
}

/// Push-constant block — model matrix + material index.
///
/// Updated per submesh draw. model (64 B) + material_idx (4 B) = 68 B
/// total; visible to both vertex and fragment stages.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    model: Mat4<f32>,
    material_idx: u32,
}

/// Per-material GPU data stored in the material SSBO.
///
/// One entry per submesh draw, indexed via `material_idx` in the
/// push constants. Indices point into the bindless texture array;
/// `emissive_idx` is negative when no emissive texture is present.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MaterialGpu {
    albedo_idx: u32,
    normal_idx: u32,
    orm_idx: u32,
    emissive_idx: i32,
}

/// Per-submesh draw record built at load time from SubMeshInfo.
#[derive(Debug)]
struct SubmeshDraw {
    first_index: u32,
    index_count: u32,
    /// Vertex base offset into the combined vertex buffer for this
    /// mesh (passed as the `vertex_offset` parameter of
    /// `vkCmdDrawIndexed`).
    vertex_offset: i32,
    /// Index into the material SSBO for this submesh.
    material_idx: u32,
    /// Index into MODEL_SCALES / MODEL_OFFSETS.
    model_idx: usize,
    /// Pre-baked TRS from the glTF node, in Z-up space.
    sub_trs: Mat4<f32>,
}

// ---------------------------------------------------------------------------
// Matrix math — right-handed, +Z up
// ---------------------------------------------------------------------------
//
// We build our own look-at and perspective helpers using vek's Vec/Mat
// primitives so the math is explicit and auditable.  vek's higher-level
// helpers (e.g. Mat4::perspective_rh_zo) are intentionally avoided.

/// Build a right-handed view matrix that places the camera at `eye`
/// looking toward `center`, with `world_up` as the reference up axis.
///
/// Convention: after this transform the camera sits at the origin and
/// looks down its **negative Z** axis (standard camera-space convention).
///
/// Column-major storage matches vek's [`Mat4`] layout.
#[rustfmt::skip]
fn look_at_rh(
    eye: Vec3<f32>,
    center: Vec3<f32>,
    world_up: Vec3<f32>,
) -> Mat4<f32> {
    // Forward: unit vector from eye toward the target.
    let f = (center - eye).normalized();
    // Right: perpendicular to forward in the horizontal plane.
    // cross(f, world_up) gives a right-hand-rule right vector.
    let r = f.cross(world_up).normalized();
    // Up: recomputed from r and f to guarantee orthogonality even
    // when world_up is not exactly perpendicular to f.
    let u = r.cross(f);

    // The view matrix is R * T, where T translates by -eye and R
    // rotates so that (r, u, -f) align with the camera axes.
    // vek's Mat4::new takes 16 scalars in column-major order
    // (all of col 0 first, then col 1, …).
    //   col 0 = [ r.x,        u.x,        -f.x,       0 ]
    //   col 1 = [ r.y,        u.y,        -f.y,       0 ]
    //   col 2 = [ r.z,        u.z,        -f.z,       0 ]
   //   col 3 = [ -dot(r,eye), -dot(u,eye), dot(f,eye), 1 ]
    Mat4::from_col_arrays(
        [
        [r.x,         u.x,         -f.x,        0.0,],
        [r.y,         u.y,         -f.y,        0.0,],
        [r.z,         u.z,         -f.z,        0.0,],
        [-r.dot(eye), -u.dot(eye), f.dot(eye),  1.0,]],
    )
}

/// Build a right-handed infinite reversed-Z perspective projection
/// matrix for Vulkan's clip space (depth range [0, 1], Y-down).
///
/// - `fov_y`: vertical field of view in radians.
/// - `aspect`: viewport width divided by height.
/// - `near`: distance to the near plane (positive).
///
/// Reversed-Z maps near → 1 and far → 0 (the far plane is at ∞).
/// Pair with `GREATER_OR_EQUAL` depth compare and a 0.0 depth clear.
/// This gives full float precision near the camera where it matters.
///
/// Derivation: z_ndc = (A·z_view + B) / (−z_view). Boundary
/// conditions near=1, far=0 as far→∞ yield A=0, B=near, so
/// z_clip = near, w_clip = −z_view, z_ndc = near/−z_view.
///
/// Negates Y (col 1 = [0, −f, 0, 0]) for Vulkan's Y-down NDC.
fn perspective_rh_zo_infinite_rev(
    fov_y: f32,
    aspect: f32,
    near: f32,
) -> Mat4<f32> {
    let f = 1.0 / (fov_y * 0.5).tan();
    // col 0 = [ f/aspect,  0,    0,  0 ]
    // col 1 = [ 0,        -f,    0,  0 ]  ← Y-down
    // col 2 = [ 0,         0,    0, -1 ]  ← A=0; w_clip = -z_view
    // col 3 = [ 0,         0, near,  0 ]  ← B=near
    Mat4::from_col_arrays([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, near, 0.0],
    ])
}

/// Build a TRS matrix from SubMeshInfo fields (Z-up, column-major).
/// `r` = [x, y, z, w] quaternion (vek convention).
#[rustfmt::skip]
fn trs_to_mat4(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> Mat4<f32> {
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let rot = Mat4::from_col_arrays([
        [1.-2.*(y*y+z*z),    2.*(x*y+z*w),    2.*(x*z-y*w), 0.],
        [   2.*(x*y-z*w), 1.-2.*(x*x+z*z),    2.*(y*z+x*w), 0.],
        [   2.*(x*z+y*w),    2.*(y*z-x*w), 1.-2.*(x*x+y*y), 0.],
        [             0.,              0.,               0.,  1.],
    ]);
    let scale = Mat4::from_col_arrays([
        [s[0],  0.,   0.,  0.],
        [  0., s[1],  0.,  0.],
        [  0.,   0., s[2], 0.],
        [  0.,   0.,   0., 1.],
    ]);
    let trans:Mat4<f32> = Mat4::translation_3d(Vec3::new(t[0], t[1], t[2]));
    trans * rot * scale
}

/// Anti-aliasing mode selected via `--aa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AntiAliasing {
    /// No anti-aliasing (`VK_SAMPLE_COUNT_1_BIT`).
    Off,
    /// 2× MSAA.
    Msaa2,
    /// 4× MSAA (default).
    Msaa4,
    /// 8× MSAA.
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
#[command(about = "Sample Vulkan app", long_about = None)]
struct CliArgs {
    /// Tracing verbosity for stdout/file logging.
    #[arg(long, default_value = "error")]
    tracing_log_level: TracingLogLevel,

    /// Vulkan validation/debug callback severity threshold.
    #[arg(long)]
    graphics_debug_level: Option<CliVulkanLogLevel>,

    /// Extra per-target tracing overrides, repeatable: e.g. --trace-target rgpu_vk=debug
    #[arg(long = "trace-target")]
    trace_target: Vec<String>,

    /// Use a dedicated transfer queue family (default: true).
    #[arg(long, default_value = "true")]
    dedicated_transfer: bool,

    /// Use a dedicated compute queue family (default: true).
    #[arg(long, default_value = "true")]
    dedicated_compute: bool,

    /// Use a dedicated present queue family (default: false).
    #[arg(long, default_value = "false")]
    dedicated_present: bool,

    /// Treat queue config as hard requirements; error if any
    /// requested axis is unsatisfied.
    #[arg(long)]
    queue_config_strict: bool,

    /// Load debug-info shader binary (`shader.debug.spv`) for RenderDoc.
    #[arg(long)]
    shader_debug_info: bool,

    /// Cap Vulkan instance API version to 1.0 and force extension
    /// code paths for all device features. Useful for testing
    /// pre-1.3 / pre-1.1 fallback paths.
    ///
    /// **Currently broken.**
    #[arg(long)]
    vk_strict_config: bool,

    /// Disable ANSI color codes in stdout log output.
    #[arg(long)]
    no_color: bool,

    /// Anti-aliasing mode (off, msaa2, msaa4, msaa8).
    #[arg(long, default_value = "msaa4")]
    aa: AntiAliasing,

    /// Error if the device does not support the requested AA mode.
    #[arg(long)]
    aa_strict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
/// Vulkan validation/debug callback severity threshold.
enum CliVulkanLogLevel {
    /// Include verbose diagnostics (maps to Vulkan VERBOSE).
    Trace,
    /// Include informational messages and above.
    Info,
    /// Include warnings and errors.
    Warn,
    /// Include errors only.
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

fn tex_vk_format(fmt: TexFormat, cs: ColorSpace) -> vk::Format {
    match (fmt, cs) {
        (TexFormat::Rgba8, ColorSpace::Srgb) => vk::Format::R8G8B8A8_SRGB,
        (TexFormat::Rgba8, ColorSpace::Linear) => vk::Format::R8G8B8A8_UNORM,
        (TexFormat::Bc4, ColorSpace::Linear) => vk::Format::BC4_UNORM_BLOCK,
        (TexFormat::Bc5, ColorSpace::Linear) => vk::Format::BC5_UNORM_BLOCK,
        (TexFormat::Bc7, ColorSpace::Srgb) => vk::Format::BC7_SRGB_BLOCK,
        (TexFormat::Bc7, ColorSpace::Linear) => vk::Format::BC7_UNORM_BLOCK,
        _ => panic!(
            "unsupported tex format combination: \
             {fmt:?} + {cs:?}"
        ),
    }
}

pub fn main() -> eyre::Result<()> {
    let app_dirs = directories::ProjectDirs::from("", "parengus", "phoenix")
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

    let mut targets: std::collections::HashMap<String, TracingLogLevel> =
        std::collections::HashMap::new();
    for spec in &cli_args.trace_target {
        if let Some(eq) = spec.find('=') {
            let (tgt, lvl) = spec.split_at(eq);
            let lvl = &lvl[1..];
            if let Ok(parsed) = lvl.parse::<TracingLogLevel>() {
                targets.insert(tgt.to_string(), parsed);
            }
        }
    }

    if let Some(gfx) = cli_args.graphics_debug_level {
        let gfx_lvl = match gfx {
            CliVulkanLogLevel::Trace => TracingLogLevel::Trace,
            CliVulkanLogLevel::Info => TracingLogLevel::Info,
            CliVulkanLogLevel::Warn => TracingLogLevel::Warn,
            CliVulkanLogLevel::Error => TracingLogLevel::Error,
        };
        use tracing_subscriber::filter::LevelFilter as LF;
        let gfx_filter: LF = LF::from(gfx_lvl);
        let rgpu_filter = targets.get("rgpu_vk").copied().map(LF::from);
        let instance_filter = targets
            .get("rgpu_vk::instance::debug_utils")
            .copied()
            .map(LF::from);

        let should_set = if let Some(inst) = instance_filter {
            gfx_filter > inst
        } else if let Some(rg) = rgpu_filter {
            gfx_filter > rg
        } else {
            let base: LF = LF::from(cli_args.tracing_log_level);
            gfx_filter > base
        };
        if should_set {
            targets
                .insert("rgpu_vk::instance::debug_utils".to_string(), gfx_lvl);
        }
    }

    let default_level = cli_args.tracing_log_level;
    let need_tracing =
        default_level != TracingLogLevel::Off || !targets.is_empty();
    if need_tracing {
        fs::create_dir_all(&log_dir)?;
        let mut log_file_path = log_dir.clone();
        log_file_path.push("log-file");
        log_file_path.set_extension("txt");
        let file_path = Some(log_file_path);
        init_default(targets, default_level, file_path, cli_args.no_color)
            .map_err(|e| eyre::eyre!("init tracing: {e}"))?;
    }

    let event_loop = winit::event_loop::EventLoop::builder().build()?;

    // SAFETY: Instance::new loads the Vulkan library via
    // ash::Entry::load (libloading). The loaded Entry must outlive
    // all Vulkan objects derived from it; wrapping the Instance in
    // an Arc ensures it stays alive at least as long as any derived
    // object does.
    let instance = Arc::new(unsafe {
        rgpu_vk::instance::Instance::new(
            "phoenix",
            cli_args.graphics_debug_level.map(Into::into),
            Some(&event_loop),
            InstanceConfig {
                surface: true,
                vk_1_0_strict: cli_args.vk_strict_config,
                physical_device_features2: true,
            },
        )
    }?);

    let device_config = DeviceConfig {
        swapchain: true,
        dynamic_rendering: true,
        synchronization2: true,
        maintenance1: false,
        shader_non_semantic_info: true,
        queue_config: QueueConfig {
            dedicated_transfer: cli_args.dedicated_transfer,
            dedicated_compute: cli_args.dedicated_compute,
            dedicated_present: cli_args.dedicated_present,
        },
        queue_config_strict: cli_args.queue_config_strict,
        min_sample_count: cli_args.aa.sample_count(),
        min_sample_count_strict: cli_args.aa_strict,
        resource_indexing: true,
        sampler_anisotropy: true,
    };

    let mut app = AppRunner(Some(App::Initializing(InitializingState {
        instance,
        device_config,
        self_dir: self_dir.to_owned(),
        shader_debug_info: cli_args.shader_debug_info,
        requested_sample_count: cli_args.aa.sample_count(),
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
const MAX_BINDLESS_TEXTURES: usize = 256;
const DEFAULT_ALBEDO_SLOT: u32 = 0;
const DEFAULT_NORMAL_SLOT: u32 = 1;
const DEFAULT_ORM_SLOT: u32 = 2;
/// World-space X offsets so the three models stand side-by-side.
/// Per-model uniform scale. The three Khronos reference models have
/// no shared scale convention; these factors compensate. Will be
/// removed once custom assets with a consistent scale replace them.
const MODEL_SCALES: [f32; 3] = [1.0, 1.0, 2.0];
/// World-space X offsets placing the three models side-by-side.
/// Temporary test-scene scaffolding; see MODEL_SCALES.
const MODEL_OFFSETS: [[f32; 3]; 3] =
    [[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];

/// Depth format candidates tried in preference order.
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
    command_buffer: ResettableCommandBuffer<Graphics>,
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
    // `None` means the window/surface is currently zero-sized. We stay in
    // Running and recreate on the next non-zero resize/scale event.
    swapchain: Option<Arc<Swapchain<WinitWindow>>>,
    /// One render-finished semaphore per swapchain image, indexed by the
    /// image index from `vkAcquireNextImageKHR`. `None` iff `swapchain`
    /// is `None`.
    render_finished_semaphores: Option<Arc<Vec<Semaphore>>>,
    /// One depth image per frame-in-flight, matched to swapchain extent.
    /// Empty when `swapchain` is `None`.
    depth_images: Vec<DepthImage>,
    /// One MSAA colour image per frame-in-flight.
    /// Empty when `sample_count == TYPE_1` or `swapchain` is `None`.
    msaa_images: Vec<MsaaImage>,
    sample_count: vk::SampleCountFlags,
    depth_format: vk::Format,
    shader: ShaderModule,
    pipeline: DynamicPipeline,
    vertex_buffer: DeviceLocalBuffer,
    index_buffer: DeviceLocalBuffer,
    submesh_draws: Vec<SubmeshDraw>,
    pipeline_color_format: vk::Format,
    command_pool: ResettableCommandPool<Graphics>,
    frames: Vec<FrameSync>,
    current_frame: usize,
    debug_counters: DebugCounters,
    per_frame_layout: Arc<DescriptorSetLayout>,
    static_layout: Arc<DescriptorSetLayout>,
    /// Shared pipeline layout referencing both descriptor set layouts.
    /// Reused when rebuilding the pipeline after a format change.
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_pool: DescriptorPool,
    /// One host-visible UBO buffer per frame in flight.
    ubo_buffers: Vec<HostVisibleBuffer>,
    /// One camera descriptor set per frame in flight, each pointing
    /// at the matching entry in `ubo_buffers`.
    camera_descriptor_sets: Vec<DescriptorSet>,
    /// Single material descriptor set — texture never changes.
    material_descriptor_set: DescriptorSet,
    /// Wall-clock time at which the app entered the Running state;
    /// used to drive the rotation animation.
    start_time: Instant,
    asset_map: AssetMap,
    textures: Vec<Texture>,
    samplers: Vec<Sampler>,
    /// Host-visible SSBO holding one MaterialGpu per submesh draw.
    material_ssbo: HostVisibleBuffer,
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
            .field("camera_set_layout", &self.per_frame_layout)
            .field("material_set_layout", &self.static_layout)
            .field("descriptor_pool", &self.descriptor_pool)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct SuspendedState {
    win: Arc<WinitWindow>,
    device: Arc<Device>,
    shader: ShaderModule,
    pipeline: DynamicPipeline,
    vertex_buffer: DeviceLocalBuffer,
    index_buffer: DeviceLocalBuffer,
    submesh_draws: Vec<SubmeshDraw>,
    pipeline_color_format: vk::Format,
    sample_count: vk::SampleCountFlags,
    command_pool: ResettableCommandPool<Graphics>,
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
    textures: Vec<Texture>,
    samplers: Vec<Sampler>,
    material_ssbo: HostVisibleBuffer,
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
                depth_images: _,
                msaa_images: _,
                sample_count,
                depth_format: _,
                shader,
                pipeline,
                vertex_buffer,
                index_buffer,
                submesh_draws,
                pipeline_color_format,
                command_pool,
                frames,
                current_frame: _,
                debug_counters,
                per_frame_layout: camera_set_layout,
                static_layout: material_set_layout,
                pipeline_layout,
                descriptor_pool,
                ubo_buffers,
                camera_descriptor_sets,
                material_descriptor_set,
                start_time,
                asset_map,
                textures,
                samplers,
                material_ssbo,
            } = running_state;

            if let Err(e) = device.wait_idle() {
                tracing::error!(
                    "Error while waiting for device idle during suspend: {}",
                    e
                );
                self.transition_to_exiting("Running", event_loop);
                return;
            }

            self.set_suspended(SuspendedState {
                win,
                device,
                shader,
                pipeline,
                vertex_buffer,
                index_buffer,
                submesh_draws,
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
                textures,
                samplers,
                material_ssbo,
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
    /// Create one binary semaphore per swapchain image for render→present
    /// synchronisation. Semaphores are named `"render finished img[N]"`.
    fn make_render_finished_semaphores(
        device: &Arc<Device>,
        image_count: usize,
    ) -> eyre::Result<Arc<Vec<Semaphore>>> {
        let sems = (0..image_count)
            .map(|i| {
                Semaphore::new(
                    device,
                    Some(&format!("render finished img[{i}]")),
                )
                .map_err(eyre::Report::from)
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        Ok(Arc::new(sems))
    }

    /// Create one depth image per frame-in-flight for `extent`.
    fn make_depth_images(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
        sample_count: vk::SampleCountFlags,
    ) -> eyre::Result<Vec<DepthImage>> {
        (0..MAX_FRAMES_IN_FLIGHT)
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

    /// Create one MSAA colour image per frame-in-flight for `extent`.
    /// Returns an empty Vec when `sample_count == TYPE_1`.
    fn make_msaa_images(
        device: &Arc<Device>,
        extent: vk::Extent2D,
        format: vk::Format,
        sample_count: vk::SampleCountFlags,
    ) -> eyre::Result<Vec<MsaaImage>> {
        if sample_count == vk::SampleCountFlags::TYPE_1 {
            return Ok(Vec::new());
        }
        (0..MAX_FRAMES_IN_FLIGHT)
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

    /// Pick the highest supported sample count ≤ `requested`.
    ///
    /// Falls back through TYPE_8 → TYPE_4 → TYPE_2 → TYPE_1.
    fn resolve_sample_count(
        device: &Device,
        requested: vk::SampleCountFlags,
    ) -> vk::SampleCountFlags {
        let props = device.properties();
        let supported = props.limits.framebuffer_color_sample_counts
            & props.limits.framebuffer_depth_sample_counts;
        for &count in &[
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ] {
            if count.as_raw() <= requested.as_raw() && supported.contains(count)
            {
                return count;
            }
        }
        vk::SampleCountFlags::TYPE_1
    }

    fn draw_frame(state: &mut RunningState) -> DrawFrameOutcome {
        // Skip if the window is currently zero-sized (no swapchain).
        if state.swapchain.is_none() {
            return DrawFrameOutcome::Success;
        }

        let frame_idx = state.current_frame;
        let frame_objs = &mut state.frames[frame_idx];

        if frame_objs.in_flight_fence.is_submitted() {
            // Wait until this frame slot's previous GPU work is done, then
            // reset the fence for reuse. This also guarantees image_available
            // is unsignaled and the command buffer is no longer in use.
            //
            // SAFETY: fence was submitted with GPU work for this frame slot;
            // the wait ensures all that work has completed.
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
        let image_available = frame_objs.image_available.raw();

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
        let render_finished = state
            .render_finished_semaphores
            .as_ref()
            .expect("render_finished_semaphores present with swapchain")[i]
            .raw();

        let elapsed = state.start_time.elapsed().as_secs_f32();
        let aspect = extent.width as f32 / extent.height as f32;

        let view = look_at_rh(
            Vec3::new(0.0, 10.0, 5.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::unit_z(),
        );
        let proj = perspective_rh_zo_infinite_rev(PI32 / 3.0, aspect, 0.1);
        let ubo = Ubo {
            view_proj: proj * view,
        };
        if let Err(e) =
            state.ubo_buffers[frame_idx].write_pod(std::slice::from_ref(&ubo))
        {
            return DrawFrameOutcome::Fatal(format!("UBO write failed: {e}"));
        }

        const TAU32: f32 = PI32 * 2.0;

        let pipeline_handle = state.pipeline.raw_pipeline();
        let frame_cmd = &mut frame_objs.command_buffer;

        // Reset and re-record the command buffer for this frame slot.
        //
        // SAFETY: fence wait above guarantees the buffer is not
        // pending on the GPU.
        if let Err(e) = unsafe { frame_cmd.reset() } {
            return DrawFrameOutcome::Fatal(format!(
                "Command buffer reset failed: {e}"
            ));
        }
        let mut rec = frame_cmd.begin_recording();

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        // Transition: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        let to_color = image_barrier2()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(subresource_range);
        let depth_image = state.depth_images[frame_idx].raw_image();
        let depth_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        // Transition: UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
        // old_layout = UNDEFINED discards previous contents, which is
        // safe because LOAD_OP_CLEAR overwrites them anyway.
        let to_depth = image_barrier2()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .image(depth_image)
            .subresource_range(depth_subresource_range);
        let mut barriers = vec![to_color, to_depth];
        if state.sample_count != vk::SampleCountFlags::TYPE_1 {
            let msaa_raw = state.msaa_images[frame_idx].raw_image();
            barriers.push(
                image_barrier2()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    )
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(msaa_raw)
                    .subresource_range(subresource_range),
            );
        }
        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(&barriers);
        // SAFETY: recording state; swapchain image, depth image, and
        // MSAA image (when present) are valid.
        unsafe { rec.pipeline_barrier2(&dep_info) };

        // SAFETY: We remember to close this, we're recording
        unsafe {
            rec.begin_debug_label(DebugLabel {
                name: "Main Draw Pass",
                ty: DebugLabelType::GraphicsPass,
            })
        };

        // Begin dynamic rendering with a clear.
        let color_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let depth_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        };
        let color_attachment = if state.sample_count
            != vk::SampleCountFlags::TYPE_1
        {
            let msaa_view = state.msaa_images[frame_idx].raw_image_view();
            // MSAA: render into the MSAA image, resolve to swapchain.
            vk::RenderingAttachmentInfo::default()
                .image_view(msaa_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(color_clear)
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(image_view)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        } else {
            vk::RenderingAttachmentInfo::default()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(color_clear)
        };
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(state.depth_images[frame_idx].raw_image_view())
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(depth_clear);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);
        // SAFETY: recording; color image in COLOR_ATTACHMENT_OPTIMAL;
        // depth image in DEPTH_STENCIL_ATTACHMENT_OPTIMAL; views valid.
        unsafe { rec.begin_rendering(&rendering_info) };

        // Bind pipeline, set dynamic viewport/scissor, draw.
        // SAFETY: inside a dynamic render pass with a compatible
        // color attachment.
        unsafe { rec.bind_graphics_pipeline(pipeline_handle) };
        // SAFETY: recording state; pipeline_layout is compatible with the
        // bound pipeline; descriptor set is valid and its buffer remains
        // alive for the duration of this frame's GPU work.
        unsafe {
            rec.bind_descriptor_sets(
                &state.pipeline_layout,
                0,
                &[
                    &state.camera_descriptor_sets[frame_idx],
                    &state.material_descriptor_set,
                ],
            )
        };
        // SAFETY: inside render pass recording; buffer is valid
        unsafe { rec.bind_vertex_buffer(0, &state.vertex_buffer, 0) };

        // Standard Vulkan viewport; Y is already corrected in the
        // projection matrix (col 1 = -f).
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
        unsafe { rec.set_viewport(std::slice::from_ref(&viewport)) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        // SAFETY: pipeline declares VK_DYNAMIC_STATE_SCISSOR.
        unsafe { rec.set_scissor(std::slice::from_ref(&scissor)) };

        // SAFETY: inside render pass recording; buffer is valid.
        unsafe {
            rec.bind_index_buffer(&state.index_buffer, 0, vk::IndexType::UINT32)
        };

        // Per-submesh draw loop: push model+material_idx, then draw.
        // SAFETY: all required dynamic state set; render pass active;
        // vertex and index buffers bound; pipeline_layout compatible.
        let rotation = Mat4::<f32>::rotation_z(elapsed / 5.0 * TAU32);
        for draw in &state.submesh_draws {
            let offset = MODEL_OFFSETS[draw.model_idx];
            let scale = Mat4::<f32>::scaling_3d(Vec3::broadcast(
                MODEL_SCALES[draw.model_idx],
            ));
            let placement = Mat4::<f32>::translation_3d(Vec3::new(
                offset[0], offset[1], offset[2],
            )) * rotation
                * scale;
            let push = PushConstants {
                model: placement * draw.sub_trs,
                material_idx: draw.material_idx,
            };
            // SAFETY: layout compatible with bound pipeline; VERTEX|FRAGMENT
            // stage and offset 0 match the declared range; push fits in the
            // minimum 128-byte guarantee.
            unsafe {
                rec.push_constants(
                    &state.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_ref(&push),
                )
            };
            // SAFETY: dynamic state set; render pass active; vertex and
            // index buffers bound; first_index + vertex_offset valid for
            // this submesh within the combined buffer.
            unsafe {
                rec.draw_indexed(
                    draw.index_count,
                    1,
                    draw.first_index,
                    draw.vertex_offset,
                    0,
                )
            };
        }

        // SAFETY: inside a dynamic render pass.
        unsafe { rec.end_rendering() };

        // SAFETY: currently in a debug label
        unsafe { rec.end_debug_label() };

        // Transition: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
        // Transfer: Graphics -> Present
        let to_present = image_barrier2()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .src_queue_family_index(state.device.graphics_queue_family())
            .dst_queue_family_index(state.device.present_queue_family())
            .subresource_range(subresource_range);
        let dep_info_present = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&to_present));
        // SAFETY: recording; image is in COLOR_ATTACHMENT_OPTIMAL.
        unsafe { rec.pipeline_barrier2(&dep_info_present) };

        rec.end_recording();

        // Submit — wait on image_available, signal render_finished,
        // signal fence when done.
        let wait_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        let signal_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(render_finished)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        // SAFETY: image_available is signaled by acquire; render_finished is
        // unsignaled; fence is unsignaled (just reset above); cmd is in the
        // executable state.
        if let Err(e) = unsafe {
            state.device.graphics_queue_submit2_one_labeled(
                frame_cmd,
                std::slice::from_ref(&wait_info),
                std::slice::from_ref(&signal_info),
                Some(&mut frame_objs.in_flight_fence),
                Some("Frame render"),
            )
        } {
            return DrawFrameOutcome::Fatal(format!(
                "Queue submit failed: {e}"
            ));
        }

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
            Ok(_)
            | Err(rgpu_vk::device::QueuePresentError::Vulkan(
                vk::Result::ERROR_OUT_OF_DATE_KHR,
            )) => DrawFrameOutcome::SwapchainOutOfDate,
            Err(e) => DrawFrameOutcome::Fatal(format!("Present failed: {e}")),
        }
    }
}

fn build_pipeline<F>(
    device: &Arc<Device>,
    shader: &ShaderModule,
    color_format: vk::Format,
    depth_format: vk::Format,
    sample_count: vk::SampleCountFlags,
    layout: Option<Arc<PipelineLayout>>,
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
    Ok(DynamicPipeline::new(
        device,
        &DynamicPipelineDesc {
            stages: &[vert, frag],
            color_attachment_formats: &[color_format],
            depth_attachment_format: Some(depth_format),
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            vertex_bindings: &vertex_bindings,
            vertex_attributes: &vertex_attributes,
            layout,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::COUNTER_CLOCKWISE,
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
                    .with_title("phoenix")
                    .with_visible(false)
                    .with_inner_size(LogicalSize {
                        width: 1600,
                        height: 900,
                    }),
            )?,
        );

        // SAFETY: Surface must be destroyed only after all swapchains
        // derived from it are destroyed and no GPU work accesses their
        // images. Swapchain holds Arc<Surface>, so the surface outlives
        // the swapchain. Swapchain replacement always calls wait_idle
        // first, and RunningState drop calls wait_idle via
        // _idle_guard — so the surface is never freed while GPU work
        // is pending.
        let surface = Arc::new(unsafe {
            Surface::new(&state.instance, Arc::clone(&win))
        }?);

        let device = Arc::new(Device::create_compatible(
            &state.instance,
            &surface,
            state.device_config,
        )?);

        let queue_config = device.queue_config();
        let dedicated_transfer = queue_config.dedicated_transfer;
        let transfer_queue_family = device.transfer_queue_family();
        let graphics_queue_family = device.graphics_queue_family();

        let sample_count =
            Self::resolve_sample_count(&device, state.requested_sample_count);
        if sample_count != state.requested_sample_count {
            tracing::warn!(
                "Requested sample count {:?} not supported; \
                 using {:?}",
                state.requested_sample_count,
                sample_count,
            );
        }

        let asset_map_path =
            state.self_dir.join("assets").join("asset_map.toml");
        let asset_map =
            AssetMap::load(&asset_map_path).map_err(|e| eyre::eyre!("{e}"))?;

        let assets_dir = state.self_dir.join("assets");

        // Load all three meshes into a combined vertex/index buffer.
        // Textures are deduped by TextureId (u64 hash). The first
        // DEFAULT_TEX_COUNT slots are reserved for the hardcoded
        // defaults; real textures follow.
        struct TexLoadInfo {
            width: u32,
            height: u32,
            format: vk::Format,
            mip_count: u32,
            name: String,
            mip_data: Vec<(Vec<u8>, usize)>,
            total_bytes: usize,
        }

        let mesh_names: [&str; 3] = ["duck", "damaged-helmet", "flight-helmet"];
        let mut scene_vertices: Vec<Vertex> = Vec::new();
        let mut scene_indices: Vec<u32> = Vec::new();
        let mut submesh_draws: Vec<SubmeshDraw> = Vec::new();
        let mut material_gpus: Vec<MaterialGpu> = Vec::new();
        // Maps TextureId hash → slot index in the bindless array.
        let mut tex_id_to_slot: HashMap<u64, u32> = HashMap::new();
        let mut tex_load_infos: Vec<TexLoadInfo> = Vec::new();

        // Default textures occupy the first DEFAULT_TEX_COUNT slots.
        // Slot 0: magenta albedo (R8G8B8A8_SRGB)
        tex_load_infos.push(TexLoadInfo {
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_SRGB,
            mip_count: 1,
            name: "default-albedo".to_owned(),
            mip_data: vec![(vec![255u8, 0, 255, 255], 0)],
            total_bytes: 4,
        });
        // Slot 1: flat normal (R8G8_UNORM, (127, 127) ≈ (0, 0, 1))
        tex_load_infos.push(TexLoadInfo {
            width: 1,
            height: 1,
            format: vk::Format::R8G8_UNORM,
            mip_count: 1,
            name: "default-normal".to_owned(),
            mip_data: vec![(vec![127u8, 127], 0)],
            total_bytes: 2,
        });
        // Slot 2: default ORM (R8G8B8A8_UNORM, O=1.0, R=0.5, M=0.0)
        // R8G8B8_UNORM is not reliably supported; use RGBA and ignore A.
        tex_load_infos.push(TexLoadInfo {
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            mip_count: 1,
            name: "default-orm".to_owned(),
            mip_data: vec![(vec![255u8, 128, 0, 255], 0)],
            total_bytes: 4,
        });

        // Resolve a TextureId to a slot, loading the texture if needed.
        // Returns the default slot for the given role on failure/absence.
        let resolve_tex = |id: TextureId,
                           role_default: u32,
                           tex_id_to_slot: &mut HashMap<u64, u32>,
                           tex_load_infos: &mut Vec<TexLoadInfo>,
                           asset_map: &AssetMap,
                           assets_dir: &std::path::Path|
         -> eyre::Result<u32> {
            if id.0 == 0 {
                return Ok(role_default);
            }
            if let Some(&s) = tex_id_to_slot.get(&id.0) {
                return Ok(s);
            }
            if tex_load_infos.len() >= MAX_BINDLESS_TEXTURES {
                eyre::bail!(
                    "too many unique textures (max {MAX_BINDLESS_TEXTURES})"
                );
            }
            let tex_name =
                asset_map.name_of(id).unwrap_or("unknown").to_owned();
            let tex_file = asset_map.get(id).ok_or_else(|| {
                eyre::eyre!("texture '{tex_name}' not in asset map")
            })?;
            let tex_asset = TexAsset::open(&assets_dir.join(tex_file))
                .map_err(|e| eyre::eyre!("load {tex_name}: {e}"))?;
            let fmt = tex_vk_format(
                tex_asset.info.format,
                tex_asset.info.color_space,
            );
            let w = tex_asset.info.width;
            let h = tex_asset.info.height;
            let mc = tex_asset.info.mip_count;
            let mut mip_data: Vec<(Vec<u8>, usize)> = Vec::new();
            let mut off: usize = 0;
            for mip_res in tex_asset.mips() {
                let mip_iter =
                    mip_res.map_err(|e| eyre::eyre!("{tex_name} mip: {e}"))?;
                let data: Vec<u8> = mip_iter.map(|[b]| b).collect();
                mip_data.push((data, off));
                off += mip_data.last().unwrap().0.len();
            }
            let slot = tex_load_infos.len() as u32;
            tex_id_to_slot.insert(id.0, slot);
            tex_load_infos.push(TexLoadInfo {
                width: w,
                height: h,
                format: fmt,
                mip_count: mc,
                name: tex_name,
                mip_data,
                total_bytes: off,
            });
            Ok(slot)
        };

        for (model_idx, &mesh_name) in mesh_names.iter().enumerate() {
            let mesh_filename = asset_map
                .get(mesh_id(mesh_name))
                .ok_or_else(|| eyre::eyre!("asset '{mesh_name}' not in map"))?;
            let mesh = MeshAsset::open(&assets_dir.join(mesh_filename))
                .map_err(|e| eyre::eyre!("load {mesh_name} mesh: {e}"))?;

            let vertex_offset = scene_vertices.len() as i32;
            let index_offset = scene_indices.len() as u32;

            let positions = mesh
                .positions()
                .map_err(|e| eyre::eyre!("{mesh_name} positions: {e}"))?;
            let tex_coords = mesh
                .tex_coords0()
                .map_err(|e| eyre::eyre!("{mesh_name} tex_coords0: {e}"))?;
            for (pos, tc) in positions.zip(tex_coords) {
                scene_vertices.push(Vertex {
                    position: pos.into(),
                    tex_coord: tc.into(),
                });
            }

            let indices_iter = mesh
                .indices()
                .map_err(|e| eyre::eyre!("{mesh_name} indices: {e}"))?;
            scene_indices.extend(indices_iter);

            // Resolve global (per-mesh) texture roles once.
            let find_global = |role: TexRole| -> TextureId {
                mesh.tex_refs
                    .iter()
                    .find(|(r, _)| *r == role)
                    .map(|(_, id)| *id)
                    .unwrap_or(TextureId::from_hash(0))
            };
            let normal_id = find_global(TexRole::Normal);
            let orm_id = find_global(TexRole::MetallicRoughness);
            let emissive_id = find_global(TexRole::Emissive);

            let normal_slot = resolve_tex(
                normal_id,
                DEFAULT_NORMAL_SLOT,
                &mut tex_id_to_slot,
                &mut tex_load_infos,
                &asset_map,
                &assets_dir,
            )?;
            let orm_slot = resolve_tex(
                orm_id,
                DEFAULT_ORM_SLOT,
                &mut tex_id_to_slot,
                &mut tex_load_infos,
                &asset_map,
                &assets_dir,
            )?;
            let emissive_slot: i32 = if emissive_id.0 != 0 {
                resolve_tex(
                    emissive_id,
                    DEFAULT_ALBEDO_SLOT,
                    &mut tex_id_to_slot,
                    &mut tex_load_infos,
                    &asset_map,
                    &assets_dir,
                )? as i32
            } else {
                -1
            };

            for (sub_idx, sub) in mesh.sub_meshes.iter().enumerate() {
                let albedo_id = mesh
                    .sub_mesh_albedos
                    .get(sub_idx)
                    .copied()
                    .unwrap_or(TextureId::from_hash(0));
                let albedo_slot = resolve_tex(
                    albedo_id,
                    DEFAULT_ALBEDO_SLOT,
                    &mut tex_id_to_slot,
                    &mut tex_load_infos,
                    &asset_map,
                    &assets_dir,
                )?;
                let material_idx = material_gpus.len() as u32;
                material_gpus.push(MaterialGpu {
                    albedo_idx: albedo_slot,
                    normal_idx: normal_slot,
                    orm_idx: orm_slot,
                    emissive_idx: emissive_slot,
                });
                let sub_trs =
                    trs_to_mat4(sub.translation, sub.rotation, sub.scale);
                submesh_draws.push(SubmeshDraw {
                    first_index: sub.index_base + index_offset,
                    index_count: sub.index_count,
                    vertex_offset,
                    material_idx,
                    model_idx,
                    sub_trs,
                });
            }
        }

        let max_mip_count = tex_load_infos
            .iter()
            .map(|t| t.mip_count)
            .max()
            .unwrap_or(1);

        let vertex_buffer_size =
            (scene_vertices.len() * size_of::<Vertex>()) as vk::DeviceSize;
        let mut staging_vertex_buffer = HostVisibleBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("vertex staging buffer"),
        )?;
        staging_vertex_buffer
            .write_pod_iter_exact(scene_vertices.into_iter())?;

        let mut vertex_buffer = DeviceLocalBuffer::new(
            &device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            Some("vertex buffer"),
        )?;

        let index_buffer_size =
            (scene_indices.len() * size_of::<u32>()) as vk::DeviceSize;
        let mut staging_index_buffer = HostVisibleBuffer::new(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some("index staging buffer"),
        )?;
        staging_index_buffer.write_pod_iter_exact(scene_indices.into_iter())?;

        let mut index_buffer = DeviceLocalBuffer::new(
            &device,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            Some("index buffer"),
        )?;

        let mut upload_command_pool = TransientCommandPool::<Transfer>::new(
            &device,
            Some("upload command pool"),
        )?;
        // SAFETY: this is the first and only allocation from this pool.
        let mut upload_cmd =
            unsafe { upload_command_pool.allocate_command_buffer()? };
        let mut upload_rec = upload_cmd.begin_recording();

        let tex_capacity = tex_load_infos.len();
        let mut used_staging_buffers = Vec::with_capacity(tex_capacity + 2);

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

        let shader_path = assets_dir.join(
            asset_map
                .get_shader(
                    shader_id("phoenix-shader"),
                    state.shader_debug_info,
                )
                .ok_or_else(|| {
                    eyre::eyre!("asset 'phoenix-shader' not in map")
                })?,
        );
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
            .unwrap_or(vk::Format::B8G8R8A8_SRGB);

        // Set 0: camera — UBO with the combined view-projection matrix.
        let per_frame_layout = Arc::new(DescriptorSetLayout::new(
            &device,
            &[
                DescriptorBindingDesc {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    binding_flags: vk::DescriptorBindingFlags::empty(),
                },
                DescriptorBindingDesc {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    count: 15,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    binding_flags: Default::default(),
                },
            ],
            Some("per frame layout"),
        )?);
        // Set 1: material — bindless texture array (binding 0,
        // PARTIALLY_BOUND so unused slots need not be written) plus
        // the material SSBO (binding 1).
        let static_layout = Arc::new(DescriptorSetLayout::new(
            &device,
            &[
                DescriptorBindingDesc {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    count: MAX_BINDLESS_TEXTURES as u32,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                },
                DescriptorBindingDesc {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    binding_flags: vk::DescriptorBindingFlags::empty(),
                },
            ],
            Some("static layout"),
        )?);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            &device,
            &PipelineLayoutDesc {
                set_layouts: &[&per_frame_layout, &static_layout],
                // One push constant range covering the full
                // PushConstants struct (model: Mat4 = 64 bytes),
                // accessible from the vertex stage only.
                push_constant_ranges: &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: size_of::<PushConstants>() as u32,
                }],
            },
        )?);

        let depth_format = device
            .find_depth_format(DEPTH_FORMAT_CANDIDATES)
            .ok_or_else(|| eyre::eyre!("No supported depth format found"))?;

        let pipeline = {
            let _span = tracing::trace_span!(
                "pipeline_create",
                color_format = ?pipeline_color_format,
            )
            .entered();
            build_pipeline(
                &device,
                &shader,
                pipeline_color_format,
                depth_format,
                sample_count,
                Some(Arc::clone(&pipeline_layout)),
                Some({
                    let idx = debug_counters.next_pipeline();
                    move || format!("main pipeline {idx}")
                }),
            )?
        };

        let mut command_pool =
            ResettableCommandPool::new(&device, Some("graphics command pool"))?;

        // Start each fence unsignaled; is_submitted() will be false on
        // the first frame so we skip the wait and go straight to
        // recording. Each frame slot owns its command buffer so the CPU
        // can encode frame N while the GPU executes frame N-1.
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

        let render_finished_semaphores = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_render_finished_semaphores(
                    &device,
                    sc.images().len(),
                )
            })
            .transpose()?;

        let depth_images = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_depth_images(
                    &device,
                    sc.extent(),
                    depth_format,
                    sample_count,
                )
            })
            .transpose()?
            .unwrap_or_default();

        let msaa_images = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_msaa_images(
                    &device,
                    sc.extent(),
                    pipeline_color_format,
                    sample_count,
                )
            })
            .transpose()?
            .unwrap_or_default();

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
        // The material set holds the bindless texture array (binding 0)
        // and the material SSBO (binding 1).
        let descriptor_pool = DescriptorPool::new(
            &device,
            (MAX_FRAMES_IN_FLIGHT + 1) as u32,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: 15 * MAX_FRAMES_IN_FLIGHT as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: MAX_BINDLESS_TEXTURES as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
            ],
            true,
            Some("descriptor pool"),
        )?;

        let camera_layouts: Vec<&DescriptorSetLayout> = (0
            ..MAX_FRAMES_IN_FLIGHT)
            .map(|_| per_frame_layout.as_ref())
            .collect();
        // SAFETY: descriptor_pool outlives camera_descriptor_sets and
        // material_descriptor_set (all stored in the same state struct,
        // dropped in declaration order — sets before pool).
        let camera_descriptor_sets =
            unsafe { descriptor_pool.allocate_sets(&camera_layouts) }?;
        for (i, (set, _buf)) in camera_descriptor_sets
            .iter()
            .zip(ubo_buffers.iter())
            .enumerate()
        {
            set.set_name(&device, Some(&format!("camera set {i}")));
        }
        // SAFETY: same guarantee as camera_descriptor_sets above.
        let mut material_sets = unsafe {
            descriptor_pool.allocate_sets(&[static_layout.as_ref()])
        }?;
        let material_descriptor_set = material_sets
            .pop()
            .expect("allocated exactly one material set");
        material_descriptor_set.set_name(&device, Some("material set"));

        // SAFETY: We are recording
        unsafe {
            upload_rec.begin_debug_label(DebugLabel {
                name: "initial buffer data upload",
                ty: command::DebugLabelType::BufferUpload,
            })
        };

        // Create one staging buffer + Texture per unique albedo, then
        // record all copies into the shared upload command buffer.
        //
        // SAFETY: upload_cmd is in the recording state (begun above);
        // all staging buffers remain alive until the fence wait below.
        unsafe {
            vertex_buffer
                .record_upload_from(&mut upload_rec, &staging_vertex_buffer)
        }?;
        used_staging_buffers.push(staging_vertex_buffer);
        // SAFETY: same as vertex buffer copy above.
        unsafe {
            index_buffer
                .record_upload_from(&mut upload_rec, &staging_index_buffer)
        }?;
        used_staging_buffers.push(staging_index_buffer);

        // SAFETY: We are recording
        unsafe { upload_rec.end_debug_label() };

        struct UploadImageCommand<'a> {
            texture: Texture,
            regions: Vec<BufferImageCopy>,
            name: &'a str,
            buffer: HostVisibleBuffer,
        }
        let mut image_upload_commands = Vec::with_capacity(tex_capacity);

        for tli in &tex_load_infos {
            let name = &tli.name;

            let mut staging = HostVisibleBuffer::new(
                &device,
                tli.total_bytes as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                Some(&format!("{} staging", tli.name)),
            )?;
            for (data, offset) in &tli.mip_data {
                staging.write_pod_at(data.as_slice(), *offset)?;
            }
            let texture = Texture::new(
                &device,
                tli.width,
                tli.height,
                tli.format,
                tli.mip_count,
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                Some(tli.name.as_str()),
            )?;
            // Copy all mip levels from staging into the texture.
            let mut regions = Vec::with_capacity(tli.mip_data.len());
            for (level, (_data, off)) in tli.mip_data.iter().enumerate() {
                let level = level as u32;
                let mip_w = (tli.width >> level).max(1);
                let mip_h = (tli.height >> level).max(1);
                let extent = vk::Extent3D {
                    width: mip_w,
                    height: mip_h,
                    depth: 1,
                };
                regions.push(texture.buffer_image_copy_for_mip(
                    *off as u64,
                    level,
                    extent,
                ));
            }

            image_upload_commands.push(UploadImageCommand {
                texture,
                buffer: staging,
                regions,
                name,
            });
        }

        let texture_pre_upload_barriers: Vec<_> = image_upload_commands
            .iter()
            .map(|c| {
                // Transition: UNDEFINED → TRANSFER_DST_OPTIMAL.
                c.texture
                    .whole_image_barrier2()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            })
            .collect();
        // SAFETY: All barriers are valid
        unsafe {
            upload_rec.pipeline_barrier2_by_barriers(
                &[],
                &[],
                &texture_pre_upload_barriers,
            );
        }
        let textures: Vec<_> = image_upload_commands
            .into_iter()
            .map(|cmd| {
                //SAFETY: upload_cmd is recording; cmd.buffer and cmd.texture are valid and remain alive for the duration of this block.
                unsafe {
                    upload_rec.begin_debug_label_lazy(|| {
                        LazyDebugLabelReturn {
                            name_ref: format!("upload {}", cmd.name),
                            ty: DebugLabelType::ImageUpload,
                        }
                    });
                    upload_rec.copy_buffer_to_image(
                        cmd.buffer.raw_buffer(),
                        cmd.texture.raw_image(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &cmd.regions,
                    );
                    upload_rec.end_debug_label();
                }

                used_staging_buffers.push(cmd.buffer);

                cmd.texture
            })
            .collect();
        let texture_post_upload_barriers: Vec<_> = textures
            .iter()
            .map(|t| {
                // Release / transition to SHADER_READ_ONLY.
                t.whole_image_barrier2()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(if dedicated_transfer {
                        vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                    } else {
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                    })
                    .dst_access_mask(if dedicated_transfer {
                        vk::AccessFlags2::NONE
                    } else {
                        vk::AccessFlags2::SHADER_READ
                    })
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(if dedicated_transfer {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    })
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
            })
            .collect();

        // Safety: Valid barriers
        unsafe {
            upload_rec.pipeline_barrier2_by_barriers(
                &[],
                &[],
                &texture_post_upload_barriers,
            );
        }

        let mut buffer_barriers = Vec::new();
        if queue_config.dedicated_transfer {
            let vertex_buffer_barrier = vertex_buffer
                .whole_buffer_barrier2()
                .src_queue_family_index(device.transfer_queue_family())
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_queue_family_index(device.graphics_queue_family())
                .dst_stage_mask(vk::PipelineStageFlags2::NONE);

            let index_buffer_barrier = index_buffer
                .whole_buffer_barrier2()
                .src_queue_family_index(transfer_queue_family)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_queue_family_index(graphics_queue_family)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE);

            buffer_barriers = vec![vertex_buffer_barrier, index_buffer_barrier];
        }
        if !buffer_barriers.is_empty() {
            // SAFETY: recording state; buffer handles valid for the
            // duration of this command buffer.
            unsafe {
                upload_rec.pipeline_barrier2_by_barriers(
                    &[],
                    &buffer_barriers,
                    &[],
                )
            };
        }

        upload_rec.end_recording();

        let opt_graphics_upload = if queue_config.dedicated_transfer {
            // Create a short command buffer on the graphics queue to acquire
            // ownership of the uploaded resources and run it after the
            // transfer semaphore signals.
            let mut graphics_upload_pool =
                TransientCommandPool::<Graphics>::new(
                    &device,
                    Some("graphics upload pool"),
                )?;
            // SAFETY: this is the first and only allocation from this pool.
            let mut graphics_upload_cmd =
                unsafe { graphics_upload_pool.allocate_command_buffer()? };
            let mut graphics_upload_rec = graphics_upload_cmd.begin_recording();

            let transfer_family = device.transfer_queue_family();
            let graphics_family = device.graphics_queue_family();
            let vb_barrier = vertex_buffer
                .whole_buffer_barrier2()
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .src_queue_family_index(transfer_family)
                .dst_queue_family_index(graphics_family);
            let ib_barrier = index_buffer
                .whole_buffer_barrier2()
                .src_access_mask(vk::AccessFlags2::NONE)
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                .src_queue_family_index(transfer_family)
                .dst_queue_family_index(graphics_family);
            // Acquire all textures on the graphics queue and
            // transition each from TRANSFER_DST to SHADER_READ_ONLY.
            let img_barriers: Vec<_> = textures
                .iter()
                .map(|tex| {
                    tex.whole_image_barrier2()
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        )
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(transfer_family)
                        .dst_queue_family_index(graphics_family)
                })
                .collect();

            let buffer_barriers = [vb_barrier, ib_barrier];
            // SAFETY: recording state and handles valid.
            unsafe {
                graphics_upload_rec.pipeline_barrier2_by_barriers(
                    &[],
                    &buffer_barriers,
                    &img_barriers,
                )
            };

            graphics_upload_rec.end_recording();
            Some((graphics_upload_cmd, graphics_upload_pool))
        } else {
            None
        };

        let mut upload_fence =
            rgpu_vk::sync::Fence::new(&device, false, Some("upload fence"))?;
        // Not always needed but it's cheap
        let transfer_semaphore =
            Semaphore::new(&device, Some("Transfer Semaphore"))?;

        if queue_config.dedicated_transfer {
            //Double submit path

            let transfer_semaphore_signal = vk::SemaphoreSubmitInfo::default()
                .semaphore(transfer_semaphore.raw())
                .stage_mask(vk::PipelineStageFlags2::TRANSFER);
            // SAFETY: upload_cmd is in the executable state; semaphore is
            // unsignaled; no fence on the transfer side.
            unsafe {
                device.transfer_queue_submit2_one_labeled(
                    &upload_cmd,
                    &[],
                    std::slice::from_ref(&transfer_semaphore_signal),
                    None,
                    Some("Upload: transfer"),
                )?
            }
            let (graphics_cb, _graphics_pool) =
                opt_graphics_upload.as_ref().expect(
                    "Somehow used dedicated transfer queue but did not record \
                 graphics queue half of upload",
                );
            let graphics_semaphore_wait = vk::SemaphoreSubmitInfo::default()
                .semaphore(transfer_semaphore.raw())
                .stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT);
            // SAFETY: graphics_cb is in the executable state; semaphore is
            // signaled by the transfer submit above; fence is unsignaled.
            unsafe {
                device.graphics_queue_submit2_one_labeled(
                    graphics_cb,
                    std::slice::from_ref(&graphics_semaphore_wait),
                    &[],
                    Some(&mut upload_fence),
                    Some("Upload: ownership transfer"),
                )?
            }
        } else {
            // SAFETY: upload_cmd is in the executable state; fence is
            // unsignaled; no semaphores needed on the unified queue path.
            unsafe {
                device.transfer_queue_submit2_one_labeled(
                    &upload_cmd,
                    &[],
                    &[],
                    Some(&mut upload_fence),
                    Some("Upload"),
                )?;
            }
        }
        // `wait` is safe and does not require additional invariants beyond
        // a valid fence; call it directly.
        upload_fence.wait(u64::MAX)?;

        // Build the full 15-entry sampler table matching the shader's
        // getSamplerIndex layout: index = filter * OOB_COUNT + addr.
        // Filters 0=NEAREST, 1=LINEAR, 2=ANISOTROPIC.
        // Addresses 0=REPEAT, 1=CLAMP_TO_EDGE, 2=MIRROR,
        //           3=TRANSPARENT_BORDER, 4=CLAMP_TO_BORDER_WHITE.
        const SAMPLER_TABLE_SIZE: usize = 15;
        let max_anisotropy = device.properties().limits.max_sampler_anisotropy;
        // (filter_filter, min_filter, anisotropy, addr_mode, border_color)
        let addr_modes = [
            (
                vk::SamplerAddressMode::REPEAT,
                vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ),
            (
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ),
            (
                vk::SamplerAddressMode::MIRRORED_REPEAT,
                vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ),
            (
                vk::SamplerAddressMode::CLAMP_TO_BORDER,
                vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            ),
            (
                vk::SamplerAddressMode::CLAMP_TO_BORDER,
                vk::BorderColor::FLOAT_OPAQUE_WHITE,
            ),
        ];
        let filter_params: [(vk::Filter, Option<f32>); 3] = [
            (vk::Filter::NEAREST, None),
            (vk::Filter::LINEAR, None),
            (vk::Filter::LINEAR, Some(max_anisotropy)),
        ];
        let mut samplers: Vec<Sampler> = Vec::with_capacity(SAMPLER_TABLE_SIZE);
        for (fi, (filter, aniso)) in filter_params.iter().enumerate() {
            for (ai, (addr, border)) in addr_modes.iter().enumerate() {
                let label = format!("sampler f{fi} a{ai}");
                samplers.push(Sampler::new(
                    &device,
                    *filter,
                    *filter,
                    *addr,
                    max_mip_count,
                    *aniso,
                    *border,
                    Some(label.as_str()),
                )?);
            }
        }

        // Build a single update that covers camera UBOs, bindless
        // textures, the sampler table, and the material SSBO.
        let n_cams = camera_descriptor_sets.len();
        let mut desc_builder =
            rgpu_vk::descriptor::DescriptorUpdateBuilder::with_capacity(
                n_cams * 2 + textures.len() + 1,
                n_cams * 16 + textures.len() + 1,
            );

        // Camera UBOs
        for (set, buf) in camera_descriptor_sets.iter().zip(ubo_buffers.iter())
        {
            let info = buf.descriptor_buffer_info(0, ubo_size);
            desc_builder.push_write_buffer_info(
                set.raw_descriptor_set(),
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                0,
                info,
            );
        }

        // Bindless texture array entries — write as SAMPLED_IMAGE (no sampler).
        for (slot, tex) in textures.iter().enumerate() {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(tex.raw_image_view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            desc_builder.push_write_image_info(
                material_descriptor_set.raw_descriptor_set(),
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                slot as u32,
                image_info,
            );
        }

        // Sampler table — write all 15 entries as one batched write
        // starting at array element 0 (matches getSamplerIndex layout).
        for set in camera_descriptor_sets.iter() {
            let write_idx = desc_builder.push_write_image_info(
                set.raw_descriptor_set(),
                1,
                vk::DescriptorType::SAMPLER,
                0,
                samplers[0].descriptor_image_info(),
            );
            for s in &samplers[1..] {
                desc_builder.push_sampler(write_idx, s);
            }
        }

        // Upload the material SSBO (host-visible; written once at load).
        let material_ssbo_size =
            (material_gpus.len() * size_of::<MaterialGpu>()) as vk::DeviceSize;
        let mut material_ssbo = HostVisibleBuffer::new(
            &device,
            material_ssbo_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            Some("material ssbo"),
        )?;
        material_ssbo.write_pod_iter_exact(material_gpus.into_iter())?;
        let buf_info =
            material_ssbo.descriptor_buffer_info(0, material_ssbo_size);
        desc_builder.push_write_buffer_info(
            material_descriptor_set.raw_descriptor_set(),
            1,
            vk::DescriptorType::STORAGE_BUFFER,
            0,
            buf_info,
        );

        // SAFETY: caller guarantees resources (buffers, images, samplers)
        // remain valid for the duration of this update; `desc_builder`
        // owns the descriptor info arrays while applying.
        unsafe { desc_builder.apply(&device) };

        // SAFETY: guard will call wait_idle in Drop. Must not be forgotten.
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
            depth_images,
            msaa_images,
            sample_count,
            depth_format,
            shader,
            pipeline,
            vertex_buffer,
            index_buffer,
            submesh_draws,
            pipeline_color_format,
            command_pool,
            frames,
            current_frame: 0,
            debug_counters,
            per_frame_layout,
            static_layout,
            pipeline_layout,
            descriptor_pool,
            ubo_buffers,
            camera_descriptor_sets,
            material_descriptor_set,
            start_time: Instant::now(),
            asset_map,
            textures,
            samplers,
            material_ssbo,
        })
    }

    fn suspended_to_running(
        state: SuspendedState,
    ) -> eyre::Result<RunningState> {
        // SAFETY: The surface outlives all derived swapchains via Arc<Surface>.
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

        let depth_format = state
            .device
            .find_depth_format(DEPTH_FORMAT_CANDIDATES)
            .ok_or_else(|| eyre::eyre!("No supported depth format found"))?;

        // Reuse the existing pipeline when the swapchain honored the
        // preferred format. Recreate it only if the surface forced a
        // different format.
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
                        build_pipeline(
                            &state.device,
                            &state.shader,
                            new_format,
                            depth_format,
                            state.sample_count,
                            Some(Arc::clone(&state.pipeline_layout)),
                            Some({
                                let idx = debug_counters.next_pipeline();
                                move || format!("main pipeline {idx}")
                            }),
                        )?
                    };
                    (pipeline, new_format)
                }
            } else {
                (state.pipeline, state.pipeline_color_format)
            };

        let render_finished_semaphores = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_render_finished_semaphores(
                    &state.device,
                    sc.images().len(),
                )
            })
            .transpose()?;

        let depth_images = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_depth_images(
                    &state.device,
                    sc.extent(),
                    depth_format,
                    state.sample_count,
                )
            })
            .transpose()?
            .unwrap_or_default();

        let msaa_images = swapchain
            .as_ref()
            .map(|sc| {
                Self::make_msaa_images(
                    &state.device,
                    sc.extent(),
                    pipeline_color_format,
                    state.sample_count,
                )
            })
            .transpose()?
            .unwrap_or_default();

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
            render_finished_semaphores,
            depth_images,
            msaa_images,
            sample_count: state.sample_count,
            depth_format,
            shader: state.shader,
            pipeline,
            vertex_buffer: state.vertex_buffer,
            index_buffer: state.index_buffer,
            submesh_draws: state.submesh_draws,
            pipeline_color_format,
            command_pool: state.command_pool,
            frames: state.frames,
            current_frame: 0,
            debug_counters,
            per_frame_layout: state.camera_set_layout,
            static_layout: state.material_set_layout,
            pipeline_layout: state.pipeline_layout,
            descriptor_pool: state.descriptor_pool,
            ubo_buffers: state.ubo_buffers,
            camera_descriptor_sets: state.camera_descriptor_sets,
            material_descriptor_set: state.material_descriptor_set,
            start_time: state.start_time,
            asset_map: state.asset_map,
            textures: state.textures,
            samplers: state.samplers,
            material_ssbo: state.material_ssbo,
        })
    }

    fn recreate_swapchain_if_needed(
        running_state: &mut RunningState,
        desired_extent: rgpu_vk::ash::vk::Extent2D,
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
                    "Error while waiting for device idle on zero \
                     extent: {}",
                    e
                );
                return false;
            }
            running_state.swapchain = None;
            running_state.render_finished_semaphores = None;
            running_state.depth_images = Vec::new();
            running_state.msaa_images = Vec::new();
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

        // Wait for all GPU work to complete before dropping the old
        // swapchain. Image views (and framebuffers in noext) must not
        // be destroyed while in-flight commands still reference them.
        if let Err(e) = running_state.device.wait_idle() {
            tracing::error!(
                "Error waiting for device idle before swapchain \
                 recreation: {}",
                e
            );
            return false;
        }

        running_state.swapchain.take();

        let sc_idx = running_state.debug_counters.next_swapchain();
        match Swapchain::new(
            &running_state.device,
            &running_state._surface,
            desired_extent,
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
                let new_extent = swapchain.extent();
                let new_image_count = swapchain.images().len();
                running_state.swapchain = Some(Arc::new(swapchain));

                match Self::make_render_finished_semaphores(
                    &running_state.device,
                    new_image_count,
                ) {
                    Ok(sems) => {
                        running_state.render_finished_semaphores = Some(sems);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Error recreating render-finished \
                             semaphores: {}",
                            e
                        );
                        return false;
                    }
                }

                match Self::make_depth_images(
                    &running_state.device,
                    new_extent,
                    running_state.depth_format,
                    running_state.sample_count,
                ) {
                    Ok(images) => {
                        running_state.depth_images = images;
                    }
                    Err(e) => {
                        tracing::error!("Error recreating depth images: {}", e);
                        return false;
                    }
                }

                match Self::make_msaa_images(
                    &running_state.device,
                    new_extent,
                    new_format,
                    running_state.sample_count,
                ) {
                    Ok(images) => {
                        running_state.msaa_images = images;
                    }
                    Err(e) => {
                        tracing::error!("Error recreating MSAA images: {}", e);
                        return false;
                    }
                }

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
                        running_state.depth_format,
                        running_state.sample_count,
                        Some(Arc::clone(&running_state.pipeline_layout)),
                        Some({
                            let idx =
                                running_state.debug_counters.next_pipeline();
                            move || format!("main pipeline {idx}")
                        }),
                    ) {
                        Ok(pipeline) => {
                            running_state.pipeline = pipeline;
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
    ) -> Option<rgpu_vk::ash::vk::Extent2D> {
        match window_event {
            WindowEvent::Resized(size) => Some(rgpu_vk::ash::vk::Extent2D {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = win.inner_size();
                Some(rgpu_vk::ash::vk::Extent2D {
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
