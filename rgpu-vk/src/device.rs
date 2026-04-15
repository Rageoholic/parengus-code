//! Logical device wrapper ([`Device`]).
//!
//! `Device` wraps a `VkDevice` and centralises all per-device state: a VMA
//! allocator, extension loaders for swapchain, dynamic rendering,
//! synchronization2, and debug utils, plus queues for graphics/present,
//! transfer, and compute roles.
//!
//! # Queue model
//!
//! Queue behaviour is controlled by [`QueueConfig`] in [`DeviceConfig`]. Three
//! independent boolean axes govern allocation:
//!
//! | Field | `true` | `false` |
//! |-------|--------|---------|
//! | `dedicated_transfer` | dedicated family | shares gfx family |
//! | `dedicated_compute` | dedicated family | shares gfx family |
//! | `parallel` | all queues per family | one queue per family |
//!
//! When multiple queues are available for a role, callers pass a `queue_index`
//! to submit methods to select one per frame in flight. Roles that share a
//! family share the same underlying `Arc<Mutex<…>>`.
//!
//! [`Device::create_compatible`] tries to honour each requested axis. When
//! `DeviceConfig::queue_config_strict = true`, any axis that could not be
//! satisfied returns [`CreateCompatibleError::QueueConfigUnsatisfied`];
//! otherwise the device uses the best configuration the hardware provides.
//!
//! # Physical device selection
//!
//! Physical device selection uses a priority-based fold: discrete GPUs outrank
//! integrated GPUs, and only devices that satisfy all required extensions and
//! queue families are considered. [`Device::create_compatible`] wraps this
//! selection and returns the highest-priority match.
//!
//! All raw Vulkan operations on the device handle are surfaced as `unsafe fn`
//! methods prefixed with `raw_` (e.g. `create_raw_buffer`). Higher-level
//! wrappers in sibling modules call these rather than accessing `ash::Device`
//! directly.

use parking_lot::{Mutex, MutexGuard};
use std::{
    collections::{HashMap, HashSet},
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use vk_mem::{
    Alloc, AllocationCreateFlags, AllocationCreateInfo, Allocator,
    AllocatorCreateInfo as VmaAllocatorCreateInfo,
};

use crate::sync::{self, Fence};
use crate::{
    instance::{FetchPhysicalDeviceError, Instance},
    surface::Surface,
};

// ---------------------------------------------------------------------------
// Queue capability markers and traits
// ---------------------------------------------------------------------------

/// Queue capability marker: supports graphics, compute, and transfer.
///
/// Device creation asserts that the selected graphics queue family also reports
/// `VK_QUEUE_COMPUTE_BIT`, so `Graphics ⟹ Compute` is always upheld at runtime.
pub struct Graphics;

/// Queue capability marker: supports compute and transfer.
pub struct Compute;

/// Queue capability marker: supports transfer only.
pub struct Transfer;

/// Implemented by queue capability markers that support graphics commands
/// (`vkCmdDraw*`, render-pass commands, etc.).
///
/// Not sealed — downstream crates may implement this for custom markers.
pub trait SupportsGraphics {}

/// Implemented by queue capability markers that support compute dispatch
/// commands.
///
/// Not sealed — downstream crates may implement this for custom markers.
pub trait SupportsCompute {}

/// Implemented by queue capability markers that support transfer commands
/// (copies and pipeline barriers).
///
/// Not all queue types imply transfer: `VK_QUEUE_VIDEO_DECODE_BIT_KHR`, for
/// example, does not. A `VideoDecode` marker would therefore not implement this
/// trait, making it a compile error to record copy or barrier commands into a
/// video-decode command buffer.
///
/// Not sealed — downstream crates may implement this for custom markers.
pub trait SupportsTransfer {}

impl SupportsGraphics for Graphics {}
impl SupportsCompute for Graphics {}
impl SupportsTransfer for Graphics {}
impl SupportsCompute for Compute {}
impl SupportsTransfer for Compute {}
impl SupportsTransfer for Transfer {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Graphics {}
    impl Sealed for super::Compute {}
    impl Sealed for super::Transfer {}
}

/// Maps a queue capability marker to the queue family index on a [`Device`].
///
/// Sealed — only the three built-in markers implement this trait.
pub trait QueueFamily: sealed::Sealed {
    /// Returns the Vulkan queue family index for this capability on `device`.
    fn family(device: &Device) -> u32;
}

impl QueueFamily for Graphics {
    fn family(device: &Device) -> u32 {
        device.graphics_queue_family()
    }
}

impl QueueFamily for Compute {
    fn family(device: &Device) -> u32 {
        device.compute_queue_family()
    }
}

impl QueueFamily for Transfer {
    fn family(device: &Device) -> u32 {
        device.transfer_queue_family()
    }
}

/// A command buffer that has been recorded and is ready for submission to a
/// queue with capability `Q`.
///
/// This trait is intentionally separate from `Recordable<Q>` to support future
/// TypeState designs where recording and submission states are encoded as
/// distinct type parameters. Future `CommandBuffer<Q, Executable>` types will
/// implement `Submittable<Q>` without needing to implement the recording
/// lifecycle.
///
/// Not sealed — downstream crates may implement this for custom types.
pub trait Submittable<Q> {
    /// The raw `vk::CommandBuffer` handle to pass to `vkQueueSubmit2`.
    fn raw(&self) -> vk::CommandBuffer;
}

enum DynamicRenderingLoader {
    /// Vulkan 1.3+: dynamic rendering is core; dispatch through `ash::Device`.
    Core,
    /// Vulkan < 1.3: loaded via `VK_KHR_dynamic_rendering`.
    Extension(ash::khr::dynamic_rendering::Device),
}

enum Synchronization2Loader {
    /// Vulkan 1.3+: synchronization2 is core; dispatch through `ash::Device`.
    Core,
    /// Vulkan < 1.3: loaded via `VK_KHR_synchronization2`.
    Extension(ash::khr::synchronization2::Device),
}

/// Describes how an allocation will be accessed by CPU and GPU.
///
/// Passed to [`Device::allocate_memory`] to select the best-matching Vulkan
/// memory type and determine whether atom-size padding is required for
/// non-coherent flush alignment.
#[derive(Copy, Clone, Debug)]
pub enum MemoryUsage {
    /// GPU-only storage. Highest bandwidth; not CPU-mappable.
    GpuOnly,
    /// CPU-writable, GPU-readable. For staging buffers and per-frame uploads.
    Upload,
    /// CPU-writable, GPU-readable. For bound buffers the CPU wants to write to
    HostVisibleBind,
    /// GPU-writable, CPU-readable. For readback.
    Download,
    /// For the weird buffers that both the CPU and GPU need to write to.
    /// Probably an incredibly niche case but it's the one missing case ignoring
    /// lazy
    UpDown,
}

impl MemoryUsage {
    /// Determine if the current memory usage type requires
    pub fn is_host_visible(&self) -> bool {
        !matches!(self, Self::GpuOnly)
    }
}

/// A logical Vulkan device and its associated per-device state.
///
/// Wraps an `ash::Device`, a VMA allocator, extension loaders for swapchain /
/// dynamic rendering / synchronization2 / debug utils, and the graphics+present
/// queue.
///
/// Constructed via [`Device::create_compatible`], which selects the best
/// physical device by priority (discrete > integrated). Raw Vulkan operations
/// are exposed as `unsafe fn` methods prefixed with `raw_`.
#[allow(dead_code)]
pub struct Device {
    parent: Arc<Instance>,
    allocator: Option<Allocator>,
    handle: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: vk::PhysicalDeviceProperties,
    swapchain_device: Option<ash::khr::swapchain::Device>,
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
    dynamic_rendering: Option<DynamicRenderingLoader>,
    synchronization2: Option<Synchronization2Loader>,
    physical_device: vk::PhysicalDevice,
    memory_budget: bool,
    /// Queue for the graphics+present role.
    ///
    /// We store exactly one `VkQueue` handle per role. Roles that share a
    /// family share the same `Arc<Mutex<…>>`.
    graphics_queue: Arc<Mutex<vk::Queue>>,
    graphics_queue_family: u32,
    /// Queue for presentation. May be the same family as graphics.
    present_queue: Arc<Mutex<vk::Queue>>,
    present_family: u32,
    transfer_queues: Arc<Mutex<vk::Queue>>,
    transfer_family: u32,
    compute_queues: Arc<Mutex<vk::Queue>>,
    compute_family: u32,
    /// The [`QueueConfig`] that was actually applied (may differ from the
    /// requested config when strict mode is off).
    queue_config: QueueConfig,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("handle", &self.handle.handle())
            .finish_non_exhaustive()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        tracing::debug!("Dropping device {:?}", self.handle.handle());
        // Ensure allocator is dropped before vkDestroyDevice.
        self.allocator = None;
        //SAFETY: All objects derived from this device should be dropped before
        //this device is dropped.
        unsafe { self.handle.destroy_device(None) };
    }
}

#[derive(Debug, Error)]
pub enum CreateCompatibleError {
    #[error(
        "Mismatched parameters to Device::create_compatible. All \
         parameters must be derived from the same instance"
    )]
    MismatchedParams,

    #[error("Host memory exhaustion while creating a compatible device")]
    MemoryExhaustion,

    #[error("Unknown Vulkan error while creating a compatible device: {0}")]
    UnknownVulkan(vk::Result),

    #[error("No suitable physical device found")]
    NoSuitableDevice,

    #[error("No queue family supporting both graphics and present")]
    NoGraphicsPresentQueue,

    #[error(
        "Requested queue config ({requested}) could not be fully \
         satisfied (achieved: {actual}) and strict mode is enabled"
    )]
    QueueConfigUnsatisfied {
        requested: QueueConfig,
        actual: QueueConfig,
    },

    #[error("Failed to create logical device: {0}")]
    DeviceCreationFailed(vk::Result),

    #[error(
        "Dynamic rendering was requested but VK_KHR_dynamic_rendering is not \
         supported by the selected physical device"
    )]
    DynamicRenderingNotAvailable,

    #[error(
        "VK_KHR_synchronization2 is not supported by the \
         selected physical device (required on Vulkan < 1.3)"
    )]
    Synchronization2NotAvailable,

    #[error(
        "VK_KHR_maintenance1 is not supported by the \
         selected physical device (required on Vulkan < 1.1)"
    )]
    Maintenance1NotAvailable,

    #[error(
        "The following DeviceConfig features require \
         physical_device_features2 on the instance, but it was \
         not enabled in InstanceConfig: {0}"
    )]
    PhysDevFeatures2Required(String),

    #[error("Failed to create GPU allocator: {0}")]
    AllocatorCreation(vk::Result),

    #[error(
        "No suitable device supports the requested sample count \
         and strict mode is enabled"
    )]
    SampleCountUnsupported,
}

#[derive(Debug, Error)]
pub enum NameObjectError {
    #[error("Invalid Vulkan object name (contains interior NUL): {0}")]
    InvalidName(std::ffi::NulError),

    #[error("Vulkan error setting object name: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum QueueSubmitError {
    #[error("Could not submit to the actual Vulkan queue: {0}")]
    SubmissionFailed(vk::Result),
    #[error(
        "Fence was not marked as ready, likely meaning reset was not called"
    )]
    FenceNotReady,
    #[error("The fence passed was from a different device")]
    MismatchedObjects,
    #[error("queue_index {0} is out of bounds for the graphics/present queue")]
    QueueIndexOutOfBounds(usize),
}

#[derive(Debug, Error)]
pub enum QueuePresentError {
    #[error("queue_index {0} is out of bounds for the graphics/present queue")]
    QueueIndexOutOfBounds(usize),
    #[error("Vulkan error during queue present: {0}")]
    Vulkan(vk::Result),
}

/// Controls how Vulkan queues are allocated for the three roles
/// (graphics+present, transfer, compute).
///
/// Each field is an independent boolean axis:
///
/// | Field | `true` | `false` |
/// |-------|--------|---------|
/// | `dedicated_transfer` | dedicated family | shares gfx family |
/// | `dedicated_compute` | dedicated family | shares gfx family |
/// | `parallel` | all queues per family | one queue per family |
///
/// The default (`true` for all) requests the most capable configuration.
/// [`Device::create_compatible`] uses whatever the hardware provides and
/// reports the achieved config via [`Device::queue_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub struct QueueConfig {
    /// Use a dedicated transfer queue family (not shared with graphics). When
    /// `false`, the graphics/present family is used for transfer.
    pub dedicated_transfer: bool,
    /// Use a dedicated compute queue family (not shared with graphics). When
    /// `false`, the graphics/present family is used for compute.
    pub dedicated_compute: bool,
    /// Use a dedicated present queue family (not shared with graphics). When
    /// `false`, the graphics family will be used for present.
    pub dedicated_present: bool,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            dedicated_transfer: true,
            dedicated_compute: true,
            dedicated_present: false,
        }
    }
}

impl std::fmt::Display for QueueConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dedicated_transfer={}, dedicated_compute={}, dedicated_present={}",
            self.dedicated_transfer,
            self.dedicated_compute,
            self.dedicated_present,
        )
    }
}

/// Opaque handle to a VMA memory allocation.
///
/// Created by [`Device::create_raw_buffer_allocated`] or
/// [`Device::create_raw_image_allocated`]. Must be destroyed alongside the
/// buffer or image it was created with.
pub(crate) struct Allocation(vk_mem::Allocation);

impl Allocation {
    /// Returns the mapped pointer for host-visible allocations, or `None` if
    /// the allocation is not persistently mapped.
    pub(crate) fn mapped_ptr(
        &self,
        device: &Device,
    ) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        let allocator = device
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        let info = allocator.get_allocation_info(&self.0);
        std::ptr::NonNull::new(info.mapped_data)
    }
}

/// Error returned when VMA fails to create a buffer or image with memory.
#[derive(Debug, Error)]
#[error("allocation failed: {0}")]
pub struct AllocateMemoryError(pub vk::Result);
#[derive(Debug, Default)]
pub struct DeviceConfig {
    pub swapchain: bool,
    pub dynamic_rendering: bool,
    pub synchronization2: bool,
    pub maintenance1: bool,
    /// When `true`, enable `VK_KHR_shader_non_semantic_info` on pre-1.3 devices
    /// that support it (core in 1.3, no-op on 1.3+). Required when loading
    /// SPIR-V compiled with non-semantic debug info (e.g. `shader.debug.spv`).
    /// Not a hard device filter: if the extension is unavailable the field is
    /// silently ignored.
    pub shader_non_semantic_info: bool,
    pub queue_config: QueueConfig,
    /// When `true`, [`Device::create_compatible`] returns
    /// [`CreateCompatibleError::QueueConfigUnsatisfied`] if any requested axis
    /// in [`queue_config`] could not be satisfied. When `false` (the default),
    /// the device uses the best configuration the hardware supports.
    pub queue_config_strict: bool,
    /// Preferred minimum MSAA sample count for colour and depth framebuffer
    /// attachments. Devices that support this count score higher during
    /// selection. Defaults to `TYPE_1` (no preference).
    pub min_sample_count: vk::SampleCountFlags,
    /// When `true`, [`Device::create_compatible`] returns
    /// [`CreateCompatibleError::SampleCountUnsupported`] if no device supports
    /// `min_sample_count`. When `false` (the default), unsupported devices are
    /// still considered but score lower.
    pub min_sample_count_strict: bool,
    /// Enable resource indexing: large partially-bound descriptor arrays
    /// with non-uniform shader indexing and update-after-bind, for
    /// sampled images, storage buffers, and storage images.
    ///
    /// Requires a subset of `VkPhysicalDeviceDescriptorIndexingFeatures`.
    /// Input-attachment, texel-buffer, and uniform-buffer subfeatures
    /// are intentionally excluded — they are unused in this codebase.
    /// Core in Vulkan 1.2; on older devices requires
    /// `VK_EXT_descriptor_indexing` and its dependency
    /// `VK_KHR_maintenance3`.
    pub resource_indexing: bool,
    /// Enable `VkPhysicalDeviceFeatures::samplerAnisotropy`.
    /// Hard device filter: devices that do not support it are skipped.
    pub sampler_anisotropy: bool,
}

/// Returns `true` if the physical device supports all synchronization2
/// sub-features. Destructures the struct exhaustively (excluding `s_type`,
/// `p_next`, and `_marker`) so the compiler catches any future field additions.
fn sync2_fully_supported(
    f: vk::PhysicalDeviceSynchronization2Features<'_>,
) -> bool {
    let vk::PhysicalDeviceSynchronization2Features {
        s_type: _,
        p_next: _,
        synchronization2,
        _marker: _,
    } = f;
    synchronization2 == vk::TRUE
}

/// Returns `true` if the physical device supports all dynamic rendering
/// sub-features. Destructures the struct exhaustively (excluding `s_type`,
/// `p_next`, and `_marker`) so the compiler catches any future field additions.
fn dynamic_rendering_fully_supported(
    f: vk::PhysicalDeviceDynamicRenderingFeatures<'_>,
) -> bool {
    let vk::PhysicalDeviceDynamicRenderingFeatures {
        s_type: _,
        p_next: _,
        dynamic_rendering,
        _marker: _,
    } = f;
    dynamic_rendering == vk::TRUE
}

/// Returns `true` if the physical device supports the minimum set of
/// descriptor indexing sub-features required for resource indexing.
/// Destructures the struct exhaustively (excluding `s_type`, `p_next`,
/// and `_marker`) so the compiler catches any future field additions.
///
/// Excluded (prefixed `_`, not required):
/// - Input-attachment features: subpass-only, incompatible with dynamic
///   rendering.
/// - Texel-buffer features: unused in this codebase; 1D sampled images
///   or SSBOs cover all our needs.
/// - Uniform-buffer non-uniform / update-after-bind: UBOs are not
///   accessed through large descriptor arrays.
fn resource_indexing_supported(
    f: vk::PhysicalDeviceDescriptorIndexingFeatures<'_>,
) -> bool {
    let vk::PhysicalDeviceDescriptorIndexingFeatures {
        s_type: _,
        p_next: _,
        shader_input_attachment_array_dynamic_indexing: _,
        shader_uniform_texel_buffer_array_dynamic_indexing: _,
        shader_storage_texel_buffer_array_dynamic_indexing: _,
        shader_uniform_buffer_array_non_uniform_indexing: _,
        shader_sampled_image_array_non_uniform_indexing,
        shader_storage_buffer_array_non_uniform_indexing,
        shader_storage_image_array_non_uniform_indexing,
        shader_input_attachment_array_non_uniform_indexing: _,
        shader_uniform_texel_buffer_array_non_uniform_indexing: _,
        shader_storage_texel_buffer_array_non_uniform_indexing: _,
        descriptor_binding_uniform_buffer_update_after_bind: _,
        descriptor_binding_sampled_image_update_after_bind,
        descriptor_binding_storage_image_update_after_bind,
        descriptor_binding_storage_buffer_update_after_bind,
        descriptor_binding_uniform_texel_buffer_update_after_bind: _,
        descriptor_binding_storage_texel_buffer_update_after_bind: _,
        descriptor_binding_update_unused_while_pending,
        descriptor_binding_partially_bound,
        descriptor_binding_variable_descriptor_count,
        runtime_descriptor_array,
        _marker: _,
    } = f;
    shader_sampled_image_array_non_uniform_indexing == vk::TRUE
        && shader_storage_buffer_array_non_uniform_indexing == vk::TRUE
        && shader_storage_image_array_non_uniform_indexing == vk::TRUE
        && descriptor_binding_sampled_image_update_after_bind == vk::TRUE
        && descriptor_binding_storage_image_update_after_bind == vk::TRUE
        && descriptor_binding_storage_buffer_update_after_bind == vk::TRUE
        && descriptor_binding_update_unused_while_pending == vk::TRUE
        && descriptor_binding_partially_bound == vk::TRUE
        && descriptor_binding_variable_descriptor_count == vk::TRUE
        && runtime_descriptor_array == vk::TRUE
}

impl Device {
    /// Create a logical device compatible with `surf`.
    ///
    /// Selects the highest-priority physical device that satisfies all
    /// requirements in `config` and can present to `surf`.
    ///
    /// The name `create_compatible` is intentional: the API does not yet expose
    /// physical devices as a first-class concept, so callers cannot select one
    /// themselves. This name signals that the selection is automatic and may
    /// change in a future API revision once physical-device enumeration is
    /// surfaced.
    pub fn create_compatible<T: HasDisplayHandle + HasWindowHandle>(
        instance: &Arc<Instance>,
        surf: &Surface<T>,
        config: DeviceConfig,
    ) -> Result<Self, CreateCompatibleError> {
        if !std::sync::Arc::ptr_eq(surf.parent(), instance) {
            return Err(CreateCompatibleError::MismatchedParams);
        }
        if !instance.phys_dev_features2_enabled() {
            let mut needs_features2: Vec<&'static str> = Vec::new();
            if config.synchronization2 {
                needs_features2.push("synchronization2");
            }
            if config.dynamic_rendering {
                needs_features2.push("dynamic_rendering");
            }
            if config.resource_indexing {
                needs_features2.push("resource_indexing");
            }
            if !needs_features2.is_empty() {
                return Err(CreateCompatibleError::PhysDevFeatures2Required(
                    needs_features2.join(", "),
                ));
            }
        }

        // Evaluate every physical device, filtering out those that lack
        // required extensions or a graphics+present queue, then score the
        // survivors so we can pick the best.
        //
        // Score: (dedicated_queue_count, device_type_priority) compared
        // lexicographically — dedicated queues matter most, then device type
        // breaks ties.
        let physical_devices = instance.fetch_raw_physical_devices()?;
        let device_type_priority = |dt: vk::PhysicalDeviceType| -> u32 {
            match dt {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            }
        };

        struct DeviceCandidate {
            handle: vk::PhysicalDevice,
            props: vk::PhysicalDeviceProperties,
            queue_families: Vec<vk::QueueFamilyProperties>,
            graphics_family: u32,
            present_family: u32,
            score: (u32, u32, u32),
            /// True when sync2 must use the extension loader.
            use_sync2_ext: bool,
            /// True when dynamic rendering must use the extension loader.
            use_dr_ext: bool,
            /// True when VK_KHR_maintenance1 must be enabled (pre-1.1 device).
            use_maintenance1_ext: bool,
            /// True when VK_KHR_shader_non_semantic_info should be enabled
            /// (available on this pre-1.3 device).
            enable_shader_non_semantic: bool,
            /// True when VK_EXT_memory_budget is supported and should be
            /// enabled.
            enable_memory_budget: bool,
            /// True when `VK_EXT_descriptor_indexing` must be enabled
            /// (pre-1.2 device).
            use_resource_indexing_ext: bool,
        }

        //Capacity here is upper bound
        let mut candidates: Vec<DeviceCandidate> =
            Vec::with_capacity(physical_devices.len());
        let mut skipped_for_sample_count = false;

        'dev: for &dev in &physical_devices {
            // SAFETY: dev was derived from instance.
            let props =
                unsafe { instance.get_raw_physical_device_properties(dev) };
            // SAFETY: dev was derived from instance.
            let queue_families = unsafe {
                instance.get_raw_physical_device_queue_family_properties(dev)
            };

            // Use the device's own reported API version so that per-device
            // capability differences are handled correctly rather than relying
            // on the single instance-level version. When the instance was
            // created with vk_1_0_strict, treat every device as pre-1.3 and
            // pre-1.1 so that the extension code paths are always exercised.
            let dev_api =
                crate::version::VkVersion::from_raw(props.api_version);
            let ver = dev_api.version;
            let is_pre_1_3 = instance.strict_1_0()
                || ver.major < 1
                || (ver.major == 1 && ver.minor < 3);
            let is_pre_1_2 = instance.strict_1_0()
                || ver.major < 1
                || (ver.major == 1 && ver.minor < 2);
            let is_pre_1_1 = instance.strict_1_0()
                || ver.major < 1
                || (ver.major == 1 && ver.minor < 1);

            // VK_KHR_swapchain is never promoted to core; always check it when
            // requested. Other extensions are only extensions on pre-1.3
            // devices.
            let needs_ext_check = config.swapchain || is_pre_1_3;
            let device_exts: Vec<vk::ExtensionProperties> = if needs_ext_check {
                // SAFETY: dev was derived from instance.
                match unsafe {
                    instance.enumerate_raw_device_extension_properties(dev)
                } {
                    Ok(exts) => exts,
                    Err(e) => {
                        tracing::debug!(
                            "Skipping {:?}: \
                                 failed to enumerate extensions: {e}",
                            props.device_name_as_c_str().unwrap_or(c"unknown"),
                        );
                        continue 'dev;
                    }
                }
            } else {
                Vec::new()
            };

            let has_ext = |name: &CStr| -> bool {
                device_exts
                    .iter()
                    .any(|e| e.extension_name_as_c_str() == Ok(name))
            };

            // VK_KHR_swapchain is always an extension; filter hard.
            if config.swapchain && !has_ext(ash::khr::swapchain::NAME) {
                tracing::debug!(
                    "Skipping {:?}: missing VK_KHR_swapchain",
                    props.device_name_as_c_str().unwrap_or(c"unknown"),
                );
                continue 'dev;
            }

            // VK_KHR_synchronization2: core in 1.3; required extension on older
            // devices when requested — hard filter.
            let use_sync2_ext = if config.synchronization2 && is_pre_1_3 {
                if has_ext(ash::khr::synchronization2::NAME) {
                    true
                } else {
                    tracing::debug!(
                        "Skipping {:?}: missing VK_KHR_synchronization2",
                        props.device_name_as_c_str().unwrap_or(c"unknown"),
                    );
                    continue 'dev;
                }
            } else {
                false
            };

            // VK_KHR_maintenance1: core in 1.1; required extension on older
            // devices when requested — hard filter.
            let use_maintenance1_ext = if config.maintenance1 && is_pre_1_1 {
                if has_ext(ash::khr::maintenance1::NAME) {
                    true
                } else {
                    tracing::debug!(
                        "Skipping {:?}: missing VK_KHR_maintenance1",
                        props.device_name_as_c_str().unwrap_or(c"unknown"),
                    );
                    continue 'dev;
                }
            } else {
                false
            };

            // VK_KHR_shader_non_semantic_info: core in 1.3; optional on older
            // devices when requested.
            let enable_shader_non_semantic = config.shader_non_semantic_info
                && is_pre_1_3
                && has_ext(ash::khr::shader_non_semantic_info::NAME);

            // VK_EXT_descriptor_indexing: core in 1.2; required extension on
            // older devices when requested — hard filter. Also requires
            // VK_KHR_maintenance3.
            let use_resource_indexing_ext =
                if config.resource_indexing && is_pre_1_2 {
                    if has_ext(ash::khr::maintenance3::NAME)
                        && has_ext(ash::ext::descriptor_indexing::NAME)
                    {
                        true
                    } else {
                        tracing::debug!(
                            "Skipping {:?}: missing \
                             VK_EXT_descriptor_indexing or \
                             VK_KHR_maintenance3",
                            props.device_name_as_c_str().unwrap_or(c"unknown"),
                        );
                        continue 'dev;
                    }
                } else {
                    false
                };

            // VK_EXT_memory_budget: optional device extension. Enables accurate
            // heap-usage/budget queries via
            // vkGetPhysicalDeviceMemoryProperties2.
            let enable_memory_budget = has_ext(ash::ext::memory_budget::NAME);

            // VK_KHR_dynamic_rendering: core in 1.3; required extension on
            // older devices when dynamic rendering is requested — hard filter.
            let use_dr_ext = if config.dynamic_rendering && is_pre_1_3 {
                if has_ext(ash::khr::dynamic_rendering::NAME) {
                    true
                } else {
                    tracing::debug!(
                        "Skipping {:?}: missing \
                             VK_KHR_dynamic_rendering",
                        props.device_name_as_c_str().unwrap_or(c"unknown"),
                    );
                    continue 'dev;
                }
            } else {
                false
            };

            // Find a family that supports both graphics and compute (the Vulkan
            // spec guarantees at least one such family exists; all real
            // desktop/mobile hardware exposes this as the primary graphics
            // family). We treat Graphics ⟹ Compute as an invariant enforced
            // here so that queue-capability marker types in the command module
            // can rely on it without runtime checks.
            let mut any_graphics_family: Option<u32> = None;
            let mut any_present_family: Option<u32> = None;
            let graphics_compute =
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
            for (idx, qf) in queue_families.iter().enumerate() {
                if any_graphics_family.is_none()
                    && qf.queue_flags.contains(graphics_compute)
                {
                    any_graphics_family = Some(idx as u32);
                    if let Ok(true) =
                        // SAFETY: `dev` and `surf` are both derived from the
                        // same `Instance`, so calling into the instance's
                        // surface support query is safe here.
                        unsafe {
                            surf.supports_queue_family(dev, idx as u32)
                        }
                        && (!(config.queue_config.dedicated_present
                            && config.queue_config_strict)
                            || any_present_family.is_none())
                    {
                        any_present_family = any_graphics_family;
                        break;
                    }
                }
                if let Ok(true) =
                    // SAFETY: `dev` and `surf` are both derived from the same
                    // `Instance`, so calling into the instance's surface
                    // support query is safe here.
                    unsafe {
                        surf.supports_queue_family(dev, idx as u32)
                    }
                {
                    any_present_family = Some(idx as u32);
                }
            }
            if any_graphics_family.is_none() || any_present_family.is_none() {
                tracing::debug!(
                    "Skipping {:?}: missing graphics or present family",
                    props.device_name_as_c_str().unwrap_or(c"unknown"),
                );
                continue 'dev;
            }

            let has_dedicated_transfer = queue_families.iter().any(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });
            let has_dedicated_present =
                any_graphics_family == any_present_family;
            let has_dedicated_compute = queue_families.iter().any(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });

            let dedicated_score = has_dedicated_transfer as u32
                + has_dedicated_compute as u32
                + !has_dedicated_present as u32;

            // Sample count support: intersect colour and depth limits.
            let supported_samples =
                props.limits.framebuffer_color_sample_counts
                    & props.limits.framebuffer_depth_sample_counts;
            let supports_sample_count =
                supported_samples.contains(config.min_sample_count);
            if config.min_sample_count_strict && !supports_sample_count {
                tracing::debug!(
                    "Skipping {:?}: does not support \
                     requested sample count {:?}",
                    props.device_name_as_c_str().unwrap_or(c"unknown"),
                    config.min_sample_count,
                );
                skipped_for_sample_count = true;
                continue 'dev;
            }

            // Feature sub-field check. Query and verify that every sub-feature
            // within each requested feature group is supported. Workaround
            // until we expose finer-grained feature selection (t044): require
            // all sub-features rather than checking individually.
            if config.synchronization2
                || config.dynamic_rendering
                || config.resource_indexing
                || config.sampler_anisotropy
            {
                let mut q_sync2 =
                    vk::PhysicalDeviceSynchronization2Features::default();
                let mut q_dr =
                    vk::PhysicalDeviceDynamicRenderingFeatures::default();
                let mut q_di =
                    vk::PhysicalDeviceDescriptorIndexingFeatures::default();
                let mut q = vk::PhysicalDeviceFeatures2::default();
                if config.synchronization2 {
                    q = q.push_next(&mut q_sync2);
                }
                if config.dynamic_rendering {
                    q = q.push_next(&mut q_dr);
                }
                if config.resource_indexing {
                    q = q.push_next(&mut q_di);
                }
                // SAFETY: dev was derived from instance; all structs in the
                // pNext chain are valid and properly initialised above.
                unsafe {
                    instance.get_physical_device_features2(dev, &mut q);
                }
                // Copy q.features.sampler_anisotropy before q_sync2/q_dr/q_di
                // are moved into the support-check functions below.
                let q_sampler_anisotropy = q.features.sampler_anisotropy;
                let supported = (!config.synchronization2
                    || sync2_fully_supported(q_sync2))
                    && (!config.dynamic_rendering
                        || dynamic_rendering_fully_supported(q_dr))
                    && (!config.resource_indexing
                        || resource_indexing_supported(q_di))
                    && (!config.sampler_anisotropy
                        || q_sampler_anisotropy == vk::TRUE);
                if !supported {
                    tracing::debug!(
                        "Skipping {:?}: missing required feature sub-fields",
                        props.device_name_as_c_str().unwrap_or(c"unknown"),
                    );
                    continue 'dev;
                }
            }

            let score = (
                dedicated_score,
                device_type_priority(props.device_type),
                supports_sample_count as u32,
            );

            candidates.push(DeviceCandidate {
                handle: dev,
                props,
                queue_families,
                graphics_family: any_graphics_family.unwrap(),
                present_family: any_present_family.unwrap(),
                score,
                use_sync2_ext,
                use_dr_ext,
                use_maintenance1_ext,
                enable_shader_non_semantic,
                enable_memory_budget,
                use_resource_indexing_ext,
            });
        }

        let best = candidates.iter().max_by_key(|c| c.score).ok_or(
            if skipped_for_sample_count {
                CreateCompatibleError::SampleCountUnsupported
            } else {
                CreateCompatibleError::NoSuitableDevice
            },
        )?;

        let physical_device = best.handle;
        let queue_families = &best.queue_families;
        let graphics_family = best.graphics_family;
        let use_sync2_ext = best.use_sync2_ext;
        let use_dr_ext = best.use_dr_ext;
        let use_maintenance1_ext = best.use_maintenance1_ext;
        let use_resource_indexing_ext = best.use_resource_indexing_ext;
        // SAFETY: physical_device was selected from this instance.
        let memory_properties = unsafe {
            instance.get_raw_physical_device_memory_properties(physical_device)
        };
        tracing::info!(
            "Selected physical device: {:?} \
             (type: {:?}, dedicated queues: {})",
            best.props.device_name_as_c_str().unwrap_or(c"unknown"),
            best.props.device_type,
            best.score.0,
        );

        // --- Family selection ---
        //
        // When dedicated_transfer/compute is requested, prefer a family with
        // the required flag but without GRAPHICS. Fall back to
        // graphics_present_family if none exists.
        //
        // When dedicated_present is not requested, prefer the graphics queue
        // else use the dedicated_present_queue.
        let dedicated_transfer_family =
            queue_families.iter().enumerate().find_map(|(idx, qf)| {
                if qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    Some(idx as u32)
                } else {
                    None
                }
            });

        let dedicated_compute_family =
            queue_families.iter().enumerate().find_map(|(idx, qf)| {
                if qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    Some(idx as u32)
                } else {
                    None
                }
            });
        let dedicated_present_family = {
            queue_families.iter().enumerate().find_map(|(idx, qf)| {
                if !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    // SAFETY: `dev` and `surf` are both derived from the same
                    // `Instance`, so calling into the instance's surface
                    // support query is safe here.
                    && let Ok(true) = unsafe {
                        surf.supports_queue_family(best.handle, idx as u32)
                    }
                {
                    Some(idx as u32)
                } else {
                    None
                }
            })
        };

        let transfer_family = if config.queue_config.dedicated_transfer {
            dedicated_transfer_family.unwrap_or(graphics_family)
        } else {
            graphics_family
        };

        let compute_family = if config.queue_config.dedicated_compute {
            dedicated_compute_family.unwrap_or(graphics_family)
        } else {
            graphics_family
        };

        let present_family = if !config.queue_config.dedicated_present {
            best.present_family
        } else {
            dedicated_present_family.unwrap_or(best.present_family)
        };

        tracing::info!(
            "Queue families — \
             graphics: {}, present: {}, transfer: {}, compute: {}",
            graphics_family,
            graphics_family,
            transfer_family,
            compute_family
        );

        // --- Queue count per family --- Always allocate exactly one queue per
        // family (no parallelism).
        let mut queue_families: HashSet<u32> = HashSet::new();
        for &family in &[graphics_family, transfer_family, compute_family] {
            queue_families.insert(family);
        }

        // --- Determine effective config and check strictness --- Each axis is
        // checked independently: dedicated_transfer and dedicated_compute
        // reflect whether each role got its own family; parallel reflects
        // whether more than one queue was allocated from the graphics/present
        // family.
        let effective_config = QueueConfig {
            dedicated_transfer: transfer_family != graphics_family,
            dedicated_compute: compute_family != graphics_family,
            dedicated_present: false,
        };

        if config.queue_config_strict {
            let req = config.queue_config;
            let eff = effective_config;
            if (req.dedicated_transfer && !eff.dedicated_transfer)
                || (req.dedicated_compute && !eff.dedicated_compute)
                || (req.dedicated_present && !eff.dedicated_present)
            {
                return Err(CreateCompatibleError::QueueConfigUnsatisfied {
                    requested: req,
                    actual: eff,
                });
            }
        }

        let priorities = [1.0];
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'_>> =
            queue_families
                .iter()
                .map(|family| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*family)
                        .queue_priorities(&priorities)
                })
                .collect();

        // Build the device extension list. A HashSet is used so that dependency
        // extensions can be inserted freely without worrying about duplicates.
        //
        // Extension dependency chains (device extensions only; instance
        // extensions such as VK_KHR_get_physical_device_properties2 are omitted
        // because they cannot appear in ppEnabledExtensionNames):
        //
        //   VK_KHR_swapchain (no device-ext deps)
        //     VK_KHR_shader_non_semantic_info (no deps) VK_EXT_memory_budget
        //   (no device-ext deps) VK_KHR_synchronization2 (no device-ext deps)
        //     VK_KHR_dynamic_rendering └── VK_KHR_depth_stencil_resolve (1.2
        //   core) └── VK_KHR_create_renderpass2 (1.2 core) ├── VK_KHR_multiview
        //     (1.1 core) └── VK_KHR_maintenance2 (1.1 core) VK_KHR_maintenance1
        //   (no deps) VK_EXT_descriptor_indexing └── VK_KHR_maintenance3 (no
        //     device-ext deps)
        let mut mandatory_exts: HashSet<&CStr> = HashSet::new();
        if config.swapchain {
            mandatory_exts.insert(ash::khr::swapchain::NAME);
        }
        if best.enable_shader_non_semantic {
            mandatory_exts.insert(ash::khr::shader_non_semantic_info::NAME);
        }
        if best.enable_memory_budget {
            mandatory_exts.insert(ash::ext::memory_budget::NAME);
        }
        if use_sync2_ext {
            mandatory_exts.insert(ash::khr::synchronization2::NAME);
        }
        if use_dr_ext {
            // Full transitive device-extension dependency chain for
            // VK_KHR_dynamic_rendering. Validation requires all deps in
            // ppEnabledExtensionNames even when promoted to core.
            mandatory_exts.insert(ash::khr::maintenance2::NAME);
            mandatory_exts.insert(ash::khr::multiview::NAME);
            mandatory_exts.insert(ash::khr::create_renderpass2::NAME);
            mandatory_exts.insert(ash::khr::depth_stencil_resolve::NAME);
            mandatory_exts.insert(ash::khr::dynamic_rendering::NAME);
        }
        if use_maintenance1_ext {
            mandatory_exts.insert(ash::khr::maintenance1::NAME);
        }
        if use_resource_indexing_ext {
            mandatory_exts.insert(ash::khr::maintenance3::NAME);
            mandatory_exts.insert(ash::ext::descriptor_indexing::NAME);
        }

        let ext_ptrs: Vec<*const i8> =
            mandatory_exts.iter().map(|e| e.as_ptr()).collect();

        // Query which features are actually supported by this physical device.
        // Use temporary query structs and copy reported fields into fresh
        // structs that will be pushed into device creation. This avoids reusing
        // the same struct instance for multiple purposes.
        let mut q_sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default();
        let mut q_dr_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut q_resource_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures::default();

        if config.synchronization2
            || config.dynamic_rendering
            || config.resource_indexing
        {
            let mut query = vk::PhysicalDeviceFeatures2::default();
            if config.synchronization2 {
                query = query.push_next(&mut q_sync2_features);
            }
            if config.dynamic_rendering {
                query = query.push_next(&mut q_dr_features);
            }
            if config.resource_indexing {
                query = query.push_next(&mut q_resource_indexing_features);
            }
            // SAFETY: physical_device was selected from this instance; all
            // structs in the pNext chain are valid and properly initialised
            // above.
            unsafe {
                instance
                    .get_physical_device_features2(physical_device, &mut query);
            }
        }

        // Copy queried values into fresh structs used for device creation so
        // that `p_next` is always intentional (default-null) and the query
        // instances are not reused. Initialize members directly (only `s_type`,
        // `p_next`, and `_marker` are left to `Default`).
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features {
            synchronization2: q_sync2_features.synchronization2,
            ..Default::default()
        };

        let mut dr_features = vk::PhysicalDeviceDynamicRenderingFeatures {
            dynamic_rendering: q_dr_features.dynamic_rendering,
            ..Default::default()
        };

        let mut resource_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures {
                shader_input_attachment_array_dynamic_indexing: q_resource_indexing_features
                    .shader_input_attachment_array_dynamic_indexing,
                shader_uniform_texel_buffer_array_dynamic_indexing: q_resource_indexing_features
                    .shader_uniform_texel_buffer_array_dynamic_indexing,
                shader_storage_texel_buffer_array_dynamic_indexing: q_resource_indexing_features
                    .shader_storage_texel_buffer_array_dynamic_indexing,
                shader_uniform_buffer_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_uniform_buffer_array_non_uniform_indexing,
                shader_sampled_image_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_sampled_image_array_non_uniform_indexing,
                shader_storage_buffer_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_storage_buffer_array_non_uniform_indexing,
                shader_storage_image_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_storage_image_array_non_uniform_indexing,
                shader_input_attachment_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_input_attachment_array_non_uniform_indexing,
                shader_uniform_texel_buffer_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_uniform_texel_buffer_array_non_uniform_indexing,
                shader_storage_texel_buffer_array_non_uniform_indexing: q_resource_indexing_features
                    .shader_storage_texel_buffer_array_non_uniform_indexing,
                descriptor_binding_uniform_buffer_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_uniform_buffer_update_after_bind,
                descriptor_binding_sampled_image_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_sampled_image_update_after_bind,
                descriptor_binding_storage_image_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_storage_image_update_after_bind,
                descriptor_binding_storage_buffer_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_storage_buffer_update_after_bind,
                descriptor_binding_uniform_texel_buffer_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_uniform_texel_buffer_update_after_bind,
                descriptor_binding_storage_texel_buffer_update_after_bind: q_resource_indexing_features
                    .descriptor_binding_storage_texel_buffer_update_after_bind,
                descriptor_binding_update_unused_while_pending: q_resource_indexing_features
                    .descriptor_binding_update_unused_while_pending,
                descriptor_binding_partially_bound: q_resource_indexing_features
                    .descriptor_binding_partially_bound,
                descriptor_binding_variable_descriptor_count: q_resource_indexing_features
                    .descriptor_binding_variable_descriptor_count,
                runtime_descriptor_array: q_resource_indexing_features.runtime_descriptor_array,
                ..Default::default()
            };

        // Policy: do not reuse feature structs — use fresh instances for
        // querying and for pushing into pNext chains. Copy reported sub-fields
        // into a new struct used for `DeviceCreateInfo` so `p_next` remains
        // intentional and clear. See: rgpu-vk/README.md#pnext-chain-policy
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&ext_ptrs);

        // On Vulkan < 1.1, extension feature structs must be chained through
        // VkPhysicalDeviceFeatures2 (from
        // VK_KHR_get_physical_device_properties2) rather than placed directly
        // in VkDeviceCreateInfo::pNext. On 1.1+ core, they go directly on
        // DeviceCreateInfo.
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        let mut use_features2 = false;
        if config.synchronization2 {
            if use_sync2_ext {
                features2 = features2.push_next(&mut sync2_features);
                use_features2 = true;
            } else {
                device_create_info =
                    device_create_info.push_next(&mut sync2_features);
            }
        }

        if config.dynamic_rendering {
            if use_dr_ext {
                features2 = features2.push_next(&mut dr_features);
                use_features2 = true;
            } else {
                device_create_info =
                    device_create_info.push_next(&mut dr_features);
            }
        }

        if config.resource_indexing {
            if use_resource_indexing_ext {
                features2 =
                    features2.push_next(&mut resource_indexing_features);
                use_features2 = true;
            } else {
                device_create_info = device_create_info
                    .push_next(&mut resource_indexing_features);
            }
        }
        // Core Vulkan 1.0 features (VkPhysicalDeviceFeatures).
        // Declared here so the reference passed to enabled_features()
        // outlives the create_ash_device call below.
        let mut core_features = vk::PhysicalDeviceFeatures::default();
        if config.sampler_anisotropy {
            core_features.sampler_anisotropy = vk::TRUE;
        }
        let need_core_features = config.sampler_anisotropy;

        if use_features2 {
            // When VkPhysicalDeviceFeatures2 goes in the pNext chain,
            // pEnabledFeatures must be NULL — set core features via
            // features2.features instead.
            if need_core_features {
                features2.features = core_features;
            }
            device_create_info = device_create_info.push_next(&mut features2);
        } else if need_core_features {
            // No features2 pNext chain: use pEnabledFeatures directly.
            device_create_info =
                device_create_info.enabled_features(&core_features);
        }

        // SAFETY: physical_device was derived from instance; device_create_info
        // is fully initialised above.
        let device = unsafe {
            instance.create_ash_device(physical_device, &device_create_info)
        }
        .map_err(CreateCompatibleError::DeviceCreationFailed)?;

        // Build per-family single `Arc<Mutex<vk::Queue>>` entries. Roles that
        // share a family share the same Arc instance, so locking any role
        // serialises on the same Mutex.
        let mut family_queues: HashMap<u32, Arc<Mutex<vk::Queue>>> =
            HashMap::new();
        for family in &queue_families {
            // SAFETY: device was just created requesting at least one queue
            // from this family; always fetch queue 0.
            let q = unsafe { device.get_device_queue(*family, 0) };
            family_queues.insert(*family, Arc::new(Mutex::new(q)));
        }

        let debug_utils_device =
            instance.create_debug_utils_device_loader(&device);

        for (family, queue) in &mut family_queues {
            debug_utils_device.as_ref().inspect(|dud| {
                let queue = Arc::get_mut(queue)
                    .expect("This Arc should not have been cloned yet")
                    .get_mut();
                let mut queue_type_strs = Vec::with_capacity(4);
                if *family == graphics_family {
                    queue_type_strs.push("graphics");
                }
                if *family == present_family {
                    queue_type_strs.push("present");
                }
                if *family == transfer_family {
                    queue_type_strs.push("transfer");
                }
                if *family == compute_family {
                    queue_type_strs.push("compute");
                }

                let queue_type_str = if queue_type_strs.is_empty() {
                    "unknown".to_string()
                } else {
                    queue_type_strs.join("+")
                };

                let queue_debug_name = std::ffi::CString::new(format!(
                    "{} Queue (family: {})",
                    queue_type_str, family
                ))
                .expect("Failed to create CString for queue debug name");
                // SAFETY: device was just created. Queue was just created from
                // this device
                unsafe {
                    let _ = dud.set_debug_utils_object_name(
                        &vk::DebugUtilsObjectNameInfoEXT::default()
                            .object_handle(*queue)
                            .object_name(&queue_debug_name),
                    );
                };
            });
        }

        let graphics_queue = family_queues[&graphics_family].clone();
        let present_queue = family_queues[&present_family].clone();
        let transfer_queues = family_queues[&transfer_family].clone();
        let compute_queues = family_queues[&compute_family].clone();

        // SAFETY: instance, device, and physical_device are valid for the
        // lifetime of the allocator, which is owned by Device and dropped
        // before vkDestroyDevice (see Device::drop).
        let allocator = unsafe {
            Allocator::new(VmaAllocatorCreateInfo::new(
                instance.ash_instance(),
                &device,
                physical_device,
            ))
        }
        .map_err(CreateCompatibleError::AllocatorCreation)?;

        Ok(Self {
            parent: instance.clone(),
            allocator: Some(allocator),
            memory_properties,
            properties: best.props,
            swapchain_device: if config.swapchain {
                Some(instance.create_swapchain_loader(&device))
            } else {
                None
            },
            debug_utils_device,
            dynamic_rendering: if config.dynamic_rendering {
                if use_dr_ext {
                    Some(DynamicRenderingLoader::Extension(
                        instance.create_dynamic_rendering_loader(&device),
                    ))
                } else {
                    Some(DynamicRenderingLoader::Core)
                }
            } else {
                None
            },
            synchronization2: if config.synchronization2 {
                Some(if use_sync2_ext {
                    Synchronization2Loader::Extension(
                        instance.create_synchronization2_loader(&device),
                    )
                } else {
                    Synchronization2Loader::Core
                })
            } else {
                None
            },
            handle: device,
            physical_device,
            memory_budget: best.enable_memory_budget,
            graphics_queue,
            graphics_queue_family: graphics_family,
            present_queue,
            present_family: graphics_family,
            transfer_queues,
            transfer_family,
            compute_queues,
            compute_family,
            queue_config: effective_config,
        })
    }

    #[inline]
    pub fn parent(&self) -> &Arc<Instance> {
        &self.parent
    }

    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns `true` if `VK_EXT_memory_budget` was enabled on this device.
    /// When true, callers may chain
    /// `vk::PhysicalDeviceMemoryBudgetPropertiesEXT` into
    /// `vkGetPhysicalDeviceMemoryProperties2` to obtain accurate per-heap usage
    /// and budget figures.
    #[inline]
    pub fn has_memory_budget(&self) -> bool {
        self.memory_budget
    }

    #[inline]
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    #[inline]
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    #[inline]
    /// Return the first format in `candidates` that supports
    /// `DEPTH_STENCIL_ATTACHMENT` in optimal tiling, or `None` if none do.
    pub fn find_depth_format(
        &self,
        candidates: &[vk::Format],
    ) -> Option<vk::Format> {
        candidates.iter().copied().find(|&fmt| {
            // SAFETY: physical_device is a valid handle selected from this
            // instance during device creation.
            let props = unsafe {
                self.parent
                    .ash_instance()
                    .get_physical_device_format_properties(
                        self.physical_device,
                        fmt,
                    )
            };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
    }

    fn alloc_create_info(usage: MemoryUsage) -> AllocationCreateInfo {
        use vk_mem::MemoryUsage as VmaUsage;
        match usage {
            MemoryUsage::GpuOnly => AllocationCreateInfo {
                usage: VmaUsage::AutoPreferDevice,
                ..Default::default()
            },
            MemoryUsage::Upload => AllocationCreateInfo {
                usage: VmaUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
            MemoryUsage::HostVisibleBind => AllocationCreateInfo {
                usage: VmaUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | AllocationCreateFlags::MAPPED,
                preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
            MemoryUsage::Download => AllocationCreateInfo {
                usage: VmaUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_RANDOM
                    | AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
            MemoryUsage::UpDown => AllocationCreateInfo {
                usage: VmaUsage::Auto,
                flags: AllocationCreateFlags::HOST_ACCESS_RANDOM
                    | AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
        }
    }

    /// Create a `VkBuffer` and bind it to a VMA allocation in one call.
    ///
    /// Returns `(buffer, allocation)`. The allocation's mapped pointer is
    /// available via [`Allocation::mapped_ptr`] for host-visible usages.
    ///
    /// # Safety
    /// `buffer_info` must be valid. The returned buffer and allocation must be
    /// destroyed together via [`destroy_raw_buffer_allocated`].
    pub(crate) unsafe fn create_raw_buffer_allocated(
        &self,
        buffer_info: &vk::BufferCreateInfo<'_>,
        usage: MemoryUsage,
    ) -> Result<(vk::Buffer, Allocation), AllocateMemoryError> {
        let alloc_info = Self::alloc_create_info(usage);
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        // SAFETY: buffer_info is valid per caller contract; allocator is live.
        unsafe { allocator.create_buffer(buffer_info, &alloc_info) }
            .map(|(buf, a)| (buf, Allocation(a)))
            .map_err(AllocateMemoryError)
    }

    /// Destroy a buffer and free its VMA allocation.
    ///
    /// # Safety
    /// `buffer` and `allocation` must have been created together by
    /// [`create_raw_buffer_allocated`] on this device, and must not be in use
    /// by the GPU.
    pub(crate) unsafe fn destroy_raw_buffer_allocated(
        &self,
        buffer: vk::Buffer,
        allocation: &mut Allocation,
    ) {
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        // SAFETY: Caller guarantees buffer and allocation validity.
        unsafe { allocator.destroy_buffer(buffer, &mut allocation.0) };
    }

    /// Create a `VkImage` and bind it to a VMA allocation in one call.
    ///
    /// # Safety
    /// `image_info` must be valid. The returned image and allocation must be
    /// destroyed together via [`destroy_raw_image_allocated`].
    pub(crate) unsafe fn create_raw_image_allocated(
        &self,
        image_info: &vk::ImageCreateInfo<'_>,
        usage: MemoryUsage,
    ) -> Result<(vk::Image, Allocation), AllocateMemoryError> {
        let alloc_info = Self::alloc_create_info(usage);
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        // SAFETY: image_info is valid per caller contract; allocator is live.
        unsafe { allocator.create_image(image_info, &alloc_info) }
            .map(|(img, a)| (img, Allocation(a)))
            .map_err(AllocateMemoryError)
    }

    /// Destroy an image and free its VMA allocation.
    ///
    /// # Safety
    /// `image` and `allocation` must have been created together by
    /// [`create_raw_image_allocated`] on this device, and must not be in use by
    /// the GPU.
    pub(crate) unsafe fn destroy_raw_image_allocated(
        &self,
        image: vk::Image,
        allocation: &mut Allocation,
    ) {
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        // SAFETY: Caller guarantees image and allocation validity.
        unsafe { allocator.destroy_image(image, &mut allocation.0) };
    }

    /// Flush a range of a host-visible allocation.
    ///
    /// Delegates to VMA, which handles non-coherent atom alignment internally.
    /// Pass `vk::WHOLE_SIZE` for `size` to flush the entire allocation.
    pub(crate) fn flush_allocation(
        &self,
        allocation: &Allocation,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<(), vk::Result> {
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop");
        allocator.flush_allocation(&allocation.0, offset, size)
    }

    #[inline]
    pub fn ash_device(&self) -> &ash::Device {
        &self.handle
    }

    /// Wait until all submitted work on this device has completed.
    ///
    /// This may block the calling thread and should generally be used for
    /// coarse-grained transitions (shutdown, suspend, swapchain teardown)
    /// rather than hot per-frame paths.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        let _span = tracing::debug_span!("device_wait_idle").entered();
        // SAFETY: `self.handle` is a valid logical device for the lifetime of
        // `self`, and this call has no additional pointer preconditions.
        unsafe { self.handle.device_wait_idle() }
    }

    #[inline]
    pub fn raw_device(&self) -> vk::Device {
        self.handle.handle()
    }

    #[inline]
    pub fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    #[inline]
    pub fn present_queue_family(&self) -> u32 {
        self.present_family
    }

    #[inline]
    pub fn transfer_queue_family(&self) -> u32 {
        self.transfer_family
    }

    #[inline]
    pub fn compute_queue_family(&self) -> u32 {
        self.compute_family
    }

    #[inline]
    pub fn queue_config(&self) -> QueueConfig {
        self.queue_config
    }
}

// image functionality
impl Device {
    /// Destroy a `VkSampler`.
    ///
    /// # Safety
    /// `sampler` must be a valid handle created from this device and not yet
    /// destroyed. No in-flight GPU work may still reference `sampler`.
    #[inline]
    pub unsafe fn destroy_raw_sampler(&self, sampler: vk::Sampler) {
        // SAFETY: Caller guarantees sampler provenance and drop ordering.
        unsafe { self.handle.destroy_sampler(sampler, None) };
    }

    /// # Safety
    /// `create_info` must reference valid Vulkan objects derived from this
    /// device. Any referenced pointers must remain valid for the duration of
    /// the call.
    #[inline]
    pub unsafe fn create_raw_image_view(
        &self,
        create_info: &vk::ImageViewCreateInfo<'_>,
    ) -> Result<vk::ImageView, vk::Result> {
        // SAFETY: Caller guarantees create_info validity and provenance.
        unsafe { self.handle.create_image_view(create_info, None) }
    }

    /// # Safety
    /// `image_view` must be a valid handle derived from this device, and all
    /// objects using it must be destroyed first.
    ///
    /// No in-flight GPU work may still reference the image view.
    #[inline]
    pub unsafe fn destroy_raw_image_view(&self, image_view: vk::ImageView) {
        // SAFETY: Caller guarantees image_view provenance and drop ordering.
        unsafe { self.handle.destroy_image_view(image_view, None) };
    }

    /// Create a `VkImage`.
    ///
    /// # Safety
    /// `create_info` must be valid and all referenced pointers must remain
    /// valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_image(
        &self,
        create_info: &vk::ImageCreateInfo<'_>,
    ) -> Result<vk::Image, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_image(create_info, None) }
    }

    /// Destroy a `VkImage`.
    ///
    /// # Safety
    /// `image` must be a valid handle created from this device and not yet
    /// destroyed. No in-flight GPU work may still reference `image`.
    #[inline]
    pub unsafe fn destroy_raw_image(&self, image: vk::Image) {
        // SAFETY: Caller guarantees image provenance and drop ordering.
        unsafe { self.handle.destroy_image(image, None) };
    }

    /// Create a `VkSampler`.
    ///
    /// # Safety
    /// `create_info` must be valid and all referenced pointers must remain
    /// valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_sampler(
        &self,
        create_info: &vk::SamplerCreateInfo<'_>,
    ) -> Result<vk::Sampler, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_sampler(create_info, None) }
    }
}

// Swapchain functionality
impl Device {
    /// # Safety
    /// `create_info` must reference valid Vulkan objects derived from this
    /// device and its parent instance. Any referenced pointers must remain
    /// valid for the duration of the call.
    ///
    /// If `create_info.old_swapchain` is non-null, that handle must be a valid
    /// swapchain created from this device.
    #[inline]
    pub unsafe fn create_raw_swapchain(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<vk::SwapchainKHR, vk::Result> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .expect("swapchain was not enabled in DeviceConfig");
        // SAFETY: Caller guarantees create_info validity and handle provenance.
        unsafe { swapchain_device.create_swapchain(create_info, None) }
    }

    /// # Safety
    /// `swapchain` must be a valid swapchain handle created from this device
    /// and not yet destroyed.
    #[inline]
    pub unsafe fn get_raw_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> Result<Vec<vk::Image>, vk::Result> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .expect("swapchain was not enabled in DeviceConfig");
        // SAFETY: Caller guarantees swapchain validity and lifetime.
        unsafe { swapchain_device.get_swapchain_images(swapchain) }
    }

    /// # Safety
    /// `swapchain` must be a valid handle derived from this device, and all
    /// child resources derived from it must be destroyed first.
    ///
    /// No in-flight GPU work may still reference the swapchain.
    #[inline]
    pub unsafe fn destroy_raw_swapchain(&self, swapchain: vk::SwapchainKHR) {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .expect("swapchain was not enabled in DeviceConfig");
        // SAFETY: Caller guarantees swapchain provenance and drop ordering.
        unsafe { swapchain_device.destroy_swapchain(swapchain, None) };
    }

    /// Acquire the next presentable swapchain image.
    ///
    /// Returns `(image_index, is_suboptimal)`. A suboptimal result means the
    /// image was acquired successfully but the swapchain no longer exactly
    /// matches the surface; recreation at the next opportunity is recommended.
    ///
    /// Returns `Err(vk::Result::ERROR_OUT_OF_DATE_KHR)` when the swapchain is
    /// incompatible with the surface and must be recreated before presentation
    /// can resume.
    ///
    /// # Safety
    /// `swapchain` must be a valid handle created from this device. `semaphore`
    /// and `fence`, when not null, must be valid unsignaled handles created
    /// from this device.
    #[inline]
    pub unsafe fn acquire_next_swapchain_image(
        &self,
        swapchain: vk::SwapchainKHR,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(u32, bool), vk::Result> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .expect("swapchain was not enabled in DeviceConfig");
        // SAFETY: Caller guarantees swapchain, semaphore, and fence validity.
        unsafe {
            swapchain_device
                .acquire_next_image(swapchain, timeout_ns, semaphore, fence)
        }
    }

    /// Present a rendered swapchain image to the surface via the
    /// graphics/present queue.
    ///
    /// Returns `Ok(true)` when the swapchain is suboptimal and should be
    /// recreated at the next opportunity.
    ///
    /// Returns `Err(vk::Result::ERROR_OUT_OF_DATE_KHR)` when recreation is
    /// mandatory before the next present.
    ///
    /// # Safety
    /// All handles in `present_info` must be valid and derived from this
    /// device. Wait semaphores must be signaled. The presented image must be in
    /// `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` and not referenced by any pending GPU
    /// work other than this presentation.
    #[inline]
    pub unsafe fn queue_present(
        &self,
        present_info: &vk::PresentInfoKHR<'_>,
    ) -> Result<bool, QueuePresentError> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .expect("swapchain was not enabled in DeviceConfig");
        let queue = self.acquire_present_queue();
        // SAFETY: Caller guarantees all handles and synchronization
        // requirements.
        unsafe { swapchain_device.queue_present(*queue, present_info) }
            .map_err(QueuePresentError::Vulkan)
    }

    #[inline]
    pub fn has_swapchain_support(&self) -> bool {
        self.swapchain_device.is_some()
    }
}

// Debug naming and label functionality
impl Device {
    /// Set a Vulkan debug name for an object owned by this device.
    ///
    /// Passing `None` as the name is treated as a no-op.
    ///
    /// # Safety
    /// `object` must be a valid Vulkan handle created from this device (or a
    /// child object associated with this device) and must remain valid for the
    /// duration of the call.
    pub unsafe fn set_object_name<H>(
        &self,
        object: H,
        name: Option<&CStr>,
    ) -> Result<(), NameObjectError>
    where
        H: vk::Handle,
    {
        let Some(debug_utils) = self.debug_utils_device.as_ref() else {
            return Ok(());
        };

        let Some(name) = name else {
            return Ok(());
        };

        let object_name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(name);

        // SAFETY: Caller guarantees object provenance and validity.
        unsafe { debug_utils.set_debug_utils_object_name(&object_name_info) }
            .map_err(NameObjectError::Vulkan)
    }

    /// Lazily set a Vulkan debug name for an object owned by this device.
    ///
    /// The closure is only called if `VK_EXT_debug_utils` is enabled. Returning
    /// `None` from the closure is treated as a no-op.
    ///
    /// # Safety
    /// `object` must be a valid Vulkan handle created from this device (or a
    /// child object associated with this device) and must remain valid for the
    /// duration of the call.
    pub unsafe fn set_object_name_lazy<H, F, T>(
        &self,
        object: H,
        name_provider: F,
    ) -> Result<(), NameObjectError>
    where
        H: vk::Handle,
        F: FnOnce() -> Option<T>,
        T: AsRef<CStr>,
    {
        if self.debug_utils_device.is_none() {
            return Ok(());
        }

        let name = name_provider();
        // SAFETY: This method shares the same safety contract as
        // set_object_name.
        unsafe {
            self.set_object_name(object, name.as_ref().map(|n| n.as_ref()))
        }
    }

    /// Begin a queue label region using the passed name
    ///
    /// # Safety
    /// Queue comes from this device
    unsafe fn begin_queue_debug_label_cstr(
        &self,
        queue: vk::Queue,
        label: Option<&CStr>,
    ) {
        if let Some(debug_utils) = self.debug_utils_device.as_ref()
            && let Some(label) = label
        {
            let label_info =
                vk::DebugUtilsLabelEXT::default().label_name(label);
            // SAFETY: Queue came from this device. We have a debug_utils_device
            // so by definition we can use debug utils functions
            unsafe {
                debug_utils.queue_begin_debug_utils_label(queue, &label_info);
            }
        }
    }

    pub fn debug_utils_enabled(&self) -> bool {
        self.debug_utils_device.is_some()
    }
    /// End a previously began queue label region
    ///
    /// # Safety
    /// We must be in a queue label on the passed Queue. Queue comes from this
    /// device.
    unsafe fn end_queue_debug_label(&self, queue: vk::Queue) {
        if let Some(debug_utils) = self.debug_utils_device.as_ref() {
            // SAFETY: Caller guarantees queue family validity. We have a
            // debug_utils_device so by definition we can use debug utils
            // functions
            unsafe { debug_utils.queue_end_debug_utils_label(queue) };
        }
    }

    /// Begin a queue label region on the graphics queue using the passed name
    ///
    /// # Safety
    /// label is valid UTF-8
    pub unsafe fn begin_graphics_queue_debug_label_cstr(
        &self,
        label: Option<&CStr>,
    ) {
        let queue = self.graphics_queue.lock();
        // SAFETY: queue comes from this device, label is valid UTF-8 per our
        // unsafe contract
        unsafe { self.begin_queue_debug_label_cstr(*queue, label) };
    }

    /// End a previously began queue label region on the graphics queue
    ///
    /// # Safety
    /// We must be in a queue debug label on the graphics queue
    pub unsafe fn end_graphics_queue_debug_label(&self) {
        let queue = self.graphics_queue.lock();

        // SAFETY: queue comes from this device. We are in a queue debug label
        // per our safety contract
        unsafe { self.end_queue_debug_label(*queue) };
    }

    /// Begin a queue label region on the present queue using the passed name
    ///
    /// # Safety
    /// label is valid UTF-8
    pub unsafe fn begin_present_queue_debug_label_cstr(
        &self,
        label: Option<&CStr>,
    ) {
        let queue = self.present_queue.lock();
        // SAFETY: queue comes from this device, label is valid UTF-8 per our
        // unsafe contract
        unsafe { self.begin_queue_debug_label_cstr(*queue, label) };
    }

    /// End a previously began queue label region on the present queue
    ///
    /// # Safety
    /// We must be in a queue debug label on the present queue
    pub unsafe fn end_present_queue_debug_label(&self) {
        let queue = self.present_queue.lock();
        // SAFETY: queue comes from this device. We are in a queue debug label
        // per our safety contract
        unsafe { self.end_queue_debug_label(*queue) };
    }

    /// Begin a queue label region on the transfer queue using the passed name
    ///
    /// # Safety
    /// label is valid UTF-8
    pub unsafe fn begin_transfer_queue_debug_label_cstr(
        &self,
        label: Option<&CStr>,
    ) {
        let queue = self.transfer_queues.lock();
        // SAFETY: queue comes from this device, label is valid UTF-8 per our
        // unsafe contract
        unsafe { self.begin_queue_debug_label_cstr(*queue, label) };
    }

    /// End a previously began queue label region on the transfer queue
    ///
    /// # Safety
    /// We must be in a queue debug label on the transfer queue
    pub unsafe fn end_transfer_queue_debug_label(&self) {
        let queue = self.transfer_queues.lock();
        // SAFETY: queue comes from this device. We are in a queue debug label
        // per our safety contract
        unsafe { self.end_queue_debug_label(*queue) };
    }

    /// Begin a queue label region on the compute queue using the passed name
    ///
    /// # Safety
    /// label is valid UTF-8
    pub unsafe fn begin_compute_queue_debug_label_cstr(
        &self,
        label: Option<&CStr>,
    ) {
        let queue = self.compute_queues.lock();
        // SAFETY: queue comes from this device, label is valid UTF-8 per our
        // unsafe contract
        unsafe { self.begin_queue_debug_label_cstr(*queue, label) };
    }

    /// End a previously began queue label region on the compute queue
    ///
    /// # Safety
    /// We must be in a queue debug label on the compute queue
    pub unsafe fn end_compute_queue_debug_label(&self) {
        let queue = self.compute_queues.lock();
        // SAFETY: queue comes from this device. We are in a queue debug label
        // per our safety contract
        unsafe { self.end_queue_debug_label(*queue) };
    }

    /// Convenience helper to set a name from UTF-8 text.
    ///
    /// Passing `None` as the name is treated as a no-op.
    ///
    /// # Safety
    /// `object` must be a valid Vulkan handle created from this device (or a
    /// child object associated with this device) and must remain valid for the
    /// duration of the call.
    pub unsafe fn set_object_name_str<H>(
        &self,
        object: H,
        name: Option<&str>,
    ) -> Result<(), NameObjectError>
    where
        H: vk::Handle,
    {
        let name = match name {
            Some(name) => {
                Some(CString::new(name).map_err(NameObjectError::InvalidName)?)
            }
            None => None,
        };

        // SAFETY: This method shares the same safety contract as
        // set_object_name.
        unsafe { self.set_object_name(object, name.as_deref()) }
    }

    // ── Command-buffer debug labels ──────────────────────────────────────
    //
    // Unlike queue labels (which annotate submissions on the host timeline),
    // command-buffer labels are recorded into the command buffer and appear on
    // the GPU timeline in tools such as RenderDoc.
    //
    // Only the _cstr primitives are provided here. Queue-label style safe &str
    // / lazy wrappers for begin can be added if needed.

    /// Begin a debug label region inside a command buffer.
    ///
    /// # Safety
    /// `command_buffer` must be a valid handle in the recording state, derived
    /// from this device. `label` must contain only valid UTF-8 bytes.
    pub unsafe fn cmd_begin_debug_label_cstr(
        &self,
        command_buffer: vk::CommandBuffer,
        label: Option<&CStr>,
        color: [f32; 4],
    ) {
        if let Some(debug_utils) = self.debug_utils_device.as_ref() {
            let label_name = label.unwrap_or(c"");
            let label_info = vk::DebugUtilsLabelEXT::default()
                .label_name(label_name)
                .color(color);
            // SAFETY: command_buffer is valid and in the recording state per
            // our contract. label_name is valid UTF-8 per our contract.
            unsafe {
                debug_utils
                    .cmd_begin_debug_utils_label(command_buffer, &label_info)
            }
        }
    }

    /// End a previously begun debug label region inside a command buffer.
    ///
    /// # Safety
    /// `command_buffer` must be a valid handle in the recording state, derived
    /// from this device. A matching `begin_cmd_debug_label_cstr` call must have
    /// been recorded into this command buffer.
    pub unsafe fn end_cmd_debug_label(
        &self,
        command_buffer: vk::CommandBuffer,
    ) {
        if let Some(debug_utils) = self.debug_utils_device.as_ref() {
            // SAFETY: command_buffer is valid and in the recording state per
            // our contract. A matching begin label was recorded per our
            // contract.
            unsafe { debug_utils.cmd_end_debug_utils_label(command_buffer) }
        }
    }
}

// Shader module functionality
impl Device {
    /// # Safety
    /// `create_info` must contain valid SPIR-V code. All referenced pointers
    /// must remain valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_shader_module(
        &self,
        create_info: &vk::ShaderModuleCreateInfo<'_>,
    ) -> Result<vk::ShaderModule, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_shader_module(create_info, None) }
    }

    /// # Safety
    /// `shader_module` must be a valid handle created from this device and not
    /// yet destroyed. All objects derived from it must be destroyed first.
    #[inline]
    pub unsafe fn destroy_raw_shader_module(
        &self,
        shader_module: vk::ShaderModule,
    ) {
        // SAFETY: Caller guarantees shader_module provenance and drop ordering.
        unsafe { self.handle.destroy_shader_module(shader_module, None) };
    }
}

// Pipeline functionality
impl Device {
    /// # Safety
    /// `create_info` must be a valid pipeline layout create info. All
    /// referenced descriptor set layouts must be valid handles created from
    /// this device.
    #[inline]
    pub unsafe fn create_raw_pipeline_layout(
        &self,
        create_info: &vk::PipelineLayoutCreateInfo<'_>,
    ) -> Result<vk::PipelineLayout, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_pipeline_layout(create_info, None) }
    }

    /// # Safety
    /// `layout` must be a valid handle created from this device and not yet
    /// destroyed. No pipeline still using this layout may be in use.
    #[inline]
    pub unsafe fn destroy_raw_pipeline_layout(
        &self,
        layout: vk::PipelineLayout,
    ) {
        // SAFETY: Caller guarantees layout provenance and drop ordering.
        unsafe { self.handle.destroy_pipeline_layout(layout, None) };
    }

    /// Create a single graphics pipeline.
    ///
    /// On partial batch failure ash returns any successfully-created pipeline
    /// handles alongside the error; this wrapper destroys them so callers never
    /// receive a mix of valid and invalid handles.
    ///
    /// # Safety
    /// `create_info` must reference valid shader stages, a valid pipeline
    /// layout, and any pNext structures, all derived from this device. All
    /// referenced pointers must remain valid for the duration of the call.
    pub unsafe fn create_raw_graphics_pipeline(
        &self,
        create_info: &vk::GraphicsPipelineCreateInfo<'_>,
    ) -> Result<vk::Pipeline, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe {
            self.handle.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(create_info),
                None,
            )
        }
        .map_err(|(partial, result)| {
            // Destroy any handles that were successfully created before the
            // failure so the caller receives nothing on error.
            for p in partial {
                if p != vk::Pipeline::null() {
                    // SAFETY: p was just created by this device.
                    unsafe { self.handle.destroy_pipeline(p, None) };
                }
            }
            result
        })
        .map(|mut pipelines| {
            debug_assert_eq!(pipelines.len(), 1);
            pipelines.remove(0)
        })
    }

    /// # Safety
    /// `pipeline` must be a valid handle created from this device and not yet
    /// destroyed. No in-flight GPU work may still reference the pipeline.
    #[inline]
    pub unsafe fn destroy_raw_pipeline(&self, pipeline: vk::Pipeline) {
        // SAFETY: Caller guarantees pipeline provenance and drop ordering.
        unsafe { self.handle.destroy_pipeline(pipeline, None) };
    }
}

// Dynamic rendering functionality
impl Device {
    #[inline]
    pub fn has_dynamic_rendering(&self) -> bool {
        self.dynamic_rendering.is_some()
    }

    /// Begin a dynamic render pass on `command_buffer`.
    ///
    /// Dispatches to the Vulkan 1.3 core entry point or the
    /// `VK_KHR_dynamic_rendering` extension entry point depending on which was
    /// available at device creation.
    ///
    /// # Safety
    /// - `command_buffer` must be a valid handle in the recording state,
    ///   derived from this device.
    /// - `rendering_info` and all objects it references (image views, resolve
    ///   attachments, etc.) must be valid for the duration of the call and the
    ///   render pass.
    /// - All referenced images must be in the layout specified in
    ///   `rendering_info`.
    #[inline]
    pub unsafe fn cmd_begin_raw_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        rendering_info: &vk::RenderingInfo<'_>,
    ) {
        let dr = self
            .dynamic_rendering
            .as_ref()
            .expect("dynamic_rendering was not enabled in DeviceConfig");
        match dr {
            DynamicRenderingLoader::Core => {
                // SAFETY: Caller guarantees command_buffer and rendering_info
                // validity.
                unsafe {
                    self.handle
                        .cmd_begin_rendering(command_buffer, rendering_info)
                };
            }
            DynamicRenderingLoader::Extension(loader) => {
                // SAFETY: Caller guarantees command_buffer and rendering_info
                // validity.
                unsafe {
                    loader.cmd_begin_rendering(command_buffer, rendering_info)
                };
            }
        }
    }

    /// End the current dynamic render pass on `command_buffer`.
    ///
    /// # Safety
    /// - `command_buffer` must be a valid handle in the recording state,
    ///   derived from this device, and currently inside a render pass begun
    ///   with [`cmd_begin_raw_rendering`](Self::cmd_begin_raw_rendering).
    #[inline]
    pub unsafe fn cmd_end_raw_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
    ) {
        let dr = self
            .dynamic_rendering
            .as_ref()
            .expect("dynamic_rendering was not enabled in DeviceConfig");
        match dr {
            DynamicRenderingLoader::Core => {
                // SAFETY: Caller guarantees command_buffer validity and render
                // pass state.
                unsafe { self.handle.cmd_end_rendering(command_buffer) };
            }
            DynamicRenderingLoader::Extension(loader) => {
                // SAFETY: Caller guarantees command_buffer validity and render
                // pass state.
                unsafe { loader.cmd_end_rendering(command_buffer) };
            }
        }
    }
}

// Synchronization2
impl Device {
    /// Internal helper to submit to a queue using the synchronization2 API with
    /// an explicit raw fence handle.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    unsafe fn queue_submit2_raw_fence(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let sync2 = self
            .synchronization2
            .as_ref()
            .expect("synchronization2 was not enabled in DeviceConfig");
        match sync2 {
            Synchronization2Loader::Core => {
                // SAFETY: `queue` was obtained from this device and remains
                // valid while this call executes. `submits` and `fence` refer
                // to resources created for this device; calling the raw
                // `queue_submit2` function pointer is safe because the
                // Synchronization2 loader was initialized during device
                // creation and provides a valid entrypoint.
                unsafe { self.handle.queue_submit2(queue, submits, fence) }
            }
            Synchronization2Loader::Extension(loader) => {
                // SAFETY: `loader` contains a valid function pointer for
                // `queue_submit2` loaded when the device was created. `queue`,
                // `submits`, and `fence` are valid and derived from this
                // device, so invoking the extension entrypoint is safe.
                unsafe { loader.queue_submit2(queue, submits, fence) }
            }
        }
        .map_err(QueueSubmitError::SubmissionFailed)
    }

    /// Internal helper to submit to a queue using the synchronization2 API with
    /// an optional safe fence reference.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    unsafe fn _queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<Fence>,
    ) -> Result<(), QueueSubmitError> {
        if fence
            .as_ref()
            .map(|f| f.parent().raw_device() != self.raw_device())
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: All handles in submits are valid and derived from this
            // device by our own safety contract. Command buffers are in the
            // executable state by our own safety contract. Wait semaphores are
            // signaled by our own safety contract. raw_fence is known to have
            // been derived from this device and is in the unsignaled state.
            unsafe { self.queue_submit2_raw_fence(queue, submits, raw_fence) }
        }
    }
    /// Submit work to the graphics/present queue using the synchronization2
    /// API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    #[inline]
    pub unsafe fn graphics_queue_submit2_raw_fence(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.graphics_queue.lock();
        // Safety: `queue` was obtained from this device and remains valid while
        // this call executes. `submits` and `fence` refer to resources created
        // for this device; calling the raw `queue_submit2` function pointer is
        // safe because the Synchronization2 loader was initialized during
        // device creation and provides a valid entrypoint.
        unsafe { self.queue_submit2_raw_fence(*queue, submits, fence) }
    }

    /// Submit work to the graphics/present queue using the synchronization2
    /// API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn graphics_queue_submit2(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: All handles in submits are valid and derived from this
            // device by our own safety contract. Command buffers are in the
            // executable state by our own safety contract. Wait semaphores are
            // signaled by our own safety contract. raw_fence is known to have
            // been derived from this device and is in the unsignaled state.
            unsafe {
                self.graphics_queue_submit2_raw_fence(submits, raw_fence)
            }?;

            if let Some(f) = fence {
                // SAFETY: This fence has just been submitted to the queue via
                // Self::graphics_present_queue_submit2_raw_fence.
                _ = unsafe { f.mark_submitted() }
            }

            Ok(())
        }
    }

    /// Submit work to the transfer queue using the synchronization2 API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    #[inline]
    pub unsafe fn transfer_queue_submit2_raw_fence(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.transfer_queues.lock();
        // Safety: `queue` was obtained from this device and remains valid while
        // this call executes. `submits` and `fence` refer to resources created
        // for this device; calling the raw `queue_submit2` function pointer is
        // safe because the Synchronization2 loader was initialized during
        // device creation and provides a valid entrypoint.
        unsafe { self.queue_submit2_raw_fence(*queue, submits, fence) }
    }

    fn is_fence_mismatched(&self, fence: &sync::Fence) -> bool {
        fence.parent().raw_device() != self.raw_device()
    }

    /// Submit work to the transfer queue using the synchronization2 API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn transfer_queue_submit2(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: All handles in submits are valid and derived from this
            // device by our own safety contract. Command buffers are in the
            // executable state by our own safety contract. Wait semaphores are
            // signaled by our own safety contract. raw_fence is known to have
            // been derived from this device and is in the unsignaled state.
            unsafe {
                self.transfer_queue_submit2_raw_fence(submits, raw_fence)
            }?;

            if let Some(f) = fence {
                // SAFETY: This fence has just been submitted to the queue via
                // Self::transfer_queue_submit2_raw_fence.
                _ = unsafe { f.mark_submitted() }
            }

            Ok(())
        }
    }

    /// Submit work to the compute queue using the synchronization2 API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    #[inline]
    pub unsafe fn compute_queue_submit2_raw_fence(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.compute_queues.lock();
        // SAFETY: `queue` was obtained from this device and remains valid while
        // this call executes. `submits` and `fence` refer to resources created
        // for this device; calling the raw `queue_submit2` function pointer is
        // safe because the Synchronization2 loader was initialized during
        // device creation and provides a valid entrypoint.
        unsafe { self.queue_submit2_raw_fence(*queue, submits, fence) }
    }

    /// Submit work to the compute queue using the synchronization2 API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn compute_queue_submit2(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: All handles in submits are valid and derived from this
            // device by our own safety contract. Command buffers are in the
            // executable state by our own safety contract. Wait semaphores are
            // signaled by our own safety contract. raw_fence is known to have
            // been derived from this device and is in the unsignaled state.
            unsafe {
                self.compute_queue_submit2_raw_fence(submits, raw_fence)
            }?;

            if let Some(f) = fence {
                // SAFETY: This fence has just been submitted to the queue via
                // Self::compute_queue_submit2_raw_fence.
                _ = unsafe { f.mark_submitted() }
            }

            Ok(())
        }
    }

    // Synchronization2 — labeled submissions
    //
    // Each public submit2 function has three string variants that follow the
    // same convention used by the queue debug-label functions:
    //
    //   _labeled_cstr   – unsafe, takes Option<&CStr> (caller guarantees UTF-8)
    //   _labeled        – safe, takes Option<&str> _labeled_lazy   – safe,
    //   takes FnOnce() -> Option<StrRef: AsRef<str>>; the label closure is only
    //                     called when debug utils is enabled, so costly label
    //                     construction is free in release builds.
    //
    // Both the raw-fence tier and the safe-Fence tier get all three variants,
    // for each of the three queues (graphics, transfer, compute).

    /// Submit work to `queue`, wrapping it in a debug-label region.
    ///
    /// Locks the queue, begins the label, submits, then ends the label. Using
    /// the existing `within_*_queue_debug_label_cstr` helpers here would
    /// deadlock because they also take the queue lock.
    ///
    /// # Safety
    /// Same as [`Self::queue_submit2_raw_fence`]. `label` must contain only
    /// valid UTF-8 bytes (it is passed directly to
    /// `begin_queue_debug_label_cstr`).
    unsafe fn queue_submit2_raw_fence_labeled_cstr(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        // SAFETY: queue is derived from this device; label is valid UTF-8 per
        // our safety contract.
        unsafe { self.begin_queue_debug_label_cstr(queue, label) };
        // SAFETY: All submit safety requirements are delegated from our own
        // safety contract. queue is derived from this device.
        let result =
            unsafe { self.queue_submit2_raw_fence(queue, submits, fence) };
        // SAFETY: queue is derived from this device. We opened a label region
        // with begin_queue_debug_label_cstr above.
        unsafe { self.end_queue_debug_label(queue) };
        result
    }

    // ── Graphics queue — labeled ─────────────────────────────────────────

    /// Submit work to the graphics queue, wrapped in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn graphics_queue_submit2_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.graphics_queue.lock();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit2_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the graphics queue, wrapped in a debug-label region (safe
    /// `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn graphics_queue_submit2_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr was built from a &str, so it is valid UTF-8.
        unsafe {
            self.graphics_queue_submit2_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the graphics queue, wrapped in a debug-label region (lazy
    /// `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled; in release builds
    /// without the extension the closure is never invoked.
    pub fn graphics_queue_submit2_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8 (built from &str). All submit
            // safety requirements are on the caller.
            unsafe {
                self.graphics_queue_submit2_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.graphics_queue_submit2_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the graphics queue using the synchronization2 API,
    /// wrapped in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn graphics_queue_submit2_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.graphics_queue.lock();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit2_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the graphics queue.
                _ = unsafe { f.mark_submitted() }
            }
            Ok(())
        }
    }

    /// Submit work to the graphics queue using the synchronization2 API,
    /// wrapped in a debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn graphics_queue_submit2_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str). Other safety
        // requirements are propagated from caller.
        unsafe {
            self.graphics_queue_submit2_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the graphics queue using the synchronization2 API,
    /// wrapped in a debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn graphics_queue_submit2_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.graphics_queue_submit2_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.graphics_queue_submit2(submits, fence) }
        }
    }

    // ── Transfer queue — labeled ─────────────────────────────────────────

    /// Submit work to the transfer queue, wrapped in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn transfer_queue_submit2_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.transfer_queues.lock();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit2_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the transfer queue, wrapped in a debug-label region (safe
    /// `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn transfer_queue_submit2_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str).
        unsafe {
            self.transfer_queue_submit2_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the transfer queue, wrapped in a debug-label region (lazy
    /// `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn transfer_queue_submit2_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. Submit safety requirements are
            // on the caller.
            unsafe {
                self.transfer_queue_submit2_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.transfer_queue_submit2_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the transfer queue using the synchronization2 API,
    /// wrapped in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn transfer_queue_submit2_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.transfer_queues.lock();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit2_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the transfer queue.
                _ = unsafe { f.mark_submitted() }
            }
            Ok(())
        }
    }

    /// Submit work to the transfer queue using the synchronization2 API,
    /// wrapped in a debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn transfer_queue_submit2_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8. Other safety requirements
        // propagated from caller.
        unsafe {
            self.transfer_queue_submit2_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the transfer queue using the synchronization2 API,
    /// wrapped in a debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn transfer_queue_submit2_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.transfer_queue_submit2_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.transfer_queue_submit2(submits, fence) }
        }
    }

    // ── Compute queue — labeled ──────────────────────────────────────────

    /// Submit work to the compute queue, wrapped in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn compute_queue_submit2_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.compute_queues.lock();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit2_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the compute queue, wrapped in a debug-label region (safe
    /// `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn compute_queue_submit2_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str).
        unsafe {
            self.compute_queue_submit2_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the compute queue, wrapped in a debug-label region (lazy
    /// `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn compute_queue_submit2_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. Submit safety requirements are
            // on the caller.
            unsafe {
                self.compute_queue_submit2_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.compute_queue_submit2_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the compute queue using the synchronization2 API, wrapped
    /// in a debug-label region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn compute_queue_submit2_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.compute_queues.lock();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit2_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the compute queue.
                _ = unsafe { f.mark_submitted() }
            }
            Ok(())
        }
    }

    /// Submit work to the compute queue using the synchronization2 API, wrapped
    /// in a debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn compute_queue_submit2_labeled(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8. Other safety requirements
        // propagated from caller.
        unsafe {
            self.compute_queue_submit2_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the compute queue using the synchronization2 API, wrapped
    /// in a debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn compute_queue_submit2_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.compute_queue_submit2_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.compute_queue_submit2(submits, fence) }
        }
    }

    // ── Typed single-buffer submit (sync2) ───────────────────────────────
    //
    // These variants accept a single `Submittable<Q>` command buffer and build
    // `vk::SubmitInfo2` internally, so the compiler enforces that the buffer's
    // queue capability matches the target queue.
    //
    // Each queue role has three string variants mirroring the multi-submit
    // family: plain, _labeled (safe &str), _labeled_lazy (FnOnce closure).

    /// Submit a single typed command buffer to the graphics queue.
    ///
    /// # Safety
    /// `cmd` must be in the executable state. `wait` semaphores must be
    /// signaled. `signal` semaphores must be unsignaled. `fence`, when `Some`,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn graphics_queue_submit2_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Graphics>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.graphics_queue_submit2(std::slice::from_ref(&submit), fence)
        }
    }

    /// Submit a single typed command buffer to the graphics queue, wrapped in a
    /// debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// Same as [`graphics_queue_submit2_one`].
    pub unsafe fn graphics_queue_submit2_one_labeled<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Graphics>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.graphics_queue_submit2_labeled(
                std::slice::from_ref(&submit),
                fence,
                label,
            )
        }
    }

    /// Submit a single typed command buffer to the graphics queue, wrapped in a
    /// debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn graphics_queue_submit2_one_labeled_lazy<B, LabelFn, StrRef>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Graphics>,
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        self.graphics_queue_submit2_labeled_lazy(
            std::slice::from_ref(&submit),
            fence,
            label_fn,
        )
    }

    // ── Transfer ─────────────────────────────────────────────────────────

    /// Submit a single typed command buffer to the transfer queue.
    ///
    /// # Safety
    /// `cmd` must be in the executable state. `wait` semaphores must be
    /// signaled. `signal` semaphores must be unsignaled. `fence`, when `Some`,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn transfer_queue_submit2_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Transfer>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.transfer_queue_submit2(std::slice::from_ref(&submit), fence)
        }
    }

    /// Submit a single typed command buffer to the transfer queue, wrapped in a
    /// debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// Same as [`transfer_queue_submit2_one`].
    pub unsafe fn transfer_queue_submit2_one_labeled<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Transfer>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.transfer_queue_submit2_labeled(
                std::slice::from_ref(&submit),
                fence,
                label,
            )
        }
    }

    /// Submit a single typed command buffer to the transfer queue, wrapped in a
    /// debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn transfer_queue_submit2_one_labeled_lazy<B, LabelFn, StrRef>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Transfer>,
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        self.transfer_queue_submit2_labeled_lazy(
            std::slice::from_ref(&submit),
            fence,
            label_fn,
        )
    }

    // ── Compute ──────────────────────────────────────────────────────────

    /// Submit a single typed command buffer to the compute queue.
    ///
    /// # Safety
    /// `cmd` must be in the executable state. `wait` semaphores must be
    /// signaled. `signal` semaphores must be unsignaled. `fence`, when `Some`,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn compute_queue_submit2_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Compute>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.compute_queue_submit2(std::slice::from_ref(&submit), fence)
        }
    }

    /// Submit a single typed command buffer to the compute queue, wrapped in a
    /// debug-label region (safe `&str` variant).
    ///
    /// # Safety
    /// Same as [`compute_queue_submit2_one`].
    pub unsafe fn compute_queue_submit2_one_labeled<B>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Compute>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.compute_queue_submit2_labeled(
                std::slice::from_ref(&submit),
                fence,
                label,
            )
        }
    }

    /// Submit a single typed command buffer to the compute queue, wrapped in a
    /// debug-label region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn compute_queue_submit2_one_labeled_lazy<B, LabelFn, StrRef>(
        &self,
        cmd: &B,
        wait: &[vk::SemaphoreSubmitInfo<'_>],
        signal: &[vk::SemaphoreSubmitInfo<'_>],
        fence: Option<&mut sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Compute>,
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        let cb_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(cmd.raw());
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(signal);
        self.compute_queue_submit2_labeled_lazy(
            std::slice::from_ref(&submit),
            fence,
            label_fn,
        )
    }

    /// Record a pipeline barrier using the synchronization2 API.
    ///
    /// # Safety
    /// `command_buffer` must be a valid handle in the recording state, derived
    /// from this device. All handles and image layouts in `dependency_info`
    /// must be valid and consistent with the command buffer's current state.
    #[inline]
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfo<'_>,
    ) {
        let sync2 = self
            .synchronization2
            .as_ref()
            .expect("synchronization2 was not enabled in DeviceConfig");
        // SAFETY: Caller guarantees command_buffer and dependency_info
        // validity.
        match sync2 {
            // SAFETY: Caller guarantees command_buffer and dependency_info
            // validity.
            Synchronization2Loader::Core => unsafe {
                self.handle
                    .cmd_pipeline_barrier2(command_buffer, dependency_info)
            },
            // SAFETY: Caller guarantees command_buffer and dependency_info
            // validity.
            Synchronization2Loader::Extension(loader) => unsafe {
                loader.cmd_pipeline_barrier2(command_buffer, dependency_info)
            },
        }
    }

    // Removed: `cmd_pipeline_barrier2_raw` helper moved to the
    // `ResettableCommandBuffer` wrapper as requested.
}

// Recording commands
impl Device {
    /// Bind a graphics pipeline for subsequent draw commands.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. `pipeline` must be a
    /// valid graphics pipeline created from this device.
    #[inline]
    pub unsafe fn cmd_bind_graphics_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
    ) {
        // SAFETY: Caller guarantees command_buffer state and pipeline validity.
        unsafe {
            self.handle.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            )
        }
    }

    /// Bind vertex buffers for subsequent draw commands.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. `buffers` and `offsets`
    /// must have equal length. All buffers must be valid handles created from
    /// this device.
    #[inline]
    pub unsafe fn cmd_bind_vertex_buffers(
        &self,
        command_buffer: vk::CommandBuffer,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        // SAFETY: Caller guarantees command_buffer state and buffer/offset
        // validity.
        unsafe {
            self.handle.cmd_bind_vertex_buffers(
                command_buffer,
                first_binding,
                buffers,
                offsets,
            )
        }
    }

    /// Record a buffer-to-buffer copy.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. `src_buffer` and
    /// `dst_buffer` must be valid handles created from this device. Regions
    /// must be valid, non-overlapping within each buffer, and within bounds.
    #[inline]
    pub unsafe fn cmd_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        // SAFETY: Caller guarantees command buffer state and copy region
        // validity.
        unsafe {
            self.handle.cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                regions,
            )
        }
    }

    /// Record a buffer-to-image copy.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. `src_buffer` must be a
    /// valid `TRANSFER_SRC` buffer. `dst_image` must be a valid image in
    /// `dst_image_layout`. Regions must be valid and within bounds.
    #[inline]
    pub unsafe fn cmd_copy_buffer_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        // SAFETY: Caller guarantees command buffer state, handle provenance,
        // and region validity.
        unsafe {
            self.handle.cmd_copy_buffer_to_image(
                command_buffer,
                src_buffer,
                dst_image,
                dst_image_layout,
                regions,
            )
        }
    }

    /// Set the viewport dynamically.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state with a pipeline bound
    /// that declares `VK_DYNAMIC_STATE_VIEWPORT`.
    #[inline]
    pub unsafe fn cmd_set_viewport(
        &self,
        command_buffer: vk::CommandBuffer,
        viewports: &[vk::Viewport],
    ) {
        // SAFETY: Caller guarantees command_buffer state and pipeline dynamic
        // state.
        unsafe { self.handle.cmd_set_viewport(command_buffer, 0, viewports) }
    }

    /// Set the scissor rectangle dynamically.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state with a pipeline bound
    /// that declares `VK_DYNAMIC_STATE_SCISSOR`.
    #[inline]
    pub unsafe fn cmd_set_scissor(
        &self,
        command_buffer: vk::CommandBuffer,
        scissors: &[vk::Rect2D],
    ) {
        // SAFETY: Caller guarantees command_buffer state and pipeline dynamic
        // state.
        unsafe { self.handle.cmd_set_scissor(command_buffer, 0, scissors) }
    }

    /// Record a non-indexed draw call.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state inside an active render
    /// pass, with a compatible graphics pipeline bound and all required dynamic
    /// state set.
    #[inline]
    pub unsafe fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        // SAFETY: Caller guarantees render pass and pipeline state validity.
        unsafe {
            self.handle.cmd_draw(
                command_buffer,
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
    /// `command_buffer` must be in the recording state. `buffer` must be a
    /// valid index buffer created from this device, bound with `INDEX_BUFFER`
    /// usage.
    #[inline]
    pub unsafe fn cmd_bind_index_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        // SAFETY: Caller guarantees command_buffer state and buffer validity.
        unsafe {
            self.handle.cmd_bind_index_buffer(
                command_buffer,
                buffer,
                offset,
                index_type,
            )
        }
    }

    /// Record an indexed draw call.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state inside an active render
    /// pass, with a compatible graphics pipeline bound, all required dynamic
    /// state set, and a valid index buffer bound.
    #[inline]
    pub unsafe fn cmd_draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // SAFETY: Caller guarantees render pass, pipeline, and index buffer
        // state validity.
        unsafe {
            self.handle.cmd_draw_indexed(
                command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
    }
}

// Queue Acquisition
impl Device {
    fn acquire_queue(
        wrapped_queue: &Arc<Mutex<vk::Queue>>,
    ) -> MutexGuard<'_, vk::Queue> {
        wrapped_queue.lock()
    }

    fn acquire_graphics_queue(&self) -> MutexGuard<'_, vk::Queue> {
        Self::acquire_queue(&self.graphics_queue)
    }

    fn acquire_present_queue(&self) -> MutexGuard<'_, vk::Queue> {
        Self::acquire_queue(&self.present_queue)
    }

    fn acquire_transfer_queue(&self) -> MutexGuard<'_, vk::Queue> {
        Self::acquire_queue(&self.transfer_queues)
    }

    fn acquire_compute_queue(&self) -> MutexGuard<'_, vk::Queue> {
        Self::acquire_queue(&self.compute_queues)
    }
}

// Queue submission (VK 1.0 core)
impl Device {
    /// Internal helper to submit to a queue using the core Vulkan 1.0 API with
    /// an explicit raw fence handle.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    unsafe fn queue_submit_raw_fence(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        // SAFETY: Caller guarantees all handle validity and state.
        unsafe { self.handle.queue_submit(queue, submits, fence) }
            .map_err(QueueSubmitError::SubmissionFailed)
    }

    /// Internal helper to submit to the graphics queue using the core Vulkan
    /// 1.0 API with a raw fence
    ///
    /// # Safety
    /// Same as graphics_queue_submit, but we must also ensure the fence is from
    /// this device and valid
    pub unsafe fn graphics_queue_submit_raw_fence(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_graphics_queue();
        // SAFETY: Caller guarantees all handle validity and state. Queue is
        // locked
        unsafe { self.queue_submit_raw_fence(*queue, submits, fence) }
    }
    /// Submit work to the graphics/present queue using the core Vulkan 1.0
    /// `vkQueueSubmit` API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when `Some`,
    /// must be in the ready state (unsignaled, not pending).
    pub unsafe fn graphics_queue_submit(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: Caller guarantees all handle validity and state.
            unsafe {
                self.graphics_queue_submit_raw_fence(submits, raw_fence)
            }?;
            if let Some(f) = fence {
                // SAFETY: fence was just submitted above.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    /// Submit work to the transfer queue using the core Vulkan 1.0
    /// `vkQueueSubmit` API with a raw fence handle.
    ///
    /// # Safety
    /// Same as `graphics_queue_submit_raw_fence` but targets the transfer
    /// queue.
    pub unsafe fn transfer_queue_submit_raw_fence(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_transfer_queue();
        // SAFETY: Caller guarantees all handle validity and state. Queue is
        // locked for the duration of this call.
        unsafe { self.queue_submit_raw_fence(*queue, submits, fence) }
    }

    /// Submit work to the transfer queue using the core Vulkan 1.0
    /// `vkQueueSubmit` API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when `Some`,
    /// must be in the ready state (unsignaled, not pending).
    pub unsafe fn transfer_queue_submit(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: Caller guarantees all handle validity and state.
            unsafe {
                self.transfer_queue_submit_raw_fence(submits, raw_fence)
            }?;
            if let Some(f) = fence {
                // SAFETY: fence was just submitted above.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    /// Submit work to the compute queue using the core Vulkan 1.0
    /// `vkQueueSubmit` API with a raw fence handle.
    ///
    /// # Safety
    /// Same as `graphics_queue_submit_raw_fence` but targets the compute queue.
    pub unsafe fn compute_queue_submit_raw_fence(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_compute_queue();
        // SAFETY: Caller guarantees all handle validity and state. Queue is
        // locked for the duration of this call.
        unsafe { self.queue_submit_raw_fence(*queue, submits, fence) }
    }

    /// Submit work to the compute queue using the core Vulkan 1.0
    /// `vkQueueSubmit` API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when `Some`,
    /// must be in the ready state (unsignaled, not pending).
    pub unsafe fn compute_queue_submit(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            // SAFETY: Caller guarantees all handle validity and state.
            unsafe { self.compute_queue_submit_raw_fence(submits, raw_fence) }?;
            if let Some(f) = fence {
                // SAFETY: fence was just submitted above.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    // VK 1.0 — labeled submissions
    //
    // Mirrors the sync2 labeled-submission set but uses `SubmitInfo` and
    // `queue_submit`. All three string variants (CStr, &str, lazy &str) are
    // provided for every queue × tier combination.

    /// Begin a debug label, submit (VK 1.0), end the label on `queue`.
    ///
    /// # Safety
    /// Same as [`Self::queue_submit_raw_fence`]. `label` must be valid UTF-8.
    unsafe fn queue_submit_raw_fence_labeled_cstr(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        // SAFETY: queue is derived from this device; label is valid UTF-8 per
        // our safety contract.
        unsafe { self.begin_queue_debug_label_cstr(queue, label) };
        // SAFETY: All submit safety requirements are delegated from our own
        // safety contract. queue is derived from this device.
        let result =
            unsafe { self.queue_submit_raw_fence(queue, submits, fence) };
        // SAFETY: queue is derived from this device. We opened a label region
        // with begin_queue_debug_label_cstr above.
        unsafe { self.end_queue_debug_label(queue) };
        result
    }

    // ── Graphics queue — VK 1.0 labeled ─────────────────────────────────

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn graphics_queue_submit_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_graphics_queue();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn graphics_queue_submit_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str).
        unsafe {
            self.graphics_queue_submit_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn graphics_queue_submit_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. Submit safety requirements are
            // on the caller.
            unsafe {
                self.graphics_queue_submit_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.graphics_queue_submit_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn graphics_queue_submit_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.acquire_graphics_queue();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the graphics queue.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn graphics_queue_submit_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8. Other safety requirements
        // propagated from caller.
        unsafe {
            self.graphics_queue_submit_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the graphics queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn graphics_queue_submit_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.graphics_queue_submit_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.graphics_queue_submit(submits, fence) }
        }
    }

    // ── Transfer queue — VK 1.0 labeled ─────────────────────────────────

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn transfer_queue_submit_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_transfer_queue();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn transfer_queue_submit_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str).
        unsafe {
            self.transfer_queue_submit_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn transfer_queue_submit_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. Submit safety requirements are
            // on the caller.
            unsafe {
                self.transfer_queue_submit_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.transfer_queue_submit_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn transfer_queue_submit_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.acquire_transfer_queue();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the transfer queue.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn transfer_queue_submit_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8. Other safety requirements
        // propagated from caller.
        unsafe {
            self.transfer_queue_submit_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the transfer queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn transfer_queue_submit_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.transfer_queue_submit_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.transfer_queue_submit(submits, fence) }
        }
    }

    // ── Compute queue — VK 1.0 labeled ──────────────────────────────────

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device. `label` must
    /// contain only valid UTF-8 bytes.
    pub unsafe fn compute_queue_submit_raw_fence_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        let queue = self.acquire_compute_queue();
        // SAFETY: queue is derived from this device. submits, fence, and label
        // satisfy our safety contract.
        unsafe {
            self.queue_submit_raw_fence_labeled_cstr(
                *queue, submits, fence, label,
            )
        }
    }

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn compute_queue_submit_raw_fence_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8 (built from &str).
        unsafe {
            self.compute_queue_submit_raw_fence_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn compute_queue_submit_raw_fence_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: vk::Fence,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. Submit safety requirements are
            // on the caller.
            unsafe {
                self.compute_queue_submit_raw_fence_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.compute_queue_submit_raw_fence(submits, fence) }
        }
    }

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `label` must contain
    /// only valid UTF-8 bytes.
    pub unsafe fn compute_queue_submit_labeled_cstr(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&CStr>,
    ) -> Result<(), QueueSubmitError> {
        if !fence.as_ref().map(|f| f.is_ready()).unwrap_or(true) {
            Err(QueueSubmitError::FenceNotReady)
        } else if fence
            .as_ref()
            .map(|f| self.is_fence_mismatched(f))
            .unwrap_or(false)
        {
            Err(QueueSubmitError::MismatchedObjects)
        } else {
            let raw_fence = fence
                .as_ref()
                .map(|f| f.raw_fence())
                .unwrap_or(vk::Fence::null());
            let queue = self.acquire_compute_queue();
            // SAFETY: All submit safety requirements satisfied by our contract.
            // label is valid UTF-8 per our contract.
            unsafe {
                self.queue_submit_raw_fence_labeled_cstr(
                    *queue, submits, raw_fence, label,
                )
            }?;
            if let Some(f) = fence {
                // SAFETY: This fence was just submitted to the compute queue.
                _ = unsafe { f.mark_submitted() };
            }
            Ok(())
        }
    }

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region (safe `&str` variant).
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled.
    pub unsafe fn compute_queue_submit_labeled(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label: Option<&str>,
    ) -> Result<(), QueueSubmitError> {
        let label_cstr = label.map(|s| {
            CString::new(s).expect("Label must not contain null bytes")
        });
        // SAFETY: label_cstr is valid UTF-8. Other safety requirements
        // propagated from caller.
        unsafe {
            self.compute_queue_submit_labeled_cstr(
                submits,
                fence,
                label_cstr.as_deref(),
            )
        }
    }

    /// Submit work to the compute queue (VK 1.0), wrapped in a debug-label
    /// region (lazy `&str` variant).
    ///
    /// `label_fn` is only called when debug utils is enabled.
    pub fn compute_queue_submit_labeled_lazy<LabelFn, StrRef>(
        &self,
        submits: &[vk::SubmitInfo<'_>],
        fence: Option<&mut crate::sync::Fence>,
        label_fn: LabelFn,
    ) -> Result<(), QueueSubmitError>
    where
        LabelFn: FnOnce() -> Option<StrRef>,
        StrRef: AsRef<str>,
    {
        if self.debug_utils_device.is_some() {
            let label_ref = label_fn();
            let label_cstr = label_ref.as_ref().map(|s| {
                CString::new(s.as_ref())
                    .expect("Label must not contain null bytes")
            });
            // SAFETY: label_cstr is valid UTF-8. All submit safety requirements
            // are on the caller.
            unsafe {
                self.compute_queue_submit_labeled_cstr(
                    submits,
                    fence,
                    label_cstr.as_deref(),
                )
            }
        } else {
            // SAFETY: propagated from caller.
            unsafe { self.compute_queue_submit(submits, fence) }
        }
    }

    // ── Typed single-buffer submit (VK 1.0) ──────────────────────────────
    //
    // Mirrors the sync2 `*_submit2_one` family but uses `vkQueueSubmit`.
    // Semaphore handles are raw `vk::Semaphore` for now; a future
    // bumpalo-backed variant will accept wrapped handles.

    /// Submit a single typed command buffer to the graphics queue (VK 1.0).
    ///
    /// # Safety
    /// `cmd` must be in the executable state. Each semaphore in `wait` must be
    /// signaled and paired with a matching entry in `wait_stages`. Each
    /// semaphore in `signal` must be unsignaled. `fence`, when `Some`, must be
    /// an unsignaled fence created from this device.
    pub unsafe fn graphics_queue_submit_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal: &[vk::Semaphore],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Graphics>,
    {
        let cb = cmd.raw();
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(wait)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(std::slice::from_ref(&cb))
            .signal_semaphores(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.graphics_queue_submit(std::slice::from_ref(&submit), fence)
        }
    }

    /// Submit a single typed command buffer to the transfer queue (VK 1.0).
    ///
    /// # Safety
    /// `cmd` must be in the executable state. Each semaphore in `wait` must be
    /// signaled and paired with a matching entry in `wait_stages`. Each
    /// semaphore in `signal` must be unsignaled. `fence`, when `Some`, must be
    /// an unsignaled fence created from this device.
    pub unsafe fn transfer_queue_submit_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal: &[vk::Semaphore],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Transfer>,
    {
        let cb = cmd.raw();
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(wait)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(std::slice::from_ref(&cb))
            .signal_semaphores(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.transfer_queue_submit(std::slice::from_ref(&submit), fence)
        }
    }

    /// Submit a single typed command buffer to the compute queue (VK 1.0).
    ///
    /// # Safety
    /// `cmd` must be in the executable state. Each semaphore in `wait` must be
    /// signaled and paired with a matching entry in `wait_stages`. Each
    /// semaphore in `signal` must be unsignaled. `fence`, when `Some`, must be
    /// an unsignaled fence created from this device.
    pub unsafe fn compute_queue_submit_one<B>(
        &self,
        cmd: &B,
        wait: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal: &[vk::Semaphore],
        fence: Option<&mut sync::Fence>,
    ) -> Result<(), QueueSubmitError>
    where
        B: Submittable<Compute>,
    {
        let cb = cmd.raw();
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(wait)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(std::slice::from_ref(&cb))
            .signal_semaphores(signal);
        // SAFETY: propagated from caller.
        unsafe {
            self.compute_queue_submit(std::slice::from_ref(&submit), fence)
        }
    }
}

// Render pass recording (VK 1.0 core)
impl Device {
    /// Record an old-style pipeline barrier (`vkCmdPipelineBarrier`).
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. All handles and image
    /// layouts in the barrier arrays must be valid and consistent with the
    /// command buffer's current state.
    // Allow extra arguments: this signature mirrors the Vulkan
    // `vkCmdPipelineBarrier` parameter groups and is kept in sync with the raw
    // API for clarity.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub unsafe fn cmd_pipeline_barrier(
        &self,
        command_buffer: vk::CommandBuffer,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_memory_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) {
        // SAFETY: Caller guarantees command_buffer state and barrier validity.
        unsafe {
            self.handle.cmd_pipeline_barrier(
                command_buffer,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            )
        }
    }

    /// Begin a render pass (`vkCmdBeginRenderPass`).
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. All objects referenced
    /// by `render_pass_begin` must be valid and derived from this device. The
    /// framebuffer's attachments must be in the layouts declared in the render
    /// pass attachment descriptions, or `UNDEFINED` when `initial_layout` is
    /// `UNDEFINED`.
    #[inline]
    pub unsafe fn cmd_begin_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        render_pass_begin: &vk::RenderPassBeginInfo<'_>,
        contents: vk::SubpassContents,
    ) {
        // SAFETY: Caller guarantees command_buffer state and render_pass_begin
        // validity.
        unsafe {
            self.handle.cmd_begin_render_pass(
                command_buffer,
                render_pass_begin,
                contents,
            )
        }
    }

    /// End the current render pass (`vkCmdEndRenderPass`).
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state inside a render pass
    /// begun with [`cmd_begin_render_pass`](Self::cmd_begin_render_pass).
    #[inline]
    pub unsafe fn cmd_end_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
    ) {
        // SAFETY: Caller guarantees active render pass state.
        unsafe { self.handle.cmd_end_render_pass(command_buffer) }
    }
}

// Buffer and memory functionality
impl Device {
    /// # Safety
    /// `create_info` must be valid and reference only objects derived from this
    /// device. All referenced pointers must remain valid for the duration of
    /// the call.
    #[inline]
    pub unsafe fn create_raw_buffer(
        &self,
        create_info: &vk::BufferCreateInfo<'_>,
    ) -> Result<vk::Buffer, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_buffer(create_info, None) }
    }

    /// # Safety
    /// `buffer` must be a valid handle created from this device and not yet
    /// destroyed. No in-flight GPU work may still reference `buffer`.
    #[inline]
    pub unsafe fn destroy_raw_buffer(&self, buffer: vk::Buffer) {
        // SAFETY: Caller guarantees buffer provenance and drop ordering.
        unsafe { self.handle.destroy_buffer(buffer, None) };
    }
}

// Command pool functionality
impl Device {
    /// # Safety
    /// `create_info` must have a valid `queue_family_index` for this device.
    /// All referenced pointers must remain valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_command_pool(
        &self,
        create_info: &vk::CommandPoolCreateInfo<'_>,
    ) -> Result<vk::CommandPool, vk::Result> {
        // SAFETY: Caller guarantees create_info validity and queue family
        // provenance.
        unsafe { self.handle.create_command_pool(create_info, None) }
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device and not yet
    /// destroyed. All command buffers allocated from it must have finished
    /// execution and must not be referenced by any pending GPU work.
    #[inline]
    pub unsafe fn destroy_raw_command_pool(&self, pool: vk::CommandPool) {
        // SAFETY: Caller guarantees pool provenance and drop ordering.
        unsafe { self.handle.destroy_command_pool(pool, None) };
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device. All command
    /// buffers allocated from it must not be pending execution on the GPU.
    #[inline]
    pub unsafe fn reset_raw_command_pool(
        &self,
        pool: vk::CommandPool,
        flags: vk::CommandPoolResetFlags,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees pool provenance and command buffer idle
        // state.
        unsafe { self.handle.reset_command_pool(pool, flags) }
    }

    /// # Safety
    /// `allocate_info.command_pool` must be a valid pool created from this
    /// device. `command_buffer_count` must be non-zero.
    #[inline]
    pub unsafe fn allocate_raw_command_buffers(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        // SAFETY: Caller guarantees allocate_info validity and pool provenance.
        unsafe { self.handle.allocate_command_buffers(allocate_info) }
    }

    /// # Safety
    /// `command_buffer` must be in the initial or executable state and must not
    /// be pending execution. All pointers in `begin_info` must remain valid for
    /// the duration of the call.
    #[inline]
    pub unsafe fn begin_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfo<'_>,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer state and begin_info
        // validity.
        unsafe { self.handle.begin_command_buffer(command_buffer, begin_info) }
    }

    /// # Safety
    /// `command_buffer` must be in the recording state.
    #[inline]
    pub unsafe fn end_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer is in the recording state.
        unsafe { self.handle.end_command_buffer(command_buffer) }
    }

    /// # Safety
    /// `command_buffer` must not be pending execution on the GPU. The pool it
    /// was allocated from must have been created with `RESET_COMMAND_BUFFER`.
    #[inline]
    pub unsafe fn reset_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferResetFlags,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer is not pending and pool flag
        // is set.
        unsafe { self.handle.reset_command_buffer(command_buffer, flags) }
    }

    /// Free command buffers back to their source pool, returning memory to the
    /// pool's internal allocator.
    ///
    /// A no-op when `command_buffers` is empty.
    ///
    /// # Safety
    /// - All handles in `command_buffers` must have been allocated from `pool`.
    /// - No buffer in `command_buffers` may be pending execution on the GPU.
    /// - The caller must externally synchronize access to `pool` (e.g. by
    ///   ensuring no other thread is allocating or resetting from it
    ///   concurrently).
    #[inline]
    pub unsafe fn free_raw_command_buffers(
        &self,
        pool: vk::CommandPool,
        command_buffers: &[vk::CommandBuffer],
    ) {
        if command_buffers.is_empty() {
            return;
        }
        // SAFETY: Caller guarantees pool/buffer provenance, idle state, and
        // external synchronization on pool.
        unsafe { self.handle.free_command_buffers(pool, command_buffers) }
    }
}

// Fence and semaphore functionality
impl Device {
    /// # Safety
    /// `create_info` must be a valid fence create info. All referenced pointers
    /// must remain valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_fence(
        &self,
        create_info: &vk::FenceCreateInfo<'_>,
    ) -> Result<vk::Fence, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_fence(create_info, None) }
    }

    /// # Safety
    /// `fence` must be a valid handle created from this device and not yet
    /// destroyed. No GPU work may reference this fence at time of destruction.
    #[inline]
    pub unsafe fn destroy_raw_fence(&self, fence: vk::Fence) {
        // SAFETY: Caller guarantees fence provenance and drop ordering.
        unsafe { self.handle.destroy_fence(fence, None) };
    }

    /// # Safety
    /// All handles in `fences` must be valid fences created from this device.
    #[inline]
    pub unsafe fn wait_for_raw_fences(
        &self,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout_ns: u64,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees fence handle validity.
        unsafe { self.handle.wait_for_fences(fences, wait_all, timeout_ns) }
    }

    /// # Safety
    /// All handles in `fences` must be valid fences created from this device
    /// and must not be currently pending on any queue submission.
    #[inline]
    pub unsafe fn reset_raw_fences(
        &self,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees fence handle validity and non-pending
        // state.
        unsafe { self.handle.reset_fences(fences) }
    }

    /// Query whether a fence is signaled.
    ///
    /// Returns `Ok(true)` if signaled, `Ok(false)` if not yet signaled.
    ///
    /// # Safety
    /// `fence` must be a valid handle created from this device and not yet
    /// destroyed.
    #[inline]
    pub unsafe fn get_raw_fence_status(
        &self,
        fence: vk::Fence,
    ) -> Result<bool, vk::Result> {
        // SAFETY: Caller guarantees fence provenance and validity.
        unsafe { self.handle.get_fence_status(fence) }
    }

    /// # Safety
    /// `create_info` must be a valid semaphore create info. All referenced
    /// pointers must remain valid for the duration of the call.
    #[inline]
    pub unsafe fn create_raw_semaphore(
        &self,
        create_info: &vk::SemaphoreCreateInfo<'_>,
    ) -> Result<vk::Semaphore, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_semaphore(create_info, None) }
    }

    /// # Safety
    /// `semaphore` must be a valid handle created from this device and not yet
    /// destroyed. No GPU work may be waiting on or about to signal it.
    #[inline]
    pub unsafe fn destroy_raw_semaphore(&self, semaphore: vk::Semaphore) {
        // SAFETY: Caller guarantees semaphore provenance and drop ordering.
        unsafe { self.handle.destroy_semaphore(semaphore, None) };
    }
}

// Descriptor set functionality
impl Device {
    /// # Safety
    /// `create_info` must be valid and reference only objects derived from this
    /// device.
    #[inline]
    pub unsafe fn create_raw_descriptor_set_layout(
        &self,
        create_info: &vk::DescriptorSetLayoutCreateInfo<'_>,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_descriptor_set_layout(create_info, None) }
    }

    /// # Safety
    /// `layout` must be a valid handle created from this device and not yet
    /// destroyed. No descriptor pool that used this layout may still exist.
    #[inline]
    pub unsafe fn destroy_raw_descriptor_set_layout(
        &self,
        layout: vk::DescriptorSetLayout,
    ) {
        // SAFETY: Caller guarantees layout provenance and ordering.
        unsafe { self.handle.destroy_descriptor_set_layout(layout, None) };
    }

    /// # Safety
    /// `create_info` must be valid and reference only objects derived from this
    /// device.
    #[inline]
    pub unsafe fn create_raw_descriptor_pool(
        &self,
        create_info: &vk::DescriptorPoolCreateInfo<'_>,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_descriptor_pool(create_info, None) }
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device and not yet
    /// destroyed. All descriptor sets allocated from it must not be referenced
    /// by any pending GPU work.
    #[inline]
    pub unsafe fn destroy_raw_descriptor_pool(&self, pool: vk::DescriptorPool) {
        // SAFETY: Caller guarantees pool provenance and ordering.
        unsafe { self.handle.destroy_descriptor_pool(pool, None) };
    }

    /// # Safety
    /// `alloc_info.descriptor_pool` must be a valid pool created from this
    /// device with sufficient capacity. All layouts in `alloc_info` must be
    /// valid handles derived from this device.
    #[inline]
    pub unsafe fn allocate_raw_descriptor_sets(
        &self,
        alloc_info: &vk::DescriptorSetAllocateInfo<'_>,
    ) -> Result<Vec<vk::DescriptorSet>, vk::Result> {
        // SAFETY: Caller guarantees alloc_info validity.
        unsafe { self.handle.allocate_descriptor_sets(alloc_info) }
    }

    /// Write or copy descriptor set updates.
    ///
    /// # Safety
    /// All handles in `descriptor_writes` and `descriptor_copies` must be valid
    /// and derived from this device. Buffer and image references in
    /// `descriptor_writes` must remain valid for as long as the descriptor set
    /// is bound in a submitted command buffer.
    #[inline]
    pub unsafe fn update_raw_descriptor_sets(
        &self,
        descriptor_writes: &[vk::WriteDescriptorSet<'_>],
        descriptor_copies: &[vk::CopyDescriptorSet<'_>],
    ) {
        // SAFETY: Caller guarantees write/copy validity.
        unsafe {
            self.handle
                .update_descriptor_sets(descriptor_writes, descriptor_copies)
        }
    }

    /// Bind descriptor sets for subsequent draw/dispatch commands.
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - `layout` must be compatible with the pipeline to be used.
    /// - All handles in `descriptor_sets` must be valid and derived from this
    ///   device.
    /// - `dynamic_offsets` must match the number of dynamic descriptors in the
    ///   bound sets.
    #[inline]
    pub unsafe fn cmd_bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        // SAFETY: Caller guarantees command buffer state, layout compatibility,
        // and descriptor set validity.
        unsafe {
            self.handle.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            )
        }
    }

    /// Record a `vkCmdPushConstants` command.
    ///
    /// # Safety
    /// - `command_buffer` must be in the recording state.
    /// - `layout` must be compatible with the pipeline that will be used for
    ///   drawing.
    /// - `stage_flags` and `offset` must match a push constant range declared
    ///   in `layout`.
    /// - `values` length must not exceed the range size.
    #[inline]
    pub unsafe fn cmd_push_constants(
        &self,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        values: &[u8],
    ) {
        // SAFETY: Caller guarantees recording state, layout compatibility,
        // stage_flags match, and range bounds.
        unsafe {
            self.handle.cmd_push_constants(
                command_buffer,
                layout,
                stage_flags,
                offset,
                values,
            )
        }
    }
}

impl From<FetchPhysicalDeviceError> for CreateCompatibleError {
    fn from(value: FetchPhysicalDeviceError) -> Self {
        match value {
            FetchPhysicalDeviceError::MemoryExhaustion => {
                Self::MemoryExhaustion
            }
            FetchPhysicalDeviceError::UnknownVulkan(e) => {
                Self::UnknownVulkan(e)
            }
        }
    }
}
