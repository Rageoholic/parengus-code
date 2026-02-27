//! Logical device wrapper ([`Device`]).
//!
//! `Device` wraps a `VkDevice` and centralises all per-device state:
//! a `gpu-allocator` allocator (behind a `Mutex`), extension loaders
//! for swapchain, dynamic rendering, synchronization2, and debug utils,
//! plus the graphics/present queue and its family index.
//!
//! Physical device selection uses a priority-based fold: discrete GPUs
//! outrank integrated GPUs, and only devices that satisfy all required
//! extensions and queue families are considered.
//! [`Device::create_compatible`] wraps this selection and returns the
//! highest-priority match.
//!
//! All raw Vulkan operations on the device handle are surfaced as
//! `unsafe fn` methods prefixed with `raw_` (e.g. `create_raw_buffer`).
//! Higher-level wrappers in sibling modules call these rather than
//! accessing `ash::Device` directly.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::{
    AllocationError, MemoryLocation,
    vulkan::{
        Allocation, AllocationCreateDesc, AllocationScheme, Allocator,
        AllocatorCreateDesc,
    },
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use crate::{
    instance::{FetchPhysicalDeviceError, Instance},
    surface::{Surface, SurfaceSupportError},
    swapchain::CreateSwapchainError,
};

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
/// Passed to [`Device::allocate_memory`] to select the best-matching
/// Vulkan memory type and determine whether atom-size padding is
/// required for non-coherent flush alignment.
#[derive(Copy, Clone, Debug)]
pub enum MemoryUsage {
    /// GPU-only storage. Highest bandwidth; not CPU-mappable.
    GpuOnly,
    /// CPU-writable, GPU-readable. For staging buffers and
    /// per-frame uploads.
    CpuToGpu,
    /// GPU-writable, CPU-readable. For readback.
    GpuToCpu,
}

/// A logical Vulkan device and its associated per-device state.
///
/// Wraps an `ash::Device`, a `gpu-allocator` allocator (behind a
/// `Mutex`), extension loaders for swapchain / dynamic rendering /
/// synchronization2 / debug utils, and the graphics+present queue.
///
/// Constructed via [`Device::create_compatible`], which selects the
/// best physical device by priority (discrete > integrated). Raw
/// Vulkan operations are exposed as `unsafe fn` methods prefixed
/// with `raw_`.
#[allow(dead_code)]
pub struct Device {
    parent: Arc<Instance>,
    allocator: Option<Mutex<Allocator>>,
    handle: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    properties: vk::PhysicalDeviceProperties,
    swapchain_device: Option<ash::khr::swapchain::Device>,
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
    dynamic_rendering: Option<DynamicRenderingLoader>,
    synchronization2: Synchronization2Loader,
    physical_device: vk::PhysicalDevice,
    memory_budget: bool,
    /// Aliased queues share the same `Arc<Mutex<vk::Queue>>` so that locking
    /// either role serializes on the same underlying resource.
    graphics_present_queue: (Arc<Mutex<vk::Queue>>, u32),
    transfer_queue: (Arc<Mutex<vk::Queue>>, u32),
    compute_queue: (Arc<Mutex<vk::Queue>>, u32),
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
        //SAFETY: All objects derived from this device should be dropped
        //before this device is dropped.
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

    #[error("Failed to create logical device: {0}")]
    DeviceCreationFailed(vk::Result),

    #[error("Error checking surface support: {0}")]
    SurfaceSupport(#[from] SurfaceSupportError),

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

    #[error("Failed to create GPU allocator: {0}")]
    AllocatorCreation(AllocationError),
}

#[derive(Debug, Error)]
pub enum DynamicRenderingError {
    #[error("Dynamic rendering is not enabled on this device")]
    NotEnabled,
}

#[derive(Debug, Error)]
pub enum NameObjectError {
    #[error("Invalid Vulkan object name (contains interior NUL): {0}")]
    InvalidName(std::ffi::NulError),

    #[error("Vulkan error setting object name: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueMode {
    /// Use dedicated transfer and compute queue families when available.
    #[default]
    Auto,
    /// Force all queue types to the same queue family.
    Unified,
    /// Force all queue types onto a single queue from one family.
    Single,
}

#[derive(Debug, Default)]
pub struct DeviceConfig {
    pub swapchain: bool,
    pub dynamic_rendering: bool,
    pub queue_mode: QueueMode,
}

impl Device {
    /// Create a logical device compatible with `surf`.
    ///
    /// Selects the highest-priority physical device that satisfies all
    /// requirements in `config` and can present to `surf`.
    ///
    /// The name `create_compatible` is intentional: the API does not yet
    /// expose physical devices as a first-class concept, so callers
    /// cannot select one themselves. This name signals that the
    /// selection is automatic and may change in a future API revision
    /// once physical-device enumeration is surfaced.
    pub fn create_compatible<T: HasDisplayHandle + HasWindowHandle>(
        instance: &Arc<Instance>,
        surf: &Surface<T>,
        config: DeviceConfig,
    ) -> Result<Self, CreateCompatibleError> {
        if !std::sync::Arc::ptr_eq(surf.parent(), instance) {
            return Err(CreateCompatibleError::MismatchedParams);
        }

        // Evaluate every physical device, filtering out those that
        // lack required extensions or a graphics+present queue, then
        // score the survivors so we can pick the best.
        //
        // Score: (dedicated_queue_count, device_type_priority)
        // compared lexicographically — dedicated queues matter most,
        // then device type breaks ties.
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
            graphics_present_family: u32,
            score: (u32, u32),
            /// True when sync2 must use the extension loader.
            use_sync2_ext: bool,
            /// True when dynamic rendering must use the extension loader.
            use_dr_ext: bool,
            /// True when VK_KHR_shader_non_semantic_info should be
            /// enabled (available on this pre-1.3 device).
            enable_shader_non_semantic: bool,
            /// True when VK_EXT_memory_budget is supported and
            /// should be enabled.
            enable_memory_budget: bool,
        }

        let mut candidates: Vec<DeviceCandidate> = Vec::new();

        'dev: for &dev in &physical_devices {
            // SAFETY: dev was derived from instance.
            let props =
                unsafe { instance.get_raw_physical_device_properties(dev) };
            // SAFETY: dev was derived from instance.
            let queue_families = unsafe {
                instance.get_raw_physical_device_queue_family_properties(dev)
            };

            // Use the device's own reported API version so that
            // per-device capability differences are handled correctly
            // rather than relying on the single instance-level version.
            let dev_api =
                crate::instance::VkVersion::from_raw(props.api_version);
            let is_pre_1_3 = dev_api.major() < 1
                || (dev_api.major() == 1 && dev_api.minor() < 3);

            // VK_KHR_swapchain is never promoted to core; always check
            // it when requested. Other extensions are only extensions on
            // pre-1.3 devices.
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

            // VK_KHR_synchronization2: core in 1.3; required extension
            // on older devices — hard filter.
            let use_sync2_ext = if is_pre_1_3 {
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

            // VK_KHR_shader_non_semantic_info: core in 1.3; optional
            // on older devices.
            let enable_shader_non_semantic =
                is_pre_1_3 && has_ext(ash::khr::shader_non_semantic_info::NAME);

            // VK_EXT_memory_budget: optional device extension.
            // Enables accurate heap-usage/budget queries via
            // vkGetPhysicalDeviceMemoryProperties2.
            let enable_memory_budget =
                has_ext(ash::ext::memory_budget::NAME);

            // VK_KHR_dynamic_rendering: core in 1.3; required extension
            // on older devices when dynamic rendering is requested —
            // hard filter.
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

            // Find a queue family that supports both graphics and
            // presentation — hard filter.
            let Some(graphics_present_family) =
                queue_families.iter().enumerate().find_map(|(idx, qf)| {
                    if !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        return None;
                    }
                    // SAFETY: dev and surf are both derived from the
                    // same instance (validated at the top of this fn).
                    let ok =
                        unsafe { surf.supports_queue_family(dev, idx as u32) };
                    match ok {
                        Ok(true) => Some(idx as u32),
                        _ => None,
                    }
                })
            else {
                tracing::debug!(
                    "Skipping {:?}: no graphics+present queue family",
                    props.device_name_as_c_str().unwrap_or(c"unknown"),
                );
                continue 'dev;
            };

            let has_dedicated_transfer = queue_families.iter().any(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });
            let has_dedicated_compute = queue_families.iter().any(|qf| {
                qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            });

            let dedicated_count =
                has_dedicated_transfer as u32 + has_dedicated_compute as u32;
            let score =
                (dedicated_count, device_type_priority(props.device_type));

            candidates.push(DeviceCandidate {
                handle: dev,
                props,
                queue_families,
                graphics_present_family,
                score,
                use_sync2_ext,
                use_dr_ext,
                enable_shader_non_semantic,
                enable_memory_budget,
            });
        }

        let best = candidates
            .iter()
            .max_by_key(|c| c.score)
            .ok_or(CreateCompatibleError::NoSuitableDevice)?;

        let physical_device = best.handle;
        let queue_families = &best.queue_families;
        let graphics_present_family = best.graphics_present_family;
        let use_sync2_ext = best.use_sync2_ext;
        let use_dr_ext = best.use_dr_ext;
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

        // Find dedicated transfer and compute queue families
        let (transfer_family, compute_family) = if matches!(
            config.queue_mode,
            QueueMode::Unified | QueueMode::Single
        ) {
            (graphics_present_family, graphics_present_family)
        } else {
            let dedicated_transfer = queue_families
                .iter()
                .enumerate()
                .find_map(|(idx, props)| {
                    if props.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    {
                        Some(idx as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(graphics_present_family);

            let dedicated_compute = queue_families
                .iter()
                .enumerate()
                .find_map(|(idx, props)| {
                    if props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    {
                        Some(idx as u32)
                    } else {
                        None
                    }
                })
                .unwrap_or(graphics_present_family);

            (dedicated_transfer, dedicated_compute)
        };

        tracing::info!(
            "Queue families — graphics+present: {}, \
             transfer: {}, compute: {}",
            graphics_present_family,
            transfer_family,
            compute_family
        );

        // Build deduplicated queue create infos
        let mut family_queue_counts: HashMap<u32, u32> = HashMap::new();
        for family in [graphics_present_family, transfer_family, compute_family]
        {
            *family_queue_counts.entry(family).or_insert(0) += 1;
        }

        // Clamp to available queue count per family, or to 1 if forced
        for (&family, count) in &mut family_queue_counts {
            if config.queue_mode == QueueMode::Single {
                *count = 1;
            } else {
                let available = queue_families[family as usize].queue_count;
                if *count > available {
                    *count = available;
                }
            }
        }

        let queue_priorities_storage: Vec<Vec<f32>> = family_queue_counts
            .iter()
            .map(|(_, &count)| vec![1.0; count as usize])
            .collect();

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'_>> =
            family_queue_counts
                .iter()
                .zip(queue_priorities_storage.iter())
                .map(|((&family, _), priorities)| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family)
                        .queue_priorities(priorities)
                })
                .collect();

        // Build extension list from the selected candidate's flags.
        let mut mandatory_exts: Vec<&CStr> = Vec::with_capacity(4);
        if config.swapchain {
            mandatory_exts.push(ash::khr::swapchain::NAME);
        }
        if best.enable_shader_non_semantic {
            mandatory_exts.push(ash::khr::shader_non_semantic_info::NAME);
        }
        if best.enable_memory_budget {
            mandatory_exts.push(ash::ext::memory_budget::NAME);
        }
        if use_sync2_ext {
            mandatory_exts.push(ash::khr::synchronization2::NAME);
        }
        if use_dr_ext {
            mandatory_exts.push(ash::khr::dynamic_rendering::NAME);
        }

        let ext_ptrs: Vec<*const i8> =
            mandatory_exts.iter().map(|e| e.as_ptr()).collect();

        // Enable synchronization2 (core 1.3 or via extension).
        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default()
                .synchronization2(true);
        // Enable dynamic rendering if requested (core 1.3 or via
        // extension).
        let mut dr_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default()
                .dynamic_rendering(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut sync2_features);
        if config.dynamic_rendering {
            device_create_info = device_create_info.push_next(&mut dr_features);
        }

        // SAFETY: physical_device was derived from instance;
        // device_create_info is fully initialised above.
        let device = unsafe {
            instance.create_ash_device(physical_device, &device_create_info)
        }
        .map_err(CreateCompatibleError::DeviceCreationFailed)?;

        // Get queues. For families with multiple queues, assign
        // incrementing indices. For families where we requested more
        // queues than available, reuse index 0.
        let mut family_next_index: HashMap<u32, u32> = HashMap::new();
        let mut get_next_queue = |family: u32| -> vk::Queue {
            let idx = family_next_index.entry(family).or_insert(0);
            let max = family_queue_counts[&family];
            let queue_idx = if *idx < max { *idx } else { 0 };
            *idx += 1;
            // SAFETY: device was just created with this family/index.
            unsafe { device.get_device_queue(family, queue_idx) }
        };

        let graphics_present_queue_handle =
            get_next_queue(graphics_present_family);
        let transfer_queue_handle = get_next_queue(transfer_family);
        let compute_queue_handle = get_next_queue(compute_family);

        // Aliased queues (same underlying VkQueue handle) must share a
        // single Mutex so that locking any role serializes on the same
        // resource.
        let gfx_queue_arc = Arc::new(Mutex::new(graphics_present_queue_handle));
        let transfer_queue_arc =
            if transfer_queue_handle == graphics_present_queue_handle {
                Arc::clone(&gfx_queue_arc)
            } else {
                Arc::new(Mutex::new(transfer_queue_handle))
            };
        let compute_queue_arc =
            if compute_queue_handle == graphics_present_queue_handle {
                Arc::clone(&gfx_queue_arc)
            } else if compute_queue_handle == transfer_queue_handle {
                Arc::clone(&transfer_queue_arc)
            } else {
                Arc::new(Mutex::new(compute_queue_handle))
            };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.ash_instance().clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(CreateCompatibleError::AllocatorCreation)?;

        Ok(Self {
            parent: instance.clone(),
            allocator: Some(Mutex::new(allocator)),
            memory_properties,
            properties: best.props,
            swapchain_device: if config.swapchain {
                Some(instance.create_swapchain_loader(&device))
            } else {
                None
            },
            debug_utils_device: instance
                .create_debug_utils_device_loader(&device),
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
            synchronization2: if use_sync2_ext {
                Synchronization2Loader::Extension(
                    instance.create_synchronization2_loader(&device),
                )
            } else {
                Synchronization2Loader::Core
            },
            handle: device,
            physical_device,
            memory_budget: best.enable_memory_budget,
            graphics_present_queue: (gfx_queue_arc, graphics_present_family),
            transfer_queue: (transfer_queue_arc, transfer_family),
            compute_queue: (compute_queue_arc, compute_family),
        })
    }

    pub fn parent(&self) -> &Arc<Instance> {
        &self.parent
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns `true` if `VK_EXT_memory_budget` was enabled on this
    /// device. When true, callers may chain
    /// `vk::PhysicalDeviceMemoryBudgetPropertiesEXT` into
    /// `vkGetPhysicalDeviceMemoryProperties2` to obtain accurate
    /// per-heap usage and budget figures.
    pub fn has_memory_budget(&self) -> bool {
        self.memory_budget
    }

    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    pub fn non_coherent_atom_size(&self) -> vk::DeviceSize {
        self.properties.limits.non_coherent_atom_size
    }

    /// Score a memory type for a given usage; returns `None` if the
    /// type is incompatible.  Higher scores are more preferred.
    fn score_memory_type(
        flags: vk::MemoryPropertyFlags,
        usage: MemoryUsage,
    ) -> Option<u32> {
        use vk::MemoryPropertyFlags as F;
        let device_local = flags.contains(F::DEVICE_LOCAL);
        let host_visible = flags.contains(F::HOST_VISIBLE);
        let host_cached = flags.contains(F::HOST_CACHED);
        match usage {
            MemoryUsage::GpuOnly => {
                // Prefer pure VRAM; penalise HOST_VISIBLE (unified).
                device_local.then_some(if host_visible { 1 } else { 2 })
            }
            MemoryUsage::CpuToGpu => {
                // Prefer DEVICE_LOCAL (ReBAR / unified memory).
                host_visible.then_some(if device_local { 2 } else { 1 })
            }
            MemoryUsage::GpuToCpu => {
                // Prefer HOST_CACHED for efficient CPU reads.
                host_visible.then_some(if host_cached { 2 } else { 1 })
            }
        }
    }

    /// Select the best Vulkan memory type index for `requirements`
    /// and `usage`.  Among types with equal score the lowest index
    /// wins, matching Vulkan's convention that earlier types in the
    /// list are more preferred within the same heap.
    fn select_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        usage: MemoryUsage,
    ) -> Option<u32> {
        self.memory_properties.memory_types
            [..self.memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .filter(|(i, _)| requirements.memory_type_bits & (1 << i) != 0)
            .filter_map(|(i, ty)| {
                Self::score_memory_type(ty.property_flags, usage)
                    .map(|s| (i as u32, s))
            })
            .max_by(|(i1, s1), (i2, s2)| s1.cmp(s2).then(i2.cmp(i1)))
            .map(|(i, _)| i)
    }

    /// Allocate device memory for the given requirements.
    ///
    /// Selects the best-matching Vulkan memory type for `usage`,
    /// narrows `requirements.memory_type_bits` to that type, then
    /// rounds `size` and `alignment` up to
    /// `VkPhysicalDeviceLimits::nonCoherentAtomSize` only when the
    /// chosen type is HOST_VISIBLE but not HOST_COHERENT.
    pub fn allocate_memory(
        &self,
        name: &str,
        requirements: vk::MemoryRequirements,
        usage: MemoryUsage,
        linear: bool,
    ) -> Result<Allocation, AllocationError> {
        let atom = self.properties.limits.non_coherent_atom_size;
        let requirements =
            if let Some(idx) = self.select_memory_type(requirements, usage) {
                use vk::MemoryPropertyFlags as F;
                let flags = self.memory_properties.memory_types[idx as usize]
                    .property_flags;
                let non_coherent_visible = flags.contains(F::HOST_VISIBLE)
                    && !flags.contains(F::HOST_COHERENT);
                let (size, alignment) = if non_coherent_visible {
                    (
                        requirements.size.div_ceil(atom) * atom,
                        requirements.alignment.max(atom),
                    )
                } else {
                    (requirements.size, requirements.alignment)
                };
                vk::MemoryRequirements {
                    size,
                    alignment,
                    memory_type_bits: 1 << idx,
                }
            } else {
                requirements
            };
        let location = match usage {
            MemoryUsage::GpuOnly => MemoryLocation::GpuOnly,
            MemoryUsage::CpuToGpu => MemoryLocation::CpuToGpu,
            MemoryUsage::GpuToCpu => MemoryLocation::GpuToCpu,
        };
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop")
            .lock()
            .expect("allocator lock poisoned");
        let mut allocator = allocator;
        allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
    }

    pub fn free_memory(
        &self,
        allocation: Allocation,
    ) -> Result<(), AllocationError> {
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is dropped only during Device::drop")
            .lock()
            .expect("allocator lock poisoned");
        let mut allocator = allocator;
        allocator.free(allocation)
    }

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

    pub fn raw_device(&self) -> vk::Device {
        self.handle.handle()
    }

    pub fn graphics_present_queue_family(&self) -> u32 {
        self.graphics_present_queue.1
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
    pub unsafe fn create_raw_swapchain(
        &self,
        create_info: &vk::SwapchainCreateInfoKHR<'_>,
    ) -> Result<vk::SwapchainKHR, CreateSwapchainError> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .ok_or(CreateSwapchainError::SwapchainNotEnabled)?;
        // SAFETY: Caller guarantees create_info validity and handle provenance.
        unsafe { swapchain_device.create_swapchain(create_info, None) }
            .map_err(CreateSwapchainError::VulkanCreate)
    }

    /// # Safety
    /// `swapchain` must be a valid swapchain handle created from this device
    /// and not yet destroyed.
    pub unsafe fn get_raw_swapchain_images(
        &self,
        swapchain: vk::SwapchainKHR,
    ) -> Result<Vec<vk::Image>, CreateSwapchainError> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .ok_or(CreateSwapchainError::SwapchainNotEnabled)?;
        // SAFETY: Caller guarantees swapchain validity and lifetime.
        unsafe { swapchain_device.get_swapchain_images(swapchain) }
            .map_err(CreateSwapchainError::VulkanGetImages)
    }

    /// # Safety
    /// `swapchain` must be a valid handle derived from this device, and all
    /// child resources derived from it must be destroyed first.
    ///
    /// No in-flight GPU work may still reference the swapchain.
    pub unsafe fn destroy_raw_swapchain(&self, swapchain: vk::SwapchainKHR) {
        if let Some(swapchain_device) = self.swapchain_device.as_ref() {
            // SAFETY: Caller guarantees swapchain provenance and drop ordering.
            unsafe { swapchain_device.destroy_swapchain(swapchain, None) };
        }
    }

    /// # Safety
    /// `create_info` must reference valid Vulkan objects derived from this
    /// device. Any referenced pointers must remain valid for the duration of
    /// the call.
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
    pub unsafe fn destroy_raw_image_view(&self, image_view: vk::ImageView) {
        // SAFETY: Caller guarantees image_view provenance and drop ordering.
        unsafe { self.handle.destroy_image_view(image_view, None) };
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
    /// `swapchain` must be a valid handle created from this device.
    /// `semaphore` and `fence`, when not null, must be valid unsignaled handles
    /// created from this device.
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
            .ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
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
    /// `VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` and not referenced by any pending
    /// GPU work other than this presentation.
    pub unsafe fn queue_present(
        &self,
        present_info: &vk::PresentInfoKHR<'_>,
    ) -> Result<bool, vk::Result> {
        let swapchain_device = self
            .swapchain_device
            .as_ref()
            .ok_or(vk::Result::ERROR_EXTENSION_NOT_PRESENT)?;
        let queue = self
            .graphics_present_queue
            .0
            .lock()
            .expect("graphics/present queue lock poisoned");
        // SAFETY: Caller guarantees all handles and synchronization
        // requirements.
        unsafe { swapchain_device.queue_present(*queue, present_info) }
    }

    pub fn has_swapchain_support(&self) -> bool {
        self.swapchain_device.is_some()
    }
}

// Debug naming functionality
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
    /// The closure is only called if `VK_EXT_debug_utils` is enabled.
    /// Returning `None` from the closure is treated as a no-op.
    ///
    /// # Safety
    /// `object` must be a valid Vulkan handle created from this device (or a
    /// child object associated with this device) and must remain valid for the
    /// duration of the call.
    pub unsafe fn set_object_name_with<H, F>(
        &self,
        object: H,
        name_provider: F,
    ) -> Result<(), NameObjectError>
    where
        H: vk::Handle,
        F: FnOnce() -> Option<CString>,
    {
        if self.debug_utils_device.is_none() {
            return Ok(());
        }

        let name = name_provider();
        // SAFETY: This method shares the same safety contract as
        // set_object_name.
        unsafe { self.set_object_name(object, name.as_deref()) }
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
}

// Shader module functionality
impl Device {
    /// # Safety
    /// `create_info` must contain valid SPIR-V code. All referenced pointers
    /// must remain valid for the duration of the call.
    pub unsafe fn create_raw_shader_module(
        &self,
        create_info: &vk::ShaderModuleCreateInfo<'_>,
    ) -> Result<vk::ShaderModule, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_shader_module(create_info, None) }
    }

    /// # Safety
    /// `shader_module` must be a valid handle created from this device and
    /// not yet destroyed. All objects derived from it must be destroyed first.
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
    /// handles alongside the error; this wrapper destroys them so callers
    /// never receive a mix of valid and invalid handles.
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
    pub unsafe fn destroy_raw_pipeline(&self, pipeline: vk::Pipeline) {
        // SAFETY: Caller guarantees pipeline provenance and drop ordering.
        unsafe { self.handle.destroy_pipeline(pipeline, None) };
    }
}

// Dynamic rendering functionality
impl Device {
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
    ///   attachments, etc.) must be valid for the duration of the call and
    ///   the render pass.
    /// - All referenced images must be in the layout specified in
    ///   `rendering_info`.
    pub unsafe fn cmd_begin_raw_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        rendering_info: &vk::RenderingInfo<'_>,
    ) -> Result<(), DynamicRenderingError> {
        match &self.dynamic_rendering {
            None => Err(DynamicRenderingError::NotEnabled),
            Some(DynamicRenderingLoader::Core) => {
                // SAFETY: Caller guarantees command_buffer and
                // rendering_info validity.
                unsafe {
                    self.handle
                        .cmd_begin_rendering(command_buffer, rendering_info)
                };
                Ok(())
            }
            Some(DynamicRenderingLoader::Extension(loader)) => {
                // SAFETY: Caller guarantees command_buffer and
                // rendering_info validity.
                unsafe {
                    loader.cmd_begin_rendering(command_buffer, rendering_info)
                };
                Ok(())
            }
        }
    }

    /// End the current dynamic render pass on `command_buffer`.
    ///
    /// # Safety
    /// - `command_buffer` must be a valid handle in the recording state,
    ///   derived from this device, and currently inside a render pass begun
    ///   with [`cmd_begin_raw_rendering`](Self::cmd_begin_raw_rendering).
    pub unsafe fn cmd_end_raw_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), DynamicRenderingError> {
        match &self.dynamic_rendering {
            None => Err(DynamicRenderingError::NotEnabled),
            Some(DynamicRenderingLoader::Core) => {
                // SAFETY: Caller guarantees command_buffer validity
                // and render pass state.
                unsafe { self.handle.cmd_end_rendering(command_buffer) };
                Ok(())
            }
            Some(DynamicRenderingLoader::Extension(loader)) => {
                // SAFETY: Caller guarantees command_buffer validity
                // and render pass state.
                unsafe { loader.cmd_end_rendering(command_buffer) };
                Ok(())
            }
        }
    }
}

// Queue submit functionality
impl Device {
    /// Submit work to the graphics/present queue using the
    /// synchronization2 API.
    ///
    /// # Safety
    /// All handles in `submits` must be valid and derived from this device.
    /// Command buffers must be in the executable state. Wait semaphores must be
    /// signaled. Signal semaphores must be unsignaled. `fence`, when not null,
    /// must be an unsignaled fence created from this device.
    pub unsafe fn graphics_present_queue_submit2(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        let queue = self
            .graphics_present_queue
            .0
            .lock()
            .expect("graphics/present queue lock poisoned");
        match &self.synchronization2 {
            // SAFETY: Caller guarantees all handle validity and
            // synchronization state.
            Synchronization2Loader::Core => unsafe {
                self.handle.queue_submit2(*queue, submits, fence)
            },
            // SAFETY: Caller guarantees all handle validity and
            // synchronization state.
            Synchronization2Loader::Extension(loader) => unsafe {
                loader.queue_submit2(*queue, submits, fence)
            },
        }
    }
}

// Recording commands
impl Device {
    /// Record a pipeline barrier using the synchronization2 API.
    ///
    /// # Safety
    /// `command_buffer` must be a valid handle in the recording state, derived
    /// from this device. All handles and image layouts in `dependency_info`
    /// must be valid and consistent with the command buffer's current state.
    pub unsafe fn cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfo<'_>,
    ) {
        // SAFETY: Caller guarantees command_buffer and
        // dependency_info validity.
        match &self.synchronization2 {
            // SAFETY: Caller guarantees command_buffer and
            // dependency_info validity.
            Synchronization2Loader::Core => unsafe {
                self.handle
                    .cmd_pipeline_barrier2(command_buffer, dependency_info)
            },
            // SAFETY: Caller guarantees command_buffer and
            // dependency_info validity.
            Synchronization2Loader::Extension(loader) => unsafe {
                loader.cmd_pipeline_barrier2(command_buffer, dependency_info)
            },
        }
    }

    /// Bind a graphics pipeline for subsequent draw commands.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state. `pipeline` must be a
    /// valid graphics pipeline created from this device.
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
    /// `command_buffer` must be in the recording state. `buffers` and
    /// `offsets` must have equal length. All buffers must be valid handles
    /// created from this device.
    pub unsafe fn cmd_bind_vertex_buffers(
        &self,
        command_buffer: vk::CommandBuffer,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        // SAFETY: Caller guarantees command_buffer state and
        // buffer/offset validity.
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

    /// Set the viewport dynamically.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state with a pipeline bound
    /// that declares `VK_DYNAMIC_STATE_VIEWPORT`.
    pub unsafe fn cmd_set_viewport(
        &self,
        command_buffer: vk::CommandBuffer,
        viewports: &[vk::Viewport],
    ) {
        // SAFETY: Caller guarantees command_buffer state and pipeline
        // dynamic state.
        unsafe { self.handle.cmd_set_viewport(command_buffer, 0, viewports) }
    }

    /// Set the scissor rectangle dynamically.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state with a pipeline bound
    /// that declares `VK_DYNAMIC_STATE_SCISSOR`.
    pub unsafe fn cmd_set_scissor(
        &self,
        command_buffer: vk::CommandBuffer,
        scissors: &[vk::Rect2D],
    ) {
        // SAFETY: Caller guarantees command_buffer state and pipeline
        // dynamic state.
        unsafe { self.handle.cmd_set_scissor(command_buffer, 0, scissors) }
    }

    /// Record a non-indexed draw call.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state inside an active render
    /// pass, with a compatible graphics pipeline bound and all required dynamic
    /// state set.
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
    /// valid index buffer created from this device, bound with
    /// `INDEX_BUFFER` usage.
    pub unsafe fn cmd_bind_index_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        // SAFETY: Caller guarantees command_buffer state and
        // buffer validity.
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
    /// `command_buffer` must be in the recording state inside an active
    /// render pass, with a compatible graphics pipeline bound, all
    /// required dynamic state set, and a valid index buffer bound.
    pub unsafe fn cmd_draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // SAFETY: Caller guarantees render pass, pipeline, and
        // index buffer state validity.
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

// Buffer and memory functionality
impl Device {
    /// # Safety
    /// `create_info` must be valid and reference only objects derived from
    /// this device. All referenced pointers must remain valid for the
    /// duration of the call.
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
    pub unsafe fn destroy_raw_buffer(&self, buffer: vk::Buffer) {
        // SAFETY: Caller guarantees buffer provenance and drop ordering.
        unsafe { self.handle.destroy_buffer(buffer, None) };
    }

    /// Query memory requirements for a buffer.
    ///
    /// # Safety
    /// `buffer` must be a valid handle created from this device.
    pub unsafe fn get_raw_buffer_memory_requirements(
        &self,
        buffer: vk::Buffer,
    ) -> vk::MemoryRequirements {
        // SAFETY: Caller guarantees buffer validity.
        unsafe { self.handle.get_buffer_memory_requirements(buffer) }
    }

    /// # Safety
    /// `allocate_info` must be valid and describe a memory type index
    /// supported by this device.
    pub unsafe fn allocate_raw_memory(
        &self,
        allocate_info: &vk::MemoryAllocateInfo<'_>,
    ) -> Result<vk::DeviceMemory, vk::Result> {
        // SAFETY: Caller guarantees allocation info validity.
        unsafe { self.handle.allocate_memory(allocate_info, None) }
    }

    /// # Safety
    /// `memory` must be a valid handle created from this device and not yet
    /// freed. No object may still be bound to `memory` at free time.
    pub unsafe fn free_raw_memory(&self, memory: vk::DeviceMemory) {
        // SAFETY: Caller guarantees memory provenance and drop ordering.
        unsafe { self.handle.free_memory(memory, None) };
    }

    /// # Safety
    /// `buffer` and `memory` must both be valid handles created from this
    /// device. `offset` must satisfy alignment/size requirements from
    /// `vkGetBufferMemoryRequirements`.
    pub unsafe fn bind_raw_buffer_memory(
        &self,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees handle validity and offset constraints.
        unsafe { self.handle.bind_buffer_memory(buffer, memory, offset) }
    }

    /// # Safety
    /// `memory` must be a valid allocation from this device. The mapped range
    /// (`offset`, `size`) must be within the allocation and obey host access
    /// synchronization requirements.
    pub unsafe fn map_raw_memory(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> Result<*mut std::ffi::c_void, vk::Result> {
        // SAFETY: Caller guarantees mapping preconditions.
        unsafe { self.handle.map_memory(memory, offset, size, flags) }
    }

    /// # Safety
    /// Every range in `memory_ranges` must reference memory allocations from
    /// this device and satisfy Vulkan flush requirements.
    pub unsafe fn flush_raw_mapped_memory_ranges(
        &self,
        memory_ranges: &[vk::MappedMemoryRange<'_>],
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees memory range validity.
        unsafe { self.handle.flush_mapped_memory_ranges(memory_ranges) }
    }

    /// # Safety
    /// `memory` must currently be mapped on this device.
    pub unsafe fn unmap_raw_memory(&self, memory: vk::DeviceMemory) {
        // SAFETY: Caller guarantees memory is currently mapped.
        unsafe { self.handle.unmap_memory(memory) };
    }
}

// Command pool functionality
impl Device {
    /// # Safety
    /// `create_info` must have a valid `queue_family_index` for this device.
    /// All referenced pointers must remain valid for the duration of the call.
    pub unsafe fn create_raw_command_pool(
        &self,
        create_info: &vk::CommandPoolCreateInfo<'_>,
    ) -> Result<vk::CommandPool, vk::Result> {
        // SAFETY: Caller guarantees create_info validity and queue
        // family provenance.
        unsafe { self.handle.create_command_pool(create_info, None) }
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device and not yet
    /// destroyed. All command buffers allocated from it must have finished
    /// execution and must not be referenced by any pending GPU work.
    pub unsafe fn destroy_raw_command_pool(&self, pool: vk::CommandPool) {
        // SAFETY: Caller guarantees pool provenance and drop ordering.
        unsafe { self.handle.destroy_command_pool(pool, None) };
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device. All command
    /// buffers allocated from it must not be pending execution on the GPU.
    pub unsafe fn reset_raw_command_pool(
        &self,
        pool: vk::CommandPool,
        flags: vk::CommandPoolResetFlags,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees pool provenance and command
        // buffer idle state.
        unsafe { self.handle.reset_command_pool(pool, flags) }
    }

    /// # Safety
    /// `allocate_info.command_pool` must be a valid pool created from this
    /// device. `command_buffer_count` must be non-zero.
    pub unsafe fn allocate_raw_command_buffers(
        &self,
        allocate_info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
        // SAFETY: Caller guarantees allocate_info validity and pool provenance.
        unsafe { self.handle.allocate_command_buffers(allocate_info) }
    }

    /// # Safety
    /// `command_buffer` must be in the initial or executable state and must
    /// not be pending execution. All pointers in `begin_info` must remain
    /// valid for the duration of the call.
    pub unsafe fn begin_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfo<'_>,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer state and
        // begin_info validity.
        unsafe { self.handle.begin_command_buffer(command_buffer, begin_info) }
    }

    /// # Safety
    /// `command_buffer` must be in the recording state.
    pub unsafe fn end_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer is in the recording state.
        unsafe { self.handle.end_command_buffer(command_buffer) }
    }

    /// # Safety
    /// `command_buffer` must not be pending execution on the GPU. The pool it
    /// was allocated from must have been created with
    /// `RESET_COMMAND_BUFFER`.
    pub unsafe fn reset_raw_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        flags: vk::CommandBufferResetFlags,
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees command_buffer is not pending
        // and pool flag is set.
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
    pub unsafe fn destroy_raw_fence(&self, fence: vk::Fence) {
        // SAFETY: Caller guarantees fence provenance and drop ordering.
        unsafe { self.handle.destroy_fence(fence, None) };
    }

    /// # Safety
    /// All handles in `fences` must be valid fences created from this device.
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
    pub unsafe fn reset_raw_fences(
        &self,
        fences: &[vk::Fence],
    ) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees fence handle validity and
        // non-pending state.
        unsafe { self.handle.reset_fences(fences) }
    }

    /// Query whether a fence is signaled.
    ///
    /// Returns `Ok(true)` if signaled, `Ok(false)` if not yet signaled.
    ///
    /// # Safety
    /// `fence` must be a valid handle created from this device and not yet
    /// destroyed.
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
    pub unsafe fn destroy_raw_semaphore(&self, semaphore: vk::Semaphore) {
        // SAFETY: Caller guarantees semaphore provenance and drop ordering.
        unsafe { self.handle.destroy_semaphore(semaphore, None) };
    }
}

// Descriptor set functionality
impl Device {
    /// # Safety
    /// `create_info` must be valid and reference only objects
    /// derived from this device.
    pub unsafe fn create_raw_descriptor_set_layout(
        &self,
        create_info: &vk::DescriptorSetLayoutCreateInfo<'_>,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_descriptor_set_layout(create_info, None) }
    }

    /// # Safety
    /// `layout` must be a valid handle created from this device
    /// and not yet destroyed. No descriptor pool that used this
    /// layout may still exist.
    pub unsafe fn destroy_raw_descriptor_set_layout(
        &self,
        layout: vk::DescriptorSetLayout,
    ) {
        // SAFETY: Caller guarantees layout provenance and ordering.
        unsafe { self.handle.destroy_descriptor_set_layout(layout, None) };
    }

    /// # Safety
    /// `create_info` must be valid and reference only objects
    /// derived from this device.
    pub unsafe fn create_raw_descriptor_pool(
        &self,
        create_info: &vk::DescriptorPoolCreateInfo<'_>,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        // SAFETY: Caller guarantees create_info validity.
        unsafe { self.handle.create_descriptor_pool(create_info, None) }
    }

    /// # Safety
    /// `pool` must be a valid handle created from this device and
    /// not yet destroyed. All descriptor sets allocated from it
    /// must not be referenced by any pending GPU work.
    pub unsafe fn destroy_raw_descriptor_pool(&self, pool: vk::DescriptorPool) {
        // SAFETY: Caller guarantees pool provenance and ordering.
        unsafe { self.handle.destroy_descriptor_pool(pool, None) };
    }

    /// # Safety
    /// `alloc_info.descriptor_pool` must be a valid pool created
    /// from this device with sufficient capacity. All layouts in
    /// `alloc_info` must be valid handles derived from this device.
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
    /// All handles in `descriptor_writes` and `descriptor_copies`
    /// must be valid and derived from this device. Buffer and image
    /// references in `descriptor_writes` must remain valid for as
    /// long as the descriptor set is bound in a submitted command
    /// buffer.
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
    /// - All handles in `descriptor_sets` must be valid and derived
    ///   from this device.
    /// - `dynamic_offsets` must match the number of dynamic
    ///   descriptors in the bound sets.
    pub unsafe fn cmd_bind_descriptor_sets(
        &self,
        command_buffer: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        // SAFETY: Caller guarantees command buffer state, layout
        // compatibility, and descriptor set validity.
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
