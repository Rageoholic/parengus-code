use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
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

#[allow(dead_code)]
pub struct Device {
    parent: Arc<Instance>,
    handle: ash::Device,
    swapchain_device: Option<ash::khr::swapchain::Device>,
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
    dynamic_rendering: Option<DynamicRenderingLoader>,
    swapchain_name_counter: AtomicU64,
    physical_device: vk::PhysicalDevice,
    graphics_present_queue: (vk::Queue, u32),
    transfer_queue: (vk::Queue, u32),
    compute_queue: (vk::Queue, u32),
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
}

#[derive(Debug, Error)]
pub enum DynamicRenderingError {
    #[error("Dynamic rendering is not enabled on this device")]
    NotEnabled,
}

#[derive(Debug, Error)]
pub enum NameObjectError {
    #[error("Debug utils extension is not enabled on this device")]
    DebugUtilsNotEnabled,

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
    pub fn create_compatible<T: HasDisplayHandle + HasWindowHandle>(
        instance: &Arc<Instance>,
        surf: &Surface<T>,
        config: DeviceConfig,
    ) -> Result<Self, CreateCompatibleError> {
        if !std::sync::Arc::ptr_eq(surf.get_parent(), instance) {
            return Err(CreateCompatibleError::MismatchedParams);
        }

        let mut mandatory_exts: Vec<&CStr> = Vec::with_capacity(2);
        if config.swapchain {
            mandatory_exts.push(ash::khr::swapchain::NAME);
        }

        // Select best physical device.
        // Gather properties and queue families upfront so we only query once
        // per device.
        // Score: (dedicated_queue_count, device_type_priority) — compared
        // lexicographically so dedicated queues matter most, then device type
        // breaks ties.
        let physical_devices = instance.fetch_physical_devices()?;
        let device_type_priority = |dt: vk::PhysicalDeviceType| -> u32 {
            match dt {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            }
        };

        struct DeviceInfo {
            handle: vk::PhysicalDevice,
            props: vk::PhysicalDeviceProperties,
            queue_families: Vec<vk::QueueFamilyProperties>,
            score: (u32, u32),
        }

        let device_infos: Vec<DeviceInfo> = physical_devices
            .iter()
            .map(|&dev| {
                //SAFETY: dev was derived from instance
                let props = unsafe { instance.get_raw_physical_device_properties(dev) };
                //SAFETY: dev was derived from instance
                let queue_families =
                    unsafe { instance.get_raw_physical_device_queue_family_properties(dev) };

                let has_dedicated_transfer = queue_families.iter().any(|qf| {
                    qf.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                });
                let has_dedicated_compute = queue_families.iter().any(|qf| {
                    qf.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !qf.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                });

                let dedicated_count = has_dedicated_transfer as u32 + has_dedicated_compute as u32;
                let score = (dedicated_count, device_type_priority(props.device_type));

                DeviceInfo {
                    handle: dev,
                    props,
                    queue_families,
                    score,
                }
            })
            .collect();

        let best = device_infos
            .iter()
            .max_by_key(|info| info.score)
            .ok_or(CreateCompatibleError::NoSuitableDevice)?;

        let physical_device = best.handle;
        let queue_families = &best.queue_families;
        tracing::info!(
            "Selected physical device: {:?} (type: {:?}, dedicated queues: {})",
            best.props.device_name_as_c_str().unwrap_or(c"unknown"),
            best.props.device_type,
            best.score.0,
        );

        // Find graphics+present queue family
        let graphics_present_family = queue_families
            .iter()
            .enumerate()
            .find_map(|(idx, props)| {
                if !props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    return None;
                }
                //SAFETY: physical_device was derived from instance, surface
                //was derived from the same instance
                let supports_present =
                    unsafe { surf.supports_queue_family(physical_device, idx as u32) };
                match supports_present {
                    Ok(true) => Some(idx as u32),
                    _ => None,
                }
            })
            .ok_or(CreateCompatibleError::NoGraphicsPresentQueue)?;

        // Find dedicated transfer and compute queue families
        let (transfer_family, compute_family) =
            if matches!(config.queue_mode, QueueMode::Unified | QueueMode::Single) {
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
            "Queue families — graphics+present: {}, transfer: {}, compute: {}",
            graphics_present_family,
            transfer_family,
            compute_family
        );

        // Build deduplicated queue create infos
        // Count how many queues we need from each family
        let mut family_queue_counts: HashMap<u32, u32> = HashMap::new();
        for family in [graphics_present_family, transfer_family, compute_family] {
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

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = family_queue_counts
            .iter()
            .zip(queue_priorities_storage.iter())
            .map(|((&family, _), priorities)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(priorities)
            })
            .collect();

        let ver = instance.get_supported_ver();
        let is_pre_1_3 = ver.major() < 1 || (ver.major() == 1 && ver.minor() < 3);

        // Enumerate device extensions once for all pre-1.3 optional checks.
        let device_exts: Vec<vk::ExtensionProperties> = if is_pre_1_3 {
            //SAFETY: physical_device was derived from instance
            unsafe { instance.enumerate_raw_device_extension_properties(physical_device) }
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // VK_KHR_shader_non_semantic_info: promoted to core in 1.3.
        if is_pre_1_3 {
            let has_non_semantic = device_exts.iter().any(|ext| {
                ext.extension_name_as_c_str() == Ok(ash::khr::shader_non_semantic_info::NAME)
            });
            if has_non_semantic {
                mandatory_exts.push(ash::khr::shader_non_semantic_info::NAME);
            }
        }

        // VK_KHR_dynamic_rendering: core in 1.3, extension on older drivers.
        // `use_dr_ext` is true only when the extension loader must be used.
        let use_dr_ext = if config.dynamic_rendering && is_pre_1_3 {
            let has_dr = device_exts
                .iter()
                .any(|ext| ext.extension_name_as_c_str() == Ok(ash::khr::dynamic_rendering::NAME));
            if has_dr {
                mandatory_exts.push(ash::khr::dynamic_rendering::NAME);
                true
            } else {
                return Err(CreateCompatibleError::DynamicRenderingNotAvailable);
            }
        } else {
            false
        };

        let ext_ptrs: Vec<*const i8> = mandatory_exts.iter().map(|e| e.as_ptr()).collect();

        // Enable synchronization2
        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        // Enable dynamic rendering if requested (core 1.3 or via extension).
        let mut dr_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut sync2_features);
        if config.dynamic_rendering {
            device_create_info = device_create_info.push_next(&mut dr_features);
        }

        //SAFETY: physical_device was derived from instance, device_create_info
        //is valid
        let device = unsafe { instance.create_ash_device(physical_device, &device_create_info) }
            .map_err(CreateCompatibleError::DeviceCreationFailed)?;

        // Get queues. For families with multiple queues, assign incrementing
        // indices. For families where we requested more queues than available,
        // reuse index 0.
        let mut family_next_index: HashMap<u32, u32> = HashMap::new();
        let mut get_next_queue = |family: u32| -> vk::Queue {
            let idx = family_next_index.entry(family).or_insert(0);
            let max = family_queue_counts[&family];
            let queue_idx = if *idx < max { *idx } else { 0 };
            *idx += 1;
            //SAFETY: device was just created with this queue family/index
            unsafe { device.get_device_queue(family, queue_idx) }
        };

        let graphics_present_queue_handle = get_next_queue(graphics_present_family);
        let transfer_queue_handle = get_next_queue(transfer_family);
        let compute_queue_handle = get_next_queue(compute_family);

        Ok(Self {
            parent: instance.clone(),
            swapchain_device: if config.swapchain {
                Some(instance.create_swapchain_loader(&device))
            } else {
                None
            },
            debug_utils_device: instance.create_debug_utils_device_loader(&device),
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
            swapchain_name_counter: AtomicU64::new(0),
            handle: device,
            physical_device,
            graphics_present_queue: (graphics_present_queue_handle, graphics_present_family),
            transfer_queue: (transfer_queue_handle, transfer_family),
            compute_queue: (compute_queue_handle, compute_family),
        })
    }

    pub fn get_parent(&self) -> &Arc<Instance> {
        &self.parent
    }

    pub fn get_physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn ash_handle(&self) -> &ash::Device {
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

    pub fn raw_handle(&self) -> vk::Device {
        self.handle.handle()
    }

    pub fn graphics_present_queue_family(&self) -> u32 {
        self.graphics_present_queue.1
    }
}

//Swapchain functionality
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

    pub fn has_swapchain_support(&self) -> bool {
        self.swapchain_device.is_some()
    }

    pub(crate) fn next_swapchain_debug_index(&self) -> u64 {
        self.swapchain_name_counter.fetch_add(1, Ordering::Relaxed) + 1
    }
}

//Debug naming functionality
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
        let debug_utils = self
            .debug_utils_device
            .as_ref()
            .ok_or(NameObjectError::DebugUtilsNotEnabled)?;

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
            return Err(NameObjectError::DebugUtilsNotEnabled);
        }

        let name = name_provider();
        // SAFETY: This method shares the same safety contract as set_object_name.
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
            Some(name) => Some(CString::new(name).map_err(NameObjectError::InvalidName)?),
            None => None,
        };

        // SAFETY: This method shares the same safety contract as set_object_name.
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
    pub unsafe fn destroy_raw_shader_module(&self, shader_module: vk::ShaderModule) {
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
    pub unsafe fn destroy_raw_pipeline_layout(&self, layout: vk::PipelineLayout) {
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
                // SAFETY: Caller guarantees command_buffer and rendering_info validity.
                unsafe { self.handle.cmd_begin_rendering(command_buffer, rendering_info) };
                Ok(())
            }
            Some(DynamicRenderingLoader::Extension(loader)) => {
                // SAFETY: Caller guarantees command_buffer and rendering_info validity.
                unsafe { loader.cmd_begin_rendering(command_buffer, rendering_info) };
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
                // SAFETY: Caller guarantees command_buffer validity and render pass state.
                unsafe { self.handle.cmd_end_rendering(command_buffer) };
                Ok(())
            }
            Some(DynamicRenderingLoader::Extension(loader)) => {
                // SAFETY: Caller guarantees command_buffer validity and render pass state.
                unsafe { loader.cmd_end_rendering(command_buffer) };
                Ok(())
            }
        }
    }
}

impl From<FetchPhysicalDeviceError> for CreateCompatibleError {
    fn from(value: FetchPhysicalDeviceError) -> Self {
        match value {
            FetchPhysicalDeviceError::MemoryExhaustion => Self::MemoryExhaustion,
            FetchPhysicalDeviceError::UnknownVulkan(e) => Self::UnknownVulkan(e),
        }
    }
}
