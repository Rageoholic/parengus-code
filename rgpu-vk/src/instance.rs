//! Vulkan instance creation and physical device enumeration.
//!
//! The central type is [`Instance`], which wraps an `ash::Instance` and
//! owns the entry-point loader, an optional debug messenger, and an
//! optional surface instance extension loader. It exposes physical device
//! queries and unsafe constructors for surfaces and logical devices.
//!
//! [`VkVersion`] is a thin newtype over the packed Vulkan version word.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use crate::surface::{
    CreateSurfaceError, SurfaceQueryError, SurfaceSupportError,
};
use std::{
    ffi::{CStr, CString},
    fmt::Debug,
    str::FromStr,
};

/// Minimum severity level for Vulkan validation layer messages.
///
/// Passed to [`Instance::new`] as `max_log_level`. Messages at or
/// above the chosen level are forwarded to the [`tracing`] subscriber;
/// lower-severity messages are suppressed. Variants are ordered
/// least-to-most severe: `Verbose < Info < Warning < Error`.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum VulkanLogLevel {
    Verbose,
    Info,
    Warning,
    Error,
}

/// A packed Vulkan API version number.
///
/// Wraps the 32-bit encoding used by `VkApplicationInfo` and
/// `vkEnumerateInstanceVersion`. Construct from components with
/// [`new`](Self::new), or wrap an already-encoded word with
/// [`from_raw`](Self::from_raw).
#[derive(Debug, Clone, Copy)]
pub struct VkVersion(u32);

impl VkVersion {
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    pub fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        Self(vk::make_api_version(variant, major, minor, patch))
    }

    pub fn variant(&self) -> u32 {
        vk::api_version_variant(self.0)
    }

    pub fn major(&self) -> u32 {
        vk::api_version_major(self.0)
    }
    pub fn minor(&self) -> u32 {
        vk::api_version_minor(self.0)
    }
    pub fn patch(&self) -> u32 {
        vk::api_version_patch(self.0)
    }

    pub fn to_tuple(&self) -> (u32, u32, u32, u32) {
        (self.variant(), self.major(), self.minor(), self.patch())
    }

    pub fn from_tuple(tuple: (u32, u32, u32, u32)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2, tuple.3)
    }

    pub fn to_raw(&self) -> u32 {
        self.0
    }
}

/// The root Vulkan object.
///
/// Owns the `ash::Entry` loader, the `ash::Instance` handle, an
/// optional debug messenger, and optional surface extension state.
/// All objects derived from an instance hold an `Arc<Instance>` to
/// keep it alive.
///
/// Construct via [`Instance::new`], which is `unsafe` because it
/// loads a Vulkan shared library through `libloading`.
pub struct Instance {
    entry: ash::Entry,
    handle: ash::Instance,
    debug_messenger:
        Option<(vk::DebugUtilsMessengerEXT, ash::ext::debug_utils::Instance)>,
    surface_instance: Option<ash::khr::surface::Instance>,
    ver: VkVersion,
}

impl Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("handle", &self.handle.handle())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum InstanceCreationError {
    #[error("Could not load Vulkan: {0}")]
    Loading(ash::LoadingError),
    #[error("Couldn't get display handle from passed value: {0}")]
    InvalidDisplayHandle(crate::RwhHandleError),
    #[error("Missing mandatory instance extensions: {0:?}")]
    MissingExtensions(Vec<String>),
    #[error("Unknown Vulkan Error {0}")]
    UnknownVulkan(vk::Result),
    #[error("Invalid app name was passed to Instance::new")]
    InvalidAppName,
}

impl Drop for Instance {
    fn drop(&mut self) {
        tracing::debug!("Dropping instance {:?}", self.handle.handle());
        if let Some((debug_messenger, debug_utils_instance)) =
            self.debug_messenger.take()
        {
            //SAFETY: last use of this debug messenger. We made this debug
            //messenger from this instance. debug_utils_instance is derived from
            //this instance
            unsafe {
                debug_utils_instance
                    .destroy_debug_utils_messenger(debug_messenger, None)
            };
        }
        //SAFETY: We are in drop so this is the last use of instance. Any given
        //derived object should be gone
        unsafe { self.handle.destroy_instance(None) };
    }
}

impl From<vk::Result> for InstanceCreationError {
    fn from(value: vk::Result) -> Self {
        InstanceCreationError::UnknownVulkan(value)
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    //SAFETY: Vulkan guarantees p_callback_data is valid
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }
        .to_string_lossy();

    let type_str = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            tracing::trace!(
                target: "rvk-debug-messenger",
                "[{}] {}",
                type_str,
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::info!(
                target: "rvk-debug-messenger",
                "[{}] {}",
                type_str,
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::warn!(
                target: "rvk-debug-messenger",
                "[{}] {}",
                type_str,
                message
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            tracing::error!(
                target: "rvk-debug-messenger",
                "[{}] {}",
                type_str,
                message
            );
        }
        _ => {
            tracing::debug!(
                target: "rvk-debug-messenger",
                "[{}] {}",
                type_str,
                message
            );
        }
    }

    vk::FALSE
}

/// Optional instance-level extensions for [`Instance::new`].
///
/// Defaults to all `false`. Set `surface` to `true` to request the
/// platform-specific surface extensions for `VkSurfaceKHR` support.
/// A `display_handle_source` must also be provided so the required
/// extension names can be enumerated.
#[derive(Debug, Default)]
pub struct InstanceExtensions {
    pub surface: bool,
}

impl Instance {
    /// Creates a new instance by loading vulkan and using the requested API
    /// version
    ///
    /// # Safety
    /// This loads vulkan using libloading, meaning that there can be
    /// arbitrary code executed. This is not great but it's *probably*
    /// fine?
    pub unsafe fn new(
        app_name: impl AsRef<str>,
        max_log_level: Option<VulkanLogLevel>,
        display_handle_source: Option<&impl HasDisplayHandle>,
        enabled_exts: InstanceExtensions,
    ) -> Result<Self, InstanceCreationError> {
        use InstanceCreationError as Error;

        let app_name_cstring = match CString::from_str(app_name.as_ref()) {
            Ok(cstr) => cstr,
            Err(_) => Err(Error::InvalidAppName)?,
        };
        //SAFETY: We pass on the burden of the safety from loading dlls to the
        //caller. As for Entry, we ensure all other vulkan objects are dropped
        //before Entry is dropped (handled in the Drop impl of Instance)
        let entry = unsafe { ash::Entry::load() }.map_err(Error::Loading)?;

        // SAFETY: entry is a live Vulkan entry (loaded on line 202);
        // vkEnumerateInstanceVersion has no preconditions beyond a valid
        // entry point.
        let api_version = unsafe { entry.try_enumerate_instance_version() }
            .unwrap_or(Some(vk::API_VERSION_1_0))
            .unwrap_or(vk::API_VERSION_1_0);
        let mut mandatory_exts = Vec::with_capacity(256);

        // Tracks whether surface extensions were actually enabled on
        // the instance. Being requested (`enabled_exts.surface`) is
        // not enough — the platform extensions are only added to
        // `mandatory_exts` when a display handle source is provided.
        let mut surface_ext_loaded = false;

        if let Some(display_handle_source) = display_handle_source
            && enabled_exts.surface
        {
            surface_ext_loaded = true;
            // ash_window will be necessary to get a surface later,
            // but surfaces are an extension. This gets those extensions
            // to start as a base to the set of mandatory extensions we
            // will almost always need.
            let ash_window_exts = ash_window::enumerate_required_extensions(
                display_handle_source
                    .display_handle()
                    .map_err(Error::InvalidDisplayHandle)?
                    .as_raw(),
            )?;

            mandatory_exts.extend(
                ash_window_exts
                    .iter()
                    //SAFETY: ash_window promises to hand us null
                    //terminated C strings in its API. This isn't
                    //enforced anywhere through any safety means
                    //but it is documented
                    .map(|ext_cstr_ptr| unsafe {
                        CStr::from_ptr(*ext_cstr_ptr)
                    }),
            );
        }

        // SAFETY: entry is a live Vulkan entry; passing None queries
        // global extensions and does not dereference any layer name.
        let instance_exts_avail =
            unsafe { entry.enumerate_instance_extension_properties(None) }?;
        // SAFETY: entry is a live Vulkan entry;
        // vkEnumerateInstanceLayerProperties has no additional preconditions.
        let instance_layers_avail =
            unsafe { entry.enumerate_instance_layer_properties() };

        let missing_exts: Vec<_> = mandatory_exts
            .iter()
            .filter(|mandatory_ext| {
                !instance_exts_avail.iter().any(|avail| {
                    avail.extension_name_as_c_str() == Ok(**mandatory_ext)
                })
            })
            .map(|ext| ext.to_string_lossy().into_owned())
            .collect();

        if !missing_exts.is_empty() {
            return Err(Error::MissingExtensions(missing_exts));
        }

        // Check if we can enable debug utils
        let debug_utils_ext_name = ash::ext::debug_utils::NAME;
        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";

        let debug_utils_available = instance_exts_avail.iter().any(|ext| {
            ext.extension_name_as_c_str() == Ok(debug_utils_ext_name)
        });

        let validation_layer_available = instance_layers_avail
            .as_ref()
            .map(|layers| {
                layers.iter().any(|layer| {
                    layer.layer_name_as_c_str() == Ok(validation_layer_name)
                })
            })
            .unwrap_or(false);

        let mut enabled_ext_ptrs: Vec<_> =
            mandatory_exts.iter().map(|ext| ext.as_ptr()).collect();
        let mut enabled_layers: Vec<*const i8> = Vec::new();

        let mut debug_messenger_create_info = if let Some(log_level) =
            max_log_level
            && debug_utils_available
            && validation_layer_available
        {
            enabled_ext_ptrs.push(debug_utils_ext_name.as_ptr());
            enabled_layers.push(validation_layer_name.as_ptr());

            let message_severity = match log_level {
                VulkanLogLevel::Verbose => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Info => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Warning => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Error => {
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
            };

            Some(
                vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(message_severity)
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback)),
            )
        } else {
            None
        };

        let engine_name = c"rvk";

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name_cstring)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(api_version);

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_ext_ptrs)
            .enabled_layer_names(&enabled_layers);

        if let Some(ref mut debug_info) = debug_messenger_create_info {
            instance_create_info = instance_create_info.push_next(debug_info);
        }

        //SAFETY: We made a valid instance_create_info
        let instance =
            unsafe { entry.create_instance(&instance_create_info, None) }?;

        let debug_messenger = if let Some(mut debug_messenger_create_info) =
            debug_messenger_create_info
        {
            //Defensive coding stuff
            debug_messenger_create_info.p_next = std::ptr::null();
            let debug_utils_instance =
                ash::ext::debug_utils::Instance::new(&entry, &instance);
            //SAFETY: Valid CI
            match unsafe {
                debug_utils_instance.create_debug_utils_messenger(
                    &debug_messenger_create_info,
                    None,
                )
            } {
                Ok(debug_messenger) => {
                    Some((debug_messenger, debug_utils_instance))
                }
                Err(e) => {
                    tracing::error!(
                        "Despite us having a valid \
                         debug_messenger_create_info \
                         We can't seem to make a debug messenger? \
                         WTF? Continuing without one but here be \
                         dragons. Actual error: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let surface_instance = surface_ext_loaded
            .then(|| ash::khr::surface::Instance::new(&entry, &instance));

        Ok(Instance {
            entry,
            handle: instance,
            debug_messenger,
            surface_instance,
            ver: VkVersion::from_raw(api_version),
        })
    }

    /// Destroy the raw VkSurfaceKHR.
    ///
    /// # Safety
    /// All objects derived from surf must be destroyed first.
    /// No in-flight GPU work may still reference `surf`.
    ///
    /// You can't use surf after this function is called (for obvious reasons)
    ///
    /// surf must be derived from this instance
    pub unsafe fn destroy_raw_surface(
        &self,
        surf: vk::SurfaceKHR,
    ) -> Result<(), DestroyRawSurfaceError> {
        if let Some(ref surface_instance) = self.surface_instance {
            // SAFETY: Surf is derived from this instance (passed on to caller)
            unsafe {
                surface_instance.destroy_surface(surf, None);
            };
            Ok(())
        } else {
            Err(DestroyRawSurfaceError::ExtensionNotLoaded)
        }
    }

    /// Get a vector of handles to available physical devices. These handles are
    /// ONLY valid in the context of this instance.
    pub fn fetch_raw_physical_devices(
        &self,
    ) -> Result<Vec<vk::PhysicalDevice>, FetchPhysicalDeviceError> {
        //SAFETY: Pretty much always fine
        match unsafe { self.handle.enumerate_physical_devices() } {
            Ok(v) => Ok(v),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
            | Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(FetchPhysicalDeviceError::MemoryExhaustion)
            }
            Err(e) => Err(FetchPhysicalDeviceError::UnknownVulkan(e)),
        }
    }

    /// Get the properties of a physical device.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    pub unsafe fn get_raw_physical_device_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        //SAFETY: physical_device was derived from this instance
        unsafe { self.handle.get_physical_device_properties(physical_device) }
    }

    /// Get the queue family properties of a physical device.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    pub unsafe fn get_raw_physical_device_queue_family_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        //SAFETY: physical_device was derived from this instance
        unsafe {
            self.handle
                .get_physical_device_queue_family_properties(physical_device)
        }
    }

    /// Get memory properties of a physical device.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    pub unsafe fn get_raw_physical_device_memory_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        // SAFETY: physical_device was derived from this instance.
        unsafe {
            self.handle
                .get_physical_device_memory_properties(physical_device)
        }
    }

    /// Create a logical device from a physical device.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    /// `create_info` must be a valid DeviceCreateInfo.
    /// Any handles referenced by `create_info` must also be derived from this
    /// instance and remain valid for the duration of the call.
    pub unsafe fn create_ash_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo<'_>,
    ) -> Result<ash::Device, vk::Result> {
        //SAFETY: physical_device was derived from this instance,
        //create_info is valid
        unsafe {
            self.handle
                .create_device(physical_device, create_info, None)
        }
    }

    /// Enumerate device extension properties for a physical device.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    pub unsafe fn enumerate_raw_device_extension_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::ExtensionProperties>, vk::Result> {
        //SAFETY: physical_device was derived from this instance
        unsafe {
            self.handle
                .enumerate_device_extension_properties(physical_device)
        }
    }

    /// The Vulkan API version negotiated at instance creation time.
    ///
    /// This is the version reported by `vkEnumerateInstanceVersion`,
    /// not necessarily the version requested by the application.
    pub fn supported_ver(&self) -> VkVersion {
        self.ver
    }

    pub fn raw_instance(&self) -> vk::Instance {
        self.handle.handle()
    }

    pub fn ash_instance(&self) -> &ash::Instance {
        &self.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_version_tuple_roundtrip() {
        let version = VkVersion::new(1, 2, 3, 4);
        let tuple = version.to_tuple();
        let rebuilt = VkVersion::from_tuple(tuple);

        assert_eq!(version.to_raw(), rebuilt.to_raw());
    }

    #[test]
    fn vk_version_raw_roundtrip() {
        let raw = vk::make_api_version(0, 1, 3, 275);
        let version = VkVersion::from_raw(raw);

        assert_eq!(version.to_raw(), raw);
        assert_eq!(version.variant(), 0);
        assert_eq!(version.major(), 1);
        assert_eq!(version.minor(), 3);
        assert_eq!(version.patch(), 275);
    }
}

#[derive(Debug, Error)]
pub enum FetchPhysicalDeviceError {
    #[error("Error fetching physical devices, memory exhaustion")]
    MemoryExhaustion,
    #[error("Error fetching physical devices, Unknown vulkan: {0}")]
    UnknownVulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum DestroyRawSurfaceError {
    #[error("Surface extension is not loaded")]
    ExtensionNotLoaded,
}

// Extensions related to surface functionality
impl Instance {
    /// Check if a queue family on a physical device supports presenting to
    /// a surface.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from this instance.
    /// `surface` must be a valid handle derived from this instance.
    pub unsafe fn get_raw_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        surface: vk::SurfaceKHR,
    ) -> Result<bool, SurfaceSupportError> {
        if let Some(ref surface_instance) = self.surface_instance {
            //SAFETY: physical_device and surface were derived from
            //this instance
            unsafe {
                surface_instance.get_physical_device_surface_support(
                    physical_device,
                    queue_family_index,
                    surface,
                )
            }
            .map_err(SurfaceSupportError::Vulkan)
        } else {
            Err(SurfaceSupportError::ExtensionNotLoaded)
        }
    }

    /// Query the surface capabilities for a physical device + surface pair.
    ///
    /// # Safety
    /// `physical_device` and `surface` must both be derived from this
    /// instance.
    pub unsafe fn get_surface_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceCapabilitiesKHR, SurfaceQueryError> {
        if let Some(ref surface_instance) = self.surface_instance {
            // SAFETY: Caller guarantees physical_device and surface provenance.
            unsafe {
                surface_instance.get_physical_device_surface_capabilities(
                    physical_device,
                    surface,
                )
            }
            .map_err(SurfaceQueryError::Vulkan)
        } else {
            Err(SurfaceQueryError::ExtensionNotLoaded)
        }
    }

    /// Query supported surface formats for a physical device + surface pair.
    ///
    /// # Safety
    /// `physical_device` and `surface` must both be derived from this
    /// instance.
    pub unsafe fn get_surface_formats(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, SurfaceQueryError> {
        if let Some(ref surface_instance) = self.surface_instance {
            // SAFETY: Caller guarantees physical_device and surface provenance.
            unsafe {
                surface_instance.get_physical_device_surface_formats(
                    physical_device,
                    surface,
                )
            }
            .map_err(SurfaceQueryError::Vulkan)
        } else {
            Err(SurfaceQueryError::ExtensionNotLoaded)
        }
    }

    /// Query supported present modes for a physical device + surface pair.
    ///
    /// # Safety
    /// `physical_device` and `surface` must both be derived from this
    /// instance.
    pub unsafe fn get_surface_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::PresentModeKHR>, SurfaceQueryError> {
        if let Some(ref surface_instance) = self.surface_instance {
            // SAFETY: Caller guarantees physical_device and surface provenance.
            unsafe {
                surface_instance.get_physical_device_surface_present_modes(
                    physical_device,
                    surface,
                )
            }
            .map_err(SurfaceQueryError::Vulkan)
        } else {
            Err(SurfaceQueryError::ExtensionNotLoaded)
        }
    }

    ///Create a raw VkSurfaceKHR.
    ///
    /// # Safety
    /// The returned surface must be destroyed before source is dropped, or when
    /// the surface is invalidated due to something like a suspend event in
    /// winit. There is a parent child relationship between both the instance
    /// and source and the returned surface.
    ///
    /// The returned surface must only be used with this instance.
    pub unsafe fn create_raw_surface<T: HasDisplayHandle + HasWindowHandle>(
        &self,
        source: &T,
    ) -> Result<vk::SurfaceKHR, CreateSurfaceError> {
        use CreateSurfaceError as Error;
        if self.surface_instance.is_some() {
            //SAFETY:
            unsafe {
                ash_window::create_surface(
                    &self.entry,
                    &self.handle,
                    source
                        .display_handle()
                        .map_err(Error::InvalidDisplayHandle)?
                        .as_raw(),
                    source
                        .window_handle()
                        .map_err(Error::InvalidWindowHandle)?
                        .as_raw(),
                    None,
                )
            }
            .map_err(Error::VulkanError)
        } else {
            Err(Error::MissingExtension)
        }
    }
}

// Device extension loader creation functionality
impl Instance {
    pub fn create_swapchain_loader(
        &self,
        device: &ash::Device,
    ) -> ash::khr::swapchain::Device {
        ash::khr::swapchain::Device::new(&self.handle, device)
    }

    pub fn create_dynamic_rendering_loader(
        &self,
        device: &ash::Device,
    ) -> ash::khr::dynamic_rendering::Device {
        ash::khr::dynamic_rendering::Device::new(&self.handle, device)
    }

    pub fn create_synchronization2_loader(
        &self,
        device: &ash::Device,
    ) -> ash::khr::synchronization2::Device {
        ash::khr::synchronization2::Device::new(&self.handle, device)
    }

    pub fn create_debug_utils_device_loader(
        &self,
        device: &ash::Device,
    ) -> Option<ash::ext::debug_utils::Device> {
        if self.debug_messenger.is_some() {
            Some(ash::ext::debug_utils::Device::new(&self.handle, device))
        } else {
            None
        }
    }
}
