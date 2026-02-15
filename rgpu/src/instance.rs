use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use crate::log::VulkanLogLevel;
use std::{
    ffi::{CStr, CString},
    fmt::Debug,
    str::FromStr,
};

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

pub struct Instance {
    entry: ash::Entry,
    handle: ash::Instance,
    debug_messenger: Option<(vk::DebugUtilsMessengerEXT, ash::ext::debug_utils::Instance)>,
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
    #[error("Could not load libvulkan: {0}")]
    LibraryLoading(libloading::Error),
    #[error("Could not load vkGetInstanceProcAddr from libvulkan")]
    MissingEntryPoint,
    #[error("Couldn't get display handle from passed value: {0}")]
    InvalidDisplayHandle(raw_window_handle::HandleError),
    #[error("Missing mandatory instance extensions: {0:?}")]
    MissingExtensions(Vec<String>),
    #[error("Unknown Vulkan Error {0}")]
    UnknownVulkan(ash::vk::Result),
    #[error("Invalid app name was passed to Instance::new")]
    InvalidAppName,
}

impl Drop for Instance {
    fn drop(&mut self) {
        tracing::debug!("Dropping instance {:?}", self.handle.handle());
        if let Some((debug_messenger, debug_utils_instance)) = self.debug_messenger.take() {
            //SAFETY: last use of this debug messenger. We made this debug
            //messenger from this instance. debug_utils_instance is derived from
            //this instance
            unsafe { debug_utils_instance.destroy_debug_utils_messenger(debug_messenger, None) };
        }
        //SAFETY: We are in drop so this is the last use of instance. Any given
        //derived object should be gone
        unsafe { self.handle.destroy_instance(None) };
    }
}

impl From<ash::vk::Result> for InstanceCreationError {
    fn from(value: ash::vk::Result) -> Self {
        InstanceCreationError::UnknownVulkan(value)
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut std::ffi::c_void,
) -> ash::vk::Bool32 {
    //SAFETY: Vulkan guarantees p_callback_data is valid
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }.to_string_lossy();

    let type_str = match message_type {
        ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    match message_severity {
        ash::vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            tracing::trace!(target: "rvk-debug-messenger", "[{}] {}", type_str, message);
        }
        ash::vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::info!(target: "rvk-debug-messenger", "[{}] {}", type_str, message);
        }
        ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::warn!(target: "rvk-debug-messenger", "[{}] {}", type_str, message);
        }
        ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            tracing::error!(target: "rvk-debug-messenger", "[{}] {}", type_str, message);
        }
        _ => {
            tracing::debug!(target: "rvk-debug-messenger", "[{}] {}", type_str, message);
        }
    }

    ash::vk::FALSE
}

#[derive(Debug, Default)]
pub struct InstanceExtensions {
    pub surface: bool,
}

impl Instance {
    /// Creates a new instance by loading vulkan and using the requested API
    /// version
    ///
    /// # Safety
    /// This loads vulkan using libloading, meaning that there can be arbitrary code
    /// executed. This is not great but it's *probably* fine?
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
        let entry = unsafe { ash::Entry::load() }.map_err(|e| match e {
            ash::LoadingError::LibraryLoadFailure(error) => Error::LibraryLoading(error),
            ash::LoadingError::MissingEntryPoint(_) => Error::MissingEntryPoint,
        })?;

        //SAFETY: Basically always fine Relax
        let api_version = unsafe { entry.try_enumerate_instance_version() }
            .unwrap_or(Some(ash::vk::API_VERSION_1_0))
            .expect("I dunno how we got here");
        let mut mandatory_exts = Vec::with_capacity(256);

        if let Some(display_handle_source) = display_handle_source
            && enabled_exts.surface
        {
            // ash_window will be necessary to get a surface later, but surfaces are
            // an extension. This gets those extensions to start as a base to the
            // set of mandatory extensions we will almost always need.
            let ash_window_exts = ash_window::enumerate_required_extensions(
                display_handle_source
                    .display_handle()
                    .map_err(|e| Error::InvalidDisplayHandle(e))?
                    .as_raw(),
            )?;

            mandatory_exts.extend(
                ash_window_exts
                    .iter()
                    //SAFETY: ash_window promises to hand us null terminated C strings
                    //in its API. This isn't enforced anywhere through any safety means
                    //but it is documented
                    .map(|ext_cstr_ptr| unsafe { CStr::from_ptr(*ext_cstr_ptr) }),
            );
        }

        //SAFETY: Pretty much always okay
        let instance_exts_avail = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        //SAFETY: Pretty much always okay
        let instance_layers_avail = unsafe { entry.enumerate_instance_layer_properties() };

        let missing_exts: Vec<_> = mandatory_exts
            .iter()
            .filter(|mandatory_ext| {
                !instance_exts_avail
                    .iter()
                    .any(|avail| avail.extension_name_as_c_str() == Ok(**mandatory_ext))
            })
            .map(|ext| ext.to_string_lossy().into_owned())
            .collect();

        if !missing_exts.is_empty() {
            return Err(Error::MissingExtensions(missing_exts));
        }

        // Check if we can enable debug utils
        let debug_utils_ext_name = ash::ext::debug_utils::NAME;
        let validation_layer_name = c"VK_LAYER_KHRONOS_validation";

        let debug_utils_available = instance_exts_avail
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(debug_utils_ext_name));

        let validation_layer_available = instance_layers_avail
            .as_ref()
            .map(|layers| {
                layers
                    .iter()
                    .any(|layer| layer.layer_name_as_c_str() == Ok(validation_layer_name))
            })
            .unwrap_or(false);

        let enable_debug_utils =
            max_log_level.is_some() && debug_utils_available && validation_layer_available;

        let mut enabled_exts: Vec<_> = mandatory_exts.iter().map(|ext| ext.as_ptr()).collect();
        let mut enabled_layers: Vec<*const i8> = Vec::new();

        let mut debug_messenger_create_info = if enable_debug_utils {
            enabled_exts.push(debug_utils_ext_name.as_ptr());
            enabled_layers.push(validation_layer_name.as_ptr());

            let log_level =
                max_log_level.expect("enable_debug_utils is true so max_log_level must be Some");
            let message_severity = match log_level {
                VulkanLogLevel::Verbose => {
                    ash::vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Info => {
                    ash::vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Warning => {
                    ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                }
                VulkanLogLevel::Error => ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            };

            Some(
                ash::vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(message_severity)
                    .message_type(
                        ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(vulkan_debug_callback)),
            )
        } else {
            None
        };

        let engine_name = c"rvk";

        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(&app_name_cstring)
            .application_version(ash::vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name)
            .engine_version(ash::vk::make_api_version(0, 0, 1, 0))
            .api_version(api_version);

        let mut instance_create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_exts)
            .enabled_layer_names(&enabled_layers);

        if let Some(ref mut debug_info) = debug_messenger_create_info {
            instance_create_info = instance_create_info.push_next(debug_info);
        }

        //SAFETY: We made a valid instance_create_info
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;

        let debug_messenger =
            if let Some(mut debug_messenger_create_info) = debug_messenger_create_info {
                //Defensive coding stuff
                debug_messenger_create_info.p_next = std::ptr::null();
                let debug_utils_instance = ash::ext::debug_utils::Instance::new(&entry, &instance);
                //SAFETY: Valid CI
                match unsafe {
                    debug_utils_instance
                        .create_debug_utils_messenger(&debug_messenger_create_info, None)
                } {
                    Ok(debug_messenger) => Some((debug_messenger, debug_utils_instance)),
                    Err(e) => {
                        tracing::error!(
                            "Despite us having a valid debug_messenger_create_info \
                            We can't seem to make a debug messenger? WTF? Continuing \
                            without one but here be dragons. Actual error: {e}"
                        );
                        None
                    }
                }
            } else {
                None
            };
        let surface_instance = Some(ash::khr::surface::Instance::new(&entry, &instance));

        Ok(Instance {
            entry,
            handle: instance,
            debug_messenger,
            surface_instance,
            ver: VkVersion::from_raw(api_version),
        })
    }

    ///Create a raw VkSurfaceKHR.
    ///
    /// # Safety
    /// The returned surface must be destroyed before source is dropped, or when
    /// the surface is invalidated due to something like a suspend event in
    /// winit. There is a parent child relationship between both the instance
    /// and source and the returned surface
    pub unsafe fn create_raw_surface<T: HasDisplayHandle + HasWindowHandle>(
        &self,
        source: &T,
    ) -> Result<vk::SurfaceKHR, CreateRawSurfaceError> {
        use CreateRawSurfaceError as Error;
        if self.surface_instance.is_some() {
            //SAFETY:
            unsafe {
                ash_window::create_surface(
                    &self.entry,
                    &self.handle,
                    source
                        .display_handle()
                        .map_err(|e| Error::DisplayHandle(e))?
                        .as_raw(),
                    source
                        .window_handle()
                        .map_err(|e| Error::WindowHandle(e))?
                        .as_raw(),
                    None,
                )
            }
            .map_err(|e| Error::OnCreate(e))
        } else {
            Err(Error::ExtensionNotLoaded)
        }
    }

    /// Destroy the raw VkSurfaceKHR.
    ///
    /// # Safety
    /// All objects derived from surf must be destroyed first.
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
    pub fn fetch_physical_devices(
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

    pub fn get_supported_ver(&self) -> VkVersion {
        self.ver
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

#[derive(Debug, Error)]
pub enum CreateRawSurfaceError {
    #[error("Error creating surface: {0}")]
    OnCreate(vk::Result),
    #[error("Unable to get display handle: {0}")]
    DisplayHandle(raw_window_handle::HandleError),
    #[error("Unable to get window handle: {0}")]
    WindowHandle(raw_window_handle::HandleError),
    #[error("Surface extension has not been loaded")]
    ExtensionNotLoaded,
}
