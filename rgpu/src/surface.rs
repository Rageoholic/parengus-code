use std::sync::Arc;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use crate::instance::Instance;

#[derive(Debug, Error)]
pub enum CreateSurfaceError {
    #[error("Couldn't get display handle: {0}")]
    InvalidDisplayHandle(raw_window_handle::HandleError),
    #[error("Couldn't get window handle: {0}")]
    InvalidWindowHandle(raw_window_handle::HandleError),
    #[error("Vulkan surface creation failed: {0}")]
    VulkanError(ash::vk::Result),
    #[error(
        "Parent instance did not have the surface extensions \
         for this platform loaded"
    )]
    MissingExtension,
}

#[derive(Debug, Error)]
pub enum SurfaceSupportError {
    #[error("Surface extension is not loaded")]
    ExtensionNotLoaded,
    #[error("Vulkan error checking surface support: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum SurfaceQueryError {
    #[error("Surface extension is not loaded")]
    ExtensionNotLoaded,
    #[error("Vulkan error querying surface: {0}")]
    Vulkan(vk::Result),
}

pub struct Surface<T: HasWindowHandle + HasDisplayHandle> {
    parent_instance: Arc<Instance>,
    handle: ash::vk::SurfaceKHR,
    _surface_source: Arc<T>,
}

struct SurfaceDebugWithSource<
    'a,
    T: HasWindowHandle + HasDisplayHandle + std::fmt::Debug,
>(&'a Surface<T>);

impl<T: HasWindowHandle + HasDisplayHandle + std::fmt::Debug> std::fmt::Debug
    for SurfaceDebugWithSource<'_, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.0.handle)
            .field("parent", &self.0.parent_instance)
            .field("source", &self.0._surface_source)
            .finish_non_exhaustive()
    }
}

impl<T: HasWindowHandle + HasDisplayHandle> std::fmt::Debug for Surface<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.handle)
            .field("parent", &self.parent_instance)
            .finish_non_exhaustive()
    }
}

impl<T: HasWindowHandle + HasDisplayHandle> Surface<T> {
    /// Creates a new surface associated with the source. We might want to
    /// separate the DisplayHandle source and the WindowHandle source but rn
    /// winit doesn't seem to require it and I feel like any good windowing lib
    /// wouldn't. We'll do some research in the future
    ///
    /// # Safety
    /// This must be dropped on events like suspend in winit due to the surface
    /// being implicitly invalidated. I'm not sure if this actually requires
    /// unsafe but I'm being aggressive here.
    ///
    /// Callers are responsible for ensuring no in-flight GPU work still
    /// references resources derived from this surface at destruction time.
    pub unsafe fn new(
        instance: &Arc<Instance>,
        source: Arc<T>,
    ) -> Result<Self, CreateSurfaceError> {
        //SAFETY: We hold Arc references to the instance and source, ensuring
        //they outlive the surface
        let surface = unsafe { instance.create_raw_surface(&source) }?;

        // SAFETY: `surface` was created from `instance` and `source` is the
        // handle provider used to create it.
        Ok(unsafe { Self::from_parts(Arc::clone(instance), surface, source) })
    }

    /// # Safety
    /// `handle` must be a valid `VkSurfaceKHR` created from `parent_instance`,
    /// and `source` must remain a valid window/display handle source for the
    /// lifetime expectations of this surface wrapper.
    pub unsafe fn from_parts(
        parent_instance: Arc<Instance>,
        handle: vk::SurfaceKHR,
        source: Arc<T>,
    ) -> Self {
        Self {
            parent_instance,
            handle,
            _surface_source: source,
        }
    }

    pub fn get_parent(&self) -> &Arc<Instance> {
        &self.parent_instance
    }

    pub fn raw_handle(&self) -> vk::SurfaceKHR {
        self.handle
    }

    /// Returns a richer debug view that includes the source when `T: Debug`.
    ///
    /// This keeps the base `Debug` impl available for all `T` without
    /// requiring `T: Debug`.
    pub fn debug_with_source(&self) -> impl std::fmt::Debug + '_
    where
        T: std::fmt::Debug,
    {
        SurfaceDebugWithSource(self)
    }

    /// Check if a queue family on a physical device supports presenting to
    /// this surface.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from the same instance
    /// as this surface.
    pub unsafe fn supports_queue_family(
        &self,
        physical_device: ash::vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> Result<bool, SurfaceSupportError> {
        //SAFETY: physical_device was derived from the same instance as this
        //surface (caller guarantees), self.handle is valid
        unsafe {
            self.parent_instance
                .get_raw_physical_device_surface_support(
                    physical_device,
                    queue_family_index,
                    self.handle,
                )
        }
    }

    /// Query swapchain surface capabilities for this surface.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from the same
    /// instance as this surface.
    pub unsafe fn query_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, SurfaceQueryError> {
        // SAFETY: Caller guarantees physical_device provenance for
        // this instance.
        unsafe {
            self.parent_instance
                .get_surface_capabilities(physical_device, self.handle)
        }
    }

    /// Query supported surface formats for this surface.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from the same
    /// instance as this surface.
    pub unsafe fn query_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, SurfaceQueryError> {
        // SAFETY: Caller guarantees physical_device provenance for
        // this instance.
        unsafe {
            self.parent_instance
                .get_surface_formats(physical_device, self.handle)
        }
    }

    /// Query supported present modes for this surface.
    ///
    /// # Safety
    /// `physical_device` must be a valid handle derived from the same
    /// instance as this surface.
    pub unsafe fn query_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, SurfaceQueryError> {
        // SAFETY: Caller guarantees physical_device provenance for
        // this instance.
        unsafe {
            self.parent_instance
                .get_surface_present_modes(physical_device, self.handle)
        }
    }
}

impl<T: HasWindowHandle + HasDisplayHandle> Drop for Surface<T> {
    fn drop(&mut self) {
        tracing::debug!("Dropping surface {:?}", self.handle);
        //SAFETY: This is being dropped which means all derived objects should
        //also be being dropped and no in-flight work may still reference it.
        let _ =
            unsafe { self.parent_instance.destroy_raw_surface(self.handle) }
                .inspect_err(|e| {
                    tracing::error!(
                        "Error while dropping surface {:?}: {e}",
                        self.handle
                    )
                });
    }
}
