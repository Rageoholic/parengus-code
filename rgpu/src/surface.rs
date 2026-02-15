use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;

use crate::instance::{CreateRawSurfaceError, Instance};

#[derive(Debug, Error)]
pub enum CreateSurfaceError {
    #[error("Couldn't get display handle: {0}")]
    InvalidDisplayHandle(raw_window_handle::HandleError),
    #[error("Couldn't get window handle: {0}")]
    InvalidWindowHandle(raw_window_handle::HandleError),
    #[error("Vulkan surface creation failed: {0}")]
    VulkanError(ash::vk::Result),
    #[error("Parent instance did not have the surface extensions for this platform loaded")]
    MissingExtension,
}

pub struct Surface<T: HasWindowHandle + HasDisplayHandle> {
    parent_instance: Arc<Instance>,
    handle: ash::vk::SurfaceKHR,
    _surface_source: Arc<T>,
}

impl<T: HasWindowHandle + HasDisplayHandle> std::fmt::Debug for Surface<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl<T: HasWindowHandle + HasDisplayHandle> Surface<T> {
    /// Creates a new surface associated with the source. We might want to
    /// separate the DisplayHandle source and the WindowHandle source but rn
    /// winit doesn't seem to require it and I feel like any good windowing lib
    /// wouldn't. We'll do some reasearch in the future
    ///
    /// # Safety
    /// This must be dropped on events like suspend in winit due to the surface
    /// being implicitly invalidated. I'm not sure if this actually requires
    /// unsafe but I'm being aggressive here
    pub unsafe fn new(
        instance: &Arc<Instance>,
        source: Arc<T>,
    ) -> Result<Self, CreateSurfaceError> {
        //SAFETY: We hold Arc references to the instance and source, ensuring
        //they outlive the surface
        let surface = unsafe { instance.create_raw_surface(&source) }?;

        Ok(Surface {
            parent_instance: Arc::clone(instance),
            handle: surface,

            _surface_source: source,
        })
    }

    pub fn get_parent(&self) -> &Arc<Instance> {
        &self.parent_instance
    }
}

impl From<CreateRawSurfaceError> for CreateSurfaceError {
    fn from(value: CreateRawSurfaceError) -> Self {
        match value {
            CreateRawSurfaceError::OnCreate(e) => CreateSurfaceError::VulkanError(e),
            CreateRawSurfaceError::DisplayHandle(handle_error) => {
                CreateSurfaceError::InvalidDisplayHandle(handle_error)
            }
            CreateRawSurfaceError::WindowHandle(handle_error) => {
                CreateSurfaceError::InvalidWindowHandle(handle_error)
            }
            CreateRawSurfaceError::ExtensionNotLoaded => CreateSurfaceError::MissingExtension,
        }
    }
}

impl<T: HasWindowHandle + HasDisplayHandle> Drop for Surface<T> {
    fn drop(&mut self) {
        tracing::debug!("Dropping surface {:?}", self.handle);
        //SAFETY: This is being dropped which means all derived objects should
        //also be being dropped.
        let _ = unsafe { self.parent_instance.destroy_raw_surface(self.handle) }.inspect_err(|e| {
            tracing::error!("Error while dropping surface {:?}: {e}", self.handle)
        });
    }
}
