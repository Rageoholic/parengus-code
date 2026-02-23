use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::device::{Device, NameObjectError};
use crate::surface::{Surface, SurfaceQueryError};

#[derive(Debug, Error)]
pub enum CreateSwapchainError {
    #[error(
        "Mismatched parameters to Swapchain::new/new_with_old. \
         Device, surface, and optional old swapchain must be \
         derived from the same instance"
    )]
    MismatchedParams,

    #[error("No supported surface formats were reported")]
    NoSurfaceFormats,

    #[error("No supported present modes were reported")]
    NoPresentModes,

    #[error("Invalid requested swapchain extent ({width}x{height})")]
    InvalidExtent { width: u32, height: u32 },

    #[error("Swapchain support was not enabled on this device")]
    SwapchainNotEnabled,

    #[error("Failed while querying surface support details: {0}")]
    SurfaceQuery(#[from] SurfaceQueryError),

    #[error("Vulkan error creating swapchain: {0}")]
    VulkanCreate(vk::Result),

    #[error("Vulkan error fetching swapchain images: {0}")]
    VulkanGetImages(vk::Result),

    #[error("Vulkan error creating swapchain image view: {0}")]
    VulkanCreateImageView(vk::Result),
}

fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    preferred_format: Option<vk::Format>,
) -> vk::SurfaceFormatKHR {
    // Try the caller's preferred format first (any color space).
    if let Some(preferred) = preferred_format
        && let Some(found) =
            formats.iter().copied().find(|f| f.format == preferred)
    {
        return found;
    }

    formats
        .iter()
        .copied()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(formats[0])
}

fn choose_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .copied()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    desired_extent: vk::Extent2D,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: desired_extent.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: desired_extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn choose_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count > 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    image_count
}

fn choose_composite_alpha(
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::CompositeAlphaFlagsKHR {
    if capabilities
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
    {
        vk::CompositeAlphaFlagsKHR::OPAQUE
    } else if capabilities
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    {
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
    } else if capabilities
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    {
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
    } else {
        vk::CompositeAlphaFlagsKHR::INHERIT
    }
}

fn create_default_swapchain_image_views<FCreate, FDestroy, FName>(
    images: &[vk::Image],
    format: vk::Format,
    swapchain_debug_index: u64,
    mut create_image_view: FCreate,
    mut destroy_image_view: FDestroy,
    mut name_image_view: FName,
) -> Result<Vec<vk::ImageView>, CreateSwapchainError>
where
    FCreate: FnMut(
        &vk::ImageViewCreateInfo<'_>,
    ) -> Result<vk::ImageView, vk::Result>,
    FDestroy: FnMut(vk::ImageView),
    FName: FnMut(usize, vk::ImageView),
{
    let mut image_views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
    for (index, image) in images.iter().copied().enumerate() {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = match create_image_view(&create_info) {
            Ok(view) => view,
            Err(e) => {
                for created_view in image_views.drain(..) {
                    destroy_image_view(created_view);
                }
                return Err(CreateSwapchainError::VulkanCreateImageView(e));
            }
        };

        let _ = swapchain_debug_index;
        name_image_view(index, image_view);
        image_views.push(image_view);
    }

    Ok(image_views)
}

pub struct Swapchain<T: HasDisplayHandle + HasWindowHandle> {
    parent_device: Arc<Device>,
    _parent_surface: Arc<Surface<T>>,
    handle: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Option<Vec<vk::ImageView>>,
    /// Serializes `vkAcquireNextImageKHR`, which the Vulkan spec requires to
    /// be externally synchronized with respect to the swapchain handle.
    acquire_lock: Mutex<()>,
}

struct SwapchainDebugWithSource<
    'a,
    T: HasDisplayHandle + HasWindowHandle + std::fmt::Debug,
>(&'a Swapchain<T>);

impl<T: HasDisplayHandle + HasWindowHandle + std::fmt::Debug> std::fmt::Debug
    for SwapchainDebugWithSource<'_, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("handle", &self.0.handle)
            .field("format", &self.0.format)
            .field("extent", &self.0.extent)
            .field("image_count", &self.0.images.len())
            .field(
                "image_view_count",
                &self.0.image_views.as_ref().map(|v| v.len()),
            )
            .field(
                "parent_surface",
                &self.0._parent_surface.debug_with_source(),
            )
            .finish_non_exhaustive()
    }
}

impl<T: HasDisplayHandle + HasWindowHandle> std::fmt::Debug for Swapchain<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("handle", &self.handle)
            .field("format", &self.format)
            .field("extent", &self.extent)
            .field("image_count", &self.images.len())
            .field(
                "image_view_count",
                &self.image_views.as_ref().map(|v| v.len()),
            )
            .finish_non_exhaustive()
    }
}

impl<T: HasDisplayHandle + HasWindowHandle> Swapchain<T> {
    /// Create a swapchain using no previous swapchain handle.
    ///
    /// For resize/recreation paths, prefer `new_with_old` so drivers can
    /// optimize resource reuse.
    ///
    /// `preferred_format` is a hint for surface format selection. When the
    /// surface supports it, the swapchain will use that format. Falls back to
    /// the default selection (B8G8R8A8_UNORM + SRGB_NONLINEAR) if not
    /// available.
    pub fn new(
        parent_device: &Arc<Device>,
        parent_surface: &Arc<Surface<T>>,
        desired_extent: vk::Extent2D,
        create_image_views: bool,
        preferred_format: Option<vk::Format>,
    ) -> Result<Self, CreateSwapchainError> {
        Self::new_with_old(
            parent_device,
            parent_surface,
            desired_extent,
            None,
            create_image_views,
            preferred_format,
        )
    }

    /// Returns a richer debug view that includes parent surface source
    /// details when `T: Debug`.
    ///
    /// This keeps the base `Debug` impl available for all `T` without
    /// requiring `T: Debug`.
    pub fn debug_with_source(&self) -> impl std::fmt::Debug + '_
    where
        T: std::fmt::Debug,
    {
        SwapchainDebugWithSource(self)
    }

    /// Create a swapchain, optionally providing an old swapchain for
    /// recreation optimization.
    ///
    /// `old_swapchain`, when provided, must originate from the same
    /// `parent_device` and `parent_surface`.
    ///
    /// The caller is responsible for synchronizing GPU usage so replacing the
    /// old swapchain is safe for the application's frame lifecycle.
    ///
    /// See [`new`](Self::new) for the semantics of `preferred_format`.
    pub fn new_with_old(
        parent_device: &Arc<Device>,
        parent_surface: &Arc<Surface<T>>,
        desired_extent: vk::Extent2D,
        old_swapchain: Option<&Self>,
        create_image_views: bool,
        preferred_format: Option<vk::Format>,
    ) -> Result<Self, CreateSwapchainError> {
        if !parent_device.has_swapchain_support() {
            return Err(CreateSwapchainError::SwapchainNotEnabled);
        }

        if desired_extent.width == 0 || desired_extent.height == 0 {
            return Err(CreateSwapchainError::InvalidExtent {
                width: desired_extent.width,
                height: desired_extent.height,
            });
        }

        if !std::sync::Arc::ptr_eq(
            parent_surface.get_parent(),
            parent_device.get_parent(),
        ) {
            return Err(CreateSwapchainError::MismatchedParams);
        }

        if let Some(old_swapchain) = old_swapchain
            && (!std::sync::Arc::ptr_eq(
                &old_swapchain.parent_device,
                parent_device,
            ) || !std::sync::Arc::ptr_eq(
                &old_swapchain._parent_surface,
                parent_surface,
            ))
        {
            return Err(CreateSwapchainError::MismatchedParams);
        }

        let physical_device = parent_device.get_physical_device();
        let present_queue_family =
            parent_device.graphics_present_queue_family();

        // SAFETY: physical_device belongs to parent_device's instance, and
        // parent_surface is derived from the same instance (validated above).
        let capabilities =
            unsafe { parent_surface.query_capabilities(physical_device) }?;
        // SAFETY: same reasoning as above.
        let formats = unsafe { parent_surface.query_formats(physical_device) }?;
        // SAFETY: same reasoning as above.
        let present_modes =
            unsafe { parent_surface.query_present_modes(physical_device) }?;

        if formats.is_empty() {
            return Err(CreateSwapchainError::NoSurfaceFormats);
        }
        if present_modes.is_empty() {
            return Err(CreateSwapchainError::NoPresentModes);
        }

        let surface_format = choose_surface_format(&formats, preferred_format);
        let present_mode = choose_present_mode(&present_modes);
        let extent = choose_extent(&capabilities, desired_extent);
        let image_count = choose_image_count(&capabilities);
        let composite_alpha = choose_composite_alpha(&capabilities);

        let queue_family_indices = [present_queue_family];

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(parent_surface.raw_handle())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(
                old_swapchain
                    .map(|swapchain| swapchain.handle)
                    .unwrap_or(vk::SwapchainKHR::null()),
            );

        // SAFETY: create info references valid handles and values selected from
        // queried surface support details.
        let handle = unsafe {
            parent_device.create_raw_swapchain(&swapchain_create_info)
        }?;
        let swapchain_debug_index = parent_device.next_swapchain_debug_index();
        // SAFETY: `handle` is a valid swapchain created from `parent_device`.
        match unsafe {
            parent_device.set_object_name_with(handle, || {
                std::ffi::CString::new(format!(
                    "Swapchain {swapchain_debug_index}"
                ))
                .ok()
            })
        } {
            Ok(()) | Err(NameObjectError::DebugUtilsNotEnabled) => {}
            Err(e) => {
                tracing::warn!("Failed to name swapchain {:?}: {e}", handle)
            }
        }
        // SAFETY: handle was created by this device's swapchain loader
        // and is valid.
        let images = unsafe { parent_device.get_raw_swapchain_images(handle) }
            .inspect_err(|_| {
                // SAFETY: handle was created above and must be
                // destroyed on early exit.
                unsafe { parent_device.destroy_raw_swapchain(handle) };
            })?;

        for (index, image) in images.iter().copied().enumerate() {
            // SAFETY: image is a valid swapchain image owned by parent_device.
            match unsafe {
                parent_device.set_object_name_with(image, || {
                    std::ffi::CString::new(format!(
                        "Swapchain {swapchain_debug_index} Image {}",
                        index + 1,
                    ))
                    .ok()
                })
            } {
                Ok(()) | Err(NameObjectError::DebugUtilsNotEnabled) => {}
                Err(e) => tracing::warn!(
                    "Failed to name swapchain image {:?}: {e}",
                    image
                ),
            }
        }

        let image_views = if create_image_views {
            Some(
                create_default_swapchain_image_views(
                    &images,
                    surface_format.format,
                    swapchain_debug_index,
                    |create_info| {
                        // SAFETY: create_info references a valid
                        // swapchain image from this device, and uses a
                        // standard 2D color subresource range.
                        unsafe {
                            parent_device.create_raw_image_view(create_info)
                        }
                    },
                    |image_view| {
                        // SAFETY: image_view was created by
                        // parent_device and must be destroyed on
                        // early exit.
                        unsafe {
                            parent_device.destroy_raw_image_view(image_view)
                        };
                    },
                    |index, image_view| {
                        // SAFETY: image_view is valid and created
                        // from parent_device.
                        match unsafe {
                            parent_device.set_object_name_with(
                                image_view,
                                || {
                                    std::ffi::CString::new(format!(
                                        "Swapchain {swapchain_debug_index} \
                                     ImageView {}",
                                        index + 1,
                                    ))
                                    .ok()
                                },
                            )
                        } {
                            Ok(())
                            | Err(NameObjectError::DebugUtilsNotEnabled) => {}
                            Err(e) => tracing::warn!(
                                "Failed to name swapchain image view {:?}: {e}",
                                image_view
                            ),
                        }
                    },
                )
                .inspect_err(|_| {
                    // SAFETY: handle was created above and must be destroyed on
                    // early exit.
                    unsafe { parent_device.destroy_raw_swapchain(handle) };
                })?,
            )
        } else {
            None
        };

        // SAFETY: `handle` and `images` are created from `parent_device`, and
        // `parent_surface` is derived from the same parent instance.
        Ok(unsafe {
            Self::from_parts(
                Arc::clone(parent_device),
                Arc::clone(parent_surface),
                handle,
                surface_format.format,
                extent,
                images,
                image_views,
            )
        })
    }

    /// # Safety
    /// `handle`, `images`, and any `image_views` must all be valid resources
    /// created from `parent_device`.
    ///
    /// `parent_surface` must be derived from the same parent instance as
    /// `parent_device`.
    ///
    /// `image_views`, when `Some`, should correspond to images in
    /// `images` (same image, format compatibility, and subresource range
    /// expectations).
    ///
    /// All resources passed here must follow Vulkan destruction ordering
    /// requirements (views before swapchain).
    pub unsafe fn from_parts(
        parent_device: Arc<Device>,
        parent_surface: Arc<Surface<T>>,
        handle: vk::SwapchainKHR,
        format: vk::Format,
        extent: vk::Extent2D,
        images: Vec<vk::Image>,
        image_views: Option<Vec<vk::ImageView>>,
    ) -> Self {
        Self {
            parent_device,
            _parent_surface: parent_surface,
            handle,
            format,
            extent,
            images,
            image_views,
            acquire_lock: Mutex::new(()),
        }
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn raw_handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn image_views(&self) -> Option<&[vk::ImageView]> {
        self.image_views.as_deref()
    }

    /// Acquire the next presentable image from the swapchain.
    ///
    /// Logically mutates the swapchain (dequeues a GPU image slot), though
    /// no Rust-visible fields change. `&self` is required because the swapchain
    /// is typically shared via `Arc`.
    ///
    /// Returns `(image_index, suboptimal)`. When `suboptimal` is `true` the
    /// swapchain is still usable but recreation is recommended.
    ///
    /// Returns `Err(vk::Result::ERROR_OUT_OF_DATE_KHR)` when the swapchain is
    /// incompatible with the surface and must be recreated.
    ///
    /// # Safety
    /// `semaphore` and `fence`, when not null, must be valid unsignaled handles
    /// created from this swapchain's device.
    pub unsafe fn acquire_next_image(
        &self,
        timeout_ns: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<(u32, bool), vk::Result> {
        let _guard = self
            .acquire_lock
            .lock()
            .expect("swapchain acquire lock poisoned");
        // SAFETY: Caller guarantees semaphore and fence validity. self.handle
        // is valid for the lifetime of this Swapchain.
        unsafe {
            self.parent_device.acquire_next_swapchain_image(
                self.handle,
                timeout_ns,
                semaphore,
                fence,
            )
        }
    }
}

impl<T: HasDisplayHandle + HasWindowHandle> Drop for Swapchain<T> {
    fn drop(&mut self) {
        tracing::debug!("Dropping swapchain {:?}", self.handle);
        // NOTE: Callers must ensure GPU synchronization before drop (for
        // example, waiting on fences/device idle) so no in-flight work still
        // references these views or the swapchain.
        for image_view in self.image_views.iter_mut().flat_map(|v| v.drain(..))
        {
            // SAFETY: image_view was created by parent_device and is being
            // destroyed during swapchain teardown.
            unsafe { self.parent_device.destroy_raw_image_view(image_view) };
        }
        // SAFETY: swapchain handle was created by parent_device and this is
        // the final destruction path for this wrapper.
        unsafe { self.parent_device.destroy_raw_swapchain(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use std::cell::RefCell;

    #[test]
    fn choose_surface_format_prefers_bgra_srgb() {
        let fallback = vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let preferred = vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };

        let chosen = choose_surface_format(&[fallback, preferred], None);
        assert_eq!(chosen.format, preferred.format);
        assert_eq!(chosen.color_space, preferred.color_space);
    }

    #[test]
    fn choose_present_mode_prefers_mailbox() {
        let chosen = choose_present_mode(&[
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
        ]);
        assert_eq!(chosen, vk::PresentModeKHR::MAILBOX);
    }

    #[test]
    fn choose_present_mode_falls_back_to_fifo() {
        let chosen = choose_present_mode(&[vk::PresentModeKHR::IMMEDIATE]);
        assert_eq!(chosen, vk::PresentModeKHR::FIFO);
    }

    #[test]
    fn choose_extent_uses_current_when_fixed() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 1280,
                height: 720,
            },
            ..Default::default()
        };

        let chosen = choose_extent(
            &capabilities,
            vk::Extent2D {
                width: 1920,
                height: 1080,
            },
        );

        assert_eq!(chosen.width, 1280);
        assert_eq!(chosen.height, 720);
    }

    #[test]
    fn choose_extent_clamps_when_variable() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 640,
                height: 480,
            },
            max_image_extent: vk::Extent2D {
                width: 1920,
                height: 1080,
            },
            ..Default::default()
        };

        let chosen = choose_extent(
            &capabilities,
            vk::Extent2D {
                width: 4000,
                height: 200,
            },
        );

        assert_eq!(chosen.width, 1920);
        assert_eq!(chosen.height, 480);
    }

    #[test]
    fn choose_image_count_respects_max_when_set() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            min_image_count: 3,
            max_image_count: 3,
            ..Default::default()
        };

        assert_eq!(choose_image_count(&capabilities), 3);
    }

    #[test]
    fn choose_composite_alpha_prefers_opaque_then_pre_multiplied() {
        let capabilities = vk::SurfaceCapabilitiesKHR {
            supported_composite_alpha:
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
                    | vk::CompositeAlphaFlagsKHR::OPAQUE,
            ..Default::default()
        };

        assert_eq!(
            choose_composite_alpha(&capabilities),
            vk::CompositeAlphaFlagsKHR::OPAQUE
        );
    }

    #[test]
    fn image_view_helper_cleans_up_on_partial_failure() {
        let images = [
            vk::Image::from_raw(1),
            vk::Image::from_raw(2),
            vk::Image::from_raw(3),
        ];
        let created_views =
            [vk::ImageView::from_raw(10), vk::ImageView::from_raw(11)];
        let create_calls = RefCell::new(0usize);
        let destroyed = RefCell::new(Vec::<vk::ImageView>::new());

        let result = create_default_swapchain_image_views(
            &images,
            vk::Format::B8G8R8A8_UNORM,
            1,
            |_| {
                let mut call = create_calls.borrow_mut();
                let ret = match *call {
                    0 => Ok(created_views[0]),
                    _ => Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
                };
                *call += 1;
                ret
            },
            |view| destroyed.borrow_mut().push(view),
            |_index, _view| {},
        );

        assert!(matches!(
            result,
            Err(CreateSwapchainError::VulkanCreateImageView(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            ))
        ));
        assert_eq!(destroyed.borrow().as_slice(), &[created_views[0]]);
    }

    #[test]
    fn image_view_helper_returns_all_views_on_success() {
        let images = [vk::Image::from_raw(1), vk::Image::from_raw(2)];
        let views =
            [vk::ImageView::from_raw(100), vk::ImageView::from_raw(101)];
        let create_calls = RefCell::new(0usize);
        let name_calls = RefCell::new(0usize);

        let result = create_default_swapchain_image_views(
            &images,
            vk::Format::B8G8R8A8_UNORM,
            2,
            |_| {
                let mut call = create_calls.borrow_mut();
                let view = views[*call];
                *call += 1;
                Ok(view)
            },
            |_view| panic!("destroy callback should not be called on success"),
            |_index, _view| {
                *name_calls.borrow_mut() += 1;
            },
        )
        .expect("helper should succeed");

        assert_eq!(result, views);
        assert_eq!(*name_calls.borrow(), 2);
    }
}
