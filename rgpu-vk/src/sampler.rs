//! Sampler wrapper ([`Sampler`]).
//!
//! A sampler encodes texture filtering and addressing state independently
//! of any particular image. One sampler can be reused with many image views.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;

/// An owned `VkSampler`.
pub struct Sampler {
    parent: Arc<Device>,
    handle: vk::Sampler,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Sampler {
    /// Create a sampler.
    ///
    /// `mag_filter` and `min_filter` control the magnification/minification
    /// filters. `address_mode` is applied to all three axes (U/V/W).
    /// Anisotropy and mip-mapping are disabled.
    pub fn new(
        device: &Arc<Device>,
        mag_filter: vk::Filter,
        min_filter: vk::Filter,
        address_mode: vk::SamplerAddressMode,
        name: Option<&str>,
    ) -> Result<Self, vk::Result> {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .anisotropy_enable(false)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);

        // SAFETY: create_info is fully initialised with no borrowed data.
        let handle =
            unsafe { device.create_raw_sampler(&create_info) }?;

        // SAFETY: handle is a valid sampler from this device.
        let name_result =
            unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name sampler {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_sampler(&self) -> vk::Sampler {
        self.handle
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        tracing::debug!("Dropping sampler {:?}", self.handle);
        // SAFETY: handle was created from parent and is owned by this
        // wrapper. No GPU work may still reference it.
        unsafe { self.parent.destroy_raw_sampler(self.handle) };
    }
}
