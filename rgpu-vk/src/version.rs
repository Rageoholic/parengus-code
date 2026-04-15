//! Vulkan-specific version encoding.
//!
//! [`VkVersion`] wraps a [`parengus_util::Version`] plus the Vulkan
//! variant bits (always 0 for standard Vulkan). It decodes from and
//! encodes to the packed `u32` used in `VkApplicationInfo` and
//! `vkEnumerateInstanceVersion`.

use ash::vk;
use parengus_util::Version;

/// A Vulkan API version number.
///
/// The semver components are exposed via the public `version` field;
/// the Vulkan variant bits (always 0 for standard Vulkan) are
/// preserved in `variant`.
///
/// Construct from a raw Vulkan word with [`from_raw`](Self::from_raw),
/// or from parts with [`new`](Self::new). Convert back with
/// [`to_raw`](Self::to_raw).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VkVersion {
    pub version: Version,
    pub variant: u32,
}

impl VkVersion {
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        Self {
            version: Version::new(
                vk::api_version_major(raw) as u16,
                vk::api_version_minor(raw),
                vk::api_version_patch(raw) as u16,
            ),
            variant: vk::api_version_variant(raw),
        }
    }

    #[inline]
    pub fn new(variant: u32, major: u16, minor: u32, patch: u16) -> Self {
        Self {
            version: Version::new(major, minor, patch),
            variant,
        }
    }

    #[inline]
    pub fn to_raw(self) -> u32 {
        vk::make_api_version(
            self.variant,
            self.version.major as u32,
            self.version.minor,
            self.version.patch as u32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_version_raw_roundtrip() {
        let raw = vk::make_api_version(0, 1, 3, 275);
        let v = VkVersion::from_raw(raw);

        assert_eq!(v.to_raw(), raw);
        assert_eq!(v.variant, 0);
        assert_eq!(v.version.major, 1);
        assert_eq!(v.version.minor, 3);
        assert_eq!(v.version.patch, 275);
    }

    #[test]
    fn vk_version_new_roundtrip() {
        let v = VkVersion::new(1, 2, 3, 4);
        let rebuilt = VkVersion::from_raw(v.to_raw());
        assert_eq!(v, rebuilt);
    }
}
