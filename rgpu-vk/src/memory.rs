//! Builder helpers for Vulkan memory-barrier structs.
//!
//! `ash` implements [`Default`] for Vulkan structs manually so that
//! `s_type` is set correctly, but leaves all other fields at their
//! zero values.  For all four barrier types this means
//! `src_queue_family_index` and `dst_queue_family_index` default to
//! **0** rather than [`vk::QUEUE_FAMILY_IGNORED`].  While `0/0` is
//! functionally harmless (equal indices → no ownership transfer), it
//! obscures intent and is a copy-paste hazard.
//!
//! Prefer the helpers in this module over calling barrier struct
//! `Default` implementations directly.

use ash::vk;

/// Start building a [`vk::ImageMemoryBarrier`] that does not
/// transfer queue-family ownership.
///
/// Equivalent to [`vk::ImageMemoryBarrier::default()`] except that
/// both `src_queue_family_index` and `dst_queue_family_index` are
/// pre-set to [`vk::QUEUE_FAMILY_IGNORED`], which is the Vulkan
/// idiom for "this barrier performs no queue-family ownership
/// transfer."  Fill in the remaining fields (stage/access masks,
/// layouts, image, subresource range) with the builder methods
/// before use.
pub fn image_barrier<'a>() -> vk::ImageMemoryBarrier<'a> {
    vk::ImageMemoryBarrier::default()
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}

/// Start building a [`vk::ImageMemoryBarrier2`] that does not
/// transfer queue-family ownership.
///
/// Equivalent to [`vk::ImageMemoryBarrier2::default()`] except that
/// both `src_queue_family_index` and `dst_queue_family_index` are
/// pre-set to [`vk::QUEUE_FAMILY_IGNORED`], which is the Vulkan
/// idiom for "this barrier performs no queue-family ownership
/// transfer."  Fill in the remaining fields (stage/access masks,
/// layouts, image, subresource range) with the builder methods
/// before use.
pub fn image_barrier2<'a>() -> vk::ImageMemoryBarrier2<'a> {
    vk::ImageMemoryBarrier2::default()
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}

/// Start building a [`vk::BufferMemoryBarrier`] that does not
/// transfer queue-family ownership.
///
/// Equivalent to [`vk::BufferMemoryBarrier::default()`] except that
/// both `src_queue_family_index` and `dst_queue_family_index` are
/// pre-set to [`vk::QUEUE_FAMILY_IGNORED`], which is the Vulkan
/// idiom for "this barrier performs no queue-family ownership
/// transfer."  Fill in the remaining fields (stage/access masks,
/// buffer, offset, size) with the builder methods before use.
pub fn buffer_barrier<'a>() -> vk::BufferMemoryBarrier<'a> {
    vk::BufferMemoryBarrier::default()
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}

/// Start building a [`vk::BufferMemoryBarrier2`] that does not
/// transfer queue-family ownership.
///
/// Equivalent to [`vk::BufferMemoryBarrier2::default()`] except that
/// both `src_queue_family_index` and `dst_queue_family_index` are
/// pre-set to [`vk::QUEUE_FAMILY_IGNORED`], which is the Vulkan
/// idiom for "this barrier performs no queue-family ownership
/// transfer."  Fill in the remaining fields (stage/access masks,
/// buffer, offset, size) with the builder methods before use.
pub fn buffer_barrier2<'a>() -> vk::BufferMemoryBarrier2<'a> {
    vk::BufferMemoryBarrier2::default()
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}
