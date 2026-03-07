//! Render pass wrapper ([`RenderPass`]).
//!
//! [`RenderPass`] wraps a `VkRenderPass` for use with the classic Vulkan
//! render pass API (as opposed to `VK_KHR_dynamic_rendering`). It describes
//! a single subpass with one color attachment and one depth/stencil
//! attachment.
//!
//! Framebuffer management is handled by [`Swapchain::create_framebuffers`],
//! which creates one `VkFramebuffer` per swapchain image using the render
//! pass and caller-supplied depth views.

use std::sync::Arc;

use ash::vk;

use crate::device::Device;

// ---------------------------------------------------------------------------
// RenderPassDesc
// ---------------------------------------------------------------------------

/// Describes a single-subpass render pass with one color and one
/// depth/stencil attachment.
///
/// # Attachment behaviour
///
/// | Attachment | `initial_layout`  | `final_layout`                      | load op | store op  |
/// |------------|-------------------|-------------------------------------|---------|-----------|
/// | Color      | `UNDEFINED`       | `PRESENT_SRC_KHR`                   | `CLEAR` | `STORE`   |
/// | Depth      | `UNDEFINED`       | `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`  | `CLEAR` | `DONT_CARE` |
///
/// Setting `initial_layout = UNDEFINED` discards previous contents, which
/// is valid when `load_op = CLEAR`. The color attachment's `final_layout =
/// PRESENT_SRC_KHR` means the render pass end automatically transitions the
/// swapchain image to a presentable layout — no explicit post-render barrier
/// is needed.
///
/// A single subpass dependency (`EXTERNAL` → subpass 0) provides the
/// necessary stage and access synchronisation for both attachments.
pub struct RenderPassDesc {
    /// Format of the color attachment (typically the swapchain format).
    pub color_format: vk::Format,
    /// Format of the depth/stencil attachment.
    pub depth_format: vk::Format,
}

// ---------------------------------------------------------------------------
// RenderPass
// ---------------------------------------------------------------------------

/// An owned wrapper around a `VkRenderPass`.
///
/// Describes a single subpass with one color attachment (swapchain image)
/// and one depth/stencil attachment. See [`RenderPassDesc`] for the fixed
/// attachment layout and load/store operations.
pub struct RenderPass {
    parent: Arc<Device>,
    handle: vk::RenderPass,
}

impl std::fmt::Debug for RenderPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPass")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl RenderPass {
    /// Create a render pass from a [`RenderPassDesc`].
    pub fn new(
        device: &Arc<Device>,
        desc: &RenderPassDesc,
    ) -> Result<Self, vk::Result> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(desc.color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let depth_attachment = vk::AttachmentDescription::default()
            .format(desc.depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let attachments = [color_attachment, depth_attachment];

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        // Subpass dependency: EXTERNAL → subpass 0.
        //
        // Color: wait for COLOR_ATTACHMENT_OUTPUT before writing to the
        // color attachment. This covers the semaphore-signaled stage from
        // vkAcquireNextImageKHR and the final_layout transition to
        // PRESENT_SRC_KHR at render pass end.
        //
        // Depth: wait for EARLY/LATE_FRAGMENT_TESTS before
        // reading/writing the depth attachment.
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        // SAFETY: create_info is fully initialised from validated
        // format and layout values above.
        let handle = unsafe {
            device.ash_device().create_render_pass(&create_info, None)
        }?;

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_render_pass(&self) -> vk::RenderPass {
        self.handle
    }

    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        tracing::debug!("Dropping render pass {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed.
        // All framebuffers and pipelines referencing this render pass
        // must be destroyed first (caller's responsibility via Arc
        // lifetimes and drop ordering).
        unsafe {
            self.parent
                .ash_device()
                .destroy_render_pass(self.handle, None)
        };
    }
}
