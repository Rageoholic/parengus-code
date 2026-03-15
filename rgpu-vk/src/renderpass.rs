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
    /// MSAA sample count. Use `TYPE_1` to disable multisampling.
    pub sample_count: vk::SampleCountFlags,
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
    ///
    /// When `desc.sample_count` is `TYPE_1` the classic 2-attachment
    /// layout (color → PRESENT, depth) is used. For any higher count
    /// a 3-attachment MSAA layout is used: MSAA color (slot 0), depth
    /// (slot 1), and a single-sample resolve (slot 2, PRESENT_SRC_KHR).
    pub fn new(
        device: &Arc<Device>,
        desc: &RenderPassDesc,
    ) -> Result<Self, vk::Result> {
        let handle = if desc.sample_count == vk::SampleCountFlags::TYPE_1 {
            Self::create_single_sample(device, desc)?
        } else {
            Self::create_multisampled(device, desc)?
        };

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    fn create_single_sample(
        device: &Arc<Device>,
        desc: &RenderPassDesc,
    ) -> Result<vk::RenderPass, vk::Result> {
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

        let dependency = Self::make_dependency();

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        // SAFETY: create_info is fully initialised from validated
        // format and layout values above.
        unsafe { device.ash_device().create_render_pass(&create_info, None) }
    }

    fn create_multisampled(
        device: &Arc<Device>,
        desc: &RenderPassDesc,
    ) -> Result<vk::RenderPass, vk::Result> {
        // Slot 0: MSAA colour — transient, resolved at subpass end.
        let msaa_color = vk::AttachmentDescription::default()
            .format(desc.color_format)
            .samples(desc.sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        // Slot 1: depth.
        let depth_attachment = vk::AttachmentDescription::default()
            .format(desc.depth_format)
            .samples(desc.sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        // Slot 2: resolve target (swapchain image).
        let resolve_attachment = vk::AttachmentDescription::default()
            .format(desc.color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachments = [msaa_color, depth_attachment, resolve_attachment];

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let resolve_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref)
            .resolve_attachments(std::slice::from_ref(&resolve_ref));

        let dependency = Self::make_dependency();

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        // SAFETY: create_info is fully initialised from validated
        // format and layout values above.
        unsafe { device.ash_device().create_render_pass(&create_info, None) }
    }

    /// Subpass dependency: EXTERNAL → subpass 0, covering colour and
    /// depth stages/accesses for both single-sample and MSAA paths.
    fn make_dependency() -> vk::SubpassDependency {
        vk::SubpassDependency::default()
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
            )
    }

    #[inline]
    pub fn raw_render_pass(&self) -> vk::RenderPass {
        self.handle
    }

    #[inline]
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
