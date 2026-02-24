use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::device::Device;
use crate::shader::EntryPoint;

// ---------------------------------------------------------------------------
// PipelineLayout
// ---------------------------------------------------------------------------

/// An owned wrapper around a `VkPipelineLayout`.
///
/// Multiple pipelines that share the same descriptor set and push-constant
/// signature can hold the layout behind an `Arc<PipelineLayout>`.
pub struct PipelineLayout {
    parent: Arc<Device>,
    handle: vk::PipelineLayout,
}

impl std::fmt::Debug for PipelineLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineLayout")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl PipelineLayout {
    /// Create an empty pipeline layout with no descriptor sets and no push
    /// constant ranges.
    pub fn new_empty(device: &Arc<Device>) -> Result<Self, vk::Result> {
        let create_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: create_info is default-initialised; it imposes no additional
        // validity requirements on the device.
        let handle =
            unsafe { device.create_raw_pipeline_layout(&create_info) }?;
        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        tracing::debug!("Dropping pipeline layout {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed during
        // teardown. All pipelines using this layout must be dropped first.
        unsafe { self.parent.destroy_raw_pipeline_layout(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CreateDynamicPipelineError {
    #[error("No shader stages provided")]
    NoStages,

    #[error("Vulkan error creating empty pipeline layout: {0}")]
    LayoutCreation(vk::Result),

    #[error("Vulkan error creating graphics pipeline: {0}")]
    PipelineCreation(vk::Result),
}

// ---------------------------------------------------------------------------
// DynamicPipelineDesc
// ---------------------------------------------------------------------------

/// Description of a [`DynamicPipeline`] to create.
///
/// # Defaults (via [`Default`])
/// | field | default |
/// |---|---|
/// | `stages` | `&[]` (must be overridden) |
/// | `color_attachment_formats` | `&[]` |
/// | `depth_attachment_format` | `None` |
/// | `layout` | `None` (empty layout is created internally) |
/// | `polygon_mode` | `FILL` |
/// | `cull_mode` | `NONE` |
/// | `front_face` | `COUNTER_CLOCKWISE` |
pub struct DynamicPipelineDesc<'a> {
    /// Shader entry points that form this pipeline's stages.
    ///
    /// Must contain at least one entry.
    pub stages: &'a [EntryPoint<'a>],

    /// Formats of the color attachments used in the dynamic render pass.
    ///
    /// Must match the attachments passed to `vkCmdBeginRendering` at draw
    /// time. Use an empty slice when rendering with no color output.
    pub color_attachment_formats: &'a [vk::Format],

    /// Format of the depth attachment. `None` means no depth attachment.
    pub depth_attachment_format: Option<vk::Format>,

    /// Pipeline layout to use.
    ///
    /// When `None` an empty layout (no descriptor sets, no push constants) is
    /// created internally and owned exclusively by the resulting pipeline. Pass
    /// an `Arc<PipelineLayout>` to share a layout across multiple pipelines.
    pub layout: Option<Arc<PipelineLayout>>,

    /// Polygon fill mode used by the rasterizer.
    pub polygon_mode: vk::PolygonMode,

    /// Face culling mode.
    pub cull_mode: vk::CullModeFlags,

    /// Winding order considered front-facing.
    pub front_face: vk::FrontFace,
}

impl Default for DynamicPipelineDesc<'_> {
    fn default() -> Self {
        Self {
            stages: &[],
            color_attachment_formats: &[],
            depth_attachment_format: None,
            layout: None,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}

// ---------------------------------------------------------------------------
// DynamicPipeline
// ---------------------------------------------------------------------------

/// A graphics pipeline built for use with dynamic rendering
/// (`VK_KHR_dynamic_rendering` / Vulkan 1.3 core).
///
/// No render pass object is required. The matching `VkRenderingInfo` must be
/// set up by the caller before issuing draw calls.
///
/// Fixed pipeline state applied during construction:
/// - Vertex input: no vertex buffers (drive vertices from push constants or
///   buffer device address)
/// - Input assembly: `TRIANGLE_LIST`
/// - Viewport/scissor: fully dynamic (`VK_DYNAMIC_STATE_VIEWPORT` +
///   `VK_DYNAMIC_STATE_SCISSOR`)
/// - Rasterization: configurable polygon mode, cull mode, front face;
///   line width fixed at 1.0
/// - Multisample: single sample, no sample shading
/// - Depth/stencil: test and write disabled
/// - Color blend: no blending, full RGBA write mask per color attachment
pub struct DynamicPipeline {
    parent: Arc<Device>,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

impl std::fmt::Debug for DynamicPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicPipeline")
            .field("handle", &self.handle)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl DynamicPipeline {
    /// Create a [`DynamicPipeline`] from a description.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils` when
    /// the extension is available. The closure is only called when the
    /// extension is enabled. Naming failures are logged as warnings and
    /// do not cause the call to fail.
    pub fn new<F>(
        device: &Arc<Device>,
        desc: &DynamicPipelineDesc<'_>,
        name: Option<F>,
    ) -> Result<Self, CreateDynamicPipelineError>
    where
        F: FnOnce() -> String,
    {
        if desc.stages.is_empty() {
            return Err(CreateDynamicPipelineError::NoStages);
        }

        let layout = match &desc.layout {
            Some(l) => Arc::clone(l),
            None => Arc::new(
                PipelineLayout::new_empty(device)
                    .map_err(CreateDynamicPipelineError::LayoutCreation)?,
            ),
        };

        let stage_create_infos: Vec<vk::PipelineShaderStageCreateInfo<'_>> =
            desc.stages
                .iter()
                .map(|ep| ep.as_pipeline_stage_create_info())
                .collect();

        let vertex_input_state =
            vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly_state =
            vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        // Viewport and scissor counts must be declared even though their
        // values are supplied dynamically.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_state =
            vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(desc.polygon_mode)
                .cull_mode(desc.cull_mode)
                .front_face(desc.front_face)
                .line_width(1.0);

        let multisample_state =
            vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state =
            vk::PipelineDepthStencilStateCreateInfo::default();

        let color_blend_attachments: Vec<
            vk::PipelineColorBlendAttachmentState,
        > = desc
            .color_attachment_formats
            .iter()
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();

        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_blend_attachments);

        let dynamic_states =
            [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let mut rendering_create_info =
            vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(desc.color_attachment_formats)
                .depth_attachment_format(
                    desc.depth_attachment_format
                        .unwrap_or(vk::Format::UNDEFINED),
                );

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(layout.raw_handle())
            .push_next(&mut rendering_create_info);

        // SAFETY: create_info references valid shader stages, a valid pipeline
        // layout, and a valid VkPipelineRenderingCreateInfo; all derived from
        // device and valid for the duration of this call.
        let handle =
            unsafe { device.create_raw_graphics_pipeline(&create_info) }
                .map_err(CreateDynamicPipelineError::PipelineCreation)?;

        // SAFETY: handle is a valid pipeline created from device.
        let name_result = unsafe {
            device.set_object_name_with(handle, || {
                std::ffi::CString::new(name?()).ok()
            })
        };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name pipeline {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
            layout,
        })
    }

    pub fn raw_handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
}

impl Drop for DynamicPipeline {
    fn drop(&mut self) {
        tracing::debug!("Dropping pipeline {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed during
        // teardown. All in-flight GPU work referencing this pipeline must be
        // completed before drop.
        unsafe { self.parent.destroy_raw_pipeline(self.handle) };
        // self.layout Arc is released here; the layout itself is destroyed
        // only when all pipelines sharing it have been dropped.
    }
}
