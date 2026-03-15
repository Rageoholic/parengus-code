//! Graphics pipeline and layout wrappers.
//!
//! Two pipeline types are provided:
//!
//! - [`DynamicPipeline`] — requires no `VkRenderPass` object; built for
//!   use with `VK_KHR_dynamic_rendering` (Vulkan 1.3 core). Render pass
//!   information is provided at draw time via `vkCmdBeginRendering`.
//! - [`RenderPassPipeline`] — paired with a `VkRenderPass` (classic
//!   Vulkan 1.0 API). Render pass and subpass index are baked in at
//!   pipeline creation time.
//!
//! [`PipelineLayout`] wraps a `VkPipelineLayout` and can be shared
//! across multiple pipelines via `Arc`. When no layout is provided in
//! the pipeline desc, an empty one is created internally.

use std::sync::Arc;

use ash::vk;
use thiserror::Error;

pub use vk::{CullModeFlags, FrontFace, VertexInputRate};

use crate::descriptor::DescriptorSetLayout;
use crate::device::Device;
use crate::shader::EntryPoint;

// ---------------------------------------------------------------------------
// PipelineLayoutDesc
// ---------------------------------------------------------------------------

/// Describes the full signature of a [`PipelineLayout`]: descriptor set
/// layouts and push constant ranges.
///
/// # Defaults (via [`Default`])
/// | field | default |
/// |---|---|
/// | `set_layouts` | `&[]` |
/// | `push_constant_ranges` | `&[]` |
#[derive(Default)]
pub struct PipelineLayoutDesc<'a> {
    /// Descriptor set layouts, in set-index order.
    pub set_layouts: &'a [&'a DescriptorSetLayout],
    /// Push constant ranges accessible by the pipeline stages.
    pub push_constant_ranges: &'a [vk::PushConstantRange],
}

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
    /// Create a pipeline layout from a [`PipelineLayoutDesc`].
    ///
    /// Use [`PipelineLayoutDesc::default`] for an empty layout (no
    /// descriptor sets, no push constants).
    pub fn new(
        device: &Arc<Device>,
        desc: &PipelineLayoutDesc<'_>,
    ) -> Result<Self, vk::Result> {
        let raw_set_layouts: Vec<vk::DescriptorSetLayout> = desc
            .set_layouts
            .iter()
            .map(|l| l.raw_descriptor_set_layout())
            .collect();
        let create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&raw_set_layouts)
            .push_constant_ranges(desc.push_constant_ranges);
        // SAFETY: create_info references valid set layouts and push
        // constant ranges for the duration of this call.
        let handle =
            unsafe { device.create_raw_pipeline_layout(&create_info) }?;
        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    /// Create an empty pipeline layout with no descriptor sets and no push
    /// constant ranges.
    pub fn new_empty(device: &Arc<Device>) -> Result<Self, vk::Result> {
        Self::new(device, &PipelineLayoutDesc::default())
    }

    #[inline]
    pub fn raw_pipeline_layout(&self) -> vk::PipelineLayout {
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

/// Describes one vertex buffer binding slot for a [`DynamicPipeline`].
#[derive(Debug, Clone, Copy)]
pub struct VertexBindingDesc {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
}

impl From<VertexBindingDesc> for vk::VertexInputBindingDescription {
    #[inline]
    fn from(value: VertexBindingDesc) -> Self {
        vk::VertexInputBindingDescription {
            binding: value.binding,
            stride: value.stride,
            input_rate: value.input_rate,
        }
    }
}

/// Describes one vertex attribute consumed by the vertex shader.
#[derive(Debug, Clone, Copy)]
pub struct VertexAttributeDesc {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

impl From<VertexAttributeDesc> for vk::VertexInputAttributeDescription {
    #[inline]
    fn from(value: VertexAttributeDesc) -> Self {
        vk::VertexInputAttributeDescription {
            location: value.location,
            binding: value.binding,
            format: value.format,
            offset: value.offset,
        }
    }
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
/// | `depth_compare_op` | `LESS_OR_EQUAL` |
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

    /// Format of the depth attachment. `None` means no depth attachment,
    /// and depth test/write are disabled regardless of `depth_compare_op`.
    pub depth_attachment_format: Option<vk::Format>,

    /// Depth comparison operator used when depth testing is enabled.
    ///
    /// Use `LESS_OR_EQUAL` for standard depth (near=0, far=1) and
    /// `GREATER_OR_EQUAL` for reversed depth (near=1, far=0).
    /// Ignored when `depth_attachment_format` is `None`.
    pub depth_compare_op: vk::CompareOp,

    /// Vertex buffer binding declarations.
    pub vertex_bindings: &'a [VertexBindingDesc],

    /// Vertex attribute declarations consumed by the vertex shader.
    pub vertex_attributes: &'a [VertexAttributeDesc],

    /// Pipeline layout to use.
    ///
    /// When `None` an empty layout (no descriptor sets, no push constants)
    /// is created internally and owned exclusively by the resulting
    /// pipeline. Pass an `Arc<PipelineLayout>` to share a layout across
    /// multiple pipelines.
    pub layout: Option<Arc<PipelineLayout>>,

    /// Polygon fill mode used by the rasterizer.
    pub polygon_mode: vk::PolygonMode,

    /// Face culling mode.
    pub cull_mode: vk::CullModeFlags,

    /// Winding order considered front-facing.
    pub front_face: vk::FrontFace,

    /// MSAA sample count for the color and depth attachments.
    /// Must match the sample count used by the render target images
    /// and the `VkRenderingInfo` at draw time.
    /// Use `TYPE_1` to disable multisampling.
    pub sample_count: vk::SampleCountFlags,
}

impl Default for DynamicPipelineDesc<'_> {
    fn default() -> Self {
        Self {
            stages: &[],
            color_attachment_formats: &[],
            depth_attachment_format: None,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            vertex_bindings: &[],
            vertex_attributes: &[],
            layout: None,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            sample_count: vk::SampleCountFlags::TYPE_1,
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
/// - Vertex input: configurable via `vertex_bindings` and
///   `vertex_attributes`
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

        let vertex_bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_bindings
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        let vertex_attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_attributes
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        let vertex_input_state =
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_bindings)
                .vertex_attribute_descriptions(&vertex_attributes);

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
                .rasterization_samples(desc.sample_count);

        let depth_stencil_state = if desc.depth_attachment_format.is_some() {
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(desc.depth_compare_op)
        } else {
            vk::PipelineDepthStencilStateCreateInfo::default()
        };

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
            .layout(layout.raw_pipeline_layout())
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

    #[inline]
    pub fn raw_pipeline(&self) -> vk::Pipeline {
        self.handle
    }

    #[inline]
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

// ---------------------------------------------------------------------------
// RenderPassPipelineDesc
// ---------------------------------------------------------------------------

/// Description of a [`RenderPassPipeline`] to create.
///
/// Attachment formats are inferred from the render pass; `depth_test` and
/// `depth_write` control whether the depth/stencil state is active.
///
/// # Defaults (via [`Default`])
/// | field | default |
/// |---|---|
/// | `stages` | `&[]` (must be overridden) |
/// | `render_pass` | `vk::RenderPass::null()` (must be overridden) |
/// | `subpass` | `0` |
/// | `layout` | `None` (empty layout is created internally) |
/// | `vertex_bindings` | `&[]` |
/// | `vertex_attributes` | `&[]` |
/// | `depth_test` | `false` |
/// | `depth_write` | `false` |
/// | `depth_compare_op` | `LESS_OR_EQUAL` |
/// | `polygon_mode` | `FILL` |
/// | `cull_mode` | `NONE` |
/// | `front_face` | `COUNTER_CLOCKWISE` |
pub struct RenderPassPipelineDesc<'a> {
    /// Shader entry points that form this pipeline's stages.
    ///
    /// Must contain at least one entry.
    pub stages: &'a [EntryPoint<'a>],

    /// The render pass this pipeline will be used with.
    pub render_pass: vk::RenderPass,

    /// Index of the subpass within `render_pass`.
    pub subpass: u32,

    /// Pipeline layout to use.
    ///
    /// When `None` an empty layout (no descriptor sets, no push constants)
    /// is created internally and owned exclusively by the resulting
    /// pipeline. Pass an `Arc<PipelineLayout>` to share a layout across
    /// multiple pipelines.
    pub layout: Option<Arc<PipelineLayout>>,

    /// Vertex buffer binding declarations.
    pub vertex_bindings: &'a [VertexBindingDesc],

    /// Vertex attribute declarations consumed by the vertex shader.
    pub vertex_attributes: &'a [VertexAttributeDesc],

    /// Enable depth testing.
    pub depth_test: bool,

    /// Enable depth writes.
    pub depth_write: bool,

    /// Depth comparison operator used when depth testing is enabled.
    pub depth_compare_op: vk::CompareOp,

    /// Polygon fill mode used by the rasterizer.
    pub polygon_mode: vk::PolygonMode,

    /// Face culling mode.
    pub cull_mode: vk::CullModeFlags,

    /// Winding order considered front-facing.
    pub front_face: vk::FrontFace,

    /// MSAA sample count. Must match the render pass `sample_count`
    /// (i.e. the number of samples used by its attachments).
    /// Use `TYPE_1` to disable multisampling.
    pub sample_count: vk::SampleCountFlags,
}

impl Default for RenderPassPipelineDesc<'_> {
    fn default() -> Self {
        Self {
            stages: &[],
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            layout: None,
            vertex_bindings: &[],
            vertex_attributes: &[],
            depth_test: false,
            depth_write: false,
            depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            sample_count: vk::SampleCountFlags::TYPE_1,
        }
    }
}

// ---------------------------------------------------------------------------
// RenderPassPipeline
// ---------------------------------------------------------------------------

/// A graphics pipeline paired with a `VkRenderPass` (classic Vulkan 1.0
/// API).
///
/// The render pass and subpass index are baked in at pipeline creation
/// time. This pipeline must only be used inside a render pass begun with
/// the compatible `VkRenderPass`.
///
/// Fixed pipeline state applied during construction:
/// - Vertex input: configurable via `vertex_bindings` and
///   `vertex_attributes`
/// - Input assembly: `TRIANGLE_LIST`
/// - Viewport/scissor: fully dynamic (`VK_DYNAMIC_STATE_VIEWPORT` +
///   `VK_DYNAMIC_STATE_SCISSOR`)
/// - Rasterization: configurable polygon mode, cull mode, front face;
///   line width fixed at 1.0
/// - Multisample: single sample, no sample shading
/// - Depth/stencil: controlled by `depth_test` and `depth_write`
/// - Color blend: no blending, full RGBA write mask on one color
///   attachment
pub struct RenderPassPipeline {
    parent: Arc<Device>,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

impl std::fmt::Debug for RenderPassPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderPassPipeline")
            .field("handle", &self.handle)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl RenderPassPipeline {
    /// Create a [`RenderPassPipeline`] from a description.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils`
    /// when the extension is available. The closure is only called when
    /// the extension is enabled. Naming failures are logged as warnings
    /// and do not cause the call to fail.
    pub fn new<F>(
        device: &Arc<Device>,
        desc: &RenderPassPipelineDesc<'_>,
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

        let vertex_bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_bindings
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        let vertex_attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_attributes
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        let vertex_input_state =
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_bindings)
                .vertex_attribute_descriptions(&vertex_attributes);

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
                .rasterization_samples(desc.sample_count);

        let depth_stencil_state =
            vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(desc.depth_test)
                .depth_write_enable(desc.depth_write)
                .depth_compare_op(desc.depth_compare_op);

        // One color blend attachment to match the single color attachment
        // declared in the render pass.
        let color_blend_attachment =
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states =
            [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

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
            .layout(layout.raw_pipeline_layout())
            .render_pass(desc.render_pass)
            .subpass(desc.subpass);

        // SAFETY: create_info references valid shader stages, a valid
        // pipeline layout, and a valid VkRenderPass; all derived from
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

    #[inline]
    pub fn raw_pipeline(&self) -> vk::Pipeline {
        self.handle
    }

    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }
}

impl Drop for RenderPassPipeline {
    fn drop(&mut self) {
        tracing::debug!("Dropping pipeline {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed
        // during teardown. All in-flight GPU work referencing this
        // pipeline must be completed before drop.
        unsafe { self.parent.destroy_raw_pipeline(self.handle) };
        // self.layout Arc is released here; the layout itself is
        // destroyed only when all pipelines sharing it have been dropped.
    }
}
