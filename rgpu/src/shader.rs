use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::device::Device;

#[derive(Debug, Error)]
pub enum CreateShaderModuleError {
    #[error("SPIR-V byte slice length ({0}) is not a multiple of 4")]
    InvalidLength(usize),

    #[error("Vulkan error creating shader module: {0}")]
    Vulkan(vk::Result),
}

/// A single pipeline stage that an entry point can be compiled for.
///
/// Used with [`ShaderModule::entry_point`] to select which stage a SPIR-V
/// entry point belongs to. Unlike [`vk::ShaderStageFlags`], this enum can only
/// represent one stage at a time, matching the semantics of a single entry
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Task,
    Mesh,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStage::TessellationControl => {
                vk::ShaderStageFlags::TESSELLATION_CONTROL
            }
            ShaderStage::TessellationEvaluation => {
                vk::ShaderStageFlags::TESSELLATION_EVALUATION
            }
            ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
            ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
        }
    }
}

pub struct ShaderModule {
    parent: Arc<Device>,
    handle: vk::ShaderModule,
}

impl std::fmt::Debug for ShaderModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderModule")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl ShaderModule {
    /// Create a shader module from raw SPIR-V bytes.
    ///
    /// `spirv_bytes` must have a length that is a multiple of 4. If the
    /// bytes are not already aligned to `u32`, they are copied internally.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils` when
    /// the extension is available. Naming failures are logged as warnings and
    /// do not cause the call to fail.
    pub fn new(
        device: &Arc<Device>,
        spirv_bytes: &[u8],
        name: Option<&str>,
    ) -> Result<Self, CreateShaderModuleError> {
        if !spirv_bytes.len().is_multiple_of(4) {
            return Err(CreateShaderModuleError::InvalidLength(
                spirv_bytes.len(),
            ));
        }

        // Reinterpret bytes as u32 words. If the slice is already u32-aligned
        // we borrow it directly; otherwise we copy into a temporary Vec.
        //
        // SAFETY: u32 has no invalid bit patterns and we verified the length
        // is a multiple of 4, so the reinterpretation is sound.
        // SPIR-V is defined as little-endian, so for the copy path we use
        // from_le_bytes rather than from_ne_bytes to be correct on all
        // platforms. The direct borrow path via align_to is only reached on
        // little-endian targets where native and SPIR-V byte order match.
        let (prefix, aligned_words, _suffix) =
            unsafe { spirv_bytes.align_to::<u32>() };
        let owned;
        let code: &[u32] = if prefix.is_empty() {
            aligned_words
        } else {
            owned = spirv_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u32>>();
            &owned
        };

        let create_info = vk::ShaderModuleCreateInfo::default().code(code);

        // SAFETY: create_info contains valid SPIR-V code words.
        let handle = unsafe { device.create_raw_shader_module(&create_info) }
            .map_err(CreateShaderModuleError::Vulkan)?;

        // SAFETY: handle is a valid shader module created from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name shader module {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    /// Create an [`EntryPoint`] view into this module for the given entry
    /// point name and shader stage.
    ///
    /// Returns `Err` only if `name` contains an interior NUL byte.
    pub fn entry_point(
        &self,
        name: &str,
        stage: ShaderStage,
    ) -> Result<EntryPoint<'_>, std::ffi::NulError> {
        Ok(EntryPoint {
            module: self,
            name: CString::new(name)?,
            stage,
        })
    }

    pub fn raw_handle(&self) -> vk::ShaderModule {
        self.handle
    }

    pub fn get_parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        tracing::debug!("Dropping shader module {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed during
        // teardown. All pipeline objects derived from this module must be
        // destroyed before this ShaderModule is dropped.
        unsafe { self.parent.destroy_raw_shader_module(self.handle) };
    }
}

/// A borrow-view pairing a [`ShaderModule`] with a specific entry point name
/// and pipeline stage.
///
/// Created via [`ShaderModule::entry_point`]. The lifetime `'a` ties this view
/// to the module it was created from, ensuring the module stays alive for as
/// long as any pipeline stage create info derived from it is in use.
#[derive(Debug)]
pub struct EntryPoint<'a> {
    module: &'a ShaderModule,
    name: CString,
    stage: ShaderStage,
}

impl<'a> EntryPoint<'a> {
    /// Build a `VkPipelineShaderStageCreateInfo` referencing this entry point.
    ///
    /// The returned struct borrows from `self`, so it must not outlive this
    /// `EntryPoint`.
    pub fn as_pipeline_stage_create_info(
        &self,
    ) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage.into())
            .module(self.module.raw_handle())
            .name(&self.name)
    }

    pub fn stage(&self) -> ShaderStage {
        self.stage
    }
}
