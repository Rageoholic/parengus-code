//! `rgpu` naming conventions:
//! - `raw_*` accessors return the Vulkan handle type from `ash::vk`.
//! - `ash_*` accessors return the corresponding `ash` wrapper object.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod buffer;
pub mod command;
pub mod device;
pub mod instance;
pub mod log;
pub mod pipeline;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;

pub use ash;
pub use raw_window_handle::HandleError as RWHHandleError;
