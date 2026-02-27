//! Thin RAII wrappers around Vulkan objects, built on [`ash`].
//!
//! > **Personal project.** This crate is not intended for general use
//! > and makes no API stability guarantees.
//!
//! # Object hierarchy
//!
//! ```text
//! Instance
//! ├── Surface<T>
//! │   └── Swapchain<T>
//! └── Device
//!     ├── HostVisibleBuffer / DeviceLocalBuffer
//!     ├── DescriptorSetLayout → DescriptorPool → DescriptorSet
//!     ├── PipelineLayout (with DescriptorSetLayout refs)
//!     ├── ShaderModule → EntryPoint → DynamicPipeline
//!     ├── ResettableCommandPool → ResettableCommandBuffer
//!     └── Fence / Semaphore
//! ```
//!
//! Each wrapper holds its parent via `Arc` so parents cannot be
//! destroyed while children are alive.
//!
//! # Naming conventions
//!
//! | prefix  | meaning                                   |
//! |---------|-------------------------------------------|
//! | `raw_*` | accepts or returns a raw `ash::vk` handle |
//! | `ash_*` | returns the `ash` wrapper object          |

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod buffer;
pub mod command;
pub mod descriptor;
pub mod device;
pub mod instance;
pub mod pipeline;
pub mod shader;
pub mod surface;
pub mod swapchain;
pub mod sync;

pub use ash;
pub use raw_window_handle::HandleError as RwhHandleError;
