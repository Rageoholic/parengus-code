#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(
    clippy::redundant_closure,
    reason = "This error sucks and I hate it cause it just ends up making \
              refactoring annoying"
)]

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
