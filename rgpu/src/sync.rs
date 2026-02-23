use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::device::{Device, NameObjectError};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CreateFenceError {
    #[error("Vulkan error creating fence: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum WaitFenceError {
    #[error("Fence wait timed out")]
    Timeout,
    #[error("Vulkan error waiting for fence: {0}")]
    Vulkan(vk::Result),
}

#[derive(Debug, Error)]
pub enum CreateSemaphoreError {
    #[error("Vulkan error creating semaphore: {0}")]
    Vulkan(vk::Result),
}

// ---------------------------------------------------------------------------
// Fence
// ---------------------------------------------------------------------------

/// An owned binary fence used for CPU–GPU synchronisation.
///
/// Use [`wait`](Self::wait) to block the CPU until the GPU signals the fence,
/// then [`reset`](Self::reset) to return it to the unsignaled state before
/// the next submission.
pub struct Fence {
    parent: Arc<Device>,
    handle: vk::Fence,
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fence")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Fence {
    /// Create a fence.
    ///
    /// `signaled` controls the initial state. Pass `true` so the first
    /// `wait` + `reset` cycle in a render loop returns immediately.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils` when
    /// the extension is available. Naming failures are logged as warnings and
    /// do not cause the call to fail.
    pub fn new(
        device: &Arc<Device>,
        signaled: bool,
        name: Option<&str>,
    ) -> Result<Self, CreateFenceError> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let create_info = vk::FenceCreateInfo::default().flags(flags);

        // SAFETY: create_info is fully initialised with no borrowed pointers.
        let handle = unsafe { device.create_raw_fence(&create_info) }
            .map_err(CreateFenceError::Vulkan)?;

        // SAFETY: handle is a valid fence created from device.
        match unsafe { device.set_object_name_str(handle, name) } {
            Ok(()) | Err(NameObjectError::DebugUtilsNotEnabled) => {}
            Err(e) => tracing::warn!("Failed to name fence {:?}: {e}", handle),
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    /// Block until the fence is signaled or `timeout_ns` nanoseconds elapse.
    ///
    /// Pass `u64::MAX` to wait indefinitely.
    pub fn wait(&self, timeout_ns: u64) -> Result<(), WaitFenceError> {
        // SAFETY: handle is a valid fence created from parent.
        unsafe { self.parent.wait_for_raw_fences(&[self.handle], true, timeout_ns) }.map_err(
            |e| {
                if e == vk::Result::TIMEOUT {
                    WaitFenceError::Timeout
                } else {
                    WaitFenceError::Vulkan(e)
                }
            },
        )
    }

    /// Reset the fence to the unsignaled state.
    ///
    /// # Safety
    /// The fence must not be currently pending on any queue submission
    /// (i.e. the GPU must have already signaled it, or it was never submitted).
    pub unsafe fn reset(&mut self) -> Result<(), vk::Result> {
        // SAFETY: Caller guarantees the fence is not pending.
        unsafe { self.parent.reset_raw_fences(&[self.handle]) }
    }

    /// Wait for the fence to be signaled and then immediately reset it.
    ///
    /// This is the canonical render-loop operation: block until the GPU
    /// finishes the previous frame, then return the fence to the unsignaled
    /// state so it can be used for the next submission.
    ///
    /// # Safety
    /// No other thread may re-submit this fence's raw handle between the wait
    /// returning and the reset completing. The `&mut` receiver prevents
    /// same-thread re-submission via `raw_handle`, but cross-thread raw handle
    /// usage is still the caller's responsibility.
    pub unsafe fn wait_and_reset(&mut self, timeout_ns: u64) -> Result<(), WaitFenceError> {
        self.wait(timeout_ns)?;
        // SAFETY: wait() succeeded so the fence is signaled and not pending.
        // &mut self prevents any same-thread re-submission of raw_handle()
        // between the wait and reset.
        unsafe { self.reset() }.map_err(WaitFenceError::Vulkan)
    }

    /// Returns `true` if the fence is currently in the signaled state.
    pub fn is_signaled(&self) -> Result<bool, vk::Result> {
        // SAFETY: handle is a valid fence created from parent.
        unsafe { self.parent.get_raw_fence_status(self.handle) }
    }

    pub fn raw_handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn get_parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        tracing::debug!("Dropping fence {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed during
        // teardown. No GPU work may reference this fence.
        unsafe { self.parent.destroy_raw_fence(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Semaphore
// ---------------------------------------------------------------------------

/// An owned binary semaphore used for GPU–GPU synchronisation.
///
/// Semaphores are signaled by one queue operation and waited on by another.
/// The CPU cannot directly query or reset a binary semaphore; they are driven
/// entirely through queue submissions.
pub struct Semaphore {
    parent: Arc<Device>,
    handle: vk::Semaphore,
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Semaphore")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Semaphore {
    /// Create a semaphore.
    ///
    /// `name` is an optional debug label applied via `VK_EXT_debug_utils` when
    /// the extension is available. Naming failures are logged as warnings and
    /// do not cause the call to fail.
    pub fn new(device: &Arc<Device>, name: Option<&str>) -> Result<Self, CreateSemaphoreError> {
        let create_info = vk::SemaphoreCreateInfo::default();

        // SAFETY: create_info is fully initialised with no borrowed pointers.
        let handle = unsafe { device.create_raw_semaphore(&create_info) }
            .map_err(CreateSemaphoreError::Vulkan)?;

        // SAFETY: handle is a valid semaphore created from device.
        match unsafe { device.set_object_name_str(handle, name) } {
            Ok(()) | Err(NameObjectError::DebugUtilsNotEnabled) => {}
            Err(e) => tracing::warn!("Failed to name semaphore {:?}: {e}", handle),
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_handle(&self) -> vk::Semaphore {
        self.handle
    }

    pub fn get_parent(&self) -> &Arc<Device> {
        &self.parent
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        tracing::debug!("Dropping semaphore {:?}", self.handle);
        // SAFETY: handle was created from parent and is being destroyed during
        // teardown. No GPU work may be waiting on or about to signal it.
        unsafe { self.parent.destroy_raw_semaphore(self.handle) };
    }
}
