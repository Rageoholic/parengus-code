//! GPU synchronisation primitives: [`Fence`] and [`Semaphore`].
//!
//! [`Fence`] is a CPU–GPU synchronisation object. The canonical
//! render-loop pattern is to create fences in the signaled state and
//! call [`wait_and_reset`](Fence::wait_and_reset) at the start of each
//! frame to block until the previous frame's GPU work completes.
//!
//! [`Semaphore`] is a GPU–GPU synchronisation object used to order
//! operations across queue submissions, typically to sequence swapchain
//! image acquisition and presentation with rendering work.

use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::device::Device;

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
    #[error("Asked to wait for fence but fence was never marked as submitted")]
    NotSubmitted,
}

#[derive(Debug, Error)]
pub enum MarkSubmittedError {
    #[error(
        "This fence is already marked as submitted but was marked \
         submitted again"
    )]
    AlreadySubmitted,
}

#[derive(Debug, Error)]
pub enum CreateSemaphoreError {
    #[error("Vulkan error creating semaphore: {0}")]
    Vulkan(vk::Result),
}

// ---------------------------------------------------------------------------
// Fence
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum FenceStatus {
    Submitted,
    Ready,
}

/// An owned binary fence used for CPU–GPU synchronisation.
///
/// Use [`wait`](Self::wait) to block the CPU until the GPU signals the fence,
/// then [`reset`](Self::reset) to return it to the unsignaled state before
/// the next submission.
pub struct Fence {
    parent: Arc<Device>,
    handle: vk::Fence,
    status: FenceStatus,
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fence")
            .field("handle", &self.handle)
            .field("status", &self.status)
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
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name fence {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
            status: if signaled {
                FenceStatus::Submitted
            } else {
                FenceStatus::Ready
            },
        })
    }

    pub fn wait_nonblocking(&self) -> Result<bool, WaitFenceError> {
        match self.wait(0) {
            Ok(_) => Ok(true),
            Err(WaitFenceError::Timeout) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn wait_and_reset_nonblocking(
        &mut self,
    ) -> Result<bool, WaitFenceError> {
        let signaled = self.wait_nonblocking()?;
        if signaled {
            // SAFETY: wait_nonblocking returned true, so the fence is
            // signaled and no longer pending on the GPU. &mut self
            // prevents same-thread re-submission via raw_fence() between
            // the wait and the reset.
            unsafe { self.reset() }.map_err(WaitFenceError::Vulkan)?;
        }
        Ok(signaled)
    }

    /// Block until the fence is signaled or `timeout_ns` nanoseconds elapse.
    ///
    /// Pass `u64::MAX` to wait indefinitely.
    pub fn wait(&self, timeout_ns: u64) -> Result<(), WaitFenceError> {
        if self.status == FenceStatus::Submitted {
            // SAFETY: handle is a valid fence created from parent.
            unsafe {
                self.parent.wait_for_raw_fences(
                    &[self.handle],
                    true,
                    timeout_ns,
                )
            }
            .map_err(|e| {
                if e == vk::Result::TIMEOUT {
                    WaitFenceError::Timeout
                } else {
                    WaitFenceError::Vulkan(e)
                }
            })
        } else {
            Err(WaitFenceError::NotSubmitted)
        }
    }

    /// Reset the fence to the unsignaled state.
    ///
    /// # Safety
    /// The fence must not be currently pending on any queue submission
    /// (i.e. the GPU must have already signaled it, or it was never submitted).
    pub unsafe fn reset(&mut self) -> Result<(), vk::Result> {
        debug_assert!(self.status == FenceStatus::Submitted);
        // SAFETY: Caller guarantees the fence is not pending.
        unsafe { self.parent.reset_raw_fences(&[self.handle]) }?;
        self.status = FenceStatus::Ready;
        Ok(())
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
    /// same-thread re-submission via `raw_fence`, but cross-thread raw-handle
    /// usage is still the caller's responsibility.
    pub unsafe fn wait_and_reset(
        &mut self,
        timeout_ns: u64,
    ) -> Result<(), WaitFenceError> {
        self.wait(timeout_ns)?;
        // SAFETY: wait() succeeded so the fence is signaled and not pending.
        // &mut self prevents any same-thread re-submission of raw_fence()
        // between the wait and reset.
        unsafe { self.reset() }.map_err(WaitFenceError::Vulkan)
    }

    /// This marks the fence as submitted, so that it can properly be waited.
    ///
    /// # Safety
    /// The fence must actually be submitted to some operation that will signal
    /// it when the operation is completed, such as vkQueueSubmit. It is
    /// undefined behavior if this operation is called while the underlying
    /// VkFence is not submitted
    pub unsafe fn mark_submitted(&mut self) -> Result<(), MarkSubmittedError> {
        if self.status == FenceStatus::Ready {
            self.status = FenceStatus::Submitted;
            Ok(())
        } else {
            Err(MarkSubmittedError::AlreadySubmitted)
        }
    }

    pub fn raw_fence(&self) -> vk::Fence {
        self.handle
    }

    pub fn parent(&self) -> &Arc<Device> {
        &self.parent
    }

    /// Is the fence in an unsignaled state where we can submit it to something
    /// like vkQueueSubmit
    pub fn is_ready(&self) -> bool {
        self.status == FenceStatus::Ready
    }

    /// Is the fence in a submitted state where we can wait on it and reset it
    pub fn is_submitted(&self) -> bool {
        self.status == FenceStatus::Submitted
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
    pub fn new(
        device: &Arc<Device>,
        name: Option<&str>,
    ) -> Result<Self, CreateSemaphoreError> {
        let create_info = vk::SemaphoreCreateInfo::default();

        // SAFETY: create_info is fully initialised with no borrowed pointers.
        let handle = unsafe { device.create_raw_semaphore(&create_info) }
            .map_err(CreateSemaphoreError::Vulkan)?;

        // SAFETY: handle is a valid semaphore created from device.
        let name_result = unsafe { device.set_object_name_str(handle, name) };
        if let Err(e) = name_result {
            tracing::warn!("Failed to name semaphore {:?}: {e}", handle);
        }

        Ok(Self {
            parent: Arc::clone(device),
            handle,
        })
    }

    pub fn raw_semaphore(&self) -> vk::Semaphore {
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
