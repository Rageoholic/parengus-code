//! Phantom marker types for suppressing `Send`/`Sync` auto-impls.

/// Makes the containing type `!Sync` while leaving `Send` unaffected.
///
/// `Cell<()>` is `!Sync` (interior mutability without locking), so a
/// `PhantomData<Cell<()>>` field suppresses the `Sync` auto-impl. The
/// type remains `Send` as long as all other fields are `Send`.
///
/// Use this when the Vulkan spec requires external synchronization for
/// operations accessed via shared references (`&T`), making it unsound
/// to share the wrapper across threads without a lock.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PhantomUnsync {
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

/// Makes the containing type `!Send` while leaving `Sync` unaffected.
///
/// A type that is `Sync` but `!Send` can be shared across threads via
/// `&T` (e.g. inspected from multiple threads) but cannot be moved to
/// another thread. Use this when a Vulkan resource is safe to read
/// concurrently but must be destroyed on the thread that created it.
#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PhantomUnsend {
    _not_send: std::marker::PhantomData<*mut ()>,
}

// SAFETY: `PhantomUnsend` carries no data and imposes no `Sync`
// restriction — only `Send` is suppressed.
unsafe impl Sync for PhantomUnsend {}
