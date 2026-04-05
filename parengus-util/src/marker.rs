//! Phantom marker types for suppressing `Send`/`Sync` auto-impls.

use std::marker::PhantomData;

/// Makes the containing type `!Sync` while leaving `Send` unaffected.
///
/// `Cell<()>` is `!Sync` (interior mutability without locking), so a
/// `PhantomData<Cell<()>>` field suppresses the `Sync` auto-impl. The
/// type remains `Send` as long as all other fields are `Send`.
///
/// Use this when a resource requires external synchronization for
/// operations accessed via shared references (`&T`), making it unsound
/// to share the wrapper across threads without a lock.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhantomUnsync {
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

/// Makes the containing type `!Send` while leaving `Sync` unaffected.
///
/// A type that is `Sync` but `!Send` can be shared across threads via
/// `&T` (e.g. inspected from multiple threads) but cannot be moved to
/// another thread. Use this when a resource is safe to read
/// concurrently but must be destroyed on the thread that created it.
#[derive(Debug, Default, Clone, Copy)]
pub struct PhantomUnsend {
    _not_send: PhantomData<*mut ()>,
}

// SAFETY: `PhantomUnsend` carries no data and imposes no `Sync`
// restriction — only `Send` is suppressed.
unsafe impl Sync for PhantomUnsend {}
