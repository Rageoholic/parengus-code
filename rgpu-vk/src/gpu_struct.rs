/// Marker trait for types that are valid SSBO array elements.
///
/// Implementors must be `bytemuck::Pod + bytemuck::Zeroable` and have
/// a size that is a multiple of 16 bytes (required by std430 array
/// stride rules).
///
/// # TODO (PSIR)
/// `gpu_ssbo!` currently emits a plain `#[repr(C)]` struct as a
/// stand-in. Eventually it should emit a layout descriptor consumed by
/// the PSIR layer, which will generate correct backend-specific
/// accessors and own all padding/alignment concerns.
pub trait GpuSsboType: bytemuck::Pod + bytemuck::Zeroable {}

/// Defines a GPU SSBO element type with compile-time alignment
/// checking.
///
/// Emits a `#[repr(C)]` struct that derives `Copy`, `Clone`,
/// `bytemuck::Pod`, and `bytemuck::Zeroable`, implements
/// [`GpuSsboType`], and asserts that its size is a multiple of 16
/// bytes at compile time.
///
/// # Usage
/// ```ignore
/// gpu_ssbo! {
///     pub MyLight {
///         color_range: [f32; 4],
///         _pad: [f32; 0],
///     }
/// }
/// // or without visibility:
/// gpu_ssbo! {
///     MyLight { color_range: [f32; 4] }
/// }
/// ```
#[macro_export]
macro_rules! gpu_ssbo_impl {
    ($vis:vis $name:ident { $($field:tt)* }) => {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        $vis struct $name { $($field)* }
        impl $crate::gpu_struct::GpuSsboType for $name {}
        const _: () = assert!(
            ::core::mem::size_of::<$name>() % 16 == 0,
            concat!(
                stringify!($name),
                " size must be a multiple of 16 bytes"
            )
        );
    };
}

#[macro_export]
macro_rules! gpu_ssbo {
    ($name:ident { $($field:tt)* }) => {
        $crate::gpu_ssbo_impl! { $name { $($field)* } }
    };
    ($vis:vis $name:ident { $($field:tt)* }) => {
        $crate::gpu_ssbo_impl! { $vis $name { $($field)* } }
    };
}
