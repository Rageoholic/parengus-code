#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

mod phoenix;

fn main() -> eyre::Result<()> {
    phoenix::main()
}
