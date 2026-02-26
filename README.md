# parengus-code

Rust workspace for Vulkan experiments.

- `rgpu-vk`: Vulkan wrapper library (uses `ash`)
- `samp-app`: sample app using `winit`

## rgpu-vk overview

`rgpu-vk` is a set of thin RAII wrappers around Vulkan objects. Each
wrapper holds its parent via `Arc` so parents cannot be destroyed
while children are alive.

Object hierarchy:

```
Instance
├── Surface<T>
│   └── Swapchain<T>
└── Device
    ├── HostVisibleBuffer / DeviceLocalBuffer
    ├── ShaderModule → EntryPoint → DynamicPipeline
    ├── ResettableCommandPool → ResettableCommandBuffer
    └── Fence / Semaphore
```

**Naming conventions**

| prefix | meaning |
|--------|---------|
| `raw_*` | accepts or returns a raw `ash::vk` handle |
| `ash_*` | returns the `ash` wrapper object |

All unsafe operations inside `unsafe fn` bodies are wrapped in
explicit `unsafe {}` blocks, satisfying
`#![deny(unsafe_op_in_unsafe_fn)]`.

## Prerequisites

- Rust toolchain (`cargo`)
- `slangc` on `PATH` (for shader compilation)
- Vulkan runtime / drivers

## Build and run (debug)

From repository root:

```sh
cargo xtask build
```

This builds the sample app, compiles shaders, and copies the
executable and debug info to the `out/` directory.

Then run the binary from `out/samp-app/debug/`.

If you have local VS Code tasks configured, you can also run the
equivalent editor task flow. (`.vscode` is gitignored in this repo.)

## Shader outputs

`cargo xtask compile-shaders` compiles both shader variants:

- `shader.spv` (default)
- `shader.debug.spv` (compiled with debug info)

Outputs are written to `out/samp-app/debug/shaders`.

## RenderDoc / shader debug info

To load debug-info shaders at runtime, pass `--shader-debug-info`:

```sh
./out/samp-app/debug/samp-app --shader-debug-info
```

If `shader.debug.spv` is missing, the app falls back to `shader.spv`.

## Useful checks

```sh
cargo check -p rgpu-vk
cargo check -p samp-app
cargo test --workspace
cargo clippy -p rgpu-vk --all-targets -- -D warnings
cargo clippy -p samp-app --all-targets -- -D warnings
```

## AI attribution

Significant portions of this repository (including code, scripts,
documentation, and refactors) were produced with AI assistance.

Treat AI-authored changes like any external contribution:

- review carefully,
- run checks/tests locally,
- and validate behavior before release.
