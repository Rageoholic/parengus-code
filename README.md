# parengus-code

Rust workspace for Vulkan experiments.

- `rgpu`: Vulkan wrapper library (uses `ash`)
- `samp-app`: sample app using `winit`

## Prerequisites

- Rust toolchain (`cargo`)
- `slangc` on `PATH` (for shader compilation)
- Vulkan runtime / drivers

## Build and run (debug)

From repository root:

```powershell
cargo build -p samp-app
./samp-app/copy-samp-app-exe-debug.ps1
./samp-app/compile-samp-app-shaders-debug.ps1
```

Then run:

```powershell
./out/samp-app/debug/samp-app.exe
```

If you have local VS Code tasks configured, you can also run the equivalent
editor task flow. (`.vscode` is gitignored in this repo.)

## Shader outputs

`./samp-app/compile-samp-app-shaders-debug.ps1` compiles both shader variants:

- `shader.spv` (default)
- `shader.debug.spv` (compiled with debug info)

Outputs are written to:

- `out/samp-app/debug/shaders`

## RenderDoc / shader debug info

To load debug-info shaders at runtime, run:

```powershell
./out/samp-app/debug/samp-app.exe --shader-debug-info
```

If `shader.debug.spv` is missing, the app falls back to `shader.spv`.

## Useful checks

```powershell
cargo check -p rgpu
cargo check -p samp-app
cargo test --workspace
cargo clippy -p rgpu --all-targets -- -D warnings
cargo clippy -p samp-app --all-targets -- -D warnings
```

## AI attribution

Significant portions of this repository (including code, scripts,
documentation, and refactors) were produced with AI assistance.

Treat AI-authored changes like any external contribution:

- review carefully,
- run checks/tests locally,
- and validate behavior before release.
