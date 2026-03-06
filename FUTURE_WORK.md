# Future Work

> Generated with Claude's assistance.

## Multisampling

The current `RenderingInfo` uses a single-sample colour attachment.
Adding MSAA requires:

- Selecting a supported sample count (`VkSampleCountFlagBits`) during
  device/swapchain setup.
- Allocating a multisampled colour attachment image and view.
- Setting `resolve_image_view` / `resolve_image_layout` on the
  `RenderingAttachmentInfo` so the MSAA image resolves into the
  swapchain image each frame.
- Wiring the sample count through `DynamicPipelineDesc` so the
  rasterisation state matches.

## Consistency of Type Layouts Between Rust and Slang

Shared types such as `Ubo` and `PushConstants` are defined independently
in Rust ([samp-app/src/main.rs](samp-app/src/main.rs)) and in Slang
([samp-app/shaders/shader.slang](samp-app/shaders/shader.slang)).
The `column_major` qualifier and `#[repr(C)]` + `bytemuck` derive macros
are currently kept in sync by hand, which is fragile. Options to
investigate:

- A code-generation step that emits Rust structs from Slang reflection
  data (via `slang-sys` or the Slang reflection API).
- A shared schema (e.g. a TOML or JSON description) that drives both
  the Slang typedef and the Rust struct.
- Compile-time assertions (`assert_eq!(size_of::<Ubo>(), EXPECTED)`)
  as a lightweight safety net until a fuller solution is in place.

## Asset Pipeline: Texture Color Space Metadata

The asset pipeline currently passes texture files through without tracking
or validating their color space. This led to an sRGB correctness bug where
the Vulkan format (`R8G8B8A8_SRGB` vs `R8G8B8A8_UNORM`) had to be chosen
by convention rather than by inspecting the source asset.

Improvements to make:

- During asset processing, inspect PNG `sRGB`, `gAMA`, and `iCCP` chunks
  (and JPEG APP2/ICC markers) to determine the source color space.
- Emit a warning at asset-build time when no color space metadata is
  found, prompting the author to either embed it or annotate the asset
  manifest entry explicitly.
- Propagate the detected color space into the asset map (e.g. as a
  `color_space = "srgb" | "linear"` field) so the loader can select the
  correct `VkFormat` without relying on convention.

## Decide If Shaders Are Assets or Code

Slang source files currently live under `samp-app/shaders/` and are
compiled at build time, but the project also has an asset pipeline
(`asset_pipeline` crate). The distinction matters for workflow and
tooling:

- **Shaders as code** — compiled by `build.rs`, errors surface as build
  failures, hot-reload requires a rebuild, SPIR-V blobs checked in or
  generated into `OUT_DIR`.
- **Shaders as assets** — processed by the asset pipeline, can be
  reloaded at runtime without relinking, fit naturally alongside
  textures and meshes, but need a runtime shader cache and error
  reporting path.

Pick one model and align the directory structure, build scripts, and
asset map accordingly.
