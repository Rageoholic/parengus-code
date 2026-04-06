---
id: t038
title: "Separate samplers from textures"
status: active
created: 2026-03-30
updated: 2026-04-05
parent: null
children: []
depends_on: []
blocked_by: []
area: renderer
---

## Context

Currently samplers and textures are treated together in some parts
of the pipeline. This leads to duplicated sampler state, and makes
descriptor/table layout and runtime binding more complex.

## Goal

Separate samplers out from textures entirely and implement a
canonical sampler table. Samplers are grouped by three axes:

- Filter mode: `nearest`, `linear`, `anisotropic`
- Out-of-bounds behavior: `wrap`, `clamp_to_edge`, `mirror`,
  `transparent_border`, `black_border`, `white_border`
- Mip filtering: `nearest`, `trilinear`

The runtime should create a small fixed table of samplers (or a
lookup into device-created sampler objects) and reference them from
material descriptors by sampler index instead of embedding sampler
state per-texture.

## Plan

- [x] Update this file with a concrete implementation plan (today)
- [ ] Design descriptor batching API
      - Define `DescriptorUpdateBuilder` / `DescriptorWrite` types in
            `rgpu-vk::descriptor`.
      - Ensure the builder owns `Vec<vk::DescriptorImageInfo>` and
            `Vec<vk::DescriptorBufferInfo>` so slices passed to Vulkan are
            stable during `vkUpdateDescriptorSets`.
- [ ] Refactor `DescriptorSet` helpers
      - Make write helpers take `&mut self` and either:
            - return a `vk::WriteDescriptorSet` plus owned infos, or
            - accept `&mut DescriptorUpdateBuilder` and append writes directly.
      - Remove per-write calls to `device.update_raw_descriptor_sets`.
- [ ] Add `DescriptorUpdateBuilder` implementation
      - Provide `new()`, `push_*` helpers, and `apply(device)` which calls
            `device.update_raw_descriptor_sets` once with all accumulated
            writes.
      - Carefully document ownership and lifetimes of info vectors.
- [ ] Add separate-sampler support
      - Update descriptor set layout creation to expose separate sampler
            bindings where desired.
      - Decide on mapping between logical material sampler indices and
            descriptor bindings (e.g., a sampler array binding or a dedicated
            sampler set).
- [ ] Implement sampler table generator (placeholder)
      - Add `rgpu_vk::sampler::SamplerTable::new(device, options)` that
            creates the canonical set of samplers and returns indices/handles.
      - Start with a minimal set (filter × oob × mip) and make it
            configurable later.
- [ ] Update call sites
      - Modify `phoenix/src/main.rs`, `samp-app/src/main.rs`,
            `samp-app-noext/src/main.rs` to:
            - Make `DescriptorSet` instances `mut` where needed.
            - Use `DescriptorUpdateBuilder` to collect writes and call `apply`.
- [ ] Validation & testing
      - Build workspace and run sample apps; visually verify texture
            sampling behavior matches expectations.
      - Add small unit / smoke tests if practical (builder lifetime,
            sampler table creation).
- [ ] Docs & follow-ups
      - Update this task with results and follow-up tasks:
            - anisotropy support
            - asset pipeline authoring changes
            - shader interface changes / bindings

## Notes

Anisotropic sampling is handled by a separate optional feature task
(t040) — the renderer must be capable of using anisotropic samplers
only if the device advertises the feature.

## Thinking

Apparently black_border is rare and somewhat redundant with transparent_border.
We're cutting it to get us down to a svelte 15. We're also dropping mip
filtering. if you want a specific mip level, ask for it explicitly. The odds of
you being in a situation where you don't know which mip you need to sample *and*
seems so rare that I don't think it's worth worrying about.