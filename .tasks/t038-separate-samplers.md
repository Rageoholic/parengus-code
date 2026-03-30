---
id: t038
title: "Separate samplers from textures"
status: active
created: 2026-03-30
updated: 2026-03-30
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

- [ ] Design the sampler table layout and canonical enum values
- [ ] Update asset pipeline / material metadata so textures and
      samplers are authored separately (or map existing texture
      sampler metadata into the table)
- [ ] Implement sampler table creation on device init / renderer
      startup and expose sampler indices in descriptor sets
- [ ] Update shaders / binding model to accept sampler index
      alongside texture index (or use combined image + sampler
      descriptors where required by API but keep logical separation)
- [ ] Add tests and visual validation to ensure expected sampler
      behaviour (wrap/mirror/clamp, trilinear vs nearest, anisotropy)

## Notes

Anisotropic sampling is handled by a separate optional feature task
(t040) — the renderer must be capable of using anisotropic samplers
only if the device advertises the feature.
