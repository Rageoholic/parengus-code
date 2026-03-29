---
id: t024
title: "Material plumbing and SSBO infrastructure"
status: active
created: 2026-03-24
updated: 2026-03-29
parent: t003
children: []
depends_on: []
blocked_by: []
area: phoenix
---

## Context

PBR shading (t003) needs material parameters and texture references
available in the runtime data path. This task builds the CPU-side
material schema, GPU SSBO upload, and descriptor layout that the PBR
shader will consume.

## Goal

Material data supports PBR inputs (albedo, normal, ORM, emissive)
and is consumable by renderer bindings via descriptor indexing and
SSBO.

## Plan

- [x] Define material parameter schema for Blinn-Phong
      (MaterialGpu: albedo_idx, normal_idx, orm_idx, emissive_idx)
- [x] Add default textures at fixed slots 0–2 (magenta albedo,
      flat normal, neutral ORM)
- [x] Collect all texture roles (albedo, normal, ORM, emissive) from
      mesh.tex_refs and sub_mesh_albedos; deduplicate by TextureId
- [x] Upload material SSBO (HostVisibleBuffer, STORAGE_BUFFER usage)
      and write to binding 1 of the material descriptor set
- [x] Enable VK_EXT_descriptor_indexing + PARTIALLY_BOUND on the
      texture array binding (binding 0); expand to 256 slots
- [x] Add write_storage_buffer and binding_flags support to rgpu-vk
- [x] Update phoenix.slang: unbounded textures[], StructuredBuffer
      materials[], material_idx push constant
- [ ] Hook parameter upload path into renderer descriptors/uniforms
      (full Blinn-Phong constants — deferred to t023/t025)

## Thinking

Implemented the material infrastructure in a single PR:
- rgpu-vk gained descriptor_indexing in DeviceConfig, binding_flags
  on DescriptorBindingDesc, and write_storage_buffer on DescriptorSet.
- phoenix now loads all four texture roles per mesh (albedo per-submesh,
  normal/ORM/emissive global). Default textures occupy slots 0–2.
- MaterialGpu uses u32 indices (not u16) to avoid the SPIR-V Int16
  capability requirement. emissive_idx is i32 with -1 meaning absent.
- The shader uses unbounded Sampler2D textures[] and StructuredBuffer
  materials[]; the fixed count 256 lives only in the Rust constant and
  descriptor layout. Exceeding 256 mesh textures is a hard error —
  `resolve_tex` returns `eyre::bail!` and startup fails rather than
  silently corrupting the descriptor set.
- Fragment output is still albedo-only; full shading is t023's job.

## Outcome

(not yet filled — task is active)
