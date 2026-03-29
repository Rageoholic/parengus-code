---
id: t003
title: "Implement PBR shading"
status: active
created: 2026-03-14
updated: 2026-03-29
parent: null
children:
  - t024
      - t035
depends_on: []
blocked_by: []
area: phoenix
---

## Context

Phoenix currently renders geometry with a placeholder material
model. PBR (Physically Based Rendering) is the target shading
model per the GDD (`private/docs/PHOENIX_GDD.md`).

As of 2026-03-29, the Blinn-Phong intermediate step (t022) was
dropped. The project now targets PBR directly. Material/SSBO
infrastructure built in t024 (albedo, normal, ORM, emissive slots;
descriptor indexing; MaterialGpu SSBO) carries forward unchanged
and is re-parented here.

## Goal

Phoenix renders objects using a PBR material model (metallic-
roughness workflow) with at minimum direct lighting support.

This remains a follow-up milestone after the Blinn-Phong renderer
baseline is complete.

## Plan

- [ ] Define PBR material parameters (albedo, metallic, roughness,
      normal, AO) and their GPU layout
- [ ] Write PBR BRDF shader (Cook-Torrance: GGX NDF, Smith G,
      Fresnel-Schlick)
- [ ] Integrate material parameters into the asset pipeline as
      texture slots
- [ ] Add a directional light uniform and basic light loop
- [ ] Verify against reference renders (e.g. Khronos glTF sample
      models)
- [ ] Add IBL (image-based lighting) as a follow-up

## Thinking

The metallic-roughness model is the glTF standard and aligns with
what the Khronos Duck and DamagedHelmet assets already exercise.
IBL can be deferred — direct lighting with a correct BRDF is a
meaningful milestone on its own.

The material/SSBO infrastructure (t024) is already in place:
four texture roles per mesh (albedo per-submesh, normal/ORM/emissive
global), default textures at slots 0–2, MaterialGpu with u32
indices, and unbounded descriptor indexing in the shader. The next
step is the PBR BRDF shader itself.

Shader implementation uses the existing Slang pipeline. PSIR (t001)
remains deferred and is not a prerequisite.

PBR also depends on the resource state tracker (t019): correct
barrier and ownership-transfer management must be in place before
draw commands can be reliably recorded.

## Outcome

(not yet filled — task is planned)
