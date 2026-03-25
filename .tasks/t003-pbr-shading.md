---
id: t003
title: "Implement PBR shading"
status: planned
created: 2026-03-14
updated: 2026-03-24
parent: null
children: []
depends_on:
  - t022
blocked_by: []
area: phoenix
---

## Context

Phoenix currently renders geometry with a placeholder material
model. PBR (Physically Based Rendering) is the target shading
model per the GDD (`private/docs/PHOENIX_GDD.md`).

As of 2026-03-24, this work is explicitly deferred while a
LearnOpenGL-style Blinn-Phong baseline is implemented first (t022).

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
what the Khronos Duck asset already exercises. IBL can be deferred
— direct lighting with a correct BRDF is a meaningful milestone
on its own.

Priority update: Blinn-Phong baseline task t022 now comes first for
near-term engine progress; this task stays planned for later.

Shader implementation targets PSIR (t001) — Slang is no longer
in use. NOTE: PSIR work has been deferred for now; PBR will be
implemented using the existing shader pipeline or a temporary
authoring path until PSIR (t001) is completed.

PBR also depends on the resource state tracker (t019): correct
barrier and ownership-transfer management must be in place before
draw commands can be reliably recorded.

## Outcome

(not yet filled — task is planned)
