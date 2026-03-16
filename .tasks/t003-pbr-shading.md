---
id: t003
title: "Implement PBR shading"
status: blocked
created: 2026-03-14
updated: 2026-03-16
parent: null
children: []
depends_on:
  - t001
  - t019
blocked_by:
  - t001
  - t019
area: phoenix
---

## Context

Phoenix currently renders geometry with a placeholder material
model. PBR (Physically Based Rendering) is the target shading
model per the GDD (`private/docs/PHOENIX_GDD.md`).

## Goal

Phoenix renders objects using a PBR material model (metallic-
roughness workflow) with at minimum direct lighting support.

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

Shader implementation targets PSIR (t001) — Slang is no longer
in use. The PBR BRDF will be written in PSIR once the IR and
its SPIR-V backend are functional.

PBR also depends on the resource state tracker (t019): correct
barrier and ownership-transfer management must be in place before
draw commands can be reliably recorded.

Dependency direction: the render graph executor depends on PBR,
not the other way around. PBR must not be listed in any render
graph task's `blocked_by`; render graph tasks that integrate PBR
should list t003 in their own `depends_on`.

## Outcome

(not yet filled — task is idea)
