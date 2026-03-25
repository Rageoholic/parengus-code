---
id: t027
title: "Import DamagedHelmet test asset"
status: active
created: 2026-03-24
updated: 2026-03-24
parent: t022
children: []
depends_on:
  - t023
  - t024
blocked_by: []
area: phoenix
---

## Context

We need a representative real-world mesh/material asset to evaluate
the Blinn-Phong baseline renderer. `assets/DamagedHelmet.gltf` is a
standard reference model and suitable for repeatable visual checks.

## Goal

DamagedHelmet is imported and renderable in Phoenix through the current
asset pipeline with a clear mapping from source material data to the
Blinn-Phong parameter set used by the renderer.

## Plan

- [ ] Verify mesh/texture paths and staging for DamagedHelmet assets
- [ ] Add import or conversion hooks required by the existing pipeline
- [ ] Ensure baker + runtime model format support multiple glTF nodes
  and multiple primitives per mesh/file
- [ ] Map source material channels into Blinn-Phong-compatible params
- [ ] Add a deterministic scene setup for renderer testing
- [ ] Confirm loading/rendering in both app variants where applicable

## Thinking

DamagedHelmet is authored for PBR, so this task includes documenting
and implementing the temporary conversion strategy for Blinn-Phong
inputs (especially specular and gloss response).

Implementation note: both the asset baker and the runtime model format
must correctly support multiple glTF nodes and multiple primitives per
mesh/file, rather than assuming a single flattened primitive.

## Outcome

(not yet filled — task is planned)
