---
id: t025
title: "Blinn-Phong lighting uniforms"
status: dropped
created: 2026-03-24
updated: 2026-03-29
parent: t022
children: []
depends_on:
  - t023
blocked_by: []
area: phoenix
---

## Context

Lighting and view-dependent specular require frame-level uniforms for
light direction/color and camera position.

## Goal

Directional light and camera/view-position parameters are provided to
the Blinn-Phong shader path each frame with stable GPU layout.

## Plan

- [ ] Define light and camera uniform block layout
- [ ] Populate per-frame uniform values from app state
- [ ] Bind and verify layout compatibility in pipeline setup
- [ ] Add a basic runtime tuning path for light intensity/specular

## Thinking

Keep this limited to one directional light for the baseline. Additional
light types remain follow-up work.

## Outcome

Dropped 2026-03-29. Parent task t022 (Blinn-Phong renderer) was
dropped; the project pivots directly to PBR (t003).
