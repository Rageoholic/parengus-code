---
id: t024
title: "Blinn-Phong material plumbing"
status: planned
created: 2026-03-24
updated: 2026-03-24
parent: t022
children: []
depends_on:
  - t023
blocked_by: []
area: phoenix
---

## Context

Blinn-Phong shading needs material-side parameters to be available in
the runtime data path and asset interfaces.

## Goal

Material data supports Blinn-Phong inputs (diffuse/albedo,
specular-strength, shininess) and is consumable by renderer bindings.

## Plan

- [ ] Define material parameter schema for Blinn-Phong
- [ ] Add runtime material structs and serialization/plumbing updates
- [ ] Hook parameter upload path into renderer descriptors/uniforms
- [ ] Add sensible defaults/fallbacks for missing parameters

## Thinking

This task should not alter long-term material architecture beyond what
is needed for the Blinn-Phong baseline.

## Outcome

(not yet filled — task is planned)
