---
id: t026
title: "Blinn-Phong visual validation"
status: planned
created: 2026-03-24
updated: 2026-03-24
parent: t022
children: []
depends_on:
  - t023
  - t024
  - t025
  - t027
blocked_by: []
area: phoenix
---

## Context

The renderer baseline needs repeatable checks to ensure visual output
matches expected Blinn-Phong behavior and remains stable over changes.

## Goal

Validate Blinn-Phong output against LearnOpenGL-style references and
record in-engine captures/checks for regressions.

## Plan

- [ ] Define comparison checklist (ambient/diffuse/specular behavior)
- [ ] Capture baseline scenes and expected camera/light settings
- [ ] Verify sibling app parity where renderer path is shared
- [ ] Document known deltas and acceptable error bounds

## Thinking

Validation should prefer deterministic scenes/settings so future
rendering changes can be compared consistently.

## Outcome

(not yet filled — task is planned)
