---
id: t022
title: "Implement LearnOpenGL-style Blinn-Phong renderer"
status: planned
created: 2026-03-24
updated: 2026-03-24
parent: null
children:
      - t023
      - t024
      - t025
      - t026
      - t027
depends_on: []
blocked_by: []
area: phoenix
---

## Context

Near-term rendering priorities have changed: PBR is deferred. The
current milestone is to establish a robust baseline renderer using a
LearnOpenGL-inspired Blinn-Phong model. This gives a clear, debuggable
lighting pipeline and unblocks scene/material iteration while broader
renderer and pipeline tasks continue.

## Goal

Phoenix renders the main scene with a Blinn-Phong shading path
(ambient + diffuse + specular) using a simple material/light model
aligned with LearnOpenGL examples, with feature parity across the app
variants where applicable.

## Plan

- [ ] Define Blinn-Phong shader inputs and UBO layout
- [ ] Implement vertex + fragment shaders for ambient/diffuse/specular
- [ ] Add material controls (albedo/diffuse map, specular strength,
      shininess)
- [ ] Add directional light parameters and camera/view-position binding
- [ ] Integrate asset/material plumbing required by the new shader path
- [ ] Validate visuals against LearnOpenGL references and in-engine
      captures
- [ ] Mirror relevant renderer-side changes in sibling app where needed

## Thinking

This task intentionally optimizes for simplicity and observability over
physical accuracy. Blinn-Phong gives deterministic behavior that is easy
to inspect while we continue maturing render-graph and resource systems.

PBR remains a follow-up goal, but should build on this baseline instead
of blocking near-term renderer progress.

## Outcome

(not yet filled — task is planned)