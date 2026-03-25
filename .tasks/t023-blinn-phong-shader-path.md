---
id: t023
title: "Blinn-Phong shader path"
status: planned
created: 2026-03-24
updated: 2026-03-24
parent: t022
children: []
depends_on: []
blocked_by: []
area: phoenix
---

## Context

The first executable milestone for the Blinn-Phong renderer is a
working shader path with stable interfaces and predictable outputs.

## Goal

Implement and wire the vertex/fragment shader pair for ambient,
diffuse, and specular (Blinn half-vector) lighting in Phoenix.

## Plan

- [ ] Define shader interface structs and binding layout
- [ ] Implement vertex transform and normal handling path
- [ ] Implement fragment Blinn-Phong terms (ambient/diffuse/specular)
- [ ] Compile and bind shaders through the current pipeline path
- [ ] Validate with a simple directional-light test scene

## Thinking

Keep this task narrowly focused on shader correctness and pipeline
binding. Material authoring, tuning controls, and validation capture
work are split into sibling child tasks.

## Outcome

(not yet filled — task is planned)
