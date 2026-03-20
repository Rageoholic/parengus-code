---
id: t020
title: "Generate Slang compiler metadata and import"
status: planned
created: 2026-03-20
updated: 2026-03-20
parent: null
children: []
depends_on: []
blocked_by: []
area: pipeline
---

## Context

The project historically used Slang for shader authoring. A Slang
compiler run can emit metadata describing resource bindings,
entry-points, and specialization information. Importing this
metadata into the engine enables compatibility for existing assets
while PSIR is not yet available.

## Goal

Provide a tooling step that runs the Slang compiler to produce a
machine-readable metadata file and import that metadata into the
engine's shader pipeline so existing shaders can be used without
manual binding annotations.

## Plan

- [ ] Define metadata schema (binding layout, resource names, types)
- [ ] Add a Slang->metadata emitter (xtask or small CLI wrapper)
- [ ] Add engine import path to consume metadata at asset compile or
      pipeline creation time
- [ ] Add tests/fixtures using the Duck asset and a small example

## Outcome

When complete, the engine can use Slang-produced metadata for shader
resource layout so PBR and other features can proceed without PSIR.
