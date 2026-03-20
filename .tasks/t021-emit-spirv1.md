---
id: t021
title: "Emit Vulkan 1.0-compatible SPIR-V from PSIR emitter"
status: planned
created: 2026-03-20
updated: 2026-03-20
parent: null
children: []
depends_on: []
blocked_by: []
area: psir
---

## Context

The PSIR project currently targets SPIR-V via `psir-spirv`. Some
platforms and validation paths require generated modules that are
compatible with the Vulkan 1.0 environment (a conservative feature
set). Providing a Vulkan 1.0-compatible emission mode eases
integration with older drivers and tooling.

## Goal

Add an emitter mode or compatibility path that produces SPIR-V
modules compatible with Vulkan 1.0 (i.e., restrict ops/capabilities
to what Vulkan 1.0 environments expect) from PSIR binary IR.

## Plan

- [ ] Audit current SPIR-V emission for opcodes and capabilities that
      require newer SPIR-V/Vulkan versions
- [ ] Add a compatibility mode that restricts emitted ops/features to
      those supported by Vulkan 1.0 and emits the appropriate
      capability/imports for that environment
- [ ] Add tests to validate generated SPIR-V with `spirv-val` using
      a Vulkan 1.0 target environment

## Outcome

PSIR can emit Vulkan 1.0-compatible SPIR-V for compatibility builds
and CI validation.
