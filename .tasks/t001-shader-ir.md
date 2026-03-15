---
id: t001
title: "Parengus Shader IR (PSIR)"
status: planned
created: 2026-03-14
updated: 2026-03-14
parent: null
children: []
depends_on: []
blocked_by: []
area: psir
---

## Context

Full IR format spec: [`docs/SHADER_IR.md`](../docs/SHADER_IR.md).
Covers register model, type system, control flow, instruction set,
module structure, resource layout, SPIR-V emission, and executor.

Planned crate structure: `psir`, `psir-spirv`, `psir-compiler`,
`psir-engine`, `psir-executor`.

## Goal

A working Parengus Shader Intermediate Representation: a typed
register-based IR with a SPIR-V lowering backend, sufficient to
compile simple shaders used by Phoenix.

## Plan

- [ ] Create `psir` crate skeleton in workspace
- [ ] Define `PsirType` enum (scalars, vectors, matrices, resources)
- [ ] Define instruction set and register model
- [ ] Implement control-flow graph representation
- [ ] Create `psir-spirv` crate; implement SPIR-V emission
- [ ] Add debug info via `NonSemantic.Shader.DebugInfo.100`
- [ ] Create `psir-compiler` (text IR → binary IR front-end)
- [ ] Wire into Phoenix shader pipeline

## Thinking

Key gotcha: vec3 std430 alignment is 16 bytes, not 12 — the type
layout helper must encode this explicitly.

### SPIR-V Target Environment (2026-03-14)

The emitter receives a `SpirvTargetEnv` (capability set) derived
from the logical device at pipeline compile time. The IR is
version-agnostic; the emitter picks ops from the capability set
(e.g. `discard` → `OpTerminateInvocation` on SPIR-V 1.6,
`OpKill` otherwise).

SPIR-V version is determined solely by `VkPhysicalDeviceProperties
::apiVersion` — not by which Vulkan extensions are enabled. None
of the current `DeviceConfig` flags (`dynamic_rendering`,
`synchronization2`, etc.) imply a SPIR-V version bump.

`psir-spirv` carries no Vulkan knowledge. `psir-engine` (which
knows both `rgpu-vk` and `psir-spirv`) does the mapping from
device capabilities to `SpirvTargetEnv`.

### `SpirvVersionRequest` on `DeviceConfig` (2026-03-14)

`DeviceConfig` will gain:

```rust
pub spirv_version: SpirvVersionRequest,  // default: None
```

```rust
pub enum SpirvVersionRequest {
    /// No preference — use whatever the selected device's
    /// Vulkan version provides by default. Default value.
    None,
    /// Prefer devices with the highest SPIR-V version during
    /// selection; enable SPIR-V-bump extensions (e.g.
    /// VK_KHR_spirv_1_4) when available.
    Highest,
    /// Require at least this SPIR-V version; reject devices
    /// below it. Also enables any extensions needed to reach
    /// the requested version.
    AtLeast(SpirvVersion),
}
```

`Device::spirv_version() -> SpirvVersion` returns the concrete
resolved version from the selected physical device's API version.

| Variant | Device selection | Extension enabling |
|---|---|---|
| `None` | No version preference | None |
| `Highest` | Prefer higher SPIR-V | Enables bump exts if avail |
| `AtLeast(V)` | Reject devices below V | Enables exts to reach V |

`samp-app-noext` uses `None` (strict VK 1.0, no extras).

## Outcome

(not yet filled — task is planned)
