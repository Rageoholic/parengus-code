---
id: t039
title: "Per-frame descriptor & command pools"
status: active
created: 2026-03-30
updated: 2026-03-30
parent: null
children: []
depends_on: []
blocked_by: []
area: renderer
---

## Context

Some parts of the renderer reuse global descriptor pools and command
pools which complicates multiframe-in-flight resource management and
makes per-frame allocator lifetimes harder to reason about.

## Goal

Switch to using per-frame-in-flight descriptor pools and command
pools where appropriate. Each frame-in-flight gets its own pools so
that allocations can be reset/trimmed when the frame completes,
reducing fragmentation and simplifying lifetime management.

## Plan

- [ ] Add `TransientCommandPool` type (created with `TRANSIENT_BIT`)
      that allocates `TransientCommandBuffer` handles — intended for
      short-lived, one-shot work (uploads, transient passes); pool is
      reset as a unit, not individual buffers
- [ ] Define a `CommandRecorder` trait that both
      `ResettableCommandBuffer` and `TransientCommandBuffer` implement,
      covering at minimum:
      - `get_raw_command_buffer(&self) -> RawCommandBufferHandle`
      - `get_recording_state(&self) -> RecordingState`
      - `set_recording_state(&mut self, state: RecordingState)`
- [ ] Create per-frame-in-flight descriptor pools (one pool per frame
      slot); allocate all per-frame descriptors from that pool at
      setup and cache them — no transient descriptor pools, no
      per-frame re-allocation
- [ ] Create per-frame-in-flight command pools (one
      `ResettableCommandPool` per frame slot) and ensure command
      buffer recording uses the correct pool for the frame being
      recorded
- [ ] Move upload/staging command buffer recording to use
      `TransientCommandPool` instead of the resettable pool
- [ ] Update phoenix (and sibling apps if `rgpu-vk` API changes
      require it) to route allocations to the per-frame pools
- [ ] Verify resets happen only after GPU idle for that frame slot
