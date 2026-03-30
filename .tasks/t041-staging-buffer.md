---
id: t041
title: "Single staging buffer with linear allocator"
status: active
created: 2026-03-30
updated: 2026-03-30
parent: null
children: []
depends_on: []
blocked_by: []
area: phoenix
---

## Context

Multiple transient uploads currently allocate many small staging
buffers and pools via `rgpu-vk` or the renderer. This causes
fragmentation and extra synchronization complexity for uploads.

## Goal

Implement a single large staging buffer in `phoenix` with a simple
linear allocator for transient uploads. The staging buffer is owned
and managed by Phoenix (not `rgpu-vk`) and is reused across
uploads until full.

Behaviour when buffer is full:

- If an allocation request cannot fit in the remaining space, submit
  the current command buffer batch that uses the staging buffer,
  wait for the associated fence, reset the fence, reset the linear
  allocator (effectively reclaiming the staging region), and then
  continue allocating.

Alignment:

- The allocator must respect device `minUniformBufferOffsetAlignment`,
  `minStorageBufferOffsetAlignment`, and any `transfer` alignment
  constraints required by the API. Allocate with the correct
  alignment per-copy so the eventual transfer (copy to GPU) is
  correctly aligned.

## Plan

- [ ] Design the staging buffer size heuristics and allocation API
- [ ] Implement a linear allocator that returns (offset, size) with
      alignment, and tracks used range
- [ ] Integrate with Phoenix upload paths: texture and buffer
      uploads should allocate from the staging buffer and record
      copies using that offset
- [ ] Implement submit/wait/reset logic in Phoenix: when staging is
      exhausted, submit recorded commands, wait on fence, reset
      allocator and fence, then continue
- [ ] Add diagnostic counters (peak usage, submits caused by
      staging exhaustion) and tests for alignment correctness

## Notes

This is intentionally implemented in Phoenix (renderer-level) so
that `rgpu-vk` remains a lightweight Vulkan wrapper without upload
policy decisions. Phoenix should orchestrate submission and the
allocator to match its render/transfer batching strategy.
