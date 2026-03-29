---
id: t036
title: "Custom Vulkan memory allocator"
status: idea
created: 2026-03-29
updated: 2026-03-29
parent: null
children: []
depends_on: []
blocked_by: []
area: rgpu-vk
---

## Context

`rgpu-vk` currently uses `gpu-allocator` for all Vulkan memory
management. It works, but it lacks features the engine will eventually
want — notably `VK_EXT_memory_budget` integration for proactive cache
eviction in the garage renderer. Building a custom allocator is
explicitly deferred until the engine's allocation patterns are well
understood; by then the requirements will be concrete rather than
speculative.

This task is captured now to record intent and known requirements,
not because it is imminent.

## Goal

`rgpu-vk` uses a custom-built Vulkan memory allocator that replaces
`gpu-allocator`, with native `VK_EXT_memory_budget` support and
allocator strategies matched to the engine's actual usage patterns.

## Plan

- [ ] Identify allocation patterns from profiling: slab for textures,
      linear/ring for per-frame staging, etc.
- [ ] Implement slab allocator for device-local image memory
- [ ] Implement linear allocator for transient staging buffers
- [ ] Integrate `VK_EXT_memory_budget`: expose per-heap budget/usage
      and use it to drive garage cache eviction
- [ ] Replace `gpu-allocator` in rgpu-vk's `Buffer` and `Image`
      wrappers
- [ ] Validate with existing tests and visual output

## Thinking

By the time this is implemented the engine will have shipped enough
features that real allocation behaviour is observable under a profiler.
Building the allocator first would mean guessing at slab sizes,
alignment requirements, and lifetime patterns. Defer until that data
exists.

Known requirements so far:
- `VK_EXT_memory_budget` for proactive VRAM pressure handling
- Supports separate strategies per usage pattern (textures vs.
  staging vs. geometry buffers)
- `memory_budget()` query exposed on `Device` for general budget
  awareness; callers decide what to do with the information

## Outcome

(not yet filled — task is an idea, explicitly deferred)