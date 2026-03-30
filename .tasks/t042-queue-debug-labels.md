---
id: t042
title: "Use Debug Utils labels for queues and submissions"
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

The renderer already applies `VK_EXT_debug_utils` labels to queue
objects. However command-buffer regions and submit-level labels are
not consistently annotated, which leaves captures with implicit
regions and makes debugging/profiling less clear.

## Goal

Ensure command-buffer regions are annotated with debug utils labels
(`vkCmdBeginDebugUtilsLabelEXT` / `vkCmdEndDebugUtilsLabelEXT`) and
optionally apply submit-level labels where useful. Queue labeling is
already in place and does not need changes.

Labeling is already supported transparently by `rgpu-vk`; Phoenix
should call the wrapper's labeling helpers rather than directly
querying the extension.

## Plan

- [ ] Add high-level labeling calls in Phoenix that invoke the
      `rgpu-vk` wrapper's debug-label helpers around command buffer
      recording regions. Do not reimplement extension probing in
      Phoenix — delegate to `rgpu-vk`.
- [ ] Annotate command-buffer regions by semantic purpose (e.g.
      `upload`, `g-buffer`, `lighting-compose`, `post-process`) so
      captures show meaningful regions
- [ ] Optionally add submit-level labels where it improves capture
      clarity; keep this opt-in
- [ ] Add an opt-in toggle and logs indicating when debug utils are
      enabled/disabled at the renderer level (this may be a thin
      passthrough to `rgpu-vk` diagnostics).
- [ ] Update capture/preflight docs and add a small visual test to
      validate labels appear in a RenderDoc/NSight capture

## Notes

This is a renderer-level concern (Phoenix) since it controls the
semantics of submissions. However, `rgpu-vk` already handles the
extension transparently; Phoenix should use `rgpu-vk`'s helpers to
emit labels for command-buffer regions and optional submit-level
annotations.
