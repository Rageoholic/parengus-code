---
id: t019
title: "Resource state tracker"
status: planned
created: 2026-03-16
updated: 2026-03-16
parent: t005
children: []
depends_on: []
blocked_by: []
area: renderer
---

## Context

The renderer needs to track the Vulkan state of every image and
buffer — owning queue family and image layout — to insert correct
barriers and queue-family ownership transfers at submit boundaries.

PBR shading (t003) depends on this tracker: a correct shading
pipeline requires reliable resource state management before
recording draw commands.

## Goal

A resource state machine that tracks per-resource Vulkan state
(queue family + image layout), creates semaphores on demand for
cross-queue synchronisation, and records the necessary barriers in
command-buffer pre- and post-ambles.

## Plan

- [ ] Define resource state type (queue family, image layout)
- [ ] Implement state tracker struct with per-resource state map
- [ ] Implement semaphore-on-demand creation for queue-family
      ownership transfers
- [ ] Integrate into renderer: caller declares resources before
      recording; tracker emits barriers in pre/post ambles
- [ ] Validate with first-frame transfer→graphics handoff:
      transfer queue releases resources and signals semaphore;
      graphics queue acquires and waits on that semaphore

## Thinking

Initial validation scenario: the first frame should produce a
submit on the transfer queue that releases vertex/texture resources
to the graphics queue family and signals a semaphore. The graphics
queue submit waits on that semaphore and performs the acquire
(queue-family ownership transfer acquire).

The renderer declares which resources it reads before recording;
the state machine produces barriers for the pre- and post-amble
of the command buffer. This avoids ad-hoc per-site barrier
insertion scattered throughout the app.

Dependency direction: the render graph executor (see sibling tasks
t006–t009) depends on PBR (t003) being available, not the other
way around. PBR must not depend on any render graph task; any
render graph task that needs PBR should list t003 in its
`depends_on`, not vice-versa.

Eventually this component will migrate into the render graph
executor: passes will declare resource usage and the executor will
drive the state tracker automatically. For now it lives as a
standalone helper consumed directly by the app.

## Outcome

(not yet filled — task is planned)
