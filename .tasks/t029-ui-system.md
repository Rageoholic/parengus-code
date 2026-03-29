---
id: t029
title: "UI system with text rendering and buttons"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: null
children:
  - t030
  - t031
  - t033
depends_on: []
blocked_by: []
area: phoenix
---

## Context

Phoenix has three distinct renderer contexts:
- **Mission renderer** — main in-game 3D scene
- **Garage renderer** — real in-game scene (e.g. VTOL mech bay) for
  mech customisation and loadout
- **Loading screen renderer** — simple 2D, no 3D scene

The UI system sits on top of all three. It must be
renderer-agnostic: self-contained with its own texture registry,
shaping state, and GPU resources, not coupled to any one
renderer's internals.

Rendering mode varies by context:
- **Mission + garage:** UI renders to a separate offscreen RGBA
  buffer; a compositing pass alpha-blends it over the scene image
  before present.
- **Loading screen:** no scene image; UI renders directly to the
  swapchain image (no compositing pass needed).

The design is driven by three sub-tasks: font pipeline (t030),
text rendering (t031), and the widget system (t033).

## Goal

A renderer-agnostic UI system that renders text and interactive
widgets, compositing onto the scene in mission/garage contexts and
rendering directly to the swapchain in the loading screen context.

## Plan

- [ ] Font pipeline: TTF as runtime asset + dynamic SDF atlas (t030)
- [ ] Text rendering system: shaped quads via SSBO (t031)
- [ ] Widget system: composable retained-mode widgets (t033)
- [ ] Offscreen UI render target + compositing pass (mission/garage)
- [ ] Direct-to-swapchain path (loading screen)

## Thinking

The separate offscreen buffer is justified by the multi-renderer
architecture: the UI system cannot draw into the mission renderer's
attachments directly without coupling to that renderer's render
pass setup. The offscreen buffer is the clean abstraction boundary.

The loading screen is simple enough that the offscreen buffer adds
no value there — the UI is the entire frame content. The UI renderer
should support both modes and the caller selects which at
initialization or per-frame.

The UI's own texture registry (separate from mesh texture registries)
is justified by the same argument: widgets are created and destroyed
independently of scene assets, and the UI must work across all three
renderer contexts without sharing lifetime assumptions.

The compositing pass is a single-triangle fullscreen shader sampling
the UI buffer and alpha-blending over the scene image. Minimal
complexity; no `vkCmdBlitImage` (which does not handle alpha).

RTL, bidirectional text, and complex-script shaping are all in
scope via rustybuzz (details in t031). Widget composability uses
a primitive/composite model (details in t033).

## Outcome

(not yet filled — task is planned)
