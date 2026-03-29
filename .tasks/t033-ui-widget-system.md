---
id: t033
title: "UI widget system"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: t029
children: []
depends_on:
  - t031
  - t032
blocked_by:
  - t031
  - t032
area: phoenix
---

## Context

With text rendering in place (t031) and an observable/event system
(t032), the engine can support a proper retained-mode widget layer:
buttons, panels, labels, bars, and other interactive UI elements
backed by persistent state and event-driven invalidation.

The widget system is part of the renderer-agnostic UI system (t029)
and maintains its own texture registry for widget-owned textures,
separate from the mesh renderer's texture registries. This is
necessary because widgets have independent lifetimes and the UI
must work across all three renderer contexts (mission, garage,
loading screen).

## Goal

Phoenix has a widget system where logical widgets (labels, buttons,
etc.) are persistent objects that reshape and re-layout only when
dirtied by the event system, and can also opt into per-frame polling
for values that change every frame.

## Plan

- [ ] Define the widget trait and lifecycle (create, dirty, gather,
      destroy)
- [ ] Implement label widget: holds a string + font + position;
      optional background rect (colour + optional border radius);
      reshapes via t031 on dirty
- [ ] Implement bar widget: filled rect with a normalised fill
      value (0.0–1.0); colour/tint configurable; no text involved,
      just a coloured quad emitted as a `UiInstance`
- [ ] Implement button widget: bounding-box hit test on CPU, visual
      state (idle/hover/pressed) via tint; fires event on click
- [ ] Wire widget dirty flags to t032 subscriptions; support polling
      opt-in for high-frequency widgets
- [ ] Implement widget tree / flat list: gather pass walks all
      widgets, skips clean ones, collects `UiInstance` records
- [ ] Integrate with the phoenix render loop

## Thinking

### RM design

Each widget stores its last-shaped `Vec<UiInstance>` output. The
gather pass re-emits cached instances for clean widgets at zero
cost. A dirty widget reshapes and updates its cache, then emits.

Widgets subscribe to observables from t032. When a subscribed value
changes, the widget is marked dirty. Polling widgets skip
subscription and mark themselves dirty unconditionally at the top
of each gather pass.

### Buttons and interaction

Input events (cursor position, click) are handled on the CPU before
the gather pass. A button checks whether the cursor is within its
bounding rect and updates its visual state accordingly, which may
mark it dirty if the state changed.

## Outcome

(not yet filled — task is planned)
