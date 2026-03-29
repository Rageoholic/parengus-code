---
id: t032
title: "Observable/event system"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: null
children: []
depends_on: []
blocked_by: []
area: phoenix
---

## Context

Both the UI system (t031) and future systems (achievements, etc.)
need a way to react to state changes without polling every frame.
The UI specifically wants retained-mode widgets that reshaping and
re-layout only when their data changes. Rather than building ad-hoc
notification mechanisms per system, a shared observable primitive
should underpin all of them.

## Goal

A general-purpose observable/event mechanism that lets systems
register callbacks (or invalidation flags) against named values or
event channels, and receive notifications when those values change.
Widgets and other consumers can also choose to poll instead of
subscribing, so the system must support both patterns.

## Plan

- [ ] Design the core primitive: typed observable value vs. untyped
      event channel vs. both; decide if callbacks are synchronous
      or deferred to a flush point each frame
- [ ] Implement the primitive in a shared crate (likely a new
      `parengus-events` or inside an existing shared crate)
- [ ] Define widget integration: a widget holds a subscription
      handle; when notified, it sets a dirty flag that triggers
      reshaping/re-layout next gather pass
- [ ] Define polling integration: a widget can also skip
      subscription and check a value directly each frame — useful
      for values that change every frame anyway (timers, counters)
- [ ] Wire UI system (t031) against the event system
- [ ] Validate with an achievement-style use case as a second
      consumer (even if the full achievement system is future work)

## Thinking

### Callback vs. polling

Subscriptions are more efficient at steady state but add lifecycle
complexity (when does a widget unsubscribe? what if the widget
is destroyed while a notification is in flight?). Polling is
trivial but wastes CPU if the value is rarely changing.

A reasonable design: observables track a **generation counter**.
A subscriber stores the last-seen generation; checking for a
change is a single integer compare. Callbacks are optional and
run at a well-defined flush point (e.g. start of the gather
pass), not immediately on write. This avoids re-entrancy problems
and keeps notification order deterministic.

### Synchronous vs. deferred

Prefer deferred (flush-point) notifications. Immediate callbacks
on write create reentrancy risks and make write order matter in
non-obvious ways. A fixed flush at the top of the frame update
is easier to reason about.

### Shared primitive

The same mechanism should serve the UI, achievements, and any
other reactive system. Do not build a UI-specific event system;
extract the primitive first and let UI sit on top.

## Outcome

(not yet filled — task is planned)
