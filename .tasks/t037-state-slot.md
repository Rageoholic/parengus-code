---
id: t037
title: "Harden state machine transitions with replace_with"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: null
children: []
depends_on: []
blocked_by: []
area: infra
---

## Context

`AppRunner(Option<App>)` and the planned `SceneState` wrapper both use
the take-then-set pattern for state machine transitions. The `None`
window between `take` and `set` is unobservable in practice (winit is
single-threaded), but there is no compile-time or runtime guarantee
that `set` is always called. Forgetting it leaves the state machine in
a permanently broken `None` state that only surfaces at the next
callback's `assert!(self.0.is_some())`.

The `replace_with` crate solves this with a closure-based API:
`replace_with_or_abort(slot, |old| new)` — the closure must return a
new value; if it panics, the process aborts rather than leaving the
slot undefined. No custom `StateSlot` type needed.

## Goal

All `AppRunner` and `SceneState` transitions use `replace_with_or_abort`
instead of manual `take_*/set_*` helpers, making it structurally
impossible to forget the `set` step.

## Plan

- [ ] Add `replace_with` to `phoenix/Cargo.toml`
- [ ] Replace `AppRunner`'s manual `take_*/set_*` helpers with
      `replace_with_or_abort` at each transition site
- [ ] Apply to `SceneState` wrapper when that is introduced

## Thinking

`replace_with_or_abort` is the right variant: on closure panic it
aborts the process, which is safer than leaving the `Option` in an
unspecified state. The closure form is also more ergonomic than a
token — the new state is computed and returned in one expression.

## Outcome

(not yet filled — task is planned)
