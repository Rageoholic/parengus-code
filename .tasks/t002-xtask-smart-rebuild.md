---
id: t002
title: "xtask: smart rebuild-required check"
status: planned
created: 2026-03-14
updated: 2026-03-14
parent: null
children: []
depends_on: []
blocked_by: []
area: infra
---

## Context

The current `cargo xtask build` (see [`xtask/src/main.rs`](
../xtask/src/main.rs)) rebuilds unconditionally. For a full
workspace build (shaders + assets + Rust) this adds unnecessary
latency on incremental development cycles.

## Goal

xtask skips steps whose inputs have not changed since the last
successful build, matching the behaviour of a proper build system.

## Plan

- [ ] Audit current xtask build steps and their input/output sets
- [ ] Design a content-hash or mtime-based staleness check
- [ ] Implement per-step "is rebuild required?" predicate
- [ ] Store last-build state in a cache file (e.g.
      `out/.xtask-cache.json`)
- [ ] Skip up-to-date steps with a clear log message
- [ ] Ensure cache is invalidated on toolchain version change

## Thinking

Cargo already handles Rust staleness; the main wins are in the
shader-compile and asset-compile steps, which re-run every time
today. A simple mtime comparison against `out/` outputs is probably
sufficient to start — content hashing is more robust but adds
complexity.

## Outcome

(not yet filled — task is planned)
