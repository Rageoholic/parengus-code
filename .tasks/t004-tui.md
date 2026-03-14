---
id: t004
title: "Implement TUI"
status: planned
created: 2026-03-14
updated: 2026-03-14
parent: null
children: []
depends_on: []
blocked_by: []
area: phoenix
---

## Context

The TUI is a **development-only simulation harness interface** — a
debugging and testing tool for the simulation layer while the
graphical renderer is under development. It is not the final player
UI.

Architecture: the TUI acts as a client of the simulation core,
interacting only through the simulation API. The underlying storage
model (ECS, etc.) is not exposed.

```
Simulation Core
       ↑
       │ Simulation API
       │
TUI Harness
Renderer (future)
```

Instrumentation hooks allow the TUI to observe simulation events
without the simulation maintaining a permanent event log:

```
Simulation Core
       ↓ emits
Simulation Events
       ↓ observed by
TUI Harness (instrumentation consumer)
```

## Goal

A working terminal UI that lets developers step/run/pause the
simulation, inspect domain entities (mechs, pilots, factions,
contracts, conflicts, locations), view the event log, and
manipulate world state for testing — all without touching
implementation internals.

## Plan

- [ ] Choose TUI library (ratatui is the leading candidate)
- [ ] Scaffold TUI crate/module in the workspace
- [ ] Implement command parser (`verb object [arguments]`)
- [ ] Simulation control commands:
      `step [n]`, `run`, `pause`, `reset`, `seed <n>`
- [ ] World overview: `world` (turn, factions, active conflicts)
- [ ] Entity inspection: `inspect <type> <id>`
      Types: mech, pilot, faction, contract, conflict, location
- [ ] Entity listing: `list <type>`
- [ ] Event log: `events`, `events last <n>`
- [ ] Debug/manipulation commands:
      `spawn`, `force contract`, `trigger event`, `damage`
- [ ] Diagnostic commands: `watch <entity> <field>`,
      `trace <entity>` (causal chain display)
- [ ] Output layout: World Status / Event Log / Command sections
- [ ] Wire instrumentation hooks into simulation core

## Thinking

**Command model:** `verb object [arguments]`, e.g.:
```
step 10
inspect mech 12
list factions
watch faction union influence
trace conflict desert_corridor
```

**Non-goals:** no final game UI, no combat visualization, no
graphical elements, no player usability constraints. Sole purpose
is simulation debugging and testing.

**Instrumentation design:** the simulation emits domain-level
events describing meaningful state transitions. The TUI consumes
these via hooks without requiring the sim to maintain long-term
event history. This enables event inspection, causal tracing, and
replay without runtime overhead in the shipped game.

**Output layout (reference):**
```
----------------------------------
World Status
----------------------------------
Turn: 143
Global Conditions: Nominal
Active Conflicts: 2

----------------------------------
Event Log
----------------------------------
[recent events]

----------------------------------
Command
----------------------------------
>
```
Exact layout is not critical as long as information is readable.

## Outcome

(not yet filled — task is planned)
