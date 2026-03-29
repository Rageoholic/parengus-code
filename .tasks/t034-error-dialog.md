---
id: t034
title: "Error dialog / fatal message display"
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

When Phoenix encounters a fatal error (missing asset, Vulkan init
failure, etc.) the process currently just panics or exits silently.
On desktop the user sees nothing useful unless they launched from a
terminal. A minimal native dialog should surface the error text
before the process exits on all desktop platforms without requiring
any platform-specific code in the engine.

## Goal

Fatal errors produce a native OS dialog showing the error message
before the process exits, regardless of whether a terminal is
attached. A single cross-platform crate handles Windows, macOS, and
Linux — no per-platform cfg blocks needed.

## Plan

- [ ] Pick a cross-platform all-in-one crate: primary candidate is
      `rfd` with the `xdg-portal` feature (no GTK compile dep,
      works in Flatpak sandboxes); `native-dialog` as fallback if
      rfd is too heavy
- [ ] Integrate with the panic handler and/or the top-level
      `eyre` report path so all fatal errors go through the dialog
- [ ] Verify behaviour when a terminal is present (dialog + stderr
      both show the message)

## Thinking

The requirement is one crate, all desktop platforms covered, no
`#[cfg(target_os = ...)]` in the engine. `rfd` satisfies this:
`MessageDialog::new().set_description(msg).show()` works on
Windows (MessageBox), macOS (NSAlert), and Linux. Use the blocking
`show()` (not `show_async()`) — correct for a panic handler
context where the event loop is either not yet running or about to
be abandoned.

**winit interaction:** for a fatal error the winit event loop is
irrelevant — init errors fire before the loop starts, and runtime
panics are unrecoverable so blocking the loop is fine. Both crates
dispatch to the main thread internally, which matches winit's
macOS requirement. On Linux, prefer `rfd` with the `xdg-portal`
feature (D-Bus, no GTK build dependency) — winit does not use GTK
so there is no conflict either way, but the portal backend is
lighter and works in Flatpak sandboxes.

**Linux portal availability:** if `xdg-desktop-portal` is not
running (headless, minimal distro), `rfd` returns `None` and no
dialog appears — the process still exits with the error on stderr.
This is acceptable: a game without a desktop environment is not a
supported target. During implementation, check whether enabling
both `xdg-portal` and `gtk3` features on `rfd` gives an automatic
fallback, or whether portal-absent means silent failure regardless.

`native-dialog` is a lighter alternative with the same API shape
but no portal support, making it potentially awkward on Linux.

## Outcome

(not yet filled — task is planned)
