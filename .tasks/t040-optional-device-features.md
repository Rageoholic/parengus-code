---
id: t040
title: "Optional device features and anisotropic sampling"
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

The renderer currently assumes certain device features are present
(e.g. anisotropic filtering). To run on a wider range of hardware we
should make some features optional and query for them at device
creation time.

## Goal

Enable anisotropic filtering as a feature the renderer can search
for and optionally enable. More generally, expose a mechanism for
optional device features and graceful fallbacks when unavailable.

## Plan

- [ ] Add device feature discovery and an optional-features table on
      `Device` creation
- [ ] Make anisotropic sampling a conditional capability: create
      anisotropic samplers only if the device supports it; otherwise
      fall back to linear filtering
- [ ] Update the asset pipeline / material compiler to mark assets
      that *prefer* anisotropy but can fall back
- [ ] Add runtime toggles / logs showing which optional features
      are enabled
- [ ] Add automated tests that simulate devices without anisotropy
      to verify correct fallback behavior
