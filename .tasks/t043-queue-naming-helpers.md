---
id: t043
title: "Re-enable queue naming and add Phoenix debug-label helpers"
status: active
created: 2026-03-30
updated: 2026-03-30
parent: null
children: []
depends_on: []
blocked_by: []
area: renderer
---

## Goal

Queue objects were previously given debug names but that logic
appears to have been commented out. Additionally, Phoenix can benefit
from small helper functions that call into `rgpu-vk`'s debug-labeling
API to annotate command-buffer regions and optional submit labels.

## Goal
Restore stable, descriptive names for queue objects (e.g.
`gfx-main-queue`, `transfer-0`, `async-compute-0`) and add a small set
of thin helpers implemented in `rgpu-vk` so Phoenix can annotate
command-buffer regions and optionally submissions by calling into
`rgpu-vk`.

Implementation location: the helpers should live in `rgpu-vk` — put
naming and queue-level helpers in `rgpu-vk/src/device.rs` and the
command-buffer region helpers (begin/end labels) in
`rgpu-vk/src/command.rs`. Phoenix will call these wrapper helpers; do
not add Phoenix-specific logic inside `rgpu-vk`.
Restore stable, descriptive names for queue objects (e.g.
`gfx-main-queue`, `transfer-0`, `async-compute-0`) and add a small
set of Phoenix helper stubs that invoke `rgpu-vk`'s labeling helpers
or extension details.

                  - In `device.rs`: re-enable queue naming and provide stable
                        queue-name helper(s)
                  - In `command.rs`: add `label_region_begin`, `label_region_end`,
                        and an optional `label_submit_opt` that Phoenix can call
                        during recording/submission


- [ ] Re-enable queue object naming code paths in `rgpu-vk` (if
      they exist commented out) or reintroduce a thin naming call at
      device/queue creation
## Notes

This task only documents the work; actual changes will be implemented
in `rgpu-vk` and should be landed as small, reviewable commits. The
helpers must be thin wrappers that delegate to the existing debug
utils handling in `rgpu-vk`.
- [ ] Add documentation showing recommended label strings and an
      example of annotating G-buffer, lighting, and upload regions
- [ ] Add a small validation capture to ensure labels appear as
      expected in RenderDoc

## Notes

This is a task description only; no code will be modified as part of
this task entry. Actual implementation should be landed in small
commits guarded by code review.
