---
id: t044
title: "Design granular subfeature exposure in DeviceConfig"
status: future
created: 2026-04-02
updated: 2026-04-02
parent: null
children: []
depends_on: []
blocked_by: []
area: rgpu-vk
---

## Context

`Device::create_compatible` enables top-level feature groups (e.g.
`descriptor_indexing: true`) by querying and passing through the entire
`VkPhysicalDevice*Features` struct. The problem is that callers and the
runtime then use those features with no record of which specific
sub-features were actually present on the selected device. Code that
relies on, say, `descriptor_binding_partially_bound` is implicitly
assuming that sub-feature was reported — but nothing enforces or even
checks that assumption. If the device happens not to support a
sub-feature, the code silently misbehaves.

## Goal

Expose the result of the physical device feature query so that runtime
code can check whether specific sub-features are available before using
them, and so that `DeviceConfig` can express required sub-features as a
hard filter during device selection.

## Plan

- [ ] Decide where the queried feature state lives after device creation
      (field on `Device`, a separate `DeviceCapabilities` struct, etc.)
- [ ] Audit which sub-features phoenix and the sample apps actually depend
      on; mark each as required (filter devices that lack it) vs. optional
      (gate the code path at runtime)
- [ ] Extend `DeviceConfig` to let callers declare required sub-features,
      and fail device selection if they are absent
- [ ] Implement runtime capability accessors so code can query "do I have
      X?" rather than assuming
- [ ] Update callers to use the accessors instead of assuming
- [ ] Update CONTRIBUTING.md with the new pattern

## Thinking

The current fix queries and passes through whatever the device reports,
which is correct at the Vulkan API level. The remaining gap is that
nothing surfaces those results to the rest of the codebase — there is no
way to ask "was `descriptor_binding_partially_bound` actually present on
the device we selected?" after the fact. Callers implicitly assume yes
because they requested the feature group. That assumption needs to become
explicit.

This task may be deferred until device selection is exposed more cleanly
to callers. In the meantime the workaround is to treat the absence of
any sub-feature within a requested group as a hard device filter — i.e.
require all reported flags to be `TRUE` before accepting the device. That
is hacky but keeps the implicit assumption safe in practice.
