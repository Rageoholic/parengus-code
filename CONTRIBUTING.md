# Contributing

> AI-assisted document — generated with Claude Code.

## Note on Private Submodule

The `private/` directory is a git submodule pointing to a private
repository. It is not available to external contributors and can be
safely ignored when working on the public codebase.

---

## Branch Names

Use `type/short-description` with kebab-case words.

| Type | Use for |
|------|---------|
| `feat/` | new feature or capability |
| `fix/` | bug fix |
| `docs/` | documentation only |
| `chore/` | maintenance, deps, tooling, CI |
| `refactor/` | restructuring without behavior change |
| `test/` | adding or updating tests |

**Examples**

```
feat/instanced-rendering
fix/swapchain-format-srgb
docs/contributing-guidelines
chore/update-ash-dep
refactor/device-selection
test/surface-lifetime
```

**Rules**
- 2–4 words after the slash; be specific enough to identify the work
- All lowercase, no underscores
- Branch off `main`; keep branches short-lived

---

## Filing Issues

Choose the right template when opening an issue
(`.github/ISSUE_TEMPLATE/`).

### Bug reports

Include:
1. **What happened** — a concise description of the unexpected behavior
2. **Steps to reproduce** — minimal code or commands to trigger it
3. **Expected behavior** — what should have happened
4. **Environment** — OS, GPU driver version, Vulkan SDK version, `rustc`
   version
5. **Relevant output** — validation layer messages, panic backtraces

### Feature requests

Include:
1. **Motivation** — the problem this solves or the use-case it enables
2. **Proposed API / behavior** — a sketch of what the change looks like
   from the caller's perspective
3. **Alternatives considered** — other approaches and why you prefer this
   one
4. **Scope** — does it affect `rgpu`, `samp-app`, or both?

### General rules
- Search for duplicates before opening a new issue
- One concern per issue; split unrelated topics
- If you used AI assistance to draft the issue body, note it at the
  bottom (see [AI Disclosure](CLAUDE.md#ai-disclosure))

---

## Vulkan Spec Links

Always link to the multi-page spec at `docs.vulkan.org`, not the
single-page version (`registry.khronos.org/…/vkspec.html`). The
single-page version is tens of MB of HTML and is not usable on
mobile or low-bandwidth connections.

Use the form:
```
https://docs.vulkan.org/spec/latest/chapters/<chapter>.html#<anchor>
```

---

## Extension Feature structs

Because we target Vulkan 1.0 + extensions, enabling optional device features
requires filling out a per-extension `VkPhysicalDevice*Features` struct. When a
feature is promoted to core, the extension struct becomes a type alias for the
core struct, so the same code works on both old and new drivers — using the
extension struct on a 1.3 device is valid and intended by the spec, not a
workaround. See the Vulkan spec: [Extending Vulkan §
Promotion](https://docs.vulkan.org/spec/latest/chapters/extensions.html#extendingvulkan-compatibility-promotion)

**Policy: query first, then pass the result to `DeviceCreateInfo`.**

Before creating the logical device, call `get_physical_device_features2` with
the feature struct(s) you care about chained into a `VkPhysicalDeviceFeatures2`.
The driver fills in which sub-features are actually supported. Pass those
same structs — unchanged — to `DeviceCreateInfo`. Never hard-code `VK_TRUE`;
enabling a feature the physical device does not report is invalid and will
trigger validation errors.

```rust
// Correct pattern
let mut my_features =
    vk::PhysicalDeviceMyFeatures::default(); // all zeros
let mut query = vk::PhysicalDeviceFeatures2::default()
    .push_next(&mut my_features);
// fills my_features with what the device actually supports
unsafe { instance.get_physical_device_features2(phys_dev, &mut query) };
// pass the filled struct to device creation — do NOT set fields to TRUE
device_create_info = device_create_info.push_next(&mut my_features);
```

**Checking feature support: exhaustive destructure.**

When writing a helper that validates whether all sub-features in a group are
supported, destructure the struct and explicitly bind every boolean field.
Use `_` only for `s_type`, `p_next`, and `_marker`. This gives compile-time
proof that no field was accidentally skipped:

```rust
fn my_feature_fully_supported(
    f: vk::PhysicalDeviceMyFeatures<'_>,
) -> bool {
    let vk::PhysicalDeviceMyFeatures {
        s_type: _,
        p_next: _,
        feature_a,
        feature_b,
        // ... every other boolean field named explicitly ...
        _marker: _,
    } = f;
    feature_a == vk::TRUE && feature_b == vk::TRUE // && ...
}
```

Do not use `..` to swallow unhandled fields — it defeats the point.
These structs are frozen by the Vulkan spec, so the exhaustive list is
a one-time cost that buys permanent safety.

The current design enables whatever the physical device reports and
requires all sub-features to be present (see the `*_fully_supported`
helpers in `device.rs`). Finer-grained per-sub-feature control is
tracked in t044.

---

## Pull Requests

### Title

Mirror the branch-name convention: `type: short description` in
sentence case.

```
feat: add instanced rendering support
fix: prefer B8G8R8A8_SRGB swapchain format
docs: add contributing guidelines
chore: update ash to 0.38
```

### Body

Use the PR template (`.github/pull_request_template.md`). Required
sections:

| Section | What to write |
|---------|---------------|
| **Summary** | Why this change exists; 1–3 bullets |
| **Changes** | What was modified at a high level |
| **Test plan** | How you verified correctness |
| **AI assistance** | Check the box if AI tools were used |

### Review checklist (author)
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] Lines ≤ 80 columns
- [ ] No new `unsafe` without a safety comment
- [ ] If the change touches `samp-app` or `samp-app-noext`, consider
      whether the sibling app needs the same fix (they share structure
      and bugs tend to appear in both)
- [ ] AI assistance disclosed if applicable
