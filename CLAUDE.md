# Project Instructions

## Repo Initialization

After cloning, activate the project's git hooks:

```
git config core.hooksPath .githooks
```

## Structure
- Cargo workspace with `resolver = "3"`, `edition = "2024"`
- Members: `rgpu` (Vulkan wrapper lib using ash),
  `samp-app` (sample app using winit)
- CI: GitHub Actions at `.github/workflows/ci.yml` —
  per-package clippy + workspace build + tests

## Coding Conventions
- `#![deny(unsafe_op_in_unsafe_fn)]` is set — all unsafe operations
  inside `unsafe fn` must be wrapped in an explicit `unsafe {}` block.
- Unsafe methods on wrapper types are prefixed with `raw`
  (e.g. `create_raw_surface`). Prefer `unsafe fn` wrappers over
  exposing raw handles directly.

## Architecture
- `Instance` wraps the ash Vulkan instance.
- `Surface<T>` holds `Arc<Instance>` and `Arc<T>` for lifetime safety.
- Device selection uses a priority-based fold over physical devices.

## Feature Unification Gotcha
Workspace feature unification can hide missing features. Always verify
individual crates with `cargo check -p <crate>` rather than relying on
a workspace-level check. rust-analyzer checks the whole workspace by
default and won't catch per-crate feature gaps.

## Line Length

Keep all lines ≤ 80 columns. `rustfmt.toml` enforces this for code via
`max_width = 80`.

For things rustfmt cannot wrap (comments, string literals, `#[derive(...)]`):
- **Comments:** Wrap manually at a word boundary before column 80.
- **String literals:** Use the escaped-newline trick (`\` at end of line
  strips the newline and leading whitespace on the next line).
- **Long `#[derive(...)]`:** Stable `rustfmt` does not wrap derive
  item lists and merges split `#[derive]` attributes back into one.
  No workaround exists on stable; slightly-over lines are accepted.

Only exceed 80 columns when there is no syntactically valid way to break
the line (e.g., a single token or URL that is itself longer than 80 chars).

## AI Disclosure

For externally visible project artifacts, explicitly disclose AI assistance.

- Keep a clear AI attribution note in `README.md`.
- When creating or updating standalone publishable text (issue bodies,
  release notes, long design docs), include an explicit AI-assistance note.
