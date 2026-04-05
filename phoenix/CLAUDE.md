LLM summary — short lines only:
- Purpose: Renderer/sample app.
- Policies: `#![deny(unsafe_op_in_unsafe_fn)]`; lines ≤ 80 cols.
- Verify: `cargo clippy -p phoenix --all-targets -- -D warnings`
- Quick checks: `cargo check -p phoenix`
- Details: see crate README for developer notes.
