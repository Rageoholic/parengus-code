LLM summary — short lines only:
- Purpose: tracing utilities used across the workspace.
- Policies: prefer small, well-tested helpers; lines ≤ 80 cols.
- Verify: `cargo clippy -p parengus-tracing --all-targets -- -D warnings`
- Quick checks: `cargo test -p parengus-tracing`
- Details: see crate README for integration notes.
