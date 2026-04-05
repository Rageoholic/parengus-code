LLM summary — short lines only:
- Purpose: compile source assets into runtime formats.
- Policies: keep output deterministic; lines ≤ 80 cols.
- Verify: `cargo clippy -p asset-compiler --all-targets -- -D warnings`
- Quick checks: `cargo test -p asset-compiler`
- Details: see crate README for asset formats and tooling.
