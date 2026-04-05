LLM summary — short lines only:
- Purpose: runtime asset loading helpers.
- Policies: prefer zero-copy where possible; lines ≤ 80 cols.
- Verify: `cargo clippy -p asset-loader --all-targets -- -D warnings`
- Quick checks: `cargo test -p asset-loader`
- Details: see crate README for API and examples.
