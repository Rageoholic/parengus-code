LLM summary — short lines only:
- Purpose: shared asset types and helpers.
- Policies: keep types stable; avoid breaking changes without notice.
- Verify: `cargo clippy -p asset-shared --all-targets -- -D warnings`
- Quick checks: `cargo test -p asset-shared`
- Details: see crate README for format specifics.
