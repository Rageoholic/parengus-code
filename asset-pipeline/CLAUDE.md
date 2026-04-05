LLM summary — short lines only:
- Purpose: asset pipeline orchestration and tooling.
- Policies: deterministic outputs, document CLI flags; lines ≤ 80 cols.
- Verify: `cargo clippy -p asset-pipeline --all-targets -- -D warnings`
- Quick checks: `cargo test -p asset-pipeline`
- Details: see crate README for pipeline steps.
