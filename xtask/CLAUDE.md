LLM summary — short lines only:
- Purpose: workspace task runner (xtask pattern).
- Policies: commands should be idempotent where reasonable; lines ≤ 80 cols.
- Verify: run the relevant `cargo xtask` commands locally.
- Quick checks: `cargo xtask build`, `cargo xtask build-samp-app`
- Details: see `xtask/src` for available tasks and flags.
