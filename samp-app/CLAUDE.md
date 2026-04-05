LLM summary — short lines only:
- Purpose: Sample app using winit; uses rgpu-vk for rendering.
- Policies: run per-crate clippy; lines ≤ 80 cols.
- Verify: `cargo clippy -p samp-app --all-targets -- -D warnings`
- Quick checks: `cargo check -p samp-app`
- Details: see crate README for usage and runtime assets.
