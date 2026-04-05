LLM summary — short lines only:
- Purpose: Vulkan wrapper crate (ash-based).
- Policies: never reuse feature structs; use fresh structs for queries
	and device creation, and ensure `p_next` is intentional when chained.
- Enforce `#![deny(unsafe_op_in_unsafe_fn)]`; lines ≤ 80 cols.
- Verify: `cargo clippy -p rgpu-vk --all-targets -- -D warnings`
- Quick checks: `cargo check -p rgpu-vk`
- Details: see README.md for human-facing guidance.
