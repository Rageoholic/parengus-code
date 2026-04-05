
# Project CLAUDE

Short machine-focused pointers. For concise LLM guidance read the
per-crate `CLAUDE.md` files in each crate root (one-line summaries
tailored for tools/LLMs). Do not assume the human-facing README is
the machine summary.

Per-crate CLAUDE files:
- [rgpu-vk/CLAUDE.md](rgpu-vk/CLAUDE.md)
- [phoenix/CLAUDE.md](phoenix/CLAUDE.md)
- [samp-app/CLAUDE.md](samp-app/CLAUDE.md)
- [samp-app-noext/CLAUDE.md](samp-app-noext/CLAUDE.md)
- [parengus-tracing/CLAUDE.md](parengus-tracing/CLAUDE.md)
- [parengus-util/CLAUDE.md](parengus-util/CLAUDE.md)
- [asset-compiler/CLAUDE.md](asset-compiler/CLAUDE.md)
- [asset-loader/CLAUDE.md](asset-loader/CLAUDE.md)
- [asset-pipeline/CLAUDE.md](asset-pipeline/CLAUDE.md)
- [asset-shared/CLAUDE.md](asset-shared/CLAUDE.md)
- [xtask/CLAUDE.md](xtask/CLAUDE.md)

Note: Not every crate is required to include a `CLAUDE.md`. If a crate
does not include one, that simply means there is no crate-specific
policy documented yet.

Quick checks (run per-crate before PRs):
```
cargo clippy -p <crate> --all-targets -- -D warnings
cargo check -p <crate>
```

Repo setup (one-time):
```
git config core.hooksPath .githooks
```

Task graph (brief):
- Tasks live in `.tasks/`. Read [.tasks/index.md](.tasks/index.md)
    for the current task list and `next_id`.
- See [.tasks/CONVENTIONS.md](.tasks/CONVENTIONS.md) for task-file
    format and conventions.

AI disclosure (brief):
- Disclose AI assistance in externally visible artifacts (issues,
    PRs, release notes, README changes).
- Keep a short AI attribution note in the crate README when content
    is AI-assisted; mark AI use in the PR template.

Vulkan spec links (brief):
- Prefer the multi-page spec at `docs.vulkan.org`.
- Use this URL form:
```
https://docs.vulkan.org/spec/latest/chapters/<chapter>.html#<anchor>
```

Branch / PR / Issue templates (brief):
- Branch naming: `type/short-description` (types: `feat`, `fix`,
    `docs`, `chore`, `refactor`, `test`). Use 2–4 kebab-case words and
    branch from `main`.
- Issue templates: use files under `.github/ISSUE_TEMPLATE/`.
- PR title: `type: short description` (sentence case). Use the PR
    template at `.github/pull_request_template.md`.
- Author checklist (keep short): run `cargo clippy`, run tests,
    keep lines ≤ 80 columns, avoid new `unsafe` without a safety note.

- **Comments:** Wrap manually at a word boundary before column 80.

Adding a policy
----------------

When adding a new policy, follow this pattern:

1. Choose scope: `crate` or `workspace`.
2. Write a short policy entry (1–4 lines) and a 1–2 line rationale.
3. If `crate`-scoped: add the policy to the crate `README.md` under a
    **Policies** or **Development Guidelines** section, and add a one-line
    summary to the crate's `CLAUDE.md`.
4. If `workspace`-scoped: add the policy to this file under a
    `Workspace Policies` heading, and mention any affected crates in their
    `CLAUDE.md` summaries.
5. Open a PR that updates the docs and code examples, include the
    rationale and tests (if applicable), and request review from the
    maintainers.

Policy template (example):

```
Title: Short policy title
Scope: crate|workspace
Rule: one-line rule statement
Rationale: 1–2 lines explaining why
Example: short code snippet or file to change
```

