# Contributing

> AI-assisted document — generated with Claude Code.

## Branch Names

Use `type/short-description` with kebab-case words.

| Type | Use for |
|------|---------|
| `feat/` | new feature or capability |
| `fix/` | bug fix |
| `docs/` | documentation only |
| `chore/` | maintenance, deps, tooling, CI |
| `refactor/` | restructuring without behavior change |
| `test/` | adding or updating tests |

**Examples**

```
feat/instanced-rendering
fix/swapchain-format-srgb
docs/contributing-guidelines
chore/update-ash-dep
refactor/device-selection
test/surface-lifetime
```

**Rules**
- 2–4 words after the slash; be specific enough to identify the work
- All lowercase, no underscores
- Branch off `main`; keep branches short-lived

---

## Filing Issues

Choose the right template when opening an issue
(`.github/ISSUE_TEMPLATE/`).

### Bug reports

Include:
1. **What happened** — a concise description of the unexpected behavior
2. **Steps to reproduce** — minimal code or commands to trigger it
3. **Expected behavior** — what should have happened
4. **Environment** — OS, GPU driver version, Vulkan SDK version, `rustc`
   version
5. **Relevant output** — validation layer messages, panic backtraces

### Feature requests

Include:
1. **Motivation** — the problem this solves or the use-case it enables
2. **Proposed API / behavior** — a sketch of what the change looks like
   from the caller's perspective
3. **Alternatives considered** — other approaches and why you prefer this
   one
4. **Scope** — does it affect `rgpu`, `samp-app`, or both?

### General rules
- Search for duplicates before opening a new issue
- One concern per issue; split unrelated topics
- If you used AI assistance to draft the issue body, note it at the
  bottom (see [AI Disclosure](CLAUDE.md#ai-disclosure))

---

## Pull Requests

### Title

Mirror the branch-name convention: `type: short description` in
sentence case.

```
feat: add instanced rendering support
fix: prefer B8G8R8A8_SRGB swapchain format
docs: add contributing guidelines
chore: update ash to 0.38
```

### Body

Use the PR template (`.github/pull_request_template.md`). Required
sections:

| Section | What to write |
|---------|---------------|
| **Summary** | Why this change exists; 1–3 bullets |
| **Changes** | What was modified at a high level |
| **Test plan** | How you verified correctness |
| **AI assistance** | Check the box if AI tools were used |

### Review checklist (author)
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo test` passes
- [ ] Lines ≤ 80 columns
- [ ] No new `unsafe` without a safety comment
- [ ] If the change touches `samp-app` or `samp-app-noext`, consider
      whether the sibling app needs the same fix (they share structure
      and bugs tend to appear in both)
- [ ] AI assistance disclosed if applicable
