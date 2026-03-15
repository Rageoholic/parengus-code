# Parengus Task Graph — Conventions

> Generated with Claude's assistance.

This document is the authoritative spec for the PTG (Parengus Task
Graph) system. Any AI assistant working in this repository should
read this file before creating or modifying task files.

---

## Purpose

PTG is a lightweight, markdown-only hierarchical task graph. Each
task is a single `.md` file with YAML frontmatter. The graph lives
in `.tasks/` and is committed to the repository. It replaces the
flat `FUTURE_WORK.md` and gives AI assistants a persistent,
structured place to record context, plans, and reasoning across
sessions.

---

## Directory Layout

```
.tasks/
  CONVENTIONS.md    <- this file
  index.md          <- entry point; read this first every session
  t001-<slug>.md    <- root task
  t002-<slug>.md    <- another root task
  t002a-<slug>.md   <- child of t002 (letter suffix marks children)
```

- One flat directory. Hierarchy lives in frontmatter, not
  subdirectories.
- Filenames: `t<id>[-<qualifier>]-<slug>.md`
  - `<id>` is the numeric ID (e.g. `001`)
  - `<qualifier>` is an optional letter suffix used to mark child
    tasks for visual grouping in directory listings (e.g. `002a`)
  - `<slug>` is 2–4 hyphenated words matching the title

---

## ID Scheme

- IDs are short sequential integers zero-padded to three digits:
  `t001`, `t002`, `t003`, …
- Children get their own sequential ID; the parent-child relationship
  is declared in frontmatter only (not in the ID).
- The current `next_id` is stored in `index.md` frontmatter. Always
  read the index before allocating a new ID.

---

## Frontmatter Specification

Every task file begins with a YAML frontmatter block. All fields use
snake_case. Lines must not exceed 80 columns.

```yaml
---
id: t001
title: "Short task title"
status: planned
# status values:
#   idea     - not yet committed, just captured
#   planned  - will be done, not started
#   active   - currently in progress
#   blocked  - cannot proceed; see blocked_by
#   done     - completed
#   dropped  - cancelled, will not be done
created: 2026-03-14
updated: 2026-03-14
parent: null          # omit or set null for root tasks
children: []          # list of child task IDs
depends_on: []        # tasks that must be done before this starts
blocked_by: []        # subset of depends_on currently blocking
area: phoenix         # free-form grouping tag
issue: 42             # GitHub issue number; omit if none
---
```

Required fields: `id`, `title`, `status`, `created`, `updated`.
All other fields are optional but should be included when known.

---

## Body Sections

The body follows the frontmatter in this fixed order. Omit a section
only if it truly does not apply yet. Wrap prose at 80 columns.

```markdown
## Context

Why this task exists. What problem it solves. Links to relevant
code, design docs, or issues.

## Goal

A crisp 1–3 sentence statement of what "done" means.

## Plan

Ordered steps. Check off completed ones; append new discoveries.

- [ ] Step one
- [ ] Step two
- [x] Step three (done 2026-03-14)

## Thinking

Design notes, trade-offs considered, dead ends encountered.
Freeform. **Append here; never rewrite existing content.**

## Outcome

Filled when status is `done` or `dropped`. Short summary of what
was delivered or why the task was abandoned. Link to PR/commit.
```

---

## Index File

`.tasks/index.md` is the entry point. Read it at the start of any
session that touches tasks.

Its frontmatter contains `next_id`. Its body contains:
1. A one-paragraph system summary with a link to this file.
2. An active-task tree (indented list showing hierarchy).
3. A table of all tasks (ID, title, status, area, issue).

The AI regenerates the index (does not diff it) whenever tasks are
created, updated, or closed. **If the index and a task file disagree,
the task file is the source of truth.**

---

## AI Interaction Conventions

### Session start
1. Read `.tasks/index.md` to get the overview and `next_id`.
2. Read the specific task file(s) relevant to the current work.
3. Read `depends_on` or parent task files if additional context
   is needed.

### Creating a new task
1. Read `index.md` to get `next_id` (e.g. `t004`).
2. Create `.tasks/t004-<slug>.md` with frontmatter + all body
   sections.
3. If it has a parent: update the parent file's `children` list
   and `updated` date.
4. Rewrite `index.md`: increment `next_id`, add a row to the table,
   update the tree.

### Updating a task mid-work
- Check off completed steps in `## Plan`.
- Append new notes to `## Thinking` (never rewrite it).
- Update the `updated` date.
- Change `status` if appropriate and update the index table row.

### Closing a task
1. Set `status: done` or `status: dropped`.
2. Fill in `## Outcome`.
3. Update the `updated` date.
4. Update `index.md`.

### Dependency discipline
Before marking a task `active`, check its `depends_on` list.
If any listed task is not `done`, either leave this task as
`planned` or move the blocking tasks into `blocked_by` and set
`status: blocked`.

---

## Relationship to Other Tooling

- **GitHub Issues** — the `issue:` field links a task file to its
  public issue. Issues track discussion; task files track AI context
  and planning. When closing a task, the corresponding PR should
  carry `Closes #N`.
- **`.local/ai/` docs** — remain as ephemeral design scratch-pads.
  When a doc graduates to active work, create a task file and link
  back to the doc in `## Context`.
- **xtask** — no `cargo xtask tasks` subcommand exists yet. Add it
  only if terminal tree rendering proves useful in practice.

---

## Quick Reference

| Action | Files touched |
|--------|---------------|
| Session start | `index.md` (read), task file (read) |
| New task | new task file + `index.md` + parent file (if any) |
| Progress update | task file only (+ `index.md` if status changes) |
| New child task | new task file + parent file + `index.md` |
| Close task | task file (status, Outcome) + `index.md` |
