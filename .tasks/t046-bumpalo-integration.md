---
id: t045
title: "Utilize bumpalo for temporary allocations"
status: planned
created: 2026-04-05
updated: 2026-04-05
parent: null
children: []
depends_on: []
blocked_by: []
area: whole-repo
---

## Context

Allocations are an operation we absolutely will want to do during the frame but
we want to adopt a more disciplined approach to it than the standard "call
malloc and pray" strategy most applications do. Most allocations can be nested
hierarchically, with only the final return value ever escaping.