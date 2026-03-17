---
id: t010
title: Explicit rendergraph dependencies
status: planned
area: renderer
parent: t005
---

Description:

RenderGraph stores explicit pass dependencies rather than deriving them
automatically from resource usage. Depends on: rendergraph_pass_definition

Subtasks:

- dependency_edge_representation
- dependency_validation
- topological_sort
