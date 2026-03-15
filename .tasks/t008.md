---
id: t008
title: RenderGraph compile phase
status: planned
area: renderer
parent: t005
---

Description:

RenderGraph has a compile phase that resolves dependencies, determines
pass order, and allocates transient resources.
Depends on: rendergraph_executor_split

Subtasks:

- rendergraph_compiler
- pass_order_resolution
- resource_lifetime_analysis
- transient_resource_allocation
- compiled_graph_representation
- compiled_graph_cache
