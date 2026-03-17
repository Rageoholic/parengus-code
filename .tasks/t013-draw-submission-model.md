---
id: t013
title: Draw submission model
status: planned
area: renderer
parent: t005
---

Description:

Hybrid draw submission. CPU groups instances by mesh and material. GPU
performs visibility culling and drives indirect draws. Depends on:
frames_in_flight

Subtasks:

- cpu_instance_batching
- mesh_material_sorting
- instance_table_generation
- compute_visibility_culling
- indirect_draw_buffer_generation
- indirect_draw_execution
