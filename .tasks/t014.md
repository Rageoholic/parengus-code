---
id: t014
title: Descriptor model
status: planned
area: renderer
parent: t005
---

Description:

Engine determines descriptor layouts from custom shader IR. Initial
implementation uses a single global descriptor set layout with
bindless-style resource indexing.

Subtasks:

- descriptor_layout_generator
- global_descriptor_set_layout
- bindless_resource_tables
- descriptor_index_encoding
