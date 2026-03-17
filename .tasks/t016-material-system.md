---
id: t016
title: Material system
status: planned
area: renderer
parent: t005
---

Description:

Materials stored using SoA layout with hot fields grouped together for
GPU/cache efficiency. Depends on: descriptor_model

Subtasks:

- material_table_definition
- hot_field_grouping
- material_texture_indices
- material_parameter_buffers
