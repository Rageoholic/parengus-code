---
id: t015
title: Shader binding model
status: planned
area: renderer
parent: t005
---

Description:

Shaders declare logical resources in custom shader IR. Engine defines
the official descriptor layout. Shader compiler rewrites shader bindings
to match the layout, enabling switching between bindless and bindful
models. Depends on: descriptor_model

Subtasks:

- shader_ir_resource_declarations
- shader_layout_resolution
- binding_rewrite_pass
- spirv_generation
