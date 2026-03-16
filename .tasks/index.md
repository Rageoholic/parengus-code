next_id: t020
---

# Task Graph Index

> Generated with Claude's assistance.

This is the entry point for the Parengus Task Graph (PTG), a
lightweight hierarchical task system where each task is a markdown
file. Read this file first each session, then open the specific task
file(s) you need. For the full specification (file format, ID scheme,
AI conventions), see [CONVENTIONS.md](CONVENTIONS.md).

---

## Active Tree

*(No active tasks yet.)*

---

## All Tasks

| ID   | Title                               | Status  | Area    | Notes          |
|------|-------------------------------------|---------|---------|----------------|
| t001 | Parengus Shader IR (PSIR)           | planned | psir    |                |
| t002 | xtask: smart rebuild-required check | planned | infra   |                |
| t003 | Implement PBR shading               | blocked | phoenix | blocked by t001, t019|
| t004 | Implement TUI                       | planned | phoenix |                |
| t005 | Import renderer task graph          | completed | renderer| import of external graph |
| t006 | RenderGraph / Executor split        | planned | renderer| rendergraph_executor_split |
| t007 | Frames-in-flight infrastructure     | planned | renderer| frames_in_flight |
| t008 | RenderGraph compile phase           | planned | renderer| rendergraph_compile_phase |
| t009 | Executor resource management        | planned | renderer| executor_resource_management |
| t010 | Explicit rendergraph dependencies   | planned | renderer| explicit_rendergraph_dependencies |
| t011 | Resource granularity                | planned | renderer| resource_granularity |
| t012 | Upload system                       | planned | renderer| upload_system |
| t013 | Draw submission model               | planned | renderer| draw_submission_model |
| t014 | Descriptor model                    | planned | renderer| descriptor_model |
| t015 | Shader binding model                | planned | renderer| shader_binding_model |
| t016 | Material system                     | planned | renderer| material_system |
| t017 | Deferred decisions (notes)          | planned | renderer| pipeline_compilation_and_caching, transient_resource_memory_aliasing |
| t018 | Future features (notes)             | planned | renderer| meshlets |
| t019 | Resource state tracker              | planned | renderer| resource_state_tracker |
