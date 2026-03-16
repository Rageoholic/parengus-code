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

| ID   | Title                                                                           | Status    | Area     | Notes                         |
|------|---------------------------------------------------------------------------------|-----------|----------|-------------------------------|
| t001 | Parengus Shader IR (PSIR)                                                       | planned   | psir     |                               |
| t002 | xtask: smart rebuild-required check                                             | planned   | infra    |                               |
| t003 | Implement PBR shading                                                           | blocked   | phoenix  | blocked by t001, t019         |
| t004 | Implement TUI                                                                   | planned   | phoenix  |                               |
| t005 | Import renderer task graph                                                      | completed | renderer | import of external graph      |
| t006 | Split renderer into logical RenderGraph and physical Executor                   | planned   | renderer |                               |
| t007 | Add frames-in-flight with per-frame allocators and state                        | planned   | renderer |                               |
| t008 | Add compile phase to RenderGraph for pass ordering and resource allocation      | planned   | renderer |                               |
| t009 | Executor owns Vulkan resource, barrier, and semaphore management                | planned   | renderer |                               |
| t010 | Store explicit pass dependencies in RenderGraph                                 | planned   | renderer |                               |
| t011 | Track resources at whole-resource granularity initially                         | planned   | renderer |                               |
| t012 | Hybrid upload: bulk startup uploads and streaming graph-pass uploads            | planned   | renderer |                               |
| t013 | Hybrid draw: CPU batches by mesh/material, GPU culls and indirect-draws         | planned   | renderer |                               |
| t014 | Derive global bindless descriptor layout from shader IR                         | planned   | renderer |                               |
| t015 | Rewrite shader IR bindings to match engine descriptor layout                    | planned   | renderer |                               |
| t016 | SoA material storage with hot fields grouped for cache efficiency               | planned   | renderer |                               |
| t017 | Deferred design decisions: pipeline caching and memory aliasing                 | planned   | renderer |                               |
| t018 | Future exploration: meshlets                                                    | planned   | renderer |                               |
| t019 | Track per-resource Vulkan queue-family and image-layout state                   | planned   | renderer |                               |
