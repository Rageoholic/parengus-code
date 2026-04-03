next_id: t045
---

# Task Graph Index

> Generated with Claude's assistance.

This is the entry point for the Parengus Task Graph (PTG), a
lightweight hierarchical task system where each task is a markdown
file. Read this file first each session, then open the specific task
file(s) you need. For the full specification (file format, ID scheme,
and AI interaction conventions), see [CONVENTIONS.md](CONVENTIONS.md).

---

## Active Tree

- t003 Implement PBR shading [active]
  - t035 Directional lighting [planned]
- t001 Parengus Shader IR (PSIR) [planned, blocked by t033, t035]

---

### Active

| ID   | Title                                                      | Area     | Notes                                           |
|------|------------------------------------------------------------|----------|-------------------------------------------------|
| t003 | Implement PBR shading                                      | phoenix  |                                                 |

### Planned

| ID   | Title                                                      | Area     | Notes                                         |
|------|------------------------------------------------------------|----------|-----------------------------------------------|
| t001 | Parengus Shader IR (PSIR)                                  | psir     |                                               |
| t002 | xtask: smart rebuild-required check                        | infra    |                                               |
| t004 | Implement TUI                                              | phoenix  |                                               |
| t006 | RenderGraph / Executor split                               | renderer |                                               |
| t007 | Frames-in-flight infrastructure                            | renderer |                                               |
| t008 | RenderGraph compile phase                                  | renderer |                                               |
| t009 | Executor resource management                               | renderer |                                               |
| t010 | Explicit rendergraph dependencies                          | renderer |                                               |
| t011 | Resource granularity                                       | renderer |                                               |
| t012 | Upload system                                              | renderer |                                               |
| t013 | Draw submission model                                      | renderer |                                               |
| t014 | Descriptor model                                           | renderer |                                               |
| t015 | Shader binding model                                       | renderer |                                               |
| t016 | Material system                                            | renderer |                                               |
| t017 | Deferred decisions (notes)                                 | renderer |                                               |
| t018 | Future features (notes)                                    | renderer |                                               |
| t019 | Resource state tracker                                     | renderer |                                               |
| t021 | Emit SPIR-V 1.0 from PSIR emitter                          | psir     |                                               |
| t029 | UI system with text rendering and buttons                  | phoenix  |                                               |
| t030 | Font import pipeline (TTF → SDF atlas)                     | pipeline | child of t029                                 |
| t031 | Text rendering system (SSBO-indexed quads)                 | phoenix  | child of t029; widgets are follow-up          |
| t032 | Observable/event system                                    | phoenix  |                                               |
| t033 | UI widget system                                           | phoenix  | child of t029                                 |
| t034 | Error dialog / fatal message display                       | infra    |                                               |
| t035 | Directional lighting                                       | phoenix  | child of t003                                 |
| t037 | Harden state machine transitions with replace_with         | infra    |                                               |
| t038 | Separate samplers from textures (sampler table)            | renderer | cleanup: sampler table and grouping           |

### Future

| ID   | Title                                                      | Area     | Notes                                           |
|------|------------------------------------------------------------|----------|-------------------------------------------------|
| t039 | Per-frame descriptor & command pools                       | renderer | cleanup: per-frame pools for descriptors/commands |
| t040 | Optional device features (anisotropic optional)            | renderer | cleanup: optional device feature handling       |
| t041 | Single staging buffer with linear allocator                | phoenix  | staging buffer & upload linear allocator        |
| t044 | Design granular subfeature exposure in DeviceConfig        | rgpu-vk  | follow-up to query-first feature fix            |

### Idea

| ID   | Title                                                      | Area     | Notes                                         |
|------|------------------------------------------------------------|----------|-----------------------------------------------|
| t036 | Custom Vulkan memory allocator                             | rgpu-vk  | explicitly deferred                           |

### Done

| ID   | Title                                                      | Area     | Notes                                         |
|------|------------------------------------------------------------|----------|-----------------------------------------------|
| t005 | Import renderer task graph                                 | renderer | import of external graph                      |
| t024 | Material plumbing and SSBO infrastructure                  | phoenix  | child of t003                                 |
| t027 | Import DamagedHelmet test asset                            | phoenix  | child of t022                                 |
| t028 | Enable asset bake caching                                  | pipeline | avoid recompressing unchanged assets          |
| t042 | Use Debug Utils labels for queues and submissions          | renderer | label queues/submissions for captures         |
| t043 | Re-enable queue naming and add Phoenix debug-label helpers | renderer | re-enable queue names; add Phoenix helper stubs |

### Dropped

| ID   | Title                                                      | Area     | Notes                                         |
|------|------------------------------------------------------------|----------|-----------------------------------------------|
| t020 | Generate Slang compiler metadata                           | pipeline |                                               |
| t022 | Implement LearnOpenGL-style Blinn-Phong renderer           | phoenix  | pivoted to PBR directly                       |
| t023 | Blinn-Phong shader path                                    | phoenix  | child of t022                                 |
| t025 | Blinn-Phong lighting uniforms                              | phoenix  | child of t022                                 |
| t026 | Blinn-Phong visual validation                              | phoenix  | child of t022                                 |
