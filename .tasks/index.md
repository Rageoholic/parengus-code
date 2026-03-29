next_id: t038
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
  - t024 Material plumbing and SSBO infrastructure [active]
  - t035 Directional lighting [planned, blocked by t024]
- t001 Parengus Shader IR (PSIR) [planned, blocked by t033, t035]

---

## All Tasks

| ID   | Title                               | Status    | Area     | Notes                    |
|------|-------------------------------------|-----------|----------|--------------------------|
| t001 | Parengus Shader IR (PSIR)           | planned   | psir     |                          |
| t002 | xtask: smart rebuild-required check | planned   | infra    |                          |
| t003 | Implement PBR shading               | active    | phoenix  |                          |
| t004 | Implement TUI                       | planned   | phoenix  |                          |
| t005 | Import renderer task graph          | completed | renderer | import of external graph |
| t006 | RenderGraph / Executor split        | planned   | renderer |                          |
| t007 | Frames-in-flight infrastructure     | planned   | renderer |                          |
| t008 | RenderGraph compile phase           | planned   | renderer |                          |
| t009 | Executor resource management        | planned   | renderer |                          |
| t010 | Explicit rendergraph dependencies   | planned   | renderer |                          |
| t011 | Resource granularity                | planned   | renderer |                          |
| t012 | Upload system                       | planned   | renderer |                          |
| t013 | Draw submission model               | planned   | renderer |                          |
| t014 | Descriptor model                    | planned   | renderer |                          |
| t015 | Shader binding model                | planned   | renderer |                          |
| t016 | Material system                     | planned   | renderer |                          |
| t017 | Deferred decisions (notes)          | planned   | renderer |                          |
| t018 | Future features (notes)             | planned   | renderer |                          |
| t019 | Resource state tracker              | planned   | renderer |                          |
| t020 | Generate Slang compiler metadata    | planned   | pipeline |                          |
| t021 | Emit SPIR-V 1.0 from PSIR emitter   | planned   | psir     |                          |
| t022 | Implement LearnOpenGL-style Blinn-Phong renderer | dropped | phoenix | pivoted to PBR directly |
| t023 | Blinn-Phong shader path            | dropped   | phoenix  | child of t022 |
| t024 | Material plumbing and SSBO infrastructure | active | phoenix | child of t003 |
| t025 | Blinn-Phong lighting uniforms      | dropped   | phoenix  | child of t022 |
| t026 | Blinn-Phong visual validation      | dropped   | phoenix  | child of t022 |
| t027 | Import DamagedHelmet test asset    | done      | phoenix  | child of t022 |
| t028 | Enable asset bake caching          | done      | pipeline | avoid recompressing unchanged assets |
| t029 | UI system with text rendering and buttons | planned | phoenix |                 |
| t030 | Font import pipeline (TTF → SDF atlas) | planned | pipeline | child of t029 |
| t031 | Text rendering system (SSBO-indexed quads) | planned | phoenix | child of t029; widgets are follow-up |
| t032 | Observable/event system                    | planned | phoenix |               |
| t033 | UI widget system                           | planned | phoenix | child of t029 |
| t034 | Error dialog / fatal message display       | planned | infra   |               |
| t035 | Directional lighting                       | planned | phoenix | child of t003 |
| t036 | Custom Vulkan memory allocator             | idea    | rgpu-vk | explicitly deferred |
| t037 | Harden state machine transitions with replace_with | planned | infra   |               |
