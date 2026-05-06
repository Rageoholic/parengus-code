---
id: t047
title: "Mesh format cleanup: strip TRS, add AABB"
status: planned
created: 2026-05-04
updated: 2026-05-04
parent: null
children: []
depends_on: []
blocked_by: []
area: pipeline
issue: null
---

## Context

The `.pmesh` format is a geometry format but currently embeds glTF
scene-graph node transforms (TRS) in each `SubMeshInfo` entry. This
is a category error — mesh files should describe geometry, not scene
placement. The TRS data crept in because glTF is a scene format and
the compiler currently bakes node world matrices into per-primitive
metadata.

Additionally, the format has no bounding volume data, which is needed
for culling at the mesh and submesh granularity, and as a future
foundation for per-meshlet bounds.

Design discussion recorded in memory:
`project_psir_design_decisions.md` (context) and session notes.

## Goal

- `SubMeshInfo` contains only geometry metadata: index range,
  `material_idx`, and AABB. No TRS.
- The glTF scene graph is fully flattened during compilation — all
  primitive positions are baked into a single coordinate space.
- `type = "mesh"` in the manifest implies flatten-scene-graph; no
  separate `gltf_usage` field needed.
- Per-submesh AABBs and a whole-mesh AABB are stored in the file.
- Albedo folds into `MeshMaterialData`; `MeshSubMeshAlbedo` section
  removed.
- Indices stored per-submesh, zero-based within the mesh's vertex
  buffer; loader concatenates at upload time.
- Format version bumped to 0.3.0.

## Plan

- [ ] **asset-shared**: bump `VERSION` to 0.3.0; add `MeshBounds`
      (kind 11) to `SectionKind`; add `Aabb` struct with
      `center: [f32; 3]` and `half_extents: [f32; 3]`; expose
      `min`/`max` as computed methods; remove `MeshSubMeshAlbedo`
      (kind 10) — albedo folds into material data.
- [ ] **asset-pipeline**: no changes needed — `type = "mesh"` already
      implies flatten-scene-graph behaviour. `type = "scene"` will
      drive different compiler behaviour when that pipeline exists.
- [ ] **asset-compiler**: flatten glTF scene graph (bake node world
      matrices into positions before storing); compute whole-mesh AABB
      from all positions; shift all positions so AABB center is at
      origin; compute per-submesh `half_extents` from shifted
      positions; compute whole-mesh `half_extents`; strip TRS from
      submesh table entries; add `material_idx: u32` to each entry;
      move albedo into `MeshMaterialData` (fold `MeshSubMeshAlbedo`
      away); emit `MeshBounds` section (1 element, 12 bytes).
- [ ] **asset-loader**: remove TRS fields from `SubMeshInfo`; add
      `aabb: Aabb`, `material_idx: u32`; remove `sub_mesh_albedos`
      from `MeshAsset`;
      add `aabb: Aabb` to `MeshAsset`; update `MeshSubMeshTable`
      decoder (new entry size); decode `MeshBounds` section.
- [ ] **lightbox/src/main.rs** (currently `phoenix`): remove
      `trs_to_mat4`; submesh model matrix is now identity (compiler
      bakes all transforms into vertex positions).
- [ ] Verify: `cargo clippy` clean on all affected crates; re-bake
      assets; confirm phoenix renders correctly.

## Thinking

TRS removal is the right call: a mesh is geometry, not a scene node.
The glTF node transforms were an artefact of using glTF (a scene
format) as a mesh source — the compiler should fully flatten them
rather than forwarding them.

Precision offsets (originally considered as `MeshOrigin`) are a scene
concern — meshes should always be authored near the origin. Dropped.

AABB stored as `center: [f32; 3]` + `half_extents: [f32; 3]` (24
bytes). Half-extents preferred over full extents — almost all AABB
math (intersection, containment, frustum culling) uses half-extents
naturally, avoiding a divide at use sites. `min`/`max` exposed as
computed methods (`center ± half_extents`).

Whole-mesh AABB has `center = (0,0,0)` by construction (compiler
normalizes positions to AABB center). Could store as `half_extents`
only (12 bytes) but using the same `Aabb` type keeps the API uniform.

`SubMeshInfo` entry layout (36 bytes):
  index_base:      u32  ( 4 B)
  index_count:     u32  ( 4 B)
  material_idx:    u32  ( 4 B)
  aabb_center:    3×f32 (12 B)
  aabb_half_ext:  3×f32 (12 B)
  — no padding; 36 bytes is fine —

`MeshBounds` section: 1 element, 24 bytes (full Aabb, center always
zero).

glTF usage tagging: considered a `gltf_usage` manifest field but
dropped — `type = "mesh"` already implies flatten-scene-graph.
When a scene pipeline exists, `type = "scene"` will drive that.
When Blender becomes the primary authoring tool, the source format
changes and the question goes away entirely.

Index format: indices are per-submesh in the format (stored verbatim
from glTF, zero-based within the mesh's vertex buffer). The loader
concatenates them into a single GPU index buffer at upload time;
`firstIndex` offsets into it per draw, `vertexOffset` is the mesh's
base into the GPU vertex buffer. No rebasing in compiler or loader.

Vertex sharing across submesh boundaries is not a real case: even
if positions and UVs happen to match at a material seam, normals and
tangents differ per material. Cross-submesh index sharing is
therefore not worth designing for.
