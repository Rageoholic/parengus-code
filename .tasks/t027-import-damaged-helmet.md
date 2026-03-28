---
id: t027
title: "Import DamagedHelmet test asset"
status: done
created: 2026-03-24
updated: 2026-03-28
parent: t022
children: []
depends_on:
  - t023
  - t024
blocked_by: []
area: phoenix
---

## Context

We need representative real-world mesh/material assets to evaluate the
Blinn-Phong baseline renderer. DamagedHelmet.gltf and FlightHelmet.gltf
are standard Khronos reference models suitable for repeatable visual checks.
DamagedHelmet has a single mesh/primitive with a PBR material set.
FlightHelmet has 6 meshes (one per part) with 5 distinct PBR materials.

## Goal

DamagedHelmet and FlightHelmet are importable and renderable in Phoenix
through the asset pipeline. The baker supports multiple glTF nodes, multiple
meshes, and multiple primitives. Material channels are stored as structured
data in .pmesh; tex_refs are auto-derived from glTF material data rather
than declared manually in the manifest.

## Plan

### Phase 0 — Documentation
- [x] Update this task file with the full plan
- [x] Update `asset-formats.md` to document MeshSubMeshTable + MeshMaterialData

### Phase 1 — Asset file housekeeping
- [ ] Rename 15 FlightHelmet textures in `assets/tex/` to short `fh-*` names
- [ ] Update `assets/FlightHelmet.gltf` image URIs to match renamed textures
- [ ] Update `assets/manifest.toml`:
  - Remove `tex_refs` from `duck` and `damaged-helmet` entries
  - Change `dh-normal` format from `bc7` to `bc5`
  - Add `flight-helmet` mesh entry + 15 FlightHelmet image entries
- [ ] Update `phoenix/assets.toml`:
  - Add `damaged-helmet` + 5 DamagedHelmet image assets
  - Add `flight-helmet` + 15 FlightHelmet image assets

### Phase 2 — Remove `tex_refs` from ManifestEntry
- [ ] Remove `tex_refs: HashMap<String, String>` from `ManifestEntry` in
  `asset-pipeline/src/lib.rs`

### Phase 3 — .pmesh sub-mesh support
- [ ] Add `MeshSubMeshTable = 8` and `MeshMaterialData = 9` to `SectionKind`
  in `asset-shared/src/lib.rs`
- [ ] Rewrite `asset-compiler/src/mesh.rs` compile() for multi-node traversal:
  - Walk all nodes depth-first; accumulate world TRS via TRS composition
  - Per primitive: apply yup_to_zup to vertices (do NOT bake node transform)
  - Convert node world TRS to Z-up engine space; store in SubMeshTable
  - Auto-derive tex_refs by matching glTF texture URI to manifest entry.file
  - Emit MeshMaterialData section (56B of material constants)
  - Emit KHR_materials_transmission warning and skip extension
- [ ] Update `asset-loader/src/lib.rs` with SubMeshInfo, SubMeshView,
  sub_mesh_count(), sub_mesh() API; remove old flat accessor methods
- [ ] Update `phoenix/src/main.rs` to use `mesh.sub_mesh(0).unwrap()` API

### Verification
- [ ] `cargo clippy` clean on all affected crates
- [ ] `cargo test -p asset-compiler`
- [ ] `cargo xtask build-phoenix`
- [ ] Visual: deterministic test scene with duck + DamagedHelmet + FlightHelmet

## Thinking

DamagedHelmet is PBR-authored; for Blinn-Phong rendering we include all
texture channels (albedo, normal, AO, metalRoughness, emissive) — the AO
channel is usable for the ambient term, and the full PBR data is preserved
for the future PBR renderer (t003).

FlightHelmet has 6 parts, each a separate glTF mesh. All nodes have identity
transforms so sub-mesh TRS will be identity. The Lenses material uses
KHR_materials_transmission (glass) — we warn and skip this extension.

Normal maps use BC5 (XY channels only); the shader reconstructs
Z = sqrt(1 - x² - y²). ORM textures use BC7.

Separate .pmat file (material shared across meshes, runtime-editable) is
out of scope; tracked under material system task (t016/t024).

The baker matches glTF texture URIs to manifest entries by `entry.file`
(forward-slash path). The TextureId hash uses the logical `entry.name`,
not the filename — these are independent per the asset ID scheme.

## Outcome

(not yet filled — task is in progress)
