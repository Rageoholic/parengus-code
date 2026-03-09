# Asset System Plan

> AI-assisted document — generated with Claude Code.

## Overview

Three new crates implement a compiled asset pipeline:

- **`asset-shared`** — binary format constants and types shared between
  compiler and loader
- **`asset-compiler`** — build-time binary invoked by xtask; converts
  raw assets to compiled formats
- **`asset-loader`** — runtime library used by apps; GPU-agnostic,
  returns CPU-side data via iterators

The existing `asset-pipeline` crate (manifest parsing, `AssetMap`
generation) is unchanged.

---

## Compiled Formats

### `.pmesh` — Compiled Mesh

**File layout:**

```
[FileHeader]
[SectionHeader; N]
[section data...]
```

**`FileHeader`:**

| Field | Type | Notes |
|-------|------|-------|
| `magic` | `u32` | `*b"PMSH"` |
| `version` | `u16` | starts at 1 |
| `asset_kind` | `u16` | `AssetKind::Mesh` |
| `section_count` | `u32` | |

All fields are read with explicit `from_le_bytes` — no `Pod` cast,
no endianness assumption.

**`SectionHeader`:**

| Field | Type | Notes |
|-------|------|-------|
| `kind` | `u32` | `SectionKind` discriminant |
| `compression` | `u32` | `Compression` discriminant |
| `_reserved` | `u32` | always 0; reserved for future use |
| `byte_offset` | `u32` | offset into file |
| `byte_len` | `u32` | uncompressed size; use for allocation |
| `compressed_byte_len` | `u32` | compressed size; use for seeking. Equal to `byte_len` if uncompressed |
| `element_count` | `u32` | number of elements (not bytes); backs `ExactSizeIterator::len()` |

**`Compression` enum:**

| Variant | Value | Notes |
|---------|-------|-------|
| `None` | 0 | section data stored as-is |
| `Lz4` | 1 | LZ4 block format |

**Section kinds for mesh** (structure of arrays — one section per
attribute):

| `SectionKind` | Contents | Required | Default compression |
|---------------|----------|----------|---------------------|
| `MeshPositions` | `[[f32; 3]]` — Z-up, little-endian | yes | `Lz4` |
| `MeshNormals` | `[[f32; 3]]` — Z-up, little-endian | yes | `Lz4` |
| `MeshTangents` | `[[f32; 4]]` — xyz + w handedness sign, little-endian | yes | `Lz4` |
| `MeshTexCoord0` | `[[f32; 2]]` — UV0, little-endian | yes | `Lz4` |
| `MeshTexCoord1` | `[[f32; 2]]` — UV1, little-endian | no | `Lz4` |
| `MeshIndices16` | `[u16]` — little-endian | one of these | `Lz4` |
| `MeshIndices32` | `[u32]` — little-endian | is required | `Lz4` |
| `MeshTexRef` | logical asset name (UTF-8, no null terminator) | no | `None` |

Optional attributes (`MeshTexCoord1`) are signaled by section
presence alone — no bitmask needed. Loaders skip unknown section
kinds, so new optional attributes can be added without a version bump
as long as they don't change the layout of existing sections.

All per-vertex sections have the same `element_count` (vertex count).
`MeshIndices` `element_count` is the index count.

The compiler always generates `MeshNormals` and `MeshTangents` if
absent in the glTF primitive (averaged face normals; Mikktspace
tangents). The coordinate transform `(x,y,z) → (x,z,-y)` (Y-up to
Z-up) is applied at compile time.

Index width is `u16` for now. Widening to `u32` requires a new
`SectionKind::MeshIndices32` variant and a `version` bump.

The compiler takes only the first mesh primitive. Multi-primitive
support is deferred.

### `.ptex` — Compiled Texture

**`FileHeader`:** same structure as `.pmesh`, magic `*b"PTEX"`,
`asset_kind = AssetKind::Texture`.

**Section kinds for texture:**

| `SectionKind` | Contents |
|---------------|----------|
| `TextureMip` | one section per mip level, in order (mip 0 first) |

**Texture metadata** is stored in the `FileHeader` or a leading
`TextureInfo` section (TBD when compiler is written):

| Field | Notes |
|-------|-------|
| `format` | `TexFormat` enum: `Bc7`, `Bc5`, `Bc4`, `Rgba8`, … |
| `color_space` | `ColorSpace` enum: `Srgb`, `Linear` |
| `width` | pixels, mip 0 |
| `height` | pixels, mip 0 |
| `mip_count` | |

The loader surfaces `format` and `color_space` so the app can select
the correct `VkFormat` (e.g. `R8G8B8A8_SRGB` vs `R8G8B8A8_UNORM`,
`BC7_SRGB_BLOCK`, etc.).

---

## Manifest Changes

### Mesh assets

```toml
[[asset]]
name = "duck"
file = "mesh/Duck.pmesh"
type = "mesh"
source_file = "mesh/Duck.gltf"
source = "https://github.com/KhronosGroup/glTF-Sample-Assets"
author = "Khronos Group"
license = "CC-BY 4.0"
```

### Image assets

```toml
[[asset]]
name = "duck-tex"
file = "tex/duck-tex.ptex"
type = "image"
source_file = "tex/DuckCM.png"
format = "bc7"
color_space = "srgb"
mips = true
```

`format`, `color_space`, and `mips` are explicit in the manifest —
no convention magic. The compiler does exactly what the manifest says.

---

## `asset-shared` Crate

**Dependencies:** none initially. No bytemuck, no serde.

Contains:

- Magic constants and `VERSION`
- `AssetKind` enum
- `SectionKind` enum
- `Compression` enum
- `FileHeader` and `SectionHeader` structs
- `TexFormat` enum
- `ColorSpace` enum

`FileHeader` and `SectionHeader` are plain Rust structs in
`asset-shared`. The compiler constructs them and serializes each field
with `to_le_bytes`; the loader reads field-by-field with
`from_le_bytes` and constructs them. No `Pod` cast, no endianness
assumption.

---

## `asset-compiler` Crate

Binary invoked by xtask:

```
asset-compiler mesh  <input.gltf> <output.pmesh>
asset-compiler image <input.png>  <output.ptex> \
    --format bc7 --color-space srgb --mips
```

Exit 0 on success, error text on stderr, non-zero exit on failure.

**Mesh compilation steps:**

1. `gltf::import(src)`
2. Read first mesh, first primitive
3. Read `POSITION` → apply Y-up to Z-up transform: `(x,y,z) → (x,z,-y)`
   → `Vec<[f32; 3]>`
4. Read `NORMAL` if present (transform to Z-up); otherwise generate
   averaged face normals → `Vec<[f32; 3]>`
5. Read `TANGENT` if present (transform to Z-up); otherwise generate
   via Mikktspace → `Vec<[f32; 4]>`
6. Read `TEXCOORD_0` → `Vec<[f32; 2]>`
7. Read `TEXCOORD_1` if present → `Vec<[f32; 2]>`
8. Read indices → `Vec<u16>` if max index ≤ `u16::MAX`, else `Vec<u32>`
9. Write texture logical name as `MeshTexRef` section (from manifest,
   not from the glTF URI)
10. LZ4-compress each attribute section and the index section
    independently
11. Write `FileHeader`, `[SectionHeader; N]`, compressed section data.
    One `SectionHeader` per attribute section present, plus indices
    and tex ref.

Note: the coordinate transform is applied here, not at runtime.
`load_duck()` in phoenix currently loads raw glTF positions without
any transform — migrating to `asset-loader` is also the fix for
that missing transform.

**Image compilation steps:**

1. Decode source image to raw RGBA8 pixels
2. Generate full mip chain
3. Compress each mip to the target BCn format
4. Write `FileHeader`, one `SectionHeader` per mip, mip data

---

## `asset-loader` Crate

**Dependencies:** `asset-shared`, LZ4 decompression crate (for now).
No `gltf`, no GPU types, no bytemuck. Additional decompression crates
added as `Compression` variants are introduced.

**Public API:**

```rust
pub mod mesh;
pub mod tex;

pub use mesh::{
    MeshAsset, Indices,
    PositionIter, NormalIter, TangentIter,
    TexCoordIter, IndexIter, TexRefIter,
};
pub use tex::TexAsset;
```

**`MeshAsset`:**

```rust
pub struct MeshAsset { /* owns decompressed attribute vecs */ }

impl MeshAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError>;

    pub fn positions(&self)   -> PositionIter<'_>;  // [f32; 3]
    pub fn normals(&self)     -> NormalIter<'_>;    // [f32; 3]
    pub fn tangents(&self)    -> TangentIter<'_>;   // [f32; 4]
    pub fn tex_coords(&self)  -> TexCoordIter<'_>;  // [f32; 2]
    pub fn tex_coords1(&self) -> Option<TexCoordIter<'_>>; // UV1
    pub fn indices(&self)     -> Indices<'_>;
    pub fn tex_refs(&self)    -> TexRefIter<'_>;    // &str
}

pub enum Indices<'a> {
    U16(IndexIter<'a, u16>),
    U32(IndexIter<'a, u32>),
}
```

All iterators implement `ExactSizeIterator`. `iter.len()` drives
staging buffer allocation:

```rust
let mesh = MeshAsset::open(&path)?;
let n = mesh.positions().len();
let staging_size = n * size_of::<[f32; 3]>();
```

Each attribute section is LZ4-decompressed on `open()` into its own
typed `Vec` (with `from_le_bytes` per element). The iterators are
slices over those owned vecs. `tex_coords1()` returns `None` if the
`MeshTexCoord1` section is absent. `indices()` returns `Indices::U16`
or `Indices::U32` depending on which section kind is present.

Apps interleave attribute arrays into their own vertex type on upload.

**`TexAsset`:**

```rust
pub struct TexAsset { /* owns decompressed mip data */ }

impl TexAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError>;

    pub fn format(&self)      -> TexFormat;
    pub fn color_space(&self) -> ColorSpace;
    pub fn width(&self)       -> u32;
    pub fn height(&self)      -> u32;
    pub fn mip_count(&self)   -> u32;
    // ExactSizeIterator<Item = u8>
    pub fn mip(&self, level: u32) -> MipIter<'_>;
}
```

**`LoadError`:**

```rust
pub enum LoadError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u16),
    WrongAssetKind,
    SectionOutOfBounds,
    InvalidUtf8,
    DecompressError,
}
```

---

## xtask Changes

### New tasks

- `build-asset-compiler` — `cargo build -p asset-compiler`
- `compile-meshes-{app}` — depends on `build-asset-compiler`; reads
  `{app}/assets.toml` to determine which mesh assets the app needs,
  then for each: invokes `asset-compiler mesh src dst` where `dst` is
  `cache/meshes/{file}`; skips entries whose `dst` is already
  up-to-date relative to `src`. The cache dir is shared — if two apps
  list the same mesh, the second task finds it already compiled and
  skips it.
- `compile-images-{app}` — same pattern for image entries, outputting
  to `cache/images/`

Both tasks are added for each app (`samp-app`, `samp-app-noext`,
`phoenix`). Each `build-{app}` gains `compile-meshes-{app}` and
`compile-images-{app}` as dependencies (alongside the existing
`compile-shaders`). `copy-assets-{app}` also depends on both compile
tasks so assets are ready before copying.

### `copy_assets` changes

The `AssetType::Mesh` branch currently copies glTF sidecars using the
`gltf` crate. Replace with: copy `.pmesh` from `cache/meshes/` to the
output `assets/` directory (same pattern as shaders from
`cache/shaders/`). Similarly, `AssetType::Image` copies `.ptex` from
`cache/images/`.

Remove `gltf = "1"` from `xtask/Cargo.toml`.

---

## Phoenix Migration

1. Delete `load_duck()`
2. Add `asset-loader` dep, remove `gltf` dep from
   `phoenix/Cargo.toml`
3. `MeshAsset::open` replaces the tuple return
4. Texture: `tex_refs()` yields the logical name `"duck-tex"`;
   look it up in the asset map, open with `TexAsset::open`
5. Phoenix interleaves `positions()` and `tex_coords()` into its own
   `Vertex` type on upload. Normals and tangents are present in the
   file but unused until lighting is added.

---

## Open / Deferred

- **Handle/ownership model** — `MeshAsset` and `TexAsset` own their
  data. A future arena or slab allocator would require a different
  shape; current design does not foreclose it.
- **Multi-primitive meshes** — compiler takes first primitive only
- **Texture reference resolution** — mesh stores logical asset name;
  exact resolution path (through asset map at runtime) TBD
- **`MeshTexRef` and multiple textures per mesh** — currently one
  ref per mesh; multi-texture meshes deferred

---

## Implementation Order

| Step | What |
|------|------|
| 1 | Add three crates to workspace `Cargo.toml` |
| 2 | Implement `asset-shared` types |
| 3 | Implement `asset-compiler` mesh subcommand |
| 4 | Test compiler standalone: `cargo run -p asset-compiler mesh Duck.gltf Duck.pmesh` |
| 5 | Implement `asset-loader` `MeshAsset` |
| 6 | Update xtask: `compile-meshes-{app}` tasks, update `copy_assets` mesh branch |
| 7 | Update `assets/manifest.toml` duck entry |
| 8 | Migrate phoenix: replace `load_duck`, wire up `MeshAsset` |
| 9 | Implement `asset-compiler` image subcommand |
| 10 | Implement `asset-loader` `TexAsset` |
| 11 | Update xtask: `compile-images-{app}` tasks |
| 12 | Migrate phoenix: replace raw PNG load with `TexAsset` |
| 13 | `cargo clippy -p asset-shared -p asset-compiler -p asset-loader -p phoenix` |
