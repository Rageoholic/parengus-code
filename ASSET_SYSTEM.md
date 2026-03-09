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
| `attribs` | `u32` | `VertexAttribs` bitmask; only meaningful for `MeshVertices` sections |
| `byte_offset` | `u32` | offset into file |
| `byte_len` | `u32` | uncompressed size; use for allocation |
| `compressed_byte_len` | `u32` | compressed size; use for seeking. Equal to `byte_len` if uncompressed |
| `element_count` | `u32` | number of elements (not bytes); backs `ExactSizeIterator::len()` |

**`Compression` enum:**

| Variant | Value | Notes |
|---------|-------|-------|
| `None` | 0 | section data stored as-is |
| `Lz4` | 1 | LZ4 block format |

**Section kinds for mesh:**

| `SectionKind` | Contents | Default compression |
|---------------|----------|---------------------|
| `MeshVertices` | `[MeshVertex]` — typed, little-endian | `Lz4` |
| `MeshIndices` | `[u16]` — little-endian | `Lz4` |
| `MeshTexRef` | logical asset name (UTF-8, no null terminator) | `None` |

**`VertexAttribs` bitmask** (stored in `SectionHeader::attribs`):

| Bit | Attribute | Notes |
|-----|-----------|-------|
| 0 | `TEX_COORD_1` | UV1; optional |

`position`, `normal`, `tangent`, and `tex_coord` (UV0) are always
present and not represented in the bitmask. The bitmask is reserved
for future optional attributes.

**In-memory vertex type:**

```rust
pub struct MeshVertex {
    pub position:   [f32; 3],         // Z-up
    pub normal:     [f32; 3],         // Z-up; generated if absent
                                      // in source
    pub tangent:    [f32; 4],         // xyz + w handedness sign;
                                      // bitangent =
                                      //   cross(normal, tangent.xyz)
                                      //   * tangent.w
                                      // generated via Mikktspace if
                                      // absent in source
    pub tex_coord:  [f32; 2],         // UV0
    pub tex_coord1: Option<[f32; 2]>, // UV1; absent if not in source
}
```

The compiler always generates `normal` and `tangent` if absent in
the glTF primitive (averaged face normals; Mikktspace tangents).
`tex_coord1` is written only if `TEXCOORD_1` is present in the
primitive; the `TEX_COORD_1` bit in `attribs` signals its presence
to the loader.

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
- `VertexAttribs` bitmask type
- `FileHeader` and `SectionHeader` structs
- `MeshVertex` struct
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
4. Read `NORMAL` if present (transform to Z-up); otherwise generate
   averaged face normals
5. Read `TANGENT` if present (transform to Z-up); otherwise generate
   via Mikktspace
6. Read `TEXCOORD_0`
7. Read `TEXCOORD_1` if present; set `TEX_COORD_1` bit in `attribs`
8. Zip into `Vec<MeshVertex>`
9. Read indices → `Vec<u16>` (fail if any index exceeds `u16::MAX`)
10. Write texture logical name as `MeshTexRef` section (from manifest,
    not from the glTF URI)
11. LZ4-compress vertex and index sections
12. Write `FileHeader`, `[SectionHeader; N]` (with `attribs` set on
    the `MeshVertices` header), compressed section data

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

pub use mesh::{MeshAsset, VertexIter, IndexIter, TexRefIter};
pub use tex::TexAsset;
```

**`MeshAsset`:**

```rust
pub struct MeshAsset { /* owns Vec<u8> of file bytes */ }

impl MeshAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError>;

    pub fn vertices(&self) -> VertexIter<'_>;
    pub fn indices(&self)  -> IndexIter<'_>;
    pub fn tex_refs(&self) -> TexRefIter<'_>;
}
```

All iterators implement `ExactSizeIterator` and yield typed values
(`MeshVertex`, `u16`). `iter.len()` drives staging buffer allocation:

```rust
let mesh = MeshAsset::open(&path)?;
let verts = mesh.vertices();
let staging_size = verts.len() * size_of::<MeshVertex>();
```

Vertex and index sections are LZ4-decompressed on `open()` into
`Vec<MeshVertex>` / `Vec<u16>` (with `from_le_bytes` per field).
The iterators are slices over those owned vecs.

**`TexAsset`:**

```rust
pub struct TexAsset { /* owns decompressed mip data */ }

impl TexAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError>;

    pub fn format(&self)     -> TexFormat;
    pub fn color_space(&self) -> ColorSpace;
    pub fn width(&self)      -> u32;
    pub fn height(&self)     -> u32;
    pub fn mip_count(&self)  -> u32;
    pub fn mip(&self, level: u32) -> MipIter<'_>; // ExactSizeIterator<Item = u8>
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
5. Phoenix converts `MeshVertex` to its own `Vertex` type. For now
   it uses only `position` and `tex_coord`; normal and tangent are
   present but unused until lighting is added.

---

## Open / Deferred

- **Handle/ownership model** — `MeshAsset` and `TexAsset` own their
  data. A future arena or slab allocator would require a different
  shape; current design does not foreclose it.
- **Multi-primitive meshes** — compiler takes first primitive only
- **Index width** — `u16` now; widening path is
  `SectionKind::MeshIndices32` + version bump
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
