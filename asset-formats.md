# Asset Binary Formats

> AI-assisted document — generated with Claude Code.

Reference for all compiled asset file formats produced by
`asset-compiler` and consumed by `asset-loader`.

---

## Shared Structures

### `FileHeader` (10 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | `magic` | `u32 LE`; `b"PMSH"` for mesh, `b"PTEX"` for texture |
| 4 | 2 | `version` | `u16 LE`; currently `1` |
| 6 | 4 | `section_count` | `u32 LE` |

The magic bytes uniquely identify the asset type; no separate
`asset_kind` field is needed.

All fields are read with explicit `from_le_bytes` — no `Pod` cast,
no endianness assumption.

### `SectionHeader` (20 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | `kind` | `u32 LE`; `SectionKind` discriminant |
| 4 | 4 | `byte_offset` | `u32 LE`; byte offset of section data in file |
| 8 | 4 | `byte_len` | `u32 LE`; uncompressed size in bytes |
| 12 | 4 | `compressed_byte_len` | `u32 LE`; on-disk size; equals `byte_len` when uncompressed |
| 16 | 4 | `element_count` | `u32 LE`; number of elements (not bytes); see per-section notes |

No padding or reserved fields. Compression is not stored in the
section header; it is determined by section kind (mesh) or by the
`TextureInfo` body (texture mips).

### `Compression` enum

| Value | Variant | Notes |
|-------|---------|-------|
| 0 | `None` | Section data stored as-is |
| 1 | `Lz4` | LZ4 **frame** format (`FrameEncoder`/`FrameDecoder`); not block format |

### `AssetId<T>`

`AssetId<T>` is the FNV-1a 64-bit hash of the UTF-8 asset name
string. It is phantom-typed by `T ∈ {Mesh, Texture, Shader}`:

```
FNV-1a 64-bit:
  offset_basis = 14695981039346656037u64
  prime        = 1099511628211u64
  for each byte b: hash = (hash ^ b as u64) * prime
```

Type aliases: `MeshId`, `TextureId`, `ShaderId`. Const helpers:
`mesh_id("name")`, `texture_id("name")`, `shader_id("name")`.

Collision check: xtask hashes all names per type domain at build
time and errors on any collision within a domain.

No checksums are stored in any asset file format. Integrity
verification (if needed) is the responsibility of the build pipeline
or packaging layer above the compiler.

---

## `.pmesh` — Compiled Mesh

- Magic: `b"PMSH"` (`u32 LE: 0x48534d50`)
- `AssetKind::Mesh = 0`

### File structure

A `.pmesh` file always begins with a `MeshSubMeshTable` section
(index 0 in the section header array). Geometry sections follow,
grouped by sub-mesh. Each sub-mesh entry in the table records which
geometry sections belong to it via a `(first_idx, section_count)`
pair that indexes into the geometry sections only (0-based; i.e.
geometry index 0 = section header index 1).

### Section Kinds

Compression for each mesh section kind is fixed and implicit — it
is not stored in the section header.

| `SectionKind`      | Value | `element_count` | Element size       | Compression |
|--------------------|-------|-----------------|--------------------|-------------|
| `MeshPositions`    | 0     | vertex count    | 12 B (`[f32; 3]`) | `Lz4`       |
| `MeshNormals`      | 1     | vertex count    | 12 B (`[f32; 3]`) | `Lz4`       |
| `MeshTangents`     | 2     | vertex count    | 16 B (`[f32; 4]`) | `Lz4`       |
| `MeshTexCoord0`    | 3     | vertex count    |  8 B (`[f32; 2]`) | `Lz4`       |
| `MeshTexCoord1`    | 4     | vertex count    |  8 B (`[f32; 2]`) | `Lz4` (optional) |
| `MeshIndices16`    | 5     | index count     |  2 B (`u16`)      | `Lz4`       |
| `MeshIndices32`    | 6     | index count     |  4 B (`u32`)      | `Lz4`       |
| `MeshTexRef`       | 7     | ref count       | 12 B              | `None`      |
| `MeshSubMeshTable` | 8     | sub-mesh count  | 48 B              | `None`      |
| `MeshMaterialData` | 9     | 1 (always)      | 56 B              | `None`      |

`MeshTexCoord1` is omitted if the source mesh has no `TEXCOORD_1`.
`MeshIndices16` is used when all indices fit in a `u16` (max index ≤
65535); `MeshIndices32` is used otherwise. A file will contain
exactly one of the two index section kinds.

`MeshSubMeshTable` **must** be the first section (section header
index 0). `MeshMaterialData` appears once per sub-mesh, after that
sub-mesh's other geometry sections.

### `MeshSubMeshTable` Entry Layout (48 bytes each)

The section data is `S × 48` bytes where `S` is `element_count`.

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0  | 12 | `translation`   | `[f32; 3] LE` — Z-up engine space |
| 12 | 16 | `rotation`      | `[f32; 4] LE` — quaternion xyzw, Z-up |
| 28 | 12 | `scale`         | `[f32; 3] LE` — local scale |
| 40 |  4 | `first_idx`     | `u32 LE` — 0-based index into geometry sections |
| 44 |  4 | `section_count` | `u32 LE` — geometry section count for this sub-mesh |

The stored TRS is the node's world transform converted from glTF
Y-up to engine Z-up space. The runtime applies `T * R * S * vertex`
to position each sub-mesh in the scene. For meshes with identity
node transforms (e.g. FlightHelmet) the TRS is
`t=[0,0,0], r=[0,0,0,1], s=[1,1,1]`.

### `MeshMaterialData` Body (56 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0  | 16 | `base_color_factor`  | `[f32; 4] LE` — RGBA |
| 16 | 12 | `emissive_factor`    | `[f32; 3] LE` |
| 28 |  4 | `metallic_factor`    | `f32 LE` |
| 32 |  4 | `roughness_factor`   | `f32 LE` |
| 36 |  4 | `normal_scale`       | `f32 LE` — `normalTexture.scale` |
| 40 |  4 | `occlusion_strength` | `f32 LE` — `occlusionTexture.strength` |
| 44 |  4 | `alpha_mode`         | `u32 LE` — `0`=Opaque, `1`=Mask, `2`=Blend |
| 48 |  4 | `alpha_cutoff`       | `f32 LE` — used when `alpha_mode == 1` |
| 52 |  4 | `double_sided`       | `u32 LE` — `0`=false, `1`=true |

Default values (when the glTF material omits a field) match the
glTF 2.0 spec: `base_color_factor = [1,1,1,1]`,
`emissive_factor = [0,0,0]`, `metallic_factor = 1.0`,
`roughness_factor = 1.0`, `normal_scale = 1.0`,
`occlusion_strength = 1.0`, `alpha_mode = 0`,
`alpha_cutoff = 0.5`, `double_sided = 0`.

### `MeshTexRef` Element Layout (12 bytes each)

```
role:     u32 LE   (TexRole discriminant)
asset_id: u64 LE   (FNV-1a 64-bit hash of texture asset name)
```

`TexRole` values: `Albedo = 0`, `Normal = 1`,
`MetallicRoughness = 2`, `Emissive = 3`, `Occlusion = 4`.

Texture references are auto-derived by the compiler from the glTF
material's texture slots. For each slot that is present, the
compiler finds the manifest image entry whose `file` path matches
the glTF image URI, then hashes that entry's logical `name` to
produce the `TextureId`. The `file` path is used only for lookup;
the stored ID is always derived from `name`. It is an error if no
manifest entry matches a referenced URI.

When `metallicRoughnessTexture` and `occlusionTexture` reference
the same image (as in FlightHelmet's ORM textures), two separate
`MeshTexRef` entries are emitted — one with role `MetallicRoughness`
and one with `Occlusion` — both pointing to the same `TextureId`.

What happens at runtime when a mesh references a texture that cannot
be found is engine-defined behaviour; the asset format itself imposes
no constraint.

### Coordinate Space

glTF Y-up → Z-up transform applied at compile time to all vertex
data (positions, normals, tangent xyz): `(x, y, z) → (x, -z, y)`.
Vertex positions are in **local Z-up space** — the node world
transform is NOT baked into vertices. Instead, each sub-mesh's
world transform is stored as TRS in the `MeshSubMeshTable` section
and applied by the runtime.

No coordinate transform is applied at load time.

---

## `.ptex` — Compiled Texture

- Magic: `b"PTEX"` (`u32 LE: 0x58455450`)
- `AssetKind::Texture = 1`

### Section Kinds

| `SectionKind` | Value | Contents | `element_count` |
|---------------|-------|----------|-----------------|
| `TextureInfo` | 200 | 24-byte metadata | 1 |
| `TextureMip`  | 100 | raw mip pixel/block data | byte count of mip |

`TextureInfo` **must** be the first section. `TextureMip` sections
follow in mip order (mip 0 first). `element_count` for `TextureMip`
is in bytes (so `byte_len == element_count`).

### `TextureInfo` Body (24 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | `format` | `u32 LE`; `TexFormat` discriminant |
| 4 | 4 | `color_space` | `u32 LE`; `ColorSpace` discriminant |
| 8 | 4 | `width` | `u32 LE`; pixels at mip 0 |
| 12 | 4 | `height` | `u32 LE`; pixels at mip 0 |
| 16 | 4 | `mip_count` | `u32 LE` |
| 20 | 4 | `compression` | `u32 LE`; `Compression` discriminant; applies to all `TextureMip` sections |

`TexFormat` values: `Rgba8 = 0`, `Bc4 = 1`, `Bc5 = 2`, `Bc7 = 3`.

`ColorSpace` values: `Srgb = 0`, `Linear = 1`.

### Compression per Format

| `TexFormat` | `TextureMip` compression |
|-------------|--------------------------|
| `Rgba8` | `Lz4` (frame format) |
| `Bc4` | `None` (BCn already compressed) |
| `Bc5` | `None` |
| `Bc7` | `None` |

### BCn Block Sizes

| Format | Block size | Block footprint |
|--------|------------|-----------------|
| BC4 | 8 B | 4×4 px |
| BC5 | 16 B | 4×4 px |
| BC7 | 16 B | 4×4 px |

### Dimension Requirement

Texture `width` and `height` must both be powers of two. The
compiler validates this after decoding the source image and returns
an error if not satisfied.
