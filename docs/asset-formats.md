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

### `SectionHeader` (24 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | `kind` | `u32 LE`; `SectionKind` discriminant |
| 4 | 4 | `compression` | `u32 LE`; `Compression` discriminant |
| 8 | 4 | `byte_offset` | `u32 LE`; byte offset of section data in file |
| 12 | 4 | `byte_len` | `u32 LE`; uncompressed size in bytes |
| 16 | 4 | `compressed_byte_len` | `u32 LE`; compressed size; equals `byte_len` if `Compression::None` |
| 20 | 4 | `element_count` | `u32 LE`; number of elements (not bytes); see per-section notes |

No padding or reserved fields.

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

---

## `.pmesh` — Compiled Mesh

- Magic: `b"PMSH"` (`u32 LE: 0x48534d50`)
- `AssetKind::Mesh = 0`

### Section Kinds

| `SectionKind` | Value | `element_count` | Element size | `Compression` |
|---------------|-------|-----------------|--------------|---------------|
| `MeshPositions`  | 0 | vertex count | 12 B (`[f32; 3]`) | `Lz4` |
| `MeshNormals`    | 1 | vertex count | 12 B (`[f32; 3]`) | `Lz4` |
| `MeshTangents`   | 2 | vertex count | 16 B (`[f32; 4]`) | `Lz4` |
| `MeshTexCoord0`  | 3 | vertex count |  8 B (`[f32; 2]`) | `Lz4` |
| `MeshTexCoord1`  | 4 | vertex count |  8 B (`[f32; 2]`) | `Lz4` (optional) |
| `MeshIndices16`  | 5 | index count  |  2 B (`u16`) | `Lz4` |
| `MeshIndices32`  | 6 | index count  |  4 B (`u32`) | `Lz4` |
| `MeshTexRef`     | 7 | ref count    | 12 B         | `None` |

`MeshTexCoord1` is omitted if the source mesh has no `TEXCOORD_1`.
`MeshIndices32` is reserved for future use; current compiler always
uses `MeshIndices16`.

### `MeshTexRef` Element Layout (12 bytes each)

```
role:     u32 LE   (TexRole discriminant)
asset_id: u64 LE   (FNV-1a 64-bit hash of texture asset name)
```

`TexRole` values: `Albedo = 0`, `Normal = 1`,
`MetallicRoughness = 2`, `Emissive = 3`, `Occlusion = 4`.

Texture references come from the manifest `[asset.tex_refs]` table,
not from the glTF URI. The compiler validates that each named asset
exists in the manifest as an image type.

### Coordinate Space

Y-up (glTF) → Z-up transform applied at compile time:
`(x, y, z) → (x, z, −y)`. No transform is applied at load time.

---

## `.ptex` — Compiled Texture

- Magic: `b"PTEX"` (`u32 LE: 0x58455450`)
- `AssetKind::Texture = 1`

### Section Kinds

| `SectionKind` | Value | Contents | `element_count` | `Compression` |
|---------------|-------|----------|-----------------|---------------|
| `TextureInfo` | 200 | 20-byte metadata | 1 | `None` |
| `TextureMip`  | 100 | raw mip pixel/block data | byte count of mip | varies |

`TextureInfo` **must** be the first section. `TextureMip` sections
follow in mip order (mip 0 first). `element_count` for `TextureMip`
equals the number of bytes of mip data (so `byte_len ==
element_count`).

### `TextureInfo` Body (20 bytes)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | `format` | `u32 LE`; `TexFormat` discriminant |
| 4 | 4 | `color_space` | `u32 LE`; `ColorSpace` discriminant |
| 8 | 4 | `width` | `u32 LE`; pixels at mip 0 |
| 12 | 4 | `height` | `u32 LE`; pixels at mip 0 |
| 16 | 4 | `mip_count` | `u32 LE` |

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
