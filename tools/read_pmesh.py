#!/usr/bin/env python3
"""Dump and validate .pmesh binary files. No third-party deps required."""
import struct
import sys

# ── Constants matching asset-shared/src/lib.rs ───────────────────────────────

PMSH_MAGIC = int.from_bytes(b"PMSH", "little")

SECTION_KINDS = {
    0: "MeshPositions",
    1: "MeshNormals",
    2: "MeshTangents",
    3: "MeshTexCoord0",
    4: "MeshTexCoord1",
    5: "MeshIndices16",
    6: "MeshIndices32",
    7: "MeshTexRef",
    8: "MeshSubMeshTable",
    9: "MeshMaterialData",
    10: "MeshSubMeshAlbedo",
    100: "TextureMip",
    200: "TextureInfo",
}

# Expected bytes per element for sanity checks (None = variable/skipped).
ELEM_SIZE = {
    0: 12,  # MeshPositions:    3×f32
    1: 12,  # MeshNormals:      3×f32
    2: 16,  # MeshTangents:     4×f32
    3: 8,   # MeshTexCoord0:    2×f32
    4: 8,   # MeshTexCoord1:    2×f32
    5: 2,   # MeshIndices16:    u16
    6: 4,   # MeshIndices32:    u32
    8: 48,  # MeshSubMeshTable: 3f+4f+3f+u32+u32
    10: 8,  # MeshSubMeshAlbedo: u64
}

TEX_ROLES = {
    0: "Albedo", 1: "Normal", 2: "MetallicRoughness",
    3: "Emissive", 4: "Occlusion",
}

# ── FNV-1a hash (matches asset-shared/src/lib.rs) ────────────────────────────

FNV_OFFSET = 14695981039346656037
FNV_PRIME  = 1099511628211


def fnv1a(s: str) -> int:
    h = FNV_OFFSET
    for b in s.encode():
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


KNOWN_ASSETS = [
    "duck-albedo",
    "dh-albedo", "dh-normal", "dh-emissive", "dh-ao", "dh-metal-rough",
    "fh-rubber-albedo", "fh-glass-albedo", "fh-metal-albedo",
    "fh-leather-albedo", "fh-lenses-albedo",
    "fh-rubber-normal", "fh-glass-normal", "fh-metal-normal",
    "fh-leather-normal", "fh-lenses-normal",
    "fh-rubber-orm", "fh-glass-orm", "fh-metal-orm",
    "fh-leather-orm", "fh-lenses-orm",
    "statue-tex",
]
HASH_TO_NAME = {fnv1a(n): n for n in KNOWN_ASSETS}

# ── Pure-Python LZ4 block decompressor ───────────────────────────────────────
# Handles lz4_flex::compress_prepend_size format:
# 4-byte LE uncompressed size followed by LZ4 block data.


def lz4_decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        raise ValueError("blob too short")
    uncompressed_size = int.from_bytes(blob[:4], "little")
    src = blob[4:]
    out = bytearray()
    pos = 0
    while pos < len(src):
        token = src[pos]; pos += 1
        lit_len = (token >> 4) & 0xF
        if lit_len == 15:
            while True:
                b = src[pos]; pos += 1
                lit_len += b
                if b != 255:
                    break
        out.extend(src[pos:pos + lit_len]); pos += lit_len
        if pos >= len(src):
            break  # end of block
        match_offset = int.from_bytes(src[pos:pos + 2], "little"); pos += 2
        match_len = (token & 0xF) + 4
        if match_len - 4 == 15:
            while True:
                b = src[pos]; pos += 1
                match_len += b
                if b != 255:
                    break
        start = len(out) - match_offset
        for i in range(match_len):
            out.append(out[start + i])
    return bytes(out[:uncompressed_size])

# ── Per-section decoders ──────────────────────────────────────────────────────


def decode_submesh_table(data: bytes, count: int) -> None:
    # SubMeshInfo: t[3f] r[4f] s[3f] index_base[u32] index_count[u32] = 48B
    for i in range(count):
        off = i * 48
        t       = struct.unpack_from("<3f", data, off)
        r       = struct.unpack_from("<4f", data, off + 12)
        s       = struct.unpack_from("<3f", data, off + 28)
        ib, ic  = struct.unpack_from("<2I", data, off + 40)
        print(f"  sub[{i}]: t={t} r={r} s={s} "
              f"index_base={ib} index_count={ic}")


def decode_texref(data: bytes, count: int) -> None:
    # TexRef: role(u32) + hash(u64) = 12 bytes
    for i in range(count):
        off      = i * 12
        role_raw, = struct.unpack_from("<I", data, off)
        hash_val, = struct.unpack_from("<Q", data, off + 4)
        role = TEX_ROLES.get(role_raw, f"unknown({role_raw})")
        name = HASH_TO_NAME.get(hash_val, f"0x{hash_val:016x}")
        print(f"  texref[{i}]: role={role} -> {name}")


def decode_subalbedo(data: bytes, count: int) -> None:
    for i in range(count):
        hash_val, = struct.unpack_from("<Q", data, i * 8)
        name = HASH_TO_NAME.get(hash_val, f"0x{hash_val:016x}")
        print(f"  albedo[{i}]: {name}")


def dump_first_elems(kind_raw: int, data: bytes, count: int) -> None:
    if kind_raw in (0, 1):  # positions / normals: 3×f32
        vals = struct.unpack_from("<3f", data, 0)
        print(f"  first: ({', '.join(f'{v:.4f}' for v in vals)})")
    elif kind_raw == 2:  # tangents: 4×f32
        vals = struct.unpack_from("<4f", data, 0)
        print(f"  first: ({', '.join(f'{v:.4f}' for v in vals)})")
    elif kind_raw in (5, 6):  # indices
        elem_size = 2 if kind_raw == 5 else 4
        fmt = "<H" if kind_raw == 5 else "<I"
        n = min(8, count)
        vals = [struct.unpack_from(fmt, data, j * elem_size)[0]
                for j in range(n)]
        suffix = "..." if count > n else ""
        print(f"  first indices: {vals}{suffix}")

# ── Top-level parser ──────────────────────────────────────────────────────────


def dump(path: str) -> bool:
    """Parse and print one .pmesh file. Returns True if all checks pass."""
    print(f"\n=== {path} ===")
    with open(path, "rb") as f:
        data = f.read()

    # FileHeader: magic(u32) version(u16) section_count(u32) = 10 bytes
    magic, version, section_count = struct.unpack_from("<IHI", data, 0)
    magic_ok = magic == PMSH_MAGIC
    print(f"magic={data[0:4].decode('latin1')!r} "
          f"({'OK' if magic_ok else 'BAD'})  "
          f"version={version}  sections={section_count}")

    ok = magic_ok
    # SectionHeader: kind(u32) byte_offset(u32) byte_len(u32)
    #                compressed_byte_len(u32) element_count(u32) = 20 bytes
    for i in range(section_count):
        off = 10 + i * 20
        (kind_raw, byte_offset, byte_len,
         comp_len, elem_count) = struct.unpack_from("<5I", data, off)
        kind_name = SECTION_KINDS.get(kind_raw, f"unknown({kind_raw})")
        is_compressed = comp_len != byte_len
        comp_label = (f"Lz4 ({comp_len / byte_len:.1%})"
                      if is_compressed else "None")
        print(f"[{i}] {kind_name}  elems={elem_count}  "
              f"bytes={byte_len}  comp={comp_label}")

        blob = data[byte_offset: byte_offset + comp_len]
        if is_compressed:
            try:
                raw = lz4_decompress(blob)
            except Exception as e:
                print(f"  DECOMPRESS ERROR: {e}")
                ok = False
                continue
        else:
            raw = blob

        # Size sanity check
        stride = ELEM_SIZE.get(kind_raw)
        if stride is not None:
            expected = elem_count * stride
            if len(raw) != expected:
                print(f"  SIZE MISMATCH: got {len(raw)}B, "
                      f"expected {expected}B ({elem_count}×{stride})")
                ok = False

        # Per-kind decode
        if kind_raw == 7:    # MeshTexRef
            decode_texref(raw, elem_count)
        elif kind_raw == 8:  # MeshSubMeshTable
            decode_submesh_table(raw, elem_count)
        elif kind_raw == 10:  # MeshSubMeshAlbedo
            decode_subalbedo(raw, elem_count)
        else:
            if elem_count > 0:
                dump_first_elems(kind_raw, raw, elem_count)

    print("Result:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "cache/compiled/duck.pmesh",
        "cache/compiled/damaged-helmet.pmesh",
        "cache/compiled/flight-helmet.pmesh",
    ]
    sys.exit(0 if all(dump(p) for p in paths) else 1)
