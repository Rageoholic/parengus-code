#!/usr/bin/env python3
"""Quick inspector for .pmesh binary files. No third-party deps required."""

import struct
import sys

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
    100: "TextureMip",
}

COMPRESSION = {0: "None", 1: "Lz4"}
ASSET_KIND   = {0: "Mesh", 1: "Texture"}

ELEM_SIZE = {
    0: 12,  # positions: 3×f32
    1: 12,  # normals:   3×f32
    2: 16,  # tangents:  4×f32
    3: 8,   # texcoord0: 2×f32
    4: 8,   # texcoord1: 2×f32
    5: 2,   # indices16: u16
    6: 4,   # indices32: u32
}


def lz4_block_decompress(blob: bytes) -> bytes:
    """
    Decompress an lz4_flex `compress_prepend_size` blob.
    Format: 4-byte LE uncompressed size + LZ4 block data.
    """
    if len(blob) < 4:
        raise ValueError("blob too short")
    uncompressed_size = int.from_bytes(blob[:4], "little")
    src = blob[4:]
    out = bytearray()
    pos = 0
    while pos < len(src):
        token = src[pos]; pos += 1
        # literal length
        lit_len = (token >> 4) & 0xF
        if lit_len == 15:
            while True:
                b = src[pos]; pos += 1
                lit_len += b
                if b != 255:
                    break
        # copy literals
        out.extend(src[pos:pos + lit_len]); pos += lit_len
        if pos >= len(src):
            break  # end of block
        # match offset (little-endian u16)
        match_offset = int.from_bytes(src[pos:pos + 2], "little"); pos += 2
        # match length
        match_len = (token & 0xF) + 4
        if match_len - 4 == 15:
            while True:
                b = src[pos]; pos += 1
                match_len += b
                if b != 255:
                    break
        # copy match
        start = len(out) - match_offset
        for i in range(match_len):
            out.append(out[start + i])
    return bytes(out[:uncompressed_size])


def parse(data: bytes) -> None:
    # FileHeader: magic(4) version(2) asset_kind(2) section_count(4) = 12B
    magic, version, kind_raw, section_count = struct.unpack_from(
        "<IHHI", data, 0
    )

    print(f"magic:         {magic:#010x}  "
          f"{'OK' if magic == PMSH_MAGIC else 'BAD - expected '
          f'{PMSH_MAGIC:#010x}'}")
    print(f"version:       {version}")
    print(f"asset_kind:    {ASSET_KIND.get(kind_raw, f'unknown({kind_raw})')}")
    print(f"section_count: {section_count}")
    print()

    sections = []
    hdr_offset = 12
    # SectionHeader: kind(4) comp(4) reserved(4) byte_offset(4)
    #                byte_len(4) compressed_byte_len(4) element_count(4) = 28B
    for _ in range(section_count):
        (kind_raw, comp_raw, _reserved,
         byte_offset, byte_len, comp_len,
         elem_count) = struct.unpack_from("<IIIIIII", data, hdr_offset)
        sections.append((kind_raw, comp_raw, byte_offset,
                         byte_len, comp_len, elem_count))
        hdr_offset += 28

    ok = True
    for i, (kind_raw, comp_raw, byte_offset,
            byte_len, comp_len, elem_count) in enumerate(sections):
        kind_name = SECTION_KINDS.get(kind_raw, f"unknown({kind_raw})")
        comp_name = COMPRESSION.get(comp_raw, f"unknown({comp_raw})")
        ratio = comp_len / byte_len if byte_len else 1.0
        print(f"[{i}] {kind_name}")
        print(f"     compression:   {comp_name}")
        print(f"     byte_offset:   {byte_offset}")
        print(f"     byte_len:      {byte_len} (uncompressed)")
        print(f"     comp_len:      {comp_len} ({ratio:.1%})")
        print(f"     element_count: {elem_count}")

        blob = data[byte_offset: byte_offset + comp_len]

        if comp_raw == 1:  # Lz4
            try:
                raw = lz4_block_decompress(blob)
            except Exception as e:
                print(f"     DECOMPRESS ERROR: {e}")
                ok = False
                print()
                continue
        else:
            raw = blob

        # Size sanity check
        expected_size = ELEM_SIZE.get(kind_raw)
        if expected_size is not None:
            expected_bytes = elem_count * expected_size
            size_ok = len(raw) == expected_bytes
            status = "OK" if size_ok else "SIZE MISMATCH"
            print(f"     decoded_bytes: {len(raw)} [{status}]")
            if not size_ok:
                print(f"       expected {expected_bytes}B "
                      f"({elem_count} × {expected_size}B)")
                ok = False

        # Per-kind extras
        if kind_raw == 7:  # MeshTexRef
            print(f"     tex_ref:       "
                  f"{raw.decode('utf-8', errors='replace')!r}")

        if kind_raw in (5, 6):  # indices
            elem_size = 2 if kind_raw == 5 else 4
            fmt = "<H" if kind_raw == 5 else "<I"
            n = min(8, elem_count)
            vals = [struct.unpack_from(fmt, raw, j * elem_size)[0]
                    for j in range(n)]
            suffix = "..." if elem_count > n else ""
            print(f"     first indices: {vals}{suffix}")

        if kind_raw in (0, 1, 2):  # positions / normals / tangents
            n_comps = 3 if kind_raw in (0, 1) else 4
            fmt = "<" + "f" * n_comps
            vals = struct.unpack_from(fmt, raw, 0)
            print(f"     first element: "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

        print()

    print("Result:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Duck.pmesh"
    parse(open(path, "rb").read())
