#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow", "numpy", "texture2ddecoder"]
# ///
"""Check that .ptex normal-map mip chains match their source images.

Usage:
  uv run tools/check_normal_mips.py
      checks all normal_map=true assets from assets/manifest.toml

  uv run tools/check_normal_mips.py cache/compiled/dh-normal.ptex ...
      checks specific .ptex files (source resolved via manifest.toml)

Exit code 0 if all assets pass, 1 otherwise.
"""
import struct
import sys
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image
import texture2ddecoder

# ── Constants matching asset-shared/src/lib.rs ────────────────────────────────

PTEX_MAGIC = int.from_bytes(b"PTEX", "little")

TEX_FORMAT   = {0: "Rgba8", 1: "Bc4", 2: "Bc5", 3: "Bc7"}
COLOR_SPACE  = {0: "Srgb", 1: "Linear"}

KIND_INFO = 200
KIND_MIP  = 100

# Flag a mip if mean absolute error exceeds this threshold.
# UASTC→BC5 transcoding gives mae ~0.003 on correct data;
# a systematic encoding bug (e.g. wrong channel) gives mae ~0.5+.
FAIL_THRESHOLD = 0.05

# ── Ptex parser ───────────────────────────────────────────────────────────────


def parse_ptex(path: Path):
    """Parse a .ptex file.

    Returns (tex_info dict, list[bytes]) where each bytes item is one
    mip's raw BC block data.
    """
    data = path.read_bytes()
    magic, version, section_count = struct.unpack_from("<IHI", data, 0)
    if magic != PTEX_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")

    tex_info = None
    mip_blobs: list[bytes] = []

    for i in range(section_count):
        off = 10 + i * 20
        (kind, byte_offset,
         byte_len, comp_len, _elem) = struct.unpack_from("<5I", data, off)
        blob = data[byte_offset: byte_offset + comp_len]

        if kind == KIND_INFO:
            fmt, cs, w, h, mip_count, compression = \
                struct.unpack_from("<6I", blob, 0)
            tex_info = {
                "format":      fmt,
                "color_space": cs,
                "width":       w,
                "height":      h,
                "mip_count":   mip_count,
                "compression": compression,
            }
        elif kind == KIND_MIP:
            mip_blobs.append(blob)

    if tex_info is None:
        raise ValueError("no TextureInfo section found")
    return tex_info, mip_blobs


# ── Mip generation (mirrors downsample_normal_mip in image.rs) ───────────────


def downsample_normal_mip(
    src: np.ndarray, dst_w: int, dst_h: int
) -> np.ndarray:
    """Box-filter a normal-map mip by averaging decoded XYZ vectors.

    Matches asset-compiler/src/image.rs:179-212 exactly:
    decode RG → XY, reconstruct Z = sqrt(1-X²-Y²), average, re-encode.
    Does NOT renormalize — shader reconstructs Z from stored XY.

    src: (h, w, 4) uint8 RGBA
    Returns: (dst_h, dst_w, 4) uint8 RGBA
    """
    h, w = src.shape[:2]
    scale_x = max(w // dst_w, 1)
    scale_y = max(h // dst_h, 1)

    # Reshape into blocks: (dst_h, scale_y, dst_w, scale_x, 4)
    blocks = src[: dst_h * scale_y, : dst_w * scale_x].reshape(
        dst_h, scale_y, dst_w, scale_x, 4
    )

    nx = blocks[..., 0].astype(np.float32) / 255.0 * 2.0 - 1.0
    ny = blocks[..., 1].astype(np.float32) / 255.0 * 2.0 - 1.0
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, None))

    ax = nx.mean(axis=(1, 3))
    ay = ny.mean(axis=(1, 3))
    az = nz.mean(axis=(1, 3))

    def enc(v: np.ndarray) -> np.ndarray:
        return np.clip(
            np.round((v + 1.0) * 127.5), 0, 255
        ).astype(np.uint8)

    return np.stack(
        [enc(ax), enc(ay), enc(az),
         np.full((dst_h, dst_w), 255, dtype=np.uint8)],
        axis=-1,
    )


def gen_mip_chain(src_rgba: np.ndarray) -> list[np.ndarray]:
    """Return the full mip chain starting from src_rgba."""
    mips = [src_rgba]
    while mips[-1].shape[0] > 1 or mips[-1].shape[1] > 1:
        prev = mips[-1]
        h, w = prev.shape[:2]
        mips.append(
            downsample_normal_mip(prev, max(w // 2, 1), max(h // 2, 1))
        )
    return mips


def mip_xy(mip: np.ndarray) -> np.ndarray:
    """Extract (h, w, 2) float32 XY normals in [-1, 1] from an RGBA mip."""
    return mip[:, :, :2].astype(np.float32) / 255.0 * 2.0 - 1.0


# ── Manifest ──────────────────────────────────────────────────────────────────


def load_normal_map_assets(manifest_path: Path) -> list[dict]:
    """Return [{name, source_path, ptex_path}] for normal_map assets."""
    with open(manifest_path, "rb") as f:
        manifest = tomllib.load(f)
    assets_dir = manifest_path.parent
    compiled   = Path("cache/compiled")
    result = []
    for asset in manifest.get("asset", []):
        if asset.get("normal_map") and asset.get("type") == "image":
            result.append({
                "name":        asset["name"],
                "source_path": assets_dir / asset["file"],
                "ptex_path":   compiled / f"{asset['name']}.ptex",
            })
    return result


# ── Per-asset check ───────────────────────────────────────────────────────────


def check(name: str, source_path: Path, ptex_path: Path) -> bool:
    """Check one normal map asset. Returns True if all checks pass."""
    if not ptex_path.exists():
        print(f"{name}: SKIP  {ptex_path} not found (run xtask bake first)")
        return True  # not a failure, just not built

    try:
        tex_info, mip_blobs = parse_ptex(ptex_path)
    except Exception as e:
        print(f"{name}: ERROR parsing ptex: {e}")
        return False

    w = tex_info["width"]
    h = tex_info["height"]
    fmt_name = TEX_FORMAT.get(tex_info["format"], str(tex_info["format"]))
    cs_name  = COLOR_SPACE.get(
        tex_info["color_space"], str(tex_info["color_space"])
    )

    errors = []
    if tex_info["color_space"] != 1:
        errors.append(f"color_space={cs_name} (expected Linear)")
    if tex_info["format"] != 2:
        errors.append(f"format={fmt_name} (expected Bc5)")

    if errors:
        print(f"{name}: FAIL  " + "; ".join(errors))
        return False

    try:
        src_img = Image.open(source_path).convert("RGBA")
    except Exception as e:
        print(f"{name}: ERROR loading source {source_path}: {e}")
        return False

    src_rgba = np.array(src_img, dtype=np.uint8)
    ref_mips = gen_mip_chain(src_rgba)

    n_mips = min(len(mip_blobs), len(ref_mips), tex_info["mip_count"])

    ok = True
    mip_stats: list[tuple[int, int, int, float, float]] = []

    for i in range(n_mips):
        mw = max(w >> i, 1)
        mh = max(h >> i, 1)

        try:
            rgba_bytes = texture2ddecoder.decode_bc5(
                mip_blobs[i], mw, mh
            )
        except Exception as e:
            print(f"{name}: ERROR decoding mip {i}: {e}")
            ok = False
            continue

        # texture2ddecoder returns BGRA; R=channel0(X), G=channel1(Y)
        # are at indices [2] and [1] respectively.
        decoded = (
            np.frombuffer(rgba_bytes, dtype=np.uint8)
            .reshape(mh, mw, 4)[:, :, [2, 1]]
            .astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        ref = mip_xy(ref_mips[i][:mh, :mw])

        diff    = np.abs(decoded - ref)
        max_err = float(diff.max())
        mae     = float(diff.mean())
        mip_stats.append((i, mw, mh, max_err, mae))

        if mae > FAIL_THRESHOLD:
            ok = False

    if not mip_stats:
        print(f"{name}: ERROR  no mips compared")
        return False

    overall_max = max(s[3] for s in mip_stats)
    overall_mae = max(s[4] for s in mip_stats)
    status      = "OK  " if ok else "FAIL"

    if ok:
        print(
            f"{name}: {status} "
            f"(mip0..mip{len(mip_stats)-1}, "
            f"max_err={overall_max:.4f}, mae={overall_mae:.4f})"
        )
    else:
        print(f"{name}: {status}")
        for i, mw, mh, max_err, mae in mip_stats:
            flag = " !" if mae > FAIL_THRESHOLD else "  "
            print(
                f"  mip{i} {mw}×{mh}{flag} "
                f"max_err={max_err:.4f}  mae={mae:.4f}"
            )

    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    manifest_path = Path("assets/manifest.toml")
    assets = load_normal_map_assets(manifest_path)

    if sys.argv[1:]:
        requested = {Path(a).resolve() for a in sys.argv[1:]}
        assets = [
            a for a in assets
            if a["ptex_path"].resolve() in requested
        ]
        if not assets:
            print("No matching normal map assets found in manifest.")
            sys.exit(1)

    if not assets:
        print("No normal_map assets found in manifest.")
        sys.exit(0)

    results = [
        check(a["name"], a["source_path"], a["ptex_path"])
        for a in assets
    ]
    sys.exit(0 if all(results) else 1)
