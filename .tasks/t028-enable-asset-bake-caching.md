id: t028
title: Enable asset bake caching
status: done
area: pipeline
parent: t002
assignee: TBD
---

## Summary

Avoid re-running expensive image/mesh compression on unchanged source files by adding a cache layer to the xtask asset bake. Cache baked outputs under `cache/compiled/` (or configurable path) and skip re-baking when inputs and relevant options are unchanged.

## Motivation

Basis UASTC→BC7 transcodes and mesh/texture processing can be CPU-intensive and slow developer iteration. A simple cache keyed by input file mtime + manifest entry + compiler args will speed subsequent builds significantly.

## Scope

- Add cache lookup in xtask/asset-bake driver before invoking per-asset compilers.
- Store baked artifacts in `cache/compiled/<asset-name>.<ext>` and record metadata (e.g., input checksum, mtime, manifest hash) alongside the artifact.
- Implement invalidation: re-bake when source file mtime or manifest entry (format, color_space, mips) changes, or when xtask is run with `--force`.
- Provide a `xtask clean-cache` subcommand to purge cache.

Out of scope for this task: distributed cache, incremental chunking, or content-addressed storage (can be future work).

## Acceptance criteria

- Running `cargo xtask build-phoenix` reuses previously baked assets when inputs/options unchanged.
- `cargo xtask build-phoenix --force` rebuilds all assets regardless of cache.
- Cache metadata is stored under `cache/compiled/` and is human-inspectable.

## Plan

1. Add a small cache util module under `xtask/src/cache.rs` that exposes `lookup(asset_name, src_path, manifest_entry) -> Option<PathBuf>` and `store(asset_name, dst_path, meta)`.
2. Wire cache lookup into `xtask` build flow before calling `asset-compiler::compile` / `asset-compiler::image::compile` / etc.
3. Implement `--force` flag passthrough to skip cache lookup.
4. Add `xtask clean-cache` command to remove `cache/compiled/` contents.
5. Add tests for cache lookup behavior (unit tests for cache util).
6. Document behavior in `.tasks/t028-enable-asset-bake-caching.md` and `README.md` (xtask usage).

## Notes

- Use a simple metadata file (TOML or JSON) with keys: input_paths, input_mtimes, manifest_hash, compiler_args_hash, timestamp.
- Use file mtimes for speed; add optional SHA256 checks when `--verify` is provided.

