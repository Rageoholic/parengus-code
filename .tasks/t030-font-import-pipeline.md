---
id: t030
title: "Font import pipeline (TTF + runtime SDF atlas)"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: t029
children: []
depends_on: []
blocked_by: []
area: pipeline
---

## Context

The UI system (t029) requires font data at runtime for both shaping
(rustybuzz) and glyph rasterization (SDF). The TTF file must be
shipped as a runtime asset. Glyphs are baked into an SDF atlas
on-demand at runtime as new glyph IDs are encountered, so there is
no offline coverage decision and no CJK scalability cliff.

## Goal

The asset pipeline copies TTF files into the output as first-class
runtime assets. At runtime, `asset-loader` exposes a `Font` type
that:
1. Holds the raw TTF bytes for rustybuzz shaping.
2. Manages a dynamic GPU texture atlas, rasterizing SDF patches
   for new glyph IDs on demand and packing them into the atlas.
3. Provides `(atlas UV rect, metrics)` lookup by `(font_id,
   glyph_id)` for the renderer.

## Plan

- [ ] Add font entries to `assets/manifest.toml` schema; copy TTF
      to output directory as a runtime asset (no compilation step)
- [ ] Choose SDF rasterizer for runtime use (e.g. `fontdue` which
      does rasterization natively, or `ttf-parser` + manual SDF
      via edt/distance transform)
- [ ] Implement atlas allocator (shelf or guillotine packing) for
      placing glyph SDF patches into a growable GPU texture
- [ ] Expose `Font` type in `asset-loader`: holds TTF bytes +
      `rustybuzz::Face` + atlas state
- [ ] Implement `Font::get_or_insert(glyph_id)` → UV rect: check
      cache, rasterize + upload if miss
- [ ] Expose atlas texture handle to the renderer (t031)

## Thinking

SDF is preferred over raw bitmaps for scale independence.

The atlas is keyed on `glyph_id` (u16, font-internal), not Unicode
codepoint. rustybuzz shaping (in t031) produces glyph IDs; those
are what hit the atlas. This correctly handles ligatures and
contextual forms where one codepoint → multiple glyph IDs or
multiple codepoints → one glyph ID.

On-demand baking means:
- No manifest coverage decision: every glyph in any font is
  reachable, including full CJK if a CJK font is loaded.
- First render of a novel glyph ID pays a rasterization cost;
  subsequent frames are cache hits. This is acceptable for a game
  where UI text changes infrequently mid-frame.
- The atlas may need to grow or page if many distinct glyphs are
  encountered. A simple approach: start with a 1K×1K texture,
  double on overflow (or add a second page).

The TTF must be available at runtime because rustybuzz reads GSUB,
GPOS, kern, hmtx, and cmap tables live during shaping. Stripping
these tables to save space would break shaping.

## Outcome

(not yet filled — task is planned)
