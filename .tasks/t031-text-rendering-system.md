---
id: t031
title: "Text rendering system (SSBO-indexed quads)"
status: planned
created: 2026-03-29
updated: 2026-03-29
parent: t029
children: []
depends_on:
  - t030
blocked_by:
  - t030
area: phoenix
---

## Context

With the font runtime from t030 (TTF + dynamic SDF atlas), the engine
needs a GPU rendering path that draws text glyphs efficiently. Text
shaping is done via rustybuzz, which converts strings to positioned
glyph IDs. The renderer draws geometry-shader-free quads via instance
ID indexing into an SSBO. Widget systems and interactivity are future
work (t033); this task is purely about getting shaped text onto the
screen.

The UI renders to a dedicated offscreen RGBA buffer (see t029);
this task owns that render target and the compositing pass that
blends it over the scene before present.

## Goal

Phoenix can render a string of text at a given screen position and
size, correctly shaped (Bidi, RTL, complex scripts) and drawn as
SDF quads via an SSBO-indexed pipeline.

## Plan

**Phase 1 — gather:** shape all text strings for the frame via
Bidi + rustybuzz; produce `(glyph_id, screen rect)` records.

**Phase 2 — atlas bake (async compute):** submit SDF rasterization
and texture uploads to the compute queue; signal a binary semaphore
on completion. Issue a queue family ownership release barrier on the
compute queue. The graphics queue waits on the semaphore and issues
the corresponding acquire barrier before the draw.

**Phase 3 — render:** write `UiInstance` array to the per-frame
SSBO and issue the draw call.

- [ ] Integrate `rustybuzz` into the gather pass: load
      `rustybuzz::Face` from the `Font` TTF bytes; shape strings
      into `(glyph_id, x_advance, y_advance, x_offset, y_offset,
      cluster)` runs
- [ ] Implement Unicode Bidi Algorithm pass before shaping to
      split mixed-direction strings into directional runs
- [ ] Implement CPU layout: accumulate per-run advances, apply
      RTL reversal; collect full `(glyph_id, screen rect)` list
- [ ] Define `UiInstance` struct (atlas UV rect, screen-space
      position + size, colour/tint)
- [ ] Implement atlas bake pass: call `Font::get_or_insert(glyph_id)`
      for all gathered glyph IDs; submit uploads to the compute
      queue; signal binary semaphore on completion
- [ ] Add queue family ownership transfer: release barrier on
      compute queue, acquire barrier on graphics queue; graphics
      queue waits on the semaphore at the UI draw step
- [ ] Write gathered instances to per-frame SSBO; issue draw
- [ ] Write vertex shader: `TRIANGLE_STRIP`, 4 vertices per
      instance; use `gl_VertexIndex` (0–3) for corner and
      `gl_InstanceIndex` to fetch position + UV from the SSBO
- [ ] Write fragment shader: sample SDF atlas, apply threshold +
      anti-aliasing; apply tint colour
- [ ] Set up pipeline (no vertex input, instance count = record
      count, alpha blend)
- [ ] Integrate into the phoenix render loop: UI pass renders to
      an offscreen RGBA buffer; a compositing pass alpha-blends
      that buffer over the scene image before present

## Thinking

### Vertex generation

Drawing quads with no vertex buffer: call `vkCmdDraw(4,
instance_count, 0, 0)` with `TRIANGLE_STRIP` topology. The vertex
shader uses `gl_VertexIndex` (already 0–3) as the corner index and
`gl_InstanceIndex` to fetch per-instance data from the SSBO.

### SDF rendering

The fragment shader samples the SDF atlas and compares the value to
a threshold (typically 0.5) with `smoothstep` for anti-aliasing.
Outline and shadow effects can be added later by varying the
threshold and adding a second pass.

### Per-frame pipeline

Text rendering is three phases per frame:

1. **Gather:** Walk all UI draw commands. Run Bidi + rustybuzz
   shaping on each text string to produce `(glyph_id, screen
   rect)` records for the whole frame. No GPU work yet.
2. **Atlas bake:** Iterate the gathered glyph IDs. For each miss,
   rasterize SDF on the CPU and upload the patch to the atlas
   texture. All uploads must complete before the draw call.
3. **Render:** Write `UiInstance` records (now with valid atlas
   UV rects from the bake pass) to the SSBO and issue the draw.

The separation is critical: interleaving atlas uploads with draw
calls would require pipeline barriers mid-pass or split render
passes. Batching all uploads in phase 2 keeps the render pass
clean.

### IM vs RM API

Not in scope for this task — t031 exposes a simple immediate-style
call (`draw_text(string, font, position, size, colour)`) that
reshapes and emits instances each frame. Widget lifecycle, retained
state, and event-driven invalidation (t032) are follow-up work.

Caching shaped output across frames is a future optimisation;
for the baseline, correctness matters more than per-frame shaping
cost.

### Font atlas texture registry

The font atlas descriptor array is a partially-bound array of
separate `VkImage`s — one slot per loaded font's atlas — consistent
with the mesh texture approach in t024. Each font can have a
different atlas size, which a single array texture would not allow.
`UiInstance.tex_idx` indexes into this array.

The array is sized at **64 slots** at pipeline-creation time. The
slot count is baked into the shader constant, so exceeding it
requires recreating the UI pipelines. When the engine needs a 65th
slot it destroys and recreates the UI descriptor layout and
pipelines with a larger count and logs a performance warning. 64 is
a generous starting point; a typical session loads at most a
handful of typefaces.

### Scope

Full Unicode support: any glyph in any loaded font is renderable
via on-demand atlas baking (t030). RTL, bidirectional text, and
complex-script shaping are all in scope via rustybuzz.

## Outcome

(not yet filled — task is planned)
