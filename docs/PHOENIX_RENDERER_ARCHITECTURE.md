# Renderer Architecture

> Written with Claude's assistance.

Phoenix uses three distinct renderer contexts. Each is a self-contained
rendering pipeline — data management architecture plus shading pipeline —
suited to its use case. The mission and garage renderers share a common
**shader library** (PBR BRDF, material SSBO layout, etc.) but have different
frame production architectures; the shared shaders are an implementation
detail, not a sign of shared architecture. The loading screen renderer is
not a 3D renderer at all — it clears the backbuffer and hands control
entirely to the UI system.

The UI system (see below) is renderer-agnostic and layers on top of all three.

---

## The Three Renderers

### Mission renderer

The primary in-game renderer. Renders the full 3D scene: world geometry,
characters, vehicles, effects. Targets an offscreen HDR colour buffer; the UI
composites over it before the final present.

All scene assets are pre-loaded during the loading screen before the mission
renderer issues its first frame. Once the mission begins, the global vertex
buffer, index buffer, and texture table are never modified. This
static-after-load invariant enables a straightforward migration to
`vkCmdDrawIndexedIndirectCount` with GPU-side culling: the draw list can be
built or filtered on the GPU without the CPU needing to update any resource
bindings. The draw indirect buffer itself is allocated at mission load time
sized to the maximum draw count and is not resized mid-mission.

### Garage renderer

The mech customisation and loadout renderer. Renders a real in-game
environment (e.g. the VTOL mech bay) with dynamic lighting and the current
mech configuration. Part previews update as the player browses loadout
options. Everything beyond the 3D scene (menus, part lists, stats) lives in
the UI layer.

**Streaming asset cache.** Unlike the mission renderer, the garage cannot
pre-load all parts at startup — the full part catalogue is too large. Instead it
maintains a per-part mesh and texture cache:

- On a cache miss, an async file I/O request is issued to load the part's
  compiled mesh and textures.
- GPU upload begins as soon as the data is available. A fence marks upload
  completion.
- Until the fence signals, that part's mesh slot is skipped in the draw list —
  no placeholder is drawn.
- Once the fence signals, the slot is live and drawn normally on the next frame.

The vertex/index buffer and texture table are therefore mutable at runtime,
unlike the mission renderer. The draw list is rebuilt CPU-side each frame based
on which cache slots are live.

**Vertex/index buffer.** The mission buffer is freed at garage entry
(see Memory pressure section), and the garage allocates its own
vertex/index buffer sized to its current streaming cache. This lets the
garage manage its buffer lifecycle independently — including shrinking
under memory pressure — without affecting mission data.

**Lighting.** Supports dynamic light changes (e.g. per-part preview lighting
rigs). The PBR shader is shared with the mission renderer via the common shader
library; only the light uniforms differ.

### Loading screen renderer

Not a 3D renderer. Its only job is to clear the backbuffer and then
let the UI system render directly to the swapchain image. There is
no scene pass, no offscreen target, and no compositing step.

---

## The UI System

The UI system is renderer-agnostic. It maintains its own GPU resources
(offscreen RGBA render target, SSBO, texture registry, font atlas) independently
of whichever renderer is active. It does not share texture registries or render
passes with the scene renderers.

### Rendering modes

| Context        | UI target            | Compositing            |
|----------------|----------------------|------------------------|
| Mission        | Offscreen RGBA buffer | Full-screen alpha-blend shader over scene image |
| Garage         | Offscreen RGBA buffer | Full-screen alpha-blend shader over scene image |
| Loading screen | Swapchain image directly | None                 |

The compositing pass is a single-triangle fullscreen shader that samples the UI
buffer and alpha-blends it over the scene image. `vkCmdBlitImage` is not used
because it does not handle alpha.

### Texture registry

The UI system owns a separate texture registry for widget-owned textures (icons,
portraits, custom widget images). This is distinct from the scene renderers'
texture registries because UI textures have independent lifetimes — they are
created and destroyed with widgets, not with scene assets — and the UI must
operate correctly across all three renderer contexts without lifetime coupling
to any one of them.

---

## Renderer switching

The active renderer changes at well-defined transition points (mission load,
garage entry, etc.). The UI system persists across transitions; only the
rendering mode (offscreen vs. direct) changes.

## Memory pressure (Android)

On Android, `winit` (via `android-activity`) surfaces a `MemoryWarning`
event. The garage renderer is a separate renderer that owns its own
vertex/index buffer (allocated fresh at garage entry after the mission
buffer is freed) and can shrink that allocation dynamically under pressure:
evict some or all cached parts, free the VkBuffer, and reallocate a smaller
one. Evicted parts reload on next access. Mission and loading screen
renderers have no purgeable cache and ignore the warning.

The mission vertex/index buffer is not a candidate for shrinking under
memory pressure: the issue is not performance (`vkAllocateMemory` /
`vkFreeMemory` are cheap one-time operations) but peak usage — resizing
requires allocating the replacement before freeing the old one, so both
are live simultaneously at the worst possible moment. The buffer is sized
to the largest mission loaded in the current session (high-water mark). It
is freed outright at garage entry; the garage allocates its own independent
buffer. At the next mission load, the mission buffer is reallocated sized
to that mission's assets (fine behind a loading screen), so the high-water
mark can shrink naturally over a session.

If the combined mission buffer + garage working set does not fit on a
target device, that is a memory budget problem to address at design time.
