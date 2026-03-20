# Parengus Shader Intermediate Representation (PSIR) — Design Sketch

> Rough design notes from design conversation, 2026-03-14. Generated with
> Claude's assistance.

## Goals

- Engine-managed resource layout (no manual binding/set indices in shaders)
- Automatic push constant packing with uniform buffer spill
- CPU-side reference executor for correctness verification
- Source-level debug info (RenderDoc) — deferred; see Open Questions
- SPIR-V as the current sole compilation target (via `rspirv`)

---

> Note (2026-03-20): PSIR work is deferred for the near term. The
> PBR shading task may proceed without PSIR; this document remains
> the spec for PSIR when work resumes.

## Crate Structure (planned)

- `psir` — IR data definitions; pure data, no engine or compiler deps
- `psir-spirv` — pure binary IR → SPIR-V lowering; no engine deps
- `psir-engine` — engine-side backend: layout assignment, calls into
  `psir-spirv`
- `psir-compiler` — frontend: text → binary IR, eventually PSL → binary
- `psir-executor` — CPU reference executor (correctness oracle)

`psir-spirv` is depended on by both `psir-engine` (runtime pipeline compilation)
and `psir-compiler` (offline tooling), so it carries no engine or compiler
dependencies — just `psir` + `rspirv`.

---

## Register Model

Typed infinite registers — each register is written exactly once (SSA), has a
type, and lives for the duration of its scope. No register reuse, no aliasing.

Every function has its own register namespace (no clobbering between caller and
callee).

### Textual Syntax

```
// Typed declaration — register index with explicit type
f32(0) := add f32(1) f32(2)

// Inferred — type comes from operand types
result := add f32(0) f32(0)

// Named — bare identifier, type inferred
albedo := sample tex(0) sampler(0) uv

// Annotated — explicit type verified against inference
quad : f32 = add double double   // errors if inferred type != f32
```

Left of `=` is always a register declaration. Right of `=` is always an
instruction. Named registers are the primary authoring surface; indexed
registers (`f32(0)`) appear in the binary IR and in low-level textual IR.
Register names are not part of the binary IR semantics. In the binary format,
only indexed registers appear in the instruction stream; names, if present,
live in a separate debug section mapping register index to interned string ID.

---

## Types

- `f32` — 32-bit float
- `f16` — 16-bit float; executor upcasts to f32, operates, downcasts on store;
  emitter wraps ops in `OpFConvert` pairs
- `u32` — 32-bit unsigned integer
- `i32` — 32-bit signed integer
- `bool` — boolean
- `vec2<T>`, `vec3<T>`, `vec4<T>` — vector; vec3 has 16-byte std430 alignment
  (footgun)
- `mat2<T>`, `mat3<T>`, `mat4<T>` — matrix; column-major (matches vek and engine
  convention)
- array — fixed-size
- `Texture2D` — opaque texture handle; only usable with texture ops or passed to
  helpers
- `Sampler` — opaque sampler handle; same rules as `Texture2D`
- bundle — anonymous multi-value group — see Bundles section

Type annotations on arithmetic/logic result registers are optional and verified
when present. Casts are their own instruction; inference cannot cross a cast
boundary.

At the binary IR level, vector types are fully concrete (`vec3<f32>`, not
generic). The textual IR can include swizzle pseudo-ops (`.x`, `.xy`, etc.) as
authoring convenience — the text→binary pass lowers them to `extract` (component
by index → scalar) and `shuffle` (component subset / reorder → vector). The
binary IR, executor, and emitter never see swizzle syntax. The textual IR does
not need to be 1:1 with the binary IR; pseudo-ops are fine.

---

## Constants and Literals

### `const` Instruction

In the binary IR, every constant value is produced by a `const` instruction that
writes a typed scalar register:

```
x  := const f32  1.0
i  := const i32  42
b  := const bool true
```

The instruction carries the type and bit pattern of the value. Constants are
ordinary registers — deduplication within a function is a compiler optimization,
not a binary IR requirement.

### Textual IR — Type Inference

In the textual IR, literals are untyped tokens. Their type is resolved from
context, in order:

1. **LHS type annotation** — `x : f32 = const 1.0` → f32 from the annotation.
2. **Typed peer operand** — `add f32(0) 1` → `1` infers as f32 from `f32(0)`.
3. **Explicit type on `const`** — `x := const f32 1.0` — required when neither
   of the above applies. Error if omitted.

No magic suffixes, no default numeric types. A literal `1.0` in an integer
context is an error; a literal `1` in a float context resolves to that float
type.

Literals in instruction argument position are sugar for an anonymous `const`
register. The text→binary pass emits the `const` instruction and substitutes its
register index.

### Composite Literals

Composite constants (vectors, matrices) are written in the textual IR as
`construct` with literal operands:

```
v : vec4<f32> = construct 0 0 0 1   // scalar types inferred from LHS
```

Each literal desugars to a scalar `const`; the binary IR contains the individual
`const` instructions followed by `construct`. The SPIR-V emitter recognizes an
all-const `construct` and folds it into a single `OpConstantComposite`.

### Instruction Immediates

Some instruction arguments are integer immediates embedded in the instruction
encoding — not registers. These exist only where the type system or code
generation structurally requires compile-time knowledge:

- `extract` — component / field index
- `shuffle` — component index list
- `switch` — case key list

A runtime register cannot be passed in an immediate position. Textual IR uses
bare integer literals for immediates; they are not desugared to `const`
instructions.

---

## Bundles (Multi-Value Groups)

Used for function call arguments and return values. Bundles are anonymous and
opaque in the IR — they are not named types.

```
// Call returning multiple values
b(0) = some_fn bundle(arg0, arg1)

// Extract by index
x := extract b(0) 0
y := extract b(0) 1
```

The emitter maps bundles to `OpTypeStruct` / `OpCompositeConstruct` /
`OpCompositeExtract` in SPIR-V. Bundle types are resolved from the callee's
declared return preamble; type-checking `extract` is just "look up index N in
the callee's return bundle type."

---

## Function Structure

Every function has a **preamble** that declares:
- Input bundle (typed by index)
- Output (return) registers (typed, named)
- Local registers (typed or inferred)

```
fn brdf
  in:  normal: vec3<f32>, view: vec3<f32>, roughness: f32
  out: color: vec4<f32>
body:
  ...
  return   // "output registers are valid", emitter branches to exit block
```

Early `return` writes the output registers then branches to the function's exit
block. No phi nodes needed at the return site because output registers are
pre-declared and written exactly once on each path.

Because output registers are declared up front and control flow is structured, a
definite-assignment pass can verify that every control flow path writes all
declared output registers before `return`. The pass only needs to recurse over
the nesting tree — no general dataflow analysis required. This is planned as a
binary IR analysis pass in `psir`.

---

## Control Flow

Structured only — no goto. Basic blocks are implicit; the emitter carves them
from the nesting structure.

### If Expression

```
result = if cond
  out: val: f32
  then:
    val := ...
  else:
    val := ...
```

Produces a bundle. Every branch must write all declared outputs. The emitter
emits `OpSelectionMerge` + `OpBranchConditional` + `OpPhi` at the merge block.

### Loop

```
loop
  carry: [i: i32 = 0 -> i_next]
  body:
    i_next := add i 1
    break_if := eq i_next 10
  break: break_if
  continue: (implicit — back-edge to header)
```

Carried variables are declared with initial values and update expressions. The
emitter emits `OpLoopMerge` (merge block + continue block) and `OpPhi` at the
loop header for each carried variable.

`break` and `continue` are explicit instructions. `break` branches to the merge
block; `continue` branches to the continue block.

### Switch

```
b(0) = switch selector
  out: val: f32, index: u32
  case 0:
    val := ...
    index := ...
  case 1:
    val := ...
    index := ...
  default:
    val := ...
    index := ...
```

Cases are integer-literal-keyed structured blocks. No fallthrough. Every case
must write all declared output registers. The binary IR stores a `[(literal,
offset, instruction_count)]` jump table after the switch opcode. The emitter
maps this to `OpSwitch`.

---

## Instruction Set

### Arithmetic
Scalar and component-wise vector/matrix operands unless noted.

- `add`, `sub`, `mul`, `div`, `rem` — binary; integer or float
- `neg` — unary negate
- `mat_mul` — matrix × matrix or matrix × vector; dispatches to the correct
  SPIR-V op based on operand types

### Comparison (produce `bool` or `vec<bool>`)

- `eq`, `ne` — equality
- `lt`, `le`, `gt`, `ge` — ordered comparison; integers or floats

### Logical (`bool` operands)

- `and`, `or`, `not` — boolean logic

### Bitwise (integer operands)

- `band`, `bor`, `bnot`, `bxor` — bitwise logic
- `shl`, `shr` — shift left / right

### Conversion

- `cast` — explicit type conversion; the only op that can change scalar type;
  inference does not cross cast boundaries

### Composite

- `construct` — build a vector or matrix from component registers
- `extract` — extract one sub-element by constant index; on a vector: yields a
  scalar; on a matrix: yields the column vector at that index; on an array:
  yields the element
- `shuffle` — reorder / subset vector components by constant index list; maps to
  `OpVectorShuffle`
- `extract_row` — extract row i from a matrix; yields a vector; more expensive
  than `extract` because matrices are column-major — decomposes to per-column
  scalar `extract`s + `construct`
- `transpose` — transpose a matrix; yields a matrix with rows and columns
  swapped

### Textures

- `sample` — sample with implicit LOD (fragment shader)
- `sample_lod` — sample with explicit LOD
- `sample_grad` — sample with explicit gradient (`dPdx`, `dPdy`)
- `texel_fetch` — integer-coordinate fetch, no sampler; optional LOD argument
  (i32), defaults to mip 0 if omitted; maps to `OpImageFetch`

### Math Intrinsics

- `dot`, `cross` — dot / cross product
- `normalize`, `length`, `distance` — vector geometry
- `reflect`, `refract` — optics
- `sqrt`, `inversesqrt`
- `abs`, `sign`
- `min`, `max`, `clamp`
- `mix` — linear interpolate (GLSL `mix` / HLSL `lerp`)
- `floor`, `ceil`, `round`, `trunc`, `fract`
- `pow`, `exp`, `exp2`, `log`, `log2`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan2`

### Select

- `select cond true_val false_val` — choose between two values based on a `bool`
  or `vec<bool>`; both operands always evaluated; maps to `OpSelect`; works on
  scalars and vectors (component-wise)

### Fragment-Only

- `dpdx`, `dpdy` — partial screen-space derivatives
- `discard` — terminate the fragment invocation; no result; emitter maps to
  `OpKill` or `OpTerminateInvocation` based on the SPIR-V target

### Control Flow

- `return` — marks output registers valid; emitter branches to function exit
  block
- `break` — branch to loop merge block
- `continue` — branch to loop continue block
- `call` — call a function; result is a bundle

---

## Module Structure

A module is the top-level compilation unit. It contains functions (entry points
and helpers). There is no module-level type block — all uniform resources are
flat primitive/vector/matrix values; the engine registry maps each
`uniform(name)` to its type.

Functions are declared with `fn <name>`. Entry points carry a `stage:` field
(`vertex`, `fragment`; `compute` is future work). Helper functions have no
`stage:` field and no `resources:` block — they are callable from any entry
point.

Resource declarations live on **entry points**, not at module level. Helper
functions receive resources as explicit arguments — they have no implicit access
to any resource.

```
fn vert
  stage: vertex
  resources:
    cam_view: mat4<f32>  uniform(camera_view)
    cam_proj: mat4<f32>  uniform(camera_proj)
    atex:     Texture2D  uniform(albedo_tex)
    asmp:     Sampler    uniform(albedo_smp)
  in:  pos: vec3<f32>  input(position)
       uv:  vec2<f32>  input(texcoord0)
  out: clip_pos: vec4<f32>  external_out(position)
       frag_uv:  vec2<f32>  output(frag_uv)
       prim_id:  u32         output(prim_id)  flat
body:
  ...
  // resources passed explicitly to helpers
  color := call brdf cam_view cam_proj normal view

fn frag
  stage: fragment
  resources:
    cam_view: mat4<f32>  uniform(camera_view)
    cam_proj: mat4<f32>  uniform(camera_proj)
    atex:     Texture2D  uniform(albedo_tex)
    asmp:     Sampler    uniform(albedo_smp)
  in:  uv:  vec2<f32>  input(frag_uv)
       pid: u32         input(prim_id)  flat
  out: color: vec4<f32>  external_out(color0)
body:
  ...

fn brdf            // helper — no stage:, no resources:
  in:  cam_view: mat4<f32>, cam_proj: mat4<f32>,
       normal: vec3<f32>, view: vec3<f32>
  out: color: vec4<f32>
body:
  ...
```

When compiling a pipeline, the engine collects the `uniform` declarations from
all linked entry points, deduplicates shared resources (matched by name and
type), and performs layout assignment once over the merged set.

### Location Assignment

All locations are implicit — the engine and compiler assign them; shader authors
never write `location(N)`.

- **Vertex inputs** — `input(name)` annotation; engine maps the name to a vertex
  buffer attribute. The name implies the type — a mismatch is an error. Local
  variable name is independent.
- **Vertex → fragment interpolants** — `output(name)` on vertex `out`,
  `input(name)` on fragment `in`; compiler matches by name and assigns a shared
  `Location`. Stage context disambiguates `input`: on a vertex shader it is a
  vertex attribute; on a fragment shader it is an interpolated varying. Optional
  `flat` qualifier suppresses interpolation (`Flat` decoration in SPIR-V).
  Integer and boolean varyings require `flat` — SPIR-V validation rejects
  interpolated integer/bool interface variables. Unmatched vertex outputs are
  written but ignored; unmatched fragment inputs are an error.
- **GPU pipeline inputs** — `external_in(name)` annotation; no location. Emitter
  owns a fixed table mapping each name to the corresponding SPIR-V `BuiltIn`
  decoration.
- **GPU pipeline outputs** — `external_out(name)` annotation; same model. Covers
  built-in outputs (`position`, `frag_depth`) and fragment color outputs
  (`color0`, `color1`, …).

Standard `external_in` names:
- `frag_coord: vec4<f32>` — fragment; window-space coord (`gl_FragCoord`)
- `vertex_index: u32` — vertex (`gl_VertexIndex`)
- `instance_index: u32` — vertex (`gl_InstanceIndex`)

Standard `external_out` names:
- `position: vec4<f32>` — vertex; clip-space position (`gl_Position`)
- `point_size: f32` — vertex (`gl_PointSize`)
- `frag_depth: f32` — fragment (`gl_FragDepth`)
- `color0: vec4<f32>` — fragment; render target attachment 0
- `color1: vec4<f32>` — fragment; render target attachment 1

New names are added to the emitter's fixed table as needed. A name not in the
table is an error.

Standard `input` names:
- `position: vec3<f32>` — object-space position
- `normal: vec3<f32>` — object-space normal
- `tangent: vec3<f32>` — object-space tangent
- `texcoord0: vec2<f32>` — primary UV set
- `texcoord1: vec2<f32>` — secondary UV set
- `color0: vec4<f32>` — vertex color

The engine unpacks all vertex data to these types before the shader sees it — no
packed formats, no raw bytes in PSIR. New names are added to both the engine and
IR together as needed.

### Binary IR Form

In the binary IR, named types become indexed type entries in a module- level
type table. Resources become indexed entries in a resource table (interned name
ID, type index, flags). Vertex input semantics are stored as interned name IDs.
Functions are ordered; entry points carry a stage tag. Location assignments are
stored as resolved indices in the binary IR after the engine/compiler pass runs.

---

## Resources and Layout

Resources are declared on entry points by local name, type, and `uniform(name)`
identifier. No binding indices or frequency annotations appear in the IR — the
engine already knows the frequency, binding policy, and layout for every named
resource from its own registry. The shader just declares which resources it
needs and under what local name.

The engine assigns concrete bindings at pipeline creation time. It may map
resources to descriptor sets, push constants, or other mechanisms as its
renderer policy dictates (e.g. bindless, multi-draw indirect). The IR is
unaffected by these decisions.

The layout engine is conservative by default: everything spills to uniform
buffers. Push constant packing is an optimization applied after correctness is
established.

Push constants follow `std430` layout rules (alignment = member size, vec3
footgun = 16-byte alignment). The layout engine owns this logic once; no shader
author needs to think about it.

Resources declared on entry points are in scope as named values in the entry
point body. They are passed to helper functions as explicit arguments — helpers
have no implicit resource access.

All five annotation forms — `uniform(name)`, `input(name)`, `output(name)`,
`external_in(name)`, and `external_out(name)` — follow the same model: a string
at authoring time, interned to an integer ID in the binary IR. At runtime only
IDs appear; no strings in the hot path.

The engine maintains a registry mapping name → ID for `uniform` and `input`
names, populated at startup from asset definitions. The compiler matches
`output(name)` / `input(name)` pairs across linked entry points and assigns
shared `Location` indices. The emitter maintains a fixed table mapping
`external_in`/`external_out` name IDs to SPIR-V `BuiltIn` decorations. In all
cases the mapping from name to implementation is owned by the consumer (engine,
compiler, or emitter), not the IR.

---

## SPIR-V Emit Notes

- The emitter accepts a SPIR-V target environment (version + capability set) as
  a configuration parameter. The IR is version-agnostic; the emitter selects ops
  based on this config (e.g. `discard` → `OpTerminateInvocation` on SPIR-V 1.6 /
  Vulkan 1.3, `OpKill` otherwise)
- `rspirv` handles opcode encoding, module structure, type deduplication
- Phi nodes are entirely an emitter concern — never appear in PSIR
- `load_uniform`, `load_input`, `store_output` are emitter-internal operations;
  they do not appear in the PSIR instruction set. Entry point resources and
  inputs are in-scope registers; the emitter generates the appropriate
  `OpLoad`/`OpStore` and `OpVariable` declarations.
- Structured control flow maps directly to SPIR-V's requirements:
  - `OpSelectionMerge` before if/switch branches
  - `OpLoopMerge` (merge + continue blocks) before loop header branch
- Each function gets its own ID namespace (matches SPIR-V per-function scope)
- `spirv-val` should be run on all emitted modules during development

---

## Executor

The executor runs PSIR shaders on the CPU as a correctness oracle. It is a
debugging tool, not used in normal engine operation.

### CSV I/O

The executor accepts RenderDoc CSV exports directly as input (no custom format).
Feed it the input CSV, run the shader, compare output against the GPU output CSV
from the same capture. This makes it straightforward to reproduce a GPU
divergence from a saved capture without re-running the engine.

CSV format is matched to RenderDoc's actual export format at implementation time
rather than specced upfront.

### Verbosity

Off by default. Two opt-in levels:

- `off` — only final output values (default)
- `named` — print each named register as it is assigned
- `all` — print every register assignment including anonymous intermediates

`named` is the practical debugging level for shader logic; `all` is for
diagnosing the emitter or executor itself.

---

## Open Questions / Future Work

- PSL (Parengus Shading Language) surface syntax — deferred, PSIR textual form
  is expressive enough to write shaders directly for now
- `switch` as sugar over nested if/else at the IR level vs. direct `OpSwitch`
  emit — current plan: direct `OpSwitch` from day one
- Storage buffers for light lists (needed for forward+ lighting)
- Compute shader entry points
- Image stores (`store_texel`) for storage images
- Shadow sampling: `sample_compare`, `sample_compare_lod` (depth texture +
  reference value; needed for shadow maps)
- Gather: `gather`, `gather_compare` (4-texel component gather; useful for PCF
  shadow filtering)
- Texture queries: `query_size` (dimensions at mip level), `query_levels` (mip
  count)
- MSAA fetch: `texel_fetch` sample-index argument for multisampled textures
- Array type syntax (element type, size) and `extract` semantics — listed in
  Types but not fully specced; defer to implementation
- Optimizer pass (or delegate to `spirv-opt`) — deferred
- Binary format section layout and debug metadata — section-based container
  with optional per-function debug sections; section header format and
  encoding deferred to implementation
- SPIR-V debug info (`NonSemantic.Shader.DebugInfo.100`) — deferred;
  depends on binary format debug section design
