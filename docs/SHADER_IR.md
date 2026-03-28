# Parengus Shader Intermediate Representation (PSIR) — Design Sketch

> Rough design notes from design conversation, 2026-03-14. Generated with
> Claude's assistance.

## Goals

- Engine-managed resource layout (no manual binding/set indices in shaders)
- Automatic push constant packing with uniform buffer spill
- CPU-side reference executor for correctness verification
- Source-level debug info (RenderDoc) — deferred; see Open Questions
- SPIR-V as the current sole compilation target (via `rspirv`)

PSIR is designed to be an efficient representation of shader work — not a
thin wrapper over SPIR-V. It has its own semantics; the SPIR-V Emit Notes
section describes how those semantics lower to SPIR-V, and that mapping
is an emitter concern. Future compilation targets (e.g. DXIL, MSL) would
add their own emit sections without changing the IR.

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

// Struct-typed register — same convention, type name as prefix
Light(0) := load_elem lights idx

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
- `f16` — 16-bit float; executor upcasts to f32, operates, downcasts on
  store; emitter uses native f16 ops when the target supports them,
  otherwise wraps each op in precision-conversion pairs (see SPIR-V
  Emit Notes)
- `ui<N>` — N-bit unsigned integer; N must be in 1–64 (inclusive)
- `i<N>` — N-bit signed integer; N must be in 1–64 (inclusive)
- Textual aliases (desugar to the parameterized form at text→binary time;
  the binary IR stores only `ui<N>` / `i<N>`):

  | Alias | Expands to  | Alias | Expands to |
  |-------|-------------|-------|------------|
  | `u8`  | `ui<8>`     | `i8`  | `i<8>`     |
  | `u16` | `ui<16>`    | `i16` | `i<16>`    |
  | `u32` | `ui<32>`    | `i32` | `i<32>`    |
  | `u64` | `ui<64>`    | `i64` | `i<64>`    |
- All `ui<N>` / `i<N>` types are unconditionally supported on any target.
  The emitter synthesizes correct behavior via masking, widening, or
  hi/lo splitting when native hardware support is absent.
- `bool` — boolean; distinct from `ui<1>` — see below
- `bool` and `ui<1>` are distinct types. `bool` is the logical type used
  with `and`/`or`/`not`/`xor`, branch conditions, and `select`. `ui<1>`
  is a 1-bit integer that participates in integer arithmetic and implicit
  widening. Neither converts implicitly to the other. To convert `bool`
  to an integer use `select b 1 0`; to convert an integer to `bool` use
  `ne x 0`. `cast` does not cross this boundary.
- `vec2<T>`, `vec3<T>`, `vec4<T>` — vector; vec3 has 16-byte std430 alignment
  (footgun)
- `mat2<T>`, `mat3<T>`, `mat4<T>` — matrix; column-major (matches vek and engine
  convention)
- array — fixed-size
- `Texture2D` — opaque resource handle; only usable with texture ops or
  passed to helpers
- `Sampler` — opaque resource handle; same rules as `Texture2D`
- `ReadBuffer<T>` — opaque read-only resource handle; T may be any
  non-opaque type (scalars, vectors, matrices, arrays, structs); declared
  as a resource on entry points with `storage(name)`; elements accessed
  via `load_elem`
- **Opaque types** (`Texture2D`, `Sampler`, `ReadBuffer<T>`) are
  resource handles — they cannot appear as struct fields, array elements,
  vector components, or `ReadBuffer` element types. They may be
  declared as entry point resources or passed as explicit helper arguments.
  `Optional<T>` where T is an opaque type is valid (e.g.
  `Optional<Texture2D>`).
- `struct Name` — named product type declared at module level; fields are
  typed and named; layout is engine-determined (see Struct Types section)
- `Optional<T>` — optional value; T may be any non-Optional type; primary
  use case is optional resource slots (e.g. emissive texture); see
  Optional section
- bundle — anonymous multi-value group — see Bundles section

Type annotations on arithmetic/logic result registers are optional and verified
when present. Casts are their own instruction; inference cannot cross a cast
boundary.

The set of types eligible to appear at shader interface positions (vertex
inputs, interpolants, uniform resources, storage buffer elements) is
exported by the engine. The IR validator checks shader declarations against
the engine's type registry; types that are valid inside a function body may
not be valid at an interface boundary (e.g. `bool` cannot be a vertex
input).

At the binary IR level, integer types are encoded as a `(width: u32,
signedness: bool)` pair in the type table. The declared width N is stored
exactly — not any machine container width.

Vector types are fully concrete (`vec3<f32>`, not generic). The textual IR
can include swizzle pseudo-ops (`.x`, `.xy`, etc.) as
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
x  := const f32    1.0
i  := const i32   42
b  := const bool  true
u  := const u8    255
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
type. Literals are range-checked against the declared width N at text→binary
time — `const ui<4> 16` is an error (16 does not fit in 4 bits; `ui<4>`
has no alias and must be written in full). Out-of-range
is always an error, never silent truncation.

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
`const` instructions followed by `construct`. The emitter can fold an
all-const `construct` into a single constant composite.

### Instruction Immediates

Some instruction arguments are integer immediates embedded in the instruction
encoding — not registers. A runtime register cannot be passed in an
immediate position. Textual IR uses bare integer literals for immediates;
they are not desugared to `const` instructions.

- `extract` — component / field index. On structs and bundles, fields
  have different types, so the index must be a compile-time constant for
  type resolution. On vectors and matrices, the component type is
  uniform; runtime indexing is not currently supported and may be
  addressed by a separate instruction (see Open Questions).
- `shuffle` — component index list. The output vector length (and
  therefore result type) is determined by how many indices are listed,
  which must be known at compile time. Runtime index values are not
  currently supported and may be addressed by a separate instruction
  (see Open Questions).
- `switch` — case key list. The keys are the values the selector is
  compared against; they form a compile-time jump table.

---

## Bundles (Multi-Value Groups)

Used for function call arguments and multi-value return values. Bundles are
anonymous and opaque in the IR — they are not named types. Exception: when
a callee declares exactly one `out:` of a named struct type, the call
result is that struct type directly rather than a single-element bundle —
no extraction step needed.

```
// Call returning multiple values
b(0) = some_fn bundle(arg0, arg1)

// Extract by index
x := extract b(0) 0
y := extract b(0) 1
```

Bundle types are resolved from the callee's declared return preamble;
type-checking `extract` is just "look up index N in the callee's return
bundle type." See SPIR-V Emit Notes for lowering details.

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

`cond` must be `bool`. `out:` is optional — omit it for pure control
flow (no value produced, no bundle result). When `out:` is present,
every branch must write all declared outputs; the declared outputs
become implicit phi nodes at the merge point.

### Loop

```
loop
  carry: [i: i32 = 0 -> i_next]
  body:
    i_next := add i 1
    break eq i_next 10
```

Loops are infinite by default. `break` and `continue` instructions placed
anywhere in the body control exit and iteration. Carried variables are
declared with initial values and next-iteration registers; they become
implicit phi nodes at the loop header. `return` and `discard` may also
appear in the loop body and exit the function or invocation immediately.
The emitter handles structured loop back-edge requirements — this is an
emitter detail with no PSIR-level meaning (see SPIR-V Emit Notes).

### If-Some Expression

```
result = if_some emissive_opt as tex
  out: c: vec4<f32>
  some:
    c := sample tex smp uv
  none:
    c := construct vec4<f32> 0 0 0 0
```

Structured optional branch. `tex` is bound to the inner value and is in
scope only inside the `some` block — it cannot be used in `none`. This is
the only way to extract an Optional's inner value; there is no standalone
`unwrap`. Every branch must write all declared output registers. The
emitter lowers `if_some` identically to `if`, extracting the inner value
into a register at the top of the `some` block.

Because `none` is an explicit block, callers can place `discard` there to
get trap-on-absent behaviour in fragment shaders — no special language
mechanism needed. There is no equivalent termination instruction for
vertex shaders in standard SPIR-V; trap-on-absent in a vertex `none`
block has no clean implementation and should be avoided.

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

`selector` must be an integer type. Cases are integer-literal-keyed
structured blocks. No fallthrough. Every case must write all declared
output registers. The binary IR stores a `[(literal,
offset, instruction_count)]` jump table after the switch opcode.

---

## Instruction Set

### Arithmetic
Scalar and component-wise vector/matrix operands unless noted.

- `add`, `sub`, `mul`, `div`, `rem` — binary; integer or float
- `neg` — unary negate; signed integers and floats only (error on unsigned)
- `mat_mul` — matrix × matrix or matrix × vector; dispatches to the correct
  SPIR-V op based on operand types
- `add_sat`, `sub_sat`, `mul_sat` — saturating binary arithmetic; integer
  only; clamps to the declared type's representable range instead of
  wrapping. Unsigned `ui<N>`: clamps to `[0, 2^N − 1]`. Signed `i<N>`:
  clamps to `[−2^(N−1), 2^(N−1) − 1]`.
- `neg_sat` — saturating unary negate; signed integers only; differs from
  `neg` only at the signed minimum (`neg_sat i8 -128` → `127`; `neg`
  would wrap to `-128`). Error on unsigned.

For all binary integer ops (`add`, `sub`, `mul`, `div`, `rem`, `and`, `or`,
`xor`, `add_sat`, `sub_sat`, `mul_sat`, `shl`, `shr`, `asr`): when
operands share the same signedness but differ in width, the narrower operand
is implicitly widened to the wider type; the result has the wider type.
Mixed signedness is always an error — use explicit `cast` or `cast_sat`
first.

### Comparison (produce `bool` or `vec<bool>`)

- `eq`, `ne` — equality
- `lt`, `le`, `gt`, `ge` — ordered comparison; integers or floats

### Logical and Bitwise

`and`, `or`, `not`, `xor` dispatch on operand type:
- `bool` operands → logical operation
- integer operands → bitwise operation

There is no separate naming — type inference determines the operation. `xor`
on `bool` is logical exclusive-or, filling a gap absent in the original spec.

- `shl` — shift left; both operands must be integers; shift amount may be
  any integer width
- `shr` — logical shift right; zero-fills high bits; natural for unsigned
  integers; shift amount may be any integer width
- `asr` — arithmetic shift right; sign-fills high bits; natural for signed
  integers; shift amount may be any integer width

### Conversion

- `cast` — explicit type conversion between numeric scalars; the only op
  that can change scalar kind; inference does not cross cast boundaries.
  Does not accept `bool` as source or target — use `select`/`ne` instead.
  - *Integer width change* (all well-defined, no UB):
    - *Narrowing* (either signedness): high bits discarded, low N bits
      retained, reinterpreted in the target signedness.
    - *Widening unsigned* (`u8` → `u32`): zero-extension.
    - *Widening signed* (`i8` → `i32`): sign-extension.
  - *Integer → float* (`i32` / `u32` → `f32`): nearest representable
    value; signed and unsigned variants are distinct.
  - *Float → integer* (`f32` → `i32` / `u32`): truncates toward zero;
    out-of-range result is implementation-defined.
  - *Float width change* (`f32` → `f16` or `f16` → `f32`): precision
    conversion.
- `cast_sat` — saturating cast; clamps to the target type's representable
  range before converting; integer→integer only (float conversions use
  `cast`). Examples: `cast_sat i8 (i32 200)` → `127`;
  `cast_sat u8 (i32 -1)` → `0`.

Comparisons (`eq`, `ne`, `lt`, `le`, `gt`, `ge`) are strict — both register
operands must share the exact same declared type. Literals in comparison
position infer their type from the typed peer operand (`eq u8 0` is
valid; `eq u8 u32` is an error). Float widths are never implicitly
promoted — use `cast` first.

### Composite

- `construct` — build a vector or matrix from component registers in
  declaration order: `construct vec4<f32> x y z w`. For structs, fields
  are specified by name in brace syntax (order independent; resolved to
  indices at text→binary time):
  ```
  light := construct Light { pos: p, color: c, radius: r }

  // same, split across lines — whitespace inside braces is insignificant
  light := construct Light {
    pos:    p,
    color:  c,
    radius: r,
  }
  ```
  The binary IR stores struct construction positionally by field index.
- `extract` — extract one sub-element by constant index; on a vector: yields
  a scalar; on a matrix: yields the column vector at that index; on an
  array or struct: yields the field/element. Structs and bundles share the
  same op — structs are named bundles with layout semantics. In the textual
  IR, `extract light pos` is sugar for `extract light 0` (resolved at
  text→binary time).
- `shuffle` — reorder / subset vector components by constant index list
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
  (i32), defaults to mip 0 if omitted

### Read Buffers

- `load_elem buf idx` — load one element of `ReadBuffer<T>` at runtime
  index `idx`; result type is T; index may be any integer type

### Optional

- `some val` — wrap a value in `Optional<T>`; T inferred from `val`
- `none Optional<T>` — absent value; T must be stated explicitly (no
  operand to infer from): `x := none Optional<Texture2D>`
- `is_some opt` → `bool` — test presence without branching; useful when
  the bool needs to be stored or passed to a helper

### Math Intrinsics

- `dot`, `cross` — dot / cross product
- `normalize`, `length`, `distance` — vector geometry
- `reflect`, `refract` — optics
- `sqrt`, `inversesqrt`
- `abs`, `sign`
- `min`, `max`, `clamp` — work on all numeric types (integers and floats);
  signed and unsigned integers use the appropriate ordered comparison
- `mix` — linear interpolate (GLSL `mix` / HLSL `lerp`); float types only
- `floor`, `ceil`, `round`, `trunc`, `fract`
- `pow`, `exp`, `exp2`, `log`, `log2`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan2`

### Select

- `select cond true_val false_val` — both operands always evaluated (not
  a branch). `bool` condition: selects between two scalars. `vec<bool>`
  condition: component-wise select between two vectors of the same shape —
  each output component is chosen independently by the corresponding
  condition component.

### Fragment-Only

- `dpdx`, `dpdy` — partial screen-space derivatives
- `discard` — terminate the fragment invocation; no result (see SPIR-V
  Emit Notes for target-dependent lowering)

### Control Flow

- `return` — marks output registers valid; emitter branches to function exit
  block
- `break` — exit the enclosing loop; optionally takes a `bool` condition
  (`break cond`) and only exits if `cond` is true
- `continue` — begin the next loop iteration; optionally takes a `bool`
  condition (`continue cond`) and only continues if `cond` is true
- `call` — call a function; result type depends on the callee's `out:`
  declaration: a single named struct `out:` yields that struct type
  directly (`light := call get_light idx`); any other case yields a
  bundle (`b(0) = call fn ...`)
- `if_some` — structured optional branch; see If-Some Expression section

---

## Module Structure

A module is the top-level compilation unit. It contains an optional
module-level type block (struct declarations only) followed by functions
(entry points and helpers).

### Struct Types

Structs are declared before all function definitions:

```
struct Light
  pos:    vec3<f32>
  color:  vec3<f32>
  radius: f32
```

Structs are named bundles with layout semantics. `extract` and `construct`
work on structs exactly as they do on bundles — no new composite ops. In
the textual IR, `extract light pos` is sugar for `extract light 0` (field
name resolved to index at text→binary time). Field names are semantic (not
debug-only): they are interned to integer IDs in the binary IR type table
entry `[(name_id: u32, type_index: u32)]`, and the engine uses them at
pipeline compile time to remap shader field accesses to the CPU-side
canonical buffer layout.

Field byte layout is engine-determined. The first-pass strategy is: the
CPU side uses a fixed canonical layout; the engine emits SPIR-V
access-chain remappings at load time. Packing optimisation is deferred.
Shader authors never write byte offsets or padding.

Functions are declared with `fn <name>`. Entry points carry a `stage:` field
(`vertex`, `fragment`; `compute` is future work). Helper functions have no
`stage:` field and no `resources:` block — they are callable from any entry
point.

Resource declarations live on **entry points**, not at module level. Helper
functions receive resources as explicit arguments — they have no implicit access
to any resource.

```
struct Light
  pos:    vec3<f32>
  color:  vec3<f32>
  radius: f32

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
       prim_id:  u32       output(prim_id)  flat
body:
  ...
  // resources passed explicitly to helpers
  color := call brdf cam_view cam_proj normal view

fn frag
  stage: fragment
  resources:
    cam_view:    mat4<f32>             uniform(camera_view)
    cam_proj:    mat4<f32>             uniform(camera_proj)
    atex:        Texture2D             uniform(albedo_tex)
    asmp:        Sampler               uniform(albedo_smp)
    emissive:    Optional<Texture2D>   uniform(emissive_tex)
    lights:      ReadBuffer<Light>  storage(lights)
  in:  uv:  vec2<f32>  input(frag_uv)
       pid: u32       input(prim_id)  flat
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

When compiling a pipeline, the engine collects the `uniform` and `storage`
declarations from all linked entry points, deduplicates shared resources
(matched by name and type), and performs layout assignment once over the
merged set.

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

Every binary module begins with a fixed header containing a magic number
and a format version. The version field allows decoders to reject
incompatible formats and supports maintaining older decoders alongside
newer ones. The exact header layout is deferred to implementation, but
the version field is a first-class design commitment — it will always be
present and always be at a stable offset in the header.

In the binary IR, named types become indexed type entries in a module-level
type table. Resources become indexed entries in a resource table (interned name
ID, type index, flags). Vertex input semantics are stored as interned name IDs.
Functions are ordered; entry points carry a stage tag. Location assignments are
stored as resolved indices in the binary IR after the engine/compiler pass runs.

Instructions are variable-width but self-describing: the opcode is a u16.
For most instructions the operand count is fixed by the opcode. The two
exceptions are `construct` and `call`, which encode an explicit operand
count immediately after the opcode. A decoder never needs to consult the
type or function table to determine instruction boundaries. No alignment
requirements — instructions and functions are packed sequentially and
read linearly.

---

## Resources and Layout

Resources are declared on entry points by local name, type, and an
annotation identifying the engine-side resource. No binding indices or
frequency annotations appear in the IR — the engine already knows the
frequency, binding policy, and layout for every named resource from its
own registry. The shader just declares which resources it needs and under
what local name.

Two resource annotations exist for data resources:
- `uniform(name)` — read-only uniform value (scalar, vector, matrix,
  texture, sampler, or `Optional<T>`)
- `storage(name)` — read buffer (`ReadBuffer<T>`); read-only by definition

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

All six annotation forms — `uniform(name)`, `storage(name)`, `input(name)`,
`output(name)`, `external_in(name)`, and `external_out(name)` — follow the
same model: a string at authoring time, interned to an integer ID in the
binary IR. At runtime only
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
- Integer types carry their exact declared width N to the emitter; container
  width selection, capability requirements, and any fallback emulation are
  emitter decisions not specced in this document.
- Saturating operations (`add_sat`, `sub_sat`, `mul_sat`, `neg_sat`,
  `cast_sat`) have no native SPIR-V equivalent for integers; the emitter
  synthesizes the correct clamp-after-op or clamp-before-convert
  sequence. This is expected to be the case for all current targets —
  it is an explicit emitter responsibility, not a gap in the IR.
- `f16` arithmetic: emitter uses native `Float16` ops when the target
  supports them; otherwise each op is wrapped in `OpFConvert` to/from
  f32 (promote → op → demote).
- `cast` numeric conversions: integer width changes lower to
  `OpSConvert`/`OpUConvert`; integer→float to
  `OpConvertSToF`/`OpConvertUToF`; float→integer to
  `OpConvertFToS`/`OpConvertFToU`; float width change to `OpFConvert`.
- Bundles lower to `OpTypeStruct` (type) / `OpCompositeConstruct`
  (construction) / `OpCompositeExtract` (extraction).
- `xor` on `bool` operands lowers to `OpLogicalNotEqual` — SPIR-V has no
  `OpLogicalXor`.
- Struct types lower to `OpTypeStruct`. Field names are not present in
  SPIR-V; the emitter uses indices only. The engine emits access-chain
  remappings (`OpAccessChain`) to bridge the shader's declared field order
  to the CPU-side canonical buffer layout.
- `ReadBuffer<T>` lowers to an `OpTypeRuntimeArray` element wrapped in
  an `OpTypeStruct` block decorated `BufferBlock` (SPIR-V 1.3) or with
  `StorageBuffer` storage class (SPIR-V 1.4+); `load_elem` lowers to
  `OpAccessChain` + `OpLoad`.
- `Optional<T>` default lowering: `(bool, T)` register pair; `is_some`
  reads the bool; `if_some` reads T into the bound register at the top of
  the `some` block. Optimized lowering for `Optional<ui<N>>` /
  `Optional<i<N>>` when N < container width: emitter packs the presence
  flag into a spare high bit of the container register.
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

Integer registers are stored internally as `u64` / `i64` and masked to N
bits on every write. This simulates exact wrap-around and saturation
semantics for any declared width N ≤ 64 without requiring native types for
every width.

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

- `f64` — 64-bit float; primary use case is compute-heavy / scientific
  workloads (physics, numerical methods); requires `Float64` Vulkan feature
  and has no viable full-fidelity emulation; co-deferred with compute shader
  entry points
- PSL (Parengus Shading Language) surface syntax — deferred, PSIR textual form
  is expressive enough to write shaders directly for now
- `switch` as sugar over nested if/else at the IR level vs. direct `OpSwitch`
  emit — current plan: direct `OpSwitch` from day one
- `WriteBuffer<T>` / `ReadWriteBuffer<T>` — writable and read-write
  storage buffers. A recurring pattern: payload buffers can be
  `WriteBuffer<T>` (e.g. draw commands, OIT nodes), but the atomic
  counter that allocates slots in those buffers requires
  `ReadWriteBuffer<u32>`. Examples: GPU culling writes
  `WriteBuffer<DrawCommand>` but needs `ReadWriteBuffer<u32>` for the
  draw count; OIT writes `WriteBuffer<OITNode>` but needs
  `ReadWriteBuffer<u32>` for head pointers and counters.
  `WriteBuffer` alone is never sufficient when dynamic allocation is
  involved. Approximate OIT via weighted blending needs only render
  targets and is available now.
- Sum types (tagged unions / enum-with-data) — natural extension once
  structs land; useful for material variant dispatch and similar patterns
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
- Dynamic composite access — runtime-indexed `extract` on vectors/matrices
  and runtime-indexed `shuffle`; likely requires separate instructions
  (e.g. `extract_dynamic`, `gather_components`) rather than relaxing the
  immediate constraint on the existing ops
- Optimizer pass (or delegate to `spirv-opt`) — deferred
- Binary format section layout and debug metadata — section-based container
  with optional per-function debug sections; section header format and
  encoding deferred to implementation
- SPIR-V debug info (`NonSemantic.Shader.DebugInfo.100`) — deferred;
  depends on binary format debug section design
