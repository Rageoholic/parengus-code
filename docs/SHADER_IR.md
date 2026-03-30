# Parengus Shader Intermediate Representation (PSIR) — Design Sketch

> Rough design notes from design conversation, 2026-03-14. Generated with
> Claude's assistance.

## Goals

- Engine-managed resource layout (no manual binding/set indices in shaders)
- Named interface contracts and explicit entry-point bindings to those
  interfaces
- No shader-authored optimization hints; optimization policy is engine-owned
- Automatic push constant packing with uniform buffer spill
- CPU-side reference executor for correctness verification
- Source-level debug info (RenderDoc) — deferred; see Open Questions
- SPIR-V as the current sole compilation target (via `rspirv`)

PSIR is designed to be an efficient representation of shader work — not a thin
wrapper over SPIR-V. It has its own semantics; the SPIR-V Emit Notes section
describes how those semantics lower to SPIR-V, and that mapping is an emitter
concern. Future compilation targets (e.g. DXIL, MSL) would add their own emit
sections without changing the IR.

---

> Note (2026-03-20): PSIR work is deferred for the near term. The PBR shading
> task may proceed without PSIR; this document remains the spec for PSIR when
> work resumes.

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
callee). Returns are passed via bundles for multiple return values or by value
for single return values.

### Textual Syntax

```
// Typed declaration — register index with explicit type
f32(0) := add f32(1) f32(2)

// Struct-typed register — same convention, type name as prefix
Light(0) := load_elem lights idx

// Inferred — type comes from operand types
result := add f32(0) f32(0)

// Named — bare identifier, type inferred
albedo := sample tex(0) uv LinearWrap

// Annotated — explicit type verified against inference
quad : f32 = add double double   // errors if inferred type != f32
```

Left of `=` is always a register declaration. Right of `=` is always an
instruction. Named registers are the primary authoring surface; indexed
registers (`f32(0)`) appear in the binary IR and in low-level textual IR.
Register names are not part of the binary IR semantics. In the binary format,
only indexed registers appear in the instruction stream; names, if present, live
in a separate debug section mapping register index to a debug-name entry.

---

## Types

- `f32` — 32-bit float
- `f16` — 16-bit float; executor upcasts to f32, operates, downcasts on store;
  emitter uses native f16 ops when the target supports them, otherwise wraps
  each op in precision-conversion pairs (see SPIR-V Emit Notes)
- `ui<N>` — N-bit unsigned integer; N must be in 1–64 (inclusive)
- `i<N>` — N-bit signed integer; N must be in 1–64 (inclusive)
- Textual aliases (desugar to the parameterized form at text→binary time; the
  binary IR stores only `ui<N>` / `i<N>`):

  | Alias | Expands to  | Alias | Expands to |
  |-------|-------------|-------|------------|
  | `u8`  | `ui<8>`     | `i8`  | `i<8>`     |
  | `u16` | `ui<16>`    | `i16` | `i<16>`    |
  | `u32` | `ui<32>`    | `i32` | `i<32>`    |
  | `u64` | `ui<64>`    | `i64` | `i<64>`    |
- All `ui<N>` / `i<N>` types are unconditionally supported on any target. The
  emitter synthesizes correct behavior via masking, widening, or hi/lo splitting
  when native hardware support is absent.
- `bool` — boolean; distinct from `ui<1>` — see below
- `bool` and `ui<1>` are distinct types. `bool` is the logical type used with
  `and`/`or`/`not`/`xor`, branch conditions, and `select`. `ui<1>` is a 1-bit
  integer that participates in integer arithmetic and implicit widening. Neither
  converts implicitly to the other. To convert `bool` to an integer use `select
  b 1 0`; to convert an integer to `bool` use `ne x 0`. `cast` does not cross
  this boundary.
- `vec2<T>`, `vec3<T>`, `vec4<T>` — vector; vec3 has 16-byte std430 alignment
  (footgun)
- `mat2<T>`, `mat3<T>`, `mat4<T>` — matrix; column-major (matches vek and engine
  convention)
- array — fixed-size
- **Texture types** — opaque resource handles. Logically each texture is a
  physical index paired with a `SamplingPolicy` describing how to interpret
  the texel data. The encoding is opaque to the shader; the shader calls
  `sample` and receives a `vec4<f32>`. May appear as struct fields, entry
  point resources, or explicit helper arguments. Cannot appear as vector
  components or `ReadBuffer` element types. `Optional<T>` is valid for any
  texture type. SPIR-V does not allow opaque types as struct members; the
  emitter unpacks them during lowering.

  Sampler mode is an immediate on every sample op (not a type or binding).
  Builtin modes: `LinearClamp` (default), `LinearWrap`, `LinearMirror`,
  `NearestClamp`, `NearestWrap`, `NearestMirror`. See Instruction Set §
  Textures for full sample op syntax.

  Concrete texture types:
  - **`Texture1D`** — 1D texture (LUTs, color grading curves). `pos`: `f32`.
    `color := sample lut u`
  - **`Texture1DArray`** — array of 1D textures. `pos`: `f32`, `layer`: int.
    `color := sample lut u layer`
  - **`Texture2D`** — 2D texture. `pos`: `vec2<f32>`.
    `color := sample tex uv LinearWrap`
  - **`TextureArray2D`** — array of 2D textures (terrain layers, sprites).
    `pos`: `vec2<f32>`, `layer`: int.
    `color := sample arr uv layer`
  - **`Texture3D`** — volumetric texture (volumetric effects, 3D LUTs).
    `pos`: `vec3<f32>`.
    `color := sample vol uvw`
  - **`TextureCube`** — cube map (environment maps, skyboxes, specular IBL).
    `pos`: `vec3<f32>` direction; need not be normalized.
    `color := sample env dir`
  - **`TextureCubeArray`** — array of cube maps (point light shadow maps).
    `pos`: `vec3<f32>` direction, `layer`: int.
    `color := sample envs dir layer`

  **`SamplingPolicy`** — a PSIR builtin sum type carried as part of every
  texture handle (not user-declarable); applies to all texture types. During
  backend lowering, a texture handle materializes as
  `{ indices: u64, policy: SamplingPolicy }` where
  `indices` packs up to 4×16-bit physical texture indices (one per output
  channel). The emitter matches on the policy to inline channel-gather and
  reconstruction logic; when the handle is statically known the branch folds
  away. Sum types are therefore present in the lowering IR at minimum even if
  not exposed at the authored PSIR surface. The lowered struct shape is not yet
  final. Efficiently lowering multi-index textures to bindful slots is an open
  question (up to 4 slots per logical texture, breaking the 1:1 assumption in
  slot allocation and DCE). sRGB decode and signed unpack are handled by the
  image view/texture format on all targets (Vulkan, DXIL, Metal), not the
  policy.

  Policy variants (non-exhaustive — new variants may be added):
  - `Standard` — pass-through; samples index slot 0.
  - `Broadcast(u2 channel)` — samples slot 0, selects one channel,
    returns `(c, c, c, 1.0)`.
  - `ReconstructNormalZ` — samples slot 0 for X and Y, computes
    `Z = sqrt(1 - X² - Y²)`, returns `(X, Y, Z, 0.0)`.
  - `SelectReconstructNormalZ(u2 x_slot, u2 y_slot)` — same but X and Y
    sourced from specified index slots.
  - `SelectMask(u2 r_slot, u2 g_slot, u2 b_slot, u2 a_slot)` — each output
    channel taken from the corresponding source slot. A post-lowering pass
    deduplicates sample ops so each slot is sampled at most once.
  - `Swizzle(u2 r, u2 g, u2 b, u2 a)` — single-slot channel reorder; each
    component selects which source channel to read from slot 0.

- `ReadBuffer<T>` — opaque read-only resource handle; T may be any non-opaque
  type (scalars, vectors, matrices, arrays, structs); declared as a resource on
  entry points with `storage(name)`; elements accessed via `load_elem`
- **Opaque types** (all texture types and `ReadBuffer<T>`) are resource handles.
  They cannot appear as vector components or `ReadBuffer` element types.
  `Optional<T>` where T is an opaque type is valid (e.g.
  `Optional<Texture2D>`). SPIR-V does not allow opaque types as struct
  members; the emitter unpacks them during lowering.
- `struct Name` — named product type declared at module level; fields are typed
  and named; layout is engine-determined (see Struct Types section)
- `Optional<T>` — optional value; T may be any non-Optional type; primary use
  case is optional resource slots (e.g. emissive texture); see Optional section
- bundle — anonymous multi-value group — see Bundles section

Type annotations on arithmetic/logic result registers are optional and verified
when present. Casts are their own instruction; inference cannot cross a cast
boundary.

The set of types eligible to appear at shader interface positions (vertex
inputs, interpolants, uniform resources, storage buffer elements) is exported by
the engine. The IR validator checks shader declarations against the engine's
type registry; types that are valid inside a function body may not be valid at
an interface boundary (e.g. `bool` cannot be a vertex input).

At the binary IR level, integer types are encoded as a `(width: u32, signedness:
bool)` pair in the type table. The declared width N is stored exactly — not any
machine container width.

Vector types are fully concrete (`vec3<f32>`, not generic). The textual IR can
include swizzle pseudo-ops (`.x`, `.xy`, etc.) as authoring convenience — the
text→binary pass lowers them to `extract` (component by index → scalar) and
`shuffle` (component subset / reorder → vector). The binary IR, executor, and
emitter never see swizzle syntax. The textual IR does not need to be 1:1 with
the binary IR; pseudo-ops are fine.

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
time — `const ui<4> 16` is an error (16 does not fit in 4 bits; `ui<4>` has no
alias and must be written in full). Out-of-range is always an error, never
silent truncation.

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
`const` instructions followed by `construct`. The emitter can fold an all-const
`construct` into a single constant composite.

### Instruction Immediates

Some instruction arguments are integer immediates embedded in the instruction
encoding — not registers. A runtime register cannot be passed in an immediate
position. Textual IR uses bare integer literals for immediates; they are not
desugared to `const` instructions.

- `extract` — component / field index. On structs and bundles, fields have
  different types, so the index must be a compile-time constant for type
  resolution. On vectors and matrices, the component type is uniform; runtime
  indexing is not currently supported and may be addressed by a separate
  instruction (see Open Questions).
- `shuffle` — component index list. The output vector length (and therefore
  result type) is determined by how many indices are listed, which must be known
  at compile time. Runtime index values are not currently supported and may be
  addressed by a separate instruction (see Open Questions).
- `switch` — case key list. The keys are the values the selector is compared
  against; they form a compile-time jump table.

---

## Bundles (Multi-Value Groups)

Used for function call arguments and multi-value return values. Bundles are
anonymous and opaque in the IR — they are not named types.

Assigning a call result to a typed register is sugar for an implicit
`extract 0` — the LHS type must match the callee's first output. Assigning
to a bundle register (or using `:=` inference) captures all outputs; a
one-element bundle is valid (if redundant). An explicit type annotation is
required to trigger implicit extract — `:=` always infers a bundle. The
type annotation may appear inline on the assignment or via a prior
declaration in the enclosing `out` block. There is no other implicit
unwrapping machinery.

```
// Typed LHS: implicit extract 0
x: vec4<f32> = some_fn bundle(arg0, arg1)

// Bundle LHS: all outputs, extract explicitly
b(0) = some_fn bundle(arg0, arg1)
y := extract b(0) 1
```

Bundle types are resolved from the callee's declared return preamble;
type-checking `extract` is just "look up index N in the callee's return bundle
type." See SPIR-V Emit Notes for lowering details.

---

## Function Structure

Every function has a **preamble** that declares:
- Input bundle (typed by index)
- Output (return) registers (typed, named)
- Local registers (typed or inferred)

```
fn brdf {
  in {
    normal: vec3<f32>
    view: vec3<f32>
    roughness: f32
  }
  out {
    color: vec4<f32>
  }
  body {
    ...
    return   // "output registers are valid", emitter branches to exit block
  }
}
```

Early `return` writes the output registers then branches to the function's exit
block. No phi nodes needed at the return site because output registers are
pre-declared and written exactly once on each path.

Because output registers are declared up front and control flow is structured, a
definite-assignment pass can verify that every control flow path writes all
declared output registers before `return`. The pass only needs to recurse over
the nesting tree — no general dataflow analysis required. This is planned as a
binary IR analysis pass in `psir`.

### Canonical Surface Shape (text IR / future PSL)

The structured textual form is block-oriented and uses explicit `in`, `out`, and
`body` sections. This is the canonical shape for authored shader text.

```
struct Light {
  position: vec3<f32>
  color: vec3<f32>
  intensity: f32
}

fn compute_pbr {
  in {
    albedo_texture: Texture2D
    normal_texture: Texture2D
    orm_texture: Texture2D
    emissive_texture: Optional<Texture2D>
    model_matrix: mat4<f32>
    view_matrix: mat4<f32>
    lights: ReadBuffer<Light>
    light_count: i32
    vertex_position: vec3<f32>
    uv: vec2<f32>
  }
  out {
    out: vec3<f32>
  }
  body {
    ...
  }
}
```

Control flow follows the same explicit-block style:

```
result = if cond {
  out { val: f32 }
  then { ... }
  else { ... }
}

loop_result = loop {
  carries {
    i: i32 = 0 -> i_next
    accum: vec3<f32> = broadcast 0 -> accum_next
  }
  out { accum_next: vec3<f32> }
  body {
    cond = ge i, light_count
    break_if cond -> accum_next
    ...
  }
}
```

`if` / `loop` forms above are syntax-level sugar for the structured IR semantics
in this document (implicit phi via `out`/`carries`, typed branch outputs,
structured merge points).

---

## Control Flow

Structured only — no goto. Basic blocks are implicit; the emitter carves them
from the nesting structure.

### If Expression

```
result = if cond {
  out { val: f32 }
  then {
    val := ...
  }
  else {
    val := ...
  }
}
```

`cond` must be `bool`. `out { ... }` is optional — omit it for pure control flow
(no value produced, no bundle result). When `out { ... }` is present, every
branch must write all declared outputs; the declared outputs become implicit phi
nodes at the merge point.

### Loop

```
loop {
  carries {
    i: i32 = 0 -> i_next
  }
  body {
    i_next := add i 1
    break_if eq i_next 10
  }
}
```

Loops may be labeled to support control transfer to outer loops:

```
loop outer {
  carries {
    i: i32 = 0 -> i_next
  }
  body {
    loop inner {
      carries {
        j: i32 = 0 -> j_next
      }
      body {
        if abort_outer {
          break outer
        }
        i_next := add i 1
        continue outer
      }
    }
    i_next := add i 1
  }
}
```

Unlabeled `break` / `break_if` / `continue` / `continue_if` target the innermost
enclosing loop. Labeled forms target a specific enclosing loop: `break <label>`,
`break_if <cond> <label>`, `continue <label>`, and `continue_if <cond> <label>`.

Loops are infinite by default. `break` / `break_if` and `continue` /
`continue_if` instructions placed anywhere in the body control exit and
iteration. Carried variables are declared with initial values and next-iteration
registers; they become implicit phi nodes at the loop header. `return` and
`discard` may also appear in the loop body and exit the function or invocation
immediately. The emitter handles structured loop back-edge requirements — this
is an emitter detail with no PSIR-level meaning (see SPIR-V Emit Notes).

Falling off the end of a loop `body` is an implicit `continue`. Explicit
`continue` is only needed for early iteration control; `continue_if` is used for
conditional iteration control.

Well-formed PSIR requires definite assignment at loop control edges:

- Before any `break` that exits a loop with declared `out` values, all loop
  outputs must be initialized on that control path.
- Before any explicit `continue` or `continue_if`, all carry next-values for the
  targeted loop must be initialized on that control path.
- On implicit continue (fallthrough at end of `body`), all carry next-values
  must also be initialized.
- A labeled `break`/`break_if`/`continue`/`continue_if` target must resolve to
  an enclosing loop label; unresolved labels are compile errors.
- Duplicate loop labels in overlapping scope are compile errors.

These are compile-time validation rules enforced by the compiler.

### If-Some Expression

```
result = if_some emissive_opt as tex {
  out { c: vec4<f32> }
  some {
    c := sample tex uv
  }
  none {
    c := construct vec4<f32> 0 0 0 0
  }
}
```

Structured optional branch. `tex` is bound to the inner value and is in scope
only inside the `some` block — it cannot be used in `none`. This is the only way
to extract an Optional's inner value; there is no standalone `unwrap`. Every
branch must write all declared output registers. The emitter lowers `if_some`
identically to `if`, extracting the inner value into a register at the top of
the `some` block.

Because `none` is an explicit block, callers can place `discard` there to get
trap-on-absent behaviour in fragment shaders — no special language mechanism
needed. There is no equivalent termination instruction for vertex shaders in
standard SPIR-V; trap-on-absent in a vertex `none` block has no clean
implementation and should be avoided.

### Switch

```
b(0) = switch selector {
  out {
    val: f32
    index: u32
  }
  case 0 {
    val := ...
    index := ...
  }
  case 1 {
    val := ...
    index := ...
  }
  default {
    val := ...
    index := ...
  }
}
```

`selector` must be an integer type. Cases are integer-literal-keyed structured
blocks. No fallthrough. Every case must write all declared output registers. The
binary IR stores a `[(literal, offset, instruction_count)]` jump table after the
switch opcode.

---

## Instruction Set

### Arithmetic
Scalar and component-wise vector/matrix operands unless noted.

- `add`, `sub`, `mul`, `div`, `rem` — binary; integer or float
- `neg` — unary negate; signed integers and floats only (error on unsigned)
- `mat_mul` — matrix × matrix or matrix × vector; dispatches to the correct
  SPIR-V op based on operand types
- `add_sat`, `sub_sat`, `mul_sat` — saturating binary arithmetic; integer only;
  clamps to the declared type's representable range instead of wrapping.
  Unsigned `ui<N>`: clamps to `[0, 2^N − 1]`. Signed `i<N>`: clamps to
  `[−2^(N−1), 2^(N−1) − 1]`.
- `neg_sat` — saturating unary negate; signed integers only; differs from `neg`
  only at the signed minimum (`neg_sat i8 -128` → `127`; `neg` would wrap to
  `-128`). Error on unsigned.

For all binary integer ops (`add`, `sub`, `mul`, `div`, `rem`, `and`, `or`,
`xor`, `add_sat`, `sub_sat`, `mul_sat`, `shl`, `shr`, `asr`): when operands
share the same signedness but differ in width, the narrower operand is
implicitly widened to the wider type; the result has the wider type. Mixed
signedness is always an error — use explicit `cast` or `cast_sat` first.

### Comparison (produce `bool` or `vec<bool>`)

- `eq`, `ne` — equality
- `lt`, `le`, `gt`, `ge` — ordered comparison; integers or floats

### Logical and Bitwise

`and`, `or`, `not`, `xor` dispatch on operand type:
- `bool` operands → logical operation
- integer operands → bitwise operation

There is no separate naming — type inference determines the operation. `xor` on
`bool` is logical exclusive-or, filling a gap absent in the original spec.

- `shl` — shift left; both operands must be integers; shift amount may be any
  integer width
- `shr` — logical shift right; zero-fills high bits; natural for unsigned
  integers; shift amount may be any integer width
- `asr` — arithmetic shift right; sign-fills high bits; natural for signed
  integers; shift amount may be any integer width

### Conversion

- `cast` — explicit type conversion between numeric scalars; the only op that
  can change scalar kind; inference does not cross cast boundaries. Does not
  accept `bool` as source or target — use `select`/`ne` instead.
  - *Integer width change* (all well-defined, no UB):
    - *Narrowing* (either signedness): high bits discarded, low N bits retained,
      reinterpreted in the target signedness.
    - *Widening unsigned* (`u8` → `u32`): zero-extension.
    - *Widening signed* (`i8` → `i32`): sign-extension.
  - *Integer → float* (`i32` / `u32` → `f32`): nearest representable value;
    signed and unsigned variants are distinct.
  - *Float → integer* (`f32` → `i32` / `u32`): truncates toward zero;
    out-of-range result is implementation-defined.
  - *Float width change* (`f32` → `f16` or `f16` → `f32`): precision conversion.
- `cast_sat` — saturating cast; clamps to the target type's representable range
  before converting; integer→integer only (float conversions use `cast`).
  Examples: `cast_sat i8 (i32 200)` → `127`; `cast_sat u8 (i32 -1)` → `0`.

Comparisons (`eq`, `ne`, `lt`, `le`, `gt`, `ge`) are strict — both register
operands must share the exact same declared type. Literals in comparison
position infer their type from the typed peer operand (`eq u8 0` is valid; `eq
u8 u32` is an error). Float widths are never implicitly promoted — use `cast`
first.

### Composite

- `broadcast` — splat one scalar value across all components of a vector. Result
  type must be `vec2<T>`, `vec3<T>`, or `vec4<T>` and is inferred from context
  (usually LHS annotation). Operand type must match the vector element type
  after literal inference. Examples: `v: vec3<f32> = broadcast 0`, `m: vec4<i32>
  = broadcast x`.
- `construct` — build a vector or matrix from component registers in declaration
  order: `construct vec4<f32> x y z w`. For structs, fields are specified by
  name in brace syntax (order independent; resolved to indices at text→binary
  time):
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
- `extract` — extract one sub-element by constant index; on a vector: yields a
  scalar; on a matrix: yields the column vector at that index; on an array or
  struct: yields the field/element. Structs and bundles share the same op —
  structs are named bundles with layout semantics. In the textual IR, `extract
  light pos` is sugar for `extract light 0` (resolved at text→binary time).
- `shuffle` — reorder / subset vector components by constant index list
- `extract_row` — extract row i from a matrix; yields a vector; more expensive
  than `extract` because matrices are column-major — decomposes to per-column
  scalar `extract`s + `construct`
- `transpose` — transpose a matrix; yields a matrix with rows and columns
  swapped

### Textures

- `sample tex pos [layer] [mode]` — sample a texture; `pos` type is
  determined by the texture type (`f32` for 1D, `vec2<f32>` for 2D,
  `vec3<f32>` for 3D); `layer` is an integer index present only for array
  texture types; `mode` defaults to `LinearClamp`. Implicit LOD —
  fragment shader only.
- `sample_lod tex pos [layer] lod [mode]` — explicit LOD level (`f32`)
- `sample_grad tex pos [layer] grad... [mode]` — explicit derivatives,
  one per texture dimension (e.g. `ddx ddy` for 2D, `ddx ddy ddz` for
  3D), each a scalar or vector matching the dimensionality of `pos`
- `texel_fetch` — integer-coordinate fetch, no sampler; optional LOD argument
  (i32), defaults to mip 0 if omitted

### Read Buffers

- `load_elem buf idx` — load one element of `ReadBuffer<T>` at runtime index
  `idx`; result type is T; index may be any integer type

### Optional

- `some val` — wrap a value in `Optional<T>`; T inferred from `val`
- `none Optional<T>` — absent value; T must be stated explicitly (no operand to
  infer from): `x := none Optional<Texture2D>`
- `is_some opt` → `bool` — test presence without branching; useful when the bool
  needs to be stored or passed to a helper

### Math Intrinsics

- `dot`, `cross` — dot / cross product
- `normalize`, `length`, `distance` — vector geometry
- `reflect`, `refract` — optics
- `sqrt`, `rsqrt`
- `abs`, `sign`
- `min`, `max`, `clamp` — work on all numeric types (integers and floats);
  signed and unsigned integers use the appropriate ordered comparison
- `mix` — linear interpolate (GLSL `mix` / HLSL `lerp`); float types only
- `floor`, `ceil`, `round`, `trunc`, `fract`
- `pow`, `exp`, `exp2`, `log`, `log2`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan2`

Transcendental precision policy: Transcendental functions (for example `sqrt`,
`rsqrt`, `pow`, `exp`, `exp2`, `log`, `log2`, and trig functions) are semantic
operations whose exact rounding and implementation are target-defined. PSIR does
not mandate bit-exact results; runtime/target implementations may produce
different numerical results. For workflows that require stronger numeric
guarantees, see the Open Questions / Future Work section for `_precise` variants
(software-only semantics and implementation notes).

### Select

- `select cond true_val false_val` — both operands always evaluated (not a
  branch). `bool` condition: selects between two scalars. `vec<bool>` condition:
  component-wise select between two vectors of the same shape — each output
  component is chosen independently by the corresponding condition component.

### Fragment-Only

- `dpdx`, `dpdy` — partial screen-space derivatives
- `discard` — terminate the fragment invocation; no result (see SPIR-V Emit
  Notes for target-dependent lowering)

### Control Flow

- `return` — marks output registers valid; emitter branches to function exit
  block
- `break` — unconditionally exit the enclosing loop. Labeled form (`break
  label`) exits the specified enclosing loop.
- `break_if` — conditional break; exits the loop only when the condition is true
  (`break_if cond`). Labeled form (`break_if cond label`) targets the specified
  enclosing loop.
- `continue` — unconditionally begin the next loop iteration. If control reaches
  the end of a loop body, iteration continues implicitly, so trailing `continue`
  is redundant. Labeled form (`continue label`) continues the specified
  enclosing loop.
- `continue_if` — conditional continue; continues only when the condition is
  true (`continue_if cond`). Labeled form (`continue_if cond label`) targets the
  specified enclosing loop.
- `call` — call a function; result type depends on the callee's `out:`
  declaration: a single named struct `out:` yields that struct type directly
  (`light := call get_light idx`); any other case yields a bundle (`b(0) = call
  fn ...`)
- `if_some` — structured optional branch; see If-Some Expression section

---

## Module Structure

A module is the top-level compilation unit. It contains:

- Optional module-level type declarations (`struct`)
- Function declarations (`fn`)
- Explicit entry-point declarations (`entry_point`) that bind a stage +
  interface name to a function

### Struct Types

Structs are declared before all function definitions:

```
struct Light {
  pos: vec3<f32>
  color: vec3<f32>
  radius: f32
}
```

Structs are named bundles with layout semantics. `extract` and `construct` work
on structs exactly as they do on bundles — no new composite ops. In the textual
IR, `extract light pos` is sugar for `extract light 0` (field name resolved to
index at text→binary time). Field names are semantic (not debug-only): they are
encoded as integer field IDs in the binary IR type table entry `[(field_id: u32,
type_index: u32)]`, and the engine uses them at pipeline compile time to remap
shader field accesses to the CPU-side canonical buffer layout.

Field byte layout is engine-determined. The first-pass strategy is: the CPU side
uses a fixed canonical layout; the engine emits SPIR-V access-chain remappings
at load time. Packing optimisation is deferred. Shader authors never write byte
offsets or padding.

Order semantics: the compiler resolves struct field names to numeric indices by
declaration order. Declaration order is significant — a shader that declares a
struct intended to match an engine-provided interface struct must use the same
field names, types, and order, or text→binary lowering will reject the module.

Optional remapping: we may optionally permit order mismatches by treating the
engine interface as the canonical source-of-truth and remapping shader-declared
fields to the engine order during text→binary lowering when names and types
match. This behavior will be opt-in (lowering flag or compiler mode) and will
emit warnings when remapping is performed to help authors avoid accidental
mismatches. Explicit per-field `field_id` annotations in shader text are
deferred — we will not require or rely on shader-authored field ids for the
foreseeable future.

Functions are declared with `fn <name>`. They contain typed `in`, `out`, and
`body` blocks. Functions are stage-agnostic by default and can be called as
helpers.

Entry points are declared separately and explicitly map a function to a shader
stage and a named interface contract name.

Entry-point argument declaration: every entry point must declare the set of
interface slots it consumes or produces (inputs and outputs) as part of the
`entry_point` block. Each declared interface slot must map to a corresponding
function parameter or output. The `->` mapping may be omitted only when the
interface slot name exactly matches the function parameter or output name;
omitting the declaration or failing to provide the corresponding function
parameter/output is a compile-time error. The compiler uses the declared mapping
to validate stage linkage, type compatibility, and to perform layout assignment.

Note on textual vs binary representation: in authored shader text the
`entry_point` declares interface slots by name (as shown below). During the
text→binary lowering pass the compiler resolves those names against the
engine-provided interface descriptor and emits a binary `entry_point` record
that references the engine-assigned numeric slot ids. Emitters and runtime
pipeline setup therefore work from numeric `slot_id` values in the binary
format; textual authoring remains name-based for readability.

Named interface contracts are not shader-authored declarations. They are
engine-generated metadata (renderer interface descriptors). During text→binary
lowering, the compiler resolves entry-point `interface:` names against those
engine-provided descriptors and emits validation errors for mismatches.

```
struct Light {
  position: vec3<f32>
  color: vec3<f32>
  intensity: f32
}

# illustrative only: this interface is engine-provided and not present
# in production shader text; shown here for clarity. This syntax is
# preliminary and not meant to be final
interface ScenePBR {
  id: 1

  // slot_id <name> <direction> <type>
  slot 1  albedo_texture      in  Texture2D
  slot 2  normal_texture      in  Texture2D
  slot 3  orm_texture         in  Texture2D
  slot 4  emissive_texture    in  Optional<Texture2D>
  slot 5  model_matrix        in  mat4<f32>
  slot 6  view_matrix         in  mat4<f32>
  slot 7  lights_buffer       in  ReadBuffer<Light>
  slot 8  light_count         in  i32
  slot 9  vertex_position     in  vec3<f32>
  slot 10 uv                  in  vec2<f32>
  slot 11 frag_color          out vec4<f32>
}

fn compute_pbr {
    in {
      albedo_texture: Texture2D
      normal_texture: Texture2D
      orm_texture: Texture2D
      emissive_texture: Optional<Texture2D>
      model_matrix: mat4<f32>
      view_matrix: mat4<f32>
      lights: ReadBuffer<Light>
      light_count: i32
      vertex_position: vec3<f32>
      uv: vec2<f32>
    }
    out {
        out: vec4<f32>
    }
    body {
        ...
    }
}

entry_point main_fragment : compute_pbr {
  stage: fragment
  interface: ScenePBR

  # explicit mappings for all interface slots (required)
  albedo_texture
  normal_texture
  orm_texture
  emissive_texture
  model_matrix
  view_matrix
  lights_buffer -> lights
  light_count
  vertex_position
  uv
  frag_color -> out
}
```

In production shader text, the `interface ScenePBR { ... }` block is not
present; it is shown here only to illustrate the engine-provided contract that
`interface: ScenePBR` refers to during lowering.

Helper functions remain ordinary `fn` declarations without stage or interface
metadata. They receive all dependencies through explicit parameters.

At pipeline compile time, the engine resolves the named interface to its
engine-generated descriptor (available uniforms, storage buffers, external
inputs, external outputs), validates the entry-point mapping, and performs final
layout assignment.

Multiple interface contracts can coexist in the engine registry (for example,
`ScenePBR`, `UI`, `PostFX`) so different rendering domains have distinct
contracts while sharing helper functions.

### Stage Link Mapping

Vertex and fragment entry points are linked at pipeline compile time.

- Default mode: fragment `in` names match vertex `out` names directly.
- Explicit mode: the pipeline composition descriptor may provide a `links` map
  to connect fragment input names to differently named vertex outputs. This
  mapping is engine-side metadata, not embedded in shader source.

Example:

```
pipeline_link {
    vertex: main_vertex
    fragment: main_fragment

    links {
        frag_uv <- vert_texcoord
        frag_n <- world_normal flat
    }
}
```

`links` rules:

- Left side is the fragment function input name.
- Right side is the vertex function output name.
- Each fragment stage-local input must be linked exactly once (by explicit map
  or same-name fallback).
- A vertex output may feed multiple fragment inputs only if all target types and
  interpolation qualifiers are compatible.
- Interpolation qualifier is part of the interpolant contract on both sides:
  vertex `out` and fragment `in`.
- Integer and boolean interpolants require `flat` on both sides.
- Qualifier mismatch is a compile-time link error.
- If `links` is present, it takes precedence for listed inputs; unlisted inputs
  fall back to same-name matching.

### Location Assignment

All locations are implicit — the engine and compiler assign them; shader authors
never write `location(N)`.

- **Interface external inputs** — declared in the engine interface contract as
  `external_in <name>: <type>`. The entry point maps each interface name to a
  function input variable.
- **Interface external outputs** — declared in the engine interface contract as
  `external_out <name>: <type>`. The entry point maps each interface name to a
  function output variable.
- **Engine-side slot class is not shader syntax** — whether an `external_in` is
  backed by uniform data, storage-backed data, vertex attributes, or built-in
  pipeline values is an engine contract concern, not a shader-language concern.
- **Vertex stage input rule** — vertex shader inputs are always sourced through
  interface `external_in` slots. There is no separate vertex `input(...)`
  semantic class.
- **Vertex → fragment interpolants** — fragment shader function inputs are the
  only stage-local `in` semantics and must match named outputs produced by the
  linked vertex shader selected by the pipeline. Matching is by pipeline-level
  explicit `links` map when present, otherwise by semantic name. Types must be
  compatible. Interpolation qualifiers are part of the type contract and must
  match between producer and consumer. Any mismatch is a compile-time link
  error. The compiler assigns locations implicitly and enforces `flat` where
  required.
- **Built-ins** — `external_in`/`external_out` names are resolved by the emitter
  via a fixed table to SPIR-V `BuiltIn` decorations when the engine contract
  marks the slot as a built-in.

Standard `external_in` names:
- `position: vec3<f32>` — vertex attribute, object-space position
- `normal: vec3<f32>` — vertex attribute, object-space normal
- `tangent: vec3<f32>` — vertex attribute, object-space tangent
- `texcoord0: vec2<f32>` — vertex attribute, primary UV set
- `texcoord1: vec2<f32>` — vertex attribute, secondary UV set
- `color0: vec4<f32>` — vertex attribute, vertex color
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

The engine unpacks all vertex attribute data to these canonical types before the
shader sees it — no packed formats, no raw bytes in PSIR. New names are added to
both the engine and IR together as needed.

### Binary IR Form

Every binary module begins with a fixed header containing a magic number and a
format version. The version field allows decoders to reject incompatible formats
and supports maintaining older decoders alongside newer ones. The exact header
layout is deferred to implementation, but the version field is a first-class
design commitment — it will always be present and always be at a stable offset
in the header.

In the binary IR, named types become indexed type entries in a module-level type
table. Resources become indexed entries in a resource table
(`interface_slot_id`, type index, flags). Vertex input semantics are stored as
interface slot IDs. Functions are ordered; entry points carry a stage tag.
Location assignments are stored as resolved indices in the binary IR after the
engine/compiler pass runs.

Instructions are variable-width but self-describing: the opcode is a u16. For
most instructions the operand count is fixed by the opcode. The two exceptions
are `construct` and `call`, which encode an explicit operand count immediately
after the opcode. A decoder never needs to consult the type or function table to
determine instruction boundaries. No alignment requirements — instructions and
functions are packed sequentially and read linearly.

---

## Resources and Layout

Resources are declared in named interface contracts and referenced by entry
points through explicit binding mappings. Interface contracts are engine-
generated metadata, not part of shader source. No binding indices or frequency
annotations appear in shader text — the engine already knows frequency, binding
policy, and layout for every named interface slot from its registry. The shader
declares which interface it targets and how interface slots map to function
variables.

Interface contracts may classify slots by engine binding class (for example:
uniform-backed data, storage-backed data, vertex attributes, built-in inputs,
built-in outputs). That classification is engine metadata used for backend
binding and lowering. Shader text continues to reference typed values through
`external_in` / `external_out` mappings at entry points.

The engine assigns concrete bindings at pipeline creation time. It may map
resources to descriptor sets, push constants, or other mechanisms as its
renderer policy dictates (e.g. bindless, multi-draw indirect). The IR is
unaffected by these decisions.

Engine decision on selector timing: the engine/runtime chooses whether a
resource selector is resolved before PSIR (bindful/specialized per-draw) or left
dynamic and lowered into descriptor indices during backend lowering. If the
selector is resolved early, PSIR never contains descriptor indices and
non-uniform descriptor-indexing concerns do not apply. If the selector is kept
dynamic, PSIR should express semantic dynamic selection (for example, "sample
material Albedo/Normal/ORM for current material"); lowering must handle
non-uniform analysis/propagation and emit target-specific descriptor indexing
and decorations. In all cases PSIR avoids encoding set/binding mechanics — it
captures only whether selection is static or dynamic so the backend can choose
bindful specialization or descriptor-indexed lowering.

Promotion decisions are engine-side policy. The engine may analyze rendergraph
usage (producer/consumer lifetimes, access frequency, fan-out, and stage
visibility) to decide which interface slots should be promoted to faster access
paths (for example push constants) versus left in descriptor-backed storage.
These decisions do not change shader source or IR semantics.

Shader text carries no optimization or placement hints. Authors provide only
semantic intent (types, dataflow, control flow, stage linkage). Backend choices
such as descriptor vs push-constant placement, promotion, residency strategy,
and packing are engine/toolchain decisions.

The layout engine is conservative by default: everything spills to uniform
buffers. Push constant packing is an optimization applied after correctness is
established.

Push constants follow `std430` layout rules (alignment = member size, vec3
footgun = 16-byte alignment). The layout engine owns this logic once; no shader
author needs to think about it.

Resources become regular typed function inputs after entry-point/interface
binding resolution. Helper functions continue to receive resources only through
explicit arguments — no implicit global resource access.

All interface slot names follow the same model: textual names in source are
resolved during text→binary lowering to integer IDs from the engine-generated
interface descriptor file. The binary IR stores only those IDs.

The engine-generated interface descriptor file defines stable integer IDs for
interfaces (`interface_id`) and for every name within each interface
(`interface_slot_id`). The compiler resolves names against that file during
text→binary lowering and emits errors for unknown or mismatched names. The
compiler also matches stage-link semantics and assigns shared `Location` indices
where applicable. The emitter maintains a fixed table mapping built-in
`external_in`/`external_out` slot IDs to SPIR-V `BuiltIn` decorations. In all
cases the mapping from authored names to runtime IDs is owned by the
engine/toolchain boundary, not the IR.

---

## Targets and Backends

On desktop, PSIR is lowered to SPIR-V at runtime. On other platforms —
consoles with fixed hardware or mobile where runtime memory and compile
time are constrained — the PSIR→target lowering may instead be performed
at asset bake time (AOT). This is an engine policy decision, not a PSIR
concern.

## SPIR-V Emit Notes

- The emitter accepts a SPIR-V target environment (version + capability set) as
  a configuration parameter. The IR is version-agnostic; the emitter selects ops
  based on this config (e.g. `discard` → `OpTerminateInvocation` on SPIR-V 1.6 /
  Vulkan 1.3, `OpKill` otherwise)
- Integer types carry their exact declared width N to the emitter; container
  width selection, capability requirements, and any fallback emulation are
  emitter decisions not specced in this document.
- Saturating operations (`add_sat`, `sub_sat`, `mul_sat`, `neg_sat`, `cast_sat`)
  have no native SPIR-V equivalent for integers; the emitter synthesizes the
  correct clamp-after-op or clamp-before-convert sequence. This is expected to
  be the case for all current targets — it is an explicit emitter
  responsibility, not a gap in the IR.
- `f16` arithmetic: emitter uses native `Float16` ops when the target supports
  them; otherwise each op is wrapped in `OpFConvert` to/from f32 (promote → op →
  demote).
- `cast` numeric conversions: integer width changes lower to
  `OpSConvert`/`OpUConvert`; integer→float to `OpConvertSToF`/`OpConvertUToF`;
  float→integer to `OpConvertFToS`/`OpConvertFToU`; float width change to
  `OpFConvert`.
- Bundles lower to `OpTypeStruct` (type) / `OpCompositeConstruct` (construction)
  / `OpCompositeExtract` (extraction).
- `xor` on `bool` operands lowers to `OpLogicalNotEqual` — SPIR-V has no
  `OpLogicalXor`.
- Struct types lower to `OpTypeStruct`. Field names are not present in SPIR-V;
  the emitter uses indices only. The engine emits access-chain remappings
  (`OpAccessChain`) to bridge the shader's declared field order to the CPU-side
  canonical buffer layout.
- `ReadBuffer<T>` lowers to an `OpTypeRuntimeArray` element wrapped in an
  `OpTypeStruct` block decorated `BufferBlock` (SPIR-V 1.3) or with
  `StorageBuffer` storage class (SPIR-V 1.4+); `load_elem` lowers to
  `OpAccessChain` + `OpLoad`.
- `Optional<T>` default lowering: `(bool, T)` register pair; `is_some` reads the
  bool; `if_some` reads T into the bound register at the top of the `some`
  block. Optimized lowering for `Optional<ui<N>>` / `Optional<i<N>>` when N <
  container width: emitter packs the presence flag into a spare high bit of the
  container register.
- `rspirv` handles opcode encoding, module structure, type deduplication
 - Non-uniform descriptor indexing: when the backend lowers dynamic selectors
   into concrete descriptor-array indices, the emitter must ensure SPIR-V
   non-uniform signaling is present at the use sites. Practically this means the
   lowering path (either in `psir-spirv` or in `psir-engine` lowering) must run
   a non-uniform taint/propagation pass from seeded dynamic selectors (flat
   vertex inputs, per-instance/material IDs, multi-draw indices) to any register
   values used for descriptor indexing. The emitter should decorate the index or
   apply the equivalent target decoration (for example the descriptor-indexing
   extension's NonUniform decoration) at the access site (index operand of an
   access or sampling op) and enable any required SPIR-V
   capabilities/extensions. If PSIR never contains indices (selection resolved
   earlier), this responsibility moves to the marshalling/lowering layer that
   materializes the indices.

- Bindful vs bindless and DCE: the bindless lowering path (descriptor arrays
  with runtime indices) is the primary driver for non-uniform handling and the
  related emitter complexity. For the bindful/specialized path the plan is to
  conservatively emit sampling/access ops for declared textures, run PSIR
  dead-code-elimination and reachability/taint analyses, and remove samples
  that cannot occur in practice. When DCE cannot remove many declared samples
  (for example more than 4), the compiler will emit a warning so authors and
  the engine can consider specialization (per-material/pipeline variants) or
  alternative lowering strategies. This approach is safe and simple: it
  avoids non-uniform descriptor-indexing when the engine specializes early,
  while providing telemetry and a controlled fallback for genuinely dynamic
  bindless workloads.
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

Integer registers are stored internally as `u64` / `i64` and masked to N bits on
every write. This simulates exact wrap-around and saturation semantics for any
declared width N ≤ 64 without requiring native types for every width.

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
- `TextureArray3D`: Vulkan has no 3D array image type, so this has no direct
  hardware mapping. Not planned; revisit if a concrete use case arises.
- `SamplingPolicy` depth/stencil variants: reading depth or stencil as a
  scalar channel requires platform-specific image view aspects and extensions
  (e.g. stencil sampling on Vulkan needs extension support). Whether these
  belong as policy variants or as an engine-level concern is unresolved.
- Transcendental `_precise` variants: Precise variants are an open design
  question. If provided, `_precise` forms (for example `exp_precise`,
  `rsqrt_precise`) must provide stronger numeric guarantees and therefore will
  need software or higher-precision implementations rather than relying on
  target fast intrinsics alone. A few important constraints and pitfalls to
  cover when designing these variants:
  - **Software implementation required:** implementing substantially higher
    precision or correctly-rounded results will generally require software
    routines (or explicit high-precision lowering) rather than emitting a single
    target transcendental instruction.
  - **Precision leakage danger:** performing some computations in higher
    precision and then failing to round or truncate intermediate results before
    returning to the shader's declared float width can silently change numerical
    behavior and ABI expectations. Lowering must explicitly manage when excess
    precision is produced and must document the rounding step.
  - **Performance and ABI:** software-accurate paths are orders of magnitude
    slower and may change calling conventions or register pressure; they should
    be opt-in and clearly documented.
  - **Lowering responsibility:** the text→binary or emitter stage must select
    the `_precise` implementation strategy (software call vs higher- precision
    hardware) and ensure correct rounding and decoration so results do not
    accidentally retain extra precision across operations.

- `f64` — 64-bit float; primary use case is compute-heavy / scientific workloads
  (physics, numerical methods); requires `Float64` Vulkan feature and has no
  viable full-fidelity emulation; co-deferred with compute shader entry points
- Full PSL (macros, modules, includes, richer ergonomics) is deferred. Core
  structured surface syntax (`struct`, `fn`, `if`, `loop`, `entry_point`) is now
  part of this design.
- `switch` as sugar over nested if/else at the IR level vs. direct `OpSwitch`
  emit — current plan: direct `OpSwitch` from day one
- `WriteBuffer<T>` / `ReadWriteBuffer<T>` — writable and read-write storage
  buffers. A recurring pattern: payload buffers can be `WriteBuffer<T>` (e.g.
  draw commands, OIT nodes), but the atomic counter that allocates slots in
  those buffers requires `ReadWriteBuffer<u32>`. Examples: GPU culling writes
  `WriteBuffer<DrawCommand>` but needs `ReadWriteBuffer<u32>` for the draw
  count; OIT writes `WriteBuffer<OITNode>` but needs `ReadWriteBuffer<u32>` for
  head pointers and counters. `WriteBuffer` alone is never sufficient when
  dynamic allocation is involved. Approximate OIT via weighted blending needs
  only render targets and is available now.
- Sum types (tagged unions / enum-with-data) — natural extension once structs
  land; useful for material variant dispatch and similar patterns
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
- Generic-length vectors (and matching matrix families) — currently the IR type
  surface is fixed to `vec2`/`vec3`/`vec4` and `mat2`/`mat3`/`mat4`. Decide
  whether to add a parameterized form (for example `vec<N, T>`) or keep
  fixed-width vector and matrix types as a permanent design choice.
- Dynamic composite access — runtime-indexed `extract` on vectors/matrices and
  runtime-indexed `shuffle`; likely requires separate instructions (e.g.
  `extract_dynamic`, `gather_components`) rather than relaxing the immediate
  constraint on the existing ops
- Optimizer pass (or delegate to `spirv-opt`) — deferred
- Binary format section layout and debug metadata — section-based container with
  optional per-function debug sections; section header format and encoding
  deferred to implementation
- SPIR-V debug info (`NonSemantic.Shader.DebugInfo.100`) — deferred; depends on
  binary format debug section design
