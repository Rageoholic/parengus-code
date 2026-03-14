# Parengus Shader Intermediate Representation (PSIR) — Design Sketch

> Rough design notes from design conversation, 2026-03-14.
> Generated with Claude's assistance.

## Goals

- Engine-managed resource layout (no manual binding/set indices in shaders)
- Automatic push constant packing with uniform buffer spill
- CPU-side reference executor for correctness verification
- Source-level debug info via `NonSemantic.Shader.DebugInfo.100` (RenderDoc)
- SPIR-V as the current sole compilation target (via `rspirv`)

---

## Crate Structure (planned)

| Crate | Role |
|---|---|
| `psir` | IR data definitions — pure data, no engine or compiler deps |
| `psir-spirv` | Pure binary IR → SPIR-V lowering (no engine deps) |
| `psir-engine` | Engine-side backend: layout assignment, calls into `psir-spirv` |
| `psir-compiler` | Frontend: text → binary IR, eventually PSL → binary |
| `psir-executor` | CPU reference executor (correctness oracle) |

`psir-spirv` is depended on by both `psir-engine` (runtime pipeline
compilation) and `psir-compiler` (offline tooling), so it carries no
engine or compiler dependencies — just `psir` + `rspirv`.

---

## Register Model

Typed infinite registers — each register is written exactly once (SSA),
has a type, and lives for the duration of its scope. No register reuse,
no aliasing.

Every function has its own register namespace (no clobbering between
caller and callee).

### Textual Syntax

```
// Typed declaration — register index with explicit type
f32(0) := load_from_uniform(camera)

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

---

## Types

| Type | Notes |
|---|---|
| `f32` | 32-bit float |
| `f16` | 16-bit float. Executor upcasts to f32, operates, downcasts on store. Emitter wraps ops in `OpFConvert` pairs. |
| `u32` | 32-bit unsigned integer |
| `i32` | 32-bit signed integer |
| `bool` | Boolean |
| `vec2<T>`, `vec3<T>`, `vec4<T>` | Vector; vec3 has 16-byte std430 alignment (footgun) |
| `mat2<T>`, `mat3<T>`, `mat4<T>` | Matrix; column-major (matches vek and engine convention) |
| struct | Anonymous, field-indexed |
| array | Fixed-size |
| `Texture2D` | Sampled texture |
| `Sampler` | Sampler or combined image sampler |
| bundle | Anonymous multi-value group — see Bundles section |

Type annotations on arithmetic/logic result registers are optional and
verified when present. Casts are their own instruction; inference cannot
cross a cast boundary.

At the binary IR level, vector types are fully concrete (`vec3<f32>`, not
generic). The textual IR can include swizzle pseudo-ops (`.x`, `.xy`,
etc.) as authoring convenience — the text→binary pass lowers them to
`extract` (component by index → scalar) and `shuffle` (component subset
/ reorder → vector). The binary IR, executor, and emitter never see
swizzle syntax. The textual IR does not need to be 1:1 with the binary
IR; pseudo-ops are fine.

---

## Constants and Literals

### `const` Instruction

In the binary IR, every constant value is produced by a `const`
instruction that writes a typed scalar register:

```
x  := const f32  1.0
i  := const i32  42
b  := const bool true
```

The instruction carries the type and bit pattern of the value. Constants
are ordinary registers — deduplication within a function is a compiler
optimization, not a binary IR requirement.

### Textual IR — Type Inference

In the textual IR, literals are untyped tokens. Their type is resolved
from context, in order:

1. **LHS type annotation** — `x : f32 = const 1.0` → f32 from the
   annotation.
2. **Typed peer operand** — `add f32(0) 1` → `1` infers as f32 from
   `f32(0)`.
3. **Explicit type on `const`** — `x := const f32 1.0` — required when
   neither of the above applies. Error if omitted.

No magic suffixes, no default numeric types. A literal `1.0` in an
integer context is an error; a literal `1` in a float context resolves
to that float type.

Literals in instruction argument position are sugar for an anonymous
`const` register. The text→binary pass emits the `const` instruction
and substitutes its register index.

### Composite Literals

Composite constants (vectors, matrices) are written in the textual IR
as `construct` with literal operands:

```
v : vec4<f32> = construct 0 0 0 1   // scalar types inferred from LHS
```

Each literal desugars to a scalar `const`; the binary IR contains the
individual `const` instructions followed by `construct`. The SPIR-V
emitter recognizes an all-const `construct` and folds it into a single
`OpConstantComposite`.

### Instruction Immediates

Some instruction arguments are integer immediates embedded in the
instruction encoding — not registers. These exist only where the type
system or code generation structurally requires compile-time knowledge:

| Instruction | Immediate argument(s) |
|---|---|
| `extract` | Component / field index |
| `shuffle` | Component index list |
| `switch` | Case key list |

A runtime register cannot be passed in an immediate position. Textual
IR uses bare integer literals for immediates; they are not desugared to
`const` instructions.

---

## Bundles (Multi-Value Groups)

Used for function call arguments and return values. Bundles are anonymous
and opaque in the IR — they are not named types.

```
// Call returning multiple values
b(0) = some_fn bundle(arg0, arg1)

// Extract by index
x := extract b(0) 0
y := extract b(0) 1
```

The emitter maps bundles to `OpTypeStruct` / `OpCompositeConstruct` /
`OpCompositeExtract` in SPIR-V. Bundle types are resolved from the
callee's declared return preamble; type-checking `extract` is just
"look up index N in the callee's return bundle type."

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

Early `return` writes the output registers then branches to the function's
exit block. No phi nodes needed at the return site because output registers
are pre-declared and written exactly once on each path.

Because output registers are declared up front and control flow is
structured, a definite-assignment pass can verify that every control
flow path writes all declared output registers before `return`. The
pass only needs to recurse over the nesting tree — no general dataflow
analysis required. This is planned as a binary IR analysis pass in
`psir`.

---

## Control Flow

Structured only — no goto. Basic blocks are implicit; the emitter carves
them from the nesting structure.

### If Expression

```
result = if cond
  out: val: f32
  then:
    val := ...
  else:
    val := ...
```

Produces a bundle. Every branch must write all declared outputs. The
emitter emits `OpSelectionMerge` + `OpBranchConditional` + `OpPhi` at
the merge block.

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

Carried variables are declared with initial values and update expressions.
The emitter emits `OpLoopMerge` (merge block + continue block) and
`OpPhi` at the loop header for each carried variable.

`break` and `continue` are explicit instructions. `break` branches to the
merge block; `continue` branches to the continue block.

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

Cases are integer-literal-keyed structured blocks. No fallthrough. Every
case must write all declared output registers. The binary IR stores a
`[(literal, offset, instruction_count)]` jump table after the switch
opcode. The emitter maps this to `OpSwitch`.

---

## Instruction Set

### Arithmetic
Scalar and component-wise vector/matrix operands unless noted.

| Op | Notes |
|---|---|
| `add`, `sub`, `mul`, `div`, `rem` | Binary; integer or float |
| `neg` | Unary negate |
| `mat_mul` | Matrix × matrix or matrix × vector; dispatches to the correct SPIR-V op based on operand types |

### Comparison (produce `bool` or `vec<bool>`)

| Op | Notes |
|---|---|
| `eq`, `ne` | Equality |
| `lt`, `le`, `gt`, `ge` | Ordered comparison; integers or floats |

### Logical (`bool` operands)

| Op | Notes |
|---|---|
| `and`, `or`, `not` | Boolean logic |

### Bitwise (integer operands)

| Op | Notes |
|---|---|
| `band`, `bor`, `bnot`, `bxor` | Bitwise logic |
| `shl`, `shr` | Shift left / right |

### Conversion

| Op | Notes |
|---|---|
| `cast` | Explicit type conversion — the only op that can change scalar type. Inference does not cross cast boundaries. |

### Composite

| Op | Notes |
|---|---|
| `construct` | Build a vector, matrix, or struct from component registers |
| `extract` | Extract one sub-element by constant index. On a vector: yields a scalar. On a matrix: yields the column vector at that index. On a struct or array: yields the field/element. |
| `shuffle` | Reorder / subset vector components by constant index list; maps to `OpVectorShuffle` |
| `extract_row` | Extract row i from a matrix; yields a vector. More expensive than `extract` because matrices are column-major — decomposes to per-column scalar `extract`s + `construct`. |
| `transpose` | Transpose a matrix; yields a matrix with rows and columns swapped. |

### Resources

| Op | Notes |
|---|---|
| `load_uniform` | Load from uniform buffer by name (text) or index (binary) |
| `load_input` | Load shader stage input (vertex attribute or fragment interpolant) |
| `store_output` | Write shader stage output |

Matrix resources can carry a `row_major` annotation. The load automatically
inserts a `transpose` so the shader always receives a column-major matrix.
The annotation is declared on the resource, not at the call site — shader
authors never handle the source memory layout directly.

### Textures

| Op | Notes |
|---|---|
| `sample` | Sample with implicit LOD (fragment shader) |
| `sample_lod` | Sample with explicit LOD |
| `sample_grad` | Sample with explicit gradient (`dPdx`, `dPdy`) |
| `texel_fetch` | Integer-coordinate fetch, no sampler. Optional LOD argument (i32); defaults to mip 0 if omitted. Maps to `OpImageFetch`. |

### Math Intrinsics

| Op | Notes |
|---|---|
| `dot`, `cross` | Dot / cross product |
| `normalize`, `length`, `distance` | Vector geometry |
| `reflect`, `refract` | Optics |
| `sqrt`, `inversesqrt` | |
| `abs`, `sign` | |
| `min`, `max`, `clamp` | |
| `mix` | Linear interpolate (GLSL `mix` / HLSL `lerp`) |
| `floor`, `ceil`, `round`, `trunc`, `fract` | |
| `pow`, `exp`, `exp2`, `log`, `log2` | |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan2` | |

### Select

| Op | Notes |
|---|---|
| `select` | `select cond true_val false_val` — choose between two values based on a bool or `vec<bool>` condition. Both operands are always evaluated. Maps to `OpSelect`. Works on scalars and vectors (component-wise). |

### Fragment-Only

| Op | Notes |
|---|---|
| `dpdx`, `dpdy` | Partial screen-space derivatives |
| `discard` | Terminate the fragment invocation; no result. Emitter maps to `OpKill` or `OpTerminateInvocation` based on the configured SPIR-V target environment. |

### Control Flow

| Op | Notes |
|---|---|
| `return` | Marks output registers valid; emitter branches to function exit block |
| `break` | Branch to loop merge block |
| `continue` | Branch to loop continue block |
| `call` | Call a function; result is a bundle |

---

## Module Structure

A module is the top-level compilation unit. It contains named type
definitions, resource declarations, and functions (entry points and
helpers).

```
types:
  Camera:
    view: mat4<f32>
    proj: mat4<f32>

resources:
  camera:     Camera    per_frame
  albedo_tex: Texture2D per_draw
  albedo_smp: Sampler   per_draw

fn vert
  stage: vertex
  in:  pos: vec3<f32>
       uv:  vec2<f32>
  out: clip_pos: vec4<f32>  builtin(position)
       frag_uv:  vec2<f32>
body:
  ...

fn frag
  stage: fragment
  in:  frag_uv: vec2<f32>
  out: color: vec4<f32>  color_out
body:
  ...

fn brdf            // helper — no stage:
  in:  normal: vec3<f32>, view: vec3<f32>
  out: color: vec4<f32>
body:
  ...
```

### Location Assignment

All locations are implicit — the engine and compiler assign them; shader
authors never write `location(N)`.

| I/O | Assignment rule |
|---|---|
| Vertex inputs | Engine matches by name to CPU-side vertex buffer attributes |
| Vertex → fragment interpolants | Compiler assigns when both stages are in the same module; matched by name. Optional `flat` annotation suppresses interpolation (`Flat` decoration in SPIR-V). |
| Fragment color outputs | `color_out` when there is exactly one; `color_out(N)` when there are multiple. Mixing bare `color_out` with indexed siblings is an error. |
| Builtins | `builtin(name)` annotation; no location. Names: `position`, `frag_coord`, `vertex_index`, `instance_index`, `point_size`, `frag_depth` |

### Binary IR Form

In the binary IR, named types become indexed type entries in a module-
level type table. Resources become indexed entries in a resource table
(name, type index, frequency, flags). Functions are ordered; entry
points carry a stage tag. Location assignments are stored as resolved
indices in the binary IR after the engine/compiler pass runs.

---

## Resources and Layout

Resources carry a **update frequency** annotation, not binding indices.
The engine assigns concrete bindings at pipeline creation time.

| Frequency | Typical engine mapping |
|---|---|
| `per_frame` | Descriptor set 0 |
| `per_draw` | Push constants (if fits), else descriptor set 1 |
| `per_object` | Descriptor set 2 |

The layout engine is conservative by default: everything spills to uniform
buffers. Push constant packing is an optimization applied after correctness
is established.

Push constants follow `std430` layout rules (alignment = member size,
vec3 footgun = 16-byte alignment). The layout engine owns this logic once;
no shader author needs to think about it.

Resources are accessed by name in the textual IR; the frontend resolves
names to indices in the binary IR.

```
camera := load_from_uniform(camera)    // named
f32(0) := load_from_uniform(0)         // indexed (binary form)
```

---

## SPIR-V Emit Notes

- The emitter accepts a SPIR-V target environment (version + capability
  set) as a configuration parameter. The IR is version-agnostic; the
  emitter selects ops based on this config (e.g. `discard` →
  `OpTerminateInvocation` on SPIR-V 1.6 / Vulkan 1.3, `OpKill`
  otherwise)
- `rspirv` handles opcode encoding, module structure, type deduplication
- Phi nodes are entirely an emitter concern — never appear in PSIR
- Structured control flow maps directly to SPIR-V's requirements:
  - `OpSelectionMerge` before if/switch branches
  - `OpLoopMerge` (merge + continue blocks) before loop header branch
- Each function gets its own ID namespace (matches SPIR-V per-function scope)
- Debug info via `NonSemantic.Shader.DebugInfo.100` extended instructions
- `spirv-val` should be run on all emitted modules during development

---

## Executor

The executor runs PSIR shaders on the CPU as a correctness oracle. It is
a debugging tool, not used in normal engine operation.

### CSV I/O

The executor accepts RenderDoc CSV exports directly as input (no custom
format). Feed it the input CSV, run the shader, compare output against
the GPU output CSV from the same capture. This makes it straightforward
to reproduce a GPU divergence from a saved capture without re-running the
engine.

CSV format is matched to RenderDoc's actual export format at
implementation time rather than specced upfront.

### Verbosity

Off by default. Two opt-in levels:

| Level | Output |
|---|---|
| `off` | Only final output values (default) |
| `named` | Print each named register as it is assigned |
| `all` | Print every register assignment including anonymous intermediates |

`named` is the practical debugging level for shader logic; `all` is for
diagnosing the emitter or executor itself.

---

## Open Questions / Future Work

- PSL (Parengus Shading Language) surface syntax — deferred, PSIR textual
  form is expressive enough to write shaders directly for now
- `switch` as sugar over nested if/else at the IR level vs. direct
  `OpSwitch` emit — current plan: direct `OpSwitch` from day one
- Storage buffers for light lists (needed for forward+ lighting)
- Compute shader entry points
- Image stores (`store_texel`) for storage images
- Shadow sampling: `sample_compare`, `sample_compare_lod` (depth
  texture + reference value; needed for shadow maps)
- Gather: `gather`, `gather_compare` (4-texel component gather;
  useful for PCF shadow filtering)
- Texture queries: `query_size` (dimensions at mip level),
  `query_levels` (mip count)
- MSAA fetch: `texel_fetch` sample-index argument for multisampled
  textures
- Optimizer pass (or delegate to `spirv-opt`) — deferred
