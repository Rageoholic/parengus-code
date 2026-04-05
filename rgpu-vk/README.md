# rgpu-vk

> **Personal project.** This crate is published for archival and
> visibility purposes; it is not intended for general use and makes
> no API stability guarantees.

Thin RAII wrappers around [Vulkan] objects, built on [`ash`].

[Vulkan]: https://www.vulkan.org/
[`ash`]: https://crates.io/crates/ash

## Object hierarchy

```
Instance
├── Surface<T>
│   └── Swapchain<T>
└── Device
    ├── HostVisibleBuffer / DeviceLocalBuffer
    ├── ShaderModule → EntryPoint → DynamicPipeline
    ├── ResettableCommandPool<Q> → ResettableCommandBuffer<Q>
    ├── TransientCommandPool<Q>  → TransientCommandBuffer<Q>
    └── Fence / Semaphore
```

Each wrapper holds its parent via `Arc` so parents cannot be
destroyed while children are alive.

## Queue capability markers

Command pools and buffers are parameterised by a queue capability
marker type:

| type        | capabilities                     |
|-------------|----------------------------------|
| `Graphics`  | graphics + compute + transfer    |
| `Compute`   | compute + transfer               |
| `Transfer`  | transfer only                    |

Recording methods on [`Recorder`] are gated by trait bounds
(`SupportsGraphics`, `SupportsCompute`) so that draw calls and render
passes cannot be recorded to a transfer-only command buffer at compile
time.

## Design policy

- **Prefer generics over concrete types.** Where a function or method
  can be written generically over a trait, it should be. This applies
  to `Recorder` (generic over `B: RecordableCommandBuffer<Q>`) and to
  buffer/image upload helpers (generic over the recorder type).

- **Traits are not sealed.** `RecordableCommandBuffer<Q>` and the
  capability traits (`SupportsGraphics`, `SupportsCompute`) are fully
  public and implementable by downstream crates. Users who need custom
  pool behaviour can implement these traits for their own types and
  pass them anywhere a standard pool/buffer is accepted.

- **pNext chain policy:** See the "pNext Chain Policy" section in
  this README.

## Naming conventions

| prefix  | meaning                                    |
|---------|--------------------------------------------|
| `raw_*` | accepts or returns a raw `ash::vk` handle  |
| `ash_*` | returns the `ash` wrapper object           |

## License

Mozilla Public License Version 2.0 — see [LICENSE] in the repository
root.

[LICENSE]: https://github.com/Rageoholic/parengus-code/blob/main/LICENSE

## Development Guidelines

- **Coding conventions**
- `#![deny(unsafe_op_in_unsafe_fn)]` is set — all unsafe operations
  inside `unsafe fn` must be wrapped in an explicit `unsafe {}` block.
- Unsafe methods on wrapper types are prefixed with `raw` (e.g.
  `create_raw_surface`). Prefer `unsafe fn` wrappers over exposing raw
  handles directly.

- **Architecture**
- `Instance` wraps the ash Vulkan instance.
- `Surface<T>` holds `Arc<Instance>` and `Arc<T>` for lifetime safety.
- Device selection uses a priority-based fold over physical devices.

- **Verification**

Always use `cargo clippy` (not just `cargo check`) to verify code after
writing it. CI runs clippy with `-D warnings` per package; rust-analyzer
surfaces clippy diagnostics in the editor but some tools (e.g. CI) are
the authoritative source. Run `cargo clippy` locally when preparing PRs.

- **Feature unification (per-crate checks)**

Workspace feature unification can hide missing features. Always verify
individual crates with `cargo check -p <crate>` rather than relying on
a workspace-level check.

- **Device feature structs**

Because we target Vulkan 1.0 + extensions, enabling optional device
features requires filling out a per-extension `VkPhysicalDevice*Features`
struct. When a feature is promoted to core, the extension struct becomes
a type alias for the core struct, so the same code works on both old and
new drivers — using the extension struct on a 1.3 device is valid and
intended by the spec.

**Policy: query first, then pass the result to `DeviceCreateInfo`.**

Before creating the logical device, call `get_physical_device_features2`
with the feature struct(s) you care about chained into a
`VkPhysicalDeviceFeatures2`. The driver fills in which sub-features are
actually supported. Pass those same structs — unchanged — to
`DeviceCreateInfo`. Never hard-code `VK_TRUE`; enabling a feature the
physical device does not report is invalid and will trigger validation
errors.

```rust
// Correct pattern
let mut my_features =
    vk::PhysicalDeviceMyFeatures::default(); // all zeros
let mut query = vk::PhysicalDeviceFeatures2::default()
    .push_next(&mut my_features);
// fills my_features with what the device actually supports
unsafe { instance.get_physical_device_features2(phys_dev, &mut query) };
// pass the filled struct to device creation — do NOT set fields to TRUE
device_create_info = device_create_info.push_next(&mut my_features);
```

**Checking feature support: exhaustive destructure.**

When writing a helper that validates whether all sub-features in a group
are supported, destructure the struct and explicitly bind every boolean
field. Use `_` only for `s_type`, `p_next`, and `_marker`. This gives
compile-time proof that no field was accidentally skipped. See the
helpers in `device.rs` for examples.

### pNext Chain Policy

For this crate prefer the safer rule: never reuse feature structs across
different purposes. Always use distinct struct instances for querying
device-reported feature values and for pushing feature structs into a
`pNext` chain (for example, `VkDeviceCreateInfo::pNext`).

Rationale
---------

Reusing the same struct instance for both query and creation can
accidentally carry pointers or state (notably `p_next`) from a prior
use and lead to subtle validation or driver bugs. Using fresh structs
prevents accidental non-null `p_next` values and makes intent explicit.

Guidance
--------

- Query into temporary structs chained on `VkPhysicalDeviceFeatures2`.
- Copy the reported boolean fields into fresh structs that you then
  place into `DeviceCreateInfo` or other pNext chains. This ensures
  `p_next` is default-null unless you intentionally chain further
  structures.
- If a struct intentionally chains further extension structs via its
  own `p_next`, add a comment explaining why and reference this policy.

Example
-------

```rust
// Query into temporary structs
let mut q_sync2 = vk::PhysicalDeviceSynchronization2Features::default();
let mut q_features2 = vk::PhysicalDeviceFeatures2::default()
    .push_next(&mut q_sync2);
unsafe { instance.get_physical_device_features2(phys_dev, &mut q_features2) };

// Copy reported flags into a fresh struct for device creation
let mut dev_sync2 = vk::PhysicalDeviceSynchronization2Features::default();
dev_sync2.synchronization2 = q_sync2.synchronization2;

// Push the fresh struct into DeviceCreateInfo
let mut device_create_info = vk::DeviceCreateInfo::default();
device_create_info = device_create_info.push_next(&mut dev_sync2);
```

You may still explicitly null `p_next` on structs when appropriate, but
the primary rule for this crate is to avoid reusing feature-struct
instances. Note that freshly-initialised feature structs (`::default()`)
already have `p_next == null`, so explicit nulling is unnecessary when
you create a new struct for device creation.


