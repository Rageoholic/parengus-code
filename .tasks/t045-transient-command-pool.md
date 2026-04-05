---
id: t045
title: "TransientCommandPool and RecordableCommandBuffer trait"
status: done
created: 2026-04-05
updated: 2026-04-05
parent: null
children: []
depends_on: []
blocked_by: []
area: rgpu-vk
---

## Context

`command.rs` only exposes `ResettableCommandPool`/`ResettableCommandBuffer`.
There is no shared trait, no transient-pool variant, and no queue-capability
enforcement on recorded commands. Upload paths in all apps use resettable
pools for one-shot work that would be better served by a transient pool.

## Goal

Introduce `RecordableCommandBuffer<Q>` as the primary trait, `Recorder<Q,B>`
as a RAII recording guard (panics on drop without `end_recording`),
`TransientCommandPool<Q>`/`TransientCommandBuffer<Q>` for pool-reset-freed
buffers, and queue capability markers (`Graphics`, `Compute`, `Transfer`)
that gate recording methods at compile time. Migrate all app upload paths to
`TransientCommandBuffer<Transfer>`.

## Plan

- [ ] Create queue capability markers and traits in `command.rs`
- [ ] Assert in device creation that the graphics queue family
      also supports compute
- [ ] Add `RecordableCommandBuffer<Q>` trait with `on_end_recording`
      and `begin_recording`
- [ ] Add `Recorder<'a, Q, B>` with all recording methods; Drop panics,
      `end_recording` uses `mem::forget`
- [ ] Add `TransientCommandPool<Q>` and `TransientCommandBuffer<Q>`
- [ ] Add `Q` type param to `ResettableCommandPool` and
      `ResettableCommandBuffer`; remove `begin()`/`end()` and old
      recording methods; impl trait
- [ ] Update `buffer.rs` and `image.rs` to accept `&mut Recorder<'_, Q, B>`
- [ ] Update `samp-app`, `samp-app-noext`, `phoenix` call sites;
      migrate upload paths to `TransientCommandBuffer<Transfer>`
- [ ] `cargo clippy` all crates, fix warnings

## Thinking

`Recorder` is generic over `B: RecordableCommandBuffer<Q>` so it can call
`B::on_end_recording` without type erasure. Recording methods carry `Q:
SupportsGraphics` / `Q: SupportsCompute` bounds. Copy and barrier ops are
ungated (all queue types support transfer). `Graphics` implies `Compute` by
device-creation assertion (Vulkan spec guarantees ≥1 family supports both;
real hardware always makes this the graphics family).

# Outcome

Developed the specification. RecordableCommandBuffer was shortened to
Recordable. Recorder is additionally parameterized on the underlying command
buffer to avoid disjoint borrows of random fields and allow the Command Buffer
to finalize its state. We also added a Submittable<Q> trait so that we can
submit using a typed API. Finally, We added HasGraphics, HasCompute, and
HasTransfer, gating their respective methods on the Recorder. This is so that we
can allow third party code to implement their own queue types, which will not be
supported by the default command pools, but can be supported in third party
code. For example, video decode queues don't necessarily have Transfer
operations enabled.

# Future work

In the future, we might also want to add a HasSparseBinding tag and a way for
the recorder to query if the underlying queue family supports certain
operations, returning an augmented subrecorder or something. Then Recorder would
Deref and DerefMut to Subrecorder<Q> while handling the end recording bit
itself. This seems like a reasonable API.

One gap in the current implementation is that we can't move Recorder between
threads. The probably shape of the API is not to make Recorder Send, but to
implement a "Half-recorded" state that would allow us to stop recording on one
thread, send the command buffer, and continue recording on another thread. This
is niche enough that I highly doubt the implementation needs this however. The
current recommended solution is to