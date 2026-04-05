# Ideas

Half-baked ideas that aren't ready for the task graph but are worth
remembering. No commitment to implement any of these.

---

## Custom secondary command buffer nesting layer

Vulkan secondary command buffers cannot nest. A custom layer sitting above
the Vulkan API could treat "nested secondaries" as a CPU-side ordering
concept only: each logical nested unit records into a flat Vulkan secondary,
and the layer assembles them into an ordered sequence of peers, inserting
whatever synchronization or state-reset glue is needed between them before
handing the list up to the primary command buffer. Nesting is never visible
to Vulkan — it is purely a CPU abstraction.

The ordering is serial and user-controlled, not a dependency graph — there
is no automatic reordering or culling. Think "command buffer chain" rather
than render graph.

Scopes can nest arbitrarily. The layer can emit glue at scope boundaries —
debug label regions are the obvious case: entering a scope records
`begin_debug_label` into the primary, exiting records `end_debug_label`.
The Vulkan secondary command buffers are the leaves of this scope tree;
the tree itself exists only on the CPU.

This implies that debug label open/close are explicit nodes in the tree,
separate from the nodes that execute command buffers. A label scope cannot
be opened inside a secondary and closed outside it (or vice versa), so
cross-buffer labels must live at the layer level as first-class scope
nodes recorded into the primary. `begin_debug_label` / `end_debug_label`
on `Recorder` remain useful for labeling sub-regions within a single
secondary buffer; these are a different granularity, not a replacement.

For VK 1.0 renderers, render passes would also be nodes in this tree.
Opening a render pass node records `vkCmdBeginRenderPass` into the primary;
child nodes are subpasses; closing it records `vkCmdEndRenderPass`. This
fits the same scope structure as debug labels and command buffer execution
nodes.

Revisit when secondary command buffer support lands.

> Generated with Claude's assistance.
