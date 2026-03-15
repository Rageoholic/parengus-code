---
id: t012
title: Upload system
status: planned
area: renderer
parent: t005
---

Description:

Hybrid upload system. Startup uploads may occur outside the RenderGraph.
Streaming uploads can be expressed as graph passes. Uploader is tied to
the Executor. Depends on: executor_resource_management

Subtasks:

- uploader_interface
- staging_buffer_system
- transfer_queue_usage
- startup_upload_pipeline
- rendergraph_upload_pass
