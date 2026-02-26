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
    ├── ResettableCommandPool → ResettableCommandBuffer
    └── Fence / Semaphore
```

Each wrapper holds its parent via `Arc` so parents cannot be
destroyed while children are alive.

## Naming conventions

| prefix  | meaning                                    |
|---------|--------------------------------------------|
| `raw_*` | accepts or returns a raw `ash::vk` handle  |
| `ash_*` | returns the `ash` wrapper object           |

## License

Mozilla Public License Version 2.0 — see [LICENSE] in the repository
root.

[LICENSE]: https://github.com/Rageoholic/parengus-code/blob/main/LICENSE
