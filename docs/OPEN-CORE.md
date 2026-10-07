# Sofuu open-core model

Sofuu is **open core**: the runtime you run is fully open source, and the
commercial layer is additional services built around it — never functionality
removed from the core.

## What's open source (MIT)

The entire execution runtime, per [`LICENSE`](../LICENSE):

- Runtime engine (QuickJS, libuv event loop, SIMD kernels)
- CLI (`run`, `chat`, `eval`, `bundle`, `add`, `doctor`, …)
- JavaScript/TypeScript APIs (`sofuu.ai`, agents, tools)
- MCP client and server, HTTP server and client
- Vector operations and all embedding spaces
- Local memory and KV stores, middleware toolchain (bundler, installer, REPL)
- SDK bindings (Swift, Kotlin, C ABI) and their samples

Use, modify, and distribute it freely, commercially or not. Build from source
with `git clone https://github.com/sofuu-runtime/sofuu && make`.

## What's proprietary

1. **The QTSQ format specification** — the design, mathematics, and spec of
   the `.qtsq` / `.qtsw` tensor format, per
   [`LICENSES/QTSQ-FORMAT.txt`](../LICENSES/QTSQ-FORMAT.txt). The runtime code
   that reads and writes it is MIT; the format itself, and independent
   competing read/write implementations outside the Sofuu ecosystem, are not.
2. **Future commercial offerings** — hosted/cloud runtime, enterprise features,
   managed infrastructure, and premium services. These will be separate
   services around the core, announced when they exist. Nothing in the list
   above will be closed to create them.

## Practical consequences

- Binaries distributed from `sofuu.xyz` are built from this MIT codebase.
- `sofuu doctor` verifies the binary you installed against the open source.
- Security or licensing questions: `hello@sofuu.xyz`.
