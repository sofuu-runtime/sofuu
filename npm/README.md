# sofuu (npm)

Sofuu is a **general-purpose runtime for AI** — built to run anywhere from
servers to smart devices. One standalone ≤ 5.5 MiB binary with a Rust/C core
(QuickJS + libuv, zero JIT): in-process local LLM inference (experimental),
autonomous agent loops, a native MCP client & server, encrypted QTSQ
cognitive vector memory, and sandboxed tools — orchestrated in TypeScript
and JavaScript. Zero daemons, zero Python, no API key to ship, no telemetry.

```bash
npx sofuu                 # interactive chat with memory that persists
npx sofuu run app.ts      # run TypeScript — no build step
npx sofuu add zod         # install a package
```

## Install

```bash
npm install -g sofuu      # or: npx sofuu — no global install needed
```

The postinstall step downloads the prebuilt binary for your platform, verifies
its SHA-256 against the published checksum, and places it under
`bin/runtime/`. If the verification fails, **nothing is installed** — this
package will not put an unverified binary on your machine.

Current release: **v0.3.0-beta.2**. The binary is revision-stamped — run
`sofuu version` and it prints the exact commit it was built from.

Supported today: macOS (arm64) and Linux, x86_64 and arm64. (Windows is on
the roadmap; the other installers already cover it.)

## Why the CLI, when there's a package?

This is the `sofuu` *CLI* — the free developer surface. The embeddable
runtime that you put inside your own app is a different artifact,
distributed through Swift Package Manager, CocoaPods, and Gradle:

| What you want | How |
|---|---|
| The `sofuu` CLI in your project | `npm i -g sofuu` (this package) |
| Embed the runtime in an iOS/macOS app | `Package.swift` or `pod 'Sofuu'` |
| Embed the runtime in an Android app | `implementation 'com.sofuu:sofuu-android:0.3.0'` |
| Link the C ABI directly | release tarballs → `docs/EMBEDDING-DIST.md` |

See https://sofuu.xyz/docs for the full picture.

## Project

Sofuu is an early-stage, bootstrapped, open-core project founded and built by
Haruhito. The source of truth for the runtime is the
[sofuu-runtime/sofuu](https://github.com/sofuu-runtime/sofuu) repository.

## What it gives you, offline and with no key

- **Embeddings with no model file** — 0 parameters, 0 bytes, microseconds a
  document, 768 dims. `ai.embed(text)`.
- **An encrypted local brain** — memory that belongs to your user, on their
  device, exportable, with no provider in the loop.
- **An agent loop + sub-agents + MCP** — tools and delegation, streamed.
- **A typed context-economy** — five tiny on-device models cut what enters
  the context window, locally.

## Environment variables (for CI and local builds)

| Variable | Effect |
|---|---|
| `SOFUU_BINARY_PATH` | Use a binary you built (`cargo build --release`) instead of downloading. |
| `SOFUU_SKIP_DOWNLOAD` | Do not fetch anything (useful in a matrix job that only needs the JS surface). |
| `SOFUU_DOWNLOAD_BASE` | Point at a different mirror of the release artifacts. |

```bash
# Build once, install everywhere in CI — no network on every job:
cargo build --release
SOFUU_BINARY_PATH="$PWD/target/release/sofuu" npm install
```

## License

The Sofuu runtime is MIT-licensed. The release binaries bundle the QTSQ
cognitive memory engine, which is **open source under the Apache License
2.0** ([github.com/haruhito2666/qtsq](https://github.com/haruhito2666/qtsq))
— full license text and third-party notices travel inside the package as
`LICENSE` and `NOTICE`.
