# sofuu (npm)

The private, offline AI runtime — offline embeddings, an encrypted local
brain, agents, and MCP in a ~3 MB binary. No API key to ship, no telemetry.

```bash
npx sofuu                 # interactive chat with memory that persists
npx sofuu run app.ts      # run TypeScript
npx sofuu add zod         # install a package
```

## Install

```bash
npm install -D sofuu      # or: npm install -g sofuu
```

The postinstall step downloads the prebuilt binary for your platform, verifies
its SHA-256 against the published checksum, and places it under
`bin/runtime/`. If the verification fails, **nothing is installed** — this
package will not put an unverified binary on your machine.

Supported today: macOS and Linux, x86_64 and arm64. (Windows is on the
roadmap; the other installers already cover it.)

## Why the CLI, when there's a package?

This is the `sofuu` *CLI* — the free developer surface. The embeddable
runtime that you put inside your own app is a different artifact, distributed
through Swift Package Manager, CocoaPods, and Gradle:

| What you want | How |
|---|---|
| The `sofuu` CLI in your project | `npm i -D sofuu` (this package) |
| Embed the runtime in an iOS/macOS app | `Package.swift` or `pod 'Sofuu'` |
| Embed the runtime in an Android app | `implementation 'com.sofuu:sofuu-android'` |
| Link the C ABI directly | release tarballs → `docs/EMBEDDING-DIST.md` |

See https://sofuu.xyz/docs for the full picture.

## What it gives you, offline and with no key

- **Embeddings with no model file** — 0 parameters, 0 bytes, ~2 µs a
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

MIT. (A build with the QTSQ encrypted-store codec linked additionally falls
under `LICENSES/QTSQ-FORMAT.txt`; the release binaries here are the
QTSQ-free, pure-MIT flavor. See `docs/EMBEDDING-DIST.md`.)
