# Sofuu CLI — prebuilt binaries

Cross-compiled from macOS (2026-08-30) with Zig as the cross-compiler.
No Docker, no VMs — see "Rebuilding" below for the exact commands.

| File | Platform | Notes |
|---|---|---|
| `sofuu-darwin-arm64` | macOS 11+ (Apple Silicon) | Full build **with QTSQ** (session store + brain persistence) |
| `sofuu-linux-x86_64` | Linux x86_64, any distro | Static musl; **with QTSQ** |
| `sofuu-linux-arm64` | Linux arm64 (Graviton / RPi / Oracle ARM) | Static musl; **with QTSQ** |
| `sofuu-windows-x86_64.exe` | Windows x64 (8.1+) | Static mingw build; TLS via Schannel (Windows's native TLS); **with QTSQ** |

All four binaries link the same QTSQ static libs (built per-target with
`zig cc` — see `scripts/cross/build_qtsq_cross.sh`): sessions persist as
`.qtsq` files, brain/KV memory persists, and the crypto suite (ML-KEM-768 /
ML-DSA-65 / Argon2) is present on every platform.

> **Redistribution note:** QTSQ is the proprietary codec checkout
> (`~/projects/black-hole-disk`, see AUTHORSHIP_SEAL.md/LEGAL.md there).
> Binaries in this folder carry it — keep them private; do not publish.

## Install

```sh
# macOS (Apple Silicon)
cp cli/sofuu-darwin-arm64 /usr/local/bin/sofuu   # or scp to the target box

# Linux x86_64 / arm64
cp cli/sofuu-linux-x86_64 /usr/local/bin/sofuu   # static — runs on any distro, no deps

# Windows: rename to sofuu.exe and put it on PATH
```

Verify: `sofuu version`

## What happens without QTSQ

**`make` now refuses to build without it.** A QTSQ-free binary is
*invisibly* broken: brain writes no-op, every session persist fails, and
— until 2026-09-25 — the chat banner still claimed `Memory: on (persists
across sessions)`. A QTSQ-free tarball shipped exactly like that on
2026-09-22 and every install since had a dead brain.

```sh
SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> make   # the supported path
SOFUU_ALLOW_NO_QTSQ=1 make                        # deliberate, degraded
```

With `SOFUU_ALLOW_NO_QTSQ=1` the build still works for CI and
third-party redistributors: chat, providers, tools, and the agent loop all
function, but sessions and brain state do not persist, and both the
banner and `sofuu doctor` say so plainly. `make` also verifies the codec
library actually exists in `SOFUU_QTSQ_DIR` before building, so a stale
path cannot silently produce a brain-less binary.

**Check any build in one command:**

```sh
sofuu doctor    # QTSQ link + a real brain write→flush→reopen→recall round-trip
```

Cross builds get QTSQ automatically via the per-target libs under
`dist/qtsq-*` built by `scripts/cross/build_qtsq_cross.sh`.

## Networking

All builds do HTTP(S) through the engine's `fetch` over a statically linked
libcurl 8.6.0:

- **Windows** — curl built with **Schannel** TLS: full `https://` support,
  certificates come from the OS store.
- **Linux** — curl built with **all TLS backends disabled** (no OpenSSL
  cross-deps): `http://` works everywhere; `https://` through the curl
  layer does not. Run behind a local proxy that terminates TLS, or build
  for the target with an mbedTLS/OpenSSL curl (see
  `scripts/cross/build_libcurl_static.sh`) to enable it.
- **macOS** — curl uses SecureTransport: full `https://` support.

## Rebuilding

Prereqs: Rust (rustup), CMake, Zig 0.13 (`make zig-install`).

```sh
make                    # macOS arm64 (native, full build)
bash scripts/cross/build_qtsq_cross.sh   # QTSQ libs for all 3 cross targets
make linux              # both Linux arches → dist/sofuu-linux-*
make windows-x86_64     # → dist/sofuu-windows-x86_64.exe
```

Scripts live in `scripts/cross/`:
`zig-cc.sh` / `zig-cxx.sh` (cc-rs-compatible `zig cc` wrappers that rewrite
Rust triples), `zig-ld.sh` (rustc linker wrapper), `zig-cc-win.sh` /
`zig-ar.sh` / `zig-ranlib.sh` (cmake wrappers for the Windows mingw target),
`build_linux_x86_64.sh`, `build_linux_arm64.sh`, `build_windows_x86_64.sh`.

Cross builds use `-p sofuu-core` (CLI package only — the desktop app's
notification stack needs a target libdbus and is out of scope here).
