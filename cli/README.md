# Sofuu CLI — local prebuilt binaries

> **Status:** these are **local build outputs of the 0.2.0-beta line**
> (cross-compiled from macOS on 2026-08-30), not the current 0.3.0-beta
> release. Only this README is tracked in git; the binaries are not. They
> predate the Kaze engine, the `sofuu script` commands and `sofuu doctor`, so
> the commands documented in the repository README do not all exist in them.
> Use the current release for anything described there.
>
> **What these binaries actually have** (`sofuu --help`, 0.2.0-beta):
> `chat`, `repl`, `run`, `eval`, `bundle`, `install`, `add`,
> `agent list|run`, `serve`, `session …`, `outputs …`, `version`,
> `licenses`, `help`. There is no `script`, `doctor`, `brain`, `drone`,
> `daemon`, `mcp`, `policy`, `tools`, `capabilities`, `conformance`,
> `compile`, `infer`, `swarm`, `dev`, `logs`, `resume` or `plugin` here: an
> unknown command prints `Unknown command '<name>'` and exits **1**.

Cross-compiled from macOS (2026-08-30) with Zig as the cross-compiler.
No Docker, no VMs — see "Rebuilding" below for the exact commands.

| File | Platform | Notes |
|---|---|---|
| `sofuu-darwin-arm64` | macOS 11+ (Apple Silicon) | Native 0.2.0-beta build, **with QTSQ** (session store + brain persistence) |
| `sofuu-linux-x86_64` | Linux x86_64, any distro | Static musl; **with QTSQ** |
| `sofuu-linux-arm64` | Linux arm64 (Graviton / RPi / Oracle ARM) | Static musl; **with QTSQ** |
| `sofuu-windows-x86_64.exe` | Windows x64 (8.1+) | Static mingw build; TLS via Schannel (Windows's native TLS); **with QTSQ** |

All four binaries link QTSQ. The cross builds use the per-target static libs
built with `zig cc` — see `scripts/cross/build_qtsq_cross.sh`. The per-target
libraries under `dist/qtsq-linux-x86_64`, `dist/qtsq-linux-arm64` and
`dist/qtsq-windows-x86_64` contain the ML-KEM-768, ML-DSA-65 and Argon2
implementations; the linked binaries expose Argon2id through the encrypted
brain container. The macOS QTSQ library is not in this folder — it comes from
a local `SOFUU_QTSQ_DIR` checkout — and no functional ML-KEM / ML-DSA surface
was exercised from these binaries, so that part of the suite is unverified
here.

> **Provenance note:** QTSQ is the open-source codec checkout
> (https://github.com/haruhito2666/qtsq, Apache-2.0; local clone at
> `~/projects/black-hole-disk`). The binaries in this folder are local
> builds — official release artifacts come from the release pipeline. The
> 0.2.0-beta banner in these binaries still prints "QTSQ proprietary"; that
> string is stale and does not describe the codec they link.

## Install

```sh
# macOS (Apple Silicon)
cp cli/sofuu-darwin-arm64 /usr/local/bin/sofuu   # or scp to the target box

# Linux x86_64 / arm64
cp cli/sofuu-linux-x86_64 /usr/local/bin/sofuu   # static — runs on any distro, no deps

# Windows: rename to sofuu.exe and put it on PATH
```

Verify: `sofuu version` (prints `0.2.0-beta` for these builds).

## QTSQ is a mandatory core component

**`make` requires QTSQ and refuses to build without it.** QTSQ provides
session persistence and brain state storage across sessions. There is no
QTSQ-free build variant.

```sh
SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> make   # required
```

`make` verifies the codec library actually exists in `SOFUU_QTSQ_DIR` before
building, ensuring every build is a full, working binary with brain and session
persistence.

**Check a build:**

```sh
# 0.3.0-beta and later:
sofuu doctor    # QTSQ link + a real brain write→flush→reopen→recall round-trip
```

`sofuu doctor` exits **0** when the install passes every check it ran and
**1** when it found a problem (a missing codec, a failed brain round-trip, a
degraded build). Capabilities it did not exercise — a model request, a tool
call, an MCP connection — are reported as *not exercised*, never as a pass,
and they do not by themselves make the exit status non-zero.

The 0.2.0-beta binaries in this folder predate `sofuu doctor` (the command
above exits 1 with `Unknown command 'doctor'`). The same round-trip has to be
driven through the API, and the write is only on disk after an explicit flush:

```sh
sofuu eval 'const b = sofuu.memory.open("./brain.qtsq", 768);
  b.remember(sofuu.ai.embedLocal("probe"), "persisted-value", "user", null);
  console.log(b.flush())'
sofuu eval 'const b = sofuu.memory.open("./brain.qtsq", 768);
  const h = b.recall(sofuu.ai.embedLocal("probe"), 1);
  console.log(h.length, h[0] && h[0].text)'
```

That round-trip was observed to succeed on `cli/sofuu-darwin-arm64` in this
folder: the second process recalled `persisted-value` from the `.qtsq` file
(re-checked 2026-11-16 — the first `eval` printed `true` for the flush, the
second printed `1 persisted-value`).
Note the 0.2.0-beta `remember` takes four arguments (`vector`, `text`,
`role`, `kv_page_id`); the current release takes two, so API examples from
the repository README do not transfer to these binaries unchanged.

Cross builds get QTSQ automatically via the per-target libs under
`dist/qtsq-*` built by `scripts/cross/build_qtsq_cross.sh`.

## Networking

HTTP(S) goes through the engine's `fetch` over libcurl. What each binary in
this folder actually links was read from the binaries themselves:

- **macOS** (`sofuu-darwin-arm64`) — links the **system**
  `/usr/lib/libcurl.4.dylib` (`otool -L`), so the curl version is the one
  macOS ships, not a bundled 8.6.0. TLS is the OS stack
  (Security.framework), so certificates come from the OS store.
- **Linux** (`sofuu-linux-x86_64`, `sofuu-linux-arm64`) — **static** builds
  (`file` reports `static-pie linked` / `statically linked`) with libcurl
  **8.6.0** built in (the binary carries libcurl's own `CLIENT libcurl 8.6.0`
  version banner) and OpenSSL as the TLS backend (libcurl's OpenSSL-backend
  error strings are present). The exact OpenSSL version these 0.2.0-beta
  binaries were built with is not recorded inside them; the current
  cross-build uses **OpenSSL 3.3.2** (`dist/openssl-*-linux-musl`).
- **Windows** (`sofuu-windows-x86_64.exe`) — static mingw build, libcurl with
  **Schannel** TLS (the binary carries Schannel's `CALG_SCHANNEL_ENC_KEY`).
  Certificates come from the OS store.

The `SOFUU_CA_BUNDLE` / `SOFUU_CA_PATH` / `SSL_CERT_FILE` overrides and the
Linux host trust-store auto-probe are **not** in these 0.2.0-beta binaries
(none of those strings are in them). They exist in the current Rust runtime
(`crates/sofuu-ffi/src/curl.rs`) — use the current release for that
behaviour.

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
