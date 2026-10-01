# Sofuu Desktop

Tauri 2 desktop shell for the Sofuu runtime. The engine (`sofuu-core` + the
QuickJS C core via `sofuu-ffi`) links in-process; the React/TypeScript
frontend is UI-only — every runtime touch lives in `src-tauri`.

## macOS

```sh
npm install
npm run tauri dev     # dev loop (beforeDevCommand runs vite)
npm run tauri build   # .app + .dmg in ../target/release/bundle/
```

API keys are stored in the macOS Keychain (`com.sofuu.desktop`); provider
models are fetched inside the engine — keys never reach the WebView.

## Windows (x64)

Build prerequisites:

- **Rust** (MSVC toolchain) + **Visual Studio Build Tools** (Desktop C++)
- **Node 20+**
- **libuv** — built from the vendored `deps/libuv` with CMake:
  ```sh
  cd deps/libuv
  cmake -S . -B build-win64 -A x64 -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
  cmake --build build-win64 --config Release
  ```
- **curl** — [vcpkg](https://vcpkg.io) `curl[schannel]` static-MD
  triplet:
  ```sh
  vcpkg install "curl[schannel]:x64-windows-static-md"
  ```

Then point the build at the two libs and bundle:

```sh
cd sofuu-desktop
export SOFUU_UV_DIR="$PWD/../deps/libuv/build-win64/Release"   # dir holding uv_a.lib
export VCPKG_INSTALLATION_ROOT="C:/vcpkg"                      # or SOFUU_CURL_STATIC_DIR=<lib dir>
npm ci
npm run tauri build -- --bundles nsis
```

`build.rs` finds `uv_a.lib` via `SOFUU_UV_DIR` and `libcurl.lib` via
`SOFUU_CURL_STATIC_DIR` or `$VCPKG_INSTALLATION_ROOT/installed/x64-windows-static-md/lib`.

### CI

> **Not built in CI for now.** The Windows workflow existed here and was
> removed alongside the other non-macOS jobs; the commands it ran are listed
> above so it can be restored verbatim. To rebuild by hand: install curl via
> the runner's preinstalled vcpkg, build vendored libuv with
> `-DBUILD_SHARED_LIBS=OFF`, then `npm run tauri build -- --bundles nsis`.

### Windows notes & limitations

- **QTSQ (full session persistence) is now buildable on Windows.** The
  proprietary QTSQ checkout ships an MSVC port (`windows/CMakeLists.txt` +
  `windows/shim/` — a single `qtsq.lib` with the qtc compressor, the crypto
  suite, and a Win32 POSIX-compat shim baked in). Build it inside the
  checkout, point `SOFUU_QTSQ_DIR` at it, and `build.rs` links it:
  brain/memory persistence and the session store work, chats survive app
  restarts. zlib comes from the same vcpkg triplet as curl
  (`x64-windows-static-md`).

  ```sh
  # inside the QTSQ checkout (see its windows/README.md for details)
  vcpkg install zlib --triplet x64-windows-static-md
  cmake -S windows -B windows/build-win64
  cmake --build windows/build-win64 --config Release   # → Release/qtsq.lib

  # then build the desktop app with
  export SOFUU_QTSQ_DIR="<path to the QTSQ checkout>"
  ```

  CI still builds **without** QTSQ (the checkout is proprietary and cannot
  ship in CI) — CI artifacts remain fail-closed: providers/models/chat
  itself are unaffected, but chats don't survive an app restart there.
- **No `Atomics.*` in engine JS.** The vendored QuickJS `CONFIG_ATOMICS`
  (pthread-dependent) is gated off for MSVC; user JS that uses `Atomics`
  throws. QuickJS's own `_WIN32` branches cover the rest of the port; three
  small POSIX shim headers live in `deps/quickjs/msvc-compat/` (only added
  to cl.exe's include path — gcc/clang never see them).
- **Glass effect**: Mica on Windows 11, acrylic on Windows 10 1809+ (the
  acrylic fallback lags on resize/drag — a known upstream limitation). On
  older systems the window is plain. Light/Dark/System themes work
  throughout; the theme setting drives the native window theme too.
- **Transparency** requires Windows 10 1809+; older systems get an opaque
  window.

## Linux (x64)

Build prerequisites (Debian/Ubuntu package names; adjust for other distros):

- **Rust** + **Node 20+**
- Tauri's Linux prerequisites:
  ```sh
  sudo apt-get install -y \
    libwebkit2gtk-4.1-dev build-essential curl wget file \
    libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
  ```
- **libcurl** headers (`libcurl4-openssl-dev` — the engine links system
  `-lcurl`) and **libdbus-1-dev** (Secret Service keyring backend)
- **libfuse2** (or `libfuse2t64`) — needed by the AppImage tooling

Then build libuv from the vendored tree and bundle:

```sh
cd deps/libuv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF
cmake --build build --config Release -j"$(nproc)"

cd ../../sofuu-desktop
npm ci
npm run tauri build -- --bundles deb,appimage
```

Packages land in `../target/release/bundle/{deb,appimage}/`. `build.rs`'s
default non-Windows link branch (system `-lcurl`, `pthread`/`m`/`dl`,
`deps/libuv/build/libuv.a`) is exactly the Linux recipe — no env vars
needed.

### CI

> **Not built in CI for now.** The Linux workflow existed here and was
> removed alongside the other non-macOS jobs; the commands it ran are listed
> above so it can be restored verbatim. To rebuild by hand: install the apt
> prerequisites, build vendored libuv with `-DBUILD_SHARED_LIBS=OFF`, then
> `npm run tauri build -- --bundles deb,appimage`.

### Linux notes & limitations

- **QTSQ on Linux = build the checkout locally.** The codec builds cleanly
  on Linux (docker gcc:13 clean-build proven) — run `make` inside the QTSQ
  checkout (needs zlib dev headers), export `SOFUU_QTSQ_DIR=<checkout path>`,
  and the app links it: full brain/memory persistence and session store.
  CI artifacts build without it (the proprietary checkout can't ship in
  CI), so they stay fail-closed — chats don't survive an app restart there.
  Providers/models/chat itself are unaffected either way.
- **API keys** persist in the freedesktop Secret Service (GNOME Keyring /
  KWallet) via keyring. A running secret-service daemon is required at
  runtime; on a headless box without one, keys simply don't persist
  (fail-closed, same as the session store).
- **Glass effect**: there is no cross-distro Mica/vibrancy equivalent, so
  the sidebar renders as a translucent pane (the CSS tint does the work);
  window transparency depends on the compositor. Light/Dark/System themes
  work throughout.
