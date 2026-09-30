# Release v0.2.0 — prepared, awaiting the owner's publish step

> Everything below is **prepared and verified locally**. The final
> tag-and-push is deliberately the owner's action (`PLAN-POSITIONING-2026.md`
> §12 keeps irreversible/publishing decisions with the owner).

## What this release is

The first release where the **headless SDK is installable**, not just
buildable. Before this, `libsofuu` existed, `PLAN-HEADLESS` H0–H6 were done,
and every documented install path was a `make` target on the author's
machine. That is the gap this release closes.

It is also the release that fixes the "docs promise more than ships" class of
bug in the SDK surface (see "SDK fixes" below), which is what the positioning
plan calls the credibility risk in §12.5.

## What is verified (run on this machine, not inferred)

| Gate | Result |
|---|---|
| `make` | builds, 3.2 MB (of a 5 MB cap) |
| `make test` (JS e2e) | 46 passed / 0 failed / 1 skipped |
| `cargo test -p sofuu-capi` | 33 passed / 0 failed |
| `make headless-test` | 5 C samples compile **and run**, all assertions pass |
| `make abi-check` | exported symbols match the baseline |
| `make size-check` | within cap |
| Swift bridge | `swiftc -typecheck -warnings-as-errors` against the real C header |
| JNI bridge | `cc -fsyntax-only` against `sofuu_embed.h` + a JDK's `jni.h` |
| `Package.swift` | `swift package dump-package` parses |
| `Sofuu.podspec` | validates (`ruby scripts/check_podspec.rb`) |
| npm package | install → sha256 verify → extract → run, end to end, live |
| npm tamper guard | a corrupted tarball is **refused** and nothing is written |

## What ships

**CLI** (unchanged mechanism, refreshed binaries with the QTSQ codec):
`sofuu-{darwin-arm64,linux-x86_64,linux-arm64}.tar.gz` + `.sha256`.

**Embeddable SDK — new in this release:**

| Artifact | Consumer |
|---|---|
| `libsofuu-darwin-arm64.tar.gz` | C/C++ on macOS |
| `libsofuu-linux-{x86_64,arm64}.tar.gz` | C/C++ on Linux (musl) |
| `libsofuu.xcframework.zip` | iOS/macOS via SwiftPM **and** CocoaPods |
| `libsofuu-android.tar.gz` (AAR + per-ABI `.so`) | Android via Gradle |
| `CHECKSUMS.txt` | everything, for verification |
| npm `sofuu` (publish separately) | `npx sofuu` |

## SDK fixes in this release (the credibility work)

These are the reasons the SDK was not usable before, each now pinned by a
test so it cannot regress:

1. **The brain is reachable from C.** `memory.open/remember/recall/count/
   flush` were documented but dead — the instance stringifies to `{}`, so a
   host could only reach memory by hand-building `rt_eval` strings. A handle
   facade fixes it; `examples/headless/c_embed.c` now does an open →
   remember → recall round-trip in its first 30 lines.
2. **`ai.stream` actually streams.** It returns an async iterable and
   ignores `onStep`, so the funnel delivered **zero** events while the header
   advertised it. The driver now pumps deltas, always ends with exactly one
   terminal `done` event, and supports a real mid-flight cancel.
3. **The first sample tells the truth.** `c_embed.c` called a funnel method
   (`version`) that does not exist and printed the resulting error envelope
   as if it were success. Every sample now asserts.
4. **The default embedding space matches the brain.** It was `sem2-64`
   (64-dim) while the brain is 768-dim and the docs called hash-v1 the
   default — so embed-then-remember was a silent dimension mismatch. The
   default is now `hash-768`, pinned by a test that runs the real path.
5. **`unknown_method` is a real error code**, distinct from `js_exception`,
   so a host can branch on "this build doesn't have that".
6. **`mcp.call` is reachable** (the client is an instance; it was
   connect-but-never-call), and `http.serve` replaces the documented-but-
   nonexistent `http.serve.start`.
7. **Doc drift is now structurally impossible**: a test parses the method
   table out of `docs/EMBEDDING.md` and fails if any listed method does not
   resolve. (It was 7 of 17 dead before this change.)

## Known, honest gaps in this release

- **Windows/WASM**: not built (v2). The npm installer refuses them with a
  clear message rather than half-installing.
- **The Android AAR is not published to Maven yet** — it is attached to the
  release as a tarball. `com.sofuu:sofuu-android` becomes a real coordinate
  once a Maven repository is configured (a Line-1 revenue prerequisite, per
  `PLAN-POSITIONING` §6).
- **The npm package is not published** — `npm/` is ready and tested; `npm
  publish` is a publishing action.
- **iOS/Android real-device verification**: the Swift bridge typechecks and
  the JNI bridge compiles, but neither has been run on a device/emulator in
  CI. Documented, not hidden.

## Publish steps (owner)

```bash
# 1. Sanity: everything green one more time, from a clean state.
export SOFUU_QTSQ_DIR=/path/to/black-hole-disk
make && make test && make headless-test && make abi-check && make size-check

# 2. Version + tag. Pushing the tag triggers release.yml, which builds the
#    CLI + all four SDK packs, attaches them, and commits the stamped
#    Package.swift back to the branch.
git add -A
git commit -m "release: v0.2.0 — installable headless SDK (SwiftPM/CocoaPods/Gradle/npm)"
git tag -a v0.2.0 -m "v0.2.0"
git push origin main --tags

# 3. After CI: publish the npm package (separate registry, separate decision).
cd npm && npm publish --access public

# 4. Refresh the public downloads the installer/npm fetch from.
#    (The macOS tarball was rebuilt with the QTSQ codec this session; the
#    other platforms in dist/ still predate the 2026-09-25 fixes and must be
#    refreshed from the new release, or the installer will keep serving a
#    brain-less binary.)
```

**Step 4 is not optional.** Until `sofuu.xyz/downloads` serves the *new*
tarballs, `npm i` and `curl | sh` will keep handing out the old binary — the
exact QTSQ-free failure that shipped on 2026-09-22.
