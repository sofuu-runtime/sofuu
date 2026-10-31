# Release v0.3.0-beta.2 — prepared, awaiting the owner's publish step

> Everything below is **prepared and verified locally**. The tag-and-push is
> deliberately the owner's action: publishing, republishing artifacts and
> stamping manifests are irreversible and need network access this session did
> not have.

## What this release is

The first release where the published binaries are **built from a commit you can
identify**. Before this, the macOS artifact on the site was 53 commits and ~35
hours behind `main`, had no revision stamp, and every runtime finding that was
really "the published binary is old" read as a live regression. `sofuu version`
now prints the revision, so that ambiguity cannot recur.

It also closes the audit: 38 of 54 findings fixed outright, 11 partial with the
remaining work written down, 4 left deliberately (see *Known, honest gaps*), and
1 shown to be a measurement error in the audit rather than a defect.

## What is verified (run on this machine, not inferred)

```
make size-check                     # macOS arm64 5,559,904 B vs 5,767,168 B cap
sh scripts/verify-dist.sh           # stamp + dirty-tree + per-artifact cap
sh tests/protocol_conformance_run.sh # 112/112
bash tests/run_js_tests.sh          # 72/72 files
sh tests/serve_test_run.sh          # 34/34 routes
cargo test -p sofuu-core --lib      # incl. 37 type-stripper tests
sh scripts/check-doc-metrics.sh     # no retired numbers in live docs
```

## What ships

| Artifact | Size now | Cap | State |
| --- | --- | --- | --- |
| macOS arm64 (this tree) | 5,559,904 B (5.30 MiB) | 5,767,168 B | **under** |
| macOS arm64 (published) | 5,655,808 B (5.39 MiB) | 5,767,168 B | under, stale |
| Linux arm64 (published) | 6,163,520 B (5.88 MiB) | 5,767,168 B | **over**, stale |
| Linux x86_64 (published) | 7,130,904 B (6.80 MiB) | 5,767,168 B | **over**, stale |

The two Linux archives must be rebuilt from this commit before they ship: they
are simultaneously over the cap and behind the engine.

## What changed

See `CHANGELOG.md` (2026-10-31) for the full list. The headline items:

- `sofuu version` carries the revision; the Makefile no longer skips strip,
  codesign and size-check; `verify-dist.sh` fails dirty-tree or unstamped builds.
- Four drivers and four Metal kernels stopped being embedded as source: the host
  build went from 320 KB **over** the cap to 207 KB **under** it.
- `listen()` refs its handle again (the README server example ran and exited),
  which also revived the MCP HTTP transport.
- `fetch` honours `AbortSignal` mid-body, serves 60 concurrent readers where it
  failed 55, and decodes gzip.
- TypeScript erasure lost two silent miscompiles: constructor parameter
  properties, and every object literal whose value was a bare identifier.
  `implements`, `abstract`, `export type`/`export interface` and generic arrows
  now erase instead of throwing.
- Policy writes validate and fail closed; `evaluate()` answers from the
  enforcement path; a per-call regexp budget bounds catastrophic backtracking.
- `structuredClone`, `Blob`, `File`, `FormData`, `Event`/`EventTarget`,
  `MessageChannel`, `navigator` and a persistent `localStorage` now exist.
- `process.argv` matches Node (`[exec, script, ...args]`) under `run` and in
  compiled artifacts.

## Known, honest gaps in this release

- **Linux artifacts are over the cap** and must be rebuilt (they are also stale).
- **SwiftPM and CocoaPods do not resolve.** The `v0.3.0-beta` tag's
  `Package.swift` has a placeholder checksum and points at a v0.2.0 asset, and no
  xcframework exists in any tag tree (`dist/` is gitignored). The site now says
  so instead of advertising an install that cannot work.
- **`namespace` and decorators** are refused with a `SyntaxError`; they need a
  transform the eraser does not implement.
- **`WeakRef`, `FinalizationRegistry`, `WebAssembly`, `Intl`, `WebSocket`,
  `EventSource`** are deliberately absent and documented as such.
- **`repair`** explains container-level damage but cannot salvage a brain whose
  codec table is gone.
- The site's font attribution gap is recorded in `sofuu-landing/THIRD_PARTY_NOTICES.md`;
  no license text was invented.

## Publish steps (owner)

```bash
# 1. Sanity, from a clean tree.
git status                                   # must be clean
make clean && make && make size-check
sh scripts/verify-dist.sh

# 2. Rebuild the Linux archives on a Linux host from this commit, then re-check
#    the cap on each (both are over today):
#      linux-arm64   6,163,520 B -> must end under 5,767,168 B
#      linux-x86_64  7,130,904 B -> must end under 5,767,168 B

# 3. Version + tag. Pushing the tag triggers release.yml, which builds the CLI
#    and the SDK packs, attaches them, and commits the stamped Package.swift.
#    The tag MUST be new: v0.3.0-beta cannot be repaired in place.
git tag -a v0.3.1 -m "sofuu v0.3.1"
git push origin v0.3.1

# 4. Stamp the SwiftPM checksum on the release commit, using the artifact that
#    is actually attached to that release (not a local build):
SOFUU_VERSION=0.3.1 ./scripts/stamp-swiftpm.sh dist/libsofuu-xcframework.zip
#    then upload that zip + its .sha256 to the release and commit Package.swift.

# 5. Publish the podspec (vendored xcframework from the release zip).

# 6. Refresh the site: public/downloads/*.tar.gz + matching *.sha256, and the
#    size table on sofuu.xyz/runtime (5.39/5.88/6.80 MiB are the OLD numbers).

# 7. npm publish (separate registry, separate decision).

# 8. Verify each artifact reports its revision:
./sofuu version    # must print "revision <tag's commit>"
```
