# Sofuu Android SDK 0.2.0 — update and publication plan

Status: implementation note for the `com.sofuu:sofuu-android:0.2.0` artifact.
Scope: the generic embedded runtime only. No host-application names, auth
vendors, pipelines, credentials, or product configuration belong in this
repository.

## Implementation status (2026-10-27)

- Source changes, JNI/Kotlin crypto helpers, publishing, and AAR verification
  are implemented in this tree.
- The release AAR builds for `arm64-v8a` and `x86_64`, is 7.7 MB, and contains
  `libsofuu.so`, `libsofuu_jni.so`, and `libc++_shared.so` per ABI.
- `publishToMavenLocal` succeeds for
  `com.sofuu:sofuu-android:0.2.0` (AAR, sources JAR, POM, module metadata).
- `make sdk-jvm-check` passes, including `sofuu.crypto.verifyJWT`, a tampered
  token rejection, SHA-256, and base64url.
- The native pack used for this AAR was built from commit `68bee868f`, not the
  current `HEAD`, because the current shared checkout has unrelated in-flight
  GPU/ML changes that do not compile for `aarch64-linux-android`. Rebuild with
  `make dist-android` once that work lands.
- Remote Maven publication is implemented but not executed: Maven Central
  needs ownership of the `com.sofuu` group plus signing credentials, and those
  are operator-owned. The namespace must not be published by guessing.
- The same layout is publicly available without credentials at
  `https://sofuu-runtime.github.io/sofuu/maven`, and the GitHub pre-release
  `android-sdk-v0.2.0` carries the AAR, sources, POM, module metadata, and
  checksums as assets.
- The 7.7 MB AAR reflects the full in-process inference path; the old Android
  script's 5 MB cap was raised to 12 MB for this artifact (override with
  `SOFUU_ANDROID_MAX_SIZE`).

## 1. What the artifact is

`com.sofuu:sofuu-android:0.2.0` is an Android library (AAR) around the
`libsofuu` C ABI. It gives a host app:

- `sofuu.ai` (local GGUF / SafeTensors inference and provider calls),
- `sofuu.agent` (agent definitions, runs, streaming, host tools),
- `sofuu.storage` (local structured documents),
- `sofuu.memory` (QTSQ-backed encrypted brain, when the build links QTSQ),
- `sofuu.crypto` (SHA-256, base64url, RS256 JWT/JWKS verification),
- `sofuu.http`, `sofuu.fs`, `sofuu.web`, `sofuu.mcp` where compiled in,
- offline embeddings, image embeddings, and provider/OS voice.

The intended local-first architecture is:

```text
Android UI
   |
   +-- Clerk or another identity SDK (optional, host-owned)
   |
   +-- JNI -> libsofuu -> JS pipeline -> local model + local storage
```

No HTTP server and no host backend are required for local data and
inference. The AAR is the whole on-device SDK.

## 2. Gaps found in the current 0.2.0 source tree

1. **No publication configuration.** `bindings/android/sofuu/build.gradle`
   has no `maven-publish`, POM, sources artifact, signing, or repository
   target. The README documents an install coordinate that is not on Maven
   Central.
2. **The prebuilt native libraries predate `sofuu.crypto`.** The staged
   `.so` files in `bindings/android/sofuu/src/main/jniLibs/` were built on
   2026-10-13; the crypto module was added on 2026-10-26.
3. **The AAR omits `libc++_shared.so`.** Both staged `libsofuu.so` files
   declare `NEEDED libc++_shared.so`, but the AAR contains only
   `libsofuu.so` and `libsofuu_jni.so`. A consumer APK would fail to load
   the runtime on a device.
4. **The Kotlin bridge exposes no crypto convenience methods** and
   `checkCaps()` does not report `crypto`, `storage`, `agent`, or `ai`.
   The generic funnel can reach them, but callers have to hand-build JSON.
5. **No JVM test covers auth.** `make sdk-jvm-check` exercises streaming and
   host tools but not `sofuu.crypto.verifyJWT`.
6. **The Gradle project has no wrapper**, no version from
   `gradle.properties` in the POM, and no `sources`/`javadoc` artifact.
7. **The QTSQ boundary is undocumented in the artifact.** The public AAR must
   remain QTSQ-free/MIT; a QTSQ-linked build has separate license terms.
   In a QTSQ-free build, `sofuu.memory` calls no-op while
   `sofuu.storage` still works.

## 3. Changes for 0.2.0

### 3.1 C ABI and `sofuu_embed.h`

The v2 ABI is frozen. This update does **not** change signatures, wire
shapes, or existing semantics. It documents the already-supported funnel
surface that local-first hosts need:

- `crypto.sha256`, `crypto.base64urlEncode`, `crypto.base64urlDecode`,
  `crypto.verifyJWT` through `sofuu_rt_call`.
- `storage.open` and the document methods through the same funnel.
- `agent.run`, `ai.stream`, and `ai.streamLocal` through
  `sofuu_rt_call_stream`.
- `memory.*` through the funnel, with the QTSQ caveat.
- A short example showing that JWT verification is a normal funnel call:
  a JSON array is spread as positional arguments, so
  `["<jwt>", "<jwks-json>", {"issuer":"..."}]` reaches
  `sofuu.crypto.verifyJWT(token, jwks, options)`.

No new exported symbol is required. Additive symbols remain allowed by the
v2 amendment ladder, but adding one just to wrap an existing funnel method
would make the ABI larger without adding capability.

### 3.2 `libsofuu`

Rebuild the Android native pack from the current source for:

- `arm64-v8a` (`aarch64-linux-android`)
- `x86_64` (`x86_64-linux-android`)

Requirements:

- release profile, stripped;
- `-Wl,-soname,libsofuu.so` so the JNI shim does not bake an absolute build
  path into `DT_NEEDED`;
- `-Wl,-z,max-page-size=16384` for Android 15+ 16 KiB-page devices;
- include `sofuu.crypto` (verify with a host-JVM test);
- QTSQ-free for the public artifact; document the QTSQ-linked alternative
  rather than publishing proprietary-linked binaries.

The native pack is generated, not committed: `.so` files are ignored by
`.gitignore`. The release process stages them into `jniLibs` and builds the
AAR.

### 3.3 AAR contents

The release AAR must contain, per ABI:

```text
jni/<abi>/libsofuu.so
jni/<abi>/libsofuu_jni.so
jni/<abi>/libc++_shared.so
```

`libc++_shared.so` is copied from the same NDK used to build the engine. The
CMake shim keeps `max-page-size=16384`.

### 3.4 Kotlin bridge

Add to `SofuuBridge`:

- `sha256(text)`, `base64urlEncode(text)`, `base64urlDecode(text)`,
  and `verifyJWT(token, jwksJson, optionsJson)` convenience methods that
  call the generic funnel and return the `result` value;
- `checkCaps()` entries for `sofuu.crypto`, `sofuu.storage`, `sofuu.ai`,
  `sofuu.agent`, and `sofuu.memory`;
- documentation that a local pipeline is driven with `eval` (bundled JS) plus
  `call`/`callStream`, and that the host must create the bridge on the
  engine thread.

Keep the bridge dependency-free. Do not add a JSON library beyond the
Android `org.json` already used by the wrapper.

### 3.5 Publication

Add `maven-publish` to `bindings/android/sofuu/build.gradle`:

```text
groupId:    com.sofuu
artifactId: sofuu-android
version:    0.2.0
packaging:  aar
```

Publication contents:

- release AAR,
- sources JAR,
- POM with name, description, URL, MIT license, and SCM,
- Gradle module metadata.

Signing and repository credentials are environment-driven and never
committed. Two targets:

1. `publishToMavenLocal` for immediate host-app consumption.
2. A remote Maven repository (Maven Central or GitHub Packages) when the
   operator supplies credentials and, for Maven Central, proof of ownership
   of the `com.sofuu` group.

The build must fail loudly if a release publication is attempted without
signing credentials when `REQUIRE_SIGNING=1` is set.

### 3.6 Tests and checks

- `make sdk-jvm-check` must additionally:
  - assert `sofuu.crypto.verifyJWT` exists;
  - verify a known RS256 token against a public test JWKS;
  - reject a tampered token;
  - assert `checkCaps()` reports `crypto` and `storage`.
- A packaging check must assert the AAR contains all three `.so` files per
  ABI and that `libsofuu.so` has no unresolved `libc++_shared.so`.
- `scripts/check_abi.sh` remains the authority for exported C symbols; no
  baseline change is expected because no symbol is added.

### 3.7 Documentation

Update:

- `bindings/android/README.md` — install, crypto/JWT example, local pipeline
  example, QTSQ caveat, publication status.
- `docs/EMBEDDING-DIST.md` — Android artifact, ABI list, QTSQ-free scope,
  Maven coordinates.
- `include/sofuu_embed.h` — funnel examples for crypto/storage/agent.
- `SOFUU-CRYPTO-JWT.md` — point at the Android convenience methods.

## 4. Acceptance checklist

- [ ] `libsofuu` rebuilt for arm64-v8a and x86_64 from current source.
- [ ] Host-JVM check proves `sofuu.crypto.verifyJWT` works through the
      bridge.
- [ ] AAR contains `libsofuu.so`, `libsofuu_jni.so`, and
      `libc++_shared.so` for both ABIs.
- [ ] AAR has a POM with `com.sofuu:sofuu-android:0.2.0`, MIT license,
      sources JAR, and Gradle module metadata.
- [ ] `publishToMavenLocal` succeeds.
- [ ] Remote publication is documented but requires operator credentials;
      no secret is added to the repository.
- [ ] QTSQ-free artifact is clearly labeled; QTSQ-linked builds are
      documented as a separate, licensed variant.

## 5. Non-goals

- No host-app pipeline, product name, auth vendor, or configuration.
- No Clerk-specific code or credentials in Sofuu.
- No HTTP server requirement for local mode.
- No public binary that links QTSQ without the separate license.
- No change to the frozen v2 ABI wire shapes.

## 6. Release commands

```bash
# 1. Build the QTSQ-free native pack (NDK required).
export ANDROID_NDK_HOME="$HOME/Library/Android/sdk/ndk/27.0.12077973"
make dist-android

# 2. Stage the pack into the AAR module (release script).
#    The generated .so files stay out of git.
bash scripts/mobile/stage_android_sdk.sh

# 3. Build and verify the AAR.
cd bindings/android
gradle :sofuu:assembleRelease :sofuu:verifyAarContents

# 4. Publish locally.
gradle :sofuu:publishToMavenLocal

# 5. Publish remotely (credentials supplied by the operator).
gradle :sofuu:publishReleasePublicationToRemoteRepository
```

A host app then uses:

```groovy
implementation 'com.sofuu:sofuu-android:0.2.0'
```
