# libsofuu — Embeddable Sofuu Distribution

> **libsofuu** lets any host application embed Sofuu's AI runtime — LLM
> streaming, offline embeddings, encrypted brain/memory, agents, MCP, HTTP,
> TypeScript — with zero TUI. QuickJS is a pure interpreter (no JIT), so
> the iOS App Store executable-memory policy is satisfied by design.
>
> This file is the *artifact + install* reference. The API contract lives in
> [`EMBEDDING.md`](./EMBEDDING.md).

## Artifacts

| Platform | Artifact | Architecture(s) | Status |
|---|---|---|---|
| macOS | `libsofuu-darwin-{arm64,x86_64}.{a,dylib}` | arm64, x86_64 | built by `make dist-macos` |
| Linux (musl) | `libsofuu-linux-{x86_64,arm64}.{a,so}` | x86_64, arm64 | built by `make dist-linux` (needs Zig) |
| iOS | `libsofuu.xcframework` | arm64 (device), arm64+x86_64 (sim) | built by `make dist-ios` (needs Xcode) |
| Android | `libsofuu-android-{aarch64,x86_64}.so` | aarch64, x86_64 | built by `make dist-android` (needs NDK) |
| Windows | `libsofuu-windows-x86_64.dll` | x86_64 | ⬜ v2 (not yet) |
| WASM/edge | `libsofuu.wasm` | wasm32 | ⬜ v2 (documented subset) |

Each pack ships a static library, a shared library, and the `sofuu_embed.h`
header. `make dist-all` builds every pack the host machine can produce
(Android excluded from `dist-all` by design — it is `make dist-android`).

**Licensing:** the runtime is MIT. Platform packs are built **QTSQ-free**
(pure MIT) by default. A build with the QTSQ codec linked additionally falls
under `LICENSES/QTSQ-FORMAT.txt` — see "QTSQ" below, because it matters.

## Install

### Swift Package Manager (iOS / macOS)

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/sofuu-runtime/sofuu.git", from: "0.2.0")
]
```

then depend on the `Sofuu` product. The package's binary target resolves the
`libsofuu.xcframework` from the matching GitHub release asset (checksum
pinned in `Package.swift`).

### CocoaPods

```ruby
pod 'Sofuu', '~> 0.2'
```

### C / C++ (macOS, Linux)

```bash
# unpack the release tarball
tar xzf libsofuu-darwin-arm64.tar.gz
cc my_app.c -I. -L. -lsofuu -o my_app
```

### Android (Gradle)

```groovy
// The AAR bundles the per-ABI .so + the Kotlin bridge.
implementation 'com.sofuu:sofuu-android:0.2.0'
```

For a manual (non-Gradle) integration, copy `libsofuu-android-<abi>.so` into
`jniLibs/<abi>/` and build `examples/headless/KotlinSample/jni_bridge.c`
with its `CMakeLists.txt`.

### npm (the `sofuu` CLI)

```bash
npx sofuu run app.ts
```

The npm package downloads the matching prebuilt binary for your platform
and verifies its sha256. See `npm/README.md`.

## Quick start (C) — the whole thing in 15 lines

```c
#include "sofuu_embed.h"
#include <stdio.h>

int main(void) {
    SofuuRuntime *rt = sofuu_rt_new(NULL);   /* NULL config → all defaults */
    char *out = NULL;
    sofuu_rt_eval(rt, "1 + 1", &out);
    printf("result: %s\n", out);              /* {"ok":true,"result":2} */
    sofuu_free(out);
    sofuu_rt_free(rt);
    return 0;
}
```

`examples/headless/c_embed.c` is the full, *asserting* version: it embeds
text, opens a brain, stores a memory, and recalls it — all offline. Run it
with `make headless-test`.

## The two things to do first

### 1. Embeddings — no key, no network, no model file

```c
/* The default space is "hash-768" (768 dims) — the SAME space the brain
 * stores in, so you can embed here and remember there with no space
 * argument. Pass "sem1-64"/"sem2-64" for the learned spaces.
 * SOFUU_ERR_UNKNOWN_SPACE (-7) for an id this build lacks. */
float *v = NULL; size_t dim = 0;
sofuu_embed_local(rt, "hello world", NULL, &v, &dim);   /* 768 dims */
sofuu_free(v);

/* Batch ingestion — one call, not N. */
float *batch = NULL; size_t rows = 0, cols = 0;
const char *texts[] = { "first doc", "second doc" };
sofuu_embed_batch(rt, texts, 2, NULL, &batch, &rows, &cols);
sofuu_free(batch);

/* Image bytes (PNG or JPEG) → img1-64, joint with sem2-64 text geometry,
 * so a text query retrieves images. */
const uint8_t *png = read_file("shot.png", &len);
sofuu_embed_image(rt, png, len, &v, &dim);

/* Which spaces are actually in this build, and which is the default? */
char *manifest = NULL;
sofuu_embed_info(rt, &manifest);
sofuu_free(manifest);
```

### 2. The brain — encrypted local memory over the JSON funnel

`memory.open` and friends return **handles** (the raw JS object does not
survive JSON). With one brain open, `handle` is optional.

```c
char *out = NULL;
sofuu_rt_call(rt, "memory.open",
              "{\"path\":\"/data/brain.qtsq\",\"dim\":768}", &out);
/* → {"ok":true,"result":{"handle":0,"dim":768}} */

sofuu_rt_call(rt, "memory.remember",
              "{\"vec\":[…768 floats…],\"text\":\"the sky is blue\","
              "\"role\":\"user\",\"kv_page_id\":0}", &out);

sofuu_rt_call(rt, "memory.recall", "{\"vec\":[…],\"k\":5}", &out);
/* → {"ok":true,"result":[{"text":"the sky is blue","score":0.98,…}]} */

sofuu_free(out);
```

See [`EMBEDDING.md`](./EMBEDDING.md) §6 for the full method table (it is
enforced by a test, so it cannot drift from the code).

## Streaming

```c
void on_event(const char *event_json, void *opaque) {
    /* intermediate: {"kind":"delta","text":"…"} (ai.stream)
     *                or agent.run's own step events
     * terminal:     {"kind":"done", …}  — ALWAYS exactly one           */
}

uint64_t cancel_id = 0;
sofuu_rt_call_stream(rt, "ai.stream", "{\"prompt\":\"…\"}",
                     on_event, my_ctx, &cancel_id);
/* from inside the callback, to stop it: */
sofuu_rt_cancel(rt, cancel_id);
```

The stream always ends with one terminal `{"kind":"done",…}` event, carrying
`deltas`/`aborted` on success or `error` on failure, so a host that waits for
`done` is never left hanging. `*out_cancel_id` is written before the stream
starts, so the callback can cancel in flight.

## Voice

```c
/* Provider audio (STT/TTS) over any OpenAI-compatible endpoint. The HOST
 * base64-encodes the audio — no encoder is baked in. Both return a malloc'd
 * JSON string the host frees with sofuu_free. */
sofuu_voice_transcribe(rt, audio_b64,
                       "{\"provider\":\"openai\",\"model\":\"whisper-1\"}", &json);
sofuu_voice_speak(rt, "hello",
                  "{\"provider\":\"openai\",\"model\":\"tts-1\",\"voice\":\"alloy\"}", &json);
```

For **on-device** speech — no network, no key, no per-call cost — use the
platform wrappers instead of the C ABI: `SofuuVoice.swift`
(`SFSpeechRecognizer` for files and live mic + `AVSpeechSynthesizer`) and
`SofuuVoice.kt` (`SpeechRecognizer` + `TextToSpeech`). Full-duplex voice
orchestration stays a sample-level recipe: OS microphone capture is not
portable C.

## QTSQ (the encrypted brain codec)

The brain's encrypted store needs the QTSQ codec. Two consequences:

- **Building the packs** requires `SOFUU_QTSQ_DIR`; `make` refuses to build
  without it. Opt out with `SOFUU_ALLOW_NO_QTSQ=1` (CI does this).
- **A QTSQ-free build is invisibly broken**: memory calls no-op and session
  persists fail while the UI still claims persistence. This shipped once
  (2026-09-22) and is why the guard exists.

Check what you actually linked, in one command:

```bash
sofuu doctor    # reports QTSQ linkage + a real write→flush→reopen round-trip
```

## CI gates

Every CI run and release builds libsofuu and runs:

1. **Headless embedding test** — compiles and *runs* `c_embed.c`,
   `c_embed_vec.c` (vector ABI), `c_llm.c` (real LLM path, mock in CI),
   `c_rlm.c`, and `c_agent.c` against `dist/libsofuu`. Every sample asserts
   its calls, so a broken sample fails the build (`make headless-test`).
2. **capi unit tests** — the funnel, memory handles, streaming, cancel,
   multi-instance isolation (`cargo test -p sofuu-capi`).
3. **ABI symbol guard** — exported public symbols diffed against
   `scripts/abi_symbols.txt`. Missing = breaking (fails); new = must be
   acknowledged (`make abi-check`).
4. **Funnel fuzz target** — `crates/fuzz/fuzz_targets/capi_call.rs` feeds
   arbitrary method/JSON to `sofuu_rt_call`; a crash is a failed run.

## Samples

| Language | Path | What it shows |
|---|---|---|
| C | `examples/headless/c_embed.c` | eval + funnel + embed + full brain round-trip (asserting) |
| C | `examples/headless/c_embed_vec.c` | vector ABI: embed/batch/info, spaces, error guards |
| C | `examples/headless/c_llm.c` | a real LLM call; mock provider in CI, live with `SOFUU_DEMO_KEY` |
| C | `examples/headless/c_rlm.c` | RLM long-context Q&A (headless proof) |
| C | `examples/headless/c_agent.c` | agent define/run/stream/cancel |
| Swift | `examples/headless/SwiftSample/SofuuBridge.swift` | iOS wrapper: embed/batch/image/info, voice, caps |
| Swift | `examples/headless/SwiftSample/SofuuVoice.swift` | on-device STT/TTS (no network) |
| Kotlin | `examples/headless/KotlinSample/SofuuBridge.kt` | Android JNI wrapper (mirror of the Swift API) |
| Kotlin | `examples/headless/KotlinSample/SofuuVoice.kt` | on-device STT/TTS (no network) |

## Size

Platform packs enforce a 5MB cap (`make size-check`; the `dist-*` scripts
check their own artifacts). Measured on macOS arm64, this workspace:

| Artifact | Size |
|---|---|
| `libsofuu.a` (release pack, QTSQ-free) | ~1.7 MB |
| `libsofuu.dylib` (release pack, QTSQ-free) | ~1.7 MB |
| `libsofuu.a` (**local** build, QTSQ linked) | ~30 MB (debug/unstripped archive) |
| `sofuu` CLI (macOS arm64, QTSQ linked) | 3.2 MB of a 5 MB cap |

The local 30 MB `.a` is not what ships: the `dist-*` scripts build
QTSQ-free and size-capped. Compare like with like.
