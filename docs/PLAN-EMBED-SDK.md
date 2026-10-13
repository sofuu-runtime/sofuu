# 📱 Plan: Sofuu Embed SDK — The Embeddable Runtime for Mobile & Edge Applications

> *The `libsofuu` SDK enabling iOS (Swift), Android (Kotlin), and Embedded Edge (C/C++) developers to embed in-process AI, persistent memory, and autonomous agents directly into native apps with zero background daemons.*

---

## 1. Executive Vision & Scope

Mobile and edge application developers face a severe dilemma when building on-device AI experiences:
1. **Raw C++ libraries (llama.cpp, ExecuTorch)** provide raw tensor math but lack a complete runtime: no persistent memory, no agent framework, no tool calling, no MCP, and fragile cross-language FFI bindings that crash host apps.
2. **Platform-locked frameworks (Apple CoreML, Google MediaPipe)** are vendor-specific, lack multi-platform parity, and offer zero cognitive memory architecture.
3. **Cloud/Python frameworks (LangChain, CrewAI)** cannot run on mobile devices and depend on remote servers, destroying user privacy and offline capabilities.

**The Sofuu Embed SDK (`libsofuu`)** solves this completely:
It is a single, self-contained library ($\le 4.4\text{ MB}$, $\le 30\text{ MB}$ runtime RSS) that drops into native mobile and edge applications. With just a few lines of Swift or Kotlin, any app gains:
* **In-Process Local LLM Inference** (GGUF & SafeTensors) streaming directly into native UI widgets.
* **Encrypted Persistent Memory (`.qtsq`)** storing user context and conversation history locally across app launches.
* **Autonomous In-App Agents & Tool Execution** capable of multi-step reasoning.
* **Host-Native Tool Bridge** allowing in-app agents to invoke native mobile APIs (Camera, Photos, Calendar, HealthKit, Location, SQLite).
* **Native Model Context Protocol (MCP)** connecting external tools.
* **Zero-JIT App Store Compliance by Design**: QuickJS is a pure C interpreter with no JIT, strictly conforming to Apple iOS App Store $W \oplus X$ executable memory rules.

---

## 2. Architecture: The Embeddable Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Host Application Layer                          │
│        iOS (SwiftUI / UIKit)       │     Android (Jetpack Compose)     │
│        Edge Appliance (C++ / Rust) │     Embedded Linux / IoT          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Clean Swift / Kotlin SDK
┌───────────────────────────────────▼────────────────────────────────────┐
│                    libsofuu C ABI v2 Layer                             │
│                  include/sofuu_embed.h                                 │
├───────────────────────────────────┬────────────────────────────────────┤
│   Lifecycle & Host Isolation      │   Streaming & Cancellation         │
│   • sofuu_rt_new / sofuu_rt_free  │   • sofuu_rt_call_stream           │
│   • Host log routing (no stdout)  │   • sofuu_rt_cancel                │
│   • Zero-exit / Catchable errors  │   • Delta & Done events            │
├───────────────────────────────────┼────────────────────────────────────┤
│   Direct Vector Entry Points      │   Host-Native Tool Bridge          │
│   • sofuu_embed_local (hash-768)  │   • Host callback registry         │
│   • sofuu_embed_image (joint)     │   • Bi-directional JSON envelopes  │
└───────────────────────────────────┴────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                  In-Process Sofuu Engine Internals                     │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   In-Process Inference  │  │    Cognitive Memory & Brain         │  │
│  │   • GGUF & SafeTensors  │  │    • QTSQ Encrypted Persistent Brain│  │
│  │   • NEON / Metal GPU    │  │    • 4-Tier CMA & HNSW Index        │  │
│  └─────────────────────────┘  └─────────────────────────────────────┘  │
│  ┌─────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   QuickJS Core (No JIT) │  │    Autonomous Agent Runtime         │  │
│  │   • Sandboxed Execution │  │    • Tool Loop & Budget Trees       │  │
│  │   • Pure C Interpreter  │  │    • Native MCP Client/Server       │  │
│  └─────────────────────────┘  └─────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Universal C ABI v2 (`include/sofuu_embed.h`)

### 3.1 Hostile-Host Hardening Rules
When Sofuu is loaded as a dynamic library inside a third-party host process (such as an iOS app), it must adhere to strict non-interference rules:
1. **Zero Termination**: The library never calls `exit()`, `abort()`, or `quick_exit()`. Any `process.exit(code)` from JavaScript throws a catchable `ExitError` with an `exitCode` property.
2. **Signal Isolation**: Signal handlers (`SIGINT`, `SIGTERM`, `SIGWINCH`) are disabled by default (`enable_signals: false`) to avoid hijacking host crash reporters (e.g. Crashlytics).
3. **No Direct Standard Output**: All `console.log/info/warn/error` statements are redirected to a host-registered callback (`sofuu_embed_set_log_cb`), preventing log pollution in release apps.
4. **Stable Error Envelopes**: Errors return negative integer status codes and populate structured JSON envelopes:
   ```json
   { "ok": false, "error": { "code": "model_not_found", "message": "Failed to open model file" } }
   ```
5. **Memory Ownership**: All buffers allocated by `libsofuu` are malloc-owned and freed via `sofuu_free(ptr)`.

### 3.2 Core ABI Functions
* `SofuuRuntime *sofuu_rt_new(const char *config_json)`: Instantiates a runtime instance configured for embedded mode, explicit brain storage path, and provider defaults.
* `void sofuu_rt_free(SofuuRuntime *rt)`: Releases all memory, caches, and thread handles.
* `int sofuu_rt_call(SofuuRuntime *rt, const char *method, const char *args_json, char **out_json)`: Synchronous generic funnel for non-streaming calls (`memory.open`, `ai.complete`, `rlm.query`).
* `int sofuu_rt_call_stream(SofuuRuntime *rt, const char *method, const char *args_json, sofuu_event_cb on_event, void *opaque, uint64_t *out_cancel_id)`: Asynchronous streaming pump delivering chunk deltas (`{"kind":"delta","text":"..."}`) and guaranteed terminal done events (`{"kind":"done",...}`).
* `int sofuu_rt_cancel(SofuuRuntime *rt, uint64_t cancel_id)`: InterruptedException token cancelling in-flight streaming turns.
* `int sofuu_embed_local(SofuuRuntime *rt, const char *text, const char *space, float **out, size_t *out_dim)`: High-speed native vector embedding without JSON serialization overhead.

---

## 4. Platform Target: iOS (Swift & SwiftUI)

### 4.1 Packaging & Distribution
* **Artifact**: `libsofuu.xcframework` (multi-architecture: `ios-arm64` device + `ios-arm64-simulator`).
* **Distribution Channels**:
  - **Swift Package Manager (`Package.swift`)**: Direct dependency in Xcode with binary checksum verification.
  - **CocoaPods (`Sofuu.podspec`)**: For enterprise iOS codebases using legacy podfiles.

### 4.2 Swift Developer Ergonomics
```swift
import SwiftUI
import Sofuu

@MainActor
class AssistantViewModel: ObservableObject {
    @Published var responseText = ""
    @Published var isGenerating = false
    
    private var engine: SofuuEngine?
    
    init() {
        guard let modelPath = Bundle.main.path(forResource: "Qwen-3-0.6B-Q4_K_M", ofType: "gguf") else { return }
        let brainUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("user_brain.qtsq")
        
        self.engine = try? SofuuEngine(config: [
            "model": modelPath,
            "brain_path": brainUrl.path
        ])
    }
    
    func send(prompt: String) async {
        guard let engine = self.engine else { return }
        isGenerating = true
        responseText = ""
        
        do {
            // Native Swift AsyncThrowingStream
            for try await token in engine.stream(prompt: prompt) {
                responseText += token
            }
        } catch {
            responseText = "Error: \(error.localizedDescription)"
        }
        isGenerating = false
    }
}
```

### 4.3 iOS-Specific Engineering Requirements
* **Zero-JIT Policy**: QuickJS interprets bytecode directly in read-only memory, satisfying App Store Review Guideline 2.5.2 ($W \oplus X$).
* **Jetsam Memory Guard**: iOS kills apps that exceed strict memory budgets (e.g. 500 MB–1.5 GB depending on device). Sofuu guarantees:
  - Runtime RSS strictly $\le 30\text{ MB}$ (excluding memory-mapped model weights).
  - Model weights mapped via zero-copy clean mmap pages that the iOS kernel can page out on memory pressure.
  - Sliding Window Attention (SWA) bounding KV cache to a fixed context limit.
* **Metal GPU Acceleration**: macOS-only today. `crates/sofuu-ffi/build.rs:149`
  compiles `src/metal/metal_engine.mm` only when `target_os == "macos"`, so the
  iOS slices of the library carry **no Metal engine**, and `scripts/dist/ios.sh`
  bundles no `default.metallib` — the shader that does ship is compiled into the
  *sample app's* bundle by Xcode, not into the framework. Making this real on
  iOS (platform-gate the engine, compile `kernels.metal` into a
  `default.metallib`, ship it as a framework resource, route to it) is
  milestone E3, not a verification step.
* **Jetsam Memory Guard**: iOS kills apps that exceed strict memory budgets
  (e.g. 500 MB–1.5 GB depending on device). What actually holds today:
  - Model weights are mapped read-only and clean (`infer/mmap.rs:32`,
    `PROT_READ | MAP_SHARED`), so the kernel can evict the pages under
    pressure — this part is real and verifiable.
  - The KV cache is **f32 and allocated for the full context at load**
    (`infer/model.rs:391`), which is ~940 MB for Qwen3-0.6B at the 4096-token
    default. It is not "model weights", so it counts against any RSS budget.
    SWA bounds attention *compute*, not this allocation.
  - There is no response to memory pressure anywhere (no
    `didReceiveMemoryWarning`, no `onTrimMemory`, no cache trim, no unload).
  - The old claim of "runtime RSS strictly ≤ 30 MB" is therefore unreachable as
    written; E5 reframes it as measure-then-reduce rather than assert.
* **App Lifecycle Handling**: Observes `scenePhase` transitions. When the app moves to background, active generation is either paused or finishes current token before suspending to prevent watchdog terminations.

---

## 5. Platform Target: Android (Kotlin & Jetpack Compose)

### 5.1 Packaging & Distribution
* **Artifact**: `com.sofuu:sofuu-android` (Gradle AAR containing `libsofuu.so` for `arm64-v8a` and `x86_64`).
* **Distribution Channels**: Maven Central / GitHub Packages.

### 5.2 Kotlin Developer Ergonomics
```kotlin
package com.example.ai

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sofuu.SofuuEngine
import com.sofuu.SofuuConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ChatViewModel(private val engine: SofuuEngine) : ViewModel() {
    private val _messages = MutableStateFlow<String>("")
    val messages = _messages.asStateFlow()

    fun ask(prompt: String) {
        viewModelScope.launch {
            // Reactive Kotlin Flow streaming
            engine.streamLocal(prompt).collect { chunk ->
                _messages.value += chunk
            }
        }
    }
}
```

### 5.3 Android-Specific Engineering Requirements
* **JNI Bridge (`cpp/jni_bridge.c`)**: High-speed, zero-copy JNI translation layer mapping C callbacks to Kotlin coroutines.
* **Hardware NEON & `sdot` Acceleration**: Native C++ microkernels built via Android NDK with `-march=armv8.2-a+dotprod` targeting modern Snapdragon and Dimensity SoCs.
* **Low Memory Killer (LMK) Protection**: Proactive memory monitors preventing Android LMK kills on devices with 4 GB–6 GB RAM.

---

## 6. The Host-Native Tool Bridge (The Mobile Superpower)

One of the most powerful capabilities of the Sofuu Embed SDK is enabling in-app agents to execute **native host functions**:

```
 [ In-App Sofuu Agent ] ──▶ Emits Tool Call: "take_photo"
            │
            ▼
 [ libsofuu C ABI ]     ──▶ Dispatches to Host Callback
            │
            ▼
 [ Host App (Swift/Kotlin) ] ──▶ Opens iOS Camera / Android CameraX
                                 Takes picture ──▶ Saves to disk
                                 Returns file path JSON
            │
            ▼
 [ libsofuu C ABI ]     ──▶ Feeds result back into Agent Loop
            │
            ▼
 [ In-App Sofuu Agent ] ──▶ Embeds image via ai.embedImage & answers user!
```

### Swift Host Tool Registration Example
```swift
// Register a native Swift tool with the Sofuu runtime
engine.registerTool(name: "get_user_location", description: "Gets device GPS coordinates") { args in
    let location = await LocationManager.shared.currentLocation()
    return ["latitude": location.lat, "longitude": location.lon]
}
```
This turns Sofuu into an intelligent in-app copilot that can take actions directly within the host app's native ecosystem.

---

## 7. Edge & Embedded Devices (C / C++ / Rust SDK)

* **Deployment Target**: Raspberry Pi 5, Nvidia Jetson Orin Nano, Rockchip RK3588, and industrial IoT hubs.
* **Artifacts**: `libsofuu.so` (Linux dynamic library) and `libsofuu.a` (static archive).
* **Usage**: Embeds directly into local C/C++ daemons or Rust edge services with zero external network access, powering local robotics, smart cameras, and voice appliances.

---

## 8. Milestone Roadmap for Embed SDK

> **Status updated 2026-10-13.** E1 and E2 ship in the C ABI, in the core JS
> runtime, in the Swift binding, and in the Kotlin AAR wrapper
> (`bindings/android/sofuu`), which carries its own streaming and host-tool
> surface verified on the host JVM (`make sdk-jvm-check`). E3–E5 remain
> open, and their premises are corrected below where the original text
> described something that does not exist.

### Milestone E1: Native Local Inference in `sofuu_rt_call_stream` — SHIPPED
- [x] `sofuu_rt_call_stream` accepts an object **or an array** in `args_json`,
      exactly like `sofuu_rt_call`. This is what makes positional methods
      reachable at all: `ai.streamLocal` takes `(modelPath, prompt, options?)`
      and the driver previously always called the method with a single object.
      `onStep` and the cancel token are injected into the trailing options
      object (appended when the caller passed none).
- [x] A local model path routes in-process from the path alone — a `.gguf`
      file, a SafeTensors / PyTorch / NumPy model *directory* with its
      `tokenizer.json`, or a bare weights file. SafeTensors directories
      previously required an explicit `provider:"local"`, so handing an
      embedded host a directory sent the request to the network as if it were
      a model name.
- [x] Swift: `Sofuu.streamLocal(modelPath:prompt:maxTokens:temperature:onEvent:)`
      and `Sofuu.streamPositional(_:_:onEvent:)` for any positional method.
- [x] Kotlin **SDK** binding (`bindings/android/sofuu`):
      `callStream(method, argsJson, listener)` with a `fun interface
      StreamListener` whose `onStreamEvent` return value is the in-flight
      cancel (the JNI slot holds a pointer to the cancel id, which
      `sofuu_rt_call_stream` writes before the first event — readable only
      from the callback, and the only thread `sofuu_rt_cancel` accepts), and
      `streamLocal(modelPath, prompt, maxTokens, temperature)`. The
      app-level bridge in `sofuu-android/` got `callStream` earlier.
- [x] Two load-blockers found and fixed while wiring this up, both in the
      shipped wrapper: it loaded `System.loadLibrary("sofuu")` — the runtime
      `.so`, which contains *none* of the JNI natives (they are compiled
      into `libsofuu_jni.so`, whose `DT_NEEDED` pulls the runtime) — so on a
      device every call would have thrown `UnsatisfiedLinkError`; and the
      shim's CMake lacked `-Wl,-z,max-page-size=16384`, which apps targeting
      Android 15+ require on 16 KiB-page devices.
- [x] The JNI text boundaries convert modified UTF-8 (JNI's "UTF-8" is
      CESU-8 for anything outside the BMP: an emoji in a prompt arrived at
      the runtime as invalid bytes and the engine saw **no arguments at
      all**; probed and fixed in both directions for eval, call, streaming
      and host tools).
- [x] Verified with real models: a 0.6B GGUF streams token deltas through the
      C ABI with a working mid-flight cancel, and an LFM2 SafeTensors directory
      generates from its path alone (`cargo test -p sofuu-capi`);
      both skip cleanly when the model is absent, since no model is checked in.
- [x] Kotlin verified without a device: `make sdk-jvm-check` compiles this
      same `jni_bridge.c` for the host OS and runs 29 assertions through the
      real wrapper and runtime (deltas, one terminal `done`, cancel after 3
      deltas → `aborted:true`, positional `streamLocal`, the UTF-8
      round-trip). The Android side is compile/link-gated separately: NDK
      for `arm64-v8a` + `x86_64`, `:sofuu:compileDebugKotlin` under the
      shipped Kotlin 1.9.24, and a full `:sofuu:assembleDebug` AAR with the
      runtime staged.

### Milestone E2: Host-Native Tool Callback Bridge — SHIPPED (C ABI + Swift + Android app + Android SDK)
- [x] `sofuu_rt_register_host_tool(rt, name, description, parameters_json, cb,
      opaque)` in `include/sofuu_embed.h`, plus
      `sofuu_rt_host_tool_resolve(rt, call_id, result_json, error)`.
- [x] Callback contract: a JSON string is a synchronous result; **NULL means
      the host took the call** and will settle it later from any thread via
      resolve, delivered to the runtime's thread through `rt/host_poke.rs`
      (`uv_async_send` — the one libuv call safe from another thread). No
      buffer-ownership handoff either way.
- [x] The wait is bounded by the agent's tool timeout (30 s per tool by
      default, `toolTimeoutMs` per agent) — which is also the timer that keeps
      the event loop alive while a call is outstanding, so no extra uv handle
      is needed and a host that never answers gets a tool-timeout error rather
      than a hung turn.
- [x] Registered tools reach agents defined *after* registration (through
      `sofuu.tools.TOOLS`) **and** agents defined from the `agents` config key
      *before* it (that path defers the unknown-builtin check to run time via
      `allowHostTools`, because registration cannot happen until
      `sofuu_rt_new` returns).
- [x] Swift: `Sofuu.registerTool(name:description:parameters:handler:)` with an
      `async throws` handler — the async shape the camera/GPS examples need.
- [x] Android: `SofuuBridge.registerHostTool` / `resolveHostTool` plus
      `DeviceTools` (device_info, vibrate, …) in `sofuu-android/`; the
      shipped SDK AAR (`bindings/android/sofuu`) has the same pair with
      `HostTool.invoke` returning null for the deferred case, and `close()`
      releasing the JNI global refs the registrations pin on the host's
      Activity.
- [x] Verified: a registered host tool runs inside a real agent turn and its
      result reaches the model's transcript (the mock provider reports
      CONFIRMED only if it sees it); a call deferred to another thread resumes
      the same turn; registration validation, unknown-id refusal and the
      JSON-string-vs-object shaping rule are all asserted
      (`cargo test -p sofuu-capi host_tool`).

### Milestone E3: iOS XCFramework Build & Metal GPU Validation — SHIPPED (engine, frameworks, simulator); device run parked
- [x] **The Metal engine compiles for iOS.** `crates/sofuu-ffi/build.rs` now
      builds `src/metal/metal_engine.mm` for `macos` **and** `ios` (device and
      simulator both report `CARGO_CFG_TARGET_OS=ios`), and the 97
      `#[cfg(target_os = "macos")]` gates across the inference engine
      (`lfm2.rs`, `block_graph.rs`, `mmap.rs`, `infer/mod.rs`) widened to
      `any(macos, ios)`. `nm` on the iOS device archive now shows the 21
      defined `sofuu_metal_*` functions (40 symbol mentions in all); before
      this it had none, and even a
      config that asked for GPU layers was a no-op on iOS — `finalize_device`
      and the Lfm2 arm of `set_gpu_layers` were themselves macOS-gated, so
      `with_local_engine` (the `ai.streamLocal` path, `enable_gpu = true`)
      silently ran CPU-only on an iPhone.
- [x] **The shader has one source of truth, and it is compiled ahead of
      time.** The `.mm` used to carry its own copy of the shader as a C
      string, and the copies had already drifted: `src/metal/kernels.metal`
      was missing the two `gemv_mlx4_affine_f16/_bf16` kernels the engine
      loads, so a metallib built from the file would have shipped without
      exactly the pipelines the engine asks for. Now `kernels.metal` is
      canonical — `build.rs` generates the runtime-fallback header from it,
      and `scripts/dist/ios.sh` compiles it into
      `Sofuu.framework/Resources/default.metallib` per slice (asserting all
      eight kernel names are present). The engine prefers the bundled
      library, verifies *every* pipeline it needs, and otherwise compiles
      the embedded copy at init — a stale or partial metallib degrades to
      the fallback instead of a nil pipeline at inference time.
      `scripts/mobile/typecheck_ios.sh` now compiles kernels.metal against
      the iOS SDK, so a shader that cannot build fails the check, not the
      user's first prompt.
- [x] **The MSL floor is 3.0, and the bf16 kernels run on the simulator.**
      Two premise corrections from probing, in order. First: the bf16 kernels
      originally used the MSL `bfloat` builtin, which `-std=metal3.0` rejects
      (probed for both SDKs) but which *also* fails at execution on the iOS
      simulator's GPU — the pipeline compiles and loads, then the command
      buffer ends `MTLCommandBufferErrorInternal` (status 5) and the output
      buffer is left untouched. Second: `bfloat` is not needed at all.
      bf16→f32 is a shift by 16 bits, and `kernels.metal` now spells that out
      (`as_type<float>((uint)h << 16)`), which is the same value for every bit
      pattern and executes everywhere. The shader therefore compiles at
      `-std=metal3.0` for macOS, the simulator and the device — matching the
      app's iOS 16.0 deployment target instead of shipping a metallib an
      iOS 16 device cannot load — and the engine prefers the precompiled
      library on every platform.
- [x] **Bugs this validation found in the engine, fixed.** (1) A failed
      command buffer was silent: every GPU entry point read its output buffer
      back after `waitUntilCompleted` without checking the status, so the
      bf16 failure above surfaced as *zeros returned as results*. All 14 wait
      sites now go through `wait_and_report()`, which logs the kernel name and
      driver error. (2) `MTLSimDriver` (the simulator's Metal driver) exports
      buffer storage over XPC shared memory, and its no-copy path hands the
      caller's pointer straight to `xpc_shmem_create`, which **aborts the
      process** (`_xpc_api_misuse`, SIGTRAP — not an error return) when the
      range is not one exportable mapping. Allocator memory is exactly that
      case: measured in a simulator probe, a 2 MiB `malloc` returns a
      page-aligned pointer whose enclosing vm region reports as a single
      16 KiB page. So `sofuu_metal_buffer_create` takes the driver's copying
      path under `TARGET_OS_SIMULATOR` (the driver mmaps its own region for
      it, verified 1 KiB…32 MiB including non-page-multiple lengths), and
      `register_zerocopy_buffer` falls back to a copy whenever no-copy is
      unavailable instead of dropping the GPU path.
- [x] **Verified on the iOS simulator, end to end.** `cargo test
      --target aarch64-apple-ios-sim` (through `xcrun simctl spawn booted`,
      with a `default.metallib` next to the test binary) is green:
      `test_metal_engine` logs "using the bundled default.metallib
      (precompiled kernels)", which only prints after every one of the eight
      pipeline states was created from it, and then passes bf16 GEMV
      (y[0] = y[500] = 1024.0), SwiGLU (5.284783), batched QKV and the
      asynchronous-overlap check in 0.52 s. An independent probe against the
      *shipped* metallib confirms `gemv_f32`, `gemv_f16` and `swiglu` return
      exact results with every command buffer Completed; macOS stays green
      with the same assertions (0.06 s).
- [x] **The framework and the app ship the library.** `scripts/dist/ios.sh
      all` builds both slices' `default.metallib` (40 KB each, all eight
      kernel names asserted present at pack time) into
      `Sofuu.framework/Resources/`, and the modulemap's
      `link framework "Metal"` is what lets a consumer link at all —
      `make ios-app` now reports BUILD SUCCEEDED with `Sofuu.debug.dylib`
      defining all 21 `sofuu_metal_*` functions and linking
      `Metal.framework` while never passing `-framework Metal` itself. The
      app bundle also carries its own `default.metallib` (8/8 kernels), so a
      real app takes the precompiled path, and the built app installs and
      launches on the simulator.
- [ ] Physical-device token generation through `sofuu-ios/`. Needs a signing
      team: `project.yml` sets `CODE_SIGN_STYLE: Automatic` with no
      `DEVELOPMENT_TEAM`, and nothing is provisioned. Verification today is
      the iOS **simulator** — the real Metal API, GPU, and our iOS-compiled
      engine (see above) — and the *device slice* is verified as a build
      artifact only: the archive defines the same 21 `sofuu_metal_*`
      functions, reports `platform: IOS`, and carries no macOS objects. Note
      that iOS 15/16 GPU support is now *permitted* (MSL 3.0) but untested on
      real older hardware — nothing has run on a device.

### Milestone E4: Android Gradle AAR & NDK Verification — OPEN
- [x] The engine cross-builds for `arm64-v8a` / `x86_64`
      (`scripts/mobile/build_engine_android.sh`, QTSQ-linked) and the app
      assembles a debug APK containing both ABIs.
- [ ] **Correcting the premise:** the app's CMake does *not* build
      `libsofuu.so` — it imports the prebuilt `jniLibs/<abi>/libsofuu.so` and
      compiles only the JNI shim. `-march=armv8.2-a+dotprod` is real but comes
      from `crates/sofuu-ffi/build.rs:144` (cc crate, all aarch64 targets), not
      from an NDK/CMake flag.
- [ ] Publish a real AAR: `bindings/android/sofuu` has no `maven-publish`
      block, so `com.sofuu:sofuu-android` resolves nowhere. Needs a Maven
      repository (a Line-1 revenue prerequisite per `PLAN-POSITIONING` §6).
- [ ] Physical-device or emulator JNI streaming run — needs hardware. No `adb`
      session is recorded anywhere in the repo.

### Milestone E5: Mobile Soak & Jetsam Memory Validation — OPEN, TARGET REFRAMED
- The original target ("runtime RSS strictly ≤ 30 MB excluding memory-mapped
  model weights") cannot hold as written: the KV cache is f32 and sized to the
  full context at load — ~940 MB for Qwen3-0.6B at the 4096-token default —
  and that is not model weights. Measure first, then reduce:
- [ ] Instrument `phys_footprint` (the value Jetsam actually limits) through
      the C ABI, report runtime / weights / KV separately, and replace this
      target with the measured numbers.
- [ ] Then reduce: f16 or quantized KV, a smaller mobile default context, an
      engine unload path (`LOCAL_ENGINE` in `rt/ai.rs` has none today), and a
      response to memory pressure (`didReceiveMemoryWarning`, `onTrimMemory`).
- [ ] 1000-token soak under Instruments (Allocations & Leaks), plus a
      `sofuu_rt_new`/`sofuu_rt_free` cycle loop. No soak harness exists today.
- [ ] Verify zero App Store compliance warnings regarding executable memory
      (QuickJS is a pure interpreter with no JIT, so this should hold — it just
      has never been checked against a real submission).
