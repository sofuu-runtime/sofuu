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
* **Metal GPU Acceleration**: Shaders in `src/metal/kernels.metal` compile into `default.metallib` bundled within the XCFramework, executing matrix multiplications directly on Apple Silicon A16/A17/M-series GPUs.
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

### Milestone E1: Native Local Inference in `sofuu_rt_call_stream`
- [ ] Connect `sofuu.ai.streamLocal()` directly into the C ABI streaming pump in `crates/sofuu-capi/src/lib.rs`.
- [ ] Allow SDK callers to pass local model paths (`.gguf` or `.safetensors` directory) and receive token deltas.

### Milestone E2: Host-Native Tool Callback Bridge
- [ ] Implement `sofuu_rt_register_host_tool(rt, name, desc, schema_json, callback, opaque)` in `include/sofuu_embed.h`.
- [ ] Add Swift and Kotlin wrappers enabling mobile developers to register native functions as agent tools.

### Milestone E3: iOS XCFramework Build & Metal GPU Validation
- [ ] Verify `scripts/dist/ios.sh` compiling `libsofuu.xcframework` with Metal GPU shaders included.
- [ ] Test real token generation on physical iPhone hardware via the `sofuu-ios/` SwiftUI sample app.

### Milestone E4: Android Gradle AAR Build & NDK Verification
- [ ] Verify `sofuu-android/` CMake build generating `libsofuu.so` for `arm64-v8a`.
- [ ] Test JNI token streaming in the Compose sample app on physical Android hardware.

### Milestone E5: Mobile Soak & Jetsam Memory Validation
- [ ] Run 1000-token continuous generation under Xcode Instruments (Allocations & Leaks) to verify RSS stays strictly $\le 30\text{ MB}$.
- [ ] Verify zero App Store compliance warnings regarding executable memory.
