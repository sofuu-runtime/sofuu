# ──────────────────────────────────────────────────────────────────
#  Sofuu (素風) — AI-Native JS Runtime
#  Build System: cargo-first (the only maintained build since M10)
# ──────────────────────────────────────────────────────────────────
#
# `make`           → Rust-first build (cargo; build.rs compiles the C
#                    engine layer: QuickJS + SIMD + http-parser)
# `make test`      → JS parity suite
# `make size-check`→ fail if binary exceeds SIZE_LIMIT (default 5MB)
# `make c-only`    → REMOVED in M10 (the pure-C binary died; see
#                    PLAN-RUST-MIGRATION.md) — prints an error
# `make linux`     → cross-compile via scripts/cross (cargo + zig)

TARGET  = sofuu
ARCH   := $(shell uname -m)

# Ubuntu/Linux arm64 reports as 'aarch64'
ifeq ($(ARCH), aarch64)
    ARCH = arm64
endif

# QTSQ codec checkout (mandatory core component).
# The cargo build reads SOFUU_QTSQ_DIR; point at a checkout, or let the
# default local checkout resolve:
#   SOFUU_QTSQ_DIR=~/projects/<checkout> make
#
# Default to local checkout if present
# PC-07 (AUDIT-2026-10-29): no absolute developer path here — `$(HOME)/…` is
# the same checkout expressed reusably, so a committed file carries no
# individual's home directory (the same rule the QTSQ probe in
# crates/sofuu-core/build.rs follows).
SOFUU_QTSQ_DIR ?= $(firstword $(wildcard $(HOME)/projects/black-hole-disk))
QTSQ_LIB_GUARD = $(firstword $(wildcard $(SOFUU_QTSQ_DIR)/libqtsq.a $(SOFUU_QTSQ_DIR)/libqtsq.dylib))

# ──────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────

.PHONY: all clean install test protocol-test serve-test c-only size-check bc-check cargo-build \
	observability-stress \
        deps scoreboard scoreboard-check \
        libsofuu zig-install linux linux-x86_64 linux-arm64 windows-x86_64 \
        release-archives dist \
        headless-test abi-check dist-macos dist-linux dist-ios dist-android dist-all \
        sdk-jvm-check android-sdk android-aar android-publish-local \
        desktop desktop-dev desktop-clean \
        ios-engine ios-engine-deps ios-project ios-typecheck ios-app ios-sim

SIZE_LIMIT ?= 5767168   # 5.5 MB hard cap (bytes)

all: $(TARGET)

# ── Rust-first build (default and only maintained build) ────────
#
# PC-07 (AUDIT-2026-10-29): every artifact built here must be independent of
# the machine that built it. rustc records the path it is handed into panic
# locations and `file!()`, and cargo hands it absolute paths — so a release
# binary shipped 87 absolute build-host paths (`~/.cargo/registry/...`,
# `~/.rustup/...`, `<checkout>/crates/...`). The prefixes need $HOME/$PWD,
# which cargo cannot expand from `.cargo/config.toml`, so they are computed
# by scripts/cross/remap_flags.sh and passed through RUSTFLAGS here (and in
# the cross scripts). A RUSTFLAGS variable replaces the config list, which is
# why the script also emits the generic prefixes the config carries.
SOFUU_REMAP_FLAGS := $(shell bash scripts/cross/remap_flags.sh 2>/dev/null)

cargo-build:
	@if [ -z "$(SOFUU_QTSQ_DIR)" ]; then \
		echo "  \033[31m✗ SOFUU_QTSQ_DIR is not set.\033[0m"; \
		echo "    QTSQ is a mandatory core component of Sofuu."; \
		echo "    Build with SOFUU_QTSQ_DIR set to a built QTSQ checkout:"; \
		echo "      SOFUU_QTSQ_DIR=~/projects/<qtsq-checkout> make"; \
		exit 1; \
	elif [ -z "$(QTSQ_LIB_GUARD)" ]; then \
		echo "  \033[31m✗ no libqtsq.a/libqtsq.dylib in SOFUU_QTSQ_DIR=$(SOFUU_QTSQ_DIR)\033[0m"; \
		echo "    QTSQ is a mandatory core component of Sofuu."; \
		exit 1; \
	fi
	RUSTFLAGS="$(SOFUU_REMAP_FLAGS) $$RUSTFLAGS" cargo build --release

$(TARGET): cargo-build
	@echo "  \033[36mCARGO\033[0m  $@  [arch=$(ARCH)]"
	@cp target/release/$(TARGET) $(TARGET)
	@if [ "$$(uname -s)" = "Darwin" ]; then strip $(TARGET) 2>/dev/null || true; codesign --force --sign - $(TARGET) 2>/dev/null || true; fi
	@echo "  \033[32mBuilt:\033[0m ./$(TARGET) ($(shell du -sh $(TARGET) | cut -f1))"

# ── Legacy pure-C build ─────────────────────────────────────────
# REMOVED in M10: every src/*.c glue file is retired (the runtime is Rust
# over the C engine layer — QuickJS/libuv/SIMD/http-parser). The C files
# the old ALL_SRCS referenced no longer exist.
c-only:
	@echo "  \033[31m✗ make c-only was removed in M10 (PLAN-RUST-MIGRATION.md).\033[0m"
	@echo "    The runtime is fully Rust (crates/) over the C engine layer;"
	@echo "    run \`make\` (cargo-first) instead."
	@exit 1

# ── Shipped .bc snapshots must match their sources ──────────────
# Release carries bytecode only (see shipped.rs), so a stale snapshot would
# ship stale driver code with no runtime signal. Byte comparison, not a hash
# file: QuickJS bytecode is deterministic for a given binary.
bc-check: $(TARGET)
	@./scripts/check-driver-bc.sh

# ── Vendored C dependencies ───────────────────────────────────────────
# The Rust build links libuv from deps/libuv/build; a fresh clone has no
# build tree, so `make` alone panics in sofuu-ffi/build.rs. This is the
# documented first step for a source build (CI runs the same two cmake
# commands).
deps:
	cmake -S deps/libuv -B deps/libuv/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build deps/libuv/build -j

# ── P6: published conformance/stability + parity scoreboard ──────────
# Generated from tracked receipts (contract/scoreboard/, stability.json,
# adoption/ledger.json); the check variant is the CI regression gate.
scoreboard:
	@python3 scripts/scoreboard.py

scoreboard-check:
	@python3 scripts/scoreboard.py --check

# ── Distribution dry-run (funnel plan F3): every channel's LOCAL half ──
# SwiftPM manifest+checksum, podspec lint, Gradle evaluate, npm pack --dry-run.
# Live pushes (pod trunk, Maven Central, npm publish) remain owner-gated.
dist-dryrun: libsofuu
	@bash scripts/dist-dryrun.sh

# ─── Universal plugin & extensibility system (PLUGIN_PLAN.md) ──────
# The 5-shield plugin runtime ships INSIDE the binary (src/js/plugins.js ->
# plugins.bc.gz plus the provider adapters driver), so these targets exercise
# the shipped engine, not a JS test double.
#
#   plugin-check   — the SDK typings cannot drift from the runtime: every
#                    documented export must exist (sofuu.d.ts parity), and
#                    the cookbook plugins must strip+load.
#   plugin-stress  — the adversarial suite: the 5 shields, concurrency,
#                    providers, mirror writes, service hooks, OpenAPI/YAML.
plugin-check: $(TARGET)
	@./$(TARGET) run scripts/stress/plugin_types_check.js
	@for f in plugins/*.ts; do ./$(TARGET) plugin validate "$$f" >/dev/null || exit 1; done; \
	echo "  ✓ cookbook plugins validate"

plugin-stress: $(TARGET)
	@./$(TARGET) run scripts/stress/plugin_stress.js

plugin-stress-fast: $(TARGET)
	@./$(TARGET) run scripts/stress/plugin_stress.js --fast

# OBSERVABILITY_PLAN: the trace substrate is the recovery path for 24/7 runs,
# so it gets the same adversarial treatment as the plugin layer — plus the
# CLI round-trip (logs / last-error / resume) that an operator actually types.
observability-stress: $(TARGET)
	@./$(TARGET) run scripts/stress/observability_stress.js
	@bash scripts/stress/observability_e2e.sh

# ── Size cap check (CI: fails if binary exceeds 5MB) ────────────
size-check: $(TARGET)
	@SIZE=$$(stat -f%z $(TARGET) 2>/dev/null || stat -c%s $(TARGET) 2>/dev/null); \
	if [ "$$SIZE" -gt "$(SIZE_LIMIT)" ]; then \
		echo "  \033[31m✗ Size $$SIZE bytes exceeds cap $(SIZE_LIMIT) bytes\033[0m"; \
		exit 1; \
	else \
		echo "  \033[32m✓ Size $$SIZE bytes ≤ $(SIZE_LIMIT) bytes\033[0m"; \
	fi

# ── Sofuu Desktop (PLAN-DESKTOP) ─────────────────────────────────
# Tauri v2 macOS app: Rust backend links sofuu-core in-process, the
# React/TS frontend is strictly UI. The 5 MB cap above is CLI-only —
# the .app bundle is expected to be ~30–60 MB and has NO size gate.
DESKTOP_DIR = sofuu-desktop

desktop:
	@echo "  \033[36mTAURI\033[0m  sofuu-desktop (release .app/.dmg)"
	cd $(DESKTOP_DIR) && npm install --no-audit --no-fund && npm run tauri build
	@echo "  \033[32m✓\033[0m bundle: target/release/bundle/ (src-tauri is a workspace member)"

desktop-dev:
	@echo "  \033[36mTAURI\033[0m  sofuu-desktop (dev)"
	cd $(DESKTOP_DIR) && npm install --no-audit --no-fund && npm run tauri dev

desktop-clean:
	rm -rf $(DESKTOP_DIR)/dist

# ── header sync: one source of truth for the C ABI ───────────────
# include/sofuu_embed.h is the contract. The mobile apps, the Android SDK
# binding and the iOS typecheck stub carry copies because they build outside
# this tree (Xcode, Gradle, Android Studio). Those copies drifted: the
# Android one still advertised a config key the header had removed, and the
# iOS stub lagged whole functions — so a binding could typecheck against a
# contract that no longer existed. Copies are byte-identical and refreshed
# here; never edit one by hand.
HEADER_COPIES = dist/sofuu_embed.h \
                sofuu-ios/.devstub/sofuu_embed.h \
                bindings/android/sofuu/src/main/cpp/include/sofuu_embed.h \
                sofuu-android/app/src/main/cpp/include/sofuu_embed.h

header-sync:
	@mkdir -p dist
	@for f in $(HEADER_COPIES); do \
		if [ -d "$$(dirname $$f)" ]; then \
			cp include/sofuu_embed.h "$$f"; \
			echo "  \033[36mSYNC\033[0m  $$f"; \
		fi; \
	done

# ── libsofuu: embeddable library (PLAN-HEADLESS H1) ──────────────
# Builds the capi crate as staticlib + cdylib, copies artifacts to dist/.
# Usage:  make libsofuu
# Output: dist/libsofuu.a  dist/libsofuu.dylib  dist/sofuu_embed.h
# The EMBEDDED dylib unwinds at the ABI boundary (panic → SOFUU_ERR_PANIC
# via catch_unwind, never a dead host app). The `sofuu` CLI binary keeps
# the workspace default panic=abort — this override applies to this build
# of the capi graph only.
libsofuu: header-sync
	@echo "  \033[36mCARGO\033[0m  libsofuu  [arch=$(ARCH)]"
	CARGO_PROFILE_RELEASE_PANIC=unwind RUSTFLAGS="$(SOFUU_REMAP_FLAGS) $$RUSTFLAGS" \
		cargo build --release -p sofuu-capi
	@mkdir -p dist
	@cp target/release/libsofuu_capi.a dist/libsofuu.a 2>/dev/null || \
		cp target/release/libsofuu.a dist/libsofuu.a 2>/dev/null || true
	@cp target/release/libsofuu_capi.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu_capi.so dist/libsofuu.so 2>/dev/null || \
		cp target/release/libsofuu.so dist/libsofuu.so 2>/dev/null || true
	@echo "  \033[32mBuilt:\033[0m"
	@ls -lh dist/libsofuu.* dist/sofuu_embed.h 2>/dev/null
	@echo ""
	@echo "  Symbol exports:"
	@nm dist/libsofuu.a 2>/dev/null | grep ' T _sofuu_' | sort || \
		nm -gU dist/libsofuu.dylib 2>/dev/null | grep ' T _sofuu_' | sort || \
		echo "  (nm not available — run manually)"

install: $(TARGET)
	@echo "  Installing sofuu to /usr/local/bin/"
	cp $(TARGET) /usr/local/bin/$(TARGET)
	@echo "  Done. Run: sofuu help"

clean:
	@echo "  Cleaning..."
	rm -f $(TARGET)
	cargo clean 2>/dev/null || true

test: $(TARGET)
	@echo ""
	@echo "\033[1m=== Sofuu Test Suite ===\033[0m"
	@echo ""
	@if [ ! -f tests/run_js_tests.sh ]; then \
		echo "\033[33m— tests/ is local-only (gitignored); nothing to run\033[0m"; \
	else bash tests/run_js_tests.sh; fi

# ─── Workstream 6 conformance (protocols/sandbox/tooling) ─────────
# Local-only runner (tests/ is gitignored). Not wired into CI yet: KNOWN-GAP
# lines are the later phases' acceptance tests and stay visible until they
# land. See docs/PLAN-PROTOCOLS-SANDBOX-TOOLING.md §5 (T5).
protocol-test: $(TARGET)
	@if [ ! -x tests/protocol_conformance_run.sh ]; then \
		echo "\033[33m— tests/ is local-only (gitignored); nothing to run\033[0m"; \
	else SOFUU=$(TARGET) tests/protocol_conformance_run.sh; fi

# H1–H3 acceptance: drives sofuu.serve.createHandler() with a fake stream
# (no model, no sockets) plus a doc↔code route check.
serve-test: $(TARGET)
	@if [ ! -x tests/serve_test_run.sh ]; then \
		echo "\033[33m— tests/ is local-only (gitignored); nothing to run\033[0m"; \
	else SOFUU=$(TARGET) tests/serve_test_run.sh; fi

# ─── Cross-compilation (cargo + zig; local convenience) ────────────
# Uses Zig as a zero-dependency cross-compiler (no Docker needed).
# First run:  make zig-install
# Then:       make linux  OR  make linux-x86_64  OR  make linux-arm64
#             make windows-x86_64   (.exe, Schannel TLS, no Docker/MSVC)
# NOTE: cross builds are QTSQ-free (the local checkout is macOS-only).

ZIG_INSTALL = scripts/cross/install_zig.sh
CROSS_SCRIPT = scripts/cross/cross_compile.sh

zig-install:
	@bash $(ZIG_INSTALL)

linux: zig-install
	@bash $(CROSS_SCRIPT) all

linux-x86_64: zig-install
	@bash $(CROSS_SCRIPT) x86_64

linux-arm64: zig-install
	@bash $(CROSS_SCRIPT) arm64

windows-x86_64: zig-install
	@bash scripts/cross/build_windows_x86_64.sh

release-archives: linux
	@echo "Archives are in dist/"
	@ls -lh dist/*.tar.gz 2>/dev/null || echo " (none yet)"

dist: all linux
	@mkdir -p dist
	@cp $(TARGET) dist/sofuu-darwin-arm64
	@stage=$$(mktemp -d); cp dist/sofuu-darwin-arm64 $$stage/; \
	  bash scripts/pack/stage_legal.sh $$stage . >/dev/null 2>&1 || true; \
	  tar -czf dist/sofuu-darwin-arm64.tar.gz -C $$stage .; rm -rf $$stage
	@echo "  \033[32m✓\033[0m dist/sofuu-darwin-arm64.tar.gz"
	@bash scripts/pack/checksums.sh
	@ls -lh dist/*.tar.gz

# ── H4: Platform packs (libsofuu for macOS, Linux, iOS, Android) ───
# Each produces dist/libsofuu-<platform>.{a,dylib/so} + dist/sofuu_embed.h.
# Run `make dist-all` to build every platform available on this host.
dist-macos:
	@bash scripts/dist/macos.sh all

dist-linux:
	@bash scripts/dist/linux.sh all

dist-ios:
	@bash scripts/dist/ios.sh all

dist-android:
	@bash scripts/dist/android.sh all

# ── Android SDK AAR (com.sofuu:sofuu-android) ───────────────────────
# The publishable artifact: QTSQ-linked native packs staged with the JNI shim
# and libc++_shared, then the release AAR. QTSQ is core, not optional.
# Credentials are operator-owned.
android-sdk: dist-android
	@bash scripts/mobile/stage_android_sdk.sh

android-aar:
	@cd bindings/android && gradle :sofuu:assembleRelease

android-publish-local:
	@cd bindings/android && gradle :sofuu:publishToMavenLocal

# ── Sofuu Mobile (Android app) ─────────────────────────────────────
# `make mobile-engine`: build the QTSQ-linked libsofuu per Android ABI
# (libuv + OpenSSL + curl + QTSQ + sofuu-capi) and stage the .so's into
# sofuu-android/app/src/main/jniLibs. Then build the app with gradle:
#   cd sofuu-android && ./gradlew :app:assembleDebug
mobile-engine:
	@bash scripts/mobile/build_engine_android.sh all

# Release engine (.so stripped, opt-z + fat LTO) — what a shipping APK
# links. Needs the runtime workspace to compile (cargo build --release).
mobile-engine-release:
	@SOFUU_ANDROID_PROFILE=release bash scripts/mobile/build_engine_android.sh all

# The app itself (Kotlin/Compose).
mobile-apk:
	@cd sofuu-android && ./gradlew :app:assembleDebug

# ── Sofuu Mobile (iOS app) ─────────────────────────────────────────
# The iOS app lives in sofuu-ios/ (SwiftUI + Swift concurrency, linking
# libsofuu through its C ABI — no JNI). Its Xcode project is generated from
# sofuu-ios/project.yml, so the .xcodeproj is a build artifact, not source.
#
#   make ios-typecheck   Swift + Metal check, ~3s, no engine needed
#   make ios-engine-deps cross-build OpenSSL + libcurl for both slices
#   make ios-engine      build dist/libsofuu.xcframework (device + sim)
#   make ios-project     regenerate Sofuu.xcodeproj from project.yml
#   make ios-app         build the app for the simulator
#
# `ios-typecheck` first: it needs nothing but Xcode, and it is where every
# Swift error will actually show up.
ios-typecheck:
	@bash scripts/mobile/typecheck_ios.sh sim

# Needs `brew install xcodegen`.
ios-project:
	@cd sofuu-ios && xcodegen generate

# The third-party TLS stack (OpenSSL → libcurl) for both iOS slices. iOS has
# no system libcurl and no system OpenSSL, so unlike Android these must be
# cross-built before the engine can link. `ios-engine` invokes this
# automatically; it is a separate target only so the slow, one-time step can
# be run and cached on its own.
ios-engine-deps:
	@bash scripts/dist/ios.sh deps

# Device + simulator slices → dist/libsofuu.xcframework, which project.yml
# links. Builds the iOS OpenSSL/libcurl stack first if it is not cached.
ios-engine:
	@bash scripts/dist/ios.sh all

ios-app: ios-project
	@cd sofuu-ios && xcodebuild -project Sofuu.xcodeproj -scheme SofuuApp \
		-sdk iphonesimulator -configuration Debug build

# Boot the app on an already-running simulator (open Simulator.app first).
ios-sim: ios-app
	@cd sofuu-ios && xcrun simctl install booted \
		"$$(xcodebuild -project Sofuu.xcodeproj -scheme SofuuApp -sdk iphonesimulator \
			-configuration Debug -showBuildSettings 2>/dev/null \
			| awk '/ BUILT_PRODUCTS_DIR /{print $$3}')/Sofuu.app" && \
		xcrun simctl launch booted com.sofuu.app

dist-all:
	@bash scripts/dist/all.sh

# ── H6: Headless CI gates ──────────────────────────────────────────
# `make headless-test`: build libsofuu + compile and RUN every sample in
#   examples/headless/ against dist/ (each one asserts its own calls).
# `make abi-check`: diff exported symbols against the checked-in baseline.
#
# examples/ is local-only (gitignored), so a fresh clone has nothing to run.
# That decision has to be made at MAKE level, not inside the recipe: each
# recipe line gets its own shell, so an `exit 0` in one line only ends that
# line's shell and every following line still runs — which is how this gate
# went red in CI ("nothing to run", then clang on a missing file).
HEADLESS_SAMPLES := $(wildcard examples/headless/*.c)

# bc freshness is a HARD dependency of the dist path: a release build
# embeds src/js/<driver>.bc with no source fallback, so a snapshot lags its
# .js → stale code ships with zero runtime signal (QTSQ Phase 6 guard).
headless-test: libsofuu bc-check
ifeq ($(strip $(HEADLESS_SAMPLES)),)
	@echo "\033[33m— examples/ is local-only (gitignored); nothing to run\033[0m"
else
	@echo ""
	@echo "\033[1m=== Headless embedding test (H6) ===\033[0m"
	@echo ""
	@echo "\033[36m--- Compile c_embed.c against libsofuu ---\033[0m"
	@cc examples/headless/c_embed.c -Idist -Ldist -lsofuu -o examples/headless/c_embed \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_embed against the library ---\033[0m"
	@examples/headless/c_embed
	@echo ""
	@echo "\033[36m--- Compile c_embed_vec.c against libsofuu ---\033[0m"
	@cc examples/headless/c_embed_vec.c -Idist -Ldist -lsofuu -o examples/headless/c_embed_vec \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_embed_vec)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_embed_vec against the library ---\033[0m"
	@examples/headless/c_embed_vec
	@echo ""
	@echo "\033[36m--- Compile c_rlm.c against libsofuu ---\033[0m"
	@cc examples/headless/c_rlm.c -Idist -Ldist -lsofuu -o examples/headless/c_rlm \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_rlm)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_rlm against the library ---\033[0m"
	@examples/headless/c_rlm
	@echo ""
	@echo "\033[36m--- Compile c_agent.c against libsofuu ---\033[0m"
	@cc examples/headless/c_agent.c -Idist -Ldist -lsofuu -o examples/headless/c_agent \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_agent)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_agent against the library ---\033[0m"
	@examples/headless/c_agent
	@echo ""
	@echo "\033[36m--- Compile c_host_tool.c against libsofuu ---\033[0m"
	@cc examples/headless/c_host_tool.c -Idist -Ldist -lsofuu -o examples/headless/c_host_tool \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_host_tool)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_host_tool against the library (E2 host tools) ---\033[0m"
	@examples/headless/c_host_tool
	@echo ""
	@echo "\033[36m--- Compile c_llm.c against libsofuu (real LLM, mock in CI) ---\033[0m"
	@cc examples/headless/c_llm.c -Idist -Ldist -lsofuu -o examples/headless/c_llm \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_llm)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_llm against the library (no key → mock) ---\033[0m"
	@examples/headless/c_llm
	@echo ""
	@echo "\033[36m--- Compile c_soak.c against libsofuu (host-abuse soak) ---\033[0m"
	@cc examples/headless/c_soak.c -Idist -Ldist -lsofuu -o examples/headless/c_soak \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_soak)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo "\033[36m--- Run c_soak (host-abuse soak, 12 fresh runtimes) ---\033[0m"
	@examples/headless/c_soak
	@echo ""
	@echo "\033[32m✓ Headless tests passed\033[0m"
	@echo ""
endif

# ── SDK JVM check: the Android AAR wrapper on the host JVM ─────────
# Compiles the AAR's JNI shim for the host OS and drives the real Kotlin
# wrapper through the real runtime — streaming deltas, mid-flight cancel,
# the UTF-8 boundary, host tools (sync + deferred) — with no device
# attached. Needs `make libsofuu` first; skips cleanly (exit 0, yellow
# line) when a host toolchain piece is missing.
sdk-jvm-check:
	@bash scripts/mobile/sdk_jvm_check.sh

abi-check: libsofuu
	@# Use the dylib/.so (public symbols are resolved at link time, not in the .a archive).
	@if [ -f dist/libsofuu.dylib ]; then LIB=dist/libsofuu.dylib; \
	 elif [ -f dist/libsofuu.so ]; then LIB=dist/libsofuu.so; \
	 else echo "\033[31m✗ No shared lib in dist/\033[0m"; exit 1; fi; \
	bash scripts/check_abi.sh $$LIB

# ── Sofuu macOS (SwiftUI app over the C ABI) ───────────────────────
# Engine artifact + typecheck for the native Mac app. Additive targets —
# the Tauri desktop targets above stay until the Swift app reaches parity.
# Output lives in sofuu-macos/engine/ (gitignored), NOT dist/ (release flows
# own that directory).
macos-engine:
	@bash scripts/dist/macos-engine.sh arm64

macos-typecheck:
	@bash sofuu-macos/scripts/typecheck.sh
