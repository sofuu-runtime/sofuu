# ──────────────────────────────────────────────────────────────────
#  Sofuu (素風) — AI-Native JS Runtime
#  Build System: cargo-first (the only maintained build since M10)
# ──────────────────────────────────────────────────────────────────
#
# `make`           → Rust-first build (cargo; build.rs compiles the C
#                    engine layer: QuickJS + SIMD + http-parser)
# `make test`      → real-resource gates + the blocked-suite report
# `make size-check`→ fail if binary exceeds SIZE_LIMIT (default 5.65 MiB = 5,924,454 B)
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

.PHONY: real-mcp-check real-memory-check real-kaze-memory-check real-kaze-memory-durable-check real-kaze-alert-check real-kaze-fail-closed-check real-kaze-gguf-check real-gates kaze-js-boundary-check js-swallow-check env-knobs-check swiftpm-consistency-check event-loop-check cold-start-bench all clean install test test-area test-rust-filter test-contract-case preflight protocol-test serve-test c-only size-check docs-lint bc-check cargo-build sandbox-stress \
	kaze-drift-check kaze-drift-strict kaze-embed embedded-kaze-gates clippy-ratchet \
	observability-stress asan fuzz-simd policy-dx modern-npm sbom sign-dist \
        deps scoreboard scoreboard-check cuda-check \
        libsofuu libsofuu-linux-x86_64 zig-install linux linux-x86_64 linux-arm64 windows-x86_64 \
        release-archives dist \
        headless-test abi-check dist-macos dist-linux dist-ios dist-android dist-all \
        sdk-jvm-check android-sdk android-aar android-publish-local \
        desktop desktop-dev desktop-clean \
        ios-engine ios-engine-deps ios-project ios-typecheck ios-app ios-sim

SIZE_LIMIT ?= 5924454   # 5.65 MiB hard cap (bytes)

# The artifact the real-resource gates exercise: an explicit SOFUU_BIN wins,
# then the debug build, then the stripped release copy at the repository root.
SOFUU_GATE_BIN = $(if $(SOFUU_BIN),$(SOFUU_BIN),$(if $(wildcard target/debug/sofuu),target/debug/sofuu,$(TARGET)))

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

# PLAN-UNIFIED-DIAGNOSTICS size trade (2026-11-10): ld64 emits an
# LC_FUNCTION_STARTS table (~26 KB in this binary) that only sampling
# profilers and external symbolication read. Release builds keep the 5.5 MiB
# gate instead; native abort backtraces are already documented as best-effort.
# Darwin-only: `-no_function_starts` is an Apple ld option.
SOFUU_LINK_FLAGS := $(shell [ "$$(uname -s)" = Darwin ] && echo "-C link-arg=-Wl,-no_function_starts")

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
	RUSTFLAGS="$(SOFUU_REMAP_FLAGS) $(SOFUU_LINK_FLAGS) $$RUSTFLAGS" cargo build --release

$(TARGET): cargo-build
	@echo "  \033[36mCARGO\033[0m  $@  [arch=$(ARCH)]"
	@rm -f $(TARGET) && cp target/release/$(TARGET) $(TARGET)
	@if [ "$$(uname -s)" = "Darwin" ]; then strip $(TARGET) 2>/dev/null || true; codesign --force --sign - $(TARGET) 2>/dev/null || true; fi
	@echo "  \033[32mBuilt:\033[0m ./$(TARGET) ($(shell du -sh $(TARGET) | cut -f1))"
	@if [ -f target/release/sofuu-agent ]; then \
		rm -f sofuu-agent && cp target/release/sofuu-agent sofuu-agent; \
		if [ "$$(uname -s)" = "Darwin" ]; then strip sofuu-agent 2>/dev/null || true; codesign --force --sign - sofuu-agent 2>/dev/null || true; fi; \
		echo "  \033[32mBuilt:\033[0m ./sofuu-agent ($$(du -sh sofuu-agent | cut -f1))"; \
	fi
	@if [ -f target/release/sofuu-micro ]; then \
		rm -f sofuu-micro && cp target/release/sofuu-micro sofuu-micro; \
		if [ "$$(uname -s)" = "Darwin" ]; then strip sofuu-micro 2>/dev/null || true; codesign --force --sign - sofuu-micro 2>/dev/null || true; fi; \
		echo "  \033[32mBuilt:\033[0m ./sofuu-micro ($$(du -sh sofuu-micro | cut -f1))"; \
	fi

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

# ── CUDA kernel gate (PLAN-GPU-CUDA C0/C1) ──────────────────────
# No GPU required: regenerates/verifies the checked-in PTX and runs the
# host-compiled kernel math against independent scalar references. The real
# GPU pass is scripts/cuda/kaggle_gpu_check.sh on Kaggle's T4.
cuda-check:
	@bash scripts/cuda/check_kernels.sh

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

# W6 (SANDBOX_UPGRADE_PLAN): the 26-case adversarial sandbox suite. Every
# case is bounded by fuel or a deadline, so a hung engine fails the run
# instead of wedging the terminal.
sandbox-stress: $(TARGET)
	@./$(TARGET) run scripts/stress/sandbox_stress.js

plugin-stress-fast: $(TARGET)
	@./$(TARGET) run scripts/stress/plugin_stress.js --fast

# OBSERVABILITY_PLAN: the trace substrate is the recovery path for 24/7 runs,
# so it gets the same adversarial treatment as the plugin layer — plus the
# CLI round-trip (logs / last-error / resume) that an operator actually types.
observability-stress: $(TARGET)
	@./$(TARGET) run scripts/stress/observability_stress.js
	@bash scripts/stress/observability_e2e.sh

# ── R7.P2: memory-safety gates on the JS→native boundary ─────────
# asan      — AddressSanitizer run of the validated-view + policy unit
#             tests. Nightly, -Zbuild-std (needs `rustup component add
#             rust-src`), host triple. Ubuntu CI runs this in
#             .github/workflows/asan.yml; macOS cannot inject the ASAN
#             runtime into rustc while it loads ASAN-built proc-macro
#             dylibs, so the target explains that instead of failing
#             with a misleading "Interceptors are not working".
# fuzz-simd — coverage-guided fuzzing of rt/view_guard (cargo-fuzz).
asan:
	@case "$$(uname -s)" in \
	  Darwin) echo "  ⚠ ASAN on macOS needs the runtime inside rustc for proc macros;"; \
	          echo "    run the ubuntu job: .github/workflows/asan.yml (make asan works on Linux)."; \
	          exit 0;; \
	esac
	@TRIPLE=$$(rustc -vV | sed -n 's/^host: //p'); \
	RUSTFLAGS="-Zsanitizer=address" ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
	cargo +nightly test -p sofuu-core --lib -Zbuild-std --target "$$TRIPLE" -- rt::view_guard

fuzz-simd:
	@if ! cargo +nightly fuzz --version >/dev/null 2>&1; then \
		echo "  ⚠ cargo-fuzz is not installed: cargo install cargo-fuzz"; exit 1; \
	fi
	@cargo +nightly fuzz run --fuzz-dir crates/fuzz fuzz_simd_bridge -- -max_len=64 -timeout=5

# ── R7.P3: policy developer-experience gates ────────────────────
# init profiles, static audit, dry-run, `why` under the layered policy, and
# the three guard flags enforced at the real gates.
policy-dx: $(TARGET)
	@bash scripts/stress/policy_dx_e2e.sh ./$(TARGET)

# ── PLAN-MODERN-NPM: npm compatibility + web streams ────────────
# Conditional "exports" (subpaths, wildcards, import vs require), WHATWG
# streams behind fetch(), and the actionable failure diagnostics. The probe
# runs against a real chunked SSE server, so streaming is proven by arrival
# time rather than by a resolved promise.
modern-npm: $(TARGET)
	@bash scripts/stress/modern_npm_e2e.sh ./$(TARGET)

# ── R7.P4: distribution integrity ───────────────────────────────
# sbom      — CycloneDX 1.5 SBOM from `cargo metadata` (stdlib Python only,
#             no cargo-cyclonedx install). Attached to every GitHub release.
# sign-dist — minisign-sign a release manifest:
#               SOFUU_MINISIGN_SECRET_KEY=~/sofuu.key make sign-dist
#             Writes <manifest>.minisig next to MANIFEST (default: dist/SHA256SUMS).
# tsbom := (no external deps)
sbom:
	@python3 scripts/sbom.py --out dist/sofuu-sbom.json

sign-dist:
	@sh scripts/sign_release.sh "$${MANIFEST:-dist/SHA256SUMS}"

# ── Size cap check (CI: fails if binary exceeds 5MB) ────────────
size-check: $(TARGET)
	@SIZE=$$(stat -f%z $(TARGET) 2>/dev/null || stat -c%s $(TARGET) 2>/dev/null); \
	if [ "$$SIZE" -gt "$(SIZE_LIMIT)" ]; then \
		echo "  \033[31m✗ Size $$SIZE bytes exceeds cap $(SIZE_LIMIT) bytes\033[0m"; \
		exit 1; \
	else \
		echo "  \033[32m✓ Size $$SIZE bytes ≤ $(SIZE_LIMIT) bytes\033[0m"; \
	fi

# ── Documentation freshness (no build required) ─────────────────
# Prose is not compiled, so a retired measurement only fails when a human reads it.
# Two of them reached the public site that way before anyone noticed. This target
# greps the live docs for numbers that have since gone false; the pattern list and
# the waiver convention live in scripts/check-doc-metrics.sh.
# Sources of truth: RULES.md §3.1 (size cap), BENCHMARKS.md (speed).
docs-lint:
	@./scripts/check-doc-metrics.sh

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

# rustc emits NO DT_SONAME for a Linux cdylib, so a host that links -lsofuu
# would bake the build-tree path into its DT_NEEDED. Name the artifact.
# (ifeq, not $(if ...): make splits the macro value on its commas.)
ifeq ($(shell uname -s),Linux)
SOFUU_SONAME_FLAGS := -C link-arg=-Wl,-soname=libsofuu.so
else
SOFUU_SONAME_FLAGS :=
endif

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
	CARGO_PROFILE_RELEASE_PANIC=unwind RUSTFLAGS="$(SOFUU_REMAP_FLAGS) $(SOFUU_SONAME_FLAGS) $$RUSTFLAGS" \
		cargo build --release -p sofuu-capi
	@mkdir -p dist
	@cp target/release/libsofuu_capi.a dist/libsofuu.a 2>/dev/null || \
		cp target/release/libsofuu.a dist/libsofuu.a 2>/dev/null || true
	@cp target/release/libsofuu_capi.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu_capi.so dist/libsofuu.so 2>/dev/null || \
		cp target/release/libsofuu.so dist/libsofuu.so 2>/dev/null || true
	@# LC_ID_DYLIB is the BUILD-TREE path of the cargo artifact, so a host that
	@# links -lsofuu would record that path (rpath cannot override an absolute
	@# id). Ship it under the relocatable id headless-test's `-Wl,-rpath,dist`
	@# and every embedding host expect; install_name_tool invalidates the ad-hoc
	@# signature, so re-sign it.
	@if [ -f dist/libsofuu.dylib ] && [ "$$(uname -s)" = Darwin ]; then \
		install_name_tool -id @rpath/libsofuu.dylib dist/libsofuu.dylib; \
		codesign -s - -f dist/libsofuu.dylib 2>/dev/null || true; \
	fi
	@echo "  \033[32mBuilt:\033[0m"
	@ls -lh dist/libsofuu.* dist/sofuu_embed.h 2>/dev/null
	@echo ""
	@echo "  Symbol exports:"
	@if [ "$$(uname -s)" = Darwin ]; then \
		nm -gU dist/libsofuu.dylib 2>/dev/null | grep ' T _sofuu_' | sort; \
	else \
		nm -D --defined-only dist/libsofuu.so 2>/dev/null | grep ' T sofuu_' | sort; \
	fi || echo "  (nm not available — run manually)"

install: $(TARGET)
	@echo "  Installing sofuu to /usr/local/bin/"
	cp $(TARGET) /usr/local/bin/$(TARGET)
	@echo "  Done. Run: sofuu help"

clean:
	@echo "  Cleaning..."
	rm -f $(TARGET) sofuu-agent sofuu-micro
	cargo clean 2>/dev/null || true

# Rust tests. SOFUU_ALLOW_LOOPBACK=1 is required, not optional: the crate's own
# http_client fixtures bind 127.0.0.1 and the SSRF gate (AUDIT-2026-11-08 §2.6)
# refuses loopback by default, so without the switch 19 of 24 transport tests
# fail with "loopback target refused" — a property of the gate, not of the
# transport. The JS suite sets the same variable for the same reason.
#
# Run it with the module filters to keep the engine tests from sharing one
# process (they serialize on rt::test_loop_lock; a few, e.g. the MCP child
# tests, can then stall each other). The plain form is what CI wants:
#   make test-rust
# and for a single area:
#   SOFUU_ALLOW_LOOPBACK=1 cargo test --release -p sofuu-core --lib rt::policy
test-rust:
	@bash scripts/preflight.sh rust

# Focused development checks never imply the full suite or release passed.
test-area:
	@bash scripts/test-area.sh "$(AREA)"

test-rust-filter:
	@test -n "$(FILTER)" || { echo 'Set FILTER to a non-empty Rust test name filter' >&2; exit 2; }
	@SOFUU_RUST_FILTER="$(FILTER)" bash scripts/preflight.sh rust

test-contract-case:
	@test -n "$(CASE)" || { echo 'Set CASE to an exact contract ID, for example T1.1' >&2; exit 2; }
	@SOFUU_CONTRACT_CASE="$(CASE)" bash scripts/contract/run_contract_tests.sh

# These targeted checks use real temporary resources. Set SOFUU_BIN to the
# candidate under test; missing prerequisites are reported as blocked.
real-mcp-check:
	@SOFUU_BIN="$(SOFUU_GATE_BIN)" python3 scripts/test-real-mcp-file.py

real-memory-check:
	@SOFUU_BIN="$(SOFUU_GATE_BIN)" python3 scripts/test-real-memory.py

real-kaze-memory-check:
	@python3 scripts/test-kaze-memory-real.py "$(SOFUU_GATE_BIN)"

real-kaze-memory-durable-check:
	@python3 scripts/test-kaze-memory-durable.py "$(SOFUU_GATE_BIN)"

real-kaze-alert-check:
	@python3 scripts/test-kaze-alert-real.py "$(SOFUU_GATE_BIN)"

real-kaze-fail-closed-check:
	@python3 scripts/test-kaze-fail-closed.py "$(SOFUU_GATE_BIN)"

real-kaze-gguf-check:
	@SOFUU_BIN="$(SOFUU_GATE_BIN)" python3 scripts/test-kaze-local-gguf-real.py

# Every real-resource gate in one run: a real MCP server, a real QTSQ store,
# the real Kaze CMA and durable stores, the real alert sinks, real local-GGUF
# inference and the event-loop benchmark. PASS / FAIL / BLOCKED per check; the
# exit code is 1 on a failure and 2 while a prerequisite is missing. A blocked
# check is a coverage gap, never a pass, and nothing is ever substituted.
real-gates:
	@bash scripts/preflight.sh real

kaze-js-boundary-check:
	@python3 scripts/test-kaze-js-boundaries.py "$(SOFUU_GATE_BIN)"

# The test suite runs the real-resource gates first, so a green run means real
# coverage, and then reports the suites that are still blocked. It exits 2
# while any of them is blocked: the JS corpus below still asserts against
# canned-response providers, so the aggregate is not green and is not run.
test: $(TARGET) kaze-drift-check clippy-ratchet js-swallow-check env-knobs-check swiftpm-consistency-check
	@echo ""
	@echo "\033[1m=== Sofuu Test Suite ===\033[0m"
	@echo ""
	@bash scripts/preflight.sh real serve js

# The full integration/release preflight. Focused changes use test-area or
# the relevant individual sections; see RULES §3.3 and TESTING.md.
preflight:
	@bash scripts/preflight.sh

# The Kaze surface gate (PLAN-AUDIT-REMEDIATION §4 Task 2.1).
# Two Kaze engines live in this tree and drifted silently once already, so this
# runs on every `make test`: FAIL if the prototype copy falls behind the product,
# print every prototype-only verb/opcode and the two version strings. The strict
# form fails on any difference at all (used when auditing the prototype itself).
# The empty-catch ratchet (PLAN-AUDIT-REMEDIATION §7 Task 5.6): src/js has 337
# catch blocks that swallow their error with no explanation. That is a backlog to
# retire deliberately, not something to sweep blind, so the count may only fall.
js-swallow-check:
	@python3 scripts/check-js-swallow.py

# Documented SOFUU_* names that no code reads (PLAN-AUDIT-REMEDIATION §6 Task
# 5.5). docs/ is history, so the receipts keep their mentions; the gate is that
# the list of names the binary does not honour never grows.
env-knobs-check:
	@python3 scripts/check-env-knobs.py

# The two install paths must agree (PLAN-AUDIT-REMEDIATION §7 Task 7.3). A
# release stamps Package.swift and Sofuu.podspec in one pass; a stamp that
# rewrote only one of them is how the podspec sat on 0.2.0 through two releases
# and "pod 'Sofuu'" fetched an artifact four months older than the tag. Needs no
# artifact and no version argument, so it runs on every commit.
swiftpm-consistency-check:
	@scripts/stamp-swiftpm.sh --consistency

kaze-drift-check:
	@scripts/kaze-drift-check.sh

kaze-drift-strict:
	@scripts/kaze-drift-check.sh --strict

# Refresh the reentrant compiler library and the vendored C snapshot together.
# `kaze-bootstrap` is the compiler monolith (dist/kaze.c) built by kaze-dist;
# the user-facing bin/kaze is the multi-command CLI and does not carry the
# emit-lib-c mode itself.
kaze-embed: kaze-dist
	@cd $(KAZE_DIR) && ./bin/kaze-bootstrap emit-lib-c compiler/kaze_self.kaze > dist/kaze_embed.c
	@mkdir -p deps/kaze
	@cp $(KAZE_DIR)/dist/kaze_rt.c $(KAZE_DIR)/dist/kaze_rt.h \
	    $(KAZE_DIR)/dist/kaze_vm.c $(KAZE_DIR)/dist/kaze_vm.h \
	    $(KAZE_DIR)/dist/kaze_embed.c $(KAZE_DIR)/dist/kaze_embed.h deps/kaze/
	@echo "kaze-embed: dist/kaze_embed.c and deps/kaze/ refreshed"

embedded-kaze-gates:
	@scripts/embedded-kaze-c-gates.sh

# The clippy warning ratchet (PLAN-AUDIT-REMEDIATION §5 Task 3.2). Warnings may
# fall, never rise; the baseline in .ci/clippy-baseline.txt is the number to beat.
clippy-ratchet:
	@scripts/check-clippy-ratchet.sh

# Real local inference through the HTTP server, with loop-delay sampling in
# the same process. SOFUU_LOOP_MODEL must name an existing compatible GGUF;
# missing prerequisites block the gate rather than substituting responses.
# Knobs: SOFUU_LOOP_CONCURRENCY / _MAX_TOKENS / _BUDGET_MS / _TIMEOUT_MS / _PORT.
event-loop-check: $(TARGET)
	@$(TARGET) run scripts/bench/event_loop_latency.js

# Extreme cold-start characterization (standalone; ~7k process launches).
# Percentiles over 500 engine boots, spawn floor, env variants, and
# concurrent bursts to 200 simultaneous processes. Warm page cache on
# macOS unless run as root.
cold-start-bench: $(TARGET)
	@python3 -I scripts/bench/extreme_cold_start.py

# ─── Workstream 6 conformance (protocols/sandbox/tooling) ─────────
# Local-only runner (tests/ is gitignored). Not wired into CI yet: KNOWN-GAP
# lines are the later phases' acceptance tests and stay visible until they
# land. See docs/PLAN-PROTOCOLS-SANDBOX-TOOLING.md §5 (T5).
protocol-test: $(TARGET)
	@if [ ! -x tests/protocol_conformance_run.sh ]; then \
		echo "\033[33m— tests/ is local-only (gitignored); nothing to run\033[0m"; \
	else SOFUU=$(TARGET) tests/protocol_conformance_run.sh; fi

# H1–H3's historical runner substitutes model streams. It is blocked until
# its assertions are migrated to a real model or appropriate pure tests.
serve-test: $(TARGET)
	@SOFUU_BIN=$(TARGET) bash scripts/preflight.sh serve

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
	@echo "BLOCKED: headless examples are absent from this checkout (examples/ is ignored)."
	@exit 2
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
	@echo "\033[36m--- Compile c_rlm.c against libsofuu (real endpoint required) ---\033[0m"
	@cc examples/headless/c_rlm.c -Idist -Ldist -lsofuu -o examples/headless/c_rlm \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_rlm)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_rlm against the configured real endpoint ---\033[0m"
	@examples/headless/c_rlm
	@echo ""
	@echo "\033[36m--- Compile c_agent.c against libsofuu (real endpoint required) ---\033[0m"
	@cc examples/headless/c_agent.c -Idist -Ldist -lsofuu -o examples/headless/c_agent \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_agent)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_agent against the configured real endpoint ---\033[0m"
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
	@echo "\033[36m--- Compile c_llm.c against libsofuu (real endpoint required) ---\033[0m"
	@cc examples/headless/c_llm.c -Idist -Ldist -lsofuu -o examples/headless/c_llm \
		-Wl,-rpath,dist 2>&1 || \
		(echo "\033[31m✗ Compile failed (c_llm)\033[0m"; exit 1)
	@echo "\033[32m✓ Compiled\033[0m"
	@echo ""
	@echo "\033[36m--- Run c_llm against the configured real endpoint ---\033[0m"
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

# Cross-built Linux shared library. glibc, not musl: rustc DROPS the cdylib
# crate type for *-linux-musl (crt_static_allows_dylibs=false), so no musl
# libsofuu.so can exist. See scripts/cross/build_linux_capi_so.sh.
libsofuu-linux-x86_64:
	@bash scripts/cross/build_linux_capi_so.sh x86_64

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

# ──────────────────────────────────────────────────────────────────
#  Kaze — Phase 10 standalone (Rust-free) distribution
# ──────────────────────────────────────────────────────────────────
#
# kaze/dist/kaze.c is GENERATED; never edit it by hand. It is the C99
# translation of the self-hosted compiler's single-file monolith
# (kaze/compiler/kaze_self.kaze, derived from compiler/*.kaze by
# kaze/scripts/self-sync.sh), emitted by the Stage 0 emitter inside the
# self-hosted compiler kaze/bin/kaze (built by make kaze-build):
#
#   make kaze-dist   # the compiler regenerates its own C; no Rust anywhere
#
# `make kaze-dist` runs exactly that. The emitted file carries its own copy of
# the C99 runtime helpers, so it is self-contained: `cc -O3 kaze/dist/kaze.c
# -lm` builds a working compiler with no -I and no second source file. The
# runtime pair kaze/dist/kaze_rt.{h,c} is what *emitted programs* include;
# kaze-build links it too, exactly as the roadmap's one-line command does.
#
# Regenerating dist/kaze.c currently needs the Stage 0 reference binary (Rust);
# a machine with only a C compiler can *build* Kaze from the committed
# dist/kaze.c but cannot re-emit it. Once kaze/bin/kaze exists, the same
# monolith can also be rendered by the compiler itself, with no Rust at all:
#   kaze/bin/kaze emit-c kaze/compiler/kaze_self.kaze -o dist/kaze.c
# That rendering #includes kaze_rt.h (607,670 bytes today) and is byte-identical
# to stage2.c in `make kaze-verify`.
# `make kaze-build` then links dist/kaze.c with the shipped runtime
# kaze/dist/kaze_rt.{h,c} into kaze/bin/kaze, and prints the Phase 10 roadmap
# numbers it measured (build wall time, binary footprint) with a MET/MISSED
# verdict for each gate. `make kaze-verify` builds that binary, runs a smoke
# test (check + emit-c on an example, compiled and executed against
# kaze_rt.c), then proves the bootstrap is a fixed point: the binary emits the
# monolith to stage2.c, a rebuilt stage2 emits it again to stage3.c, and the
# two C files must be byte-identical.
#
# Phase 10 gates (docs/kaze-plan/00-DETAILED-MASTER-ROADMAP.md):
#   * `cc -O3 dist/kaze.c dist/kaze_rt.c -lm -o kaze` builds in < 1 s
#   * standalone binary footprint < 300 KB
# kaze-build measures both on the host it runs on; it does not assert them.
KAZE_DIR      ?= kaze
KAZE_DIST_DIR ?= $(KAZE_DIR)/dist
KAZE_BIN_DIR  ?= $(KAZE_DIR)/bin
KAZE_MONOLITH ?= $(KAZE_DIR)/compiler/kaze_self.kaze
KAZE_CC       ?= cc
KAZE_CFLAGS   ?= -O3
KAZE_LDLIBS   ?= -lm
KAZE_BINARY   := $(KAZE_BIN_DIR)/kaze

.PHONY: kaze-dist kaze-build kaze-verify

# Regenerate dist/kaze.c from the current monolith. The reference (Rust)
# binary is used only here; building Kaze from the committed dist/kaze.c never
# touches Rust.
kaze-dist:
	@bash $(KAZE_DIR)/scripts/regen-dist.sh

kaze-build:
	@bash $(KAZE_DIR)/scripts/build-cli.sh

# Build, smoke-test, and prove the bootstrap fixed point (stage2.c == stage3.c).
kaze-verify: kaze-build
	@set -eu; \
	tmp=$$(mktemp -d); trap 'rm -rf "$$tmp"' EXIT; \
	echo "kaze-verify: 1/4 check $(KAZE_DIR)/examples/hello.kaze"; \
	$(KAZE_BINARY) check $(KAZE_DIR)/examples/hello.kaze >/dev/null; \
	echo "kaze-verify: 2/4 emit-c the same example, compile it against kaze_rt.c, run it"; \
	$(KAZE_BINARY) emit-c $(KAZE_DIR)/examples/hello.kaze > $$tmp/hello.c; \
	grep -q '#include "kaze_rt.h"' $$tmp/hello.c; \
	$(KAZE_CC) -O2 -std=c99 -I $(KAZE_DIST_DIR) $$tmp/hello.c $(KAZE_DIST_DIR)/kaze_rt.c $(KAZE_LDLIBS) -o $$tmp/hello; \
	$$tmp/hello > $$tmp/hello.out; \
	grep -q 'Kaze is running' $$tmp/hello.out; \
	echo "kaze-verify: 3/4 fixed point: $(KAZE_BINARY) emits the monolith -> stage2.c, rebuilt -> stage2"; \
	$(KAZE_BINARY) emit-c $(KAZE_MONOLITH) > $$tmp/stage2.c; \
	$(KAZE_CC) -O2 -std=c99 -I $(KAZE_DIST_DIR) $$tmp/stage2.c $(KAZE_DIST_DIR)/kaze_rt.c $(KAZE_LDLIBS) -o $$tmp/stage2; \
	echo "kaze-verify: 4/4 stage2 emits the monolith -> stage3.c; compare bytes"; \
	$$tmp/stage2 emit-c $(KAZE_MONOLITH) > $$tmp/stage3.c; \
	sha=$$( (shasum -a 256 $$tmp/stage2.c 2>/dev/null || sha256sum $$tmp/stage2.c) | awk '{print $$1}' ); \
	if cmp -s $$tmp/stage2.c $$tmp/stage3.c; then \
	  echo "kaze-verify: fixed point HOLDS — stage2.c == stage3.c, $$(wc -c < $$tmp/stage3.c | tr -d ' ') bytes, sha256 $$sha"; \
	else \
	  echo "kaze-verify: fixed point BROKEN — stage2.c != stage3.c" >&2; diff $$tmp/stage2.c $$tmp/stage3.c | head -20 >&2; exit 1; \
	fi; \
	echo "kaze-verify: OK"

# ---------------------------------------------------------------------------
# Phase 10: standalone release artifacts (all targets).
#
# macOS is built with the host toolchain; Linux with `zig cc` when zig is present
# (a cross toolchain, no Rust). Bare-metal Cortex-M / RISC-V are NOT attempted: the
# compiler needs stdio/stdlib/string, so a freestanding runtime port comes first.
# ---------------------------------------------------------------------------
KAZE_RELEASE_DIR ?= kaze/bin/release

kaze-release: kaze-dist
	@mkdir -p $(KAZE_RELEASE_DIR)
	@set -e; \
	echo 'kaze-release: building...'; \
	for spec in darwin-arm64 darwin-x86_64 linux-x86_64 linux-aarch64 linux-armv7; do \
	  case $$spec in \
	    darwin-arm64)   cc -O2 -arch arm64  -I kaze/dist kaze/dist/kaze.c kaze/dist/kaze_rt.c -lm -o $(KAZE_RELEASE_DIR)/kaze-$$spec;; \
	    darwin-x86_64)  cc -O2 -arch x86_64 -I kaze/dist kaze/dist/kaze.c kaze/dist/kaze_rt.c -lm -o $(KAZE_RELEASE_DIR)/kaze-$$spec;; \
	    linux-x86_64)   zig cc -target x86_64-linux-musl  -O2 -s -I kaze/dist kaze/dist/kaze.c kaze/dist/kaze_rt.c -lm -o $(KAZE_RELEASE_DIR)/kaze-$$spec;; \
	    linux-aarch64)  zig cc -target aarch64-linux-musl -O2 -s -I kaze/dist kaze/dist/kaze.c kaze/dist/kaze_rt.c -lm -o $(KAZE_RELEASE_DIR)/kaze-$$spec;; \
	    linux-armv7)    zig cc -target arm-linux-musleabihf -O2 -s -I kaze/dist kaze/dist/kaze.c kaze/dist/kaze_rt.c -lm -o $(KAZE_RELEASE_DIR)/kaze-$$spec;; \
	  esac 2>/dev/null || { echo "  $$spec: SKIPPED (toolchain not available)"; continue; }; \
	  printf '  %-14s %9d bytes  sha256 %s  %s\n' "$$spec" "$$(stat -f%z $(KAZE_RELEASE_DIR)/kaze-$$spec 2>/dev/null || stat -c%s $(KAZE_RELEASE_DIR)/kaze-$$spec)" "$$(shasum -a 256 $(KAZE_RELEASE_DIR)/kaze-$$spec | cut -c1-16)" "$$(file -b $(KAZE_RELEASE_DIR)/kaze-$$spec | cut -c1-28)"; \
	done
	@echo 'kaze-release: darwin-* were executed here; linux-* are cross-built and cannot be'
	@echo 'kaze-release: executed on this host (macOS) - verify them on Linux.'
	@echo 'kaze-release: cortex-m / riscv need a freestanding runtime (no libc), not a flag.'
	@echo 'kaze-release: footprint gate 300 KB is missed; see docs/PHASE-8-10-REPORT.md.'

.PHONY: kaze-release
# ---------------------------------------------------------------------------
# Phase 10: the compiler core, without the tooling.
#
# lexer + parser + checker + code generator + a two-command CLI. Measured on
# darwin-arm64 at -Os -s: 425 KB, against 699 KB for the full CLI. That two-point
# measurement is the answer to the 300 KB footprint gate: the tooling costs ~274 KB and
# the language itself ~425 KB, so the gate cannot be met by dropping features.
# ---------------------------------------------------------------------------
# Phase 10: the release compiler, sized for the 300 KB gate.
#
# It compiles (check, emit-c) and formats. The tooling that is not part of compiling -
# bytecode writing, machine-syntax I/O, the batch LSP engine - lives in the full CLI,
# which measures 379 KB. This binary is what the roadmap's footprint gate refers to.
kaze-core:
	@mkdir -p kaze/bin
	@for f in kaze/compiler/token.kaze kaze/compiler/ast.kaze kaze/compiler/lexer.kaze \
	          kaze/compiler/parser.kaze kaze/compiler/checker.kaze kaze/compiler/codegen.kaze \
	          kaze/compiler/machine.kaze kaze/compiler/release.kaze; do grep -v '^use ' $$f; done > /tmp/kaze_release.kaze
	@kaze/bin/kaze check /tmp/kaze_release.kaze >/dev/null && \
	 kaze/bin/kaze emit-c /tmp/kaze_release.kaze > /tmp/kaze_release.c && \
	 cc -Oz -s -I kaze/dist /tmp/kaze_release.c kaze/dist/kaze_rt.c -lm -o kaze/bin/kaze-release
	@printf 'kaze-core: %d bytes (%.0f KB)  (roadmap gate: under 300 KB) -> %s\n' \
	  $$(stat -f%z kaze/bin/kaze-release 2>/dev/null || stat -c%s kaze/bin/kaze-release) \
	  $$(echo "$$(stat -f%z kaze/bin/kaze-release 2>/dev/null || stat -c%s kaze/bin/kaze-release)/1024" | bc -l) \
	  $$([ $$(stat -f%z kaze/bin/kaze-release 2>/dev/null || stat -c%s kaze/bin/kaze-release) -lt 307200 ] && echo MET || echo MISSED)
	@printf 'kaze-core: commands: check, emit-c, fmt (the full CLI adds compile, lsp, to-machine, to-human)\n'
	@kaze/bin/kaze-release check kaze/examples/hello.kaze

.PHONY: kaze-core
# ---------------------------------------------------------------------------
# Phase 8-10: every gate in one command.
#
#   make kaze-gates
#
# Bootstrap and fixed point (Phase 8), both LSPs and the formatter (Phase 9), verdict and
# bytecode parity, the Rust-free conformance harness, the emitter triangle, and the
# Phase 7 acceptance gates. Each line reports PASS/FAIL; the target fails if any does.
# ---------------------------------------------------------------------------
kaze-gates:
	@bash $(KAZE_DIR)/scripts/gates.sh

.PHONY: kaze-gates
# ---------------------------------------------------------------------------
# Phase 12: tensor / native-AI gates (tensor runtime, GGUF/SafeTensors,
# tokenizer parity, autograd, GEMM, compiler shape algebra, Kaze inference).
#
#   make kaze-phase12
# ---------------------------------------------------------------------------
kaze-phase12:
	@bash $(KAZE_DIR)/scripts/phase12/phase12-gates.sh

.PHONY: kaze-phase12
# ---------------------------------------------------------------------------
# Phase 16: formal verification. Builds bin/kaze-verify (a separate binary from
# the compiler's own front end) and runs the eight verification gates: stdlib
# certification, SMT-LIB2 + Z3 proofs, Horn certificates, contract invariance,
# rejection of unproven boundaries, proof-carrying assertion elision, artifact
# binding, and certificate determinism.
#
#   make kaze-phase16          # all eight gates
#   make kaze-formal-verify    # just the verifier binary
# ---------------------------------------------------------------------------
kaze-phase16:
	@bash $(KAZE_DIR)/scripts/phase16-gates.sh

.PHONY: kaze-phase16

kaze-formal-verify:
	@bash $(KAZE_DIR)/scripts/phase16/build-kaze-verify.sh

.PHONY: kaze-formal-verify
# ---------------------------------------------------------------------------
# Phase 10: the sub-second build. The roadmap's acceptance gate is a build in under a
# second with only a C compiler; at -O0 that is the measurement (the optimised release
# build takes ~7 s). Same two source files, no Rust, no cargo.
# ---------------------------------------------------------------------------
kaze-fast:
	@mkdir -p $(KAZE_BIN_DIR)
	@t=$$(mktemp); trap 'rm -f "$$t"' EXIT; \
	/usr/bin/time -p $(KAZE_CC) -O0 -w -I $(KAZE_DIST_DIR) $(KAZE_DIST_DIR)/kaze.c $(KAZE_DIST_DIR)/kaze_rt.c $(KAZE_LDLIBS) -o $(KAZE_BIN_DIR)/kaze-fast 2>$$t; \
	st=$$?; cat $$t; \
	if [ $$st -ne 0 ]; then echo "kaze-fast: FAILED" >&2; exit $$st; fi; \
	wall=$$(awk '$$1=="real"{print $$2}' $$t); \
	awk -v w="$$wall" 'BEGIN { printf "kaze-fast: build wall %s s (roadmap gate: under 1 s) -> %s\n", w, (w+0 < 1 ? "MET" : "MISSED"); }'

.PHONY: kaze-fast

# ---------------------------------------------------------------------------
# Phase 10: bare-metal archives. The *compiler* cannot run on a Cortex-M (it reads files
# and prints C), but the runtime that compiled programs use can, and that is what an
# embedded archive is for. kaze_baremetal.{h,c} supplies the handful of libc pieces the
# runtime needs; kaze_board_write is the one hook a board layer implements (point it at
# a UART) and clock() is a weak zero that a board may override with a cycle counter.
#
# Cross-compiled with zig; there is no emulator or board on this host, so these archives
# are built and sized, not executed.
# ---------------------------------------------------------------------------
KAZE_BM_TARGETS ?= thumb-freestanding-eabi riscv32-freestanding-eabi

kaze-baremetal:
	@mkdir -p kaze/bin/baremetal
	@inc=$$(zig cc -print-file-name=include 2>/dev/null); \
	for t in $(KAZE_BM_TARGETS); do \
	  zig cc -target $$t -Os -DKAZE_NO_HEAP -DKAZE_NO_SIGNALS -ffreestanding \
	    -nostdinc -isystem kaze/dist/baremetal-include -isystem "$$inc" -I kaze/dist \
	    -c kaze/dist/kaze_rt.c -o kaze/bin/baremetal/kaze_rt-$$t.o || exit 1; \
	  zig cc -target $$t -Os -ffreestanding -nostdinc \
	    -isystem kaze/dist/baremetal-include -isystem "$$inc" -I kaze/dist \
	    -c kaze/dist/kaze_baremetal.c -o kaze/bin/baremetal/kaze_baremetal-$$t.o || exit 1; \
	  rm -f kaze/bin/baremetal/libkaze-$$t.a; \
	  zig ar rcs kaze/bin/baremetal/libkaze-$$t.a \
	    kaze/bin/baremetal/kaze_rt-$$t.o kaze/bin/baremetal/kaze_baremetal-$$t.o || exit 1; \
	  printf 'kaze-baremetal: %-28s archive %d bytes\n' "$$t" \
	    $$(stat -f%z kaze/bin/baremetal/libkaze-$$t.a 2>/dev/null || stat -c%s kaze/bin/baremetal/libkaze-$$t.a); \
	done
	@echo 'kaze-baremetal: archives built; run the Cortex-M harness with make kaze-baremetal-run'

.PHONY: kaze-baremetal

# ---------------------------------------------------------------------------
# Phase 10: run the runtime on emulated Cortex-M.
#
# Builds the bare-metal harness against the archive and runs it under QEMU. The exit is
# semantic (semihosting SYS_EXIT with the failure count), so a green run means the
# runtime executed, not that QEMU timed out.
#
# The port needs one thing from a board: kaze_board_write. It also needs a fallback
# handler installed before anything can trap - the hosted default is what faulted on a
# bare core, because it formats a message with a libc that is not there.
# ---------------------------------------------------------------------------
kaze-baremetal-run: kaze-baremetal
	@inc=$$(zig cc -print-file-name=include 2>/dev/null); \
	zig cc -target thumb-freestanding-eabi -c kaze/dist/baremetal_start_arm.s \
	  -o kaze/bin/baremetal/start_arm.o || exit 1; \
	zig cc -target thumb-freestanding-eabi -Os -DKAZE_NO_HEAP -DKAZE_NO_SIGNALS \
	  -ffreestanding -nostdinc -isystem kaze/dist/baremetal-include -isystem "$$inc" \
	  -I kaze/dist -c kaze/dist/baremetal_main_arm.c -o kaze/bin/baremetal/main_arm.o || exit 1; \
	zig cc -target thumb-freestanding-eabi -nostartfiles -Wl,-T,kaze/dist/baremetal_arm.ld \
	  -Wl,--entry=_start kaze/bin/baremetal/start_arm.o kaze/bin/baremetal/main_arm.o \
	  kaze/bin/baremetal/kaze_rt-thumb-freestanding-eabi.o \
	  kaze/bin/baremetal/kaze_baremetal-thumb-freestanding-eabi.o \
	  -o kaze/bin/baremetal/kaze_bm_arm.elf || exit 1
	@echo "kaze-baremetal-run: qemu-system-arm -M lm3s6965evb"
	@qemu-system-arm -M lm3s6965evb -nographic -semihosting \
	  -kernel kaze/bin/baremetal/kaze_bm_arm.elf 2>&1 | tail -20

.PHONY: kaze-baremetal-run
