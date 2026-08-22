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

# QTSQ codec checkout (proprietary, optional). The cargo build reads
# SOFUU_QTSQ_DIR; point at a checkout to enable brain persistence +
# session store, or leave unset for a degraded build (CI does the same).
QTSQ_DIR ?= /Users/priyanshuboruah/projects/black-hole-disk

# ──────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────

.PHONY: all clean install test bench c-only size-check cargo-build \
        libsofuu zig-install linux linux-x86_64 linux-arm64 release-archives dist \
        headless-test abi-check dist-macos dist-linux dist-ios dist-android dist-all

SIZE_LIMIT ?= 5242880   # 5 MB hard cap (bytes)

all: $(TARGET)

# ── Rust-first build (default and only maintained build) ────────
cargo-build:
	cargo build --release

$(TARGET): cargo-build
	@echo "  \033[36mCARGO\033[0m  $@  [arch=$(ARCH)]"
	@cp target/release/$(TARGET) $(TARGET)
	@if [ "$$(uname -s)" = "Darwin" ]; then codesign --force --sign - $(TARGET) 2>/dev/null || true; fi
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

# ── Size cap check (CI: fails if binary exceeds 5MB) ────────────
size-check: $(TARGET)
	@SIZE=$$(stat -f%z $(TARGET) 2>/dev/null || stat -c%s $(TARGET) 2>/dev/null); \
	if [ "$$SIZE" -gt "$(SIZE_LIMIT)" ]; then \
		echo "  \033[31m✗ Size $$SIZE bytes exceeds cap $(SIZE_LIMIT) bytes\033[0m"; \
		exit 1; \
	else \
		echo "  \033[32m✓ Size $$SIZE bytes ≤ $(SIZE_LIMIT) bytes\033[0m"; \
	fi

# ── libsofuu: embeddable library (PLAN-HEADLESS H1) ──────────────
# Builds the capi crate as staticlib + cdylib, copies artifacts to dist/.
# Usage:  make libsofuu
# Output: dist/libsofuu.a  dist/libsofuu.dylib  dist/sofuu_embed.h
libsofuu:
	@echo "  \033[36mCARGO\033[0m  libsofuu  [arch=$(ARCH)]"
	cargo build --release -p sofuu-capi
	@mkdir -p dist
	@cp target/release/libsofuu_capi.a dist/libsofuu.a 2>/dev/null || \
		cp target/release/libsofuu.a dist/libsofuu.a 2>/dev/null || true
	@cp target/release/libsofuu_capi.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu.dylib dist/libsofuu.dylib 2>/dev/null || \
		cp target/release/libsofuu_capi.so dist/libsofuu.so 2>/dev/null || \
		cp target/release/libsofuu.so dist/libsofuu.so 2>/dev/null || true
	@cp include/sofuu_embed.h dist/sofuu_embed.h
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
	@bash tests/run_js_tests.sh

bench: $(TARGET)
	@bash bench/run_bench.sh

# ─── Cross-compilation (cargo + zig; local convenience) ────────────
# Uses Zig as a zero-dependency cross-compiler (no Docker needed).
# First run:  make zig-install
# Then:       make linux  OR  make linux-x86_64  OR  make linux-arm64
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

release-archives: linux
	@echo "Archives are in dist/"
	@ls -lh dist/*.tar.gz 2>/dev/null || echo " (none yet)"

dist: all linux
	@mkdir -p dist
	@cp $(TARGET) dist/sofuu-darwin-arm64
	@tar -czf dist/sofuu-darwin-arm64.tar.gz -C dist sofuu-darwin-arm64
	@echo "  \033[32m✓\033[0m dist/sofuu-darwin-arm64.tar.gz"
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

dist-all:
	@bash scripts/dist/all.sh

# ── H6: Headless CI gates ──────────────────────────────────────────
# `make headless-test`: build libsofuu + compile c_embed.c + c_rlm.c + c_agent.c + run all.
# `make abi-check`: diff exported symbols against the checked-in baseline.
headless-test: libsofuu
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
	@echo "\033[32m✓ Headless tests passed\033[0m"
	@echo ""

abi-check: libsofuu
	@# Use the dylib/.so (public symbols are resolved at link time, not in the .a archive).
	@if [ -f dist/libsofuu.dylib ]; then LIB=dist/libsofuu.dylib; \
	 elif [ -f dist/libsofuu.so ]; then LIB=dist/libsofuu.so; \
	 else echo "\033[31m✗ No shared lib in dist/\033[0m"; exit 1; fi; \
	bash scripts/check_abi.sh $$LIB
