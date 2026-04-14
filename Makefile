# ──────────────────────────────────────────────────────────────────
#  Sofuu (素風) — AI-Native JS Runtime
#  Build System: simple Make, no CMake required
# ──────────────────────────────────────────────────────────────────

CC      = cc
TARGET  = sofuu
ARCH   := $(shell uname -m)

# ---- Directories ----
SRC_DIR   = src
DEPS_DIR  = deps
QJS_DIR   = $(DEPS_DIR)/quickjs
UV_DIR    = $(DEPS_DIR)/libuv

# ---- Compiler flags ----
QJS_VERSION := $(shell cat $(QJS_DIR)/VERSION 2>/dev/null || echo "2024-01-13")

CFLAGS  = -O2 -Wall -Wextra -Wno-unused-parameter \
          -Wno-sign-compare \
          -Wno-cast-function-type \
          -I$(QJS_DIR) \
          -I$(UV_DIR)/include \
          -I$(SRC_DIR) \
          -I$(SRC_DIR)/engine \
          -I$(SRC_DIR)/modules \
          -I$(SRC_DIR)/io \
          -I$(SRC_DIR)/http \
          -I$(SRC_DIR)/mcp \
          -Ideps/http-parser \
          -I$(SRC_DIR)/ts \
          -I$(SRC_DIR)/simd \
          -I$(SRC_DIR)/npm \
          -I$(SRC_DIR)/repl \
          -I$(SRC_DIR)/bundler \
          -D_GNU_SOURCE \
          -DCONFIG_VERSION=\"$(QJS_VERSION)\"

# Feature Flags
SOFUU_MEMORY ?= 0
SOFUU_LLM    ?= 0

ifeq ($(SOFUU_MEMORY),1)
    CFLAGS += -DSOFUU_MEMORY=1
endif

ifeq ($(SOFUU_LLM),1)
    CFLAGS += -DSOFUU_LLM=1
endif

# macOS-specific
UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
    CFLAGS += -D__APPLE__
endif

# Architecture-specific SIMD flags
ifeq ($(ARCH), arm64)
    CFLAGS  += -march=armv8-a
    SIMD_SRC = $(SRC_DIR)/simd/neon.c
else ifeq ($(ARCH), x86_64)
    CFLAGS  += -mavx2 -mfma
    SIMD_SRC = $(SRC_DIR)/simd/avx.c
else
    SIMD_SRC = $(SRC_DIR)/simd/neon.c   # scalar fallback
endif


# Optional: readline for REPL
READLINE := $(shell pkg-config --libs readline 2>/dev/null)
ifneq ($(READLINE),)
    CFLAGS  += -DSOFUU_HAVE_READLINE
    LDFLAGS += $(READLINE)
endif

# ---- Source files ----
# QuickJS core (we compile it as part of our build — no separate lib step)
QJS_SRCS = \
    $(QJS_DIR)/quickjs.c \
    $(QJS_DIR)/quickjs-libc.c \
    $(QJS_DIR)/libregexp.c \
    $(QJS_DIR)/libunicode.c \
    $(QJS_DIR)/cutils.c \
    $(QJS_DIR)/libbf.c

# Our runtime sources
SOFUU_SRCS = \
    $(SRC_DIR)/main.c \
    $(SRC_DIR)/sofuu.c \
    $(SRC_DIR)/engine/engine.c \
    $(SRC_DIR)/modules/mod_console.c \
    $(SRC_DIR)/modules/mod_process.c \
    $(SRC_DIR)/modules/mod_ai.c \
    $(SRC_DIR)/io/promises.c \
    $(SRC_DIR)/io/loop.c \
    $(SRC_DIR)/io/timer.c \
    $(SRC_DIR)/io/fs.c \
    $(SRC_DIR)/io/subprocess.c \
    $(SRC_DIR)/http/client.c \
    $(SRC_DIR)/http/server.c \
    $(SRC_DIR)/http/sse.c \
    $(SRC_DIR)/mcp/mcp.c \
    $(SRC_DIR)/ts/stripper.c \
    $(SRC_DIR)/npm/resolver.c \
    $(SRC_DIR)/npm/cjs.c \
    $(SRC_DIR)/repl/repl.c \
    $(SRC_DIR)/bundler/bundler.c \
    $(SIMD_SRC) \
    deps/http-parser/http_parser.c

# Memory and LLM modules are disabled by default (require local C library)
# Enable with: make SOFUU_MEMORY=1 SOFUU_LLM=1
ifeq ($(SOFUU_MEMORY),1)
    SOFUU_SRCS += src/memory/qtsq_adapter.c \
                  src/memory/hnsw.c \
                  src/memory/mod_memory.c \
                  src/memory/mod_kv.c \
                  src/memory/mod_agent.c
endif

ifeq ($(SOFUU_LLM),1)
    SOFUU_SRCS += src/llm/llm_local.c
endif

ALL_SRCS = $(QJS_SRCS) $(SOFUU_SRCS)
UV_LIB   = $(UV_DIR)/build/libuv.a

# ---- Link flags ----
LDFLAGS = -lpthread -lm -lcurl

ifeq ($(SOFUU_MEMORY),1)
    QTSQ_DIR = black-hole-disk
    LDFLAGS += -L$(QTSQ_DIR) -lqtsq -lz
endif

# macOS has no need for -ldl, Linux does
ifneq ($(UNAME), Darwin)
    LDFLAGS += -ldl
endif

# ──────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────

.PHONY: all clean install test bench

all: $(TARGET)

$(TARGET): $(ALL_SRCS) $(UV_LIB)
	@echo "  \033[36mCC\033[0m  $@  [arch=$(ARCH)]"
	$(CC) $(CFLAGS) -o $@ $(ALL_SRCS) $(UV_LIB) $(LDFLAGS)
	@echo "  \033[32mBuilt:\033[0m ./$(TARGET) ($(shell du -sh $(TARGET) | cut -f1))"

install: $(TARGET)
	@echo "  Installing sofuu to /usr/local/bin/"
	cp $(TARGET) /usr/local/bin/$(TARGET)
	@echo "  Done. Run: sofuu help"

clean:
	@echo "  Cleaning..."
	rm -f $(TARGET)

test: $(TARGET)
	@echo ""
	@echo "\033[1m=== Sofuu Test Suite ===\033[0m"
	@echo ""
	@echo "\033[36m--- Priority 1: fetch/GC/rejection ---\033[0m"
	./$(TARGET) run examples/priority1_test.js
	@echo ""
	@echo "\033[36m--- Priority 2: process/signals/stdin ---\033[0m"
	./$(TARGET) run examples/priority2_test.js
	@echo ""
	@echo "\033[36m--- TypeScript stripper ---\033[0m"
	./$(TARGET) run examples/ts_test.ts
	@echo ""
	@echo "\033[36m--- SIMD vectors ---\033[0m"
	./$(TARGET) run examples/simd_test.js
	@echo ""
	@echo "\033[32m=== All tests passed! ===\033[0m"
	@echo ""

bench: $(TARGET)
	@bash bench/run_bench.sh

# ─── Cross-compilation targets ───────────────────────────────────────────────
# Uses Zig as a zero-dependency cross-compiler (no Docker needed).
# First run:  make zig-install
# Then:       make linux  OR  make linux-x86_64  OR  make linux-arm64

ZIG_INSTALL = scripts/cross/install_zig.sh
CROSS_SCRIPT = scripts/cross/cross_compile.sh

.PHONY: zig-install linux linux-x86_64 linux-arm64 release-archives dist

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
	@echo "  [32m✓[0m dist/sofuu-darwin-arm64.tar.gz"
	@ls -lh dist/*.tar.gz

