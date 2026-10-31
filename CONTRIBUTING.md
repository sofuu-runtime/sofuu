# Contributing to Sofuu

Sofuu is an **open-source project with a solo-developer governance model** (similar to SQLite: *open source, not open contribution*).

The runtime is 100% open source under the permissive **MIT License** — you are completely free to inspect the source, fork it, compile it locally, experiment with modifications, and build tools on top of it.

---

## 🏛️ Development Philosophy

To maintain:
1. **Airtight Architectural Cohesion**: A single unified design vision across memory, event loops, sandboxes, and agent loops.
2. **Strict Binary Size Cap**: Keeping the uncompressed stripped binary strictly $\le 5.5\text{ MB}$ with zero external shared libraries.
3. **High Security Standards**: Avoiding external supply-chain vulnerabilities in core execution paths.

**All core code is developed, audited, and maintained exclusively by the creator.** 

> ⚠️ **Pull requests containing code changes or feature additions will not be merged.** Please do not spend your valuable time writing speculative PRs for the core runtime.

---

## 💡 How You Can Contribute

Even without submitting code PRs to core, community feedback is vital to shaping Sofuu:

### 1. Detailed Bug Reports
If you run into an unexpected crash, edge case in JavaScript execution, or platform discrepancy:
- Open a [GitHub Issue](https://github.com/sofuu-runtime/sofuu/issues).
- Provide your OS, CPU architecture, and a **minimal reproducible script** (`repro.js`).

### 2. Feature Requests & Architectural Feedback
- Share how you are using Sofuu and what capabilities or APIs would make your agent workflows smoother.
- Propose ideas via GitHub Issues or Discussions.

### 3. Build & Share Plugins
- Build custom tools, MCP integrations, or plugins using the official plugin API.
- Share your community plugins and integrations publicly for others to use.

### 4. Documentation & Typo Fixes
- Small pull requests fixing typos, broken links, or clarifying documentation in Markdown files are warmly welcomed.

---

## 🛠️ Building & Testing Locally (For Your Own Fork)

`make` on its own does not build: the runtime links QTSQ (the codec behind
`sofuu.memory`) statically, and both build scripts refuse to configure without
it. The prerequisite list, in the order the failures appear:

```bash
# 1. libuv — vendored, but cargo needs a build tree of it first
make deps

# 2. QTSQ — a separate checkout that has to produce libqtsq.a
git clone https://github.com/haruhito2666/qtsq ~/projects/qtsq
#    building it needs zlib headers (Debian/Ubuntu: zlib1g-dev)
cd ~/projects/qtsq
make CPPFLAGS="-D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE"
#    On Linux that still fails: its Makefile compiles src/qtsq_metal.mm on
#    every target, which needs Apple's Objective-C++ toolchain
#    (`cc: cannot execute cc1objplus`). Until that guard lands upstream,
#    compile the C translation units directly and archive them:
#      cc -O2 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -c src/*.c
#      ar rcs libqtsq.a *.o

# 3. Link dependencies of the engine: libcurl and libssl headers
#    Debian/Ubuntu: sudo apt-get install -y libcurl4-openssl-dev libssl-dev

# 4. Build and gate (SOFUU_QTSQ_DIR points at the checkout from step 2)
SOFUU_QTSQ_DIR=~/projects/qtsq make
SOFUU_QTSQ_DIR=~/projects/qtsq make size-check   # hard cap: 5,767,168 B (RULES.md 3.1)

# Run the test suite
make test

# Verify runtime health
./sofuu doctor
```

macOS needs none of the apt packages: zlib and libcurl come from the system,
and the QTSQ checkout builds with plain `make`. Steps 2 and 3 are the sequence
the 2026-10-31 audit had to find by hand on a clean Ubuntu 24.04 host (findings
P1-7 and P2-12). CI skips step 2 by downloading a prebuilt QTSQ artifact from
a repository secret (`SOFUU_QTSQ_CI_URL`), which is why the public path was
never written down.

Thank you for supporting Sofuu and respecting this governance model!
