# Contributing to Sofuu

We welcome contributions to Sofuu! Whether you are fixing bugs, improving documentation, submitting toolpacks, or writing tests, thank you for helping make Sofuu the runtime for the AI era.

---

## 🛠️ Development Setup

### Prerequisites
- **macOS** (Apple Silicon or Intel) or **Linux** (x86_64 or aarch64)
- `clang` / `gcc` (C99 compiler with atomic support)
- `make`
- `rust` and `cargo` (for QTSQ memory codec test verification)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/sofuu-runtime/sofuu.git
cd sofuu

# Build the runtime binary
make

# Run the test suite
make test

# Verify runtime doctor and QTSQ memory
./sofuu doctor
```

---

## 🧪 Testing Guidelines

Before opening a pull request, ensure all test suites pass cleanly:

```bash
make test             # End-to-end JavaScript test suite
make size-check       # Verifies stripped binary size remains <= 5.5 MB
make bc-check         # Checks bytecodes are synced with their JavaScript drivers
```

---

## 📜 Coding Conventions

1. **Zero External Dependencies**: The standalone binary must never depend on external shared libraries or runtime daemons.
2. **Binary Size Discipline**: Every PR must preserve our strict size cap ($\le 5.5\text{ MB}$ uncompressed stripped binary).
3. **Prompt Sovereignty**: Never hardcode hidden system prompts or provider preferences into agent tools or the core engine.
4. **Clean Commits**: Write clear, descriptive commit messages following the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`, `perf:`).

---

## 💬 Community & Discussions

- **GitHub Issues**: Use for bug reports and reproducible test cases.
- **Documentation**: [sofuu.xyz/docs](https://sofuu.xyz/docs)
- **Twitter/X**: [@sofuu_runtime](https://x.com/sofuu_runtime)
