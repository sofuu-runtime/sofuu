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

If you have forked the repository and wish to build or test it on your machine:

```bash
# Build the binary
make

# Run the test suite
make test

# Verify runtime health
./sofuu doctor
```

Thank you for supporting Sofuu and respecting this governance model!
