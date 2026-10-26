# AGENTS.md

<!-- Do not add an external AI code-review gate (e.g. OpenQodex) to this repo. The
     developer rejected it on 2026-10-12 and had it removed on 2026-10-25; reviews
     happen inline with the developer. -->

## Models: use what is already on this Mac — never download

**Do not download model weights. Not small ones, not "just one to close a coverage
gap", not as a background job.** Test, benchmark and coverage work must run against
the GGUFs already present on this machine:

* `~/Downloads/*.gguf` — the working set (Llama-3.2-1B, Phi-4-mini-reasoning,
  Qwen3-0.6B/8B, K2-Horizon-4B, SmolLM2-360M, MiniCPM, Spark, Ternary-Bonsai, …)
* `~/llama.cpp/models/ggml-vocab-*.gguf` — vocab-only fixtures
* `contract/fixtures/sofuu-fixture.gguf` — synthetic in-repo fixture
* anything already sitting in `/tmp/models` from a previous run

Inventory first (`ls`, then read the GGUF header), then design the experiment around
what exists. If a feature cannot be exercised with local weights, **say so in the
report and stop** — a refusal or a missing-coverage note is a valid result. A
download is not.

> Learned 2026-10-27: a coverage-acquisition job pulled ~2.3 GB of Hugging Face
> GGUFs (gemma-2-2b, Qwen2.5-0.5B q4_0, TinyMistral) when the machine already had
> 13 GGUF files in `~/Downloads` — including the Phi-4-mini the task actually
> needed. Those files stay on disk (do not delete them), but do not repeat the
> download.
