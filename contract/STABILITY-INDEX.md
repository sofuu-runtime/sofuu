# Stability index (generated)

Generated from [`contract/stability.json`](stability.json) — audit date 2026-10-23.
Every public symbol carries exactly one tier. Do not edit by hand.

**Legend:** `T1` = guaranteed: conformance-tested, semver-protected · `T2` = supported: documented, may change with a minor bump and a deprecation note · `T3` = experimental: may change or vanish; never required by a guide or an app

## C ABI symbols (34 entries — T1 18, T2 16, T3 0)

| symbol | tier |
|---|---|
| `sofuu_destroy` | T1 |
| `sofuu_embed_abi_version` | T1 |
| `sofuu_embed_batch` | T1 |
| `sofuu_embed_image` | T1 |
| `sofuu_embed_info` | T1 |
| `sofuu_embed_local` | T1 |
| `sofuu_embed_set_log_cb` | T2 |
| `sofuu_engine_ctx` | T1 |
| `sofuu_eval_file` | T1 |
| `sofuu_eval_string` | T1 |
| `sofuu_flush_jobs` | T2 |
| `sofuu_flush_jobs_bounded` | T2 |
| `sofuu_free` | T1 |
| `sofuu_get_engine` | T2 |
| `sofuu_init` | T1 |
| `sofuu_loop_close` | T2 |
| `sofuu_loop_get` | T2 |
| `sofuu_loop_init` | T2 |
| `sofuu_loop_run` | T2 |
| `sofuu_loop_shutdown_engine` | T2 |
| `sofuu_rt_call` | T1 |
| `sofuu_rt_call_stream` | T1 |
| `sofuu_rt_cancel` | T1 |
| `sofuu_rt_eval` | T2 |
| `sofuu_rt_eval_nb` | T2 |
| `sofuu_rt_free` | T1 |
| `sofuu_rt_host_tool_resolve` | T2 |
| `sofuu_rt_install_rejection_tracker` | T2 |
| `sofuu_rt_new` | T1 |
| `sofuu_rt_register_host_tool` | T2 |
| `sofuu_rt_report_pending_rejections` | T2 |
| `sofuu_run_jobs` | T2 |
| `sofuu_voice_speak` | T1 |
| `sofuu_voice_transcribe` | T1 |

## CLI commands (21 entries — T1 13, T2 8, T3 0)

| symbol | tier |
|---|---|
| `add` | T2 |
| `agent` | T1 |
| `agents` | T1 |
| `bundle` | T2 |
| `chat` | T1 |
| `compile-bc` | T2 |
| `doctor` | T1 |
| `eval` | T1 |
| `help` | T1 |
| `infer` | T1 |
| `install` | T2 |
| `licenses` | T2 |
| `output` | T2 |
| `outputs` | T2 |
| `repl` | T1 |
| `run` | T1 |
| `serve` | T1 |
| `session` | T1 |
| `sessions` | T1 |
| `swarm` | T2 |
| `version` | T1 |

## HTTP routes (11 entries — T1 6, T2 5, T3 0)

| symbol | tier |
|---|---|
| `GET /health` | T2 |
| `GET /memories` | T2 |
| `GET /recall` | T1 |
| `GET /v1/models` | T1 |
| `POST /chat/completions` | T2 |
| `POST /forget` | T1 |
| `POST /infer` | T2 |
| `POST /remember` | T1 |
| `POST /share` | T2 |
| `POST /v1/chat/completions` | T1 |
| `POST /v1/completions` | T1 |

## Config keys (43 entries — T1 11, T2 27, T3 5)

| symbol | tier |
|---|---|
| `active` | T1 |
| `api_key` | T2 |
| `api_key(legacy flat)` | T3 |
| `archive_auto_recover` | T2 |
| `archive_enabled` | T2 |
| `archive_max_age_secs` | T2 |
| `archive_max_bytes` | T2 |
| `archive_max_count` | T2 |
| `archive_recovery_budget_chars` | T2 |
| `archive_redact` | T2 |
| `archive_retain_final` | T2 |
| `archive_retain_partial` | T2 |
| `archive_retain_tool_results` | T2 |
| `base_url` | T2 |
| `base_url(legacy flat)` | T3 |
| `brain` | T1 |
| `brain_path` | T1 |
| `budget_usd` | T1 |
| `ctx_window` | T1 |
| `ctx_window_explicit` | T2 |
| `effort` | T1 |
| `embed_model` | T2 |
| `embed_provider` | T2 |
| `ghost` | T2 |
| `max_output` | T2 |
| `memory_backend` | T1 |
| `ml` | T2 |
| `model` | T1 |
| `model(legacy flat)` | T3 |
| `no_think_models` | T2 |
| `permissions` | T1 |
| `pricing` | T1 |
| `profile` | T2 |
| `profile(legacy flat)` | T3 |
| `provider(legacy flat)` | T3 |
| `providers[]` | T1 |
| `recall_budget` | T2 |
| `recall_min` | T2 |
| `rlm` | T2 |
| `rss_warn_mb` | T2 |
| `spend_total_usd` | T2 |
| `sync` | T2 |
| `theme` | T2 |

## JS namespaces

| namespace | tier | members | T1 members |
|---|---|---|---|
| `sofuu.agent` | T1 | 25 | 12 |
| `sofuu.ai` | T1 | 21 | 13 |
| `sofuu.aotEval` | T3 | 1 | 0 |
| `sofuu.buffer` | T3 | 1 | 0 |
| `sofuu.capabilities` | T1 | 1 | 1 |
| `sofuu.chat` | T1 | 18 | 16 |
| `sofuu.context` | T2 | 1 | 0 |
| `sofuu.extractor` | T3 | 7 | 0 |
| `sofuu.fs` | T1 | 9 | 8 |
| `sofuu.http` | T1 | 3 | 2 |
| `sofuu.kv` | T2 | 4 | 0 |
| `sofuu.mcp` | T1 | 4 | 4 |
| `sofuu.memory` | T1 | 28 | 12 |
| `sofuu.ml` | T2 | 9 | 0 |
| `sofuu.parse` | T3 | 1 | 0 |
| `sofuu.policy` | T2 | 5 | 0 |
| `sofuu.process` | T1 | 11 | 11 |
| `sofuu.rlm` | T2 | 4 | 0 |
| `sofuu.sandbox` | T2 | 3 | 0 |
| `sofuu.spawn` | T1 | 2 | 2 |
| `sofuu.storage` | T2 | 6 | 0 |
| `sofuu.tools` | T1 | 13 | 11 |
| `sofuu.web` | T1 | 5 | 3 |

### `sofuu.agent` members

| member | tier |
|---|---|
| `CORE_PROMPT` | T2 |
| `VERSION` | T2 |
| `brainFor` | T3 |
| `cancel` | T1 |
| `contextWindow` | T2 |
| `create` | T3 |
| `current` | T2 |
| `define` | T1 |
| `getPermissions` | T1 |
| `getTrace` | T1 |
| `isTransientProviderError` | T3 |
| `list` | T1 |
| `loadDir` | T1 |
| `mapContext` | T2 |
| `maxOutputFor` | T2 |
| `modelCaps` | T2 |
| `pendingApprovals` | T1 |
| `pinProjectFact` | T2 |
| `rememberGist` | T2 |
| `renderTrace` | T1 |
| `resolveApproval` | T1 |
| `resume` | T2 |
| `run` | T1 |
| `runMany` | T1 |
| `setPermissions` | T1 |

### `sofuu.ai` members

| member | tier |
|---|---|
| `applyChatTemplate` | T2 |
| `complete` | T1 |
| `dot` | T1 |
| `embed` | T1 |
| `embedBatch` | T1 |
| `embedImage` | T1 |
| `embedInfo` | T2 |
| `embedInfoV2` | T2 |
| `embedLocal` | T1 |
| `embedLocalSemantic` | T2 |
| `embedLocalSemanticV2` | T2 |
| `estimateTokens` | T2 |
| `inferLocal` | T1 |
| `l2` | T1 |
| `modelCaps` | T2 |
| `resolveCaps` | T2 |
| `similarity` | T1 |
| `speak` | T1 |
| `stream` | T1 |
| `streamLocal` | T1 |
| `transcribe` | T1 |

### `sofuu.chat` members

| member | tier |
|---|---|
| `cancel` | T1 |
| `clear` | T1 |
| `clearSessions` | T1 |
| `compact` | T1 |
| `deleteSession` | T1 |
| `detectCaps` | T2 |
| `init` | T1 |
| `newSession` | T1 |
| `resolveApproval` | T1 |
| `resolveCaps` | T2 |
| `resume` | T1 |
| `sessionTurns` | T1 |
| `sessions` | T1 |
| `setPermissions` | T1 |
| `setProject` | T1 |
| `state` | T1 |
| `submit` | T1 |
| `usage` | T1 |

### `sofuu.extractor` members

| member | tier |
|---|---|
| `applyFacts` | T3 |
| `cleanJsonText` | T3 |
| `extract` | T3 |
| `parseFactsJson` | T3 |
| `processTurn` | T3 |
| `processTurnAsync` | T3 |
| `shouldExtract` | T3 |

### `sofuu.fs` members

| member | tier |
|---|---|
| `appendFile` | T1 |
| `exists` | T1 |
| `mkdir` | T1 |
| `readFile` | T1 |
| `readFileBytes` | T1 |
| `readdir` | T1 |
| `realpath` | T2 |
| `rm` | T1 |
| `writeFile` | T1 |

### `sofuu.kv` members

| member | tier |
|---|---|
| `flush` | T2 |
| `open` | T2 |
| `save` | T2 |
| `search` | T2 |

### `sofuu.mcp` members

| member | tier |
|---|---|
| `MCPClient` | T1 |
| `MCPServer` | T1 |
| `connect` | T1 |
| `serve` | T1 |

### `sofuu.memory` members

| member | tier |
|---|---|
| `clearWorking` | T2 |
| `close` | T1 |
| `compact` | T2 |
| `consolidate` | T2 |
| `count` | T1 |
| `decayTick` | T2 |
| `embeddingInfo` | T2 |
| `export` | T1 |
| `flush` | T1 |
| `forget` | T1 |
| `importRecords` | T1 |
| `inspect` | T2 |
| `kvHints` | T2 |
| `learn` | T1 |
| `list` | T1 |
| `markPositive` | T2 |
| `migrate` | T2 |
| `open` | T1 |
| `promoteWorking` | T2 |
| `recall` | T1 |
| `remember` | T1 |
| `rememberEntity` | T2 |
| `rememberEntityTurn` | T3 |
| `rememberWorking` | T2 |
| `retain` | T2 |
| `retainAgainst` | T2 |
| `search` | T1 |
| `stats` | T2 |

### `sofuu.ml` members

| member | tier |
|---|---|
| `alloc` | T2 |
| `compaction` | T2 |
| `feedback` | T3 |
| `freshness` | T2 |
| `info` | T2 |
| `relevance` | T2 |
| `supervisor` | T2 |
| `track` | T3 |
| `workset` | T3 |

### `sofuu.policy` members

| member | tier |
|---|---|
| `clear` | T2 |
| `evaluate` | T2 |
| `get` | T2 |
| `load` | T2 |
| `set` | T2 |

### `sofuu.process` members

| member | tier |
|---|---|
| `argv` | T1 |
| `chdir` | T1 |
| `cwd` | T1 |
| `env` | T1 |
| `exit` | T1 |
| `off` | T1 |
| `on` | T1 |
| `pid` | T1 |
| `platform` | T1 |
| `runtime` | T1 |
| `version` | T1 |

### `sofuu.rlm` members

| member | tier |
|---|---|
| `VERSION` | T2 |
| `logRoute` | T3 |
| `query` | T2 |
| `route` | T2 |

### `sofuu.storage` members

| member | tier |
|---|---|
| `Database` | T2 |
| `KeyValueStore` | T2 |
| `MessagesCollection` | T2 |
| `TableHandle` | T2 |
| `TracesCollection` | T2 |
| `open` | T2 |

### `sofuu.tools` members

| member | tier |
|---|---|
| `ENABLED` | T2 |
| `GROUP` | T1 |
| `TOOLS` | T1 |
| `VERSION` | T2 |
| `bash` | T1 |
| `code_eval` | T1 |
| `edit_file` | T1 |
| `glob` | T1 |
| `grep` | T1 |
| `list_dir` | T1 |
| `read_file` | T1 |
| `todo_write` | T1 |
| `write_file` | T1 |

### `sofuu.web` members

| member | tier |
|---|---|
| `ENGINES` | T3 |
| `TOOLS` | T1 |
| `VERSION` | T2 |
| `open` | T1 |
| `search` | T1 |

## Known gaps (planned, not yet public)

| symbol | surface | target tier | phase |
|---|---|---|---|
| `sofuu_capabilities_json()` | cabi | T1 | P1 |
| `sofuu.model.describe()` | js | T1 | P1 |
| `sofuu.model.list()` | js | T1 | P1 |
| `sofuu.session (namespace)` | js | T1 | P3 |
| `GET /capabilities` | http | T1 | P1 |
| `POST /v1/embeddings` | http | T2 | P3 |
| `session export/import (CLI + JS + HTTP)` | all | T1 | P3 |

