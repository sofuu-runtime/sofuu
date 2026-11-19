# memory-stores-authorization-failure-durability — Memory stores: authorization, failure honesty, durability

Scope of this worker: `crates/sofuu-core/src/rt/memory.rs`,
`crates/sofuu-core/src/rt/memory_adapters.rs`,
`crates/sofuu-core/src/rt/kaze_memory.rs`,
`crates/sofuu-core/src/rt/storage.rs`, `crates/sofuu-core/src/rt/wal.rs`.
Division D02 of `docs/plans/PLAN-REAL-EXECUTION-REMEDIATION.md` (R3 items 3–5).

All rows below describe the **working tree** as of 2026-11-16 ~13:50 IST.
The working tree was already dirty when this worker started; nothing was
reverted, stashed, committed or checked out.

## Checks performed

| Check | Command / source read | Result | Evidence |
|---|---|---|---|
| D02: `memory.save` / `memory.recall` need an explicit family grant (H-2) | read `crates/sofuu-core/src/domains/robotics/hardware.rs:31-35` (`EFFECTOR_FAMILIES = caps::FAMILIES`, which contains `memory`), `:183-194` (`needs_backend_grant`), and `crates/sofuu-core/src/domains/kaze/vm.rs:914-922` (`authorise` runs for **every** capability call) | pass | `memory` is in `EFFECTOR_FAMILIES`, so a grant is required. Observed live: `sofuu script run read.kaze --policy allow.json --json` → `E_CAP_MISSING`, `"the host did not grant 'memory'. Grant it explicitly, e.g. --cap memory (H-2/backend)"`. |
| D02: the policy applies to the memory family (H-3) | read `hardware.rs:195-221`; the request built for the decision is `{"op":"tool.run","tool":"kaze.memory.<op>"}` | pass | Observed live with `--policy deny.json` (`{"tools":{"deny":["kaze.memory.*"]}}`): `E_CAP_DENIED`, `"tool 'kaze.memory.recall' is denied by policy (R7.6/tools.deny) (H-3/policy)"`, `status: "denied"`. A missing policy is `E_CAP_DENIED` too (`hardware.rs:195-197`), and an empty `tools.allow` grants nothing (`hardware.rs:212-216`). |
| D02: a configured-but-absent store fails closed, never with a canned value | read `crates/sofuu-core/src/domains/kaze/mod.rs:681-708` (`build_cap_host`) and `crates/sofuu-core/src/domains/kaze/caps.rs:497-501` (the runtime host's `_` arm) | pass | With no `--cap memory=…` the base `RuntimeHost` answers `memory.*` with `E_CAP_MISSING: embedded Kaze has no available backend for memory.save`. A grant without a backend (`--cap memory`) resolves to the same host. |
| D02: a store that exists but cannot be read is an error, not an empty result | read `crates/sofuu-core/src/rt/kaze_memory.rs:245-283` (`read_records`) | pass | Observed live: a directory at the store path → `E_KAZE_MEMORY`, `"cannot read durable memory store /tmp/kmem/dirstore.json: Is a directory (os error 21)"`; a store holding `{not json` → `E_KAZE_MEMORY`, `"corrupt storage data in /tmp/kmem/bad.json: invalid type: map, expected a sequence"`. A store that does not exist yet is a genuine empty result (`read_records` → `NotFound` → `Ok(vec![])`). |
| D02: a write that cannot reach the disk is an error, never a success | read `kaze_memory.rs:298-320` (`write_records`: stage → `wal::publish_paths` → rename) | pass | Observed live: `memory=file:` pointing at a directory → `E_KAZE_MEMORY`, `"cannot read durable memory store …: Is a directory"`, `ok: false`. The commit itself is a staged write plus `wal::publish_paths` (fsync the bytes, rename over the target, fsync the directory), so a failed publish returns an error and removes the staging file. |
| D02: the durable claim is proven by reopening the store in a **separate process** | read `scripts/test-kaze-memory-durable.py`; ran it against the local artifact | pass | `python3 scripts/test-kaze-memory-durable.py target/debug/sofuu` → `PASS: Kaze durable memory (process reopen, persistence, isolation, bytecode, standalone)`. The script writes in process 1, exits, and recalls in process 2 (`:54-62`), checks policy denial and approval-required (`:64-70`), checks that a new ephemeral process sees nothing (`:72-74`), and re-checks the same store through `.kazebc` (`:87-91`) and a compiled standalone binary (`:93-97`). |
| D02: the ephemeral store is real, per-run and isolated | ran `scripts/test-kaze-memory-real.py` | pass | `python3 scripts/test-kaze-memory-real.py target/debug/sofuu` → `PASS: Kaze per-run CMA, isolation, policy, limits, source, bytecode, JS and standalone`. |
| D02: the native QTSQ store really writes, reopens in another process, isolates and reports a flush failure | read `scripts/contract/harness/real_memory_store.js`; ran `scripts/test-real-memory.py` | pass | `SOFUU_BIN=$PWD/target/debug/sofuu python3 scripts/test-real-memory.py` → `PASS: real memory write, separate-process reopen, store isolation, readonly and flush failure`. The harness asserts a coded error on a readonly store, on a separate store, and on a real filesystem failure (`chmod 0500` on the store directory). |
| D02: the stored value encoding on disk is the documented one | observed the store file after a real write | pass | `--cap memory=file:$PWD/durable.json` wrote `[{"label":"k","value":"v1"},{"label":"k","value":"v2"}]` — one JSON object per record, in write order, with the label as the key. |
| D02: recall matches the label exactly and returns every value under it | ran a Kaze script that saves `k→v1`, `k→v2` and recalls `k` | pass | `result: "[\"v1\",\"v2\"]"` on both the ephemeral and the durable store; the durable file above holds both records. |
| D02: a value that cannot be encoded is refused | read `kaze_memory.rs:405-455` (`to_json`) | pass (type-checked) | A non-finite number fails `Number::from_f64` → `E_KAZE_MEMORY: memory numbers must be finite`; a non-text record key, a duplicate key, a value deeper than 16 levels or with more than 4096 nodes are refused the same way. New test `kaze_memory.rs:634-651`. |
| D02: the WAL commit path never reports a success it did not perform | read `crates/sofuu-core/src/rt/wal.rs:238-270` (`publish_paths`), `:296-430` (`Journal`) and the two call sites in `crates/sofuu-core/src/rt/memory.rs:660-682` (page publish) and `:6188-6212` (index publish) | pass (after this change) | Before: when the intent could not be recorded, or the publish failed, the journal file was left behind describing a commit that was no longer pending — it blocked the next writer in that directory for the 30 s stale window and `recover` replayed an intent nobody held. Now the journal is cleared in both outcomes (`memory.rs:676`, `memory.rs:6204`), and the staging file is removed only when the publish failed. |
| D02: the WAL directory fsync covers a relative store path | read `wal.rs:254-278` | pass (after this change) | `Path::parent` of a bare relative name is `Some("")`; the old code then opened `""`, which always fails, so the directory the rename changed was never fsynced for a store named `store.json` in the working directory. `target_dir` now resolves that case to `.` (`wal.rs:272-278`), with a pure-function test (`wal.rs:646-655`). |
| D02: the native structured store (`sofuu.storage`) opens, inserts, queries, counts, flushes and closes honestly | read `crates/sofuu-core/src/rt/storage.rs:114-235` (open), `:320-404` (insert), `:406-573` (query/flush) and the JS bindings `:622-935` | pass (after this change) | A corrupt manifest, an unreadable segment and a directory that cannot be listed now fail the open with `E_STORAGE: …`; a failed flush keeps the manager dirty and is reported by `flush()` and `close()`; `count()` for an unknown table is `None` (a typed error), not `0`. |
| D02: a malformed query must not widen a read | read `storage.rs:709-800` | pass (after this change) | The query JSON used to decode with `unwrap_or(Value::Null)`, and a null query carries **no filter** — a typo in the query string silently widened the read to the whole table and the caller received every row as though it had asked for them. `parse_query` now refuses anything that is not a JSON object (`storage.rs:781-846`), with a test (`storage.rs:1174-1197`). |
| D02: a failed serialization must not become an empty result set | read `storage.rs:752-772` | pass (after this change) | `serde_json::to_string(&rows).unwrap_or_else(|_| "[]")` answered `[]` when the result set could not be serialized — a storage failure presented as "no matching rows". It now throws `E_STORAGE: cannot serialize the query result`. |
| D02: the store that answered must be named in the trace | read `crates/sofuu-core/src/domains/kaze/caps.rs:365-390` (`note_with_backend`, `Action.backend`) and `Action::render` | pass (after this change) | The memory hosts recorded their actions through `note()`, which names the backend `local` — the same token for a per-run store and a committed one. Both hosts now record their own backend (`memory-ephemeral` / `memory-durable`, `kaze_memory.rs:73-76`), so the trace line, the CLI text output and the CLI JSON `actions` array distinguish them. New test `kaze_memory.rs:622-639`. |
| D02: a durable writer must be able to arbitrate its own read-modify-write | read `crates/sofuu-core/src/rt/brain_lock.rs:246-375` (`acquire`) and `kaze_memory.rs:325-355` (`ensure_writer`) | pass (after this change) | `brain_lock::acquire` degrades to an **unlocked** handle when the lock file cannot be taken (any failure other than a conflict). The durable write is a read-modify-write, so two unarbitrated writers would each publish their own set and one record would vanish with both writes reported as successful. An unarbitrated writer is now refused (`kaze_memory.rs:342`). New test `kaze_memory.rs:657-676`. |
| D02: the store size bound is enforced before the file is loaded | read `kaze_memory.rs:245-283` | pass (after this change) | `read_records` now checks the file size against the documented 256 KiB bound before reading it, instead of loading an arbitrarily large file into memory. New test `kaze_memory.rs:641-655`. |
| D02: the Kaze KV store open failure is typed | read `crates/sofuu-core/src/rt/memory.rs:5805-5975` (`kv_store_open`), `:6227-6280` (`js_kv_open`) | pass (after this change) | `kv_store_open` now returns `Result<_, String>` and `js_kv_open` throws a coded `E_STORE` error carrying the reason, instead of a bare `InternalError: Failed to initialize KV`. |
| D02: the pluggable adapter loader fails closed | read `crates/sofuu-core/src/rt/memory_adapters.rs:34-82` and `src/js/memory_api.js:105-112` | pass | An un-inflatable or un-evaluatable adapter source prints the reason and answers `false`; the JS side turns that into a typed `E_ADAPTER` refusal (`memory_api.js:109`). No change needed in this file. |
| D02: timeouts and cancellation | read `crates/sofuu-core/src/domains/kaze/vm.rs:579-599` (the deadline poll at every loop head), `crates/sofuu-core/src/rt/deadline.rs`, `crates/sofuu-core/src/domains/kaze/mod.rs:834-837` | unverified | See *Blocked or unverified* below. |
| The edits compile | `cargo check -p sofuu-core --lib --tests` | pass | Exit status 0. Cargo reports the test targets as compiled (`sofuu-core (bin "sofuu" test) generated 5 warnings`); the only warnings are pre-existing ones in `doctor.rs`, `chat.rs`, `session.rs`, `domains/memory/cma.rs`, `domains/infer/*` and `rt/mcp.rs`. No warning or error comes from the five files this worker touched. |
| The Rust unit tests | — | **not run** (build/test policy: the coordinator owns `cargo build` / `cargo test`) | The new and updated unit tests are **type-checked only** (`cargo check --lib --tests`, exit 0). They must be executed by the coordinator. |
| The Python memory checks against the current binary | `scripts/test-kaze-memory-durable.py`, `scripts/test-kaze-memory-real.py`, `scripts/test-real-memory.py` | pass (against the **pre-change** artifact) | The artifact is `target/debug/sofuu`, sha256 `64a7da0f010ba41480a27b48f2bb15efafdc12ba297657c53abcc2113291972f`, built 12:23 — **before** this worker's edits. The receipts above therefore prove the pre-existing behaviour, not the changed behaviour. The changed behaviour is covered by `cargo check` and by the new unit tests, which the coordinator must run. |

## Gaps found and fixed

| Gap | Fix | Path:line |
|---|---|---|
| A malformed `sofuu.storage` query decoded to a **null query**, which carries no filter at all: a typo silently widened the read to the whole table and the caller received every row as though it had asked for them | `parse_query` refuses any query that is not a JSON object, and `js_storage_query` throws `E_STORAGE: the query is not valid JSON` / `E_STORAGE: the query must be a JSON object` | `crates/sofuu-core/src/rt/storage.rs:781-846`, `:728-736` |
| A result set that could not be serialized was answered as `[]` — a storage failure presented as "no matching rows" | the serialization failure now throws `E_STORAGE: cannot serialize the query result` | `crates/sofuu-core/src/rt/storage.rs:752-768` |
| `storage.open` handed out a handle even when the manager could not be registered, so the caller held a handle that named no store | the registry lock failure now throws `storage registry is unavailable` and no handle is returned | `crates/sofuu-core/src/rt/storage.rs:645-651` |
| `storage.close` reported `false` for a failed close but the reason was lost | the close now logs the reason (`[sofuu/storage] the close could not persist the pending rows: …`) next to the boolean it returns | `crates/sofuu-core/src/rt/storage.rs:900-925` |
| The WAL journal was left behind when the intent could not be recorded or the publish failed: it described a commit that was no longer pending, blocked the next writer in that directory for the 30 s stale window, and `recover` replayed an intent nobody held | the journal is cleared in both outcomes — `commit()` when the rename landed, `abandon()` when it did not — and the staging file is removed only on failure | `crates/sofuu-core/src/rt/wal.rs:411-424`; `crates/sofuu-core/src/rt/memory.rs:663-682`, `:6192-6212` |
| A store named by a bare relative path (`store.json` in the working directory) never had its directory fsynced: `Path::parent` returns `Some("")` and `File::open("")` always fails, so the rename went unrecorded across a power loss | `target_dir` resolves the empty parent to `.` | `crates/sofuu-core/src/rt/wal.rs:254-278` |
| Both memory hosts recorded their actions with the backend `local`, so the trace could not tell a per-run store from a committed one | each host records its own backend (`memory-ephemeral`, `memory-durable`) through `note_with_backend`, so the trace line, the CLI text and the CLI JSON name the store that answered | `crates/sofuu-core/src/rt/kaze_memory.rs:73-76`, `:131-138`, `:172-179`, `:371-379`, `:406-414` |
| A durable writer that could not take the store's writer lock was allowed to write anyway. The durable write is a read-modify-write, so two unarbitrated writers would each publish their own set and one record would vanish — with **both** writes reported as successful | `ensure_writer` refuses an unarbitrated (`degraded`) lock with `E_KAZE_MEMORY: … cannot arbitrate concurrent writers: …` | `crates/sofuu-core/src/rt/kaze_memory.rs:325-355` |
| The durable store's size bound was only checked **after** the whole file had been loaded into memory | `read_records` checks the file size against the documented bound before reading it | `crates/sofuu-core/src/rt/kaze_memory.rs:245-283` |
| `sofuu.kv.open` reported a store failure as a bare `InternalError: Failed to initialize KV`, with the reason only on stderr | `kv_store_open` returns the reason and `js_kv_open` throws a coded `E_STORE` error carrying it | `crates/sofuu-core/src/rt/memory.rs:5805-5815`, `:5852-5860`, `:5940-5948`, `:6262-6272` |
| The stored contract (value encoding, key/query semantics, isolation, durability, size bounds) was scattered across the code and never stated in one place | the module documentation of `kaze_memory.rs` now states all five, with the size bounds and the durable-claim proof | `crates/sofuu-core/src/rt/kaze_memory.rs:26-57` |
| The recall semantics were never covered by a test, and neither were the store-provenance, the size bound, the unarbitrated writer or the malformed query | new tests: exact-match recall, an unencodable value, the trace's store name, the oversized store, the unarbitrated writer, the malformed query, the relative-target directory resolution | `crates/sofuu-core/src/rt/kaze_memory.rs:573-676`, `crates/sofuu-core/src/rt/storage.rs:1174-1197`, `crates/sofuu-core/src/rt/wal.rs:646-655` |

## Register rows

| Division | Surface | Status | Evidence / remaining gap |
|---|---|---|---|
| D02 memory | Kaze `memory.save` / `memory.recall` on the ephemeral CMA (`memory=ephemeral`) | `real` | `crates/sofuu-core/src/rt/kaze_memory.rs` (`EphemeralMemoryHost`) over Sofuu's real CMA (`crate::memory::cma::Cma`). `scripts/test-kaze-memory-real.py` → PASS. A run-scoped store: it is **not** durable and the trace says so (`memory-ephemeral`). |
| D02 memory | Kaze `memory.save` / `memory.recall` on the durable file store (`memory=durable`, `memory=file:<path>`) | `real` | `DurableMemoryHost`: every save is staged, fsynced and renamed over the target before the call returns (`kaze_memory.rs:298-320`). `scripts/test-kaze-memory-durable.py` → PASS. |
| D02 memory | The durable claim across a process boundary | `real` | `scripts/test-kaze-memory-durable.py` writes in process 1, exits, and recalls in process 2; the same store is re-read through `.kazebc` and through a compiled standalone binary. The Rust equivalent (`a_durable_record_survives_the_process_that_wrote_it`) spawns a child process and reads the record back — **type-checked, not executed** by this worker. |
| D02 memory | Store isolation (path ↔ path, durable ↔ ephemeral, run ↔ run) | `real` | `scripts/test-kaze-memory-durable.py` (an ephemeral run in a new process sees nothing) and `scripts/test-kaze-memory-real.py` (isolation, limits). Rust tests for path isolation and durable↔ephemeral isolation exist and are type-checked. |
| D02 memory | Authorization of the memory family (declaration H-1, grant H-2, policy H-3) | `real` | `crates/sofuu-core/src/domains/kaze/vm.rs:914-928` runs `hardware::authorise` for every capability call; the decision is `kaze.memory.<op>`. Observed live: `E_CAP_MISSING` without a grant, `E_CAP_DENIED` under `tools.deny`, `E_CAP_APPROVAL_REQUIRED` under `approval.mode: always` (`scripts/test-kaze-memory-durable.py:64-70`). |
| D02 memory | Missing configuration / denied access on `memory.*` | `real` | No `--cap memory=…` → the base runtime host answers `E_CAP_MISSING: embedded Kaze has no available backend for memory.save` (`caps.rs:497-501`); a denied policy never reaches the store (`vm.rs:915-922`). |
| D02 memory | An absent, unreadable, corrupt or oversized store | `real` | `read_records` (`kaze_memory.rs:245-283`): absent → genuine empty result; unreadable → `E_KAZE_MEMORY: cannot read durable memory store …`; unparsable → `E_KAZE_MEMORY: corrupt storage data in …`; over the 256 KiB bound → `E_KAZE_MEMORY: … larger than the … byte limit`. All four observed live except the last (type-checked test). |
| D02 memory | A write that cannot reach the disk | `real` | `write_records` stages, publishes atomically and returns `E_KAZE_MEMORY: cannot commit durable memory store …` when the publish fails; the staging file is removed. Observed live (a directory at the store path): `ok: false`, `E_KAZE_MEMORY`. |
| D02 memory | Ephemeral vs durable provenance in the API, the CLI output and the trace | `real` | The backend selection is `memory=ephemeral` / `memory=durable` / `memory=file:<path>`; the resolved backend is in the JS report's `backends` array (`domains/kaze/mod.rs:1287-1295`); the trace line now carries `[real: memory-ephemeral]` / `[real: memory-durable]` (`kaze_memory.rs:73-76`, `caps.rs:151-175` renders `[<mode>: <backend>]`). |
| D02 memory | WAL / storage commit path (atomic publish, write-ahead journal, directory fsync) | `real` | `crates/sofuu-core/src/rt/wal.rs`: `publish_paths` = fsync the staged bytes → rename → fsync the directory; `Journal` records the intent (target, staging path, sha256) and `recover` rolls a crashed commit forward or back. The journal is now cleared in both outcomes of a publish attempt. |
| D02 memory | Native QTSQ brain store (`sofuu.memory.open` / `remember` / `recall` / `flush` / `close`) | `real` | `scripts/test-real-memory.py` → PASS (write, separate-process reopen, isolation, readonly refusal, a real filesystem flush failure). `crates/sofuu-core/src/rt/memory.rs` throws coded errors (`E_CORRUPT`, `E_LOCKED`, `E_ENOSPC`, `E_STORE`, `E_INVALID_PATH`, `E_ACCES`). |
| D02 memory | Native structured store (`sofuu.storage`: open, insert, query, count, flush, close) | `real` | `crates/sofuu-core/src/rt/storage.rs`: a corrupt manifest, an unreadable segment and an unlistable directory fail the open with `E_STORAGE:`; `count()` for an unknown table is a typed error, not `0`; a failed flush keeps the manager dirty and is reported by `flush()` and `close()`; a malformed query is refused instead of widening the read. |
| D02 memory | Timeout and cancellation **during** a memory operation | `unverified` | The VM polls the host deadline at every loop head (`domains/kaze/vm.rs:579-599`) and the fuel budget stops a runaway script with `E_TIMEOUT`, so a run cannot continue past its deadline. A memory operation itself is a bounded, synchronous, local call with no internal retry and therefore no mid-operation cancellation point; a store on a hung network filesystem would block past the deadline. Not exercised on this machine. |
| D02 memory | Two **concurrent** writers on one durable path | `unverified` | The writer lock is taken on the first save (`kaze_memory.rs:325-355`), and an unarbitrated writer is now refused, but two live processes racing on one store were not exercised here. `scripts/test-kaze-memory-durable.py` runs its processes sequentially. |
| D02 memory | A store on a filesystem that refuses the directory fsync | `unverified` | The publish path treats a failed directory fsync as best-effort: the bytes are durable and the rename landed, so the write is reported as done, but the directory entry's own durability across a power loss is not guaranteed. Recorded as a limitation rather than as a claim. |

## Blocked or unverified

* **The new Rust unit tests were not executed.** Build/test policy puts
  `cargo build` / `cargo test` with the coordinator. `cargo check -p sofuu-core
  --lib --tests` passes (exit 0), so the tests compile, but their assertions
  are unproven here: the exact-match recall order, the trace's store name, the
  oversized-store refusal, the unarbitrated-writer refusal, the malformed
  query and the relative-target directory resolution. The coordinator must run
  them before the register rows are promoted.
* **The Python memory receipts are for the pre-change artifact.** The debug
  binary used by `scripts/test-kaze-memory-durable.py`,
  `scripts/test-kaze-memory-real.py` and `scripts/test-real-memory.py` was
  built at 12:23, before this worker's edits
  (sha256 `64a7da0f010ba41480a27b48f2bb15efafdc12ba297657c53abcc2113291972f`).
  They prove the pre-existing behaviour. They must be re-run against a build
  that contains these changes; the changes that could alter them are the
  durable writer lock, the store size bound and the WAL journal handling.
* **Timeout and cancellation inside a memory operation are unverified** (see
  the register rows).
* **Concurrent writers on one durable path are unverified.**
* **The directory-fsync limitation is unverified** on a filesystem that
  refuses it.
* No model weights were downloaded, read or needed for this area.

## Notes for the coordinator

* **Shared-file changes I did not make** (they are outside my scope; the
  coordinator or the owning worker should apply them):
  * `src/js/storage.js` — `Database.count(table)` returns `0` when the handle
    is closed (a closed store reported as an empty table), and `Database.close()`
    discards the boolean `__sofuu_storage_close` returns, so a close that
    could not persist its rows is silent on the JS side. The native side now
    returns `false` and logs the reason (`storage.rs:900-925`); the JS side
    should surface it.
  * `src/js/storage.js` — `Database.insert` returns a row id for a row that is
    only in memory until the debounced flush; a program that never flushes or
    closes loses the row silently, because the storage registry is a
    `OnceLock` static whose `StorageManager`s are never dropped at process
    exit. A durable claim for `sofuu.storage` needs an explicit flush or
    close.
  * `crates/sofuu-core/src/domains/kaze/mod.rs` — `build_cap_host` opens the
    durable store at the default path `.sofuu/kaze_memory.json` relative to
    the working directory; the report names the resolved backend but the CLI
    output does not name the store path for a `memory=durable` run. Worth a
    line in the CLI output if the durable claim must be readable from the
    output alone.
  * `crates/sofuu-core/src/domains/kaze/caps.rs` — the durable store's writer
    lock lives beside the store as `<store>.lock`; a run that writes leaves
    that file behind. It is empty apart from the holder's pid and is released
    by the kernel when the process dies, but it is a side file the CLI should
    mention if the store directory is documented.
* **The `Action::render` change by the Kaze worker** (the trace line now ends
  in `[<mode>: <backend>]`) is what makes the memory store provenance
  visible. This worker's `memory-ephemeral` / `memory-durable` backend names
  ride on that rendering; if the rendering changes, these two constants are
  the only thing to re-check.
* **`StorageManager::count` now returns `Option<usize>`** (the earlier
  swarm's change). The only in-tree callers are in `storage.rs` itself;
  `crates/sofuu-capi` does not use it. `cargo check` confirms this.
* **The `contract/CAPABILITY-REGISTER.md` D02 row** currently says the memory
  surfaces are `real`. That claim still holds, and this worker's rows add the
  detail (the failure modes, the WAL/journal behaviour, the provenance and the
  two `unverified` rows) — but the durable-store row should not be read as
  covering the changed code until the coordinator re-runs the three Python
  memory checks and the Rust unit tests against a build that contains these
  changes.
