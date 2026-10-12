# PLAN — Stable Hybrid Storage for Main Sofuu

> Status: core main-Sofuu storage phases implemented; output archive and
> optional maintenance remain deferred
>
> Date: 2026-08-31
>
> Scope: the main Rust Sofuu runtime and CLI in
> /Users/priyanshuboruah/ai-native-js-runtime
>
> This document is the design contract for the implementation. The core
> storage work described in Phases 1–5 is now implemented in the main Rust
> Sofuu binary; the remaining phases are intentionally still future work.

## 1. Final design decision

Main Sofuu should use bounded QTSQ segment files plus a small QTSQ manifest:

~~~text
append event
  -> update one bounded active segment
  -> rotate it to an immutable segment when a limit is reached
  -> publish one new manifest atomically
  -> read only the segments needed by the caller
~~~

This is the stable middle ground:

- it preserves fast incremental durability;
- it avoids one file per event;
- it avoids rewriting one growing session file;
- it keeps the QTSQ header overhead amortized across event batches;
- it allows recent-context and peer reads without decoding all history;
- it preserves legacy files and does not require a QTSQ repository change.

There must never be one file per token, one file per stream delta, or one
unbounded session file.

## 2. Current repository facts

The repository currently contains two storage paths. They must not be merged
accidentally.

| Area | Current behavior | Treatment in this plan |
| --- | --- | --- |
| Main Rust session mesh | crates/sofuu-core/src/session.rs stores each session as one flat QTSQ file, rewrites it on every persisted event, caps events at 300, and keeps liveness in plaintext registry.json | Add a segmented main-Sofuu store and retain the flat reader |
| Main Rust reads | load_session_data decodes the complete flat session; peer polling loads each peer session | Add manifest, recent, and sequence-range reads |
| Sofuu QTSQ bridge | crates/sofuu-ffi/src/qtsq.rs stores payloads below 500 KiB raw and unencrypted, and payloads at or above 500 KiB compressed and plaintext; old encrypted files remain readable | Reuse this bridge and close any Sofuu-side large-write atomicity gap |
| Desktop JavaScript path | src/js/chat.js uses sessions/<id>/session.qtsq and one event file per desktop conversation event | Do not modify desktop in this plan; keep the main store in a distinct namespace |
| Output archive | PLAN-OUTPUT-ARCHIVE.md defines timestamped logical output records under .sofuu/outputs/ | Keep it additive; use its own indexed policy |

The main Rust write path is Session::persist. It serializes all retained
SessionData and sends it through qtsq_session_save. The main read paths include
load_session_data, past_turns, session_context, and PeerWatch::poll.

This distinction matters: the main Rust problem is primarily rewrite and
decode amplification; the desktop problem is primarily file-count
amplification.

## 3. Problems and desired result

### 3.1 Current main-runtime problem

Every main session event currently causes the complete retained session payload
to be serialized and written again. Every peer poll or context request can
decode the complete session file.

The existing 300-event cap limits the worst case, but it does not prevent:

- repeated rewriting of events that have not changed;
- repeated QTSQ encode/decode work;
- unnecessary reads when only the last few events are needed;
- larger latency as event text grows.

### 3.2 Current desktop problem

The desktop path writes one event per file. That provides good crash locality,
but long sessions create many directory entries, many opens, and many QTSQ
headers.

The main-Sofuu plan must not reuse the desktop directory schema. The user
previously requested main Sofuu work without desktop changes, so this design
keeps the namespaces separate.

### 3.3 Desired result

For main Sofuu, a normal event should rewrite only a bounded active segment and
its small manifest. Closed segments must never be rewritten during ordinary
chat. Reads should use the manifest to select only the required segment files.

## 4. Goals

The implementation must:

1. reduce write amplification for main Sofuu;
2. bound the active rewrite size;
3. keep every published file independently valid and checksummed;
4. make interrupted writes recoverable;
5. load only metadata, recent events, or an explicit sequence range;
6. preserve the current 300-event and 20-note limits in release one;
7. preserve legacy flat QTSQ session reads;
8. preserve Session behavior and existing session commands;
9. make peer polling incremental;
10. support concurrent sessions and safe same-session writes;
11. use the current Sofuu QTSQ bridge;
12. keep new records unencrypted;
13. provide explicit migration, repair, statistics, and rollback;
14. keep output-archive failures non-fatal to a chat turn.

## 5. Non-goals

The first implementation must not:

- modify /Users/priyanshuboruah/projects/black-hole-disk;
- build or modify sofuu-desktop;
- change src/js/chat.js;
- replace the desktop conv/ layout;
- make a full transcript one unbounded file;
- create token, character, or stream-delta files;
- silently increase the 300-event mesh retention cap;
- rewrite every old session at startup;
- add new encryption to local session/output records;
- require random-access support not supplied by QTSQ;
- add ML parameters, retraining, remote storage, or telemetry;
- inject all old segments into every model request;
- make outputs/ the mandatory source of truth for session resume.

## 6. Storage contract

### 6.1 Source of truth

For a new main-Sofuu session, the normal source of truth is:

1. one valid committed manifest;
2. immutable sealed segment files;
3. at most one mutable active segment.

Temporary and orphan files are recovery candidates only. They are never normal
read targets.

The manifest is authoritative for ordinary reads. Segment contents are used to
repair a manifest after an interrupted transaction. A manifest entry must not
be trusted until its path, session ID, range, size, and optional hash have been
validated.

### 6.2 Main-Sofuu namespace

The main runtime must not use the desktop path:

~~~text
sessions/<session-id>/
~~~

Use:

~~~text
sessions/<session-id>.store/
~~~

The suffix is intentional. It prevents a main segmented manifest from being
confused with the desktop schema-2 session index.

### 6.3 Proposed layout

~~~text
<project>/
  .sofuu/
    sessions/
      registry.json
      .lock
      <legacy-session-id>.qtsq
      <session-id>.store/
        manifest.qtsq
        .lock
        segments/
          segment-00000001-00000064.qtsq
          segment-00000065-00000128.qtsq
          active-00000129.qtsq
          *.tmp
~~~

The existing registry.json remains plaintext liveness bookkeeping. It is not
conversation payload and should not be routed through the encrypted QTSQ vault.

### 6.4 File roles

| File | Purpose | Write rule |
| --- | --- | --- |
| registry.json | project-wide session summary and heartbeat | existing project lock and atomic replacement |
| .lock | registry lock | advisory lock |
| <id>.qtsq | legacy flat main session | read-only after migration |
| <id>.store/manifest.qtsq | metadata, retention state, and segment index | temporary file, sync, atomic replace |
| segments/segment-start-end.qtsq | immutable event batch | publish once, never mutate |
| segments/active-start.qtsq | current bounded event batch | temporary file, sync, atomic replace |
| *.tmp | incomplete candidate | ignored by normal readers |
| <id>.store/.lock | same-session writer lock | advisory lock |

## 7. QTSQ responsibilities

### 7.1 Why QTSQ helps Sofuu

QTSQ is useful here as a payload container because it supplies:

- a custom, versioned file boundary;
- type and format identification;
- checksums;
- one loader for manifests and segments;
- raw storage for small local payloads;
- compression for large aggregate payloads;
- compatibility with older encrypted records.

QTSQ is not a database, directory index, lock manager, or transaction
coordinator. Sofuu must provide those layers.

### 7.2 Required write policy

Use the existing Sofuu QTSQ policy for every new manifest and segment:

- serialized payload below 500 KiB: raw QTSQ, no compressor, no encryption;
- serialized payload at or above 500 KiB: current QTC compression, plaintext
  writer, no encryption.

The threshold applies to the complete serialized manifest/segment payload.
Batching several small events is therefore useful: one raw QTSQ header is
amortized across the batch.

The 103 B to 4.6 KB issue was caused by the old security block. This plan must
not restore that block. A small aggregate segment can still be larger than its
JSON bytes because of a QTSQ header and filesystem metadata, but it must not
incur the old encryption expansion.

### 7.3 Required read policy

Readers must:

1. load raw current records;
2. load current compressed plaintext records;
3. retain legacy encrypted-read compatibility;
4. validate QTSQ type, checksum, and payload schema;
5. reject wrong-session, malformed, truncated, or unsafe records;
6. never decrypt a new raw record;
7. never pass codec or filesystem error text to the model as trusted context.

### 7.4 Sofuu atomicity requirement

The current small raw QTSQ path already performs a temporary write, sync, and
rename. The large compressed path must receive the same outer atomic guarantee
before segmented writes are enabled.

This is a Sofuu-side requirement. First try to implement it in the Sofuu FFI
wrapper or segment-store writer by writing to a same-directory temporary target
and then renaming. Modify the QTSQ repository only if an audit proves that the
existing public API makes this impossible.

## 8. Versioned payload schemas

Manifest and segment payloads are JSON inside QTSQ. They are not the same as
the reconstructed legacy SessionData object.

### 8.1 Manifest

~~~json
{
  "format": "sofuu.session.manifest",
  "storage_schema": 1,
  "session_id": "s-...",
  "created": 1780000000,
  "host": "machine",
  "cwd": "/project",
  "model": "model-name",
  "provider": "provider-name",
  "task": null,
  "notes": [],
  "next_seq": 129,
  "retained_from_seq": 1,
  "event_count": 128,
  "last_event_t": 1780000012,
  "segments": [
    {
      "id": "segment-00000001-00000064",
      "file": "segments/segment-00000001-00000064.qtsq",
      "state": "sealed",
      "start_seq": 1,
      "end_seq": 64,
      "event_count": 64,
      "payload_bytes": 12000,
      "sha256": "..."
    },
    {
      "id": "active-00000065",
      "file": "segments/active-00000065.qtsq",
      "state": "active",
      "start_seq": 65,
      "end_seq": 128,
      "event_count": 64,
      "payload_bytes": 11000,
      "sha256": "..."
    }
  ]
}
~~~

Required invariants:

- format equals sofuu.session.manifest;
- storage_schema is separate from the QTSQ container version;
- session_id equals the owning session ID;
- next_seq is one greater than the highest published sequence;
- event_count equals the sum of published segment counts;
- ranges are ordered and non-overlapping;
- at most one segment is active;
- sealed ranges are contiguous after retained_from_seq;
- file paths are relative and confined to the store segments directory;
- payload_bytes and sha256 describe the serialized segment payload;
- future unknown fields are ignored where safe.

### 8.2 Segment

~~~json
{
  "format": "sofuu.session.segment",
  "storage_schema": 1,
  "session_id": "s-...",
  "start_seq": 65,
  "end_seq": 128,
  "events": [
    {
      "seq": 65,
      "t": 1780000001,
      "kind": "prompt",
      "text": "..."
    }
  ]
}
~~~

Required invariants:

- format equals sofuu.session.segment;
- session_id matches the manifest;
- event sequences are strictly increasing;
- first and last event sequences equal start_seq and end_seq;
- event count equals end_seq minus start_seq plus one;
- event text and current event kinds remain intact;
- no event from another session is accepted;
- QTSQ checksum and JSON validation both succeed.

### 8.3 Sequence compatibility

The current main SessionEvent has timestamp, kind, and text but no persisted
sequence. The segmented layer must use an internal stored-event shape with seq.

The implementation should:

- preserve the public/reconstructed SessionEvent API where possible;
- assign a monotonic sequence at persistence time;
- reconstruct legacy sequence values from legacy event order;
- never use wall-clock timestamps as the sole ordering key;
- persist next_seq in the manifest;
- make sequence assignment deterministic under the session lock.

## 9. Segment sizing and retention

These defaults are starting values to benchmark, not values to tune by guess.

### 9.1 Soft rotation thresholds

Before adding an event, rotate if the new active payload would exceed either:

- 64 events; or
- 256 KiB serialized payload.

This bounds ordinary active rewrites to a small fraction of a long session.

### 9.2 Hard thresholds

An active segment must not exceed:

- 128 events; or
- 1 MiB serialized payload.

If one event is larger than the soft limit, store it as a single-event segment.
If one event exceeds the hard limit, send it through the large QTSQ path and
record a diagnostic. Never silently truncate or split event text.

### 9.3 Compression interaction

Normal active and sealed segments should remain below 500 KiB and therefore use
raw QTSQ. Offline compaction may create a payload at or above 500 KiB, in which
case the existing compressed plaintext path is used.

Rotation must not be designed merely to trigger compression. Compression is
for genuinely large payloads.

### 9.4 Retention

Release one preserves:

- 300 retained session events;
- 20 retained notes.

When the event cap is crossed:

1. remove complete oldest sealed segments first;
2. if the boundary splits a segment, rewrite only that segment's retained
   suffix;
3. update retained_from_seq;
4. commit the new manifest before deleting anything;
5. record the removed sequence range for diagnostics.

This session mesh is not the permanent full conversation archive. Durable
final/partial/error outputs remain covered by the output archive plan.

## 10. Write lifecycle

### 10.1 New session

1. Validate the project root and generate the existing safe session ID.
2. Create the store and segments directories.
3. Create the per-session lock path.
4. Build the first active segment containing the start event.
5. Write it to a same-directory temporary file.
6. Sync it and rename it to the active filename.
7. Build the manifest with the active range.
8. Write, sync, and atomically rename manifest.qtsq.
9. Update registry.json under the existing registry lock.
10. Return only after the manifest can be read and validated.

If manifest publication fails, the store is unpublished and can be repaired or
quarantined. The process must not report successful persistence.

### 10.2 Ordinary append

Hold the per-session lock only for storage work. Never hold it during network
requests, model inference, or terminal interaction.

1. Read and validate the manifest.
2. Load only the active segment.
3. Assign the next sequence.
4. Append the new event to the active in-memory batch.
5. Apply current event and note retention.
6. If limits remain satisfied:
   - serialize the new active segment;
   - write it to a temporary file;
   - sync and atomically replace the active path;
   - update the manifest in a temporary file;
   - sync and atomically replace the manifest.
7. Otherwise execute the rotation transaction.
8. Release the lock.

If active replacement succeeds but manifest replacement fails, the active file
is still a valid bounded snapshot. On the next open, recovery compares the
active content with the manifest and repairs forward when safe.

### 10.3 Rotation transaction

When a new event would exceed a soft limit:

1. Lock the session store.
2. Read and validate the current manifest and active segment.
3. Form the sealed batch from the current active events.
4. Form the next active batch containing the new event.
5. Write the sealed batch to a temporary filename.
6. Sync the sealed temporary file.
7. Write the next active batch to a temporary filename.
8. Sync the active temporary file.
9. Rename both to unique final filenames.
10. Write one manifest referencing the sealed and new active files.
11. Sync and atomically replace the manifest.
12. Delete the old active file only after manifest publication.
13. Release the lock.

The old manifest must remain usable until the new manifest is committed. Do not
rename away the only file referenced by the old manifest before the new
manifest has a complete replacement view.

### 10.4 Failed writes

A storage failure must:

- return a structured non-fatal error;
- leave the in-memory session usable where possible;
- not turn a successful model response into a failed turn;
- not claim an event is committed when validation failed;
- leave valid candidates available to repair;
- avoid exposing raw paths and errors as model instructions.

## 11. Read behavior

### 11.1 Store detection

For a validated session ID:

1. use <id>.store/manifest.qtsq if it validates as the main segmented format;
2. otherwise use legacy flat <id>.qtsq if present;
3. otherwise report missing data;
4. never classify desktop <id>/session.qtsq as the main segmented manifest.

A valid segmented store wins after migration. The old flat file remains intact
until an explicit retention decision removes it.

### 11.2 Metadata-only operations

Session listing should read registry.json and, when needed, only manifests. It
must not decode every segment just to list IDs, models, status, or ranges.

Manifest validation should check:

- safe session ID;
- safe relative segment paths;
- ordered ranges;
- event and byte bounds;
- file existence;
- optional payload hash.

Payload checksum/decode may be deferred until a segment is selected, but a
selected segment must be fully validated before returning events.

### 11.3 Recent context

Implement a recent-event reader that:

1. reads the manifest;
2. walks segments from newest to oldest;
3. decodes only enough segments to satisfy the requested event/token budget;
4. stops as soon as the budget is met;
5. returns events in ascending sequence order.

past_turns should walk backward until the existing ten prompt/answer turns are
available. session_context should read only the existing recent event window.

### 11.4 Peer polling

PeerWatch::poll should stop loading every peer's full session on every poll.
Instead:

- retain the last observed sequence per peer;
- read the peer manifest;
- skip peers whose next_seq has not advanced;
- load only the active/new sealed segments containing newer sequences;
- preserve the current notice cap and event-kind filtering;
- advance the observed sequence only after valid decoding;
- emit one diagnostic identity for a corrupt segment and do not skip past it.

This preserves near-real-time peer notices while eliminating repeated full
session decode work.

### 11.5 Compatibility full load

Keep load_session_data for callers that need a complete reconstructed
SessionData. Its segmented implementation may decode all retained segments.

Ordinary recent context and peer polling must not call the full-load function.
The reconstructed result must preserve metadata, task, notes, event order, event
kinds, text, and legacy behavior.

## 12. Crash recovery and repair

### 12.1 Recovery triggers

Run recovery:

- when a segmented manifest is missing or invalid;
- after a storage error;
- on explicit sofuu session repair;
- optionally during idle maintenance.

Do not scan every session on every ordinary turn.

### 12.2 Recovery scan

A repair operation is confined to the owning store:

1. validate every candidate filename;
2. ignore unsafe paths;
3. validate QTSQ and JSON for each candidate;
4. check session ID, ranges, event order, counts, and checksums;
5. classify files as published, active, orphaned, corrupt, or incomplete;
6. choose the longest valid contiguous sequence from retained_from_seq;
7. prefer a valid manifest-published range;
8. promote a valid orphan only when it extends that contiguous range;
9. never skip a sequence gap;
10. write a rebuilt manifest atomically;
11. quarantine corrupt files with deterministic names;
12. report repaired, orphaned, missing, and corrupt counts.

### 12.3 Crash matrix

| Interruption point | Required result |
| --- | --- |
| Before temporary file completes | old manifest and old active remain usable |
| After temporary sync, before rename | candidate is ignored or promoted only if fully valid |
| After sealed rename, before active rename | old manifest remains authoritative; sealed file is an orphan candidate |
| After active rename, before manifest rename | old view remains readable; both new files can be reconciled |
| After manifest rename | new sealed and active pair is authoritative |
| During cleanup | valid published files remain; cleanup retries later |

No repair path may create overlapping ranges, duplicate sequences, or silently
join events across an unrecoverable gap.

### 12.4 Corrupt or missing segments

If a required segment is corrupt or missing:

- stop at the first safe valid prefix;
- mark the session degraded;
- do not fabricate text;
- do not replay later segments out of order;
- return a clear diagnostic only when the caller requests that context;
- keep unrelated chat operations available.

## 13. Legacy and migration policy

### 13.1 Reader first

Implement segmented reading and validation before switching new writes. Legacy
flat files and older encrypted QTSQ records remain readable.

### 13.2 New sessions

After the writer rollout, new main sessions use <id>.store/ and do not create
new flat <id>.qtsq files.

### 13.3 Existing sessions

Migration is explicit or lazy per session, never a startup bulk rewrite:

1. acquire the relevant locks;
2. load and validate the legacy flat file;
3. convert retained events into bounded segments;
4. write a complete staged store;
5. validate every staged file and the staged manifest;
6. rename the staged directory to <id>.store/;
7. leave the legacy flat file untouched;
8. record source hash and migration time;
9. select the segmented store only after its manifest validates.

If migration fails, remove only the incomplete staging directory and continue
using the legacy flat file.

### 13.4 Rollback

A rollback must not rewrite or delete data:

- stop creating new segmented stores;
- continue reading valid segmented stores;
- use legacy files for sessions not yet migrated;
- retain both readers until a later release has a tested removal policy.

## 14. Main Sofuu code structure

Keep public session behavior in crates/sofuu-core/src/session.rs, but move
storage mechanics into a focused internal module, for example:

~~~text
crates/sofuu-core/src/session_store.rs
~~~

The module should own:

- store discovery and path validation;
- manifest and segment schemas;
- QTSQ load/save calls;
- atomic writes and syncing;
- per-session locking;
- active append and rotation;
- retention;
- recent/range reads;
- repair;
- migration.

The public Session methods should delegate to this module. Callers should not
know whether a session is legacy or segmented.

Expected storage operations:

- create;
- append_event;
- update_metadata;
- read_manifest;
- read_recent;
- read_since;
- read_range;
- read_all_retained;
- repair;
- migrate_legacy;
- remove.

Use the existing sessions .lock for registry read-modify-write and a
per-session .store/.lock for manifest/segment transactions. Lock creation must
not allow a forged session ID to escape the sessions directory.

## 15. Output archive boundary

The output archive remains specified in
PLAN-OUTPUT-ARCHIVE.md. The hybrid policy for it is:

~~~text
.sofuu/
  outputs/
    index.qtsq
    records/
      2026/08/31/
        <output-id>.qtsq
~~~

For the first output release:

- keep one file per logical output;
- write final, partial, error, recovery, summary, and material tool-result
  records;
- never write one file per token or stream delta;
- use date sharding and a bounded metadata index;
- use raw QTSQ below 500 KiB and compressed plaintext QTSQ at or above 500 KiB;
- retrieve only selected records;
- keep output failures non-fatal to the current chat turn.

Do not pack all outputs into a large file in the first release. Without a
Sofuu-level random-access record index, reading one output from a pack would
require decoding the whole pack. Add immutable output packs only after
measurements prove that date sharding and index rotation are insufficient.

## 16. Compaction

Rotation is the normal write path. Compaction is optional maintenance.

Only compact sealed segments:

- during an explicit maintenance command;
- during idle time;
- after segment-count or index-size thresholds are measured;
- never while a model request waits for the session lock.

Compaction transaction:

1. lock the store;
2. choose contiguous sealed segments;
3. fully validate them;
4. preserve all sequence numbers and timestamps;
5. write a replacement segment to a temporary file;
6. sync and publish it;
7. write a manifest referencing the replacement;
8. sync and publish the manifest;
9. delete old segments only after manifest commit.

Target replacement payload: at most 1 MiB in the first compactor. If it reaches
500 KiB, use the existing compressed plaintext path. Compaction must never change
logical events or retention boundaries.

## 17. Concurrency and filesystem requirements

### 17.1 Registry

Keep the current project registry lock for registry read-modify-write. Registry
writes remain bounded and atomic. A heartbeat must not hold the session lock.

### 17.2 Same-session writes

All writers for one main session use the per-session lock. A reader may be
lock-free because manifest and segment publication is atomic, but it must
validate the selected file.

### 17.3 Different sessions

Different session stores may append concurrently. They must not share a single
global segment lock.

### 17.4 Temporary files

Temporary names must:

- live in the target directory;
- include a unique process/counter suffix;
- never be selected by normal readers;
- be removable or repairable after a crash.

All published segment and manifest writes must flush/sync the file before
rename. Directory syncing should be used where supported after publication of a
new filename.

## 18. Privacy requirements

New session and output payloads are intentionally unencrypted per the user's
decision. Therefore:

- request owner-only permissions where supported;
- keep full event text out of manifests and indexes;
- keep secrets out of diagnostics;
- bound error summaries;
- retain current peer-text sanitization;
- label recovered historical data as untrusted before model injection;
- retain decryption only for legacy reads;
- never call the vault-encryption function for new segmented records.

Do not apply lossy redaction to canonical session text without a separate
product decision. Canonical history and diagnostic summaries must remain
distinguishable.

## 19. CLI and diagnostics

Add focused main-Sofuu maintenance commands:

~~~text
sofuu session inspect <id>
sofuu session repair <id>
sofuu session migrate <id>
sofuu session storage-stats <id>
~~~

They should report:

- legacy or segmented format;
- manifest schema;
- segment count and ranges;
- active event count and bytes;
- retained sequence range;
- raw and compressed record counts;
- legacy encrypted record count;
- missing, corrupt, orphaned, and quarantined counts;
- last migration/repair result;
- measured bytes written/read when metrics are enabled.

Do not print full prompt/answer text by default. Add explicit selected-event
display only when requested.

## 20. Test plan

### 20.1 Schema and path tests

- manifest and segment round-trip;
- required-field and discriminator validation;
- unknown-field compatibility;
- unsupported schema rejection;
- invalid session ID rejection;
- absolute/traversal path rejection;
- safe segment filename validation;
- wrong-session segment rejection;
- ordered/non-overlapping range validation.

### 20.2 QTSQ policy tests

- small manifest is raw and unencrypted;
- small segment is raw and unencrypted;
- payload one byte below 500 KiB does not compress;
- payload exactly 500 KiB uses the current compression path;
- new writes never call vault encryption;
- large writes are atomic at the Sofuu boundary;
- legacy encrypted records still load;
- checksum failure fails closed.

### 20.3 Append and rotation tests

- first event creates a valid manifest and active segment;
- append rewrites only the bounded active segment;
- exact soft-limit boundary rotates deterministically;
- hard limits are enforced;
- a single large event is not silently split;
- sequence numbers stay contiguous;
- clock rollback does not reorder events;
- no overlapping segment ranges;
- old active is retained until manifest commit;
- temporary files are ignored by normal reads.

### 20.4 Crash tests

Inject failure after:

- temporary write;
- temporary sync;
- sealed rename;
- active rename;
- manifest temporary write;
- manifest sync;
- manifest rename;
- cleanup.

Every restart/repair must yield the last committed data or a valid
durably-present extension. It must never yield a torn manifest, malformed QTSQ,
duplicate sequence, silent gap, or half-written event.

### 20.5 Read-path tests

- legacy flat sessions load unchanged;
- segmented sessions reconstruct equivalent SessionData;
- recent reads decode only required segments;
- peer polling returns each notice once;
- unchanged peers cause no segment decode;
- corruption stops at the safe prefix;
- missing optional data is non-fatal;
- desktop-shaped directories are not misclassified.

### 20.6 Migration and concurrency tests

- migration preserves order, notes, and metadata;
- migration is idempotent;
- migration failure leaves the legacy file untouched;
- staged stores are invisible until manifest validation;
- source hash is recorded;
- two sessions update the registry without lost entries;
- same-session writers serialize;
- different sessions append concurrently;
- cleanup cannot delete another process's newly published file.

### 20.7 Performance tests

Use synthetic sessions with 10, 64, 300, and 3,000 events, plus:

- repetitive text;
- mixed text;
- incompressible text;
- one event near 500 KiB.

Measure:

- append p50/p95;
- bytes written per event;
- QTSQ encode/decode time;
- file opens during recent reads;
- file opens during peer polling;
- resume latency;
- manifest size;
- segment sizes;
- file count and directory listing time.

Initial gates:

- ordinary append writes one bounded active segment plus one manifest, not the
  entire retained history;
- recent reads open only the manifest and required segments;
- unchanged peer polling decodes no segment payload;
- a 300-event session uses no more than six normal segment payload files;
- thresholds are changed only after target-machine measurements.

## 21. Implementation phases

### Phase 0 — Baseline

Read-only:

- instrument current Session::persist, load_session_data, past_turns, and
  PeerWatch::poll;
- record bytes written, files opened, decode count, and latency;
- create legacy fixtures;
- audit large-write atomicity;
- confirm main-only build/test scope.

Gate: repeatable baseline results exist.

### Phase 1 — Storage primitives

Add only:

- schemas;
- safe path/store discovery;
- atomic synced write helper;
- session lock;
- QTSQ bridge adapter;
- validators.

Gate: path, atomicity, and QTSQ policy tests pass.

### Phase 2 — Reader first

Add:

- segmented manifest/segment reader;
- recent/range readers;
- legacy fallback;
- corruption/degraded state.

Do not switch new writes.

Gate: all current legacy session tests pass.

### Phase 3 — New segmented writer

Use <id>.store/ only for newly created main sessions:

- active writes;
- rotation;
- manifest transactions;
- retention;
- unencrypted policy;
- large-write atomicity.

Gate: lifecycle, crash, QTSQ, and concurrency tests pass.

### Phase 4 — Incremental reads

Change past_turns, session_context, and PeerWatch::poll to use bounded/range
reads. Keep load_session_data for full reconstruction and compatibility.

Gate: behavior matches the old path and decode/open counts improve.

### Phase 5 — Explicit migration and repair

Add migrate, repair, inspect, and storage-stats commands. Preserve legacy files
and stage migrations atomically.

Gate: migration is idempotent and failure-safe.

### Phase 6 — Output archive

Implement the separate output archive plan with logical records, date sharding,
selective retrieval, and no token files.

Gate: output persistence is non-fatal and historical context is bounded.

### Phase 7 — Optional maintenance

Only after measurements, add compaction, manifest rotation, orphan cleanup, and
retention tuning.

Gate: maintenance has crash tests and preserves logical sequences.

## 22. Future implementation scope

Expected main-Sofuu scope:

- crates/sofuu-core/src/session.rs;
- a focused storage module under crates/sofuu-core/src/;
- crates/sofuu-ffi/src/qtsq.rs only if the large-write wrapper cannot remain
  outside the bridge;
- main Sofuu tests and fixtures;
- CLI help and storage documentation.

Explicitly excluded:

- sofuu-desktop/;
- src/js/chat.js;
- /Users/priyanshuboruah/projects/black-hole-disk;
- unrelated ML/model code;
- unrelated desktop changes.

Before any future edit, compare each target with the current working tree and
preserve existing user changes.

## 23. Acceptance checklist

### Correctness

- [ ] New main sessions use <id>.store/.
- [ ] Legacy flat sessions remain readable.
- [ ] Desktop directories are not misclassified.
- [ ] Every published segment has valid QTSQ and schema validation.
- [ ] Ranges are ordered, contiguous after retention, and non-overlapping.
- [ ] No event is duplicated or silently dropped.
- [ ] Existing 300-event and 20-note limits remain in release one.
- [ ] Existing session commands and peer behavior remain functional.

### Durability

- [ ] Manifest and segment writes use same-directory temporary files, sync, and
      atomic rename.
- [ ] Large QTSQ writes receive the same atomic guarantee.
- [ ] Old active files are retained until manifest commit.
- [ ] Every injected crash point is recoverable.
- [ ] Corrupt data fails closed and is repairable.

### Efficiency

- [ ] Normal append does not rewrite the full retained session.
- [ ] Recent context does not decode all segments.
- [ ] Unchanged peer polling does not decode segment payloads.
- [ ] No token/delta files are created.
- [ ] Thresholds are benchmarked on the target machine.
- [ ] Compression is used only at or above 500 KiB.

### QTSQ and privacy

- [ ] No QTSQ repository source was changed.
- [ ] New main records are unencrypted.
- [ ] Legacy encrypted records remain readable.
- [ ] Manifests contain metadata, not full text.
- [ ] File permissions and diagnostic handling are tested.

### Scope and release

- [ ] Main Sofuu builds/tests without building desktop.
- [ ] Existing dirty changes remain untouched.
- [ ] Migration is not a startup bulk rewrite.
- [ ] Rollback retains both readers.
- [ ] Output archive remains additive.

## 24. Defaults to review before implementation

Unless discussion changes them:

1. store path: <id>.store/;
2. manifest: manifest.qtsq;
3. one mutable active segment;
4. soft rotation: 64 events or 256 KiB;
5. hard active limit: 128 events or 1 MiB;
6. session retention: 300 events and 20 notes;
7. raw below 500 KiB; compressed plaintext at or above 500 KiB;
8. no encryption for new records;
9. legacy encrypted reads retained;
10. per-session lock plus existing registry lock;
11. reader-first rollout;
12. new sessions segmented, old sessions explicit migration;
13. one logical output file in the first output archive;
14. no desktop changes;
15. no QTSQ repository changes;
16. no automatic historical context injection.

## 25. Final recommendation

This design gives main Sofuu bounded active writes, immutable recoverable
history, selective reads, and a clear QTSQ boundary. It fixes the real main
runtime cost without replacing one problem with a giant file or a directory
containing one file for every event.

The core Phase 1–5 implementation is complete and has focused QTSQ, lifecycle,
rotation, migration, repair, and incremental-read coverage. Before shipping
the deferred output archive or optional maintenance phases, add target-machine
baseline measurements and fault-injection crash tests for every transaction
boundary. The namespace, 500 KiB threshold, transaction ordering, migration
policy, and output boundary remain the release contract.
