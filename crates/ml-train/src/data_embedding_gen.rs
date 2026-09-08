//! Procedural, seeded corpus generator for the semantic-embedder trainer.
//!
//! The curated groups in `data_embedding.rs` cover real Sofuu topics but are
//! far too small to generalize from — the rejected v1 candidate trained on 16
//! families and scored 0.417 recall@5 against hash-v1's 0.849 on the §10
//! acceptance corpus.  This module generates hundreds of distinct families
//! across the eight acceptance-category structures plus three everyday
//! categories, with randomized slot values drawn from large pools and
//! per-family template shuffles, so families are separable only by their
//! content words — never by surface form.
//!
//! Everything is seeded: the same corpus seed reproduces the dataset
//! byte-for-byte, which keeps retrained artifacts reproducible.

use std::collections::{HashMap, HashSet};

use crate::data_embedding;
use crate::train::Rng;

/// Training categories.  0..8 mirror the §10 acceptance categories by name
/// (failure mining maps eval misses onto these); 8..11 add everyday-domain
/// breadth so the projector sees more than work-shaped text.
pub const CAT_NAMES: [&str; 11] = [
    "paraphrase",
    "facts",
    "documentation",
    "code",
    "paths",
    "errors",
    "versions",
    "dates",
    "decisions",
    "preferences",
    "incidents",
];

/// Pseudo-category id for the curated Sofuu groups (no val split there).
pub const CURATED_CAT: usize = CAT_NAMES.len();

pub const BASE_FAMILIES: usize = 26;
pub const VAL_FAMILIES: usize = 8;

pub const CORPUS_SEED: u64 = 0xC0FF_EE20_2609_0401;

/// Optional corpus-seed override (SOFUU_EMB_CORPUS_SEED, hex or decimal) so
/// multi-corpus training runs can randomize the generated dataset itself.
pub fn corpus_seed_or_env() -> u64 {
    match std::env::var("SOFUU_EMB_CORPUS_SEED") {
        Ok(v) => u64::from_str_radix(v.trim_start_matches("0x").trim(), 16)
            .or_else(|_| v.trim().parse::<u64>().map(|x| x))
            .unwrap_or(CORPUS_SEED),
        Err(_) => CORPUS_SEED,
    }
}

#[derive(Clone)]
pub struct GenFamily {
    pub memories: Vec<String>,
    /// trained as anchors (empty for validation families)
    pub train_queries: Vec<String>,
    /// held-out probes; never trained
    pub val_queries: Vec<String>,
}

pub struct GenCat {
    pub train: Vec<GenFamily>,
    pub val: Vec<GenFamily>,
}

pub struct Corpus {
    pub cats: Vec<GenCat>,
    /// curated Sofuu-topic groups; every example is a train anchor
    pub curated: Vec<Vec<String>>,
    pub chatter: Vec<String>,
}

// ── slot pools ──────────────────────────────────────────────────────────

const PROJ_PRE: [&str; 16] = [
    "lumen", "vector", "pine", "cobalt", "ember", "quartz", "willow", "harbor", "ridge", "delta",
    "north", "storm", "glass", "fable", "onyx", "sable",
];
const PROJ_SUF: [&str; 16] = [
    "cache", "forge", "metrics", "relay", "vault", "stream", "ledger", "signal", "spool", "grid",
    "loom", "port", "stack", "keeper", "bridge", "path",
];

const TOOLS: [&str; 40] = [
    "rusqlite",
    "AES-GCM",
    "React",
    "Celery",
    "io_uring",
    "Redis",
    "Kafka",
    "Postgres",
    "gRPC",
    "protobuf",
    "nginx",
    "envoy",
    "Terraform",
    "Kubernetes",
    "Docker",
    "FFmpeg",
    "SQLite",
    "RocksDB",
    "ClickHouse",
    "Prometheus",
    "Grafana",
    "etcd",
    "MinIO",
    "NATS",
    "RabbitMQ",
    "Elasticsearch",
    "MongoDB",
    "Memcached",
    "Varnish",
    "HAProxy",
    "systemd",
    "ZeroMQ",
    "LMDB",
    "DuckDB",
    "Polars",
    "Arrow",
    "Tantivy",
    "wasmtime",
    "Brotli",
    "Zstd",
];

const PURPOSES: [&str; 36] = [
    "the session registry",
    "sector encryption",
    "the settings UI",
    "invoice scheduling",
    "the write path",
    "asset cache invalidation",
    "log shipping",
    "the auth boundary",
    "request tracing",
    "config hot-reload",
    "the job queue",
    "binary releases",
    "schema migrations",
    "rate limiting",
    "full-text search",
    "the telemetry sink",
    "feature flags",
    "backup rotation",
    "the upload pipeline",
    "session storage",
    "the token cache",
    "webhook signing",
    "error aggregation",
    "the CDN edge",
    "latency budgets",
    "secret rotation",
    "the audit trail",
    "quota enforcement",
    "email delivery",
    "the export pipeline",
    "index compaction",
    "connection pooling",
    "the deploy pipeline",
    "user avatars",
    "the media transcoder",
    "push notifications",
];

const FNAMES: [&str; 48] = [
    "align_chunks",
    "purge_stale",
    "split_cli",
    "with_backoff",
    "drop_dups",
    "throttle",
    "map_ranges",
    "fold_batches",
    "scan_tree",
    "prune_index",
    "pack_frames",
    "parse_header",
    "diff_rows",
    "shard_keys",
    "merge_runs",
    "bucket_by_day",
    "walk_deps",
    "carry_flags",
    "trim_edges",
    "stamp_meta",
    "spin_workers",
    "gate_traffic",
    "seed_cache",
    "drain_queue",
    "pool_sockets",
    "sort_bins",
    "clip_tail",
    "hash_slices",
    "batch_replies",
    "stage_upload",
    "tally_votes",
    "bridge_streams",
    "crop_paths",
    "round_stats",
    "lift_gates",
    "patch_refs",
    "sweep_tombstones",
    "coalesce_writes",
    "mirror_state",
    "step_clock",
    "load_table",
    "route_keys",
    "fade_logs",
    "brace_match",
    "roll_files",
    "seek_gap",
    "lean_pool",
    "trace_hops",
];

const FARGS: [&str; 16] = [
    "the corpus path",
    "the argv slice",
    "the cache root",
    "the bucket key",
    "the record list",
    "the open handle",
    "the frame buffer",
    "the config table",
    "the work queue",
    "the segment id",
    "the peer list",
    "the token stream",
    "the page map",
    "the epoch counter",
    "the source dir",
    "the output sink",
];

const FDESCS: [&str; 32] = [
    "splits documents into overlapping chunks and returns their offsets",
    "walks entries older than the TTL and unlinks them",
    "tokenizes quoted arguments and returns the flag table",
    "reruns a fallible call with growing delays",
    "removes near-identical rows keeping the newest",
    "token-buckets calls per key and sleeps on overflow",
    "maps byte ranges onto aligned blocks",
    "folds small batches into one write",
    "walks the tree depth-first and yields paths",
    "drops stale postings and compacts the index",
    "packs frames head-to-tail without padding",
    "reads the header and validates the magic",
    "compares rows and emits a minimal patch",
    "splits keys across shards by prefix",
    "merges adjacent runs and dedups by key",
    "groups events into daily buckets",
    "visits dependencies in topological order",
    "propagates flags through the call chain",
    "trims whitespace and boundary bytes",
    "stamps rows with a monotonic version",
    "starts a bounded worker pool",
    "gates traffic when the budget is spent",
    "warms the cache from a seed list",
    "drains the queue with backpressure",
    "keeps warm sockets and evicts idle ones",
    "bins values and returns medians",
    "cuts the tail past the quantile",
    "hashes slices into a rolling fingerprint",
    "batches replies until the window closes",
    "stages an upload with resume support",
    "counts votes and resolves ties",
    "splices two streams without blocking",
];

const FVERBS: [&str; 32] = [
    "chunk documents",
    "clear expired cache entries",
    "parse command line flags",
    "retry a flaky call",
    "remove duplicate rows",
    "apply rate limiting",
    "align block ranges",
    "batch small writes",
    "walk a directory tree",
    "prune a stale index",
    "pack binary frames",
    "validate a file header",
    "diff two row sets",
    "shard keys by prefix",
    "merge sorted runs",
    "bucket events by day",
    "order dependencies",
    "thread feature flags",
    "trim string edges",
    "version stamp rows",
    "pool worker threads",
    "throttle over-budget traffic",
    "warm a cold cache",
    "drain a work queue",
    "keep sockets warm",
    "take the median of bins",
    "clip heavy tails",
    "fingerprint byte slices",
    "batch outgoing replies",
    "resume large uploads",
    "resolve tied votes",
    "splice live streams",
];

const PATH_SUFFIX: [&str; 8] = [
    "src/core/config.rs",
    "src/net/pool.rs",
    "src/store/wal.rs",
    "src/ui/theme.rs",
    "src/codec/frame.rs",
    "src/cli/args.rs",
    "src/db/migrate.rs",
    "src/auth/token.rs",
];

const PATH_NOTES: [&str; 24] = [
    "parses the TOML once at boot",
    "mirrors the engine config into the WebView",
    "resolves the workspace root first",
    "validates the schema before scan",
    "guards the codec offset table",
    "only reads public/site.json",
    "caches the parsed table in a OnceLock",
    "edits flow through the Tauri commands",
    "paths are resolved relative to the workspace",
    "validation runs before any scanner starts",
    "offset 5016 is integrity-checked here",
    "the static export has no server config",
    "reloads on SIGHUP without a restart",
    "the loader fails closed on bad input",
    "defaults live next to the binary",
    "watch mode debounces writes",
    "tokens are read from the keyring first",
    "the migration gate runs before open",
    "every field has a schema doc string",
    "env overrides beat file values",
    "the table is frozen after boot",
    "imports merge into the same namespace",
    "rotation happens on size, not time",
    "the fallback chain ends in defaults",
];

const ERR_COMP: [&str; 32] = [
    "scheduler",
    "indexer",
    "gateway",
    "replicator",
    "wal-writer",
    "compactor",
    "authorizer",
    "broker",
    "ingestor",
    "planner",
    "watcher",
    "migrator",
    "resolver",
    "archiver",
    "emitter",
    "scorer",
    "tracker",
    "cleaner",
    "loader",
    "router",
    "packer",
    "sampler",
    "limiter",
    "healer",
    "catcher",
    "stacker",
    "merger",
    "seeker",
    "builder",
    "sender",
    "keeper",
    "pusher",
];

const ERR_MSG: [&str; 32] = [
    "worker pool drained while a batch was in flight",
    "mmap window grew past the address space limit",
    "upstream closed the stream before the trailer",
    "sequence gap detected between mirrors",
    "segment header checksum did not match",
    "snapshot pin count went negative",
    "lease expired mid-renewal",
    "the queue exceeded the high watermark",
    "handshake timed out after three attempts",
    "clock skew exceeded the lease window",
    "the schema hash changed under a live reader",
    "the quota ledger went inconsistent",
    "a duplicate commit raced the compaction",
    "the retry budget was spent before success",
    "the feature gate referenced a missing flag",
    "the tombstone outlived its segment",
    "the warm pool never refilled",
    "the fingerprint column overflowed",
    "the shard map pointed at a retired node",
    "the cookie jar expired mid-session",
    "the drain never reached quiescence",
    "the writer fell behind the WAL tail",
    "the ring buffer wrapped before the flush",
    "the sentinel file was missing at boot",
    "the probe timeout beat the slow start",
    "the generator skipped a sequence value",
    "the loader hit a truncated dictionary",
    "the queue depth violated the SLO",
    "the estimator drifted past the band",
    "the migration stalled at step three",
    "the reaper deleted a pinned page",
    "the proxy looped a redirect twice",
];

const ERR_FIX: [&str; 32] = [
    "cap concurrent batches at two",
    "shard the index by prefix",
    "enable TCP keepalive on the proxy",
    "resync from the primary snapshot",
    "truncate to the last good frame",
    "rebuild the pin table at open",
    "shorten the lease and renew early",
    "drain to the low watermark first",
    "raise the handshake deadline to 5s",
    "step the clock from the leader",
    "pin the schema hash at open",
    "rebuild the ledger from the journal",
    "serialize commits through the gate",
    "reserve budget for one final try",
    "bake the flag list into the binary",
    "cap tombstone lifetime to the segment",
    "pre-warm the pool at boot",
    "widen the column and migrate rows",
    "retire the node from the shard map",
    "refresh cookies before long jobs",
    "wait for quiescence before handoff",
    "fsync the tail before advancing",
    "flush at half the ring size",
    "recreate the sentinel on start",
    "extend slow start by one window",
    "reseed the generator from the journal",
    "rebuild the dictionary index",
    "rebalance the queue depth",
    "recenter the estimator each minute",
    "resume from the checkpoint",
    "honor pins during reaping",
    "cap redirects at one hop",
];

const LIBS: [&str; 14] = [
    "tokio", "serde", "reqwest", "hyper", "axum", "sqlx", "clap", "tracing", "regex", "libc",
    "curl", "zlib", "rusqlite", "notify",
];

const VERS: [&str; 28] = [
    "1.35.0", "1.38.1", "1.40.2", "1.0.210", "1.0.203", "1.0.219", "0.7.2", "0.9.1", "2.4.1",
    "2.6.0", "3.1.2", "1.79.0", "1.81.0", "4.5.1", "5.0.4", "0.11.3", "1.4.2", "2.2.7", "3.0.9",
    "0.6.5", "1.12.0", "2.9.3", "1.7.8", "0.8.6", "1.2.4", "2.1.5", "3.3.0", "1.6.1",
];

const TEAMS: [&str; 16] = [
    "platform", "runtime", "billing", "infra", "mobile", "security", "tools", "data", "web", "api",
    "qa", "docs", "sre", "design", "ml", "devrel",
];

const THINGS: [&str; 72] = [
    "the mesh sync beta",
    "the RLM sandbox GA",
    "the invoice export",
    "the colo migration",
    "the offline mode",
    "the audit remediation",
    "the keyring migration",
    "the tool-cache rewrite",
    "the ledger backfill",
    "the DNS cutover",
    "the sync protocol freeze",
    "the dependency sweep",
    "the settings redesign",
    "the search rollout",
    "the quota overhaul",
    "the batch runner",
    "the webhook replay",
    "the cache layer rewrite",
    "the onboarding flow",
    "the export scheduler",
    "the index rebuild",
    "the alert pipeline",
    "the billing cutover",
    "the proxy upgrade",
    "the session GC",
    "the snapshot format bump",
    "the test sharder",
    "the docs migration",
    "the model picker refresh",
    "the retention policy",
    "the perf harness",
    "the replica swap",
    "the schema registry",
    "the upload resume",
    "the CLI theming",
    "the handoff export",
    "the search reindex",
    "the billing hedge",
    "the key rotation",
    "the proxy failover",
    "the queue split",
    "the cache pre-warm",
    "the alert dedup",
    "the quota dashboard",
    "the export digest",
    "the session replay",
    "the rollout tracker",
    "the coverage push",
    "the flake budget",
    "the docs refresh",
    "the API freeze",
    "the schema annex",
    "the index shrink",
    "the WAL archive",
    "the snapshot prune",
    "the token scope split",
    "the client refresh",
    "the desktop theming",
    "the CLI plugins",
    "the brain export",
    "the mesh quorum",
    "the relay upgrade",
    "the webhook ledger",
    "the invoice PDF",
    "the refund runner",
    "the dunning emails",
    "the audit bundle",
    "the cost allocator",
    "the usage meter",
    "the perf dashboard",
    "the oncall handbook",
    "the rollback rehearsal",
];

const DATES: [&str; 28] = [
    "October 3",
    "October 7",
    "October 10",
    "October 14",
    "October 17",
    "October 21",
    "October 24",
    "October 28",
    "October 30",
    "November 3",
    "November 4",
    "November 7",
    "November 11",
    "November 14",
    "November 18",
    "November 21",
    "November 25",
    "November 28",
    "December 2",
    "December 5",
    "December 9",
    "December 12",
    "December 16",
    "December 19",
    "December 23",
    "January 6",
    "January 9",
    "January 13",
];

const DEPS: [&str; 20] = [
    "the keyring migration",
    "the tool-cache rewrite",
    "the ledger backfill",
    "the DNS cutover",
    "the schema freeze",
    "the proxy upgrade",
    "the token refresh",
    "the replica swap",
    "the quota rework",
    "the audit sign-off",
    "the design review",
    "the perf sign-off",
    "the data backfill",
    "the client update",
    "the docs pass",
    "the beta feedback",
    "the load test",
    "the security review",
    "the cost review",
    "the rollback drill",
];

const RETRIES: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

const SUBJ: [&str; 140] = [
    "the ingest pipeline",
    "the billing webhook",
    "the session store",
    "the login flow",
    "the deploy checklist",
    "the release train",
    "the export scheduler",
    "the search index",
    "the token cache",
    "the proxy pool",
    "the quota ledger",
    "the audit log",
    "the retry queue",
    "the schema registry",
    "the snapshot writer",
    "the keyring sync",
    "the upload path",
    "the alert router",
    "the batch runner",
    "the CLI shell",
    "the settings cache",
    "the mesh peer",
    "the WAL writer",
    "the compactor",
    "the test sharder",
    "the docs build",
    "the model picker",
    "the retention job",
    "the perf bench",
    "the replica set",
    "the feature flags",
    "the cost report",
    "the webhook gateway",
    "the cron scheduler",
    "the metrics agent",
    "the backup runner",
    "the log tailer",
    "the rate limiter",
    "the idempotency cache",
    "the deploy gate",
    "the smoke suite",
    "the ledger writer",
    "the sync engine",
    "the replica ladder",
    "the shard mapper",
    "the lease manager",
    "the queue drain",
    "the export viewer",
    "the diff renderer",
    "the alias table",
    "the route table",
    "the probe runner",
    "the cache warmer",
    "the pool sizing",
    "the quota meter",
    "the billing reconciler",
    "the invoice pricer",
    "the refund flow",
    "the dunning pass",
    "the signup funnel",
    "the invite flow",
    "the session GC",
    "the token refresh",
    "the device registry",
    "the presence service",
    "the mail renderer",
    "the push fanout",
    "the webhook signer",
    "the audit viewer",
    "the retention sweeper",
    "the perf tracker",
    "the flake hunter",
    "the config loader",
    "the health prober",
    "the traffic shaper",
    "the schema linter",
    "the secret vault",
    "the disk watermark",
    "the gc tuner",
    "the edge router",
    "the fan-out writer",
    "the dead-letter bin",
    "the backfill job",
    "the cold storage tier",
    "the warm standby",
    "the read replica",
    "the write buffer",
    "the spill file",
    "the sort merge",
    "the bloom filter",
    "the compaction gate",
    "the tombstone sweep",
    "the catalog cache",
    "the plan analyzer",
    "the slow query log",
    "the trace sampler",
    "the span exporter",
    "the metric scraper",
    "the alert silencer",
    "the on-call rota",
    "the status page",
    "the incident bot",
    "the changelog linter",
    "the license scan",
    "the SBOM builder",
    "the image builder",
    "the layer cache",
    "the registry mirror",
    "the pull-through proxy",
    "the admission check",
    "the node drainer",
    "the pod evictor",
    "the cordon step",
    "the rollout brake",
    "the canary analysis",
    "the fault injector",
    "the load profile",
    "the soak runner",
    "the bench harness",
    "the fuzz target",
    "the seed corpus",
    "the repro script",
    "the minidump parser",
    "the symbol server",
    "the flame graph",
    "the alloc tracker",
    "the page cache",
    "the io scheduler",
    "the ring buffer",
    "the epoch clock",
    "the lease clock",
    "the heartbeat pinger",
    "the peer gossip",
    "the anti-entropy pass",
    "the merkle tree",
    "the snapshot log",
    "the redo player",
    "the undo cleaner",
    "the vacuum pass",
    "the freeze frame",
];

const NUMS: [&str; 16] = [
    "three",
    "five",
    "seven",
    "ten",
    "twelve",
    "sixteen",
    "twenty",
    "thirty",
    "forty-five",
    "sixty",
    "one hundred",
    "two hundred",
    "500",
    "1,000",
    "2,000",
    "4,096",
];

const DECISIONS: [&str; 32] = [
    "adopted an append-only log",
    "moved releases to Tuesdays",
    "standardized on int8 artifacts",
    "split the monolith by domain",
    "froze the public API",
    "switched to quarterly audits",
    "banned silent fallbacks",
    "adopted trunk-based merges",
    "moved backups to nightly",
    "required two reviewers",
    "capped sessions at seven",
    "pinned docs to the release",
    "gated deploys on smoke tests",
    "adopted error budgets",
    "moved config to code",
    "banned broad exceptions",
    "required rollback drills",
    "standardized on UTF-8 logs",
    "moved metrics to push",
    "capped prompts at 8 KiB",
    "batched small writes",
    "required signed artifacts",
    "versioned every export",
    "shadow-tested new models",
    "auto-retried idempotent jobs",
    "paired on-call with sre",
    "rotated keys monthly",
    "kept staging data synthetic",
    "reviewed all shell tools",
    "tracked latency in-app",
    "pinned minor versions",
    "deleted stale branches weekly",
];

const REASONS: [&str; 32] = [
    "it makes history replayable",
    "it keeps the week predictable",
    "it keeps artifacts small",
    "it keeps teams unblocked",
    "it avoids breaking clients",
    "it keeps findings fresh",
    "it makes failures loud",
    "it keeps reviews fast",
    "it bounds data loss",
    "it keeps quality up",
    "it keeps memory bounded",
    "it ships docs with code",
    "it catches bad builds early",
    "it keeps toil honest",
    "it survives audits",
    "it keeps errors actionable",
    "it keeps deploys safe",
    "it keeps logs searchable",
    "it keeps dashboards live",
    "it protects long contexts",
    "it smooths disk load",
    "it blocks tampering",
    "it keeps exports stable",
    "it de-risks rollouts",
    "it avoids duplicate work",
    "it shares the load",
    "it limits blast radius",
    "it protects user data",
    "it keeps tools reviewable",
    "it keeps regressions visible",
    "it avoids surprise majors",
    "it keeps the tree clean",
];

const PREFS: [&str; 24] = [
    "prefers dark mode",
    "wants compact tables",
    "likes daily digests",
    "avoids weekend pings",
    "prefers text over calls",
    "wants raw diffs",
    "likes short summaries",
    "needs screen-reader labels",
    "prefers keyboard-first flows",
    "wants export warnings",
    "dislikes autoplay",
    "prefers UTC timestamps",
    "wants inline previews",
    "likes strict lints",
    "prefers staged rollouts",
    "wants quiet hours",
    "needs offline sync",
    "prefers plain-text mail",
    "wants confirm-on-delete",
    "likes auto-save",
    "prefers large fonts",
    "wants host key checks",
    "likes weekly reports",
    "prefers manual deploys",
];

const VALUES: [&str; 24] = [
    "at least on this laptop",
    "since the retina upgrade",
    "after the reorg",
    "for client work",
    "on the shared box",
    "during on-call",
    "in the staging account",
    "when traveling",
    "on the demo cluster",
    "for the archive project",
    "after the incident",
    "with the new team",
    "on the train",
    "for long-running jobs",
    "in the docs repo",
    "during freezes",
    "on the kiosk machine",
    "for external reviews",
    "on the build agent",
    "in the sandbox",
    "since the migration",
    "for pair sessions",
    "on the VM",
    "when presenting",
];

const SEVS: [&str; 5] = ["sev1", "sev2", "sev3", "sev4", "sev5"];

const ACTIONS: [&str; 24] = [
    "paged the on-call",
    "rolled back the deploy",
    "drained the queue",
    "failed over to the replica",
    "throttled new signups",
    "bumped the timeout",
    "added a circuit breaker",
    "replayed from the journal",
    "evicted the cache",
    "raised the rate limit",
    "isolated the bad shard",
    "froze the schema",
    "pinned the old image",
    "scaled the pool",
    "switched the provider",
    "opened a status page",
    "paused the cron",
    "rebuilt the index",
    "rotated the credentials",
    "quarantined the topic",
    "resumed with backpressure",
    "warm-restarted the pool",
    "switched to the secondary",
    "held the release",
];

const DAYS: [&str; 14] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "last night",
    "this morning",
    "yesterday",
    "during the freeze",
    "ahead of the launch",
    "after the deploy",
    "mid-migration",
];

const CHAT_A: [&str; 10] = [
    "the coffee machine",
    "my desk chair",
    "the office printer",
    "the neighbor's cat",
    "the elevator",
    "the parking lot",
    "the weather app",
    "the meeting room TV",
    "the kettle",
    "the bike lane",
];

const CHAT_B: [&str; 10] = [
    "is broken again",
    "looks great today",
    "needs a cleanup",
    "made a weird noise",
    "was booked twice",
    "ran out of sugar",
    "lost the connection",
    "won the bracket",
    "drips when it boils",
    "floods when it rains",
];

// ── category templates (3-of-5 memories, 4-of-6 queries per family) ─────

const FACTS_MEM: [&str; 5] = [
    "In {p} we use {t} for {u}; that call was made early and written into the log.",
    "Decision record — {p}: {t} handles {u}. One tool per job, no duplicates.",
    "{u} inside {p} runs on {t} (settled with the team, see the notes).",
    "For {u}, {p} standardized on {t} after the evaluation round.",
    "{p} fact sheet: {t} is the tool behind {u}.",
];
const FACTS_Q: [&str; 6] = [
    "what does {p} use for {u}?",
    "{p}: which tool covers {u}?",
    "is {t} the tool for {u} in {p}?",
    "which component handles {u} in {p}?",
    "who owns {u} in {p} — is it {t}?",
    "{p} {u} — what backs it?",
];

const DOC_MEM: [&str; 5] = [
    "{l} retry guide: the client retries a failed request up to {n} times before the error reaches the caller.",
    "Working with {l}: attempts are capped at {n}; each retry waits longer than the last.",
    "{l} operational notes — after {n} failed attempts the client surfaces the error instead of retrying.",
    "{l} config: retry limit {n}, backoff exponential, errors surface to the caller.",
    "The {l} client gives up after {n} attempts; whatever happened, the caller sees the real error.",
];
const DOC_Q: [&str; 6] = [
    "how many times does {l} retry a failed request?",
    "does {l} back off between retries?",
    "what happens when {l} exhausts its attempts?",
    "{l} maximum attempts",
    "when does {l} stop retrying?",
    "who surfaces the error from {l} after retries fail?",
];

const CODE_MEM: [&str; 5] = [
    "Snippet from {p}: {f}({a}) — {d}. Call it after the handle is open.",
    "In {p} the helper {f} {d}; example: {f}({a});",
    "{p} utility: {f} takes {a} and {d}.",
    "{f} in {p} is the entry point for {w}; pass {a}.",
    "From the {p} sources: {f}({a}) {d}.",
];
const CODE_Q: [&str; 6] = [
    "how do I {w} in {p}?",
    "{p} helper for {w}",
    "which function handles {w} in {p}?",
    "show me the {w} call in {p}",
    "what does {f} do in {p}?",
    "does {p} use {f} for {w}?",
];

const PATHS_MEM: [&str; 5] = [
    "{p}/{s} — {n1}.",
    "The {stem} code lives at {p}/{s}; {n2}.",
    "{n3}: edit {p}/{s} and rebuild.",
    "In {p} you will find it under {s}; {n1}.",
    "{p} keeps {stem} logic in {s}, where {n2}.",
];
const PATHS_Q: [&str; 6] = [
    "where does {p} keep its {syn} logic?",
    "{p} {stem} file location",
    "which file loads {syn} in {p}?",
    "edit location for {stem} in {p}",
    "where is {p}'s {stem} code?",
    "{p}: path to the {stem} module",
];

/* Role synonyms: the acceptance corpus asks for "settings" when the memory
 * says config.rs — the query must associate the file's ROLE, not echo its
 * name.  One synonym pair per path suffix. */
const PATHS_SYN: [&str; 8] = [
    "settings",
    "sockets",
    "journal",
    "appearance",
    "packets",
    "flags",
    "upgrades",
    "credentials",
];

const ERR_MEM: [&str; 5] = [
    "{comp} crashed with error {c}: {m}. Fix: {f}.",
    "Known issue {c} in {comp} — {m}. Workaround: {f}.",
    "Error {c} ({comp}): {m}. We {f} until the patch lands.",
    "{c} fired again in {comp}: {m}. Runbook says: {f}.",
    "Postmortem note — {comp} raised {c} ({m}); the fix was to {f}.",
];
const ERR_Q: [&str; 6] = [
    "what is error {c} in {comp}?",
    "{comp} failing with {c}",
    "how do I fix {c}?",
    "meaning of error {c}",
    "{c} keeps happening — what now?",
    "who owns {comp} error {c}?",
];

const VER_MEM: [&str; 5] = [
    "{p} pins {l} to v{v}. Do not bump without the compat matrix.",
    "{l} v{v} in {p} — locked until the next major audit.",
    "Upgrade rule: {p} stays on {l} v{v}.",
    "{p} depends on {l} {v}; treat it as pinned.",
    "Version note — {l} {v} for {p}, no exceptions.",
];
const VER_Q: [&str; 6] = [
    "which {l} version does {p} use?",
    "{p} {l} pin",
    "is {l} bumped in {p}?",
    "what {l} release is {p} on?",
    "can I upgrade {l} in {p}?",
    "{p} dependency: {l} version?",
];

const DATE_MEM: [&str; 5] = [
    "{t} ships {thing} on {d}. Blocked if {dep} slips.",
    "Release plan: {thing} lands {d} ({t}).",
    "{thing} — target {d}, owner {t}, depends on {dep}.",
    "Calendar note: {t} wants {thing} done by {d}; {dep} is the risk.",
    "{t} committed to {thing} for {d}, assuming {dep} holds.",
];
const DATE_Q: [&str; 6] = [
    "when does {t} ship {thing}?",
    "{thing} release date",
    "what is the target date for {thing}?",
    "is {thing} scheduled before the audit?",
    "who ships {thing}, and when?",
    "{thing} — what blocks it?",
];

const PARA_MEM: [&str; 3] = [
    "We decided {subj} {mech}; {det} — re-check in {num} days.",
    "Decision from the {subj} review: {mech}. Context: {det}.",
    "{subj} {mech} now. Reason: {det}. Owner review in {num} weeks.",
];
const PARA_Q: [&str; 4] = [
    "{q_mech} — what does {subj} do here?",
    "what did we decide about {subj}?",
    "for {subj}: {q_det}?",
    "{subj}: {q_mech}?",
];

/* Mechanism-only families: memories state a mechanism in one vocabulary,
 * the query asks for it in another, and NOTHING else is shared — no
 * subject, no named entity.  This is the hardest paraphrase skill in the
 * acceptance corpus: retrieval must work purely through meaning of the
 * mechanism phrase.  Duplicates are prevented by the mech slot key. */
const MECHONLY_MEM: [&str; 3] = [
    "Standing policy: we {mech}, because {det}.",
    "How it works here: we {mech}. The reason was {det}.",
    "Note for the team — by default we {mech}; {det}.",
];
const MECHONLY_Q: [&str; 3] = [
    "{q_mech}?",
    "quick check: {q_mech}?",
    "remind me — {q_mech} here?",
];

/* Paraphrase pairs: the memory states the mechanism in one vocabulary, the
 * query asks for it in another — the matching signal is the shared subject
 * plus meaning, not echoed tokens (mirrors the acceptance corpus, whose
 * queries rephrase rather than quote the memories). */
const PARA_MECH: [(&str, &str); 59] = [
    ("compresses every container with qtc before writing to disk", "how files get squeezed before storage"),
    ("retries failed deliveries five times with exponential backoff", "how many attempts a failing delivery gets"),
    ("keeps four warm sockets and evicts idle ones after thirty seconds", "how the connection pool is sized"),
    ("caps memory recall at two thousand tokens per turn", "how much remembered context a prompt may carry"),
    ("pins the library version until the next compat audit", "when the dependency version may change"),
    ("routes every write through the WAL before the page cache", "in what order writes reach disk"),
    ("signs each webhook with the rotating key", "how deliveries are authenticated"),
    ("shards the index by key prefix", "how the index is split up"),
    ("drains the queue before shutting down", "what happens to pending work at shutdown"),
    ("evicts cache entries oldest-first under pressure", "which entries go first when the cache is full"),
    ("batches small writes into one flush per second", "how tiny writes are grouped"),
    ("expires sessions after seven idle days", "when unused sessions die"),
    ("mirrors state to the replica every five seconds", "how often the replica catches up"),
    ("requires two reviewers for every release", "how many sign-offs a release needs"),
    ("encrypts sectors with AES-GCM keys from the vault", "how data at rest is protected"),
    ("buckets events into one-minute windows", "how events are grouped in time"),
    ("fails closed when the config hash mismatches", "what happens on a bad config"),
    ("warms the pool from the journal at boot", "how the cache fills after a restart"),
    ("stamps each row with a monotonic version", "how rows are ordered"),
    ("throttles tenants that exceed their quota", "what happens when a tenant overruns"),
    ("reserves ten percent of the disk as a safety watermark", "how much free space is kept back"),
    ("drops the oldest shard once the index passes twelve segments", "what happens when segments pile up"),
    ("coalesces adjacent deletes into one tombstone pass", "when the delete markers get merged"),
    ("recomputes the digest only when the file size changes", "what triggers a hash recompute"),
    ("prefetches the next page while the current one renders", "what loads ahead of the user"),
    ("renews the lease a minute before it lapses", "when the lock gets extended"),
    ("compresses cold rows with a stronger, slower codec", "how rarely-read data is stored"),
    ("routes reads to the nearest healthy replica", "where lookups are served from"),
    ("defers index maintenance to the nightly window", "when search structures get rebuilt"),
    ("counts a request as failed only after three consecutive timeouts", "when a call counts as down"),
    ("snapshots memory before every destructive migration", "what protects against a bad schema change"),
    ("spills oversized aggregates to a temp file on disk", "what happens when a rollup is too big"),
    ("redacts secrets before any log line leaves the process", "how credentials stay out of the logs"),
    ("stagger restarts across the fleet one node at a time", "how upgrades avoid taking everything down"),
    ("reconciles divergent replicas with an anti-entropy pass", "how copies that drifted get fixed"),
    ("increments a per-object epoch on every mutation", "how stale writes are detected"),
    ("blocks a deploy if the error budget is spent", "what gates a release when reliability is low"),
    ("weights the hash ring by each node's capacity", "how keys are spread across machines"),
    ("collapses duplicate alerts within a two-minute window", "how notification storms are tamed"),
    ("propagates deletes via a background sweep, not the write path", "when removals actually happen"),
    ("admits new connections slowly while the pool warms", "how the system behaves right after a restart"),
    ("pins each tenant to one writer to keep ordering", "how concurrent updates to one tenant are handled"),
    ("backs off p95-gated traffic shaping before shedding requests", "what happens first when latency climbs"),
    ("materializes the expensive view nightly instead of per query", "how the costly report is served"),
    ("validates every payload against the frozen schema", "what checks incoming data"),
    ("buffers acknowledgements until a batch of one kilobyte fills", "when confirmations get sent"),
    ("annotates each trace with the deploy id that produced it", "how a trace is tied to a release"),
    ("replays the journal forward on crash recovery", "how the store heals after a crash"),
    ("caps any single tenant at forty percent of total capacity", "how much of the system one tenant can eat"),
    ("determines the winner by vector clock, not wall time", "how conflicting writes are resolved"),
    ("rewrites heavy pages in place during compaction", "what happens to big records at merge time"),
    ("publishes the manifest only after every shard acknowledges", "when a release becomes visible"),
    ("sweeps orphaned blobs the morning after expiry", "when unreferenced storage is reclaimed"),
    ("enforces the memory ceiling with an allocator hook, not the OS", "how runaway allocations are stopped"),
    ("collects only every fiftieth trace to bound overhead", "how much of the traffic is observed"),
    ("re-encrypts each archive under a fresh data key every quarter", "when stored data is re-keyed"),
    ("collapses the plan cache when a new schema lands", "what invalidates cached queries"),
    ("fences old primaries with an epoch token before promoting", "how split-brain is prevented"),
    ("smuggles priority work past the queue via a side lane", "how urgent tasks jump the line"),
];

const PARA_DET: [(&str, &str); 61] = [
    ("it smooths disk load on the shared volume", "is protecting the shared volume the goal"),
    ("the provider flakes under load", "provider flakiness is involved"),
    ("old builds must still read the containers", "old builds still matter here"),
    ("the pool starved during the morning peak", "the morning peak played a part"),
    ("surprise majors broke clients last quarter", "past upgrades broke clients"),
    ("the audit needs reproducible history", "reproducibility is required"),
    ("duplicate sends angered customers", "customers were affected"),
    ("cold starts hurt the p99", "latency is the concern"),
    ("backups grew past the retention window", "storage growth is the reason"),
    ("pin fights corrupted two snapshots", "snapshots were involved"),
    ("retries were invisible in the logs", "observability is the issue"),
    ("the queue starved small tenants", "fairness is the concern"),
    ("the schema drifted between services", "schema drift matters here"),
    ("thundering herds took down the upstream", "the upstream needs protection"),
    ("memory grew flat-out during the soak", "memory growth was observed"),
    ("the tail latency doubled under load", "tail latency is the driver"),
    ("the auditor asked for explicit choices", "compliance drove it"),
    ("restarts used to lose warm state", "restarts matter"),
    ("the journal replays out of order", "ordering is not guaranteed"),
    ("the demo cluster shares the quota", "a shared environment is involved"),
    ("the incident review demanded a kill switch", "an outage review shaped it"),
    ("a regulatory freeze lands next spring", "an upcoming regulation matters"),
    ("the on-call was drowning in pages", "alert fatigue drove it"),
    ("the crawler hammered us every midnight", "an external client caused it"),
    ("the embedded build has no allocator", "a constrained target shaped it"),
    ("two datacenters must see identical writes", "cross-region consistency is the reason"),
    ("the pagers fired twice for the same fault", "duplicate notifications were the pain"),
    ("a botched rollback left half the fleet stale", "a past rollback failure shaped it"),
    ("the customer contract promises single-digit recovery", "an SLA is behind it"),
    ("the old client still speaks version one", "legacy protocol support matters"),
    ("finance flagged the runaway storage bill", "cost pressure is the driver"),
    ("the soak test found a slow leak", "a leak was observed"),
    ("the security scan flagged unsigned artifacts", "a security finding drove it"),
    ("the field team needed offline installs", "offline operation is required"),
    ("the upstream bug zeroed out our counters", "a vendor defect played a part"),
    ("a firmware clock skew broke ordering", "clock skew was involved"),
    ("the demo must never show an empty screen", "a visible failure drove it"),
    ("the legal review requires seven-year records", "record-keeping law is the reason"),
    ("the merge freeze lands every December", "a seasonal freeze matters"),
    ("one bad deploy took the whole region out", "blast radius is the concern"),
    ("the migration tool cannot stream writes", "a tooling limit shaped it"),
    ("support kept reopening the same ticket", "repeat support load drove it"),
    ("the partner feed arrives unordered", "unordered input is the norm"),
    ("the first attempt stalled mid-flight", "a stalled run was observed"),
    ("the threat model assumes a hostile tenant", "hostile tenants are assumed"),
    ("the board asked for cost per query", "reportability drove it"),
    ("the old format broke past ten million rows", "a scale limit was hit"),
    ("the regression slipped through with no test", "a test gap shaped it"),
    ("the on-call runbook assumed manual steps", "runbook simplification is the goal"),
    ("the mobile team asked for smaller payloads", "bandwidth on mobile is the reason"),
    ("the cache poisoned itself on a bad warm", "a past cache bug drove it"),
    ("the partner contract forbids data egress", "a contractual limit is behind it"),
    ("the hardware dedupes our writes badly", "hardware behavior is involved"),
    ("the compliance audit samples raw traffic", "audit requirements drove it"),
    ("the failure only shows under weekend load", "weekend traffic patterns matter"),
    ("the vendor patch fixed the wrong layer", "a vendor patch was involved"),
    ("the previous attempt churned the whole cache", "past churn was the pain"),
    ("the storm test melted the old limits", "a stress test drove it"),
    ("a hot loop burned a whole core", "cpu cost was the pain"),
    ("the zero-downtime promise is contractual", "uptime is contractually required"),
    ("the debug build hid it for months", "a slow discovery shaped the fix"),
];

const DEC_MEM: [&str; 5] = [
    "{p} {dec} because {r}.",
    "Team decision — {p}: {dec}; the reason was {r}.",
    "{p} policy: {dec} ({r}).",
    "After the review, {p} {dec} — {r}.",
    "Standing rule in {p}: {dec}, since {r}.",
];
const DEC_Q: [&str; 6] = [
    "why does {p} {dec}?",
    "{p} policy on this",
    "what did {p} decide?",
    "who decided {p} should {dec}?",
    "{p}: is {dec} still the rule?",
    "reason behind the {p} decision",
];

const PREF_MEM: [&str; 5] = [
    "{u} {pref} — {v}.",
    "Remember: {u} {pref} ({v}).",
    "User note — {u}: {pref}, {v}.",
    "{u} told us they {pref}; applies {v}.",
    "Profile: {u} {pref}. Context: {v}.",
];
const PREF_Q: [&str; 6] = [
    "what does {u} prefer?",
    "{u}'s preferences",
    "does {u} {pref}?",
    "remember anything about {u}?",
    "how should we treat {u}?",
    "{u}: what do they like?",
];

const INC_MEM: [&str; 5] = [
    "Incident on {s} ({sev}) {day}: we {a}.",
    "{s} went down {day} ({sev}); response was to {a}.",
    "{day}'s {sev} on {s}: {a}.",
    "Postmortem — {s}, {sev}, {day}: first move was to {a}.",
    "Timeline: {day}, {s} {sev}. Action taken: {a}.",
];
const INC_Q: [&str; 6] = [
    "what happened to {s} {day}?",
    "{s} incident details",
    "how did we respond to the {s} {sev}?",
    "was {s} down {day}?",
    "who handled the {s} incident?",
    "{s} {sev} — what was done?",
];

// ── OOD domains (stress-suite only; never part of the training corpus) ──

pub const OOD_DOMAINS: [&str; 8] = [
    "cooking",
    "travel",
    "music",
    "sports",
    "weather",
    "gardening",
    "fitness",
    "gaming",
];

const COOK_DISH: [&str; 12] = [
    "mushroom risotto",
    "cold brew concentrate",
    "sourdough starter",
    "chicken adobo",
    "lentil soup",
    "banana bread",
    "shakshuka",
    "pork carnitas",
    "miso-glazed eggplant",
    "overnight oats",
    "tomato confit",
    "black bean chili",
];
const COOK_ING: [&str; 12] = [
    "arborio rice",
    "coarse grounds",
    "rye flour",
    "soy vinegar",
    "red lentils",
    "overripe bananas",
    "farm eggs",
    "dried chiles",
    "white miso",
    "rolled oats",
    "roma tomatoes",
    "three bean mix",
];
const COOK_TECH: [&str; 12] = [
    "low and slow",
    "a hot sear first",
    "an overnight rest",
    "gentle folding",
    "a tight simmer",
    "a water bath",
    "dry brining",
    "resting the meat",
    "salting early",
    "toasting the spices",
    "deglazing the pan",
    "carryover heat",
];
const COOK_MIN: [&str; 12] = [
    "10",
    "15",
    "20",
    "25",
    "30",
    "40",
    "45",
    "60",
    "90",
    "2 hours",
    "3 hours",
    "overnight",
];

const TRV_CITY: [&str; 12] = [
    "Lisbon",
    "Kyoto",
    "Oaxaca",
    "Tbilisi",
    "Porto",
    "Hanoi",
    "Valparaiso",
    "Ljubljana",
    "Fez",
    "Busan",
    "Tallinn",
    "Medellin",
];
const TRV_SEAS: [&str; 12] = [
    "late September",
    "shoulder season",
    "early June",
    "the dry month",
    "carnival week",
    "harvest season",
    "the quiet weeks",
    "off-peak winter",
    "fiesta time",
    "monsoon edge",
    "spring break",
    "high summer",
];
const TRV_LAND: [&str; 12] = [
    "the tram line 28",
    "the bamboo grove",
    "the market halls",
    "the sulfur baths",
    "the river bridges",
    "the old quarter",
    "the funicular",
    "the castle hill",
    "the tanneries",
    "the fish market",
    "the singing dunes",
    "the cable cars",
];
const TRV_TIP: [&str; 12] = [
    "book trains a day ahead",
    "carry cash for markets",
    "skip the hop-on bus",
    "eat where workers eat",
    "go at opening hour",
    "learn five phrases",
    "avoid the free walking tour",
    "take the slow boat",
    "stay near a transit hub",
    "pack a rain shell",
    "reserve museums online",
    "walk the ridge at dawn",
];

const MUS_INSTR: [&str; 12] = [
    "the dreadnought",
    "the upright bass",
    "the hammered dulcimer",
    "the marimba",
    "the cajon",
    "the fretless bass",
    "the nyckelharpa",
    "the tabla set",
    "the baritone uke",
    "the Wurlitzer",
    "the bandoneon",
    "the koto",
];
const MUS_GENRE: [&str; 12] = [
    "desert blues",
    "bossa nova",
    "krautrock",
    "gamelan",
    "bluegrass",
    "dub techno",
    "field hollers",
    "shoegaze",
    "highlife",
    "modal jazz",
    "chanson",
    "drum & bass",
];
const MUS_TECH: [&str; 12] = [
    "drop D tuning",
    "walking bass lines",
    "brushes on the snare",
    "a clave pattern",
    "palm muting",
    "side-stick accents",
    "open-string drones",
    "cross-rhythm loops",
    "fingerpicking",
    "half-time feel",
    "call and response",
    "tape saturation",
];
const MUS_SONG: [&str; 12] = [
    "the bridge section",
    "the opening riff",
    "the 12-bar form",
    "the outro jam",
    "the horn stab",
    "the bass drop",
    "the second verse",
    "the drum break",
    "the chorus hook",
    "the intro vamp",
    "the solo section",
    "the tag ending",
];

const SPT_TEAM: [&str; 12] = [
    "the rowing eight",
    "the futsal side",
    "the cycling squad",
    "the volleyball club",
    "the rugby sevens",
    "the swim squad",
    "the cricket eleven",
    "the handball team",
    "the relay team",
    "the hockey line",
    "the climbing team",
    "the badminton pair",
];
const SPT_DRILL: [&str; 12] = [
    "pyramid intervals",
    "shadow drills",
    "tempo runs",
    "sprint ladders",
    "scrimmage sets",
    "fartlek sessions",
    "plate circuits",
    "reaction sprints",
    "small-sided games",
    "hill repeats",
    "technical circuits",
    "recovery spins",
];
const SPT_PLAYER: [&str; 12] = [
    "the setter",
    "the sweep rower",
    "the goalkeeper",
    "the anchor leg",
    "the point guard",
    "the fly half",
    "the lead-out rider",
    "the middle blocker",
    "the striker",
    "the libero",
    "the opening batsman",
    "the sprinter",
];
const SPT_STAT: [&str; 12] = [
    "split times",
    "possession rate",
    "serve accuracy",
    "lap deltas",
    "conversion rate",
    "turnover count",
    "power profile",
    "sprint speed",
    "pass completion",
    "heart-rate zones",
    "VO2 estimates",
    "field percentage",
];

const WTH_PLACE: [&str; 12] = [
    "the coast range",
    "the high desert",
    "the river delta",
    "the lake district",
    "the gulf coast",
    "the plateau",
    "the fjord towns",
    "the steppe",
    "the island chain",
    "the foothills",
    "the valley floor",
    "the northern pass",
];
const WTH_PHEN: [&str; 12] = [
    "morning fog banks",
    "monsoon cells",
    "lake-effect snow",
    "dry lightning",
    "katabatic winds",
    "a heat dome",
    "squall lines",
    "a polar plunge",
    "hail swaths",
    "a flash-flood watch",
    "freezing rain",
    "a dust front",
];
const WTH_SEAS: [&str; 12] = [
    "late October",
    "the shoulder weeks",
    "peak summer",
    "the wet season",
    "early spring",
    "harvest time",
    "the dry spell",
    "first frost",
    "storm season",
    "the thaw",
    "high pressure weeks",
    "the monsoon edge",
];
const WTH_ADV: [&str; 12] = [
    "grit the passes early",
    "stage pumps at low points",
    "warn the orchard crews",
    "hold the ferries",
    "pre-position generators",
    "clear the culverts",
    "open cooling centers",
    "delay the burn window",
    "shelter the livestock",
    "back up the well pumps",
    "close the ridge trail",
    "stage sandbags",
];

const GRD_PLANT: [&str; 12] = [
    "the tomato beds",
    "the espalier pears",
    "the pollinator strip",
    "the asparagus crowns",
    "the garlic rows",
    "the blueberry hedge",
    "the herb spiral",
    "the potato towers",
    "the cover crop",
    "the rhubarb patch",
    "the grape arbor",
    "the salad beds",
];
const GRD_SOIL: [&str; 12] = [
    "heavy clay",
    "sandy loam",
    "raised compost",
    "sheet-mulched beds",
    "chalky ground",
    "aged manure",
    "leaf mold",
    "biochar mix",
    "silt over gravel",
    "wood-chip paths",
    "worm castings",
    "spent mushroom compost",
];
const GRD_SEAS: [&str; 12] = [
    "late frost window",
    "first warm week",
    "midsummer",
    "the dry spell",
    "harvest push",
    "shoulder season",
    "first rains",
    "putting the beds down",
    "pruning month",
    "seed-starting week",
    "mulch month",
    "cleanup week",
];
const GRD_TASK: [&str; 12] = [
    "side-dress with compost",
    "thin the seedlings",
    "net the berries",
    "hill the potatoes",
    "top-dress the garlic",
    "prune the laterals",
    "water deeply twice a week",
    "undersow clover",
    "stake the heavy heads",
    "split the crowns",
    "flip the compost",
    "lay the drip line",
];

const FIT_MOVE: [&str; 12] = [
    "the trap-bar deadlift",
    "weighted carries",
    "the split squat",
    "push-up ladders",
    "the hip hinge drill",
    "sled pushes",
    "the pull-up pyramid",
    "farmer walks",
    "the goblet squat",
    "band pull-aparts",
    "plank pull-throughs",
    "the wall sit finisher",
];
const FIT_MUSC: [&str; 12] = [
    "posterior chain",
    "grip and forearms",
    "the upper back",
    "core bracing",
    "glutes and hams",
    "shoulder stabilizers",
    "the lats",
    "hip flexors",
    "calves and ankles",
    "rotator cuff",
    "the neck and traps",
    "adductors",
];
const FIT_REP: [&str; 12] = [
    "3×5",
    "5×3",
    "4×8",
    "2×10",
    "EMOM 10",
    "3 rounds of 12",
    "5×5",
    "10×1 heavy",
    "3×15 light",
    "ladder 1-2-3",
    "30 seconds on",
    "AMRAP 8",
];
const FIT_PLAN: [&str; 12] = [
    "after the easy run",
    "on lower days",
    "twice a week",
    "as the finisher",
    "before the long ride",
    "in the morning slot",
    "on deload weeks",
    "every other day",
    "as active recovery",
    "post-physio",
    "during lunch break",
    "before the spar",
];

const GAM_TITLE: [&str; 12] = [
    "the extraction shooter",
    "the city builder",
    "the deck-runner",
    "the farming sim",
    "the tactics remake",
    "the souls-like",
    "the rally sim",
    "the colony builder",
    "the puzzle box game",
    "the life sim",
    "the mech arena",
    "the roguelike deck",
];
const GAM_BOSS: [&str; 12] = [
    "the second-phase flagship",
    "the swamp wyrm",
    "the clockwork sentinel",
    "the flood gate guardian",
    "the twin duel",
    "the gauntlet keeper",
    "the sand leviathan",
    "the vault warden",
    "the storm drake",
    "the final auditor",
    "the mirror fight",
    "the siege breaker",
];
const GAM_MECH: [&str; 12] = [
    "parry windows",
    "the fatigue system",
    "tile adjacency rules",
    "the ammo economy",
    "permadeath branches",
    "the supply chain",
    "stagger damage",
    "the event deck",
    "the morale meter",
    "zoning tools",
    "the crafting tree",
    "the zone timer",
];
const GAM_PATCH: [&str; 12] = [
    "the season-4 rework",
    "the hotfix patch",
    "the balance pass",
    "the new expansion",
    "the netcode update",
    "the difficulty patch",
    "the mod tools drop",
    "the DLC launch",
    "the ranked reset",
    "the engine upgrade",
    "the roadmap stream",
    "the anniversary event",
];

// ── helpers ─────────────────────────────────────────────────────────────

fn fill(t: &str, subs: &[(&str, &str)]) -> String {
    let mut s = t.to_string();
    for (k, v) in subs {
        s = s.replace(&format!("{{{k}}}"), v);
    }
    s
}

fn project_name(pre: usize, suf: usize) -> String {
    format!("{}-{}", PROJ_PRE[pre], PROJ_SUF[suf])
}

/// Draws slot-index tuples that are unique within one category instance, so
/// families never share their full slot signature.  Falls back to accepting a
/// repeat after 5,000 tries — with pool sizes in the hundreds and at most a
/// few dozen families per instance, that path is unreachable in practice.
struct Picker {
    rng: Rng,
    used: HashSet<String>,
}

impl Picker {
    fn new(seed: u64) -> Self {
        Self {
            rng: Rng(seed),
            used: HashSet::new(),
        }
    }

    fn draw(
        &mut self,
        n: usize,
        pool_sizes: &[usize],
        key: &dyn Fn(&[usize]) -> String,
    ) -> Vec<Vec<usize>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut tries = 0usize;
            loop {
                tries += 1;
                let idx: Vec<usize> = pool_sizes
                    .iter()
                    .map(|&sz| (self.rng.next_u32() as usize) % sz)
                    .collect();
                let k = key(&idx);
                if !self.used.contains(&k) || tries > 5_000 {
                    self.used.insert(k);
                    out.push(idx);
                    break;
                }
            }
        }
        out
    }

    fn take(&mut self, n: usize, total: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..total).collect();
        self.rng.shuffle(&mut idx);
        idx.truncate(n);
        idx
    }

    /// A slot draw that is NOT part of the family key (may repeat across
    /// families — e.g. the same tool used by two projects).
    fn free(&mut self, sz: usize) -> usize {
        (self.rng.next_u32() as usize) % sz
    }
}

fn build_family(
    p: &mut Picker,
    val: bool,
    mem_t: &[&str],
    q_t: &[&str],
    subs: &[(&str, &str)],
) -> GenFamily {
    let mem_idx = p.take(3, mem_t.len());
    let memories = mem_idx
        .iter()
        .map(|&i| fill(mem_t[i], subs))
        .collect::<Vec<_>>();
    if val {
        let q_idx = p.take(2, q_t.len());
        GenFamily {
            memories,
            train_queries: Vec::new(),
            val_queries: q_idx.iter().map(|&i| fill(q_t[i], subs)).collect(),
        }
    } else {
        let q_idx = p.take(4, q_t.len());
        let train_queries = q_idx.iter().map(|&i| fill(q_t[i], subs)).collect();
        GenFamily {
            memories,
            train_queries,
            val_queries: Vec::new(),
        }
    }
}

fn fam_facts(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = (project, purpose): every facts template names at least these
    // two, so same-key families would print contradictory answers.
    let draws = p.draw(
        n,
        &[PROJ_PRE.len(), PROJ_SUF.len(), PURPOSES.len()],
        &|i| format!("{}-{}/{}", PROJ_PRE[i[0]], PROJ_SUF[i[1]], PURPOSES[i[2]]),
    );
    draws
        .iter()
        .map(|ix| {
            let proj = project_name(ix[0], ix[1]);
            let tool = TOOLS[p.free(TOOLS.len())];
            build_family(
                p,
                val,
                &FACTS_MEM,
                &FACTS_Q,
                &[("p", proj.as_str()), ("t", tool), ("u", PURPOSES[ix[2]])],
            )
        })
        .collect()
}

fn fam_documentation(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = library (must be unique per family: the queries ask for the
    // retry count, so two families on one library would contradict).
    const STEM: [&str; 24] = [
        "netclient", "queueman", "storerite", "authkit", "logfly", "cacherite", "fleetd",
        "plotkit", "schemadoc", "batcher", "relic", "tunnel", "watchdog", "ledgerlite", "packrat",
        "keyringd", "sweeper", "messenger", "sampler", "gater", "meterman", "profiler", "circuit",
        "handoff",
    ];
    const VARIANT: [&str; 8] = ["", "-ng", "-x", "-pro", "-lite", "-go", "-rs", "-cloud"];
    let libs: Vec<String> = VARIANT
        .iter()
        .flat_map(|v| STEM.iter().map(move |s| format!("{s}{v}")))
        .collect();
    let draws = p.draw(n, &[libs.len()], &|i| libs[i[0]].clone());
    draws
        .iter()
        .map(|ix| {
            let retry = RETRIES[p.free(RETRIES.len())].to_string();
            build_family(
                p,
                val,
                &DOC_MEM,
                &DOC_Q,
                &[("l", libs[ix[0]].as_str()), ("n", retry.as_str())],
            )
        })
        .collect()
}

fn fam_code(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = project: several code queries name only the project, so one
    // project carries exactly one helper family.
    let draws = p.draw(n, &[PROJ_PRE.len(), PROJ_SUF.len()], &|i| {
        format!("{}-{}", PROJ_PRE[i[0]], PROJ_SUF[i[1]])
    });
    draws
        .iter()
        .map(|ix| {
            let proj = project_name(ix[0], ix[1]);
            let f = FNAMES[p.free(FNAMES.len())];
            let a = FARGS[p.free(FARGS.len())];
            let d = FDESCS[p.free(FDESCS.len())];
            let w = FVERBS[p.free(FVERBS.len())];
            build_family(
                p,
                val,
                &CODE_MEM,
                &CODE_Q,
                &[
                    ("p", proj.as_str()),
                    ("f", f),
                    ("a", a),
                    ("d", d),
                    ("w", w),
                ],
            )
        })
        .collect()
}

fn fam_paths(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // ONE suffix shared by every family in this instance — the acceptance
    // corpus's hard-negative shape (same src/core/config.rs in six projects,
    // only the project tells them apart).
    let s_idx = p.free(PATH_SUFFIX.len());
    let suffix = PATH_SUFFIX[s_idx];
    let stem = suffix
        .rsplit('/')
        .next()
        .and_then(|f| f.split('.').next())
        .unwrap_or("core");
    let syn = PATHS_SYN[s_idx];
    let draws = p.draw(n, &[PROJ_PRE.len(), PROJ_SUF.len()], &|i| {
        format!("{}-{}", PROJ_PRE[i[0]], PROJ_SUF[i[1]])
    });
    draws
        .iter()
        .map(|ix| {
            let proj = project_name(ix[0], ix[1]);
            let n1 = PATH_NOTES[p.free(PATH_NOTES.len())];
            let n2 = PATH_NOTES[p.free(PATH_NOTES.len())];
            let n3 = PATH_NOTES[p.free(PATH_NOTES.len())];
            build_family(
                p,
                val,
                &PATHS_MEM,
                &PATHS_Q,
                &[
                    ("p", proj.as_str()),
                    ("s", suffix),
                    ("stem", stem),
                    ("syn", syn),
                    ("n1", n1),
                    ("n2", n2),
                    ("n3", n3),
                ],
            )
        })
        .collect()
}

fn fam_errors(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = error code, spread across the band like the acceptance corpus
    // (E4102 … E4471); a dense band over-smoothed code discrimination.
    let draws = p.draw(n, &[900], &|i| format!("E4{:03}", 100 + i[0]));
    draws
        .iter()
        .map(|ix| {
            let code = format!("E4{:03}", 100 + ix[0]);
            let comp = ERR_COMP[p.free(ERR_COMP.len())];
            let m = ERR_MSG[p.free(ERR_MSG.len())];
            let f = ERR_FIX[p.free(ERR_FIX.len())];
            build_family(
                p,
                val,
                &ERR_MEM,
                &ERR_Q,
                &[("c", code.as_str()), ("comp", comp), ("m", m), ("f", f)],
            )
        })
        .collect()
}

fn fam_versions(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = (project, library): 14 libs across many families guarantees the
    // same-lib different-version hard negatives of the acceptance corpus,
    // while one project never pins one lib twice.
    let draws = p.draw(
        n,
        &[PROJ_PRE.len(), PROJ_SUF.len(), LIBS.len()],
        &|i| {
            format!(
                "{}-{}/{}",
                PROJ_PRE[i[0]], PROJ_SUF[i[1]], LIBS[i[2]]
            )
        },
    );
    draws
        .iter()
        .map(|ix| {
            let proj = project_name(ix[0], ix[1]);
            let v = VERS[p.free(VERS.len())];
            build_family(
                p,
                val,
                &VER_MEM,
                &VER_Q,
                &[("p", proj.as_str()), ("l", LIBS[ix[2]]), ("v", v)],
            )
        })
        .collect()
}

fn fam_dates(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = thing: the queries ask about the thing alone, so things are
    // unique per family; teams repeat (real calendars do too).
    let draws = p.draw(n, &[THINGS.len()], &|i| THINGS[i[0]].to_string());
    draws
        .iter()
        .map(|ix| {
            let t = TEAMS[p.free(TEAMS.len())];
            let d = DATES[p.free(DATES.len())];
            let dep = DEPS[p.free(DEPS.len())];
            build_family(
                p,
                val,
                &DATE_MEM,
                &DATE_Q,
                &[("thing", THINGS[ix[0]]), ("t", t), ("d", d), ("dep", dep)],
            )
        })
        .collect()
}

fn fam_paraphrase(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = subject: half the paraphrase queries name only the subject.
    // Mechanism/detail pairs are drawn freely; their (mem, query) halves use
    // deliberately different vocabulary, so retrieval must work through
    // meaning, not echoed tokens.
    let subj_n = (n + 1) / 2;
    let mech_n = n - subj_n;
    let mut out = Vec::with_capacity(n);
    let draws = p.draw(subj_n, &[SUBJ.len()], &|i| SUBJ[i[0]].to_string());
    for ix in draws.iter() {
        let (mech, q_mech) = PARA_MECH[p.free(PARA_MECH.len())];
        let (det, q_det) = PARA_DET[p.free(PARA_DET.len())];
        let num = NUMS[p.free(NUMS.len())];
        let mem_idx = p.take(3, PARA_MEM.len());
        let memories = mem_idx
            .iter()
            .map(|&ti| {
                fill(
                    PARA_MEM[ti],
                    &[
                        ("subj", SUBJ[ix[0]]),
                        ("mech", mech),
                        ("det", det),
                        ("num", num),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let q_idx = p.take(4, PARA_Q.len());
        let queries: Vec<String> = q_idx
            .iter()
            .map(|&qi| {
                fill(
                    PARA_Q[qi],
                    &[
                        ("subj", SUBJ[ix[0]]),
                        ("q_mech", q_mech),
                        ("q_det", q_det),
                    ],
                )
            })
            .collect();
        out.push(if val {
            GenFamily {
                memories,
                train_queries: Vec::new(),
                val_queries: queries.into_iter().take(2).collect(),
            }
        } else {
            GenFamily {
                memories,
                train_queries: queries,
                val_queries: Vec::new(),
            }
        });
    }
    // mechanism-only families: key = mech index (uniqueness without a
    // subject); the hardest shape — queries share zero anchor tokens with
    // the memories.
    if mech_n > 0 {
        let mdraws = p.draw(mech_n, &[PARA_MECH.len()], &|i| i[0].to_string());
        for mix in mdraws.iter() {
            let (mech, q_mech) = PARA_MECH[mix[0]];
            let di = p.free(PARA_DET.len());
            let (det, _) = PARA_DET[di];
            out.push(build_family(
                p,
                val,
                &MECHONLY_MEM,
                &MECHONLY_Q,
                &[("mech", mech), ("det", det), ("q_mech", q_mech)],
            ));
        }
    }
    out
}

fn fam_decisions(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = project ("{p} policy on this" names only the project).
    let draws = p.draw(n, &[PROJ_PRE.len(), PROJ_SUF.len()], &|i| {
        format!("{}-{}", PROJ_PRE[i[0]], PROJ_SUF[i[1]])
    });
    draws
        .iter()
        .map(|ix| {
            let proj = project_name(ix[0], ix[1]);
            let dec = DECISIONS[p.free(DECISIONS.len())];
            let r = REASONS[p.free(REASONS.len())];
            build_family(
                p,
                val,
                &DEC_MEM,
                &DEC_Q,
                &[("p", proj.as_str()), ("dec", dec), ("r", r)],
            )
        })
        .collect()
}

fn fam_preferences(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = user (queries ask "what does {u} prefer?").
    const NAME: [&str; 20] = [
        "ana", "bruno", "chiara", "diego", "elif", "farid", "gwen", "hiro", "ines", "jonas",
        "katya", "liam", "mara", "nils", "olga", "priya", "quinn", "raul", "sana", "tomas",
    ];
    const ROLE: [&str; 4] = ["-dev", "-ops", "-qa", "-pm"];
    let users: Vec<String> = ROLE
        .iter()
        .flat_map(|r| NAME.iter().map(move |n| format!("{n}{r}")))
        .collect();
    let draws = p.draw(n, &[users.len()], &|i| users[i[0]].clone());
    draws
        .iter()
        .map(|ix| {
            let pref = PREFS[p.free(PREFS.len())];
            let v = VALUES[p.free(VALUES.len())];
            build_family(
                p,
                val,
                &PREF_MEM,
                &PREF_Q,
                &[("u", users[ix[0]].as_str()), ("pref", pref), ("v", v)],
            )
        })
        .collect()
}

fn fam_incidents(p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    // key = service ("{s} incident details" names only the service).
    const SVC: [&str; 24] = [
        "auth-api", "billing-worker", "search-index", "media-gw", "sync-relay", "export-svc",
        "notify-svc", "ledger-db", "edge-cache", "queue-main", "upload-svc", "session-store",
        "web-frontend", "job-runner", "metrics-ingest", "log-shipper", "key-vault", "cron-runner",
        "cdn-origin", "graphql-api", "pdf-render", "mail-relay", "webhook-hub", "feature-flags",
    ];
    const ENV: [&str; 4] = ["", "-staging", "-eu", "-canary"];
    let services: Vec<String> = ENV
        .iter()
        .flat_map(|e| SVC.iter().map(move |s| format!("{s}{e}")))
        .collect();
    let draws = p.draw(n, &[services.len()], &|i| services[i[0]].clone());
    draws
        .iter()
        .map(|ix| {
            let sev = SEVS[p.free(SEVS.len())];
            let a = ACTIONS[p.free(ACTIONS.len())];
            let day = DAYS[p.free(DAYS.len())];
            build_family(
                p,
                val,
                &INC_MEM,
                &INC_Q,
                &[
                    ("s", services[ix[0]].as_str()),
                    ("sev", sev),
                    ("a", a),
                    ("day", day),
                ],
            )
        })
        .collect()
}

fn build_cat(ci: usize, p: &mut Picker, n: usize, val: bool) -> Vec<GenFamily> {
    match ci {
        0 => fam_paraphrase(p, n, val),
        1 => fam_facts(p, n, val),
        2 => fam_documentation(p, n, val),
        3 => fam_code(p, n, val),
        4 => fam_paths(p, n, val),
        5 => fam_errors(p, n, val),
        6 => fam_versions(p, n, val),
        7 => fam_dates(p, n, val),
        8 => fam_decisions(p, n, val),
        9 => fam_preferences(p, n, val),
        10 => fam_incidents(p, n, val),
        _ => unreachable!("category index out of range"),
    }
}

fn chatter(p: &mut Picker, n: usize) -> Vec<String> {
    let draws = p.draw(n, &[CHAT_A.len(), CHAT_B.len()], &|i| {
        format!("{}/{}", CHAT_A[i[0]], CHAT_B[i[1]])
    });
    draws
        .iter()
        .map(|ix| format!("{} {}", CHAT_A[ix[0]], CHAT_B[ix[1]]))
        .collect()
}

/// Build the deterministic training corpus.  `boost` maps an 8-name eval
/// category to extra train-family count (failure mining); extra categories
/// and the curated groups are never boosted.
pub fn corpus(seed: u64, boost: &HashMap<String, usize>) -> Corpus {
    let mut p = Picker::new(seed);
    let mut cats = Vec::with_capacity(CAT_NAMES.len());
    for (ci, name) in CAT_NAMES.iter().enumerate() {
        let extra = boost.get(*name).copied().unwrap_or(0);
        let train = build_cat(ci, &mut p, BASE_FAMILIES + extra, false);
        let val = build_cat(ci, &mut p, VAL_FAMILIES, true);
        cats.push(GenCat { train, val });
    }
    // hand-written associative scenario families ride in the paraphrase
    // category: the acceptance corpus's hardest skill cannot come from slot
    // templates, only from real rephrased scenarios.
    for spec in crate::data_embedding::ASSOC_FAMILIES {
        assert_eq!(spec.len(), 7, "assoc family shape: 3 memories + 4 queries");
        cats[0].train.push(GenFamily {
            memories: spec[..3].iter().map(|s| s.to_string()).collect(),
            train_queries: spec[3..].iter().map(|s| s.to_string()).collect(),
            val_queries: Vec::new(),
        });
    }
    // short-query scenario families: bare noun-phrase probes over long
    // memories ("retry policy" style heads that never appear verbatim).
    for spec in crate::data_embedding::ASSOC_SHORT {
        assert_eq!(spec.len(), 7, "assoc short family shape");
        cats[0].train.push(GenFamily {
            memories: spec[..3].iter().map(|s| s.to_string()).collect(),
            train_queries: spec[3..].iter().map(|s| s.to_string()).collect(),
            val_queries: Vec::new(),
        });
    }
    for spec in crate::data_embedding::ASSOC_VAL {
        assert_eq!(spec.len(), 7, "assoc val family shape");
        cats[0].val.push(GenFamily {
            memories: spec[..3].iter().map(|s| s.to_string()).collect(),
            train_queries: Vec::new(),
            val_queries: spec[3..].iter().map(|s| s.to_string()).collect(),
        });
    }
    let curated = data_embedding::groups()
        .iter()
        .map(|g| g.examples.iter().map(|s| s.to_string()).collect())
        .collect();
    let chat = chatter(&mut p, 16);
    let c = Corpus {
        cats,
        curated,
        chatter: chat,
    };
    assert_unique_texts(&c);
    c
}

fn assert_unique_texts(c: &Corpus) {
    let mut seen = HashSet::new();
    let mut check = |t: &str| {
        assert!(seen.insert(t.to_string()), "duplicate corpus text: {t:?}");
    };
    for cat in &c.cats {
        for fam in cat.train.iter().chain(cat.val.iter()) {
            for t in fam.memories.iter().chain(&fam.train_queries).chain(&fam.val_queries) {
                check(t);
            }
        }
    }
    for g in &c.curated {
        for t in g {
            check(t);
        }
    }
    for t in &c.chatter {
        check(t);
    }
}

// ── OOD domains (stress suite) ──────────────────────────────────────────

const OOD_MEM: [&str; 3] = [
    "About {a}: the trick is {b} with {c}, and you should {d}.",
    "Notes on {a} — {b} matters most; {c} helps, and {d}.",
    "{a} works best {c}; {b} is the detail people miss, then {d}.",
];
const OOD_Q: [&str; 4] = [
    "how do I get {a} right?",
    "what matters for {a}?",
    "{a} — any tips?",
    "is {b} really needed for {a}?",
];

struct OodPools {
    a: &'static [&'static str],
    b: &'static [&'static str],
    c: &'static [&'static str],
    d: &'static [&'static str],
}

fn ood_pools(domain: usize) -> OodPools {
    match domain {
        0 => OodPools {
            a: &COOK_DISH,
            b: &COOK_ING,
            c: &COOK_TECH,
            d: &COOK_MIN,
        },
        1 => OodPools {
            a: &TRV_CITY,
            b: &TRV_LAND,
            c: &TRV_SEAS,
            d: &TRV_TIP,
        },
        2 => OodPools {
            a: &MUS_GENRE,
            b: &MUS_INSTR,
            c: &MUS_TECH,
            d: &MUS_SONG,
        },
        3 => OodPools {
            a: &SPT_TEAM,
            b: &SPT_DRILL,
            c: &SPT_STAT,
            d: &SPT_PLAYER,
        },
        4 => OodPools {
            a: &WTH_PLACE,
            b: &WTH_PHEN,
            c: &WTH_SEAS,
            d: &WTH_ADV,
        },
        5 => OodPools {
            a: &GRD_PLANT,
            b: &GRD_SOIL,
            c: &GRD_SEAS,
            d: &GRD_TASK,
        },
        6 => OodPools {
            a: &FIT_MOVE,
            b: &FIT_MUSC,
            c: &FIT_REP,
            d: &FIT_PLAN,
        },
        _ => OodPools {
            a: &GAM_TITLE,
            b: &GAM_MECH,
            c: &GAM_PATCH,
            d: &GAM_BOSS,
        },
    }
}

/// Generate `n` fresh families for an OOD domain.  The stress suite uses a
/// different seed than any training run, so OOD instances stay unseen even
/// if a domain name is later adopted for training.
pub fn ood_domain_families(domain: usize, seed: u64, n: usize) -> Vec<GenFamily> {
    let pools = ood_pools(domain);
    let mut p = Picker::new(seed);
    // key = pool-a slot: the queries name only {a}, so a-values must be
    // unique per family (n ≤ 12 per domain holds).
    let draws = p.draw(n, &[pools.a.len()], &|i| i[0].to_string());
    draws
        .iter()
        .map(|ix| {
            let subs = [
                ("a", pools.a[ix[0]]),
                ("b", pools.b[p.free(pools.b.len())]),
                ("c", pools.c[p.free(pools.c.len())]),
                ("d", pools.d[p.free(pools.d.len())]),
            ];
            let memories = OOD_MEM
                .iter()
                .map(|t| fill(t, &subs))
                .collect::<Vec<_>>();
            let queries = OOD_Q
                .iter()
                .map(|t| fill(t, &subs))
                .collect::<Vec<_>>();
            GenFamily {
                memories,
                train_queries: queries,
                val_queries: Vec::new(),
            }
        })
        .collect()
}
