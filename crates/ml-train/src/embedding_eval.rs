// ml-train/src/embedding_eval.rs — PLAN-TINY-SEMANTIC-EMBEDDER §10 acceptance
// harness for the semantic memory embedder.
//
//   cargo run -p ml-train --release -- embed-eval
//
// Runs the BAKED runtime artifact (sofuu-core::embedding, artifact id pinned
// by cargo test) against the hash-v1 baseline over a deterministic,
// Sofuu-shaped corpus in the plan's 8 categories, applies the §10.1 gates
// verbatim, and checks the §10.3 resource budgets. Dev-only: no network, no
// keys, no external files beyond the in-tree artifact.
//
// Hard negatives are structural: within every category the six families share
// one sentence template and differ only in slot values, so the distractors are
// exactly the plan's cases — same path different project (paths), same library
// different version (versions), same message shape different component
// (errors), same event shape different date (dates). Retrieval runs through
// the real Cma::recall path (HNSW + tier/strength scoring), which is the
// candidate-set proxy for the plan's end-to-end gate (§10.1 item 5).

use std::path::PathBuf;
use std::time::Instant;

use sofuu_core::embedding::{
    artifact_id_for, hash_v1_features, QuantizedProjector, HASH_DIM, MODEL_ID, SEMANTIC_DIM,
};
use sofuu_core::memory::cma::Cma;

const TOP_K: usize = 5;
const P95_SAMPLES: usize = 200;
/// Round-6 two-channel fusion constants — PRE-REGISTERED in
/// PLAN-TINY-SEMANTIC-EMBEDDER.md (round 6) before any grading: α = 10 is
/// the original RRF default, N = 20 = 4× the final top-5. Not tuned on §10.
const FUSE_ALPHA: f64 = 10.0;
pub(crate) const FUSE_TOP_N: usize = 20;
/// Round-7 admission cap — PRE-REGISTERED in PLAN-TINY-SEMANTIC-EMBEDDER.md
/// (round 7) before any grading: at most 2 of the final 5 slots may go to
/// candidates with NO hash support (rank_H > N). Fixed by principle (the
/// lexical channel keeps majority control of the final list; the semantic
/// channel may add up to two unconfirmed finds), NOT tuned on §10. It
/// targets the displacement mechanism round 6's pre-registration itself
/// predicted and round 6's grade confirmed.
const FUSE_SEM_ONLY_MAX: usize = 2;
/// §10.1 gate values, verbatim from the plan.
const G1_PARAPHRASE_MIN: f32 = 0.85;
const G1_MARGIN: f32 = 0.10;
const G2_FACTS_MIN: f32 = 0.90;
const G3_MARGIN: f32 = 0.01; // ≤ 1pp below hash-v1
const G5_MARGIN: f32 = 0.01; // ≤ 1pp overall regression
const G6_MIN_WINS: usize = 4;
/// §10.3 payload budget, 32 KiB by default.  Width experiments may raise it
/// via SOFUU_EMB_BUDGET_KIB — the override is always printed, never silent.
fn budget_payload_bytes() -> u64 {
    std::env::var("SOFUU_EMB_BUDGET_KIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32)
        * 1024
}
const BUDGET_INIT_MS: f64 = 25.0;
const BUDGET_P95_MS: f64 = 10.0;

const CAT_NAMES: [&str; 8] = [
    "paraphrase", "facts", "documentation", "code", "paths", "errors", "versions", "dates",
];
const CHATTER_TAG: (usize, usize) = (CAT_NAMES.len(), 0);

struct Family {
    memories: Vec<String>,
    queries: Vec<String>,
}

struct Category {
    name: &'static str,
    families: Vec<Family>,
}

// ── corpus ──────────────────────────────────────────────────────────────
// 8 categories × 6 families × (3 memories + 4 queries). Deterministic:
// every string is a literal or a format! of literals — no RNG anywhere.

fn paraphrase() -> Category {
    let fam = |ms: [&str; 3], qs: [&str; 4]| Family {
        memories: ms.iter().map(|s| s.to_string()).collect(),
        queries: qs.iter().map(|s| s.to_string()).collect(),
    };
    Category {
        name: "paraphrase",
        families: vec![
            fam(
                [
                    "We decided the session store compresses with qtc before it ever touches disk; zlib is only the fallback for containers that must stay readable by old builds.",
                    "Decision from the storage review: qtc is the codec for session state, zlib stays as the legacy fallback path.",
                    "Session state hits the disk only through qtc compression now. The zlib fallback exists purely so older containers still open.",
                ],
                [
                    "how are session files compressed on disk?",
                    "which codec does the session store use?",
                    "what happens if qtc is unavailable for a session container?",
                    "session persistence compression choice",
                ],
            ),
            fam(
                [
                    "Webhook deliveries from the billing provider retry five times with exponential backoff, then park in the dead-letter table for manual replay.",
                    "Billing webhooks: five attempts, exponential backoff between tries, after that they land in the dead-letter queue.",
                    "The billing integration gives up after the fifth delivery attempt and parks the event for a human to replay.",
                ],
                [
                    "how many times do billing webhooks redeliver?",
                    "what happens to a webhook that keeps failing?",
                    "where do failed payment events end up?",
                    "webhook delivery retry policy",
                ],
            ),
            fam(
                [
                    "Every release ships through the checklist: changelog written, versions bumped, binaries cross-built, smoke test on the packed artifact.",
                    "The release process is fixed: changelog, version bump, cross-compile all targets, unpack-and-run smoke check.",
                    "Before any release goes out we write the changelog, bump versions, build every target and smoke test the bundle.",
                ],
                [
                    "what must happen before a release goes out?",
                    "steps for shipping a new build",
                    "release day routine",
                    "how do we cut a release?",
                ],
            ),
            fam(
                [
                    "The login timeouts came from connection pool exhaustion; the fix keeps a warm pool of four sockets and drops idle ones after thirty seconds.",
                    "Root cause of the auth outages: the connection pool ran dry under load. We now hold four warm sockets and idle-evict at thirty seconds.",
                    "Users saw login failures because the pool starved; keeping four sockets warm and evicting idle ones after 30s fixed it.",
                ],
                [
                    "why were users failing to sign in?",
                    "what fixed the auth timeouts?",
                    "login outage root cause",
                    "how is the connection pool sized?",
                ],
            ),
            fam(
                [
                    "Memory recall is capped at two thousand tokens per turn so a single lookup can never crowd out the working context.",
                    "We limit retrieval to 2k tokens a turn; nothing a memory lookup returns may crowd out live conversation.",
                    "Recall budget: at most 2000 tokens of remembered context per request, enforced before the prompt is assembled.",
                ],
                [
                    "how much memory can be pulled into a prompt?",
                    "is there a cap on remembered context?",
                    "token limit for memory retrieval per turn",
                    "recall budget size",
                ],
            ),
            fam(
                [
                    "The Windows build compiles with MSVC only; the shim layer maps the POSIX calls and vcpkg supplies zlib and curl.",
                    "Porting notes: target MSVC on Windows, keep the POSIX shims thin, take zlib and libcurl from vcpkg.",
                    "On Windows we build with the MSVC toolchain; a small shim covers POSIX bits and vcpkg provides the C dependencies.",
                ],
                [
                    "which toolchain does the Windows build use?",
                    "where do zlib and curl come from on Windows?",
                    "how is the app ported to Windows?",
                    "windows build dependencies",
                ],
            ),
        ],
    }
}

fn facts() -> Category {
    let defs: [(&str, &str, &str); 6] = [
        ("sofuu-cli", "rusqlite", "the session registry"),
        ("black-hole-disk", "AES-GCM", "sector encryption"),
        ("sofuu-desktop", "React", "the settings UI"),
        ("billing-api", "Celery", "invoice scheduling"),
        ("nimbus-db", "io_uring", "the write path"),
        ("atlas-web", "Redis", "asset cache invalidation"),
    ];
    Category {
        name: "facts",
        families: defs
            .iter()
            .map(|&(p, t, u)| Family {
                memories: vec![
                    format!("In {p} we use {t} for {u}; that was settled at the kickoff and written into the decision log."),
                    format!("Decision record — {p}: {t} handles {u}. Don't introduce a second tool for the same job."),
                    format!("{u} inside {p} runs on {t} (agreed with the team, see the architecture notes)."),
                ],
                queries: vec![
                    format!("what does {p} use for {u}?"),
                    format!("{p}: which tool covers {u}?"),
                    format!("is {t} the tool for {u} in {p}?"),
                    format!("which component handles {u} in {p}?"),
                ],
            })
            .collect(),
    }
}

fn documentation() -> Category {
    let defs: [(&str, u32); 6] = [
        ("netclient", 4),
        ("queueman", 2),
        ("storerite", 3),
        ("authkit", 5),
        ("logfly", 3),
        ("cacherite", 6),
    ];
    Category {
        name: "documentation",
        families: defs
            .iter()
            .map(|&(l, n)| Family {
                memories: vec![
                    format!("{l} retry guide: the client retries a failed request up to {n} times before the error reaches the caller."),
                    format!("Working with {l}: attempts are capped at {n}; each retry waits longer than the last (exponential backoff)."),
                    format!("{l} operational notes — after {n} failed attempts the client surfaces the underlying error instead of retrying again."),
                ],
                queries: vec![
                    format!("how many times does {l} retry a failed request?"),
                    format!("does {l} back off between retries?"),
                    format!("what happens when {l} exhausts its attempts?"),
                    format!("{l} maximum attempts"),
                ],
            })
            .collect(),
    }
}

fn code() -> Category {
    let defs: [(&str, &str, &str, &str, &str); 6] = [
        ("rag-index", "align_chunks", "the corpus path", "splits documents into overlapping chunks and returns their offsets", "chunk documents"),
        ("cache-clear", "purge_stale", "the cache root", "walks entries older than the TTL and unlinks them", "clear expired cache entries"),
        ("parse-args", "split_cli", "the argv slice", "tokenizes quoted arguments and returns the flag table", "parse command line flags"),
        ("retry-wrap", "with_backoff", "the closure", "reruns a fallible call with growing delays", "retry a flaky call"),
        ("dedupe", "drop_dups", "the record list", "removes near-identical rows keeping the newest", "remove duplicate rows"),
        ("rate-gate", "throttle", "the bucket key", "token-buckets calls per key and sleeps on overflow", "apply rate limiting"),
    ];
    Category {
        name: "code",
        families: defs
            .iter()
            .map(|&(p, f, a, d, w)| Family {
                memories: vec![
                    format!("Snippet from {p}: {f}({a}) — {d}. Call it after the handle is open."),
                    format!("In {p} the helper {f} {d}; example: {f}({a});"),
                    format!("{p} utility: {f} takes {a} and {d}."),
                ],
                queries: vec![
                    format!("how do I {w} in {p}?"),
                    format!("{p} helper for {w}"),
                    format!("which function handles {w} in {p}?"),
                    format!("show me the {w} call in {p}"),
                ],
            })
            .collect(),
    }
}

fn paths() -> Category {
    // Hard negatives by construction: the SAME path suffix in six projects.
    let defs: [(&str, &str, &str, &str); 6] = [
        ("sofuu-cli", "parses the TOML once at boot", "it caches the parsed table in a OnceLock", "To add a setting"),
        ("sofuu-desktop", "mirrors the engine config into the WebView", "edits flow through the Tauri commands", "To expose a new pref"),
        ("helmor", "resolves the workspace root first", "paths are resolved relative to the workspace", "To relocate state"),
        ("mimosa", "validates the schema before scan", "validation runs before any scanner starts", "To add a rule class"),
        ("qtsq-checkout", "guards the codec offset table", "offset 5016 is integrity-checked here", "To touch the codec"),
        ("landing-site", "only reads public/site.json", "the static export has no server config", "To change metadata"),
    ];
    Category {
        name: "paths",
        families: defs
            .iter()
            .map(|&(p, n1, n2, n3)| Family {
                memories: vec![
                    format!("{p}/src/core/config.rs — {n1}."),
                    format!("The config loader lives at {p}/src/core/config.rs; {n2}."),
                    format!("{n3}: edit {p}/src/core/config.rs and rebuild."),
                ],
                queries: vec![
                    format!("where is the config loader in {p}?"),
                    format!("{p} config file path"),
                    format!("which file loads settings in {p}?"),
                    format!("settings loader location in {p}"),
                ],
            })
            .collect(),
    }
}

fn errors() -> Category {
    let defs: [(&str, &str, &str, &str); 6] = [
        ("E4102", "scheduler", "worker pool drained while a batch was in flight", "cap concurrent batches at two"),
        ("E4117", "indexer", "mmap window grew past the address space limit", "shard the index by prefix"),
        ("E4233", "gateway", "upstream closed the stream before the trailer", "enable TCP keepalive on the proxy"),
        ("E4388", "replicator", "sequence gap detected between mirrors", "resync from the primary snapshot"),
        ("E4405", "wal-writer", "segment header checksum did not match", "truncate to the last good frame"),
        ("E4471", "compactor", "snapshot pin count went negative", "rebuild the pin table at open"),
    ];
    Category {
        name: "errors",
        families: defs
            .iter()
            .map(|&(c, comp, m, f)| Family {
                memories: vec![
                    format!("{comp} crashed with error {c}: {m}. Fix: {f}."),
                    format!("Known issue {c} in {comp} — {m}. Workaround: {f}."),
                    format!("Error {c} ({comp}): {m}. We {f} until the upstream patch lands."),
                ],
                queries: vec![
                    format!("what is error {c} in {comp}?"),
                    format!("{comp} failing with {c}"),
                    format!("how do I fix {c}?"),
                    format!("meaning of error {c}"),
                ],
            })
            .collect(),
    }
}

fn versions() -> Category {
    // Hard negatives by construction: the same library pinned to conflicting
    // versions across projects.
    let defs: [(&str, &str, &str); 6] = [
        ("sofuu-cli", "tokio", "1.38.1"),
        ("sofuu-desktop", "tokio", "1.35.0"),
        ("atlas-web", "tokio", "1.40.2"),
        ("billing-api", "serde", "1.0.210"),
        ("nimbus-db", "serde", "1.0.203"),
        ("helmor", "serde", "1.0.219"),
    ];
    Category {
        name: "versions",
        families: defs
            .iter()
            .map(|&(p, l, v)| Family {
                memories: vec![
                    format!("{p} pins {l} to v{v}. Do not bump without the compat matrix."),
                    format!("{l} v{v} in {p} — locked until the next major audit."),
                    format!("Upgrade rule: {p} stays on {l} v{v}."),
                ],
                queries: vec![
                    format!("which {l} version does {p} use?"),
                    format!("{p} {l} pin"),
                    format!("is {l} bumped in {p}?"),
                    format!("what {l} release is {p} on?"),
                ],
            })
            .collect(),
    }
}

fn dates() -> Category {
    let defs: [(&str, &str, &str, &str); 6] = [
        ("platform", "the mesh sync beta", "October 14", "the keyring migration"),
        ("runtime", "the RLM sandbox GA", "October 21", "the tool-cache rewrite"),
        ("billing", "the invoice export", "October 7", "the ledger backfill"),
        ("infra", "the colo migration", "November 4", "the DNS cutover"),
        ("mobile", "the offline mode", "November 18", "the sync protocol freeze"),
        ("security", "the audit remediation", "October 30", "the dependency sweep"),
    ];
    Category {
        name: "dates",
        families: defs
            .iter()
            .map(|&(t, thing, d, dep)| Family {
                memories: vec![
                    format!("{t} ships {thing} on {d}. Blocked if {dep} slips."),
                    format!("Release plan: {thing} lands {d} ({t})."),
                    format!("{thing} — target {d}, owner {t}, depends on {dep}."),
                ],
                queries: vec![
                    format!("when does {t} ship {thing}?"),
                    format!("{thing} release date"),
                    format!("what is the target date for {thing}?"),
                    format!("is {thing} scheduled before the audit?"),
                ],
            })
            .collect(),
    }
}

fn chatter() -> Vec<String> {
    [
        "hey, did you watch the match last night?",
        "remind me to water the plants when I get home",
        "the coffee machine on floor three is broken again",
        "I liked that book about the lighthouse keeper",
        "lunch tomorrow at the noodle place?",
        "my back hurts from the new chair",
        "the cat knocked over the monitor cable",
        "weekend plans: hike if the weather holds",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// ── evaluation ──────────────────────────────────────────────────────────

#[derive(Clone)]
struct BackendScores {
    /// mean recall@5 per category (family hit anywhere in top 5)
    recall: [f32; 8],
    /// mean precision@5 per category (fraction of top 5 inside target family)
    precision: [f32; 8],
    /// mean hard-negative contamination per category
    /// (top-5 hits from a sibling family of the same category)
    contamination: [f32; 8],
    overall_recall: f32,
    overall_precision: f32,
}

fn eval_backend(
    name: &str,
    dim: usize,
    embed: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, (usize, usize))],
    queries: &[(String, (usize, usize))],
    n_fams: usize,
) -> BackendScores {
    let mut cma = Cma::new(dim);
    // id → (category, family); remember() may return an existing id when its
    // near-dup gate fires, so build the map from the returned ids.
    let mut id_tag: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    for (text, tag) in records {
        let id = cma.remember(&embed(text), text, "note", 0);
        assert!(id >= 0, "{name}: remember failed for {text:?}");
        id_tag.insert(id as u32, *tag);
    }
    // every family must have survived dedup with at least one record
    // (chatter shares one out-of-range tag; exclude it from the count)
    let mut fam_seen = std::collections::HashSet::new();
    for tag in id_tag.values() {
        if tag.0 < CAT_NAMES.len() {
            fam_seen.insert(*tag);
        }
    }
    let expected: usize = CAT_NAMES.len() * n_fams;
    assert_eq!(
        fam_seen.len(),
        expected,
        "{name}: dedup collapsed some families ({} of {} present)",
        fam_seen.len(),
        expected
    );

    score_queries(
        &id_tag,
        queries,
        &mut |q| {
            cma.recall(&embed(q), TOP_K)
                .into_iter()
                .map(|h| h.id)
                .collect()
        },
    )
}

/// Shared §10 metric accounting: given each query's final top-K candidate
/// ids (best first) and the id→(category, family) map, compute per-category
/// recall@5 / precision@5 / contamination exactly as the single-backend
/// eval does. Used by eval_backend and the round-6 fused pipeline so the
/// metrics stay byte-identical across candidate kinds.
fn score_queries(
    id_tag: &std::collections::HashMap<u32, (usize, usize)>,
    queries: &[(String, (usize, usize))],
    top_k_of: &mut dyn FnMut(&str) -> Vec<u32>,
) -> BackendScores {
    // Per-query accounting: recall@5 = query has ≥1 target-family hit in its
    // top-5; precision@5 = target slots / 5 for that query (top_k < 5 slots
    // are still charged — a short candidate list is not free).
    let n_q_per_cat = CAT_NAMES.len();
    let mut hit_q = [[0u32; 3]; 8]; // per category: [queries w/ target hit, queries w/ same-cat sibling hit, queries]
    let mut tgt_slots = [0u32; 8];
    let mut tot_slots = [0u32; 8];
    for (qtext, (qcat, qfam)) in queries {
        let ids = top_k_of(qtext);
        let mut tgt = 0u32;
        let mut sib = 0u32;
        for id in &ids {
            let tag = id_tag.get(id).copied().unwrap_or(CHATTER_TAG);
            if tag.0 == *qcat {
                if tag.1 == *qfam {
                    tgt += 1;
                } else {
                    sib += 1;
                }
            }
        }
        hit_q[*qcat][2] += 1;
        if tgt > 0 {
            hit_q[*qcat][0] += 1;
        }
        if sib > 0 {
            hit_q[*qcat][1] += 1;
        }
        tgt_slots[*qcat] += tgt;
        tot_slots[*qcat] += TOP_K as u32;
    }
    let mut recall = [0f32; 8];
    let mut precision = [0f32; 8];
    let mut contamination = [0f32; 8];
    let q_per_cat = queries.len() as f32 / n_q_per_cat as f32;
    for c in 0..CAT_NAMES.len() {
        recall[c] = hit_q[c][0] as f32 / q_per_cat;
        precision[c] = tgt_slots[c] as f32 / tot_slots[c] as f32;
        contamination[c] = hit_q[c][1] as f32 / hit_q[c][2] as f32;
    }
    let q_all = queries.len() as f32;
    let tgt_q_all: u32 = hit_q.iter().map(|h| h[0]).sum();
    BackendScores {
        recall,
        precision,
        contamination,
        overall_recall: tgt_q_all as f32 / q_all,
        overall_precision: tgt_slots.iter().sum::<u32>() as f32
            / tot_slots.iter().sum::<u32>() as f32,
    }
}

/// The shared RRF fusion rule (round 6, pre-registered; round 7 adds the
/// admission cap, also pre-registered): given each channel's ranked hit ids
/// (best first) and the per-store id→record-index maps, return the fused
/// top-TOP_K record indices. One rule, one place — used by eval_fused and by
/// embed-verify's fused probe so the graded pipeline and the generalization
/// probe cannot drift apart.
pub(crate) fn fuse_top5(
    ids_s: &[u32],
    ids_h: &[u32],
    sem_id_to_idx: &std::collections::HashMap<u32, usize>,
    hash_id_to_idx: &std::collections::HashMap<u32, usize>,
) -> Vec<u32> {
    // idx → (fused score, rank_S, rank_H); absent rank = usize::MAX
    let mut votes: std::collections::HashMap<u32, (f64, usize, usize)> =
        std::collections::HashMap::new();
    for (r, id) in ids_s.iter().enumerate() {
        if let Some(&idx) = sem_id_to_idx.get(id) {
            let e = votes
                .entry(idx as u32)
                .or_insert((0.0, usize::MAX, usize::MAX));
            e.0 += 1.0 / (FUSE_ALPHA + (r + 1) as f64);
            e.1 = e.1.min(r + 1);
        }
    }
    for (r, id) in ids_h.iter().enumerate() {
        if let Some(&idx) = hash_id_to_idx.get(id) {
            let e = votes
                .entry(idx as u32)
                .or_insert((0.0, usize::MAX, usize::MAX));
            e.0 += 1.0 / (FUSE_ALPHA + (r + 1) as f64);
            e.2 = e.2.min(r + 1);
        }
    }
    let mut cand: Vec<(f64, usize, usize, u32)> = votes
        .into_iter()
        .map(|(idx, (s, rs, rh))| (s, rs, rh, idx))
        .collect();
    cand.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    // Round-7 admission cap: a candidate with no hash support (absent from
    // channel H's top-N → rank_H == usize::MAX) may take at most
    // FUSE_SEM_ONLY_MAX of the final slots; overflow passes to the next
    // hash-supported candidate. Deterministic, same order as the score sort.
    let mut out: Vec<u32> = Vec::with_capacity(TOP_K);
    let mut sem_only_used = 0usize;
    for (_, _, rh, idx) in cand {
        if out.len() == TOP_K {
            break;
        }
        if rh == usize::MAX {
            if sem_only_used == FUSE_SEM_ONLY_MAX {
                continue;
            }
            sem_only_used += 1;
        }
        out.push(idx);
    }
    out
}

/// The fused two-channel store (round 6, pre-registered): a sem64 Cma and a
/// hash768 Cma over the same records, candidates keyed by RECORD INDEX (the
/// stores have separate id namespaces; near-dup last-wins). `top5` applies
/// the shared RRF rule.  Used by eval_fused, embed-verify's fused probe, and
/// embed-stress so the graded pipeline and the generalization probes cannot
/// drift apart.
pub(crate) struct FusedStore {
    cma_sem: Cma,
    cma_hash: Cma,
    sem_id_to_idx: std::collections::HashMap<u32, usize>,
    hash_id_to_idx: std::collections::HashMap<u32, usize>,
    idx_tag: std::collections::HashMap<u32, (usize, usize)>,
}

impl FusedStore {
    pub(crate) fn new(
        sem_fn: &dyn Fn(&str) -> Vec<f32>,
        hash_fn: &dyn Fn(&str) -> Vec<f32>,
        records: &[(String, (usize, usize))],
    ) -> FusedStore {
        let mut cma_sem = Cma::new(SEMANTIC_DIM);
        let mut cma_hash = Cma::new(HASH_DIM);
        let mut sem_id_to_idx: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        let mut hash_id_to_idx: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        let mut idx_tag: std::collections::HashMap<u32, (usize, usize)> =
            std::collections::HashMap::new();
        for (i, (text, tag)) in records.iter().enumerate() {
            let s = cma_sem.remember(&sem_fn(text), text, "note", 0);
            let h = cma_hash.remember(&hash_fn(text), text, "note", 0);
            assert!(s >= 0 && h >= 0, "fused: remember failed for {text:?}");
            sem_id_to_idx.insert(s as u32, i);
            hash_id_to_idx.insert(h as u32, i);
            idx_tag.insert(i as u32, *tag);
        }
        FusedStore {
            cma_sem,
            cma_hash,
            sem_id_to_idx,
            hash_id_to_idx,
            idx_tag,
        }
    }

    /// Distinct in-category families with at least one record reachable
    /// through one channel's store (the near-dup gate can collapse records;
    /// callers assert none of the graded families disappeared).
    pub(crate) fn present_families(
        &self,
        records: &[(String, (usize, usize))],
        sem: bool,
        n_cats: usize,
    ) -> usize {
        let map = if sem { &self.sem_id_to_idx } else { &self.hash_id_to_idx };
        let mut fam = std::collections::HashSet::new();
        for &idx in map.values() {
            let tag = records[idx].1;
            if tag.0 < n_cats {
                fam.insert(tag);
            }
        }
        fam.len()
    }

    /// Fused top-TOP_K record indices for a query (both vectors precomputed).
    pub(crate) fn top5(&mut self, sem_q: &[f32], hash_q: &[f32]) -> Vec<u32> {
        let ids_s: Vec<u32> = self
            .cma_sem
            .recall(sem_q, FUSE_TOP_N)
            .into_iter()
            .map(|h| h.id)
            .collect();
        let ids_h: Vec<u32> = self
            .cma_hash
            .recall(hash_q, FUSE_TOP_N)
            .into_iter()
            .map(|h| h.id)
            .collect();
        fuse_top5(&ids_s, &ids_h, &self.sem_id_to_idx, &self.hash_id_to_idx)
    }

    pub(crate) fn idx_tags(&self) -> &std::collections::HashMap<u32, (usize, usize)> {
        &self.idx_tag
    }
}

/// Round-6 two-channel candidate (pre-registered in the plan): reciprocal
/// rank fusion of channel S (the sem64 tower) and channel H (hash-v1 768-dim
/// features recomputed at query time from the raw text the store keeps —
/// zero params, zero new features). score(c) = 1/(α+rank_S) + 1/(α+rank_H)
/// over each channel's top-N. Channel-PRESERVING: a find from either channel
/// keeps its full vote, so the tower's low-lexical-overlap paraphrase finds
/// are never demoted by the lexical channel (a plain hash rerank would).
/// Final list = top-5 by fused score; deterministic tie-break: score desc,
/// rank_S asc, rank_H asc, record index asc.
fn eval_fused(
    sem_fn: &dyn Fn(&str) -> Vec<f32>,
    hash_fn: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, (usize, usize))],
    queries: &[(String, (usize, usize))],
    n_fams: usize,
) -> BackendScores {
    let mut store = FusedStore::new(sem_fn, hash_fn, records);
    // every family must have survived each channel's near-dup gate with at
    // least one reachable record (chatter shares one out-of-range tag)
    let expected: usize = CAT_NAMES.len() * n_fams;
    for (label, sem) in [("sem", true), ("hash", false)] {
        let present = store.present_families(records, sem, CAT_NAMES.len());
        assert_eq!(
            present, expected,
            "fused/{label}: dedup collapsed some families ({present} of {expected} present)"
        );
    }

    // score_queries holds the tag map across the call while the closure
    // mutates the store (recall reinforcement) — pass an owned copy.
    let idx_tags = store.idx_tags().clone();
    score_queries(&idx_tags, queries, &mut |q| store.top5(&sem_fn(q), &hash_fn(q)))
}

/// Diagnostic: how far a query lands from its target family versus sibling
/// families of the same category, in the raw embedding space (before HNSW).
/// Explains a gate failure that the HNSW numbers only hint at.
fn separation(
    embed: &dyn Fn(&str) -> Vec<f32>,
    records: &[(String, (usize, usize))],
    queries: &[(String, (usize, usize))],
) -> (f64, f64) {
    let mut same = 0f64;
    let mut same_n = 0u64;
    let mut cross = 0f64;
    let mut cross_n = 0u64;
    for (qtext, qtag) in queries {
        let q = embed(qtext);
        for (text, tag) in records {
            let t = embed(text);
            let dot: f32 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
            if *tag == *qtag {
                same += dot as f64;
                same_n += 1;
            } else if tag.0 == qtag.0 {
                cross += dot as f64;
                cross_n += 1;
            }
        }
    }
    (same / same_n.max(1) as f64, cross / cross_n.max(1) as f64)
}

// ── budgets (§10.3) ─────────────────────────────────────────────────────
fn artifact_path() -> PathBuf {
    // manifest dir = <root>/crates/ml-train → one pop lands on <root>/crates
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.push("sofuu-core");
    path.push("src");
    path.push("embedding");
    path.push("weights_v1.sem");
    path
}

/// Candidate-artifact override: SOFUU_EMBED_EVAL_ARTIFACT may point at a
/// retrain candidate.  The path is canonicalized and allow-listed to /tmp or
/// the workspace tree — an arbitrary filesystem read is not acceptable.
fn resolve_artifact_path() -> Result<PathBuf, String> {
    let override_path = match std::env::var("SOFUU_EMBED_EVAL_ARTIFACT") {
        Ok(p) => p,
        Err(_) => return Ok(artifact_path()),
    };
    let requested = PathBuf::from(&override_path);
    let canonical = requested
        .canonicalize()
        .map_err(|e| format!("candidate artifact {}: {e}", requested.display()))?;
    let tmp_root = std::path::Path::new("/tmp")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"));
    let manifest_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    if canonical.starts_with(&tmp_root) || canonical.starts_with(&manifest_root) {
        Ok(canonical)
    } else {
        Err(format!(
            "candidate artifact {} is outside the allowed /tmp and workspace roots",
            canonical.display()
        ))
    }
}

/// The artifact under test, generation-dispatched by its magic word:
/// SEM1 (trigram tower, `sofuu_core::embedding`) or SEM2 (trigram tower +
/// learned word table, `embedding_v2`).  The §10 harness is
/// generation-agnostic — it only ever needs `forward_text`.
pub(crate) enum Artifact {
    V1(QuantizedProjector),
    V2(crate::embedding_v2::QuantizedV2),
}

impl Artifact {
    pub(crate) fn forward_text(&self, t: &str) -> [f32; SEMANTIC_DIM] {
        match self {
            Artifact::V1(m) => m.forward(&hash_v1_features(t)),
            Artifact::V2(m) => {
                m.forward(&hash_v1_features(t), &crate::embedding_v2::tokenize(t))
            }
        }
    }
}

/// The artifact under test.  When SOFUU_EMBED_EVAL_ARTIFACT is unset, this
/// is the in-tree artifact the runtime bakes.
pub(crate) struct SemanticModel {
    pub bytes: Vec<u8>,
    pub model: Artifact,
    pub kind: &'static str,
    pub id: String,
    pub path: PathBuf,
    pub blob_version: u32,
}

pub(crate) fn load_semantic_model() -> Result<SemanticModel, String> {
    let path = resolve_artifact_path()?;
    let bytes = std::fs::read(&path)
        .map_err(|e| format!("artifact missing at {}: {e}", path.display()))?;
    let magic = if bytes.len() >= 4 {
        u32::from_le_bytes(bytes[0..4].try_into().unwrap())
    } else {
        0
    };
    let (model, kind) = if magic == crate::embedding_v2::V2_MAGIC {
        let m = crate::embedding_v2::QuantizedV2::from_blob(&bytes)
            .map_err(|e| format!("SEM2 artifact invalid: {e}"))?;
        (Artifact::V2(m), "SEM2")
    } else {
        let m = QuantizedProjector::from_blob(&bytes)
            .map_err(|e| format!("artifact invalid: {e}"))?;
        (Artifact::V1(m), "SEM1")
    };
    let blob_version = if bytes.len() >= 8 {
        u32::from_le_bytes(bytes[4..8].try_into().unwrap())
    } else {
        0
    };
    Ok(SemanticModel {
        id: artifact_id_for(&bytes),
        model,
        kind,
        bytes,
        path,
        blob_version,
    })
}

/// Semantic embed fn for a loaded model (unit 64-dim outputs).  When
/// SOFUU_EMB_ANCHOR is set (> 0), the hybrid lens from embedding_anchor.rs
/// is applied — the same blend the trainer used, so a candidate trained
/// with an anchor channel is graded exactly as it would deploy.
pub(crate) fn semantic_embed(m: &SemanticModel) -> impl Fn(&str) -> Vec<f32> + '_ {
    let (k, w) = crate::embedding_anchor::anchor_cfg();
    let mat = if k > 0 {
        crate::embedding_anchor::anchor_matrix(k)
    } else {
        Vec::new()
    };
    move |t: &str| {
        let mut f = m.model.forward_text(t).to_vec();
        if k == 0 {
            f
        } else {
            let x = hash_v1_features(t);
            crate::embedding_anchor::unit(&mut f);
            let a = crate::embedding_anchor::anchor_forward(&x, &mat, k);
            crate::embedding_anchor::blend(&f, &a, k, w)
        }
    }
}

pub(crate) struct EvalCorpus {
    pub records: Vec<(String, (usize, usize))>,
    pub queries: Vec<(String, (usize, usize))>,
    pub n_fams: usize,
}

/// Shared by `embed-eval` (acceptance gates) and `embed-stress`
/// (variant/typo probes reuse the same queries).
pub(crate) fn build_corpus() -> EvalCorpus {
    let cats = vec![
        paraphrase(),
        facts(),
        documentation(),
        code(),
        paths(),
        errors(),
        versions(),
        dates(),
    ];
    let n_fams = cats[0].families.len();
    debug_assert!(cats.iter().all(|c| c.families.len() == n_fams));
    let mut records: Vec<(String, (usize, usize))> = chatter()
        .into_iter()
        .map(|t| (t, CHATTER_TAG))
        .collect();
    let mut queries: Vec<(String, (usize, usize))> = Vec::new();
    for (ci, cat) in cats.iter().enumerate() {
        for (fi, fam) in cat.families.iter().enumerate() {
            for m in &fam.memories {
                records.push((m.clone(), (ci, fi)));
            }
            for q in &fam.queries {
                queries.push((q.clone(), (ci, fi)));
            }
        }
    }
    EvalCorpus {
        records,
        queries,
        n_fams,
    }
}

/// Failure-mining output consumed by the trainer: one line per category,
/// `name<TAB>semantic_recall<TAB>hash_recall`.  The trainer boosts the
/// failing categories' family counts for the next candidate.
fn write_mining(scores_sem: &BackendScores, scores_hash: &BackendScores) {
    let mut out = String::from("# name\tsem_r5\thash_r5\n");
    let mut c = 0usize;
    while c < CAT_NAMES.len() {
        out.push_str(&format!(
            "{}\t{:.4}\t{:.4}\n",
            CAT_NAMES[c], scores_sem.recall[c], scores_hash.recall[c]
        ));
        c += 1;
    }
    let _ = std::fs::write("/tmp/sofuu_embed_mining.tsv", out);
}

struct Budgets {
    payload: u64,
    init_ms: f64,
    p95_ms: f64,
}

fn measure_budgets(m: &SemanticModel) -> Result<Budgets, String> {
    let payload = m.bytes.len() as u64;
    let unit = "fix the login timeout bug in the scheduler pool ";
    let big = unit.repeat(8192 / unit.len() + 1);
    let big = &big[..8192]; // exactly 8 KiB of input, per the plan's bar
    let t0 = Instant::now();
    let first = m.model.forward_text(big);
    let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(first.len(), SEMANTIC_DIM);
    let mut samples = Vec::with_capacity(P95_SAMPLES);
    for i in 0..P95_SAMPLES {
        let probe = format!("{big} variation {}", i % 7);
        let t = Instant::now();
        let _ = m.model.forward_text(&probe);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = samples[(P95_SAMPLES * 95) / 100];
    Ok(Budgets { payload, init_ms, p95_ms: p95 })
}

// ── driver ──────────────────────────────────────────────────────────────

pub fn run_eval() -> i32 {
    println!("PLAN-TINY-SEMANTIC-EMBEDDER §10 acceptance harness");
    let sm = match load_semantic_model() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let twochannel = std::env::var("SOFUU_EMBED_EVAL_MODE").as_deref() == Ok("twochannel");
    let override_note = if std::env::var("SOFUU_EMBED_EVAL_ARTIFACT").is_ok() {
        " (candidate override)"
    } else {
        ""
    };
    let candidate_note = if twochannel {
        " | candidate: two-channel RRF (S sem64 + H hash768, α=10 N=20, sem-only≤{FUSE_SEM_ONLY_MAX})"
    } else {
        ""
    };
    println!(
        "env: {} {} | mode: {} | artifact: {} ({} v{}, {}) {}{}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(debug_assertions) { "debug (budgets are advisory here — the §10.3 bars are for release builds)" } else { "release" },
        MODEL_ID,
        sm.kind,
        sm.blob_version,
        sm.id,
        override_note,
        candidate_note,
    );

    let budgets = match measure_budgets(&sm) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("harness setup failed: {e}");
            return 2;
        }
    };

    let corpus = build_corpus();
    let records = corpus.records;
    let queries = corpus.queries;
    let n_fams = corpus.n_fams;
    let sem_fn = semantic_embed(&sm);
    println!(
        "corpus: {} records ({} chatter distractors) + {} queries, top-{} via Cma::recall",
        records.len(),
        records.len() - CAT_NAMES.len() * n_fams * 3, // 3 memories per family
        queries.len(),
        TOP_K
    );

    // how many records actually survived each backend's near-dup gate?
    {
        let probe = |dim: usize, embed: &dyn Fn(&str) -> Vec<f32>| {
            let mut cma = Cma::new(dim);
            for (text, _) in &records {
                cma.remember(&embed(text), text, "note", 0);
            }
            cma.len()
        };
        println!(
            "near-dup survival: hash {} of {} records, semantic {} of {}",
            probe(HASH_DIM, &hash_v1_features),
            records.len(),
            probe(SEMANTIC_DIM, &sem_fn),
            records.len()
        );
    }

    let semantic_scores =
        eval_backend("semantic", SEMANTIC_DIM, &sem_fn, &records, &queries, n_fams);
    let hash_scores = eval_backend("hash-v1", HASH_DIM, &hash_v1_features, &records, &queries, n_fams);

    println!("\ncategory       hash R@5  sem R@5   delta   sem P@5  sem hardneg-contam");
    for c in 0..CAT_NAMES.len() {
        println!(
            "{:<14} {:<9.3} {:<9.3} {:<+7.3} {:<8.3} {:.3}",
            CAT_NAMES[c],
            hash_scores.recall[c],
            semantic_scores.recall[c],
            semantic_scores.recall[c] - hash_scores.recall[c],
            semantic_scores.precision[c],
            semantic_scores.contamination[c],
        );
    }
    println!(
        "{:<14} {:<9.3} {:<9.3} {:<+7.3} {:<8.3} {:.3}",
        "OVERALL",
        hash_scores.overall_recall,
        semantic_scores.overall_recall,
        semantic_scores.overall_recall - hash_scores.overall_recall,
        semantic_scores.overall_precision,
        semantic_scores.contamination.iter().sum::<f32>() / CAT_NAMES.len() as f32,
    );

    // Round-6: when SOFUU_EMBED_EVAL_MODE=twochannel, the candidate graded
    // by the §10.1 gates is the fused two-channel pipeline (pre-registered
    // in the plan). The pure-sem table above still prints for transparency;
    // corpus, metrics, and gate thresholds are untouched.
    let graded = if twochannel {
        let fused = eval_fused(&sem_fn, &hash_v1_features, &records, &queries, n_fams);
        println!(
            "\nfused two-channel candidate (RRF α={FUSE_ALPHA} N={FUSE_TOP_N}, sem-only≤{FUSE_SEM_ONLY_MAX}, top-{TOP_K}):"
        );
        println!("category       hash R@5  fused R@5  delta   fused P@5  fused hardneg-contam");
        for c in 0..CAT_NAMES.len() {
            println!(
                "{:<14} {:<9.3} {:<9.3} {:<+7.3} {:<9.3} {:.3}",
                CAT_NAMES[c],
                hash_scores.recall[c],
                fused.recall[c],
                fused.recall[c] - hash_scores.recall[c],
                fused.precision[c],
                fused.contamination[c],
            );
        }
        println!(
            "{:<14} {:<9.3} {:<9.3} {:<+7.3} {:<9.3} {:.3}",
            "OVERALL",
            hash_scores.overall_recall,
            fused.overall_recall,
            fused.overall_recall - hash_scores.overall_recall,
            fused.overall_precision,
            fused.contamination.iter().sum::<f32>() / CAT_NAMES.len() as f32,
        );
        // Informational: the fused pipeline runs BOTH embeds per query.
        // §10.3 budgets gate the artifact, not the pipeline — the extra
        // compute is reported here, never hidden.
        let unit = "fix the login timeout bug in the scheduler pool ";
        let big = unit.repeat(8192 / unit.len() + 1);
        let big = &big[..8192];
        let mut samples = Vec::with_capacity(P95_SAMPLES);
        for i in 0..P95_SAMPLES {
            let probe = format!("{big} variation {}", i % 7);
            let t = Instant::now();
            let a = sem_fn(&probe);
            let b = hash_v1_features(&probe);
            std::hint::black_box((&a, &b));
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  fused pipeline p95 (both embeds, 8 KiB input, {} samples): {:.3} ms (informational)",
            P95_SAMPLES,
            samples[(P95_SAMPLES * 95) / 100]
        );
        fused
    } else {
        semantic_scores.clone()
    };

    let (h_same, h_cross) = separation(&hash_v1_features, &records, &queries);
    let (s_same, s_cross) = separation(&sem_fn, &records, &queries);
    println!("\nraw-space separation (cosine, query→own family vs →sibling families):");
    println!(
        "  hash-v1:  same {:.3} / cross {:.3} (gap {:+.3})",
        h_same,
        h_cross,
        h_same - h_cross
    );
    println!(
        "  semantic: same {:.3} / cross {:.3} (gap {:+.3})",
        s_same,
        s_cross,
        s_same - s_cross
    );

    write_mining(&semantic_scores, &hash_scores);

    // Per-query miss listing for training diagnostics (SOFUU_EMBED_EVAL_DEBUG=1).
    if std::env::var("SOFUU_EMBED_EVAL_DEBUG").as_deref() == Ok("1") {
        println!("\nmissed semantic queries (category / query):");
        let mut id_tag: std::collections::HashMap<u32, (usize, usize)> =
            std::collections::HashMap::new();
        let mut cma = Cma::new(SEMANTIC_DIM);
        for (text, tag) in &records {
            let id = cma.remember(&sem_fn(text), text, "note", 0);
            id_tag.insert(id as u32, *tag);
        }
        for (qtext, qtag) in &queries {
            let hits5 = cma.recall(&sem_fn(qtext), TOP_K);
            let hit = hits5
                .iter()
                .any(|h| id_tag.get(&h.id) == Some(qtag));
            if !hit {
                println!("  {:<14} {}", CAT_NAMES[qtag.0], qtext);
                for h in &hits5 {
                    let t = id_tag
                        .get(&h.id)
                        .map(|t| {
                            format!(
                                "{},{}",
                                CAT_NAMES.get(t.0).copied().unwrap_or("chatter"),
                                t.1
                            )
                        })
                        .unwrap_or_else(|| "chatter".to_string());
                    println!(
                        "      -> [{t}] score {:.3} {}",
                        h.score,
                        if id_tag.get(&h.id) == Some(qtag) { " <== own" } else { "" }
                    );
                }
            }
        }
    }

    // ── §10.1 gates ── (thresholds verbatim from the plan; they read the
    // graded candidate — the fused pipeline in twochannel mode)
    let mut failed: Vec<String> = Vec::new();
    let g1a = graded.recall[0] >= G1_PARAPHRASE_MIN;
    let g1b = graded.recall[0] >= hash_scores.recall[0] + G1_MARGIN;
    if !(g1a && g1b) {
        failed.push(format!(
            "(1) paraphrase: candidate R@5 {:.3} vs min {:.2} / hash {:.3} +10pp",
            graded.recall[0], G1_PARAPHRASE_MIN, hash_scores.recall[0]
        ));
    }
    if graded.recall[1] < G2_FACTS_MIN {
        failed.push(format!(
            "(2) facts: candidate R@5 {:.3} < {:.2}",
            graded.recall[1], G2_FACTS_MIN
        ));
    }
    for c in [3usize, 4, 5, 6] {
        if graded.recall[c] < hash_scores.recall[c] - G3_MARGIN {
            failed.push(format!(
                "(3) exact retrieval [{}]: candidate {:.3} < hash {:.3} - 1pp",
                CAT_NAMES[c], graded.recall[c], hash_scores.recall[c]
            ));
        }
    }
    if graded.overall_precision < hash_scores.overall_precision {
        failed.push(format!(
            "(4) hard-negative precision: candidate P@5 {:.3} < hash {:.3}",
            graded.overall_precision, hash_scores.overall_precision
        ));
    }
    if graded.overall_recall < hash_scores.overall_recall - G5_MARGIN {
        failed.push(format!(
            "(5) end-to-end recall (Cma::recall candidate-set proxy): candidate {:.3} < hash {:.3} - 1pp",
            graded.overall_recall, hash_scores.overall_recall
        ));
    }
    let wins = (0..CAT_NAMES.len())
        .filter(|&c| graded.recall[c] > hash_scores.recall[c])
        .count();
    if wins < G6_MIN_WINS {
        failed.push(format!(
            "(6) category wins: {wins} < {G6_MIN_WINS} of 8"
        ));
    }

    // ── §10.3 budgets ──
    let budget_payload = budget_payload_bytes();
    let budget_note = if std::env::var("SOFUU_EMB_BUDGET_KIB").is_ok() {
        " (experimental override)"
    } else {
        ""
    };
    println!("\nbudgets (§10.3):");
    println!(
        "  payload: {} B (limit {} B{}) {}",
        budgets.payload,
        budget_payload,
        budget_note,
        if budgets.payload <= budget_payload { "OK" } else { "FAIL" }
    );
    println!(
        "  init (bake + first 8 KiB forward): {:.2} ms (limit {:.0} ms) {}",
        budgets.init_ms,
        BUDGET_INIT_MS,
        if budgets.init_ms <= BUDGET_INIT_MS { "OK" } else { "FAIL" }
    );
    println!(
        "  p95 forward, 8 KiB input ({} samples): {:.3} ms (limit {:.0} ms reference) {}",
        P95_SAMPLES,
        budgets.p95_ms,
        BUDGET_P95_MS,
        if budgets.p95_ms <= BUDGET_P95_MS { "OK" } else { "FAIL" }
    );
    println!(
        "  migrated vector stream: {} f32 vs hash {} f32 = 1/{:.1} (informational)",
        SEMANTIC_DIM,
        HASH_DIM,
        HASH_DIM as f32 / SEMANTIC_DIM as f32
    );
    if budgets.payload > budget_payload {
        failed.push(format!(
            "budget payload {} > {} B{}",
            budgets.payload, budget_payload, budget_note
        ));
    }
    if budgets.init_ms > BUDGET_INIT_MS {
        failed.push(format!("budget init {:.2} > {:.0} ms", budgets.init_ms, BUDGET_INIT_MS));
    }
    if budgets.p95_ms > BUDGET_P95_MS {
        failed.push(format!("budget p95 {:.3} > {:.0} ms", budgets.p95_ms, BUDGET_P95_MS));
    }

    if failed.is_empty() {
        println!("\nVERDICT: PASS — all §10.1 gates and §10.3 budgets met. Cutover may proceed.");
        0
    } else {
        println!("\nVERDICT: FAIL — {} gate(s)/budget(s) failed:", failed.len());
        for f in &failed {
            println!("  - {f}");
        }
        println!("Per the plan: do NOT replace the current embedder; keep the candidate as an offline experiment.");
        1
    }
}
