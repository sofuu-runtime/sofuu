// ml-train/src/data_relevance.rs — synthetic dataset for the relevance gate
// (PLAN-ML-GATES §6/§9).
//
// Each example is ONE target candidate inside a constructed candidate set
// (task + menu of files/memories/search results + already-kept + recent
// visible text). Label 1 = deserves the tokens (use), 0 = don't pull it in.
// Labels are MECHANICAL — they follow from construction, organized around
// §9's decision classes so the net can only pass by learning the signals:
//
//   USE:  definition site of the task-named symbol; synonym/needle match
//         (low lexical, high semantic); changelog when the task is about
//         versions; test when the task is about behaviour; direct on-task.
//   SKIP: unrelated call-site mention; license/generated/lockfile/minified;
//         near-duplicate of an already-kept candidate; wrong-topic trap
//         (high lexical, low semantic); off-task; already visible; changelog
//         when the task is NOT about versions; test for a typo fix.
//
// The last group (task-dependent labels) is where parameters earn their keep:
// the label flips with the TASK, not the candidate — no threshold on a single
// feature can express it.
//
// Channel-coverage rule (the freshness lesson): every feature channel is
// ACTIVE in training. Holdout families test unseen CONSTRUCTIONS of trained
// channels (and unseen source domains — web/memory-shaped candidates), never
// an untrained channel.

use sofuu_core::ml::relevance::features::{
    extract_all, CandKind, CandidateInput, RelevanceContext,
};

use crate::train::Example;

/* ── Slot banks ──────────────────────────────────────────────────── */

/// Disjoint domains. Target content and task draw from the SAME domain for
/// the on-task families and DIFFERENT domains for the off-task families.
const DOMAINS: &[(&str, &[&str])] = &[
    ("database migration", &["database", "migration", "schema", "tables", "columns", "indexes"]),
    ("auth sessions", &["auth", "session", "token", "login", "cookie", "credentials"]),
    ("rate limiting", &["rate", "limit", "throttle", "requests", "bucket", "burst"]),
    ("web scraper", &["scraper", "crawl", "pages", "html", "links", "fetch"]),
    ("cache eviction", &["cache", "eviction", "entries", "lru", "ttl", "hits"]),
    ("log pipeline", &["logs", "pipeline", "ingest", "shards", "batches", "retention"]),
];

const SYMBOLS: &[&str] = &[
    "parse_config", "TokenBucket", "evict_expired", "walk_links", "apply_schema",
    "session_from_cookie", "RateGate", "compact_batch", "ShardWriter", "build_index",
];

const PATHS: &[&str] = &[
    "src/config.rs", "src/auth/session.rs", "src/limit/throttle.rs", "src/scrape/walker.rs",
    "src/cache/lru.rs", "src/logs/shipper.rs", "crates/core/engine.rs", "tools/gen_schema.py",
    "config/limits.toml", "tests/pipeline_test.rs",
];

/// General on-task phrasings (name a symbol + its domain).
const TASKS_GENERAL: &[&str] = &[
    "fix the {s} function in the {d} module",
    "why does {s} return the wrong result in {d}",
    "refactor {s} without breaking the {d} callers",
    "debug the regression in {s} used by {d}",
    "add error handling to {s} in the {d} code",
    "explain how {s} works in the {d} path",
];

/// Tasks about versions/upgrades (make changelogs useful).
const TASKS_VERSION: &[&str] = &[
    "what changed in the latest {d} release",
    "which {d} version introduced the breaking change",
    "list the deprecated {d} APIs before we upgrade",
    "summarize the {d} changelog for the upgrade",
];

/// Tasks about behaviour (make tests useful).
const TASKS_BEHAVIOUR: &[&str] = &[
    "what does {s} do when the {d} input is empty",
    "how does {s} behave under concurrent {d} load",
    "does {s} handle the edge case in {d} correctly",
    "what is the expected behaviour of {s} for {d}",
];

/// Typo/formatting tasks (tests are NOT useful here).
const TASKS_TYPO: &[&str] = &[
    "fix the typo in the {s} doc comment",
    "rename the misspelled variable in {s}",
    "format the {s} file with the standard style",
    "correct the grammar in the {d} readme",
];

/// Goal tasks — name the DOMAIN goal but NOT a code symbol. Used by the
/// synonym/needle families so the answer can echo the domain words
/// morphologically without a shared symbol token dominating the match.
const TASKS_GOAL: &[&str] = &[
    "finish the {d} work",
    "get the {d} task done",
    "what is left to complete the {d} effort",
    "wrap up the remaining {d} work",
    "how do we complete the {d} step",
];

/// Off-task distractor tasks (for the wrong-topic and off-task families the
/// target content is on THIS domain while the task is on another).
const RECENT_GENERIC: &[&str] = &[
    "let us keep going with the current approach and re-run the suite",
    "the last change compiled cleanly, next step is the integration check",
    "I pushed the fix we discussed, please review it when ready",
    "sounds good, continue with the next item on the list",
    "nothing blocked on my side, carry on with the implementation",
];

/// Filler distractor candidates (off-topic, low-signal) that populate every
/// candidate set so BM25 has a real collection to rank against.
const FILLER_CANDS: &[&str] = &[
    "meeting notes from last sprint: priorities unchanged, next review on friday",
    "an old design memo about the previous architecture, mostly superseded",
    "the cafeteria menu for this week, unrelated to the engineering work",
    "a draft blog post announcing the team offsite, still in review",
    "housekeeping: the ci runner was restarted this morning, no action needed",
    "a weather report snapshot archived by the monitoring job",
];

/// Never-use class content (license / generated / lockfile / minified).
const NEVER_USE: &[&str] = &[
    "permission is hereby granted, free of charge, to any person obtaining a copy \
     of this software. the software is provided as is, without warranty of any kind. \
     copyright notice shall be included in all copies. all rights reserved.",
    "// code generated by protoc-gen. do not edit. this file is generated \
     automatically from the schema definition. machine generated output.",
    "lockfile maintained by the package manager. checksum sha256 recorded for \
     every dependency. do not edit by hand. package-lock version three.",
    "var _0x4a2b=function(a,b){return a+b};_0x4a2b(1,2);minified bundle output \
     with no source mapping available for this artifact.",
];

/// Changelog / version content (useful only for version tasks).
const CHANGELOG: &[&str] = &[
    "changelog version 2.0: breaking change to the {d} api, deprecated the old \
     constructor, upgrade guide added, semver major bump tagged yesterday",
    "release notes 1.4: the {d} module gained a new option, no breaking changes, \
     version bumped, see the migration guide for details",
];

/// Test content (useful only for behaviour tasks).
const TEST_CONTENT: &[&str] = &[
    "#[test] fn {s}_empty_input() {{ let out = {s}(\"\"); assert!(out.is_empty()); }} \
     #[test] fn {s}_happy_path() {{ assert_eq!({s}(\"x\"), \"x\"); }} unit test suite",
    "describe(\"{s}\", () => {{ it(\"should handle the {d} edge case\", () => {{ \
     expect({s}(null)).toBeUndefined(); }}); }}); test case spec",
];

/// Synonym/needle content: answers the task but phrased in DIFFERENT word
/// forms so exact-word overlap stays partial while morphological similarity
/// is dense — every domain stem appears in inflected form (migrates↔
/// migration, evicts↔eviction, throttles↔throttle). This is the gap the
/// stem channel + char-trigram cosine exploit that a flat exact-word
/// overlap threshold cannot. Keyed by domain index.
const SYNONYM_ANSWERS: &[&str] = &[
    // database migration
    "the routine migrates the storage schema: it rewrites every table, \
     redefines the columns, and rebuilds the indexes before swapping the \
     database live",
    // auth sessions
    "the auth check validates the login credentials, verifies the session \
     badge against the clock, and rejects the request once the token has \
     lapsed",
    // rate limiting
    "the governor refills the quota bucket on a fixed cadence and throttles \
     any request that arrives after the rate limit for the burst is \
     exhausted",
    // web scraper
    "the scraper crawls each page, follows the links it finds, and stores \
     the html markup until the crawl frontier is empty",
    // cache eviction
    "the pruner evicts the least recently used cache entries whose ttl has \
     expired, making room in the lru for fresh data",
    // log pipeline
    "the pipeline ingests the logs in batches and ships each shard to \
     storage once the retention window closes",
];

/// SECOND synonym bank — fresh morphological echoes used by the TRAINING
/// synonym family (R29), so the stem channel sees two independent phrasings
/// of each domain's needle and does not memorize bank 1; the test-fold
/// synonym (R15) stays a fresh construction.
const SYNONYM_ANSWERS_2: &[&str] = &[
    // database migration — echoes the "migrat" stem + schema nouns
    "the job migrates the database to its new layout: every table is \
     rewritten, the columns are remapped, and the indexes are rebuilt \
     before the cutover",
    // auth sessions — echoes auth/login/session/token/credential stems
    "the auth guard re-checks the login credential, revalidates the session \
     against the issued token, and turns away the request once it lapses",
    // rate limiting — echoes rate/limit/bucket/throttle stems
    "the limiter refills the bucket on schedule and throttles any request \
     that arrives after the rate limit runs out",
    // web scraper — echoes scrap/crawl/page/link/html stems
    "the scraper walks the pages one by one, keeps the html it needs, and \
     queues the links until the crawl is done",
    // cache eviction — echoes evict/cache/lru/ttl stems
    "the sweeper evicts the cache entries that aged out of the lru once \
     their ttl is past, freeing slots for new data",
    // log pipeline — echoes pipeline/log/shard/retention stems
    "the pipeline batches the logs and moves each shard into storage when \
     the retention period ends",
];

/* ── Candidate + example construction ───────────────────────────── */

#[derive(Clone)]
struct Cand {
    text: String,
    kind: CandKind,
    strength: f32,
    role: u8,
    path: String,
}

fn cfile(text: &str, path: &str) -> Cand {
    Cand { text: text.to_string(), kind: CandKind::File, strength: 0.0, role: 0, path: path.to_string() }
}
fn cother(text: &str) -> Cand {
    Cand { text: text.to_string(), kind: CandKind::Other, strength: 0.0, role: 0, path: String::new() }
}
fn cweb(text: &str) -> Cand {
    Cand { text: text.to_string(), kind: CandKind::Web, strength: 0.0, role: 0, path: String::new() }
}
fn cmem(text: &str, strength: f32, role: u8) -> Cand {
    Cand { text: text.to_string(), kind: CandKind::Memory, strength, role, path: String::new() }
}

fn fill(tpl: &str, d: &str, s: &str) -> String {
    tpl.replace("{d}", d).replace("{s}", s)
}

fn tokens_of(text: &str) -> u32 {
    (text.len() as f32 / 4.0).ceil() as u32
}

/// One training example: a candidate set plus the target candidate's index.
/// The set is rotated by a deterministic per-example hash so the target
/// lands at a VARIED position — in production the retriever's order is
/// arbitrary, so the position channel must not stay constant-last (a
/// constant feature the net could fold into its bias and then misread at
/// serve time). Kept indices are remapped through the rotation.
fn make(
    task: &str,
    recent: &str,
    cands: Vec<Cand>,
    kept: Vec<usize>,
    target: usize,
    y: f32,
    group: u32,
) -> Example {
    let len = cands.len();
    let mut h: u32 = group.wrapping_mul(0x9E37_79B9);
    for b in task.as_bytes() {
        h = h.wrapping_mul(31).wrapping_add(*b as u32);
    }
    for b in cands[target].text.as_bytes() {
        h = h.wrapping_mul(31).wrapping_add(*b as u32);
    }
    let rot = if len > 1 { (h as usize) % len } else { 0 };
    let remap = |old: usize| (old + len - rot) % len;
    let rotated: Vec<Cand> = (0..len).map(|new_i| cands[(new_i + rot) % len].clone()).collect();
    let new_target = remap(target);
    let new_kept: Vec<usize> = kept.iter().map(|&i| remap(i)).collect();

    let inputs: Vec<CandidateInput> = rotated
        .iter()
        .map(|c| CandidateInput {
            text: &c.text,
            kind: c.kind,
            strength: c.strength,
            role: c.role,
            path: &c.path,
        })
        .collect();
    let ctx = RelevanceContext { task, recent, candidates: &inputs, kept: &new_kept };
    let feats = extract_all(&ctx);
    let _ = tokens_of; // reserved for future size-aware families
    Example {
        x: feats[new_target].to_vec(),
        y,
        group,
        text: rotated[new_target].text.clone(),
        task: task.to_string(),
    }
}

/// A realistic distractor set: 1–4 off-topic fillers (set size VARIES so
/// the set-size channel is trained across production shapes) plus, two
/// thirds of the time, a strong ON-TASK competitor file that even mentions
/// the task symbol. The retriever's menu already looks plausible — the
/// gate must separate within a competitive set where BM25 rank/score take
/// INTERIOR values (a competitor that matches several query words pushes
/// one-word matches into the 0.2–0.4 normalized band), not only against
/// strawman fillers (the bimodal-BM25 pathology that let an early bake
/// mis-score sets with several task-matching candidates).
fn distractors(di: usize, k: usize) -> Vec<Cand> {
    let (domain, words) = DOMAINS[di];
    let w = words[k % words.len()];
    let sym = SYMBOLS[(di + k) % SYMBOLS.len()];
    let mut out = Vec::new();
    let n_fillers = 1 + (k % 4);
    for j in 0..n_fillers {
        let text = FILLER_CANDS[(di * 3 + k + j) % FILLER_CANDS.len()];
        out.push(cother(text));
    }
    if k % 3 != 0 {
        let comp = format!(
            "overview of the {domain} module: the {w} layout, the {w} \
             configuration, and how {sym} fits into the {w} flow"
        );
        out.push(cfile(&comp, PATHS[(di + k + 5) % PATHS.len()]));
    }
    out
}

/* ── The dataset ─────────────────────────────────────────────────── */

pub fn build() -> Vec<Example> {
    let mut out = Vec::new();

    for (di, (domain, words)) in DOMAINS.iter().enumerate() {
        // A different domain for off-task content. 24 variants per domain:
        // the modulo arithmetic cycles every bank for broad lexical coverage.
        for k in 0..24usize {
            let (odomain, owords) = DOMAINS[(di + 1 + k) % DOMAINS.len()];
            let w = words[k % words.len()];
            let ow = owords[k % owords.len()];
            let sym = SYMBOLS[(di + k) % SYMBOLS.len()];
            let path = PATHS[(di + k) % PATHS.len()];
            let recent = RECENT_GENERIC[(di + k) % RECENT_GENERIC.len()];

            // ── USE families ──────────────────────────────────────────

            // R0 (pos): definition site of the task-named symbol. The task
            // names `sym`; the target is the file that DEFINES it.
            {
                let task = fill(TASKS_GENERAL[(di + k) % TASKS_GENERAL.len()], domain, sym);
                let def = format!("fn {sym} implements the core {w} step of the {domain} \
                                   path and returns the computed {w} result");
                let mut cands = distractors(di, k);
                cands.push(cfile(&def, path));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 0));
            }

            // R1 (pos): synonym/needle — answers the task in DIFFERENT word
            // FORMS: the task names the domain goal, the answer echoes the
            // domain words morphologically (migrates↔migration) so exact-word
            // overlap stays partial while char-trigram similarity holds. A
            // flat word-overlap threshold cannot catch it; the net must
            // combine the morphological cosine with the partial overlap.
            {
                let task = fill(TASKS_GOAL[(di + k + 1) % TASKS_GOAL.len()], domain, sym);
                let needle = SYNONYM_ANSWERS[di % SYNONYM_ANSWERS.len()];
                let mut cands = distractors(di, k + 1);
                cands.push(cother(needle));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 1));
            }

            // R2 (pos): changelog content + version task → use.
            {
                let task = fill(TASKS_VERSION[(di + k) % TASKS_VERSION.len()], domain, sym);
                let log = fill(CHANGELOG[(di + k) % CHANGELOG.len()], domain, sym);
                let mut cands = distractors(di, k + 2);
                cands.push(cfile(&log, "CHANGELOG.md"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 2));
            }

            // R3 (pos): test content + behaviour task → use.
            {
                let task = fill(TASKS_BEHAVIOUR[(di + k) % TASKS_BEHAVIOUR.len()], domain, sym);
                let test = fill(TEST_CONTENT[(di + k) % TEST_CONTENT.len()], domain, sym);
                let mut cands = distractors(di, k + 3);
                cands.push(cfile(&test, "tests/module_test.rs"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 3));
            }

            // R4 (pos): direct on-task content — high BM25 AND high semantic.
            // The easy positive that anchors the use class.
            {
                let task = fill(TASKS_GENERAL[(di + k + 2) % TASKS_GENERAL.len()], domain, sym);
                let direct = format!("the {domain} code around {sym}: the {w} handling \
                                      lives here and the {w} values are computed inline");
                let mut cands = distractors(di, k + 4);
                cands.push(cfile(&direct, path));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 4));
            }

            // ── SKIP families ─────────────────────────────────────────

            // R5 (neg): unrelated call-site mention of the task symbol — the
            // symbol appears but carries no task-relevant information.
            {
                let task = fill(TASKS_GENERAL[(di + k + 3) % TASKS_GENERAL.len()], domain, sym);
                let mention = format!("the {odomain} handler calls {sym} once during \
                                       setup and otherwise deals with {ow} only");
                let mut cands = distractors(di, k + 5);
                cands.push(cfile(&mention, PATHS[(di + k + 3) % PATHS.len()]));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 5));
            }

            // R6 (neg): never-use class (license/generated/lockfile/minified).
            {
                let task = fill(TASKS_GENERAL[(di + k + 4) % TASKS_GENERAL.len()], domain, sym);
                let never = NEVER_USE[(di + k) % NEVER_USE.len()];
                let mut cands = distractors(di, k + 6);
                cands.push(cfile(never, "LICENSE"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 6));
            }

            // R7 (neg): near-duplicate of an already-kept candidate. The kept
            // copy is index 0; the target repeats it verbatim.
            {
                let task = fill(TASKS_GENERAL[(di + k + 5) % TASKS_GENERAL.len()], domain, sym);
                let body = format!("the {domain} implementation of {sym} with the {w} \
                                    step described in detail");
                let mut cands = vec![cfile(&body, path)];
                cands.extend(distractors(di, k + 7));
                cands.push(cfile(&body, path)); // verbatim repeat
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![0], target, 0.0, 7));
            }

            // R8 (neg): wrong-topic trap — ONE of the task's own words
            // appears in a different sense, buried in unrelated scheduling
            // content. The trap word is always a domain-NAME word (one of
            // the first two words of the domain), so the trap is guaranteed
            // to share exactly one query word with the task — the f4/f36
            // ≈ 0.2–0.33 regime must be densely trained as skip, or the
            // net keys on the embedder's noisy cosine instead.
            {
                let task = fill(TASKS_GENERAL[(di + k) % TASKS_GENERAL.len()], domain, sym);
                let tw = words[k % 2];
                let trap = format!("office logistics memo: the {tw} of the team to the \
                                    annex building is scheduled for next month; desks, \
                                    badges, and parking slots are assigned in the attached \
                                    roster. contact facilities for exceptions.");
                let mut cands = distractors(di, k + 8);
                cands.push(cother(&trap));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 8));
            }

            // R9 (neg): off-task — low lexical AND low semantic. The easy
            // negative that anchors the skip class.
            {
                let task = fill(TASKS_GENERAL[(di + k + 1) % TASKS_GENERAL.len()], domain, sym);
                let offtask = format!("a short note about the {odomain} dashboard and \
                                       its {ow} metrics, unrelated to anything else");
                let mut cands = distractors(di, k + 9);
                cands.push(cother(&offtask));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 9));
            }

            // R10 (neg): already visible — the target text is already in the
            // recent history, so re-sending it wastes tokens.
            {
                let task = fill(TASKS_GENERAL[(di + k + 2) % TASKS_GENERAL.len()], domain, sym);
                let visible = format!("the {domain} snippet with the {w} detail we \
                                       already looked at");
                let recent_with = format!("{recent}\n{visible}");
                let mut cands = distractors(di, k + 10);
                cands.push(cother(&visible));
                let target = cands.len() - 1;
                out.push(make(&task, &recent_with, cands, vec![], target, 0.0, 10));
            }

            // R11 (neg): changelog content but the task is NOT about versions
            // (task-dependent flip of R2).
            {
                let task = fill(TASKS_GENERAL[(di + k + 3) % TASKS_GENERAL.len()], domain, sym);
                let log = fill(CHANGELOG[(di + k + 1) % CHANGELOG.len()], domain, sym);
                let mut cands = distractors(di, k + 11);
                cands.push(cfile(&log, "CHANGELOG.md"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 11));
            }

            // R12 (neg): test content but the task is a typo/formatting fix
            // (task-dependent flip of R3).
            {
                let task = fill(TASKS_TYPO[(di + k) % TASKS_TYPO.len()], domain, sym);
                let test = fill(TEST_CONTENT[(di + k + 1) % TEST_CONTENT.len()], domain, sym);
                let mut cands = distractors(di, k + 12);
                cands.push(cfile(&test, "tests/module_test.rs"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 12));
            }

            // ── Held-out constructions (val) ───────────────────────────
            // R13/R14 validate UNSEEN CONSTRUCTIONS of trained channels
            // (a memory-shaped use candidate, a fresh never-use phrasing);
            // the kind channels themselves are trained on R17–R20 below.

            // R13 (val pos): memory-shaped candidate that answers the task.
            {
                let task = fill(TASKS_GENERAL[(di + k + 4) % TASKS_GENERAL.len()], domain, sym);
                let mem = format!("remembered fact: in the {domain} module, {sym} \
                                   computes the {w} result exactly as the task needs");
                let mut cands = distractors(di, k + 13);
                cands.push(cmem(&mem, 0.8, 2));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 13));
            }

            // R14 (val neg): held-out never-use construction (fresh phrasing).
            {
                let task = fill(TASKS_GENERAL[(di + k + 5) % TASKS_GENERAL.len()], domain, sym);
                let gen = format!("this file is generated by the {ow} toolchain. do not \
                                   edit it directly; regenerate it from the {ow} source \
                                   schema instead. auto-generated artifact.");
                let mut cands = distractors(di, k + 14);
                cands.push(cfile(&gen, "src/gen.rs"));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 14));
            }

            // R15 (test pos): web-result-shaped answer (held-out domain). A
            // search snippet that answers a GOAL task, phrased like a result
            // card — the synonym answer echoes the domain words
            // morphologically, so the web kind one-hot + morphological
            // channels must combine (the kind channel is trained on R18).
            {
                let task = fill(TASKS_GOAL[(di + k) % TASKS_GOAL.len()], domain, sym);
                let snippet = SYNONYM_ANSWERS[di % SYNONYM_ANSWERS.len()];
                let web = format!("search result: {snippet} — discussed in the {domain} \
                                   guide with examples and caveats");
                let mut cands = distractors(di, k + 15);
                cands.push(cweb(&web));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 15));
            }

            // R16 (test neg): held-out wrong-topic trap (fresh construction —
            // one domain-NAME word in a different sense, buried in unrelated
            // content; the training analogues are R8/R23/R25). Shares exactly
            // one query word so the f4/f36 ≈ 0.2–0.33 skip regime is what
            // gets measured on the holdout.
            {
                let task = fill(TASKS_GENERAL[(di + k + 1) % TASKS_GENERAL.len()], domain, sym);
                let tw = words[k % 2];
                let trap2 = format!("retrospective writeup: the {tw} question came up \
                                     during the quarterly review; the facilitation notes, \
                                     attendance list, and follow-up owners are recorded \
                                     here for the record.");
                let mut cands = distractors(di, k + 16);
                cands.push(cother(&trap2));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 16));
            }

            // ── Kind-channel coverage in TRAINING (channel-coverage rule) ──
            // The web + memory one-hots must be ACTIVE with correct labels in
            // the training split; val/test then hold out fresh CONSTRUCTIONS
            // (R13/R15), never an untrained channel.

            // R17 (train pos): memory-shaped candidate that answers the task
            // directly (names the symbol + domain words).
            {
                let task = fill(TASKS_GENERAL[(di + k + 2) % TASKS_GENERAL.len()], domain, sym);
                let mem = format!("remembered fact: {sym} lives in the {domain} module \
                                   and computes the {w} result; the {w} step runs before \
                                   the callers see the value");
                let mut cands = distractors(di, k + 17);
                cands.push(cmem(&mem, 0.9, 2));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 17));
            }

            // R18 (train pos): web-result card quoting the definition site —
            // high lexical AND morphological match, web kind.
            {
                let task = fill(TASKS_GENERAL[(di + k + 3) % TASKS_GENERAL.len()], domain, sym);
                let web = format!("search result: how {sym} works in the {domain} path — \
                                   the {w} step explained with a worked example and notes \
                                   on the {w} edge cases");
                let mut cands = distractors(di, k + 18);
                cands.push(cweb(&web));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 18));
            }

            // R19 (train neg): memory candidate about a DIFFERENT domain —
            // kind alone must not earn tokens; content still decides.
            {
                let task = fill(TASKS_GENERAL[(di + k + 4) % TASKS_GENERAL.len()], domain, sym);
                let mem = format!("remembered fact: the {odomain} dashboard tracks the \
                                   {ow} metrics and refreshes them every hour");
                let mut cands = distractors(di, k + 19);
                cands.push(cmem(&mem, 0.7, 2));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 19));
            }

            // R20 (train neg): web result about an unrelated topic — stale
            // marketing content with no task words.
            {
                let task = fill(TASKS_GENERAL[(di + k + 5) % TASKS_GENERAL.len()], domain, sym);
                let web = "search result: top ten tips for planning a team offsite — \
                           venue checklists, catering options, and agenda templates \
                           for large groups";
                let mut cands = distractors(di, k + 20);
                cands.push(cweb(web));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 20));
            }

            // ── Template decorrelation ─────────────────────────────────
            // Every task bank must appear with BOTH labels, or the net can
            // key on the task phrasing itself instead of the task×content
            // relation (fold-2 collapse in CV: held-out goal tasks were
            // unseen with positives, so goal-task ⇒ skip was learned).

            // R21 (train pos): goal task + direct on-task file content.
            {
                let task = fill(TASKS_GOAL[(di + k + 2) % TASKS_GOAL.len()], domain, sym);
                let direct = format!("the {domain} module: the {w} handling lives here, \
                                      the {w} values are computed inline, and the {w} \
                                      step finishes before the callers resume");
                let mut cands = distractors(di, k + 21);
                cands.push(cfile(&direct, path));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 21));
            }

            // R22 (train neg): general task + off-topic content from another
            // domain (general tasks must not read as use-by-phrasing).
            {
                let task = fill(TASKS_GENERAL[(di + k) % TASKS_GENERAL.len()], domain, sym);
                let offtask = format!("a short note about the {odomain} dashboard and \
                                       its {ow} metrics, unrelated to anything else");
                let mut cands = distractors(di, k + 22);
                cands.push(cother(&offtask));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 22));
            }

            // R23 (train neg): goal task + one-word trap (goal tasks must
            // not read as use-by-phrasing; the training analogue of R16).
            {
                let task = fill(TASKS_GOAL[(di + k + 3) % TASKS_GOAL.len()], domain, sym);
                let tw = words[k % 2];
                let trap = format!("office logistics memo: the {tw} of the team to the \
                                    annex building is scheduled for next month; desks, \
                                    badges, and parking slots are assigned in the \
                                    attached roster. contact facilities for exceptions.");
                let mut cands = distractors(di, k + 23);
                cands.push(cother(&trap));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 23));
            }

            // R24 (train neg): version task + off-topic content (version
            // tasks must not read as use-by-phrasing either).
            {
                let task = fill(TASKS_VERSION[(di + k + 1) % TASKS_VERSION.len()], domain, sym);
                let offtask = format!("a short note about the {odomain} dashboard and \
                                       its {ow} metrics, unrelated to anything else");
                let mut cands = distractors(di, k + 24);
                cands.push(cother(&offtask));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 24));
            }

            // R25 (train neg): a SECOND wrong-topic trap framing (vendor
            // invoice mentioning one domain word in a billing sense) so the
            // trap side of the boundary is defined by more than one phrasing.
            {
                let task = fill(TASKS_GENERAL[(di + k + 1) % TASKS_GENERAL.len()], domain, sym);
                let tw = words[k % 2];
                let trap = format!("vendor invoice summary: the {tw} charge appears on \
                                    the march statement; accounting requires the receipt \
                                    before the expense is approved for processing");
                let mut cands = distractors(di, k + 25);
                cands.push(cother(&trap));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 25));
            }

            // ── Partial overlap with already-kept context ──────────────
            // f16 (max cosine to a kept candidate) is only trained at ~1.0
            // (R7 verbatim dups) unless we construct the INTERIOR regime:
            // a target that partially resembles kept context. The label
            // must then ride on content, not on the resemblance.

            // R26 (train neg): the target rephrases part of a kept candidate
            // and adds nothing task-relevant — moderate f16 reads as
            // redundancy, and the weak task match decides: skip.
            {
                let task = fill(TASKS_GENERAL[(di + k + 2) % TASKS_GENERAL.len()], domain, sym);
                let kept_text = format!("the {domain} notes from earlier: the {w} plan \
                                         and the {sym} summary we agreed on");
                let target_text = format!("a follow-up memo that rephrases the {w} plan \
                                           in slightly different words without adding \
                                           any new detail");
                let mut cands = vec![cfile(&kept_text, path)];
                cands.extend(distractors(di, k + 26));
                cands.push(cother(&target_text));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![0], target, 0.0, 26));
            }

            // R27 (train pos): the target builds on a kept candidate but
            // carries NEW task-relevant detail — moderate f16 must not
            // auto-skip; the content match decides: use.
            {
                let task = fill(TASKS_GENERAL[(di + k + 3) % TASKS_GENERAL.len()], domain, sym);
                let kept_text = format!("the {domain} notes from earlier: the {w} plan \
                                         we agreed on last week");
                let target_text = format!("fn {sym} implements the {w} step: the exact \
                                           code the task needs, with the {w} handling \
                                           computed inline");
                let mut cands = vec![cfile(&kept_text, path)];
                cands.extend(distractors(di, k + 27));
                cands.push(cfile(&target_text, PATHS[(di + k + 7) % PATHS.len()]));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![0], target, 1.0, 27));
            }

            // ── Hard-regime augmentation in TRAINING ────────────────────
            // R28/R29 densify the two hardest regimes so the boundary is
            // learned, not memorized: a fresh trap framing (R28) and a
            // second synonym phrasing (R29). R15/R16 stay the fresh TEST
            // constructions; val {13,14} stays clean for selection.

            // R28 (train neg): hard trap, fresh framing — one domain-NAME word
            // in a different sense inside unrelated content.
            {
                let task = fill(TASKS_GENERAL[(di + k + 4) % TASKS_GENERAL.len()], domain, sym);
                let tw = words[k % 2];
                let trap = format!("catering order confirmation: the {tw} for the \
                                    all-hands lunch is booked for the main room; \
                                    dietary preferences were collected and the menu \
                                    is final. contact the kitchen for changes.");
                let mut cands = distractors(di, k + 28);
                cands.push(cother(&trap));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 0.0, 28));
            }

            // R29 (train pos): hard synonym, fresh phrasing (bank 2) — goal
            // task, morphological echoes, low exact-word overlap.
            {
                let task = fill(TASKS_GOAL[(di + k + 4) % TASKS_GOAL.len()], domain, sym);
                let needle = SYNONYM_ANSWERS_2[di % SYNONYM_ANSWERS_2.len()];
                let mut cands = distractors(di, k + 29);
                cands.push(cother(needle));
                let target = cands.len() - 1;
                out.push(make(&task, recent, cands, vec![], target, 1.0, 29));
            }
        }
    }

    out
}
