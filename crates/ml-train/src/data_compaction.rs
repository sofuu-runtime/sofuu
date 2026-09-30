// ml-train/src/data_compaction.rs — synthetic dataset for the compaction
// gate (PLAN-ML-GATES §12).
//
// Each example is ONE target segment inside a constructed conversation
// context (task + recent keep-window + optional summary + sibling
// segments). Label 1 = disposable (compact me), 0 = load-bearing (keep).
// Labels are MECHANICAL — they follow from construction, so the net can
// only pass by learning the signals §12 names: referenced-by-recent,
// duplication, boilerplate, retrievability, decision language, recency.
//
// Channel-coverage rule (the freshness lesson): every feature channel
// must be ACTIVE in training. Holdout families test unseen CONSTRUCTIONS
// of trained channels, never a channel the net has never seen.

use sofuu_core::ml::compaction::features::{
    extract_all, CompactionContext, SegKind, SegmentInput,
};

use crate::train::Example;

/* ── Slot banks ──────────────────────────────────────────────────── */

/// Six disjoint domains — target content and task/recent draw from
/// DIFFERENT domains unless the family is about the interaction.
const DOMAINS: &[(&str, &[&str])] = &[
    ("database migration", &["database", "migration", "schema", "tables", "columns", "indexes"]),
    ("auth sessions", &["auth", "session", "token", "login", "cookie", "credentials"]),
    ("rate limiting", &["rate", "limit", "throttle", "requests", "bucket", "burst"]),
    ("web scraper", &["scraper", "crawl", "pages", "html", "links", "fetch"]),
    ("cache eviction", &["cache", "eviction", "entries", "lru", "ttl", "hits"]),
    ("log pipeline", &["logs", "pipeline", "ingest", "shards", "batches", "retention"]),
];

const PATHS: &[&str] = &[
    "scripts/migrate_db.rs", "src/auth/session.rs", "src/limit/throttle.rs",
    "src/scrape/walker.rs", "src/cache/lru.rs", "src/logs/shipper.rs",
    "crates/core/src/engine.rs", "tools/gen_schema.py", "config/limits.toml",
    "tests/pipeline_test.rs",
];

const SYMBOLS: &[&str] = &[
    "parse_config", "TokenBucket", "uv_async_send", "evict_expired",
    "ShardWriter", "walk_links", "session_from_cookie", "apply_schema",
    "RateGate", "compact_batch",
];

const TASKS: &[&str] = &[
    "finish the {d} work",
    "fix the failing test in the {d} code",
    "what is left to do on the {d} task?",
    "review the latest changes to the {d} module",
    "prepare the {d} changes for release",
    "debug the regression reported in {d}",
    "refactor the {d} hot path without breaking callers",
    "write up the status of the {d} effort",
    "check whether the {d} benchmarks improved",
    "close out the remaining {d} review comments",
];

/// Recent-window lines that do NOT reference the target (generic
/// continuation of the task domain).
const RECENT_GENERIC: &[&str] = &[
    "let us keep going with the current approach and re-run the suite",
    "the last change compiled cleanly, next step is the integration check",
    "I pushed the fix we discussed, please review it when ready",
    "sounds good, continue with the next item on the list",
    "the output looks right so far, proceed to the following stage",
    "nothing blocked on my side, carry on with the implementation",
    "the reviewer approved the direction, keep the momentum going",
    "one more pass over the tests and we can move forward",
    "the numbers match expectations, on to the next piece",
];

/// Filler sibling lines (old, low-signal, but NOT the target).
const FILLERS: &[&str] = &[
    "earlier note about the {d} work: progress is on track",
    "status update from the {d} front: nothing new to report",
    "a quick reminder that the {d} review is scheduled for later",
    "housekeeping: the {d} branch was rebased this morning",
    "context from the {d} standup: priorities unchanged",
    "the {d} dashboard shows the usual metrics today",
    "an old memo about the {d} rollout, mostly administrative",
    "planning artifact for the {d} milestone, already superseded",
];

/// Recent-window lines that RE-mention a path or symbol (reference
/// signal for the keep families) — varied phrasings so the net learns
/// the MECHANISM (token re-appears), not one sentence.
const RECENT_REF_TPLS: &[&str] = &[
    "please check {} again, the behaviour there decides the next step",
    "look back at {} — the answer depends on what it does",
    "the detail we need is in {}, revisit it before continuing",
    "whatever {} says is what we should follow here",
];

const RECENT_REF_TPLS_2: &[&str] = &[
    "one more look at {} before we wrap up",
    "keep {} in mind while finishing this",
    "the follow-up question is about {} specifically",
    "return to {} once the current step is done",
];

const VERBOSE_SCAFFOLDS: &[&str] = &[
    "running build for the {w} component\ncompiling module one\ncompiling module two\n\
     compiling module three\nlinking artifacts\nemitting warnings about unused imports\n\
     build finished with zero errors and twelve warnings\nartifacts written to the out \
     directory\nchecksums recorded for every produced file",
    "test run for the {w} suite\ntest one passed in twelve milliseconds\ntest two passed \
     in eight milliseconds\ntest three passed in thirty milliseconds\ntest four passed in \
     five milliseconds\nsummary: four passed, zero failed, zero ignored\ncoverage report \
     generated with line coverage at eighty two percent",
    "directory listing for the {w} tree\nsrc with fourteen files\ntests with six files\n\
     docs with three files\nconfiguration with two files\nreadme and license at the root\n\
     total size two hundred and forty kilobytes across twenty five entries",
    "dependency resolution for the {w} project\nresolving registry index\ndownloading \
     crate one version one point two\ndownloading crate two version zero point nine\n\
     verifying checksums\nlocking the graph with forty two packages\nno conflicts found",
    "profiling pass over the {w} hot path\nsampling at one kilohertz for ten seconds\n\
     top frame: serialization at twenty two percent\nsecond frame: hashing at fourteen \
     percent\nthird frame: allocation at nine percent\nflame graph written to disk",
    "lint pass across the {w} sources\nchecking style rules\nchecking clippy lints\n\
     three warnings about redundant closures\ntwo warnings about needless borrows\n\
     no errors reported\nsuggestions appended to the report file",
    "commit log for the {w} branch\ncommit one: initial scaffold\ncommit two: wire the \
     config loader\ncommit three: add the happy path tests\ncommit four: fix the edge \
     case found in review\ncommit five: tidy the module docs\nfive commits listed",
    "environment report for the {w} runner\noperating system: macos arm six four\n\
     rust toolchain: stable\nmemory: sixteen gigabytes\ndisk free: one hundred and \
     twenty gigabytes\nshell: zsh\nlocale: en us utf eight",
    "benchmark table for the {w} path\nbaseline: one hundred operations per second\n\
     candidate: one hundred and twelve operations per second\ndelta: twelve percent\n\
     variance: plus or minus three percent across five runs\nnotes recorded",
];

const BOILERPLATES: &[&str] = &[
    "cookies help us deliver our services. by using this site you agree to our use of \
     cookies. privacy policy and terms of service apply. navigation: home, about, \
     contact, sitemap. all rights reserved.",
    "this software is provided as is, without warranty of any kind, express or implied. \
     in no event shall the authors be liable for any claim, damages or other liability. \
     permission is granted to anyone to use this software for any purpose.",
    "subscribe to our newsletter for updates. follow us on social media. copyright \
     notice: all content on this page is protected. disclaimer: the views expressed \
     here do not necessarily reflect those of the organization.",
    "advertisement. sponsored content follows. click here to learn more about our \
     partners. this message was brought to you by our advertising partners. special \
     offer ends soon. terms and conditions apply.",
    "page one of twelve. next page. previous page. sort by relevance. filter results. \
     showing ten of one hundred and twenty entries. pagination controls. footer with \
     legal links and site map.",
    "end user license agreement. section one: acceptance of terms. section two: grant \
     of license. section three: restrictions. section four: termination. section five: \
     governing law and jurisdiction.",
    "thank you for visiting our website. this page uses cookies to improve your \
     experience. accept all cookies or manage preferences. read our cookie policy for \
     more information about the data we collect.",
    "press release boilerplate: the company is a leading provider of innovative \
     solutions for modern enterprises. for more information please visit the investor \
     relations page. media inquiries welcome at the address below.",
    "job listings widget: we are hiring across multiple teams. competitive salary and \
     benefits. remote friendly. equal opportunity employer. apply through the careers \
     portal linked below.",
];

const DECISION_SCAFFOLDS: &[&str] = &[
    "we decided to use {s} for this part — it keeps the hot path simple",
    "decision: the {d} module keeps {s} as the single entry point, that is settled",
    "the plan is to route everything through {s}; we agreed on that yesterday",
    "conclusion from the review: {s} stays, the alternative was rejected",
    "we will use {s} and must not change it without a new decision",
    "requirement accepted: {s} is the contract every caller follows from now on",
    "we chose {s} after the comparison — the trade-offs are documented and settled",
    "final call: {s} handles this responsibility, nothing else gets the job",
    "agreed in the review: the {d} work standardizes on {s} from here on",
];

const ERROR_SCAFFOLDS: &[&str] = &[
    "error: the {w} step failed with exit code one\nthread panicked at assertion \
     failed: expected four entries, found zero\nbacktrace printed to the log",
    "test failed in the {w} suite: expected success but the call returned not found\n\
     stack trace follows\nframe one, frame two, frame three",
    "fatal: could not open the {w} input\nthe underlying error was permission denied\n\
     the process exited with a failure status",
    "exception in the {w} worker: null reference at line eighty four\nthe batch was \
     rolled back and the error recorded",
    "build failed for the {w} target: unresolved import, missing symbol\nthe compiler \
     emitted one error and stopped",
    "the {w} job exited with a failure: connection refused while reaching the \
     upstream service\nretry budget exhausted after three attempts",
    "assertion failed in the {w} unit test: left and right differ\nleft: four\n\
     right: zero\nthe test harness recorded the mismatch and aborted the run",
    "timeout: the {w} step did not finish within the allotted window\nthe process was \
     killed and the partial output discarded",
];

const INSTRUCTION_SCAFFOLDS: &[&str] = &[
    "before anything else, make sure the {d} tests stay green",
    "keep the {d} public API unchanged while you refactor",
    "when you touch the {d} code, update the changelog too",
    "do not merge the {d} branch until the review passes",
    "run the {d} benchmark after every change and note the numbers",
    "the {d} fix must land before the release cut",
    "always run the formatter before committing the {d} changes",
    "keep every {d} commit small and reviewable",
    "document any {d} behaviour change in the migration notes",
];

const CHITCHAT: &[&str] = &[
    "ok", "sounds good", "sure, go ahead", "thanks for checking",
    "got it", "nice, that works", "fine by me", "great, keep it up",
    "no problem", "acknowledged", "yep, all good", "cool, carry on",
    "perfect, thanks", "alright then", "yes, that is fine", "cheers",
];

/// Recent-window lines that reference the target only by PARAPHRASE —
/// domain words and structure re-appear, but the path/symbol itself is
/// NOT repeated verbatim (real references are usually paraphrases).
/// Trains the reference channel to be robust to paraphrase (H15); the
/// second bank is the held-out VAL construction (H16).
const WEAK_REF_TPLS: &[&str] = &[
    "go back to the {d} file we looked at earlier — the handling there is what \
     the flow builds on",
    "the {d} code we read before handles the rest of the flow, return to that part",
    "revisit the {d} source from earlier, the handling there sits under the flow",
    "the file from the {d} work we opened before still carries the handling we need",
];

const WEAK_REF_TPLS_2: &[&str] = &[
    "the {d} part we went through before carries what the next step depends on",
    "look at that {d} file from earlier again, it still drives the logic we care about",
    "the earlier {d} reading is the piece to keep in mind here",
    "whatever that {d} source said before still guides the logic downstream",
];

/// Old assistant narration that restates what is already visible —
/// no decision language, no error content, never referenced. Trains
/// the MECHANISM H7 tests held out: an old assistant segment is only
/// load-bearing when referenced or carrying decision/error language.
/// (Must avoid every DECISION_WORDS / ERROR_WORDS entry.)
const NARRATION_SCAFFOLDS: &[&str] = &[
    "recap of the {w} pass: the steps ran in order, the output matched what the \
     earlier messages already show, and there is nothing new beyond that",
    "to summarize where the {w} work stands: everything proceeded as described \
     above, no surprises surfaced, the state is exactly as previously reported",
    "walking through what just happened with {w}: the first step produced its \
     usual output, the second step repeated the same numbers, and nothing else \
     is worth restating",
    "a restatement of the {w} status: the same points as before still hold, the \
     details are unchanged, and the earlier description remains accurate",
    "narrating the {w} sequence again for clarity: step one did what it always \
     does, step two followed, and the overall picture is identical to last time",
    "revisiting the {w} rundown: the items listed earlier are still the items, \
     the ordering is the same, and no detail has shifted since then",
    "as mentioned while working on {w}, the flow is exactly what the previous \
     turns describe, so repeating it here adds nothing",
    "the {w} walkthrough once more: inputs were read, processing continued, \
     outputs appeared, all of which is already visible above",
    "summing up the {w} activity in plain words: routine progress, ordinary \
     output, and a state that mirrors the earlier report line for line",
];

const OFFTASK_REASONS: &[&str] = &[
    "thinking out loud: the {w} approach reminds me of a completely different system \
     where we tried streaming aggregation, but that was another project and the \
     constraints there do not apply here at all",
    "a side note on the {w} design: historically this pattern comes from message \
     queues, though the analogy only goes so far and the comparison is not useful \
     for the current problem",
    "I wondered whether the {w} layer could be rewritten from scratch, but that is a \
     detour we are not taking; parking the idea entirely",
    "an observation about the {w} naming convention: it resembles the old internal \
     style guide, which is trivia rather than anything actionable",
    "musings on how the {w} docs could be reorganized someday; not relevant to the \
     task at hand and purely speculative",
    "a tangent on the {w} tooling: there was once an internal experiment with a \
     different generator, but it was abandoned long ago and has no bearing here",
    "idle thought about the {w} naming: it would read better reversed, though that is \
     cosmetics and we are not doing cosmetics right now",
    "recalling how the {w} review went last quarter — interesting history, but it \
     does not change anything about today's work",
];

/* ── Construction helpers ────────────────────────────────────────── */

#[derive(Clone)]
struct Seg {
    text: String,
    age: u32,
    kind: u8, // 0 user, 1 assistant, 2 tool_call, 3 tool_result
    retr: bool,
}

fn tokens_of(text: &str) -> u32 {
    (text.len() as u32 / 4).max(4)
}

fn fill(tpl: &str, d: &str, w: &str, s: &str) -> String {
    tpl.replace("{d}", d).replace("{w}", w).replace("{s}", s)
}

/// One training example: a context plus the target segment's index.
fn make(
    task: &str,
    summary: &str,
    recent: &str,
    segs: Vec<Seg>,
    target: usize,
    y: f32,
    group: u32,
) -> Example {
    let inputs: Vec<SegmentInput> = segs
        .iter()
        .map(|s| SegmentInput {
            text: &s.text,
            tokens: tokens_of(&s.text),
            age_steps: s.age,
            kind: match s.kind {
                0 => SegKind::User,
                1 => SegKind::Assistant,
                2 => SegKind::ToolCall,
                _ => SegKind::ToolResult,
            },
            retrievable: s.retr,
            already_compacted: false,
        })
        .collect();
    let ctx = CompactionContext { task, summary, recent, segments: &inputs };
    let feats = extract_all(&ctx);
    Example {
        x: feats[target].to_vec(),
        y,
        group,
        text: segs[target].text.clone(),
        task: task.to_string(),
    }
}

fn sibling(text: &str, age: u32, kind: u8) -> Seg {
    Seg { text: text.to_string(), age, kind, retr: false }
}

/* ── The dataset ─────────────────────────────────────────────────── */

pub fn build() -> Vec<Example> {
    let mut out = Vec::new();

    for (di, (domain, _words)) in DOMAINS.iter().enumerate() {
        // A different domain for the target's content (off-task by
        // construction unless a family says otherwise). 36 variants per
        // domain: the modulo index arithmetic cycles every bank, so each
        // family gets broad lexical coverage.
        for k in 0..36usize {
            let (tdomain, twords) = DOMAINS[(di + 1 + k) % DOMAINS.len()];
            let w = twords[k % twords.len()];
            let task = fill(TASKS[(di + k) % TASKS.len()], domain, "", "");
            let recent = RECENT_GENERIC[(di + k) % RECENT_GENERIC.len()];
            let filler1 = sibling(
                &fill(FILLERS[(di * 7 + k) % FILLERS.len()], domain, "", ""),
                10 + (k % 4) as u32,
                1,
            );
            // Large verbose sibling in EVERY context: real histories mix
            // big tool dumps with short messages, so relative size
            // (f6/f8) must never become a discriminator — keep-vs-
            // disposable has to ride on content channels (reference,
            // decision/instruction/error words, duplication, …). The
            // sibling's SIZE VARIES (1–3×) so no family can anchor on
            // "I am the largest segment".
            let big_text_base =
                fill(VERBOSE_SCAFFOLDS[(di * 5 + k + 4) % VERBOSE_SCAFFOLDS.len()], tdomain, w, "");
            let big_text = match k % 3 {
                0 => big_text_base.clone(),
                1 => format!("{big_text_base}\n{big_text_base}"),
                _ => format!("{big_text_base}\n{big_text_base}\n{big_text_base}"),
            };
            let big_sib = sibling(&big_text, 11, 3);

            // H0: old verbose tool output, never referenced → disposable.
            let verbose = fill(VERBOSE_SCAFFOLDS[(di * 5 + k) % VERBOSE_SCAFFOLDS.len()], tdomain, w, "");
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&verbose, 6 + (k % 6) as u32, 3),
                     sibling(recent, 0, 0)],
                2, 1.0, 0,
            ));

            // H4: pure boilerplate, old → disposable. Size varies (full
            // vs half) so the boilerplate channel (f14) is learned, not
            // "I happen to be the largest segment".
            let bp_full = BOILERPLATES[(di + k) % BOILERPLATES.len()];
            let bp: String = if k % 2 == 1 {
                bp_full.chars().take(200).collect()
            } else {
                bp_full.to_string()
            };
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&bp, 7 + (k % 5) as u32, 3),
                     sibling(recent, 0, 0)],
                2, 1.0, 4,
            ));

            // H6: old retrievable read (file/grep result), unreferenced →
            // disposable (cheap to fetch again). HALF ARE ON-TASK: a
            // re-fetchable read stays disposable even when its content
            // matches the task — retrievability, not topic, decides.
            let read = if k % 2 == 0 {
                format!(
                    "contents of {}: lines one through forty covering the {} work, \
                     headers and rows as printed, nothing else",
                    PATHS[(di + k) % PATHS.len()],
                    domain
                )
            } else {
                format!(
                    "contents of {}: lines one through forty of the {} listing, headers \
                     and rows as printed, nothing else",
                    PATHS[(di + k) % PATHS.len()],
                    tdomain
                )
            };
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(),
                     Seg { text: read, age: 8 + (k % 4) as u32, kind: 3, retr: true },
                     sibling(recent, 0, 0)],
                2, 1.0, 6,
            ));

            // H10: old trivial chit-chat → disposable (content, not
            // relative size, is the discriminator — big sibling present).
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(),
                     sibling(CHITCHAT[(di * 5 + k) % CHITCHAT.len()], 9, 0),
                     sibling(recent, 0, 0)],
                2, 1.0, 10,
            ));

            // H1: recent-window segments → keep (all kinds).
            let recent_seg = sibling(
                &format!("latest step on the {} task: {}", domain, recent),
                (k % 2) as u32,
                (k % 4) as u8,
            );
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&verbose, 8, 3), recent_seg],
                2, 0.0, 1,
            ));

            // H2: old segment whose path/symbol is RE-mentioned by the
            // recent window → keep (the load-bearing signal).
            let anchor = if k % 2 == 0 {
                PATHS[(di * 5 + k) % PATHS.len()]
            } else {
                SYMBOLS[(di + k) % SYMBOLS.len()]
            };
            let referenced = format!(
                "the {} logic lives in {} and controls the {} behaviour",
                domain, anchor, domain
            );
            let recent_line = RECENT_REF_TPLS[(di + k) % RECENT_REF_TPLS.len()]
                .replace("{}", anchor);
            out.push(make(
                &task, "", &recent_line,
                vec![filler1.clone(), big_sib.clone(), sibling(&referenced, 6 + (k % 5) as u32, 1),
                     sibling(&recent_line, 0, 0)],
                2, 0.0, 2,
            ));

            // H5: old decision statement → keep.
            let decision = fill(
                DECISION_SCAFFOLDS[(di + k) % DECISION_SCAFFOLDS.len()],
                domain,
                twords[(k + 1) % twords.len()],
                SYMBOLS[(di * 5 + k) % SYMBOLS.len()],
            );
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&decision, 7 + (k % 5) as u32, 1),
                     sibling(recent, 0, 0)],
                2, 0.0, 5,
            ));

            // H8: tool result carrying an error → keep (the model must
            // react to it).
            let err = fill(ERROR_SCAFFOLDS[(di + k) % ERROR_SCAFFOLDS.len()], tdomain, w, "");
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&err, 5 + (k % 4) as u32, 3),
                     sibling(recent, 0, 0)],
                2, 0.0, 8,
            ));

            // H9: old user instruction → keep (content, not relative
            // size, is the discriminator — big sibling present).
            let instr = fill(INSTRUCTION_SCAFFOLDS[(di * 5 + k) % INSTRUCTION_SCAFFOLDS.len()], domain, "", "");
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&instr, 8 + (k % 4) as u32, 0),
                     sibling(recent, 0, 0)],
                2, 0.0, 9,
            ));

            // H3: near-duplicate of an EARLIER sibling → the repeat is
            // disposable (trains the dup channel; H11 tests it held out).
            let original = format!(
                "listing for the {} area: entries alpha, beta, gamma — three items",
                tdomain
            );
            out.push(make(
                &task, "", recent,
                vec![sibling(&original, 9, 3), filler1.clone(), big_sib.clone(),
                     sibling(&original, 2 + (k % 3) as u32, 3),
                     sibling(recent, 0, 0)],
                3, 1.0, 3,
            ));

            // H7: old off-task reasoning, unreferenced, unretrievable →
            // disposable (superseded musings; VAL holdout).
            let musing = fill(OFFTASK_REASONS[(di + k) % OFFTASK_REASONS.len()], tdomain, w, "");
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&musing, 8 + (k % 4) as u32, 1),
                     sibling(recent, 0, 0)],
                2, 1.0, 7,
            ));

            // H14: old assistant narration, unreferenced, no decision or
            // error language → disposable (trains the channel H7 holds
            // out: old assistant segments without load-bearing markers).
            let narration =
                fill(NARRATION_SCAFFOLDS[(di * 3 + k) % NARRATION_SCAFFOLDS.len()], tdomain, w, "");
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&narration, 6 + (k % 6) as u32, 1),
                     sibling(recent, 0, 0)],
                2, 1.0, 14,
            ));

            // H11: duplicate with different scaffolding (TEST holdout).
            let dump = format!(
                "checksum table for the {} artifacts: entry one verified, entry two \
                 verified, entry three verified",
                tdomain
            );
            out.push(make(
                &task, "", recent,
                vec![sibling(&dump, 10, 3), filler1.clone(), big_sib.clone(),
                     sibling(&dump, 3 + (k % 2) as u32, 3),
                     sibling(recent, 0, 0)],
                3, 1.0, 11,
            ));

            // H12: decision in held-out phrasing (VAL holdout keep class).
            let decision2 = format!(
                "settled: the {} path goes through {} — that is the agreed contract",
                domain,
                SYMBOLS[(di + k + 3) % SYMBOLS.len()]
            );
            out.push(make(
                &task, "", recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&decision2, 6 + (k % 5) as u32, 1),
                     sibling(recent, 0, 0)],
                2, 0.0, 12,
            ));

            // H13: referenced-in-recent with held-out phrasing (TEST
            // holdout keep class).
            let anchor2 = PATHS[(di + k + 5) % PATHS.len()];
            let referenced2 = format!(
                "note: {} owns the {} handling and its behaviour is final",
                anchor2, domain
            );
            let recent_line2 = RECENT_REF_TPLS_2[(di + k) % RECENT_REF_TPLS_2.len()]
                .replace("{}", anchor2);
            out.push(make(
                &task, "",
                &recent_line2,
                vec![filler1.clone(), big_sib.clone(), sibling(&referenced2, 7 + (k % 4) as u32, 1),
                     sibling(&recent_line2, 0, 0)],
                2, 0.0, 13,
            ));

            // H15: old segment referenced only by PARAPHRASE in the
            // recent window (no verbatim anchor) → keep. Trains the
            // reference channel to survive paraphrase; H16 holds it out.
            let anchor_w = if k % 2 == 0 {
                PATHS[(di * 3 + k) % PATHS.len()]
            } else {
                SYMBOLS[(di * 7 + k) % SYMBOLS.len()]
            };
            let weak_referenced = format!(
                "the {} handling sits in {} and the rest of the flow builds on it",
                domain, anchor_w
            );
            let weak_recent =
                WEAK_REF_TPLS[(di + k) % WEAK_REF_TPLS.len()].replace("{d}", domain);
            out.push(make(
                &task, "", &weak_recent,
                vec![filler1.clone(), big_sib.clone(), sibling(&weak_referenced, 6 + (k % 5) as u32, 1),
                     sibling(&weak_recent, 0, 0)],
                2, 0.0, 15,
            ));

            // H16: paraphrased reference in held-out phrasing (VAL
            // holdout — boundary samples that keep threshold selection
            // honest instead of bottoming out on perfectly separable
            // folds).
            let anchor_v = PATHS[(di * 7 + k + 2) % PATHS.len()];
            let weak_referenced2 = format!(
                "earlier reading: {} carries the {} logic and downstream code follows it",
                anchor_v, domain
            );
            let weak_recent2 =
                WEAK_REF_TPLS_2[(di + k) % WEAK_REF_TPLS_2.len()].replace("{d}", domain);
            out.push(make(
                &task, "", &weak_recent2,
                vec![filler1.clone(), big_sib.clone(), sibling(&weak_referenced2, 7 + (k % 4) as u32, 1),
                     sibling(&weak_recent2, 0, 0)],
                2, 0.0, 16,
            ));
        }
    }

    /* ── CONFLICT TRUTH TABLE (groups 17–24) ──────────────────────────
     *
     * Hardening (2026-09-25). Every family above is PURE: one reason to
     * keep or drop, and the classes do not overlap in feature space. A
     * logistic regression separates them perfectly, which is why
     * `gate-eval` first reported compaction as a tie with the linear
     * reference (F1 1.000 both) — the benchmark could not tell whether
     * the 8 KB network earned its place.
     *
     * The eight families below are a full truth table over the three
     * channels that actually argue with each other:
     *
     *     R = referenced by the recent keep-window   (f9/f10)
     *     T = retrievable, cheap to re-fetch         (f13)
     *     D = carries decision language              (f23)
     *
     *     R T D | label
     *     ───────┼────────
     *     1 0 0  | keep      referenced and not re-fetchable
     *     1 1 0  | dispose   referenced, but a re-fetch is cheaper
     *     0 1 0  | dispose   old and re-fetchable
     *     0 0 0  | dispose   old, plain, unreferenced chatter
     *     1 1 1  | keep      the decision outranks the re-fetch
     *     1 0 1  | keep      decision + reference
     *     0 1 1  | keep      a decision is load-bearing on its own
     *     0 0 1  | keep      ditto
     *
     * The R/T/D table above turned out to be LINEARLY SEPARABLE (verified
     * by brute-force search: a single plane splits all eight cells), so
     * it could not answer the question either — a linear model scored
     * 1.000 on it. A conflict only counts if no straight line can solve
     * it.
     *
     * GROUPS 25–26 are therefore an XOR family, the smallest pattern that
     * is provably non-separable in binary coordinates (exhaustive search
     * over a coefficient grid finds no separator). The compaction
     * semantics behind it:
     *
     *   R = the recent keep-window cites this segment
     *   D = the segment records a decision
     *
     *     R D | label   reading
     *     ────┼────────  ──────────────────────────────────────────────
     *     1 0 | keep     one clean reason to keep it: it was cited
     *     0 1 | keep     one clean reason to keep it: it decided something
     *     1 1 | DISPOSE  cited for an unrelated reason AND carrying a
     *                    decision is a coincidence of vocabulary, not a
     *                    reason to keep — the cite is incidental
     *     0 0 | DISPOSE  no cite, no decision: ordinary churn
     *
     * i.e. keep iff EXACTLY ONE of {cited, decided} holds. No weighted
     * sum of the two channels reproduces that: the positive cells sit
     * on the diagonal and the negatives off it, so any line that catches
     * both positives also catches a negative.
     *
     * This is the measurement the original benchmark could not make. If
     * the network clears this and the linear reference does not, the
     * 8 KB is buying real capacity; if BOTH clear it, the data is still
     * too easy and the honest conclusion is that the gate does not need
     * a network. Either outcome is useful; a tie reported as a tie is
     * the point. */
    for (di, (domain, _words)) in DOMAINS.iter().enumerate() {
        for k in 0..36usize {
            let (tdomain, twords) = DOMAINS[(di + 2 + k) % DOMAINS.len()];
            let w = twords[(k * 3 + 1) % twords.len()];
            let task = fill(TASKS[(di + k * 5) % TASKS.len()], domain, "", "");
            let path = PATHS[(di * 3 + k) % PATHS.len()];
            let filler = sibling(
                &fill(FILLERS[(di * 11 + k) % FILLERS.len()], domain, "", ""),
                12 + (k % 3) as u32,
                1,
            );
            let age = 5 + (k % 7) as u32;

            // A keep-window that cites the target (R = 1) and one that
            // does not (R = 0). The citation is lexical, so f10 fires.
            let citing_recent = format!(
                "going back to what you said about {path} and the {w} numbers — \
                 we still need that before the {} work",
                tdomain
            );
            let neutral_recent = format!(
                "moving on to the {domain} side now, the {w} part is done and closed"
            );
            // Bodies differing ONLY in decision language, so the D
            // channel is the single difference between the pair.
            let plain_body = format!(
                "the {w} column in {path} is 4200 wide and the {} rows follow the \
                 same order as the previous run",
                tdomain
            );
            let decide_body = format!(
                "we decided to pin the {w} column in {path} at 4200; that choice is \
                 final and the {} rows must match it",
                tdomain
            );

            // One cell emitter, parameterised by the retrievable flag, so
            // both the XOR (25/26) and the R/T/D table (17–24) share a
            // single borrow of `out`.
            let mut cell = |r: bool, t: bool, d: bool, y: f32, group: u32| {
                let recent = if r { citing_recent.clone() } else { neutral_recent.clone() };
                let body = if d { decide_body.clone() } else { plain_body.clone() };
                out.push(make(
                    &task,
                    "",
                    &recent,
                    vec![
                        filler.clone(),
                        Seg { text: body, age, kind: 1, retr: t },
                        sibling(&recent, 0, 0),
                    ],
                    1,
                    y,
                    group,
                ));
            };

            /* Groups 17–24: the R/T/D disagreement table — a THIRD
             * channel (retrievable) joins the XOR's two, and the
             * combinations are not all learnable from the XOR pair
             * alone. Held out of training entirely, so the network is
             * asked to apply the interaction it learned on 25/26 to
             * conflicts it has never seen. */
            cell(true, false, false, 0.0, 17); // cited, unique            → keep
            cell(true, true, false, 1.0, 18); // cited, re-fetchable       → dispose
            cell(false, true, false, 1.0, 19); // old + re-fetchable        → dispose
            cell(false, false, false, 1.0, 20); // no cite, no decision      → dispose
            cell(true, true, true, 0.0, 21); // cited + decided + refetch → keep
            cell(true, false, true, 0.0, 22); // cited + decided           → keep
            cell(false, true, true, 0.0, 23); // decided, re-fetchable     → keep
            cell(false, false, true, 0.0, 24); // decided alone             → keep

            // Groups 25/26: the XOR itself (TRAIN — see the trainer's
            // split comment; a pattern never trained on is unlearnable).
            cell(true, false, false, 0.0, 25); // R=1 D=0 → keep
            cell(false, false, true, 0.0, 25); // R=0 D=1 → keep
            cell(true, false, true, 1.0, 26); // R=1 D=1 → dispose
            cell(false, false, false, 1.0, 26); // R=0 D=0 → dispose
        }
    }

    /* Augmentation: ~1/5 of the examples get an existing summary that
     * PARAPHRASES the target (redundancy channel) — label invariant: a
     * segment already absorbed into the summary is AT LEAST as
     * disposable, never more load-bearing. Only applied to disposable
     * families so the invariant is airtight. */
    let mut aug = Vec::new();
    for (i, e) in out.iter().enumerate() {
        if i % 5 == 0 && e.y >= 0.5 && [0u32, 4, 6, 10, 14].contains(&e.group) {
            let mut c = e.clone();
            // Redundancy shows up in feature 12 — recompute with summary.
            let head: String = e.text.chars().take(60).collect();
            let summary = format!("earlier material covered: {head}");
            let segs = [
                sibling(&summary, 12, 1),
                sibling(&e.text, 8, 3),
                sibling("keep going", 0, 0),
            ];
            c.x = {
                let inputs: Vec<SegmentInput> = segs
                    .iter()
                    .map(|s| SegmentInput {
                        text: &s.text,
                        tokens: tokens_of(&s.text),
                        age_steps: s.age,
                        kind: SegKind::ToolResult,
                        retrievable: false,
                        already_compacted: false,
                    })
                    .collect();
                let ctx = CompactionContext {
                    task: &e.task,
                    summary: &summary,
                    recent: "keep going",
                    segments: &inputs,
                };
                extract_all(&ctx)[1].to_vec()
            };
            aug.push(c);
        }
    }
    out.extend(aug);
    out
}
