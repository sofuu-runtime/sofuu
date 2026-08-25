// ml-train/src/data_freshness.rs — the freshness training set
// (PLAN-ML-GATES §5/§9).
//
// Labels are MECHANICAL — each follows from the construction, never from
// case-by-case opinion:
//
//   label = task_is_time_sensitive AND content_is_possibly_stale
//
// Content families (one CV group each — generalization is proven by
// holding whole families out):
//   G0 explicit-old-date      possibly stale  (the easy class)
//   G1 explicit-recent-date   fresh
//   G2 implicit stale vocab   possibly stale, NO date (the learnable mass)
//   G3 implicit fresh vocab   fresh
//   G4 legacy-version docs    possibly stale (legacy markers make it real)
//   G5 changelog (old year)   possibly stale
//   G6 timeless textbook      never stale
//   G7 hedging language       possibly stale
//   G8 news URL with old year possibly stale
//   G9 future-year roadmap    never stale (future ≠ stale — feature 12)
//
// The majority of positives are IMPLICIT (no date present) — if explicit
// dates fully decided it, a rule would do (§5). Crossing every content
// example with both a time-sensitive and a timeless task teaches the
// interaction: "latest QuickJS release" from a 2019 page → verify;
// "what is a B-tree" from the same page → fine.
//
// LEXICAL DIVERSITY is load-bearing: the embedder is char-trigram TF-IDF,
// so if one exact phrase predicted the label the net would memorize the
// phrase and invert on unseen constructions (that failure mode was
// observed and fixed here — every scaffold/phrase bank is varied so only
// the SHARED signal — years, staleness semantics, task interaction —
// predicts the label).

use sofuu_core::ml::freshness::features::{extract, FreshnessInput, SourceKind};

use crate::train::Example;

pub const NOW_YEAR: u32 = 2026;

const LIBS: &[&str] = &[
    "QuickJS", "libuv", "serde", "tokio", "React", "Vue", "Express", "Django", "Flask",
    "NumPy", "pandas", "TensorFlow", "PyTorch", "webpack", "Vite", "esbuild", "Redis",
    "PostgreSQL", "SQLite", "ZeroMQ", "gRPC", "Electron", "Tauri", "LLVM", "Zlib",
    "OpenSSL", "curl", "nginx", "Memcached", "RabbitMQ",
];

const CONCEPTS: &[&str] = &[
    "B-tree", "hash table", "event loop", "promise", "garbage collector", "mutex",
    "trie", "bloom filter", "quicksort", "TCP handshake", "TLS certificate",
    "reference counting", "copy on write", "tail call", "closure", "virtual memory",
    "skip list", "readers-writer lock", "arena allocator", "continuation",
];

const SERVICES: &[&str] = &[
    "CloudRun", "Lambda", "S3 storage", "the Gemini API", "OpenAI embeddings",
    "GitHub Actions", "CircleCI", "Datadog", "Cloudflare Workers", "Fly.io",
];

const TOPICS: &[&str] = &[
    "the QuickJS project", "WebAssembly GC", "the Rust edition", "HTTP/3 adoption",
    "the Bun runtime", "ARM servers", "RISC-V laptops", "JS engine market share",
    "the SQLite release cycle", "edge computing pricing",
];

const TS_TASKS: &[&str] = &[
    "what is the latest version of {lib}",
    "current release of {lib}",
    "is {lib} still maintained",
    "what are the newest features in {lib}",
    "how much does {svc} cost now",
    "latest news about {topic}",
    "which {lib} version should I use today",
    "what is the current status of {topic}",
    "recent updates to {lib}",
    "what does {svc} charge this year",
];

const TL_TASKS: &[&str] = &[
    "what is a {concept}",
    "explain how a {concept} works",
    "what is the difference between a {concept} and a stack",
    "why do programs use a {concept}",
    "give me an example of a {concept}",
    "what is the big-O of operations on a {concept}",
    "when should I prefer a {concept}",
    "describe the invariants of a {concept}",
];

fn fill(tpl: &str, lib: &str, concept: &str, svc: &str, topic: &str) -> String {
    tpl.replace("{lib}", lib)
        .replace("{concept}", concept)
        .replace("{svc}", svc)
        .replace("{topic}", topic)
}

fn time_sensitive_tasks() -> Vec<String> {
    let mut out = Vec::new();
    for (i, tpl) in TS_TASKS.iter().enumerate() {
        for j in 0..6 {
            let lib = LIBS[(i * 3 + j) % LIBS.len()];
            let svc = SERVICES[(i + j) % SERVICES.len()];
            let topic = TOPICS[(i * 2 + j) % TOPICS.len()];
            out.push(fill(tpl, lib, "", svc, topic));
        }
    }
    out
}

fn timeless_tasks() -> Vec<String> {
    let mut out = Vec::new();
    for (i, tpl) in TL_TASKS.iter().enumerate() {
        for j in 0..6 {
            let concept = CONCEPTS[(i * 3 + j) % CONCEPTS.len()];
            out.push(fill(tpl, "", concept, "", ""));
        }
    }
    out
}

/* ── Phrase banks — varied so no single trigram run predicts a label ── */

const STALE_PHRASES: &[&str] = &[
    "This project has been deprecated and is no longer maintained.",
    "The repository is archived and read-only; development has stopped.",
    "This tool was discontinued and superseded by a ground-up rewrite.",
    "Legacy module, removed from the current distribution and kept for reference.",
    "This package reached end of life and receives no security fixes.",
    "Abandoned by its authors; the issue tracker was closed permanently.",
    "Unmaintained since the original team moved on to other work.",
    "This release line is obsolete; upstream suggests migrating away.",
    "The project sunset its public infrastructure last spring.",
    "Retired: the maintainers recommend a drop-in replacement instead.",
    "Development halted and the CI pipelines were switched off.",
    "This branch is dead code, preserved only for historical interest.",
    "Support ended; no further patches will be published for it.",
    "The package was yanked from the registry after the final release.",
];

const STALE_SCAFFOLDS: &[&str] = &[
    "{lib} user guide. {phrase} The guide covers installation, configuration and common patterns.",
    "{phrase} Notes on getting started with {lib} follow below.",
    "Working with {lib}: setup, tips and examples. {phrase}",
    "{lib} overview article. {phrase} Usage patterns are described next.",
    "Introduction to {lib} for newcomers. {phrase} The API tour continues below.",
];

const FRESH_PHRASES: &[&str] = &[
    "We just shipped a new version with major improvements.",
    "Announced today: a brand new build is now available for download.",
    "This is the latest release, published this week.",
    "Fresh out of the oven: the newest update introduces a redesigned API.",
    "Currently maintained with weekly releases and active development.",
    "A new stable version landed this morning after a short beta.",
    "The team released an updated build with long-requested features.",
    "Now available: the most recent release, rolled out hours ago.",
    "A modern rewrite, actively developed and updated frequently.",
    "The newest edition launched this month with broad compatibility.",
    "Just published: release candidates promoted to stable today.",
    "This current generation is supported with rapid patch cycles.",
];

const FRESH_SCAFFOLDS: &[&str] = &[
    "{lib} news. {phrase} Download the update from the project page.",
    "{phrase} Details about {lib} follow.",
    "Release post for {lib}. {phrase}",
    "{lib} status update: {phrase}",
];

const HEDGES: &[&str] = &[
    "This information may be outdated; as of last check the details were accurate.",
    "Note: this guide might have changed since it was written. Verify before relying on it.",
    "At the time of writing this was correct, but there could be newer information.",
    "These instructions were valid previously; check whether they still apply.",
    "Details below were accurate when published but may not be current anymore.",
    "Treat the following as possibly stale — re-verify anything important.",
    "The author warns this write-up could lag behind the real state of things.",
    "Snapshot from an earlier review; some facts might have drifted since.",
];

const LEGACY_MARKERS: &[&str] = &[
    "documents the legacy API kept for compatibility with the older release line",
    "covers the previous generation interface, superseded by the current one",
    "describes the retired module layout from the older series",
    "explains the deprecated call conventions of the earlier major version",
    "details the obsolete configuration format phased out upstream",
];

const URL_HOSTS: &[&str] = &[
    "news.example.com", "tech.example.org", "daily.example.net", "wire.example.io",
];

const BOILERPLATE: &str = "Cookies help us deliver our services. By using this site you \
    agree to our use of cookies. Privacy policy and terms of service apply. Navigation: \
    home, about, contact, sitemap. All rights reserved.";

/// (text, possibly_stale, group) — one entry per content construction.
fn content_bank() -> Vec<(String, bool, u32)> {
    let mut bank: Vec<(String, bool, u32)> = Vec::new();

    // G0: explicit old dates (2015..2022 — all ≥3 years behind NOW_YEAR).
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..4 {
            let y = 2015 + ((i + k * 3) % 8);
            let scaffold = match (i + k) % 3 {
                0 => format!(
                    "{lib} documentation. Updated on March {}, {y}. This page describes \
                     installing and configuring {lib}.",
                    4 + i % 20
                ),
                1 => format!(
                    "Posted January {}, {y}: our review of {lib} and how it fits modern \
                     workflows.",
                    2 + k % 26
                ),
                _ => format!(
                    "{lib} tutorial, last revised {y}-0{}-1{}. Follow the steps to set up \
                     {lib} from scratch.",
                    1 + k % 9,
                    1 + i % 9
                ),
            };
            bank.push((scaffold, true, 0));
        }
    }

    // G1: explicit recent dates (NOW_YEAR-1..NOW_YEAR).
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..3 {
            let y = NOW_YEAR - 1 + ((i + k) % 2) as u32;
            let scaffold = match (i + k) % 2 {
                0 => format!(
                    "{lib} release announcement, published January {}, {y}. The new build \
                     ships performance improvements and bug fixes.",
                    3 + i % 24
                ),
                _ => format!(
                    "On {y}-0{}-0{}, the {lib} team cut a new release with updated \
                     dependencies.",
                    1 + (i + k) % 9,
                    1 + i % 27
                ),
            };
            bank.push((scaffold, false, 1));
        }
    }

    // G2: implicit stale vocabulary — NO date anywhere (the learnable mass).
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..4 {
            let phrase = STALE_PHRASES[(i * 5 + k * 3) % STALE_PHRASES.len()];
            let text = STALE_SCAFFOLDS[(i + k) % STALE_SCAFFOLDS.len()]
                .replace("{lib}", lib)
                .replace("{phrase}", phrase);
            bank.push((text, true, 2));
        }
    }

    // G3: implicit fresh vocabulary.
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..3 {
            let phrase = FRESH_PHRASES[(i * 7 + k * 5) % FRESH_PHRASES.len()];
            let text = FRESH_SCAFFOLDS[(i + k) % FRESH_SCAFFOLDS.len()]
                .replace("{lib}", lib)
                .replace("{phrase}", phrase);
            bank.push((text, false, 3));
        }
    }

    // G4: legacy-version docs — version strings PLUS explicit legacy
    // markers (a bare version number carries no staleness information the
    // model could ever see; the markers make the label learnable).
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..4 {
            let major = i % 3; // 0.x–2.x era versions
            bank.push((
                format!(
                    "{lib} {major}.{} reference manual. It {}. Install with the package \
                     manager: add {lib} {major}.{} to your dependencies.",
                    2 + k,
                    LEGACY_MARKERS[(i + k) % LEGACY_MARKERS.len()],
                    2 + k
                ),
                true,
                4,
            ));
        }
    }

    // G5: changelog with an old year.
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..3 {
            let y = 2018 + ((i + k) % 5);
            bank.push((
                format!(
                    "{lib} changelog. v{}.{k} ({y}-06-01): breaking changes, bug fixes, \
                     removed legacy endpoints. v{}.{k} ({y}-02-11): initial stable API.",
                    2 + i % 3,
                    1 + i % 3
                ),
                true,
                5,
            ));
        }
    }

    // G6: timeless textbook content — never stale.
    for (i, concept) in CONCEPTS.iter().enumerate() {
        for k in 0..4 {
            bank.push((
                format!(
                    "A {concept} is a fundamental data structure used in systems \
                     programming. This article explains the invariants, the core \
                     operations, and the classic trade-offs of the {concept}. {}",
                    match (i + k) % 3 {
                        0 => "Complexity analysis included.",
                        1 => "Diagrams and pseudocode provided.",
                        _ => "Worked examples accompany each section.",
                    }
                ),
                false,
                6,
            ));
        }
    }

    // G7: hedging language — the author already doubts currency.
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..3 {
            bank.push((
                format!(
                    "{lib} setup notes. {} Steps: install, configure, run.",
                    HEDGES[(i * 3 + k) % HEDGES.len()]
                ),
                true,
                7,
            ));
        }
    }

    // G8: news/blog URL shape with an old year in the path.
    for (i, topic) in TOPICS.iter().enumerate() {
        for k in 0..4 {
            let y = 2017 + ((i + k) % 6);
            let host = URL_HOSTS[(i + k) % URL_HOSTS.len()];
            bank.push((
                format!(
                    "https://{host}/{y}/{:02}/article-about-{} — reporting on {topic}: \
                     what happened, who is involved, and what comes next.",
                    1 + (i + k) % 12,
                    topic.replace(' ', "-"),
                ),
                true,
                8,
            ));
        }
    }

    // G9: future-year roadmaps — a future year is NOT staleness.
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..2 {
            bank.push((
                format!(
                    "{lib} roadmap for {} and beyond: planned features, scheduled \
                     milestones, and the long-term vision for the project.",
                    NOW_YEAR + 1 + ((i + k) % 2) as u32
                ),
                false,
                9,
            ));
        }
    }

    // G10: neutral zero-signal content — no dates, no versions, no stale or
    // fresh vocabulary. Nothing to verify, so the label is 0 for BOTH task
    // kinds. This family must exist in TRAINING or the net never learns
    // that "time-sensitive task + no temporal evidence" means no nudge —
    // it would default to "time-sensitive task → verify" and nudge on
    // everything (the observed failure mode before this family landed).
    let neutral_scaffolds: &[&str] = &[
        "{lib} API reference: modules, functions, and typical usage patterns.",
        "Overview of {lib}: architecture, main components, and integration points.",
        "How {lib} configuration files are structured, with annotated examples.",
        "A tour of the {lib} codebase: directory layout and key abstractions.",
        "{lib} internals: data flow, threading model, and extension hooks.",
        "Getting productive with {lib}: workflows, conventions, and tooling.",
    ];
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..4 {
            bank.push((
                neutral_scaffolds[(i + k * 2) % neutral_scaffolds.len()]
                    .replace("{lib}", lib),
                false,
                10,
            ));
        }
    }

    // G11: URL shape with a RECENT year — the shape alone is not
    // staleness, the year decides. Trains the url-shape channel G8 tests
    // (without this family the url_shape feature is zero across training
    // and its weight never leaves the decay floor — the same pathology
    // hedging had before G7 moved into training).
    for (i, topic) in TOPICS.iter().enumerate() {
        for k in 0..3 {
            let y = NOW_YEAR - 1 + ((i + k) % 2) as u32;
            let host = URL_HOSTS[(i + k) % URL_HOSTS.len()];
            bank.push((
                format!(
                    "https://{host}/{y}/{:02}/coverage-of-{} — an update on {topic}: \
                     the latest developments, the people involved, and what is next.",
                    1 + (i + k) % 12,
                    topic.replace(' ', "-"),
                ),
                false,
                11,
            ));
        }
    }

    // G12: future-year plans in varied phrasings — a future year is not
    // staleness even for a time-sensitive task. Trains the suppression
    // channel G9 tests (feature 7 is zero everywhere else in training).
    let future_phrases: &[&str] = &[
        "is scheduled for", "is planned for", "will arrive in", "lands in",
    ];
    for (i, lib) in LIBS.iter().enumerate() {
        for k in 0..2 {
            let y = NOW_YEAR + 1 + ((i + k) % 2) as u32;
            bank.push((
                format!(
                    "{lib} release plan: the next major version {} {y}, bringing the \
                     redesigned query planner and the new storage engine.",
                    future_phrases[(i + k) % future_phrases.len()]
                ),
                false,
                12,
            ));
        }
    }

    bank
}

/// The full dataset: every content example crossed with time-sensitive and
/// timeless tasks (the interaction that earns the parameters, §9), plus
/// rule-based augmentation whose labels follow deterministically from the
/// transformation (padding with boilerplate must not flip the label).
pub fn build() -> Vec<Example> {
    let ts = time_sensitive_tasks();
    let tl = timeless_tasks();
    let bank = content_bank();
    let mut out = Vec::new();
    let kinds = [SourceKind::Web, SourceKind::File, SourceKind::Tool, SourceKind::Memory];

    for (bi, (text, possibly_stale, group)) in bank.iter().enumerate() {
        let task_pairs = [
            ts[bi % ts.len()].clone(),
            tl[bi % tl.len()].clone(),
            ts[(bi * 7 + 3) % ts.len()].clone(),
        ];
        for (ti, task) in task_pairs.iter().enumerate() {
            let ts_task = ti != 1; // pairs 0 and 2 are time-sensitive
            let y = if ts_task && *possibly_stale { 1.0 } else { 0.0 };
            let kind = kinds[(bi + ti) % kinds.len()];
            let (strength, age_days) = if kind == SourceKind::Memory {
                (0.3 + 0.05 * (bi % 10) as f32, 30.0 * (bi % 20) as f32)
            } else {
                (0.0, 0.0)
            };
            let inp = FreshnessInput {
                text,
                task,
                kind,
                strength,
                age_days,
                now_year: NOW_YEAR,
            };
            out.push(Example {
                x: extract(&inp).to_vec(),
                y,
                group: *group,
                text: text.clone(),
                task: task.clone(),
            });

            // Augmentation: pad ~1/4 of the examples with boilerplate.
            // Label is invariant by construction — length must not flip it.
            if (bi + ti) % 4 == 0 {
                let padded = format!("{text}\n\n{BOILERPLATE}\n\n{BOILERPLATE}");
                let inp2 = FreshnessInput { text: &padded, ..inp };
                out.push(Example {
                    x: extract(&inp2).to_vec(),
                    y,
                    group: *group,
                    text: padded,
                    task: task.clone(),
                });
            }
        }
    }
    out
}
