// ml-train/src/data_supervisor.rs — the supervisor dataset (PLAN-ML-GATES
// §11, construction rules §9).
//
// One example per (trajectory, candidate action) pair, label 1 = "this
// action is waste right now — nudge", 0 = "let it run". Families S0–S25
// cover the decision classes of §11: duplication, re-reads, breadth,
// off-task targets, spinning, error retries, budget pressure, skip-set
// advice, useless-tool history, and the loop-boundary pseudo-action.
//
// Two constructions carry the generalization burden (lessons from the
// relevance phase):
//
// - THE STEM RULE. Every legitimate action's target/pattern echoes a TASK
//   stem (verify_signature → signature_pad; "rewrites the schema" →
//   schema_version), so the morphological channel f2 separates "exploring
//   the task's domain" from wandering. Every off-task trap shares at most
//   a SYMBOL word (handle_delivery → "office plant delivery schedule"),
//   never a task word — a trap that echoed the task would be a weak
//   positive in the off-task region and poison the boundary.
// - CHANNEL COVERAGE + CLEAN SELECTION. Every feature channel is active
//   in TRAINING (near-dup S18, paraphrased dups S15, skip-set S13, loop
//   S20–S23, …); the val families (S24 held-out clean, S26 held-out
//   clean positives) hold out CONSTRUCTIONS of trained channels, never
//   an untrained channel, and stay CLEAN — a hard val family the net
//   can't yet do collapses threshold selection to the floor (the
//   relevance-phase lesson; hard regimes like S15 belong in training).

use sofuu_core::ml::supervisor::features::{
    extract, SupervisorContext, TrajCall, LOOP_TOOL,
};

use crate::train::{Example, Rng};

struct Domain {
    task: &'static str,
    /// On-task file paths (index 0 is the primary file).
    paths: &'static [&'static str],
    /// On-task grep symbols — from index 1 on, each echoes a task stem.
    symbols: &'static [&'static str],
    /// Off-task targets: share at most a SYMBOL word, never a task word.
    off_path: &'static str,
    off_pattern: &'static str,
}

const TRAIN_DOMAINS: &[Domain] = &[
    Domain {
        task: "fix the verify_signature function in the payment webhook handler",
        paths: &["src/payment/webhook.rs", "src/payment/sign.rs", "tests/payments_webhook.rs"],
        symbols: &["verify_signature", "signature_pad", "webhook_route"],
        off_path: "docs/office_plant_delivery.md",
        off_pattern: "office plant delivery schedule",
    },
    Domain {
        task: "finish the database migration that rewrites the schema",
        paths: &["src/migration/migrate.rs", "src/migration/schema.rs", "tests/migrate_test.rs"],
        symbols: &["rewrite_table", "schema_version", "migrate_lock"],
        off_path: "notes/potluck_signup_sheet.md",
        off_pattern: "potluck signup sheet",
    },
    Domain {
        task: "fix the eviction policy in the lru cache",
        paths: &["src/cache/lru.rs", "src/cache/ttl.rs", "tests/lru_test.rs"],
        symbols: &["evict_expired", "cache_touch", "lru_evict"],
        off_path: "docs/team_lunch_rotation.md",
        off_pattern: "team lunch rotation sign-up",
    },
    Domain {
        task: "repair the session check in the auth guard",
        paths: &["src/auth/guard.rs", "src/auth/session.rs", "tests/auth_test.rs"],
        symbols: &["revalidate_session", "session_store", "guard_token"],
        off_path: "hr/mentorship_pairing.md",
        off_pattern: "mentorship pairing list",
    },
    Domain {
        task: "make the page crawler resume after a network drop",
        paths: &["src/crawler/crawl.rs", "src/crawler/queue.rs", "tests/crawl_test.rs"],
        symbols: &["crawler_queue", "crawler_retry", "page_fetch"],
        off_path: "ops/roof_repair_quote.md",
        off_pattern: "roof repair quote comparison",
    },
    Domain {
        task: "batch the log pipeline shards into storage",
        paths: &["src/pipeline/batch.rs", "src/pipeline/ship.rs", "tests/pipeline_test.rs"],
        symbols: &["batch_shards", "pipeline_flush", "log_ship"],
        off_path: "facilities/gym_membership_discount.md",
        off_pattern: "gym membership discount form",
    },
];

/// Val domains — fresh wording for the held-out clean family (S24).
const VAL_DOMAINS: &[Domain] = &[
    Domain {
        task: "finish the invoice reconciliation job",
        paths: &["src/invoice/reconcile.rs", "src/invoice/ledger.rs", "tests/invoice_test.rs"],
        symbols: &["reconcile_ledger", "invoice_close", "ledger_flag"],
        off_path: "docs/billing_offsite.md",
        off_pattern: "billing offsite agenda",
    },
    Domain {
        task: "fix the sidebar overlap in the settings layout",
        paths: &["src/settings/layout.css", "src/settings/sidebar.css", "tests/layout_test.js"],
        symbols: &["sidebar_grid", "settings_stack", "layout_overlap"],
        off_path: "docs/color_poll_results.md",
        off_pattern: "color poll results for the lounge",
    },
];

/// Test domains — fresh domains for the held-out replays (S16) and the
/// fresh off-task traps (S25).
const TEST_DOMAINS: &[Domain] = &[
    Domain {
        task: "roll out the gateway config change",
        paths: &["src/gateway/config.yaml", "src/gateway/upstream.yaml", "tests/gateway_test.sh"],
        symbols: &["gateway_upstream", "gateway_route", "config_reload"],
        off_path: "docs/upstream_canoe_trip.md",
        off_pattern: "upstream canoe trip signup",
    },
    Domain {
        task: "stop the audio buffer underrun in the player",
        paths: &["src/audio/player.rs", "src/audio/mixer.rs", "tests/player_test.rs"],
        symbols: &["refill_buffer", "audio_clock", "player_underrun"],
        off_path: "docs/clock_tower_tour.md",
        off_pattern: "clock tower tour signup",
    },
];

/* ── Args + call builders (canonical sorted-key JSON, matching the JS
 *    mlCanon/mlSig/mlTarget contract: target = path || pattern) ───── */

fn read_a(path: &str) -> String {
    format!("{{\"path\":\"{path}\"}}")
}

/// Parent directory of a repo path ("src/payment/webhook.rs" →
/// "src/payment") — the target of the clean list_dir follow-up.
fn dir_of(path: &str) -> String {
    path.rsplit_once('/')
        .map(|(d, _)| d.to_string())
        .unwrap_or_else(|| ".".to_string())
}
fn read_a_off(path: &str, off: u32) -> String {
    format!("{{\"offset\":{off},\"path\":\"{path}\"}}")
}
fn grep_a(pat: &str) -> String {
    format!("{{\"pattern\":\"{pat}\"}}")
}
fn grep_a_in(path: &str, pat: &str) -> String {
    format!("{{\"path\":\"{path}\",\"pattern\":\"{pat}\"}}")
}
fn edit_a(path: &str) -> String {
    format!("{{\"new_text\":\"fixed\",\"old_text\":\"broken\",\"path\":\"{path}\"}}")
}
fn write_a(path: &str, content_chars: usize) -> String {
    format!("{{\"content\":\"{}\",\"path\":\"{path}\"}}", "x".repeat(content_chars))
}
fn bash_a(cmd: &str) -> String {
    format!("{{\"command\":\"{cmd}\"}}")
}
fn web_a(query: &str) -> String {
    format!("{{\"query\":\"{query}\"}}")
}

fn tc(tool: &str, args: &str, target: &str, chars: u32, errored: bool) -> TrajCall {
    TrajCall {
        tool: tool.to_string(),
        sig: format!("{tool}:{args}"),
        target: target.to_string(),
        result_chars: chars,
        errored,
    }
}

fn read_call(path: &str, chars: u32) -> TrajCall {
    tc("read_file", &read_a(path), path, chars, false)
}
fn read_call_off(path: &str, off: u32, chars: u32) -> TrajCall {
    tc("read_file", &read_a_off(path, off), path, chars, false)
}
fn grep_call(pat: &str, chars: u32) -> TrajCall {
    tc("grep", &grep_a(pat), pat, chars, false)
}
fn edit_call(path: &str) -> TrajCall {
    tc("edit_file", &edit_a(path), path, 140, false)
}
fn bash_call(cmd: &str, chars: u32) -> TrajCall {
    tc("bash", &bash_a(cmd), "", chars, false)
}

fn ctx(
    dom: &Domain,
    budget: u32,
    calls: Vec<TrajCall>,
    skip: Vec<String>,
    tool: &str,
    args: &str,
    target: &str,
) -> SupervisorContext {
    let step = calls.len() as u32 + 1;
    SupervisorContext {
        task: dom.task.to_string(),
        budget,
        calls,
        skip_targets: skip,
        tool: tool.to_string(),
        args_text: args.to_string(),
        target: target.to_string(),
        sig: if tool == LOOP_TOOL { String::new() } else { format!("{tool}:{args}") },
        step,
    }
}

fn ex(c: SupervisorContext, y: f32, group: u32) -> Example {
    let desc = format!("{} {} {}", c.tool, c.target, c.args_text);
    let task = c.task.clone();
    Example { x: extract(&c).to_vec(), y, group, text: desc, task }
}

fn good_chars(rng: &mut Rng) -> u32 {
    800 + (rng.next_u32() % 3200)
}

/* ── Families ────────────────────────────────────────────────────── */

/// S0 (neg): a clean FIRST call — read the primary file or grep the
/// primary symbol, empty trajectory.
fn s0_first_call(out: &mut Vec<Example>, rng: &mut Rng, n: usize, domains: &[Domain], group: u32) {
    for i in 0..n {
        let dom = &domains[i % domains.len()];
        let (tool, args, target) = if i % 2 == 0 {
            ("read_file", read_a(dom.paths[0]), dom.paths[0].to_string())
        } else {
            ("grep", grep_a(dom.symbols[0]), dom.symbols[0].to_string())
        };
        out.push(ex(ctx(dom, 16, vec![], vec![], tool, &args, &target), 0.0, group));
    }
}

/// S1 (neg): clean FOLLOW-UP calls — the trajectory is healthy reads/
/// greps/a test run; the next call stays on task (later paths/symbols,
/// each echoing a task stem, or a build/test command).
fn s1_followup(out: &mut Vec<Example>, rng: &mut Rng, n: usize, domains: &[Domain], group: u32) {
    for i in 0..n {
        let dom = &domains[i % domains.len()];
        let mut calls = vec![read_call(dom.paths[0], good_chars(rng))];
        if i % 3 != 0 {
            calls.push(grep_call(dom.symbols[0], 300 + rng.next_u32() % 1200));
        }
        if i % 4 == 0 {
            calls.push(bash_call("cargo test", 400 + rng.next_u32() % 900));
        }
        let variant = i % 4;
        let (tool, args, target) = match variant {
            0 => ("read_file", read_a(dom.paths[1]), dom.paths[1].to_string()),
            1 => ("grep", grep_a(dom.symbols[1]), dom.symbols[1].to_string()),
            2 => ("bash", bash_a("cargo build"), String::new()),
            // Listing the directory that holds the file just read — the
            // "what else is here" follow-up. Without this the net only
            // ever sees list_dir as the degenerate root listing (S5).
            _ => {
                let dir = dir_of(dom.paths[0]);
                ("list_dir", read_a(&dir), dir)
            }
        };
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, &target), 0.0, group));
    }
}

/// S2 (pos): EXACT duplicate — an earlier call's tool+args reappear
/// verbatim (same canonical sig).
fn s2_exact_dup(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let dup = if i % 2 == 0 {
            read_call(dom.paths[0], good_chars(rng))
        } else {
            grep_call(dom.symbols[0], 600)
        };
        let mut calls = vec![dup.clone()];
        if i % 3 != 0 {
            calls.push(grep_call(dom.symbols[1], 500));
        }
        if i % 4 == 0 {
            calls.push(read_call(dom.paths[1], good_chars(rng)));
        }
        // The duplicate is the action; its original sits earlier.
        let c = ctx(dom, 16, calls, vec![], &dup.tool, &dup.sig.splitn(2, ':').nth(1).unwrap_or("").to_string(), &dup.target);
        out.push(ex(c, 1.0, 2));
    }
}

/// S3 (pos): re-reading an UNCHANGED file with different args (an offset
/// read) — the re-read rule's territory, learned not just ruled.
fn s3_reread_unchanged(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[i % dom.paths.len()];
        let mut calls = vec![read_call(p, good_chars(rng))];
        if i % 2 == 0 {
            calls.push(grep_call(dom.symbols[1], 450));
        }
        let off = 40 + (rng.next_u32() % 40) * 4;
        out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a_off(p, off), p), 1.0, 3));
    }
}

/// S4 (neg): re-reading a file AFTER a write to it — legitimate.
fn s4_reread_after_write(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[0];
        let calls = vec![read_call(p, good_chars(rng)), edit_call(p)];
        let off = 20 + (rng.next_u32() % 20) * 8;
        out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a_off(p, off), p), 0.0, 4));
    }
}

/// S5 (pos): TOO-BROAD calls — degenerate patterns and root listings.
fn s5_too_broad(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    let broads: &[(&str, &str, &str)] = &[
        ("grep", ".", "."),
        ("grep", "**", "**"),
        ("glob", "*", "*"),
        ("list_dir", "/", "/"),
        ("list_dir", "", ""),
        ("grep", "*", "*"),
    ];
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let (tool, pat, target) = broads[i % broads.len()];
        let args = if tool == "list_dir" {
            if pat.is_empty() { "{}".to_string() } else { read_a(pat) }
        } else {
            grep_a(pat)
        };
        let mut calls = vec![];
        if i % 3 == 0 {
            calls.push(read_call(dom.paths[0], good_chars(rng)));
        }
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, target), 1.0, 5));
    }
}

/// S6 (pos): OFF-TARGET calls — the target shares at most a symbol word
/// with the run, never a task word (the stem rule). Trajectory kept to
/// 0–1 calls so the class isn't separable by "has history" alone; the
/// similarity channels must carry it.
fn s6_off_task(out: &mut Vec<Example>, rng: &mut Rng, n: usize, domains: &[Domain], group: u32) {
    for i in 0..n {
        let dom = &domains[i % domains.len()];
        let mut calls = vec![];
        if i % 2 == 0 {
            calls.push(read_call(dom.paths[0], good_chars(rng)));
        }
        let (tool, args, target) = if i % 2 == 0 {
            ("grep", grep_a(dom.off_pattern), dom.off_pattern.to_string())
        } else {
            ("read_file", read_a(dom.off_path), dom.off_path.to_string())
        };
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, &target), 1.0, group));
    }
}

/// S7 (pos): SPINNING — repeated offset reads of one file, each
/// returning almost nothing; the action is yet another one.
fn s7_spinning(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[i % dom.paths.len()];
        let depth = 3 + (rng.next_u32() % 3) as u32; // 3-5 prior reads
        let calls: Vec<TrajCall> = (0..depth)
            .map(|k| read_call_off(p, k * 50, 15 + rng.next_u32() % 45))
            .collect();
        let off = depth * 50;
        out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a_off(p, off), p), 1.0, 7));
    }
}

/// S8 (pos): RETRYING an errored call with identical args.
fn s8_retry_errored(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let bad = if i % 2 == 0 {
            tc("grep", &grep_a(dom.symbols[1]), dom.symbols[1], 0, true)
        } else {
            tc("read_file", &read_a(dom.paths[1]), dom.paths[1], 0, true)
        };
        let mut calls = vec![read_call(dom.paths[0], good_chars(rng))];
        if i % 3 == 0 {
            calls.push(grep_call(dom.symbols[0], 700));
        }
        calls.push(bad.clone());
        let args = bad.sig.splitn(2, ':').nth(1).unwrap_or("").to_string();
        out.push(ex(ctx(dom, 16, calls, vec![], &bad.tool, &args, &bad.target), 1.0, 8));
    }
}

/// S9 (neg): narrowing after an error — same tool, TIGHTER args (a scoped
/// path, a more specific pattern). The advice "narrow it" was taken.
fn s9_narrow_retry(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let bad = tc("grep", &grep_a("error"), "error", 0, true);
        let calls = vec![read_call(dom.paths[0], good_chars(rng)), bad];
        let (args, target) = if i % 2 == 0 {
            (grep_a_in(dom.paths[0], dom.symbols[1]), dom.symbols[1].to_string())
        } else {
            (grep_a(dom.symbols[1]), dom.symbols[1].to_string())
        };
        out.push(ex(ctx(dom, 16, calls, vec![], "grep", &args, &target), 0.0, 9));
    }
}

/// S10 (pos): LATE off-task — over budget, the run drifts; the next call
/// is an unrelated wander. The interaction class (f31): the SAME call
/// early is merely curious, late it is waste.
fn s10_late_off_task(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let mut calls = Vec::new();
        let depth = 8 + (rng.next_u32() % 5) as usize; // 8-12 prior calls
        for k in 0..depth {
            if k == depth - 1 && i % 2 == 0 {
                // The run already started drifting.
                calls.push(grep_call(dom.off_pattern, 90));
            } else if k % 3 == 2 {
                calls.push(grep_call(dom.symbols[k % dom.symbols.len()], 300 + rng.next_u32() % 800));
            } else {
                calls.push(read_call(dom.paths[k % dom.paths.len()], good_chars(rng)));
            }
        }
        // step = calls+1 lands at 9-13 vs budget 8 → over.
        let c = ctx(dom, 8, calls, vec![], "grep", &grep_a(dom.off_pattern), dom.off_pattern);
        out.push(ex(c, 1.0, 10));
    }
}

/// S11 (neg): LATE but on-task and progressing — near the budget edge the
/// run is writing the fix. Budget pressure alone must not fire.
fn s11_late_on_task(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let mut calls = Vec::new();
        for k in 0..9 {
            if k % 4 == 3 {
                calls.push(edit_call(dom.paths[0]));
            } else if k % 3 == 2 {
                calls.push(grep_call(dom.symbols[k % dom.symbols.len()], 500));
            } else {
                calls.push(read_call(dom.paths[k % dom.paths.len()], good_chars(rng)));
            }
        }
        let p = dom.paths[0];
        out.push(ex(ctx(dom, 12, calls, vec![], "edit_file", &edit_a(p), p), 0.0, 11));
    }
}

/// S12 (neg): WRITE/EDIT progress — the run read the file, now it fixes
/// it. Write content length varies so the args-size channel is active.
fn s12_write_progress(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[i % dom.paths.len()];
        let mut calls = vec![read_call(p, good_chars(rng))];
        if i % 2 == 0 {
            calls.push(bash_call("cargo test", 600));
        }
        let (tool, args) = if i % 3 == 0 {
            ("write_file", write_a(p, 200 + (rng.next_u32() % 1300) as usize))
        } else {
            ("edit_file", edit_a(p))
        };
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, p), 0.0, 12));
    }
}

/// S13 (pos): the target is in the pre-turn "probably not needed" advice
/// (relevance skip set) — even though it looks on-task.
fn s13_skip_set(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let mut calls = vec![];
        if i % 2 == 0 {
            calls.push(read_call(dom.paths[0], good_chars(rng)));
        }
        let (tool, args, target) = if i % 2 == 0 {
            ("read_file", read_a(dom.paths[2]), dom.paths[2].to_string())
        } else {
            ("grep", grep_a(dom.symbols[1]), dom.symbols[1].to_string())
        };
        let skip = vec![target.clone()];
        out.push(ex(ctx(dom, 16, calls, skip, tool, &args, &target), 1.0, 13));
    }
}

/// S14 (pos): a tool that TWICE returned nothing useful is called again.
fn s14_useless_tool(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let (tool, mk): (&str, fn(&Domain, usize) -> (String, String)) = if i % 2 == 0 {
            // web_search args carry "query" — mlTarget() sees no path or
            // pattern, so the target is "" at runtime too.
            ("web_search", |d: &Domain, k: usize| (web_a(d.symbols[k % d.symbols.len()]), String::new()))
        } else {
            ("grep", |d: &Domain, k: usize| (grep_a(d.symbols[k % d.symbols.len()]), d.symbols[k % d.symbols.len()].to_string()))
        };
        let (a1, t1) = mk(dom, 0);
        let (a2, t2) = mk(dom, 1);
        let calls = vec![
            read_call(dom.paths[0], good_chars(rng)),
            tc(tool, &a1, &t1, rng.next_u32() % 60, false),
            tc(tool, &a2, &t2, rng.next_u32() % 60, false),
        ];
        let (a3, t3) = mk(dom, 2);
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &a3, &t3), 1.0, 14));
    }
}

/// S15 (pos, TRAIN): PARAPHRASED duplicates — same call, differently
/// spelled target (double slash, "./" prefix, a trailing-space pattern).
/// The sig differs; the near-dup cosine channel (trained on S18) must
/// catch it. Hard regime → TRAINING (the val fold stays clean; a hard
/// case the net can't yet do collapses threshold selection to the floor).
fn s15_paraphrase_dup(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[0];
        let calls = vec![read_call(p, good_chars(rng)), grep_call(dom.symbols[0], 500)];
        let paraphrased = match i % 3 {
            0 => p.replace('/', "//"),                    // src//payments//webhook.rs
            1 => format!("./{p}"),                        // ./src/payment/webhook.rs
            _ => format!("{} ", dom.symbols[0]),          // "verify_signature "
        };
        let (tool, args, target) = if i % 3 == 2 {
            ("grep", grep_a(&paraphrased), paraphrased)
        } else {
            ("read_file", read_a(&paraphrased), paraphrased)
        };
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, &target), 1.0, 15));
    }
}

fn s15_paraphrase_dup_in(out: &mut Vec<Example>, n: usize, dom: &Domain, group: u32) {
    for i in 0..n {
        let p = dom.paths[0];
        let calls = vec![read_call(p, 1200), grep_call(dom.symbols[0], 500)];
        let paraphrased = match i % 3 {
            0 => p.replace('/', "//"),
            1 => format!("./{p}"),
            _ => format!("{} ", dom.symbols[0]),
        };
        let (tool, args, target) = if i % 3 == 2 {
            ("grep", grep_a(&paraphrased), paraphrased)
        } else {
            ("read_file", read_a(&paraphrased), paraphrased)
        };
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, &target), 1.0, group));
    }
}

/// S16 (mixed, TEST): held-out DOMAINS — the core constructions replayed
/// in the gateway/audio domains, which no training family touches.
/// Balanced neg/pos per domain so the test fold's majority baseline is
/// not trivially strong.
fn s16_heldout_domains(out: &mut Vec<Example>, rng: &mut Rng) {
    for dom in TEST_DOMAINS {
        let doms = std::slice::from_ref(dom);
        s0_first_call(out, rng, 60, doms, 16);
        s1_followup(out, rng, 60, doms, 16);
        s2_dup_in(out, rng, 30, dom);
        s5_broad_in(out, rng, 30, dom);
        s6_off_task(out, rng, 30, doms, 16);
        s15_paraphrase_dup_in(out, 30, dom, 16);
    }
}

fn s2_dup_in(out: &mut Vec<Example>, rng: &mut Rng, n: usize, dom: &Domain) {
    for i in 0..n {
        let dup = read_call(dom.paths[0], good_chars(rng));
        let calls = vec![dup.clone(), grep_call(dom.symbols[1], 500)];
        let args = dup.sig.splitn(2, ':').nth(1).unwrap_or("").to_string();
        out.push(ex(ctx(dom, 16, calls, vec![], &dup.tool, &args, &dup.target), 1.0, 16));
    }
}

fn s5_broad_in(out: &mut Vec<Example>, rng: &mut Rng, n: usize, dom: &Domain) {
    let broads: &[(&str, &str, &str)] =
        &[("grep", ".", "."), ("glob", "*", "*"), ("list_dir", "/", "/")];
    for i in 0..n {
        let (tool, pat, target) = broads[i % broads.len()];
        let args = if tool == "list_dir" { read_a(pat) } else { grep_a(pat) };
        let mut calls = vec![];
        if i % 2 == 0 {
            calls.push(read_call(dom.paths[0], good_chars(rng)));
        }
        out.push(ex(ctx(dom, 16, calls, vec![], tool, &args, target), 1.0, 16));
    }
}

/// S17 (mixed, TRAIN): BOUNDARY samples — the mildly-broad and the
/// almost-legitimate, so threshold selection sees a real margin.
fn s17_boundary(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        match i % 4 {
            // pos: wildcard-heavy but not degenerate.
            0 => {
                let calls = vec![read_call(dom.paths[0], good_chars(rng))];
                out.push(ex(ctx(dom, 16, calls, vec![], "grep", &grep_a("src/**"), "src/**"), 1.0, 17));
            }
            // pos: third offset read of an unchanged file.
            1 => {
                let p = dom.paths[1];
                let calls = vec![read_call(p, 900), read_call_off(p, 50, 60)];
                out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a_off(p, 100), p), 1.0, 17));
            }
            // neg: re-read after a write two calls back.
            2 => {
                let p = dom.paths[0];
                let calls = vec![read_call(p, 1200), edit_call(p), grep_call(dom.symbols[1], 400)];
                out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a_off(p, 30), p), 0.0, 17));
            }
            // neg: a new on-task file mid-run.
            _ => {
                let calls = vec![read_call(dom.paths[0], good_chars(rng)), grep_call(dom.symbols[0], 700)];
                out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a(dom.paths[2]), dom.paths[2]), 0.0, 17));
            }
        }
    }
}

/// S18 (pos, TRAIN): NEAR-duplicates with a different sig — "./" prefixes
/// and double slashes preserve most char trigrams, so the cosine channel
/// (f4) learns "same call, respelled" from these.
fn s18_near_dup(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[i % dom.paths.len()];
        let calls = vec![read_call(p, good_chars(rng))];
        let respelled = if i % 2 == 0 { format!("./{p}") } else { p.replace('/', "//") };
        out.push(ex(ctx(dom, 16, calls, vec![], "read_file", &read_a(&respelled), &respelled), 1.0, 18));
    }
}

/// S19 (neg): SIMILAR-but-different — a second symbol from the same
/// domain (echoing a task stem) after the first. Overlap with the
/// trajectory must not read as duplication.
fn s19_similar_different(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let calls = vec![
            read_call(dom.paths[0], good_chars(rng)),
            grep_call(dom.symbols[0], 800),
        ];
        let sym = dom.symbols[1 + (i % (dom.symbols.len() - 1))];
        out.push(ex(ctx(dom, 16, calls, vec![], "grep", &grep_a(sym), sym), 0.0, 19));
    }
}

/// S20 (pos, TRAIN): LOOP boundary while spinning.
fn s20_loop_spinning(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let p = dom.paths[i % dom.paths.len()];
        let depth = 4 + (rng.next_u32() % 3) as u32;
        let calls: Vec<TrajCall> = (0..depth)
            .map(|k| read_call_off(p, k * 40, 10 + rng.next_u32() % 50))
            .collect();
        out.push(ex(ctx(dom, 12, calls, vec![], LOOP_TOOL, "", ""), 1.0, 20));
    }
}

/// S21 (neg, TRAIN): LOOP boundary while progressing.
fn s21_loop_progress(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let calls = vec![
            read_call(dom.paths[0], good_chars(rng)),
            grep_call(dom.symbols[1], 900),
            edit_call(dom.paths[0]),
            read_call_off(dom.paths[0], 40, 1100),
        ];
        out.push(ex(ctx(dom, 16, calls, vec![], LOOP_TOOL, "", ""), 0.0, 21));
    }
}

/// S22 (pos, TRAIN): LOOP boundary over budget with no recent progress.
fn s22_loop_stalled(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let mut calls = Vec::new();
        for k in 0..10 {
            if k % 3 == 2 {
                calls.push(tc("grep", &grep_a(dom.symbols[k % dom.symbols.len()]), dom.symbols[k % dom.symbols.len()], 0, true));
            } else {
                calls.push(grep_call(dom.symbols[k % dom.symbols.len()], 20 + rng.next_u32() % 40));
            }
        }
        out.push(ex(ctx(dom, 8, calls, vec![], LOOP_TOOL, "", ""), 1.0, 22));
    }
}

/// S23 (neg, TRAIN): LOOP boundary early and healthy.
fn s23_loop_early(out: &mut Vec<Example>, rng: &mut Rng, n: usize) {
    for i in 0..n {
        let dom = &TRAIN_DOMAINS[i % TRAIN_DOMAINS.len()];
        let depth = 1 + (rng.next_u32() % 3) as usize;
        let mut calls = Vec::new();
        for k in 0..depth {
            calls.push(read_call(dom.paths[k % dom.paths.len()], good_chars(rng)));
        }
        out.push(ex(ctx(dom, 16, calls, vec![], LOOP_TOOL, "", ""), 0.0, 23));
    }
}

/// S24 (neg, VAL): held-out CLEAN — first/follow-up constructions in the
/// fresh val domains. Keeps threshold selection honest and un-collapsed.
fn s24_heldout_clean(out: &mut Vec<Example>, rng: &mut Rng, n_each: usize) {
    for dom in VAL_DOMAINS {
        let doms = std::slice::from_ref(dom);
        s0_first_call(out, rng, n_each, doms, 24);
        s1_followup(out, rng, n_each, doms, 24);
    }
}

/// S26 (pos, VAL): held-out CLEAN POSITIVES — exact duplicates, too-broad
/// calls, AND off-task traps in the fresh val domains. The channels (f3/f5
/// dup, f16/f17 breadth, the zero-echo similarity profile) are trained on
/// S2/S5/S6; the val fold gets both classes ACROSS the decisive echo/
/// no-echo axis so threshold selection sees the real margin, not a
/// perfectly separable fold with no boundary samples.
fn s26_val_waste(out: &mut Vec<Example>, rng: &mut Rng, n_each: usize) {
    for dom in VAL_DOMAINS {
        let doms = std::slice::from_ref(dom);
        s2_dup_in(out, rng, n_each, dom);
        s5_broad_in(out, rng, n_each, dom);
        s6_off_task(out, rng, n_each, doms, 26);
        // Re-group the recently pushed examples (the shared builders tag
        // them 16/5/6) into the val family.
        for e in out.iter_mut().rev().take(n_each * 3) {
            e.group = 26;
        }
    }
}

/// S25 (pos, TEST): held-out TRAPS — fresh-domain off-task targets that
/// share a SYMBOL word (never a task word), plus the two-error retry.
fn s25_heldout_traps(out: &mut Vec<Example>, rng: &mut Rng, n_each: usize) {
    for dom in TEST_DOMAINS {
        let doms = std::slice::from_ref(dom);
        s6_off_task(out, rng, n_each, doms, 25);
        for i in 0..n_each {
            let bad = tc("read_file", &read_a(dom.paths[1]), dom.paths[1], 0, true);
            let calls = vec![read_call(dom.paths[0], good_chars(rng)), bad.clone(), bad.clone()];
            let args = bad.sig.splitn(2, ':').nth(1).unwrap_or("").to_string();
            out.push(ex(ctx(dom, 16, calls, vec![], &bad.tool, &args, &bad.target), 1.0, 25));
        }
    }
}

pub fn build() -> Vec<Example> {
    let mut rng = Rng(0x5150_7E11);
    let mut out = Vec::new();
    s0_first_call(&mut out, &mut rng, 240, TRAIN_DOMAINS, 0);
    s1_followup(&mut out, &mut rng, 320, TRAIN_DOMAINS, 1);
    s2_exact_dup(&mut out, &mut rng, 200);
    s3_reread_unchanged(&mut out, &mut rng, 180);
    s4_reread_after_write(&mut out, &mut rng, 180);
    s5_too_broad(&mut out, &mut rng, 200);
    s6_off_task(&mut out, &mut rng, 320, TRAIN_DOMAINS, 6);
    s7_spinning(&mut out, &mut rng, 180);
    s8_retry_errored(&mut out, &mut rng, 160);
    s9_narrow_retry(&mut out, &mut rng, 180);
    s10_late_off_task(&mut out, &mut rng, 180);
    s11_late_on_task(&mut out, &mut rng, 180);
    s12_write_progress(&mut out, &mut rng, 220);
    s13_skip_set(&mut out, &mut rng, 120);
    s14_useless_tool(&mut out, &mut rng, 160);
    s15_paraphrase_dup(&mut out, &mut rng, 180);
    s16_heldout_domains(&mut out, &mut rng);
    s17_boundary(&mut out, &mut rng, 200);
    s18_near_dup(&mut out, &mut rng, 200);
    s19_similar_different(&mut out, &mut rng, 260);
    s20_loop_spinning(&mut out, &mut rng, 120);
    s21_loop_progress(&mut out, &mut rng, 160);
    s22_loop_stalled(&mut out, &mut rng, 120);
    s23_loop_early(&mut out, &mut rng, 160);
    s24_heldout_clean(&mut out, &mut rng, 50);
    s25_heldout_traps(&mut out, &mut rng, 40);
    s26_val_waste(&mut out, &mut rng, 40);
    out
}
