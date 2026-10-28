// conformance.rs — the P5 per-release conformance report (`sofuu conformance`).
//
// One JSON document that says what THIS binary is and what it claims: the
// capability manifest, the live-vs-claimed capability posture (the R7
// install-integrity claim, evaluated at runtime), the platform, version, and
// — when the caller has one — the 52-case contract-suite results. A release
// attaches this next to the checksums (`scripts/conformance_report.sh` wraps
// suite + report into one artifact).

use serde_json::{json, Value};
use sofuu_ffi::SofuuRuntime;

/// Run the probe on `rt`, return the posture value.
fn probe_rt(rt: &SofuuRuntime) -> Result<Value, String> {
    // The probe is an async IIFE that writes the posture file; the loop is
    // drained with run_jobs so the file is on disk when eval returns.
    let rc = rt.eval_string(
        "globalThis.__probeResult = (async function () {\n  function t(expr) { try { return typeof eval(expr); } catch (e) { return 'error: ' + (e && e.message || e); } }\n  const mem_live = t('sofuu.memory && sofuu.memory.remember');\n  const mem_entry = mem_live === 'function'\n    ? { claimed: 'full', live: 'full', ok: true }\n    : { claimed: 'full', live: mem_live, ok: false, note: 'a valid Sofuu build must expose sofuu.memory.remember (QTSQ is a mandatory core component)' };\n  const posture = {\n    memory: mem_entry,\n    session_document: { claimed: 'full (P3)', live: t('sofuu.session && sofuu.session.export'),\n      ok: t('sofuu.session && sofuu.session.export') === 'function' },\n    events: { claimed: 'full (P2)', live: t('sofuu.events && sofuu.events.append && sofuu.events.render'),\n      ok: t('sofuu.events && sofuu.events.append && sofuu.events.render') === 'function' },\n    async_contract: { claimed: 'full (P2)', live: t('sofuu.async && sofuu.async.makeError'),\n      ok: t('sofuu.async && sofuu.async.makeError') === 'function' },\n    policy: { claimed: 'one evaluator (P4)', live: t('sofuu.policy && sofuu.policy.evaluate'),\n      ok: t('sofuu.policy && sofuu.policy.evaluate') === 'function' },\n  };\n  const path = '.sofuu/conformance-live-' + process.pid + '.json';\n  await sofuu.fs.writeFile(path, JSON.stringify(posture));\n  return path;\n})()",
        "<conformance>",
    );
    rt.run_jobs();
    if rc != 0 {
        return Err("the capability probe failed to evaluate".into());
    }
    let Ok(entries) = std::fs::read_dir(".sofuu") else {
        return Err("cannot read .sofuu for the probe result".into());
    };
    let mut found: Option<Value> = None;
    let mut stale: Vec<std::path::PathBuf> = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("conformance-live-") && name.ends_with(".json") {
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    if v.get("memory").is_some() {
                        found = Some(v);
                    }
                }
            }
            stale.push(e.path());
        }
    }
    for p in stale {
        let _ = std::fs::remove_file(p);
    }
    found.ok_or_else(|| "the probe wrote no posture file".to_string())
}

/// Build the per-release report. `suite` = the contract runner's results
/// when the caller has that file (`--suite`).
pub fn build_report(rt: &SofuuRuntime, version: &str, suite: Option<Value>) -> Result<Value, String> {
    let posture = probe_rt(rt)?;
    let manifest: Value =
        serde_json::from_str(&sofuu_core::capabilities::manifest_json()).unwrap_or(Value::Null);
    let all_ok = posture
        .as_object()
        .map(|o| o.values().all(|v| v.get("ok").and_then(Value::as_bool).unwrap_or(false)))
        .unwrap_or(false);
    Ok(json!({
        "schema_version": 1,
        "kind": "sofuu-conformance",
        "version": version,
        "platform": { "os": std::env::consts::OS, "arch": std::env::consts::ARCH },
        "generated_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "manifest": manifest,
        "capability_posture": posture,
        "capability_posture_ok": all_ok,
        "install_integrity": {
            "script": "scripts/verify-dist.sh (checksums + capabilities-vs-claim + version parity)",
            "runtime_part_ok": all_ok,
        },
        "contract_suite": suite.unwrap_or(Value::Null),
        "how_to_reproduce": "bash scripts/contract/run_contract_tests.sh; bash scripts/verify-dist.sh; sofuu conformance",
    }))
}

/// `sofuu conformance [--suite <file>]` — print the report JSON.
pub fn cmd_conformance(rt: &SofuuRuntime, version: &str, args: &[String]) -> i32 {
    let mut suite = None;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--suite" && i + 1 < args.len() {
            if let Ok(text) = std::fs::read_to_string(&args[i + 1]) {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    suite = Some(v);
                }
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    match build_report(rt, version, suite) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
            if report.get("capability_posture_ok").and_then(Value::as_bool) != Some(true) {
                eprintln!("\x1b[33mWarning:\x1b[0m capability posture has non-full entries (recorded in capability_posture; engine-lane findings stay visible — contract/matrix.json)");
            }
            // The report is the deliverable; the hard gate lives in
            // verify-dist.sh, which knows the build variant's claim.
            0
        }
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m conformance: {e}");
            1
        }
    }
}
