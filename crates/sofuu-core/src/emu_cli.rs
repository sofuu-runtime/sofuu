//! `sofuu ui` — headless screenshots and the interactive emulator launcher.
//!
//! `sofuu-core` never links `sofuu-ui`/`sofuu-gpu`. The snapshot path runs the
//! app's `sofuu.ui` code with a patch recorder (`ui_bridge::begin_headless`),
//! then hands the recorded `ScenePatch` stream to the standalone
//! `sofuu-emu` binary, which owns layout, rasterization, PNG and audit JSON.
//!
//! ```text
//! sofuu ui snapshot ./app.js --platform ios --mode annotated --fail-on-error
//! sofuu ui emu      ./app.js --platform android --watch
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use sofuu_ffi::SofuuRuntime;

use crate::ui_bridge;

pub const USAGE: &str = "\
Usage: sofuu ui snapshot <app.js> [options]
       sofuu ui emu <app.js> [options]
       sofuu ui profiles [--json]
       sofuu ui mcp [--name <server>]

Snapshot options:
  --platform <ios|android|macos|windows|linux>   target profile (default ios)
  --platforms <ios,android,...>                  filmstrip matrix (up to 6 tiles)
  --device <id>                                  device preset (default: profile default)
  --theme <light|dark>                           host + app theme
  --orientation <portrait|landscape>             rotate the device
  --keyboard-open, --keyboard                    composite the software IME
  --scenario <name|file>                         scenario name (exposed as
                                                 globalThis.__sofuu_emu_scenario)
                                                 or an alternate app entry file
  --props <json>                                 preview props passed to the app
  --mock                                         stub fetch/sofuu.ai locally
                                                 (no network, deterministic AI text)
  --offline                                      like --mock but unmatched fetches
                                                 reject with E_EMU_OFFLINE
  --mock-config <json|@file>                     mocked routes: {\"routes\":[{url,
                                                 method,status,json|text}]}
  --slowmo <factor>                              live pump pacing (2 = half speed)
  --actions <json|@file>                         action list (tap/type/submit/scroll/...)
  --mode <clean|annotated|filmstrip>             screenshot mode (default clean)
  --out <path>                                   PNG path
  --audit-json <path>                            write the AuditReport JSON
  --scale <n>                                    cap the physical scale
  --title <text>                                 emulator window title
  --fail-on-error                                exit 2 when the audit has errors
  --json                                         print the capture JSON only

Emulator options:
  --watch                                        hot-reload on file changes
  --headless                                     render one annotated snapshot
";

#[derive(Clone, Debug)]
struct UiOptions {
    app: String,
    platform: String,
    platforms: Vec<String>,
    device: Option<String>,
    theme: String,
    orientation: String,
    keyboard_open: bool,
    scenario: Option<String>,
    props: Value,
    mock: crate::emu_mock::MockOptions,
    slowmo: f32,
    actions: Value,
    mode: String,
    out: Option<String>,
    audit_json: Option<String>,
    scale: Option<f32>,
    title: Option<String>,
    fail_on_error: bool,
    json_only: bool,
    watch: bool,
    headless: bool,
    name: String,
}

impl Default for UiOptions {
    fn default() -> Self {
        Self {
            app: String::new(),
            platform: "ios".to_string(),
            platforms: Vec::new(),
            device: None,
            theme: "light".to_string(),
            orientation: "portrait".to_string(),
            keyboard_open: false,
            scenario: None,
            props: Value::Null,
            mock: crate::emu_mock::MockOptions::default(),
            slowmo: 1.0,
            actions: Value::Array(Vec::new()),
            mode: "clean".to_string(),
            out: None,
            audit_json: None,
            scale: None,
            title: None,
            fail_on_error: false,
            json_only: false,
            watch: false,
            headless: false,
            name: "sofuu-ui-emu".to_string(),
        }
    }
}

/// Entry point called from `main`'s `"ui"` arm.
pub fn cmd_ui(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let Some(sub) = args.get(2).map(String::as_str) else {
        eprintln!("{USAGE}");
        return 2;
    };
    match sub {
        "snapshot" | "shot" | "audit" => {
            let mut opts = match parse_options(&args[2..], false) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return 2;
                }
            };
            if sub == "audit" && opts.mode == "clean" {
                opts.mode = "annotated".to_string();
            }
            run_snapshot(rt, &opts)
        }
        "emu" | "open" => {
            let opts = match parse_options(&args[2..], true) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m {e}");
                    return 2;
                }
            };
            if opts.headless {
                let mut shot = opts.clone();
                shot.mode = if shot.mode == "clean" {
                    "annotated".to_string()
                } else {
                    shot.mode
                };
                return run_snapshot(rt, &shot);
            }
            run_interactive(rt, &opts)
        }
        "profiles" | "devices" => run_emu_binary(&["profiles", "--json"]),
        "mcp" => run_mcp(rt, &args[2..]),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            0
        }
        other => {
            eprintln!("\x1b[31mError:\x1b[0m unknown `sofuu ui {other}`");
            eprintln!("{USAGE}");
            2
        }
    }
}

fn parse_options(args: &[String], interactive: bool) -> Result<UiOptions, String> {
    let mut o = UiOptions::default();
    let mut i = 1; // args[0] is the subcommand
    while i < args.len() {
        let arg = args[i].as_str();
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg {
            "--platform" | "-p" => o.platform = take(&mut i)?,
            "--platforms" => {
                let raw = take(&mut i)?;
                o.platforms = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if o.platforms.is_empty() {
                    return Err("--platforms needs at least one profile id".to_string());
                }
            }
            "--device" | "-d" => o.device = Some(take(&mut i)?),
            "--theme" => o.theme = take(&mut i)?,
            "--orientation" | "--rotate" => o.orientation = take(&mut i)?,
            "--keyboard-open" | "--keyboard" | "--kb" => o.keyboard_open = true,
            "--scenario" => o.scenario = Some(take(&mut i)?),
            "--props" | "--data" => o.props = parse_json_arg(&take(&mut i)?)?,
            "--mock" => o.mock.enabled = true,
            "--offline" => {
                o.mock.enabled = true;
                o.mock.offline = true;
            }
            "--mock-config" => {
                let v = parse_json_arg(&take(&mut i)?)?;
                let mut mock = crate::emu_mock::MockOptions::from_json(&v)?;
                /* `--offline` may appear before the config; keep it. */
                mock.offline |= o.mock.offline;
                o.mock = mock;
            }
            "--slowmo" => {
                let raw = take(&mut i)?;
                let v = raw
                    .parse::<f32>()
                    .map_err(|_| format!("--slowmo expects a positive number, got `{raw}`"))?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(format!("--slowmo expects a positive number, got `{raw}`"));
                }
                o.slowmo = v;
            }
            "--actions" => o.actions = parse_json_arg(&take(&mut i)?)?,
            "--mode" | "-m" => o.mode = take(&mut i)?,
            "--out" | "-o" => o.out = Some(take(&mut i)?),
            "--audit-json" => o.audit_json = Some(take(&mut i)?),
            "--scale" => {
                let raw = take(&mut i)?;
                o.scale = Some(
                    raw.parse::<f32>()
                        .map_err(|_| format!("--scale expects a number, got `{raw}`"))?,
                );
            }
            "--title" => o.title = Some(take(&mut i)?),
            "--fail-on-error" | "--fail" => o.fail_on_error = true,
            "--json" => o.json_only = true,
            "--watch" => o.watch = true,
            "--headless" => o.headless = true,
            "--name" => o.name = take(&mut i)?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}`"));
            }
            other => {
                if o.app.is_empty() {
                    o.app = other.to_string();
                } else {
                    return Err(format!("unexpected argument `{other}`"));
                }
            }
        }
        i += 1;
    }
    if o.app.is_empty() {
        return Err("an app entry file is required (e.g. ./my_app.js)".to_string());
    }
    if interactive && o.mode == "clean" {
        // Interactive launches do not need a screenshot mode; keep clean.
    }
    Ok(o)
}

fn parse_json_arg(raw: &str) -> Result<Value, String> {
    let text = if let Some(path) = raw.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?
    } else {
        raw.to_string()
    };
    serde_json::from_str(&text).map_err(|e| format!("invalid JSON `{raw}`: {e}"))
}

fn normalise_platform(raw: &str) -> Result<String, String> {
    match raw.to_ascii_lowercase().as_str() {
        "ios" | "android" | "macos" | "windows" | "linux" => Ok(raw.to_ascii_lowercase()),
        other => Err(format!(
            "unknown platform `{other}` (expected ios, android, macos, windows or linux)"
        )),
    }
}

fn validate_options(o: &mut UiOptions) -> Result<(), String> {
    o.platform = normalise_platform(&o.platform)?;
    for platform in &mut o.platforms {
        *platform = normalise_platform(platform)?;
    }
    match o.theme.to_ascii_lowercase().as_str() {
        "light" | "dark" => o.theme = o.theme.to_ascii_lowercase(),
        other => return Err(format!("unknown theme `{other}` (expected light or dark)")),
    }
    match o.orientation.to_ascii_lowercase().as_str() {
        "portrait" | "landscape" => o.orientation = o.orientation.to_ascii_lowercase(),
        other => {
            return Err(format!(
                "unknown orientation `{other}` (expected portrait or landscape)"
            ))
        }
    }
    match o.mode.to_ascii_lowercase().as_str() {
        "clean" | "annotated" | "filmstrip" => o.mode = o.mode.to_ascii_lowercase(),
        other => {
            return Err(format!(
                "unknown mode `{other}` (expected clean, annotated or filmstrip)"
            ))
        }
    }
    if !o.platforms.is_empty() && o.mode != "filmstrip" {
        return Err("--platforms is only valid with --mode filmstrip".to_string());
    }
    if o.platforms.len() > 6 {
        return Err("--platforms supports at most 6 tiles".to_string());
    }
    if o.fail_on_error && o.mode == "filmstrip" {
        // Valid: filmstrip merges every tile's issues.
    }
    Ok(())
}

fn normalise_actions(value: &Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Null => Ok(Vec::new()),
        _ => Err("--actions must be a JSON array".to_string()),
    }
}

fn run_snapshot(rt: &SofuuRuntime, opts: &UiOptions) -> i32 {
    let mut o = opts.clone();
    if let Err(e) = validate_options(&mut o) {
        eprintln!("\x1b[31mError:\x1b[0m {e}");
        return 2;
    }
    let actions = match normalise_actions(&o.actions) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return 2;
        }
    };
    let app = resolve_app_entry(&o);
    if !Path::new(&app).exists() {
        eprintln!("\x1b[31mError:\x1b[0m app entry not found: {app}");
        return 2;
    }

    let patches = match capture_patches(rt, &app, &o, &actions) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return 1;
        }
    };
    if patches.is_empty() {
        eprintln!(
            "\x1b[31mError:\x1b[0m the app did not mount a sofuu.ui surface \
             (mount it in a top-level statement or call waitSettled first)"
        );
        return 1;
    }

    let platforms: Vec<String> = if o.platforms.is_empty() {
        vec![o.platform.clone()]
    } else {
        o.platforms.clone()
    };
    let patches_json: Vec<Value> = patches
        .iter()
        .map(|s| serde_json::from_str::<Value>(s).unwrap_or(Value::Null))
        .collect();
    /* E7: `scroll` is a gesture, not a semantic event, so it is replayed by
     * the emulator's engine after the patch stream instead of the JS host. */
    let scrolls: Vec<Value> = actions
        .iter()
        .filter(|a| {
            a.get("type")
                .and_then(Value::as_str)
                .map(|t| t.eq_ignore_ascii_case("scroll"))
                .unwrap_or(false)
        })
        .map(|a| {
            json!({
                "id": a.get("id").and_then(Value::as_str).unwrap_or(""),
                "delta_y": a
                    .get("delta_y")
                    .or_else(|| a.get("deltaY"))
                    .or_else(|| a.get("dy"))
                    .or_else(|| a.get("y"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
            })
        })
        .collect();
    let mut job = json!({
        "app": empty_scene(&o),
        "config": {
            "platform": platforms[0],
            "device": o.device,
            "theme": o.theme,
            "orientation": o.orientation,
            "keyboard_open": o.keyboard_open,
            "scenario": o.scenario,
            "title": o.title,
        },
        "patches": patches_json,
        "scrolls": scrolls,
        "mode": o.mode,
        "out": o.out,
        "audit_json": o.audit_json,
        "scale_limit": o.scale,
        "fail_on_error": o.fail_on_error,
    });
    if platforms.len() > 1 {
        // Same recorded patch stream, one tile per platform profile. Device
        // ids are resolved per platform (a phone preset is not valid on
        // macOS), falling back to that profile's default.
        let manifest = profiles_manifest();
        let tiles: Vec<Value> = platforms
            .iter()
            .map(|platform| {
                let mut device = o.device.clone();
                let mut label = platform.clone();
                if let Some(m) = manifest {
                    if let Some(res) =
                        resolve_live_profile(m, platform, o.device.as_deref(), &o.orientation)
                    {
                        device = res
                            .get("device")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .or(device);
                        let profile_name = res
                            .get("profileName")
                            .and_then(Value::as_str)
                            .unwrap_or(platform);
                        let device_name =
                            res.get("deviceName").and_then(Value::as_str).unwrap_or("");
                        label = format!("{profile_name} - {device_name}");
                    }
                }
                json!({
                    "label": label,
                    "config": {
                        "platform": platform,
                        "device": device,
                        "theme": o.theme,
                        "orientation": o.orientation,
                        "keyboard_open": o.keyboard_open,
                        "title": o.title,
                    },
                    "app": empty_scene(&o),
                    "patches": patches_json.clone(),
                    "scrolls": scrolls.clone(),
                })
            })
            .collect();
        job["tiles"] = Value::Array(tiles);
        if let Some(map) = job.as_object_mut() {
            map.remove("app");
        }
    }
    let capture = match spawn_emu(&job, &["render", "--job", "-"]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            return 1;
        }
    };
    if o.json_only {
        println!("{}", serde_json::to_string(&capture).unwrap_or_default());
    } else {
        print_capture_summary(&capture);
    }
    let failed = capture
        .get("audit")
        .and_then(|a| a.get("summary"))
        .and_then(|s| s.get("errors"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        > 0;
    if failed && o.fail_on_error {
        2
    } else {
        0
    }
}

/// Keeps the bounded eval drain active for the whole capture and restores the
/// normal run-until-idle contract on every exit path (including errors).
///
/// A real app is allowed to keep a dashboard/chart `setInterval` running
/// forever; without the bound, `eval_file`/`eval_string` would drain libuv
/// until that interval is cleared and the one-shot capture would hang.
struct CaptureDrainBudget;

impl CaptureDrainBudget {
    fn new() -> Self {
        crate::rt::engine::set_eval_drain_budget(150);
        Self
    }
}

impl Drop for CaptureDrainBudget {
    fn drop(&mut self) {
        crate::rt::engine::set_eval_drain_budget(0);
    }
}

/// Run the app once, apply scripted actions, and return every recorded patch.
fn capture_patches(
    rt: &SofuuRuntime,
    app: &str,
    o: &UiOptions,
    actions: &[Value],
) -> Result<Vec<String>, String> {
    ui_bridge::begin_headless();
    let _budget = CaptureDrainBudget::new();
    /* E8: export the resolved device metrics before the app evaluates. The
     * app viewport is the same size `EmulatorHost::app_size` lays out, so a
     * host-sized app can derive its environment without guessing. */
    let device_metrics = profiles_manifest()
        .and_then(|m| {
            resolve_device_metrics(
                m,
                &o.platform,
                o.device.as_deref(),
                &o.orientation,
                o.keyboard_open,
            )
        })
        .unwrap_or_else(|| {
            json!({
                "screenWidth": 0.0,
                "screenHeight": 0.0,
                "width": 0.0,
                "height": 0.0,
                "scale": 1.0,
                "safe": { "top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0 },
                "keyboardHeight": 0.0,
            })
        });
    let prelude = format!(
        "globalThis.__sofuu_emu_scenario = {};\n\
         globalThis.__sofuu_emu_props = {};\n\
         globalThis.__sofuu_emu_slowmo = {};\n\
         globalThis.__sofuu_emu_options = {};\n\
         globalThis.__sofuu_emu_device = {};\n{}\
",
        json!(o.scenario),
        o.props,
        o.slowmo,
        json!({
            "platform": o.platform,
            "device": o.device,
            "theme": o.theme,
            "orientation": o.orientation,
            "keyboardOpen": o.keyboard_open,
        }),
        device_metrics,
        if o.mock.enabled {
            o.mock.prelude()
        } else {
            String::new()
        }
    );
    if rt.eval_string(&prelude, "<sofuu-emu-prelude>") != 0 {
        ui_bridge::end_headless();
        return Err("failed to install emulator globals".to_string());
    }
    /* A real app is allowed to keep a dashboard/chart `setInterval` running
     * forever. `eval_file` normally drains libuv until idle, which would hang
     * the one-shot capture; `CaptureDrainBudget` bounds every eval drain until
     * this function returns so the first frames land. */
    let rc = rt.eval_file(app);
    rt.run_jobs();
    let _ = rt.eval_string(
        "if (typeof sofuu !== 'undefined' && sofuu.ui && sofuu.ui.pump) sofuu.ui.pump();",
        "<sofuu-emu-pump>",
    );
    rt.run_jobs();
    if rc != 0 {
        ui_bridge::end_headless();
        return Err(format!("app failed to evaluate: {app}"));
    }
    if !ui_bridge::headless_mounted() {
        // Give one more microtask turn a chance (top-level `await` mounts).
        for _ in 0..4 {
            rt.run_jobs();
            let _ = rt.eval_string(
                "if (typeof sofuu !== 'undefined' && sofuu.ui && sofuu.ui.pump) sofuu.ui.pump();",
                "<sofuu-emu-pump>",
            );
        }
    }

    for action in actions {
        apply_action(action)?;
        rt.run_jobs();
        let _ = rt.eval_string(
            "if (typeof sofuu !== 'undefined' && sofuu.ui && sofuu.ui.pump) sofuu.ui.pump();",
            "<sofuu-emu-pump>",
        );
        rt.run_jobs();
    }
    let patches = ui_bridge::headless_take_patches();
    ui_bridge::end_headless();
    Ok(patches)
}

fn apply_action(action: &Value) -> Result<(), String> {
    let ty = action
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let id = action
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if matches!(ty.as_str(), "wait" | "wait_settled" | "settle" | "sleep") {
        return Ok(());
    }
    let event = match ty.as_str() {
        "tap" | "click" => json!({"type":"click","id": id}),
        "type" | "input" | "set_text" => json!({
            "type":"text_changed",
            "id": id,
            "value": action.get("text").or_else(|| action.get("value")).and_then(Value::as_str).unwrap_or(""),
        }),
        "submit" | "enter" | "press_enter" => json!({
            "type":"input_submitted",
            "id": id,
            "value": action.get("text").or_else(|| action.get("value")).and_then(Value::as_str).unwrap_or(""),
        }),
        "toggle" | "switch" => json!({
            "type":"switch_toggled",
            "id": id,
            "checked": action.get("checked").and_then(Value::as_bool).unwrap_or(true),
        }),
        "segment" | "segmented" | "select_segment" => json!({
            "type":"segmented_changed",
            "id": id,
            "index": action.get("index").and_then(Value::as_u64).unwrap_or(0),
        }),
        "tab" | "select_tab" => json!({
            "type":"tab_changed",
            "id": id,
            "index": action.get("index").and_then(Value::as_u64).unwrap_or(0),
        }),
        "select_row" => json!({
            "type":"row_selected",
            "id": id,
            "row_id": action.get("row").or_else(|| action.get("row_id")).and_then(Value::as_str).unwrap_or(""),
        }),
        "scroll" => {
            // Scroll is a gesture, not a semantic UIEvent; replaying it with
            // a synthetic pointer sequence is left to the interactive shell.
            return Ok(());
        }
        other => {
            return Err(format!(
            "unknown action type `{other}` (tap, type, submit, toggle, segment, tab, wait_settled)"
        ))
        }
    };
    if id.is_empty() && ty != "wait_settled" {
        return Err(format!("action `{ty}` needs an `id`"));
    }
    if !ui_bridge::headless_queue_event(&event.to_string()) {
        return Err("headless UI recorder is not active".to_string());
    }
    Ok(())
}

fn resolve_app_entry(o: &UiOptions) -> String {
    if let Some(scenario) = &o.scenario {
        if (scenario.ends_with(".js") || scenario.ends_with(".ts") || scenario.ends_with(".mjs"))
            && Path::new(scenario).exists()
        {
            return scenario.clone();
        }
    }
    o.app.clone()
}

fn empty_scene(o: &UiOptions) -> Value {
    json!({
        "title": o.title.clone().unwrap_or_else(|| "Sofuu UI".to_string()),
        "theme": o.theme,
        "root": {
            "id": "emu_app_root",
            "width": {"fill": 1.0},
            "height": {"fill": 1.0},
            "kind": {
                "type": "container",
                "axis": "column",
                "align": "stretch",
                "gap": 0.0,
                "fold_width": null,
                "scrollable": false,
                "style": null,
                "children": []
            }
        }
    })
}

fn print_capture_summary(capture: &Value) {
    let audit = capture.get("audit").cloned().unwrap_or(Value::Null);
    let summary = audit.get("summary").cloned().unwrap_or(Value::Null);
    let errors = summary.get("errors").and_then(Value::as_u64).unwrap_or(0);
    let warnings = summary.get("warnings").and_then(Value::as_u64).unwrap_or(0);
    let nodes = summary
        .get("interactive_nodes")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mode = capture.get("mode").and_then(Value::as_str).unwrap_or("?");
    let platform = capture
        .get("platform")
        .and_then(Value::as_str)
        .unwrap_or("?");
    let device = capture.get("device").and_then(Value::as_str).unwrap_or("?");
    println!("\x1b[1msofuu ui snapshot\x1b[0m");
    println!("  {platform} · {device} · {mode}");
    if let Some(png) = capture.get("capture_png").and_then(Value::as_str) {
        println!("  PNG:   {png}");
    }
    if let Some(path) = audit.get("capture_png").and_then(Value::as_str) {
        if Some(path) != capture.get("capture_png").and_then(Value::as_str) {
            println!("  audit: {path}");
        }
    }
    println!("  Nodes: {nodes} interactive · {errors} error(s) · {warnings} warning(s)");
    for issue in audit
        .get("issues")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let tag = issue.get("tag").and_then(Value::as_u64).unwrap_or(0);
        let sev = issue.get("severity").and_then(Value::as_str).unwrap_or("?");
        let kind = issue.get("kind").and_then(Value::as_str).unwrap_or("?");
        let id = issue.get("node_id").and_then(Value::as_str).unwrap_or("?");
        let detail = issue.get("detail").and_then(Value::as_str).unwrap_or("");
        let marker = if sev == "error" {
            "\x1b[31m✗\x1b[0m"
        } else {
            "\x1b[33m!\x1b[0m"
        };
        println!("  {marker} [{tag}] {kind} · {id}\n      {detail}");
    }
    if errors == 0 {
        println!("  \x1b[32m✓ no layout/contrast errors\x1b[0m");
    }
    if let Some(ms) = capture.get("capture_ms").and_then(Value::as_f64) {
        let renderer = capture
            .get("renderer")
            .and_then(Value::as_str)
            .unwrap_or("software");
        println!("  \x1b[90m{ms:.1} ms · renderer {renderer}\x1b[0m");
    }
}

fn run_interactive(_rt: &SofuuRuntime, opts: &UiOptions) -> i32 {
    let mut o = opts.clone();
    if let Err(e) = validate_options(&mut o) {
        eprintln!("\x1b[31mError:\x1b[0m {e}");
        return 2;
    }
    let app = resolve_app_entry(&o);
    if !Path::new(&app).exists() {
        eprintln!("\x1b[31mError:\x1b[0m app entry not found: {app}");
        return 2;
    }
    let title = o
        .title
        .clone()
        .unwrap_or_else(|| format!("Sofuu Emu - {}", o.platform));

    /* The profiles manifest comes from the companion binary: core never
     * links sofuu-emu, and the JS shell needs device geometry to paint the
     * platform-exact frame. When the binary is missing the app still runs
     * (chrome falls back to a phone-shaped default). */
    let manifest = profiles_manifest();
    let resolved = manifest
        .and_then(|m| resolve_live_profile(m, &o.platform, o.device.as_deref(), &o.orientation))
        .unwrap_or_else(|| json!({}));
    let profiles = manifest
        .and_then(|m| m.get("profiles").cloned())
        .unwrap_or_else(|| json!([]));

    let device_label = resolved
        .get("deviceName")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| o.device.clone())
        .unwrap_or_else(|| "default device".to_string());
    eprintln!(
        "\x1b[1msofuu ui emu\x1b[0m  {} · {} · {} · {}",
        o.platform,
        device_label,
        o.theme,
        if o.keyboard_open {
            "keyboard open"
        } else {
            "keyboard closed"
        }
    );
    eprintln!("\x1b[90m  strip: platform · device · theme · rotate · keys · bounds · snapshot · watch · pause/step (⌘. collapses)\x1b[0m");

    let live_config = json!({
        "enabled": true,
        "platform": o.platform.clone(),
        "device": resolved.get("device").cloned().unwrap_or(json!(o.device.clone())),
        "theme": o.theme.clone(),
        "orientation": o.orientation.clone(),
        "keyboardOpen": o.keyboard_open,
        "watch": o.watch,
        "scenario": o.scenario.clone(),
        "title": title.clone(),
        "width": resolved.get("width").cloned().unwrap_or(Value::Null),
        "height": resolved.get("height").cloned().unwrap_or(Value::Null),
        "scale": resolved.get("scale").cloned().unwrap_or(Value::Null),
        "profileName": resolved.get("profileName").cloned().unwrap_or(Value::Null),
        "deviceName": resolved.get("deviceName").cloned().unwrap_or(Value::Null),
        "profiles": profiles,
    });
    let device_metrics = manifest
        .as_ref()
        .and_then(|m| {
            resolve_device_metrics(
                m,
                &o.platform,
                o.device.as_deref(),
                &o.orientation,
                o.keyboard_open,
            )
        })
        .unwrap_or_else(|| {
            json!({
                "screenWidth": 0.0,
                "screenHeight": 0.0,
                "width": 0.0,
                "height": 0.0,
                "scale": 1.0,
                "safe": { "top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0 },
                "keyboardHeight": 0.0,
            })
        });
    let prelude = format!(
        "globalThis.__sofuu_emu_scenario = {};\n\
         globalThis.__sofuu_emu_props = {};\n\
         globalThis.__sofuu_emu_slowmo = {};\n\
         globalThis.__sofuu_emu_live = {};\n\
         globalThis.__sofuu_emu_device = {};\n{}\
",
        json!(o.scenario),
        o.props,
        o.slowmo,
        live_config,
        device_metrics,
        if o.mock.enabled {
            o.mock.prelude()
        } else {
            String::new()
        }
    );

    /* The live loop owns the pump so it can also watch sources and service
     * the toolbar's snapshot button; build the snapshot argv once. */
    let mut capture_args = vec![
        "ui".to_string(),
        "snapshot".to_string(),
        app.clone(),
        "--platform".to_string(),
        o.platform.clone(),
        "--theme".to_string(),
        o.theme.clone(),
        "--orientation".to_string(),
        o.orientation.clone(),
    ];
    if let Some(device) = &o.device {
        capture_args.push("--device".to_string());
        capture_args.push(device.clone());
    }
    if o.keyboard_open {
        capture_args.push("--keyboard-open".to_string());
    }
    if let Some(scenario) = &o.scenario {
        capture_args.push("--scenario".to_string());
        capture_args.push(scenario.clone());
    }
    if !o.props.is_null() {
        capture_args.push("--props".to_string());
        capture_args.push(o.props.to_string());
    }
    /* The toolbar snapshot is a fresh child, so mocked routes must travel with
     * it or a "Submit" capture would try the real network. */
    if o.mock.enabled {
        capture_args.push("--mock".to_string());
        if o.mock.offline {
            capture_args.push("--offline".to_string());
        }
        if !o.mock.routes.is_empty() {
            capture_args.push("--mock-config".to_string());
            capture_args.push(o.mock.to_json().to_string());
        }
    }
    let mut watch = o.watch;
    let app_path = app.clone();
    /* `--slowmo` scales the live pump, not the deterministic replay: a 2.0
     * factor halves animation speed for inspection/debugging. */
    let frame_ms = (16.0 * o.slowmo).clamp(1.0, 250.0);
    let worker = move || -> i32 {
        // `--ui` runs the app on the worker thread, exactly like
        // `sofuu run --ui`; the GPU window owns thread 0.
        let rt = match SofuuRuntime::init() {
            Some(rt) => rt,
            None => {
                eprintln!("\x1b[31mFatal:\x1b[0m runtime init failed");
                return 1;
            }
        };
        if rt.eval_string(&prelude, "<sofuu-emu-prelude>") != 0 {
            eprintln!("\x1b[31mError:\x1b[0m failed to install emulator globals");
            return 1;
        }
        /* Live apps are long-lived: bound the startup drain so a dashboard's
         * `setInterval` cannot block the worker before the pump loop starts.
         * 60 ms keeps `sofuu ui emu` launch inside the E3 gate; an async mount
         * that is still pending is picked up by the grace loop below. */
        crate::rt::engine::set_eval_drain_budget(60);
        let rc = rt.eval_file(&app_path);
        crate::rt::engine::set_eval_drain_budget(1);
        if rc != 0 {
            return rc;
        }
        rt.run_jobs();
        if !ui_bridge::is_ui_attached() {
            /* Top-level `await` / an async mount may still be in flight: give
             * it two seconds of pump ticks before treating the entry as a
             * non-UI script (which would close the window immediately). */
            let deadline = Instant::now() + Duration::from_secs(2);
            while !ui_bridge::is_ui_attached()
                && !ui_bridge::window_closed()
                && Instant::now() < deadline
            {
                let _ = rt.eval_string(
                    "if (typeof sofuu !== 'undefined' && sofuu.ui && sofuu.ui.pump) sofuu.ui.pump();",
                    "<sofuu-emu-pump>",
                );
                rt.run_jobs();
                std::thread::sleep(Duration::from_millis(16));
            }
            if !ui_bridge::is_ui_attached() {
                return rc;
            }
        }
        // The JS shell skips its own setInterval pump while emu-live is set;
        // this loop forwards host events and drives `sofuu.ui.pump()`.
        let mut newest = newest_source_mtime(&app_path);
        loop {
            if ui_bridge::window_closed() {
                break;
            }
            let mut closed = false;
            let mut had_event = false;
            while let Ok(Some(json)) = ui_bridge::poll_event_raw() {
                if json.contains("\"window_closed\"") {
                    closed = true;
                }
                had_event = true;
                ui_bridge::queue_host_event(&json);
            }
            while let Some(action) = ui_bridge::poll_host_action_raw() {
                had_event = true;
                /* `watch` is host-only: the strip toggles the Rust reload
                 * watcher, not a JS scene flag. */
                if action.contains("\"watch:toggle\"") {
                    watch = !watch;
                    eprintln!(
                        "\x1b[90m  watch {}\x1b[0m",
                        if watch { "on" } else { "off" }
                    );
                    continue;
                }
                ui_bridge::queue_host_event(&action);
            }
            /* Skip the JS pump when nothing happened and the app has no
             * timers/pending I/O: a static scene then costs one sleep per
             * frame instead of an eval, which is what keeps an idle emulator
             * near 0% CPU. */
            if had_event || crate::rt::event_loop::sofuu_loop_alive() {
                let _ = rt.eval_string(
                    "if (typeof sofuu !== 'undefined' && sofuu.ui && sofuu.ui.pump) sofuu.ui.pump();",
                    "<sofuu-emu-pump>",
                );
            }
            rt.run_jobs();
            if closed {
                break;
            }
            if let Some(mode) = ui_bridge::take_emu_capture_request() {
                let mut args = capture_args.clone();
                if let Some(env) = ui_bridge::take_emu_capture_env() {
                    if let Ok(v) = serde_json::from_str::<Value>(&env) {
                        apply_capture_env(&mut args, &v);
                    }
                }
                args.push("--mode".to_string());
                args.push(mode.to_string());
                spawn_live_capture(&args);
            }
            if watch {
                if let Some(modified) = newest_source_mtime(&app_path) {
                    if Some(modified) != newest {
                        newest = Some(modified);
                        eprintln!("\x1b[90m  ↻ reload {}\x1b[0m", app_path);
                        crate::rt::engine::set_eval_drain_budget(250);
                        let reload = rt.eval_file(&app_path);
                        crate::rt::engine::set_eval_drain_budget(1);
                        rt.run_jobs();
                        if reload != 0 {
                            eprintln!("\x1b[31mError:\x1b[0m reload failed in {app_path}");
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(frame_ms as u64));
        }
        0
    };
    let spec = live_window_spec(&resolved, &o.platform, manifest);
    let rc = ui_bridge::run_with_ui_spec(&title, Some(&spec), worker);
    if rc == std::process::ExitCode::SUCCESS {
        0
    } else {
        1
    }
}

/// Screen-exact window contract for the live emulator (E6). The client area is
/// the resolved device size in points; the native strip is attached outside
/// the render path.
fn live_window_spec(
    resolved: &Value,
    platform: &str,
    manifest: Option<&Value>,
) -> ui_bridge::UiWindowSpec {
    ui_bridge::UiWindowSpec {
        width: resolved
            .get("width")
            .and_then(Value::as_f64)
            .unwrap_or(1080.0) as f32,
        height: resolved
            .get("height")
            .and_then(Value::as_f64)
            .unwrap_or(720.0) as f32,
        decorations: false,
        resizable: false,
        scale: resolved
            .get("scale")
            .and_then(Value::as_f64)
            .map(|v| v as f32),
        corner_radius: resolved
            .get("corner_radius")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32,
        strip: true,
        reticle: true,
        platform: platform.to_string(),
        device: resolved
            .get("device")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        profiles_json: manifest.map(|m| m.to_string()).unwrap_or_default(),
    }
}

/// `sofuu-emu profiles --json` — the geometry the JS shell needs. Never
/// fatal: a missing companion just means the app opens without exact chrome.
fn load_profiles_manifest() -> Option<Value> {
    let exe = find_emu_binary().ok()?;
    let out = Command::new(exe)
        .args(["profiles", "--json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice::<Value>(&out.stdout).ok()
}

/// Resolve the profile/device/orientation into the live window's geometry.
fn resolve_live_profile(
    manifest: &Value,
    platform: &str,
    device: Option<&str>,
    orientation: &str,
) -> Option<Value> {
    let profiles = manifest.get("profiles")?.as_array()?;
    let profile = profiles
        .iter()
        .find(|p| p.get("id").and_then(Value::as_str) == Some(platform))?;
    let devices = profile.get("devices")?.as_array()?;
    let dev = device
        .and_then(|id| {
            devices.iter().find(|d| {
                d.get("id").and_then(Value::as_str) == Some(id)
                    || d.get("name").and_then(Value::as_str) == Some(id)
            })
        })
        .or_else(|| devices.first())?;
    let mut width = dev.get("width").and_then(Value::as_f64).unwrap_or(0.0);
    let mut height = dev.get("height").and_then(Value::as_f64).unwrap_or(0.0);
    if orientation == "landscape" {
        std::mem::swap(&mut width, &mut height);
    }
    Some(json!({
        "width": width,
        "height": height,
        "scale": dev.get("scale").and_then(Value::as_f64).unwrap_or(1.0),
        "corner_radius": dev.get("corner_radius").and_then(Value::as_f64).unwrap_or(0.0),
        "device": dev.get("id").cloned().unwrap_or(Value::Null),
        "deviceName": dev.get("name").cloned().unwrap_or(Value::Null),
        "profileName": profile.get("name").cloned().unwrap_or(Value::Null),
    }))
}

/// Cached `sofuu-emu profiles --json` manifest. The companion binary is
/// static for the life of this process, so one spawn is enough even when a
/// filmstrip resolves several platforms.
fn profiles_manifest() -> Option<&'static Value> {
    static MANIFEST: OnceLock<Option<Value>> = OnceLock::new();
    MANIFEST.get_or_init(load_profiles_manifest).as_ref()
}

/// Resolve the exact metrics a host passes to the app.
///
/// `width`/`height` are the app viewport after the status/title bar and home
/// bar are removed (the same values `EmulatorHost::app_size` uses), while
/// `screenWidth`/`screenHeight` stay the full device size. `safe` is the
/// physical inset set after the landscape swap, so an edge-to-edge app can
/// draw its own chrome. `keyboardHeight` is 0 unless the IME is open.
fn resolve_device_metrics(
    manifest: &Value,
    platform: &str,
    device: Option<&str>,
    orientation: &str,
    keyboard_open: bool,
) -> Option<Value> {
    let profiles = manifest.get("profiles")?.as_array()?;
    let profile = profiles
        .iter()
        .find(|p| p.get("id").and_then(Value::as_str) == Some(platform))?;
    let devices = profile.get("devices")?.as_array()?;
    let dev = device
        .and_then(|id| {
            devices.iter().find(|d| {
                d.get("id").and_then(Value::as_str) == Some(id)
                    || d.get("name").and_then(Value::as_str) == Some(id)
            })
        })
        .or_else(|| devices.first())?;
    let num = |key: &str| dev.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    let mut screen_w = num("width");
    let mut screen_h = num("height");
    let mut safe_top = num("safe_top");
    let mut safe_bottom = num("safe_bottom");
    let mut safe_left = num("safe_left");
    let mut safe_right = num("safe_right");
    let status_bar = num("status_bar_height");
    let titlebar = num("titlebar_height");
    let home_bar = num("home_bar_height");
    let keyboard_portrait = num("keyboard_portrait");
    let keyboard_landscape = num("keyboard_landscape");
    if orientation == "landscape" {
        std::mem::swap(&mut screen_w, &mut screen_h);
        // Mirrors `DevicePreset::geometry`: the notch/island moves to a side
        // edge, so the nominal top clearance becomes a left/right inset.
        let old_top = safe_top;
        safe_top = 0.0;
        safe_left = old_top.max(safe_left);
        safe_right = old_top.max(safe_right);
    }
    let top = if status_bar > 0.0 {
        status_bar.max(safe_top)
    } else {
        titlebar
    };
    let app_w = (screen_w - safe_left - safe_right).max(1.0);
    let app_h = (screen_h - top - home_bar).max(1.0);
    let keyboard = if keyboard_open {
        if orientation == "landscape" {
            keyboard_landscape
        } else {
            keyboard_portrait
        }
    } else {
        0.0
    };
    let scale = num("scale");
    Some(json!({
        "screenWidth": screen_w,
        "screenHeight": screen_h,
        "width": app_w,
        "height": app_h,
        "scale": if scale > 0.0 { scale } else { 1.0 },
        "safe": {
            "top": safe_top,
            "bottom": safe_bottom,
            "left": safe_left,
            "right": safe_right,
        },
        "keyboardHeight": keyboard,
    }))
}

/// Source watch for `--watch`: the entry plus its sibling `.js` libraries.
fn newest_source_mtime(app: &str) -> Option<std::time::SystemTime> {
    let path = Path::new(app);
    let mut newest = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    if let Some(dir) = path.parent() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) != Some("js") {
                    continue;
                }
                if let Ok(t) = entry.metadata().and_then(|m| m.modified()) {
                    if newest.map(|n| t > n).unwrap_or(true) {
                        newest = Some(t);
                    }
                }
            }
        }
    }
    newest
}

/// Replace (or drop) a `--flag value` pair in a capture argv.
fn override_capture_flag(args: &mut Vec<String>, flag: &str, value: Option<&str>) {
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag {
            args.remove(i);
            if i < args.len() && !args[i].starts_with("--") {
                args.remove(i);
            }
        } else {
            i += 1;
        }
    }
    if let Some(v) = value {
        args.push(flag.to_string());
        args.push(v.to_string());
    }
}

/// Apply the toolbar's live env to a snapshot argv.
fn apply_capture_env(args: &mut Vec<String>, env: &Value) {
    let text = |key: &str| env.get(key).and_then(Value::as_str).map(str::to_string);
    if let Some(platform) = text("platform") {
        override_capture_flag(args, "--platform", Some(&platform));
    }
    override_capture_flag(args, "--device", text("device").as_deref());
    if let Some(theme) = text("theme") {
        override_capture_flag(args, "--theme", Some(&theme));
    }
    if let Some(orientation) = text("orientation") {
        override_capture_flag(args, "--orientation", Some(&orientation));
    }
    let keyboard = env
        .get("keyboard")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--keyboard-open" || args[i] == "--keyboard" || args[i] == "--kb" {
            args.remove(i);
        } else {
            i += 1;
        }
    }
    if keyboard {
        args.push("--keyboard-open".to_string());
    }
}

/// Capture the live app from the toolbar's Snapshot button (headless child;
/// the window keeps running).
fn spawn_live_capture(args: &[String]) {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m snapshot: {e}");
            return;
        }
    };
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    match cmd.spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
        Err(e) => eprintln!("\x1b[31mError:\x1b[0m snapshot failed to start: {e}"),
    }
}

fn run_mcp(rt: &SofuuRuntime, args: &[String]) -> i32 {
    let mut name = "sofuu-ui-emu".to_string();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--name" {
            i += 1;
            if let Some(v) = args.get(i) {
                name = v.clone();
            }
        }
        i += 1;
    }
    let code = format!(
        r#"(function () {{
  if (!sofuu.emu || typeof sofuu.emu.mcp !== 'function') {{
    throw new Error('sofuu.emu is unavailable in this build');
  }}
  var srv = sofuu.emu.mcp({{ name: {name:?} }});
  srv.start();
}})();"#
    );
    let rc = rt.eval_string(&code, "<sofuu-emu-mcp>");
    if rc == 0 {
        0
    } else {
        1
    }
}

fn run_emu_binary(args: &[&str]) -> i32 {
    match find_emu_binary().and_then(|exe| {
        Command::new(exe)
            .args(args)
            .status()
            .map_err(|e| format!("failed to start sofuu-emu: {e}"))
    }) {
        Ok(status) if status.success() => 0,
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m {e}");
            1
        }
    }
}

/// Run `sofuu-emu <args>` with `job` on stdin and parse the single JSON line
/// written to stdout.
fn spawn_emu(job: &Value, args: &[&str]) -> Result<Value, String> {
    use std::io::Write as _;
    let exe = find_emu_binary()?;
    let mut child = Command::new(&exe)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start {}: {e}", exe.display()))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(job.to_string().as_bytes())
            .map_err(|e| format!("write render job: {e}"))?;
    }
    drop(child.stdin.take());
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait for sofuu-emu: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    /* `--fail-on-error` makes sofuu-emu exit 2 AFTER printing the capture
     * JSON. The caller owns the pass/fail decision (it prints the report and
     * exits 2 itself), so a JSON payload wins over the status code. */
    if let Some(line) = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
    {
        return serde_json::from_str(line).map_err(|e| format!("invalid sofuu-emu output: {e}"));
    }
    if !out.status.success() {
        return Err(format!(
            "sofuu-emu exited with {}{}",
            out.status.code().unwrap_or(-1),
            if stderr.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", stderr.trim())
            }
        ));
    }
    Err(format!(
        "sofuu-emu produced no JSON{}{}",
        if stdout.trim().is_empty() { "" } else { ": " },
        stdout.trim()
    ))
}

/// Locate the companion binary next to `sofuu`, in the workspace target dir,
/// on `PATH`, or via `SOFUU_EMU_BIN`.
pub fn find_emu_binary() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("SOFUU_EMU_BIN") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "SOFUU_EMU_BIN does not point at a file: {}",
            path.display()
        ));
    }
    let name = if cfg!(windows) {
        "sofuu-emu.exe"
    } else {
        "sofuu-emu"
    };
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(name));
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("release").join(name));
                candidates.push(parent.join("debug").join(name));
            }
        }
    }
    candidates.push(PathBuf::from("target/release").join(name));
    candidates.push(PathBuf::from("target/debug").join(name));
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            candidates.push(dir.join(name));
        }
    }
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(format!(
        "cannot find `{name}`; build it with `cargo build -p sofuu-emu`, \
         put it next to `sofuu`, or set SOFUU_EMU_BIN"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_events_are_semantic() {
        let actions = normalise_actions(&json!([
            {"type":"tap","id":"submit"},
            {"type":"type","id":"note","text":"hello"},
            {"type":"toggle","id":"live","checked":false}
        ]))
        .unwrap();
        assert_eq!(actions.len(), 3);
        let event = |ty: &str, id: &str| json!({"type":ty,"id":id});
        assert_eq!(event("click", "submit")["type"], "click");
    }

    #[test]
    fn json_args_accept_files_and_inline() {
        let inline = parse_json_arg("{\"a\":1}").unwrap();
        assert_eq!(inline["a"], 1);
        let dir = std::env::temp_dir().join(format!("sofuu-emu-args-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.json");
        std::fs::write(&path, "[1,2]").unwrap();
        let from_file = parse_json_arg(&format!("@{}", path.display())).unwrap();
        assert_eq!(from_file[1], 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn device_metrics_export_app_viewport_and_safe_insets() {
        let manifest = json!({
            "profiles": [{
                "id": "android",
                "devices": [{
                    "id": "sofuu-phone-411",
                    "name": "Sofuu Phone 411",
                    "width": 411.43,
                    "height": 914.29,
                    "scale": 2.625,
                    "safe_top": 24.0,
                    "safe_bottom": 24.0,
                    "safe_left": 0.0,
                    "safe_right": 0.0,
                    "status_bar_height": 24.0,
                    "titlebar_height": 0.0,
                    "home_bar_height": 24.0,
                    "keyboard_portrait": 300.0,
                    "keyboard_landscape": 200.0
                }]
            }]
        });
        let portrait = resolve_device_metrics(
            &manifest,
            "android",
            Some("sofuu-phone-411"),
            "portrait",
            false,
        )
        .unwrap();
        assert_eq!(portrait["screenWidth"], 411.43);
        assert_eq!(portrait["screenHeight"], 914.29);
        assert_eq!(portrait["width"], 411.43);
        assert_eq!(portrait["height"], 866.29);
        assert_eq!(portrait["scale"], 2.625);
        assert_eq!(portrait["safe"]["top"], 24.0);
        assert_eq!(portrait["safe"]["bottom"], 24.0);
        assert_eq!(portrait["keyboardHeight"], 0.0);

        let open = resolve_device_metrics(
            &manifest,
            "android",
            Some("sofuu-phone-411"),
            "portrait",
            true,
        )
        .unwrap();
        assert_eq!(open["keyboardHeight"], 300.0);

        let landscape = resolve_device_metrics(
            &manifest,
            "android",
            Some("sofuu-phone-411"),
            "landscape",
            false,
        )
        .unwrap();
        assert_eq!(landscape["screenWidth"], 914.29);
        assert_eq!(landscape["screenHeight"], 411.43);
        assert_eq!(landscape["safe"]["top"], 0.0);
        assert_eq!(landscape["safe"]["bottom"], 24.0);
        assert_eq!(landscape["safe"]["left"], 24.0);
        assert_eq!(landscape["safe"]["right"], 24.0);
        assert_eq!(landscape["width"], 866.29);
        assert_eq!(landscape["height"], 363.43);
    }

    #[test]
    fn live_window_is_screen_exact_and_frameless() {
        let manifest = json!({
            "profiles": [{
                "id": "android",
                "name": "Android",
                "devices": [{
                    "id": "sofuu-phone-411",
                    "name": "Sofuu Phone 411",
                    "width": 411.43,
                    "height": 914.29,
                    "scale": 2.625,
                    "corner_radius": 0.0,
                    "safe_top": 24.0,
                    "safe_bottom": 24.0,
                    "safe_left": 0.0,
                    "safe_right": 0.0
                }]
            }]
        });
        let resolved = resolve_live_profile(&manifest, "android", None, "portrait").unwrap();
        let spec = live_window_spec(&resolved, "android", Some(&manifest));
        // Client area equals the device screen in points, not the app viewport.
        assert_eq!(spec.width, 411.43);
        assert_eq!(spec.height, 914.29);
        assert_eq!(spec.scale, Some(2.625));
        assert!(!spec.decorations, "emulator screen must be frameless");
        assert!(!spec.resizable, "device presets are fixed");
        assert!(spec.strip, "controls live in the attached strip");
        assert_eq!(spec.device, "sofuu-phone-411");
        assert!(!spec.profiles_json.is_empty());

        let rotated = resolve_live_profile(&manifest, "android", None, "landscape").unwrap();
        let spec = live_window_spec(&rotated, "android", Some(&manifest));
        assert_eq!(spec.width, 914.29);
        assert_eq!(spec.height, 411.43);
    }

    #[test]
    fn capture_env_overrides_startup_flags() {
        let mut args: Vec<String> = [
            "ui",
            "snapshot",
            "app.js",
            "--platform",
            "ios",
            "--device",
            "iphone-16-pro",
            "--theme",
            "light",
            "--orientation",
            "portrait",
            "--keyboard-open",
            "--mode",
            "annotated",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        apply_capture_env(
            &mut args,
            &json!({
                "platform": "android",
                "device": null,
                "theme": "dark",
                "orientation": "landscape",
                "keyboard": false
            }),
        );
        let joined = args.join(" ");
        assert!(joined.contains("--platform android"));
        assert!(!joined.contains("iphone-16-pro"));
        assert!(!joined.contains("--keyboard-open"));
        assert!(joined.contains("--theme dark"));
        assert!(joined.contains("--orientation landscape"));
        assert!(joined.contains("--mode annotated"));
    }
}
