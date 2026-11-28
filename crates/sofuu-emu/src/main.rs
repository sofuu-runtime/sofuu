//! `sofuu-emu` companion binary.
//!
//! `sofuu-core` (and tests) drive it over JSON so the shipped CLI never links
//! the emulator. Commands:
//!
//! ```text
//! sofuu-emu profiles [--json]
//! sofuu-emu render   --job job.json [--mode clean|annotated|filmstrip]
//!                    [--out shot.png] [--audit-json audit.json] [--fail-on-error]
//! sofuu-emu render   --job -            # read the job from stdin
//! sofuu-emu serve                       # JSON-lines render server
//! ```

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use sofuu_emu::screenshot::{run_job, CaptureMode, RenderJob};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("sofuu-emu: {err}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(ExitCode::SUCCESS);
    };
    match command {
        "profiles" | "list" => {
            if args.iter().any(|a| a == "--json") {
                println!("{}", sofuu_emu::profiles_json());
            } else {
                for profile in sofuu_emu::profiles() {
                    println!("{} ({})", profile.name, profile.id);
                    for device in &profile.devices {
                        println!(
                            "  {:<22} {:>4}x{:<4} @{:.1}x{}",
                            device.id,
                            device.width as i32,
                            device.height as i32,
                            device.scale,
                            if device.is_touch(&profile) { "  touch" } else { "" }
                        );
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        "render" | "snapshot" | "audit" => render(args),
        "serve" => serve(),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(ExitCode::SUCCESS)
        }
        "--version" | "-V" | "version" => {
            println!("sofuu-emu {}", sofuu_emu::VERSION);
            Ok(ExitCode::SUCCESS)
        }
        other => Err(format!(
            "unknown command `{other}` (try `profiles`, `render`, `serve`)"
        )),
    }
}

fn render(args: &[String]) -> Result<ExitCode, String> {
    let mut job_path: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut audit_json: Option<PathBuf> = None;
    let mut mode: Option<CaptureMode> = None;
    let mut fail_on_error = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--job" | "--input" | "-j" => {
                i += 1;
                job_path = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "--job needs a path (or `-`)".to_string())?,
                );
            }
            "--out" | "-o" => {
                i += 1;
                out = Some(PathBuf::from(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "--out needs a path".to_string())?,
                ));
            }
            "--audit-json" => {
                i += 1;
                audit_json = Some(PathBuf::from(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "--audit-json needs a path".to_string())?,
                ));
            }
            "--mode" | "-m" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--mode needs clean|annotated|filmstrip".to_string())?;
                mode = Some(CaptureMode::parse(value)?);
            }
            "--fail-on-error" => fail_on_error = true,
            "--help" | "-h" => {
                print_help();
                return Ok(ExitCode::SUCCESS);
            }
            other => return Err(format!("unknown render flag `{other}`")),
        }
        i += 1;
    }
    let Some(job_path) = job_path else {
        return Err("render needs --job <path|->".to_string());
    };
    let text = if job_path == "-" {
        std::io::read_to_string(std::io::stdin())
            .map_err(|e| format!("read stdin: {e}"))?
    } else {
        std::fs::read_to_string(&job_path).map_err(|e| format!("{job_path}: {e}"))?
    };
    let mut job = RenderJob::from_json(&text)?;
    if let Some(mode) = mode {
        job.mode = mode;
    }
    if out.is_some() {
        job.out = out.map(|p| p.display().to_string());
    }
    if audit_json.is_some() {
        job.audit_json = audit_json.map(|p| p.display().to_string());
    }
    job.fail_on_error |= fail_on_error;

    let errors = job.fail_on_error;
    let captured = run_job(job)?;
    println!("{}", captured.to_json());
    if errors && captured.audit.has_errors() {
        Ok(ExitCode::from(2))
    } else {
        Ok(ExitCode::SUCCESS)
    }
}

fn serve() -> Result<ExitCode, String> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(err) => return Err(format!("read stdin: {err}")),
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "ping" {
            let _ = writeln!(stdout, "{{\"ok\":true,\"version\":\"{}\"}}", sofuu_emu::VERSION);
            let _ = stdout.flush();
            continue;
        }
        if line == "profiles" {
            let _ = writeln!(stdout, "{}", sofuu_emu::profiles_json());
            let _ = stdout.flush();
            continue;
        }
        let response = match RenderJob::from_json(line).and_then(run_job) {
            Ok(captured) => serde_json::json!({ "ok": true, "capture": captured }),
            Err(err) => serde_json::json!({ "ok": false, "error": err }),
        };
        let _ = writeln!(stdout, "{response}");
        let _ = stdout.flush();
    }
    Ok(ExitCode::SUCCESS)
}

fn print_help() {
    println!(
        "sofuu-emu {version}\n\
         \n\
         USAGE:\n\
         \x20 sofuu-emu profiles [--json]\n\
         \x20 sofuu-emu render --job <job.json|-> [--mode clean|annotated|filmstrip]\n\
         \x20                    [--out shot.png] [--audit-json audit.json] [--fail-on-error]\n\
         \x20 sofuu-emu serve\n\
         \n\
         A render job is JSON:\n\
         \x20 {{ \"config\": {{ \"platform\": \"ios\", \"device\": \"iphone-16-pro\" }},\n\
         \x20   \"app\": <Scene>, \"mode\": \"annotated\", \"out\": \"/tmp/shot.png\" }}\n\
         \n\
         Exit codes: 0 success, 1 bad job/IO, 2 audit errors with --fail-on-error.",
        version = sofuu_emu::VERSION
    );
}
