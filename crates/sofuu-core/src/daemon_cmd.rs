// daemon_cmd.rs — `sofuu daemon`: 24/7 autonomous background agent supervisor.
//
// Manages continuous background agent daemons via native OS service managers:
// - macOS: launchd plists in ~/Library/LaunchAgents/com.sofuu.<name>.plist
// - Linux: systemd user units in ~/.config/systemd/user/sofuu-<name>.service
// - Foreground: `sofuu daemon run <target>` runs the supervisor loop directly.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

extern "C" fn sig_handler(_: libc::c_int) {
    RUNNING.store(false, Ordering::SeqCst);
}

pub fn print_help() {
    println!(
        "Usage: sofuu daemon <command> [options]\n\n\
         Autonomous 24/7 background agent service supervisor.\n\
         Deploys, manages, and monitors long-running agent daemons via native OS service\n\
         managers (launchd on macOS, systemd on Linux).\n\n\
         Commands:\n\
         \x20 install <target> [options]  Install and start agent as persistent OS background service\n\
         \x20 status [name]              Show status of running/registered agent daemons\n\
         \x20 logs <name> [options]      View or stream agent daemon execution logs\n\
         \x20 stop <name>                Stop running agent daemon service\n\
         \x20 start <name>               Start installed agent daemon service\n\
         \x20 restart <name>             Restart agent daemon service\n\
         \x20 uninstall <name>           Stop and completely remove agent daemon service\n\
         \x20 run <target> [options]     Run foreground supervisor loop without OS service registration\n\n\
         Install options:\n\
         \x20 --name <name>              Service name (default: stem of target binary/script)\n\
         \x20 --every <interval>         Execution cycle interval (e.g. 500ms, 2s, 10s, 1m, 1h; default: 5s)\n\
         \x20 --interval <val>           Alias for --every\n\
         \x20 --entry <fn>               Routine to execute in agent\n\
         \x20 --arg <k=v>                Pass key-value argument (repeatable)\n\
         \x20 --cap <name>               Grant capability backend (repeatable)\n\
         \x20 --fuel <n>                 Instruction fuel budget (default: 200000)\n\
         \x20 --keep-alive               Auto-restart daemon process if it exits\n\
         \x20 --no-start                 Install service configuration without immediate start\n\n\
         Log options:\n\
         \x20 -f, --follow               Follow log output continuously\n\
         \x20 -n <lines>                 Number of lines to show (default: 50)\n\
         \x20 --err                      Show stderr log instead of stdout\n"
    );
}

pub fn run(argv: &[String]) -> i32 {
    let Some(subcmd) = argv.first().map(String::as_str) else {
        print_help();
        return 1;
    };

    match subcmd {
        "-h" | "--help" | "help" => {
            print_help();
            0
        }
        "install" => cmd_install(&argv[1..]),
        "status" => cmd_status(&argv[1..]),
        "logs" | "log" => cmd_logs(&argv[1..]),
        "stop" => cmd_stop(&argv[1..]),
        "start" => cmd_start(&argv[1..]),
        "restart" => cmd_restart(&argv[1..]),
        "uninstall" | "remove" => cmd_uninstall(&argv[1..]),
        "run" => cmd_run(&argv[1..]),
        unknown => {
            eprintln!("\x1b[31mError:\x1b[0m unknown daemon command '{unknown}'\n");
            print_help();
            1
        }
    }
}

// ── Paths and environment helpers ───────────────────────────────────

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn daemon_log_dir() -> PathBuf {
    home_dir().join(".sofuu").join("logs").join("daemon")
}

#[cfg(target_os = "macos")]
fn launch_agents_dir() -> PathBuf {
    home_dir().join("Library").join("LaunchAgents")
}

#[cfg(target_os = "macos")]
fn plist_path_for(name: &str) -> PathBuf {
    launch_agents_dir().join(format!("com.sofuu.{name}.plist"))
}

#[cfg(target_os = "linux")]
fn systemd_user_dir() -> PathBuf {
    home_dir().join(".config").join("systemd").join("user")
}

#[cfg(target_os = "linux")]
fn service_path_for(name: &str) -> PathBuf {
    systemd_user_dir().join(format!("sofuu-{name}.service"))
}

fn clean_service_name(s: &str) -> String {
    let clean: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    if clean.is_empty() { "agent".to_string() } else { clean }
}

fn parse_interval_ms(s: &str) -> u64 {
    let s = s.trim().to_lowercase();
    if s.ends_with("ms") {
        s[..s.len() - 2].parse::<u64>().unwrap_or(5000).max(10)
    } else if s.ends_with('s') {
        let secs = s[..s.len() - 1].parse::<f64>().unwrap_or(5.0).max(0.1);
        (secs * 1000.0) as u64
    } else if s.ends_with('m') {
        let mins = s[..s.len() - 1].parse::<f64>().unwrap_or(1.0).max(0.01);
        (mins * 60.0 * 1000.0) as u64
    } else if s.ends_with('h') {
        let hrs = s[..s.len() - 1].parse::<f64>().unwrap_or(1.0).max(0.001);
        (hrs * 3600.0 * 1000.0) as u64
    } else if s.ends_with('d') {
        let days = s[..s.len() - 1].parse::<f64>().unwrap_or(1.0).max(0.0001);
        (days * 86400.0 * 1000.0) as u64
    } else if let Ok(val) = s.parse::<u64>() {
        if val >= 100 { val } else { val * 1000 }
    } else {
        5000
    }
}

fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    if days > 0 {
        format!("{days}d {hours}h {mins}m")
    } else if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else if mins > 0 {
        format!("{mins}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

// ── Command: install ───────────────────────────────────────────────

fn cmd_install(args: &[String]) -> i32 {
    let mut target_arg: Option<String> = None;
    let mut custom_name: Option<String> = None;
    let mut interval = "5s".to_string();
    let mut entry: Option<String> = None;
    let mut arg_pairs: Vec<String> = Vec::new();
    let mut caps: Vec<String> = Vec::new();
    let mut fuel = 200000u64;
    let mut keep_alive = true;
    let mut no_start = false;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--name" && i + 1 < args.len() {
            custom_name = Some(args[i + 1].clone());
            i += 2;
        } else if (a == "--every" || a == "--interval") && i + 1 < args.len() {
            interval = args[i + 1].clone();
            i += 2;
        } else if let Some(val) = a.strip_prefix("--every=") {
            interval = val.to_string();
            i += 1;
        } else if let Some(val) = a.strip_prefix("--interval=") {
            interval = val.to_string();
            i += 1;
        } else if a == "--entry" && i + 1 < args.len() {
            entry = Some(args[i + 1].clone());
            i += 2;
        } else if a == "--arg" && i + 1 < args.len() {
            arg_pairs.push(args[i + 1].clone());
            i += 2;
        } else if let Some(val) = a.strip_prefix("--arg=") {
            arg_pairs.push(val.to_string());
            i += 1;
        } else if a == "--cap" && i + 1 < args.len() {
            caps.push(args[i + 1].clone());
            i += 2;
        } else if let Some(val) = a.strip_prefix("--cap=") {
            caps.push(val.to_string());
            i += 1;
        } else if a == "--fuel" && i + 1 < args.len() {
            fuel = args[i + 1].parse().unwrap_or(200000);
            i += 2;
        } else if a == "--no-keep-alive" {
            keep_alive = false;
            i += 1;
        } else if a == "--keep-alive" {
            keep_alive = true;
            i += 1;
        } else if a == "--no-start" {
            no_start = true;
            i += 1;
        } else if !a.starts_with('-') && target_arg.is_none() {
            target_arg = Some(a.clone());
            i += 1;
        } else {
            i += 1;
        }
    }

    let Some(target) = target_arg else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon install <target>' requires a target binary or script");
        eprintln!("Example: sofuu daemon install ./example-agents/paper_trader/paper-trader --every 2s");
        return 1;
    };

    let target_path = Path::new(&target);
    if !target_path.exists() {
        eprintln!("\x1b[31mError:\x1b[0m target '{}' not found on disk", target_path.display());
        return 1;
    }

    let canonical_target = match fs::canonicalize(target_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m failed to resolve path '{}': {e}", target_path.display());
            return 1;
        }
    };

    let name = match custom_name {
        Some(n) => clean_service_name(&n),
        None => clean_service_name(
            canonical_target
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("agent"),
        ),
    };

    let work_dir = canonical_target
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_string_lossy()
        .to_string();

    let is_kaze = canonical_target
        .extension()
        .is_some_and(|ext| ext == "kaze" || ext == "kazebc");

    let sofuu_exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "sofuu".to_string());

    let mut exec_args: Vec<String> = Vec::new();
    let exec_cmd: String;

    if is_kaze {
        exec_cmd = sofuu_exe;
        exec_args.push("daemon".to_string());
        exec_args.push("run".to_string());
        exec_args.push(canonical_target.to_string_lossy().to_string());
        exec_args.push("--every".to_string());
        exec_args.push(interval.clone());
        if let Some(ref e) = entry {
            exec_args.push("--entry".to_string());
            exec_args.push(e.clone());
        }
        for ap in &arg_pairs {
            exec_args.push("--arg".to_string());
            exec_args.push(ap.clone());
        }
        for c in &caps {
            exec_args.push("--cap".to_string());
            exec_args.push(c.clone());
        }
        if fuel != 200000 {
            exec_args.push("--fuel".to_string());
            exec_args.push(fuel.to_string());
        }
    } else {
        // Standalone compiled binary
        exec_cmd = canonical_target.to_string_lossy().to_string();
        exec_args.push("--daemon".to_string());
        exec_args.push("--every".to_string());
        exec_args.push(interval.clone());
        if let Some(ref e) = entry {
            exec_args.push("--entry".to_string());
            exec_args.push(e.clone());
        }
        for ap in &arg_pairs {
            exec_args.push("--arg".to_string());
            exec_args.push(ap.clone());
        }
        for c in &caps {
            exec_args.push("--cap".to_string());
            exec_args.push(c.clone());
        }
        if fuel != 200000 {
            exec_args.push("--fuel".to_string());
            exec_args.push(fuel.to_string());
        }
    }

    let log_dir = daemon_log_dir();
    if let Err(e) = fs::create_dir_all(&log_dir) {
        eprintln!("\x1b[31mError:\x1b[0m failed to create log directory '{}': {e}", log_dir.display());
        return 1;
    }

    let stdout_log = log_dir.join(format!("{name}.stdout.log"));
    let stderr_log = log_dir.join(format!("{name}.stderr.log"));

    #[cfg(target_os = "macos")]
    {
        let agents_dir = launch_agents_dir();
        if let Err(e) = fs::create_dir_all(&agents_dir) {
            eprintln!("\x1b[31mError:\x1b[0m failed to create LaunchAgents directory: {e}");
            return 1;
        }

        let plist_path = plist_path_for(&name);
        let mut prog_args_xml = format!("        <string>{exec_cmd}</string>\n");
        for arg in &exec_args {
            let escaped = arg.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
            prog_args_xml.push_str(&format!("        <string>{escaped}</string>\n"));
        }

        let plist_content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sofuu.{name}</string>
    <key>ProgramArguments</key>
    <array>
{prog_args_xml}    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <{keep_alive}/>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
</dict>
</plist>
"#,
            stdout_log.to_string_lossy(),
            stderr_log.to_string_lossy()
        );

        if let Err(e) = fs::write(&plist_path, plist_content) {
            eprintln!("\x1b[31mError:\x1b[0m failed to write plist '{}': {e}", plist_path.display());
            return 1;
        }

        let mut pid_str = "-".to_string();
        let mut is_active = false;

        if !no_start {
            // Unload if previously running
            let _ = Command::new("launchctl")
                .arg("unload")
                .arg(&plist_path)
                .output();

            // Load and start
            let load_res = Command::new("launchctl")
                .arg("load")
                .arg("-w")
                .arg(&plist_path)
                .output();

            match load_res {
                Ok(out) if out.status.success() => {
                    is_active = true;
                    std::thread::sleep(Duration::from_millis(150));
                    if let Ok(list_out) = Command::new("launchctl").arg("list").output() {
                        let text = String::from_utf8_lossy(&list_out.stdout);
                        let label = format!("com.sofuu.{name}");
                        for line in text.lines() {
                            if line.contains(&label) {
                                let parts: Vec<&str> = line.split_whitespace().collect();
                                if let Some(p) = parts.first() {
                                    if *p != "-" {
                                        pid_str = p.to_string();
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    eprintln!("\x1b[33mWarning:\x1b[0m launchctl load exited with: {err}");
                }
                Err(e) => {
                    eprintln!("\x1b[33mWarning:\x1b[0m failed to invoke launchctl: {e}");
                }
            }
        }

        println!("\x1b[32m[DAEMON]\x1b[0m Installed 24/7 background service for '\x1b[1m{name}\x1b[0m'");
        println!("  Service ID:  com.sofuu.{name}");
        println!("  Target:      {}", canonical_target.display());
        println!("  Interval:    {interval}");
        if is_active {
            if pid_str != "-" {
                println!("  Status:      \x1b[32mRUNNING\x1b[0m (PID: {pid_str})");
            } else {
                println!("  Status:      \x1b[32mLOADED\x1b[0m (managed by launchd)");
            }
        } else {
            println!("  Status:      \x1b[33mINSTALLED\x1b[0m (stopped, use 'sofuu daemon start {name}')");
        }
        println!("  Config:      {}", plist_path.display());
        println!("  Stdout:      {}", stdout_log.display());
        println!("  Stderr:      {}", stderr_log.display());
        println!("\nQuick commands:");
        println!("  sofuu daemon status {name}");
        println!("  sofuu daemon logs {name} -f");
        println!("  sofuu daemon stop {name}");
        println!("  sofuu daemon uninstall {name}\n");
        0
    }

    #[cfg(target_os = "linux")]
    {
        let systemd_dir = systemd_user_dir();
        if let Err(e) = fs::create_dir_all(&systemd_dir) {
            eprintln!("\x1b[31mError:\x1b[0m failed to create systemd user directory: {e}");
            return 1;
        }

        let svc_path = service_path_for(&name);
        let mut full_cmd = exec_cmd.clone();
        for a in &exec_args {
            full_cmd.push(' ');
            full_cmd.push_str(a);
        }

        let restart_pol = if keep_alive { "always" } else { "on-failure" };
        let unit_content = format!(
            r#"[Unit]
Description=Sofuu Autonomous Agent - {name}
After=network.target

[Service]
Type=simple
ExecStart={full_cmd}
WorkingDirectory={work_dir}
StandardOutput=append:{}
StandardError=append:{}
Restart={restart_pol}
RestartSec=3

[Install]
WantedBy=default.target
"#,
            stdout_log.to_string_lossy(),
            stderr_log.to_string_lossy()
        );

        if let Err(e) = fs::write(&svc_path, unit_content) {
            eprintln!("\x1b[31mError:\x1b[0m failed to write systemd service unit '{}': {e}", svc_path.display());
            return 1;
        }

        let _ = Command::new("systemctl").arg("--user").arg("daemon-reload").output();

        let mut is_active = false;
        if !no_start {
            let start_res = Command::new("systemctl")
                .arg("--user")
                .arg("enable")
                .arg("--now")
                .arg(format!("sofuu-{name}.service"))
                .output();

            if let Ok(out) = start_res {
                if out.status.success() {
                    is_active = true;
                }
            }
        }

        println!("\x1b[32m[DAEMON]\x1b[0m Installed 24/7 background service for '\x1b[1m{name}\x1b[0m'");
        println!("  Service ID:  sofuu-{name}.service");
        println!("  Target:      {}", canonical_target.display());
        println!("  Interval:    {interval}");
        println!("  Status:      {}", if is_active { "\x1b[32mACTIVE\x1b[0m" } else { "\x1b[33mINSTALLED\x1b[0m" });
        println!("  Unit:        {}", svc_path.display());
        println!("  Stdout:      {}", stdout_log.display());
        println!("  Stderr:      {}", stderr_log.display());
        println!("\nQuick commands:");
        println!("  sofuu daemon status {name}");
        println!("  sofuu daemon logs {name} -f");
        println!("  sofuu daemon stop {name}");
        println!("  sofuu daemon uninstall {name}\n");
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("\x1b[33m[Notice]\x1b[0m OS background service auto-installation is supported on macOS (launchd) and Linux (systemd).");
        println!("On this platform, use foreground supervisor mode:");
        println!("  sofuu daemon run {} --every {interval}", canonical_target.display());
        0
    }
}

// ── Command: status ────────────────────────────────────────────────

fn cmd_status(args: &[String]) -> i32 {
    let name_query = args.first().map(|s| clean_service_name(s));

    #[cfg(target_os = "macos")]
    {
        let agents_dir = launch_agents_dir();
        let list_out = Command::new("launchctl").arg("list").output();
        let launchctl_text = list_out
            .as_ref()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        if let Some(ref name) = name_query {
            let plist = plist_path_for(name);
            let label = format!("com.sofuu.{name}");
            let is_installed = plist.exists();
            let mut pid = "-".to_string();
            let mut is_running = false;

            for line in launchctl_text.lines() {
                if line.contains(&label) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(p) = parts.first() {
                        if *p != "-" {
                            pid = p.to_string();
                            is_running = true;
                        } else {
                            is_running = true; // loaded in launchd
                        }
                    }
                }
            }

            println!("\x1b[1m=== Sofuu Daemon Status: {name} ===\x1b[0m");
            println!("  Service ID:  {label}");
            println!("  Config:      {}", plist.display());
            println!("  Installed:   {}", if is_installed { "YES" } else { "NO" });
            if is_running {
                if pid != "-" {
                    println!("  State:       \x1b[32mRUNNING\x1b[0m (PID: {pid})");
                } else {
                    println!("  State:       \x1b[32mLOADED\x1b[0m (waiting / scheduled by launchd)");
                }
            } else {
                println!("  State:       \x1b[31mSTOPPED\x1b[0m");
            }

            let log_dir = daemon_log_dir();
            let stdout_file = log_dir.join(format!("{name}.stdout.log"));
            let stderr_file = log_dir.join(format!("{name}.stderr.log"));
            if stdout_file.exists() {
                let sz = fs::metadata(&stdout_file).map(|m| m.len()).unwrap_or(0);
                println!("  Stdout:      {} ({:.1} KB)", stdout_file.display(), sz as f64 / 1024.0);
            }
            if stderr_file.exists() {
                let sz = fs::metadata(&stderr_file).map(|m| m.len()).unwrap_or(0);
                if sz > 0 {
                    println!("  Stderr:      {} ({:.1} KB)", stderr_file.display(), sz as f64 / 1024.0);
                }
            }

            // Print last 5 log lines if stdout exists
            if stdout_file.exists() {
                if let Ok(content) = fs::read_to_string(&stdout_file) {
                    let lines: Vec<&str> = content.lines().collect();
                    if !lines.is_empty() {
                        println!("\n  \x1b[1mRecent Activity (last {} lines):\x1b[0m", lines.len().min(5));
                        let start = if lines.len() > 5 { lines.len() - 5 } else { 0 };
                        for l in &lines[start..] {
                            println!("    {l}");
                        }
                    }
                }
            }
            println!();
            return 0;
        }

        // List all daemons
        println!("\x1b[1m{:<20} {:<12} {:<10} {:<12} {:<28}\x1b[0m", "NAME", "STATUS", "PID", "LOG SIZE", "SERVICE ID");
        println!("{}", "-".repeat(84));

        let mut found = 0;
        if let Ok(entries) = fs::read_dir(&agents_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().to_string();
                if fname.starts_with("com.sofuu.") && fname.ends_with(".plist") {
                    found += 1;
                    let name = fname["com.sofuu.".len()..fname.len() - ".plist".len()].to_string();
                    let label = format!("com.sofuu.{name}");
                    let mut pid = "-".to_string();
                    let mut state = "\x1b[31mSTOPPED\x1b[0m".to_string();

                    for line in launchctl_text.lines() {
                        if line.contains(&label) {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if let Some(p) = parts.first() {
                                if *p != "-" {
                                    pid = p.to_string();
                                    state = "\x1b[32mRUNNING\x1b[0m".to_string();
                                } else {
                                    state = "\x1b[32mLOADED\x1b[0m".to_string();
                                }
                            }
                        }
                    }

                    let stdout_file = daemon_log_dir().join(format!("{name}.stdout.log"));
                    let log_size = if stdout_file.exists() {
                        let sz = fs::metadata(&stdout_file).map(|m| m.len()).unwrap_or(0);
                        format!("{:.1} KB", sz as f64 / 1024.0)
                    } else {
                        "-".to_string()
                    };

                    println!("{:<20} {:<21} {:<10} {:<12} {:<28}", name, state, pid, log_size, label);
                }
            }
        }

        if found == 0 {
            println!("  (No active Sofuu daemons installed in ~/Library/LaunchAgents)");
            println!("  Use 'sofuu daemon install <target>' to register an autonomous agent.");
        }
        println!();
        0
    }

    #[cfg(target_os = "linux")]
    {
        let systemd_dir = systemd_user_dir();
        if let Some(ref name) = name_query {
            let unit = service_path_for(name);
            println!("\x1b[1m=== Sofuu Daemon Status: {name} ===\x1b[0m");
            println!("  Service Unit: {}", unit.display());
            println!("  Installed:    {}", if unit.exists() { "YES" } else { "NO" });

            let status_cmd = Command::new("systemctl")
                .arg("--user")
                .arg("status")
                .arg(format!("sofuu-{name}.service"))
                .output();

            if let Ok(out) = status_cmd {
                let text = String::from_utf8_lossy(&out.stdout);
                println!("\n{text}");
            }
            return 0;
        }

        println!("\x1b[1m{:<20} {:<12} {:<28}\x1b[0m", "NAME", "STATUS", "UNIT");
        println!("{}", "-".repeat(64));
        let mut found = 0;
        if let Ok(entries) = fs::read_dir(&systemd_dir) {
            for entry in entries.flatten() {
                let fname = entry.file_name().to_string_lossy().to_string();
                if fname.starts_with("sofuu-") && fname.ends_with(".service") {
                    found += 1;
                    let name = fname["sofuu-".len()..fname.len() - ".service".len()].to_string();
                    let is_active = Command::new("systemctl")
                        .arg("--user")
                        .arg("is-active")
                        .arg(&fname)
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false);

                    let state = if is_active { "\x1b[32mACTIVE\x1b[0m" } else { "\x1b[31mINACTIVE\x1b[0m" };
                    println!("{:<20} {:<21} {:<28}", name, state, fname);
                }
            }
        }
        if found == 0 {
            println!("  (No active Sofuu daemons found in ~/.config/systemd/user)");
        }
        println!();
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Status inquiry for background services is available on macOS and Linux.");
        0
    }
}

// ── Command: logs ──────────────────────────────────────────────────

fn cmd_logs(args: &[String]) -> i32 {
    let mut name_opt: Option<String> = None;
    let mut follow = false;
    let mut lines = 50usize;
    let mut show_err = false;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "-f" || a == "--follow" {
            follow = true;
            i += 1;
        } else if a == "--err" || a == "--stderr" {
            show_err = true;
            i += 1;
        } else if (a == "-n" || a == "--lines") && i + 1 < args.len() {
            lines = args[i + 1].parse().unwrap_or(50);
            i += 2;
        } else if !a.starts_with('-') && name_opt.is_none() {
            name_opt = Some(clean_service_name(a));
            i += 1;
        } else {
            i += 1;
        }
    }

    let Some(name) = name_opt else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon logs <name>' requires a service name");
        return 1;
    };

    let log_file = if show_err {
        daemon_log_dir().join(format!("{name}.stderr.log"))
    } else {
        daemon_log_dir().join(format!("{name}.stdout.log"))
    };

    if !log_file.exists() {
        eprintln!("\x1b[31mError:\x1b[0m log file '{}' does not exist", log_file.display());
        eprintln!("Has daemon '{name}' run yet? Check 'sofuu daemon status'.");
        return 1;
    }

    if follow {
        let mut child = match Command::new("tail")
            .arg("-n")
            .arg(lines.to_string())
            .arg("-f")
            .arg(&log_file)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m failed to spawn tail: {e}");
                return 1;
            }
        };
        let _ = child.wait();
        0
    } else {
        match fs::read_to_string(&log_file) {
            Ok(content) => {
                let all_lines: Vec<&str> = content.lines().collect();
                let start = if all_lines.len() > lines { all_lines.len() - lines } else { 0 };
                for l in &all_lines[start..] {
                    println!("{l}");
                }
                0
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m failed to read '{}': {e}", log_file.display());
                1
            }
        }
    }
}

// ── Command: stop ──────────────────────────────────────────────────

fn cmd_stop(args: &[String]) -> i32 {
    let Some(name) = args.first().map(|s| clean_service_name(s)) else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon stop <name>' requires a service name");
        return 1;
    };

    #[cfg(target_os = "macos")]
    {
        let plist = plist_path_for(&name);
        if !plist.exists() {
            eprintln!("\x1b[31mError:\x1b[0m service '{name}' configuration not found at '{}'", plist.display());
            return 1;
        }

        let res = Command::new("launchctl").arg("unload").arg(&plist).output();
        match res {
            Ok(o) if o.status.success() => {
                println!("\x1b[32m[DAEMON]\x1b[0m Stopped service '{name}' (unloaded from launchd)");
                0
            }
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr);
                println!("\x1b[33m[DAEMON]\x1b[0m Service '{name}' stopped: {err}");
                0
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m failed to invoke launchctl: {e}");
                1
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let res = Command::new("systemctl")
            .arg("--user")
            .arg("stop")
            .arg(format!("sofuu-{name}.service"))
            .output();

        match res {
            Ok(o) if o.status.success() => {
                println!("\x1b[32m[DAEMON]\x1b[0m Stopped service '{name}'");
                0
            }
            Ok(o) => {
                eprintln!("\x1b[31mError:\x1b[0m {}", String::from_utf8_lossy(&o.stderr));
                1
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
                1
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Service stop is supported on macOS and Linux.");
        0
    }
}

// ── Command: start ─────────────────────────────────────────────────

fn cmd_start(args: &[String]) -> i32 {
    let Some(name) = args.first().map(|s| clean_service_name(s)) else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon start <name>' requires a service name");
        return 1;
    };

    #[cfg(target_os = "macos")]
    {
        let plist = plist_path_for(&name);
        if !plist.exists() {
            eprintln!("\x1b[31mError:\x1b[0m service '{name}' configuration not found at '{}'", plist.display());
            return 1;
        }

        // Unload first to ensure clean state
        let _ = Command::new("launchctl").arg("unload").arg(&plist).output();
        let res = Command::new("launchctl").arg("load").arg("-w").arg(&plist).output();

        match res {
            Ok(o) if o.status.success() => {
                println!("\x1b[32m[DAEMON]\x1b[0m Started service '{name}' (loaded into launchd)");
                0
            }
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr);
                eprintln!("\x1b[31mError:\x1b[0m launchctl load failed: {err}");
                1
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m failed to invoke launchctl: {e}");
                1
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let res = Command::new("systemctl")
            .arg("--user")
            .arg("start")
            .arg(format!("sofuu-{name}.service"))
            .output();

        match res {
            Ok(o) if o.status.success() => {
                println!("\x1b[32m[DAEMON]\x1b[0m Started service '{name}'");
                0
            }
            Ok(o) => {
                eprintln!("\x1b[31mError:\x1b[0m {}", String::from_utf8_lossy(&o.stderr));
                1
            }
            Err(e) => {
                eprintln!("\x1b[31mError:\x1b[0m {e}");
                1
            }
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Service start is supported on macOS and Linux.");
        0
    }
}

// ── Command: restart ───────────────────────────────────────────────

fn cmd_restart(args: &[String]) -> i32 {
    let rc = cmd_stop(args);
    if rc != 0 {
        // Continue anyway to attempt restart
    }
    std::thread::sleep(Duration::from_millis(200));
    cmd_start(args)
}

// ── Command: uninstall ─────────────────────────────────────────────

fn cmd_uninstall(args: &[String]) -> i32 {
    let Some(name) = args.first().map(|s| clean_service_name(s)) else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon uninstall <name>' requires a service name");
        return 1;
    };

    #[cfg(target_os = "macos")]
    {
        let plist = plist_path_for(&name);
        if plist.exists() {
            let _ = Command::new("launchctl").arg("unload").arg(&plist).output();
            let _ = fs::remove_file(&plist);
            println!("\x1b[32m[DAEMON]\x1b[0m Uninstalled service '{name}' and removed '{}'", plist.display());
            println!("  (Log files preserved in ~/.sofuu/logs/daemon/{name}.*.log)");
            0
        } else {
            eprintln!("\x1b[33mWarning:\x1b[0m service '{name}' configuration not found at '{}'", plist.display());
            1
        }
    }

    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("systemctl")
            .arg("--user")
            .arg("stop")
            .arg(format!("sofuu-{name}.service"))
            .output();
        let _ = Command::new("systemctl")
            .arg("--user")
            .arg("disable")
            .arg(format!("sofuu-{name}.service"))
            .output();

        let svc = service_path_for(&name);
        if svc.exists() {
            let _ = fs::remove_file(&svc);
            let _ = Command::new("systemctl").arg("--user").arg("daemon-reload").output();
            println!("\x1b[32m[DAEMON]\x1b[0m Uninstalled service '{name}'");
            0
        } else {
            eprintln!("\x1b[33mWarning:\x1b[0m service '{name}' not found");
            1
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        println!("Service uninstall is supported on macOS and Linux.");
        0
    }
}

// ── Command: run (Foreground Supervisor Loop) ──────────────────────

fn cmd_run(args: &[String]) -> i32 {
    let mut target_arg: Option<String> = None;
    let mut interval_raw = "5s".to_string();
    let mut entry: Option<String> = None;
    let mut arg_pairs: Vec<String> = Vec::new();
    let mut caps: Vec<String> = Vec::new();
    let mut fuel = 200000u64;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if (a == "--every" || a == "--interval") && i + 1 < args.len() {
            interval_raw = args[i + 1].clone();
            i += 2;
        } else if let Some(val) = a.strip_prefix("--every=") {
            interval_raw = val.to_string();
            i += 1;
        } else if let Some(val) = a.strip_prefix("--interval=") {
            interval_raw = val.to_string();
            i += 1;
        } else if a == "--entry" && i + 1 < args.len() {
            entry = Some(args[i + 1].clone());
            i += 2;
        } else if a == "--arg" && i + 1 < args.len() {
            arg_pairs.push(args[i + 1].clone());
            i += 2;
        } else if let Some(val) = a.strip_prefix("--arg=") {
            arg_pairs.push(val.to_string());
            i += 1;
        } else if a == "--cap" && i + 1 < args.len() {
            caps.push(args[i + 1].clone());
            i += 2;
        } else if let Some(val) = a.strip_prefix("--cap=") {
            caps.push(val.to_string());
            i += 1;
        } else if a == "--fuel" && i + 1 < args.len() {
            fuel = args[i + 1].parse().unwrap_or(200000);
            i += 2;
        } else if !a.starts_with('-') && target_arg.is_none() {
            target_arg = Some(a.clone());
            i += 1;
        } else {
            i += 1;
        }
    }

    let Some(target) = target_arg else {
        eprintln!("\x1b[31mError:\x1b[0m 'sofuu daemon run <target>' requires a target file or binary");
        return 1;
    };

    let target_path = Path::new(&target);
    if !target_path.exists() {
        eprintln!("\x1b[31mError:\x1b[0m target '{}' not found on disk", target_path.display());
        return 1;
    }

    let interval_ms = parse_interval_ms(&interval_raw);
    let is_kaze = target_path
        .extension()
        .is_some_and(|ext| ext == "kaze" || ext == "kazebc");

    // If it's a standalone binary executable, delegate directly to its built-in daemon loop!
    if !is_kaze {
        let mut cmd = Command::new(target_path);
        cmd.arg("--daemon");
        cmd.arg("--every").arg(&interval_raw);
        if let Some(ref e) = entry {
            cmd.arg("--entry").arg(e);
        }
        for ap in &arg_pairs {
            cmd.arg("--arg").arg(ap);
        }
        for c in &caps {
            cmd.arg("--cap").arg(c);
        }
        if fuel != 200000 {
            cmd.arg("--fuel").arg(fuel.to_string());
        }

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let err = cmd.exec();
            eprintln!("\x1b[31mError:\x1b[0m failed to exec '{}': {err}", target_path.display());
            return 1;
        }

        #[cfg(not(unix))]
        {
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("\x1b[31mError:\x1b[0m failed to spawn '{}': {e}", target_path.display());
                    return 1;
                }
            };
            return child.wait().map(|s| s.code().unwrap_or(0)).unwrap_or(1);
        }
    }

    // Otherwise, it's a .kaze or .kazebc script: run the Rust supervisor loop!
    unsafe {
        libc::signal(libc::SIGINT, sig_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, sig_handler as libc::sighandler_t);
    }

    let stem = target_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("agent");

    println!("\x1b[35m[DAEMON]\x1b[0m Starting 24/7 supervisor loop for {stem}");
    println!(
        "\x1b[35m[DAEMON]\x1b[0m Interval: {interval_raw} ({interval_ms}ms) · Fuel: {fuel} · PID: {}",
        std::process::id()
    );
    println!("\x1b[35m[DAEMON]\x1b[0m Autonomous recovery active · Press Ctrl+C to stop gracefully.");

    let mut cycles = 0u64;
    let mut successes = 0u64;
    let mut failures = 0u64;
    let start_time = Instant::now();

    let mut opts = crate::sfs::RunOptions {
        fuel,
        entry,
        ..Default::default()
    };
    for c in &caps {
        let (fam, dev) = if let Some((f, d)) = c.split_once('=') {
            (f.trim().to_string(), d.trim().to_string())
        } else {
            (c.trim().to_string(), "sim".to_string())
        };
        opts.grants.push(fam.clone());
        if dev != "sim" {
            opts.backend.push((fam, dev));
        }
    }

    for ap in &arg_pairs {
        if let Some((k, v)) = ap.split_once('=') {
            let val = serde_json::from_str(v).unwrap_or_else(|_| serde_json::Value::String(v.to_string()));
            opts.args.insert(k.trim().to_string(), val);
        }
    }

    while RUNNING.load(Ordering::SeqCst) {
        cycles += 1;
        let elapsed = format_duration(start_time.elapsed());
        println!("\x1b[36m[CYCLE]\x1b[0m #{cycles} · uptime {elapsed}");

        let outcome = if target_path.extension().is_some_and(|e| e == "kazebc") {
            fs::read(target_path)
                .map_err(|e| crate::sfs::KazeError::new("E_KAZE_IO", e.to_string()))
                .and_then(|bytes| {
                    crate::sfs::isa::deserialize(&bytes)
                        .map_err(|e| crate::sfs::KazeError::new("E_KAZE_BADCODE", e))
                })
                .and_then(|m| crate::sfs::run_module(m, &target_path.to_string_lossy(), &opts))
        } else {
            crate::sfs::run_file(&target_path.to_string_lossy(), &opts)
        };

        match outcome {
            Ok(report) => {
                successes += 1;
                for act in &report.actions {
                    if act.contains("ai.") {
                        println!("\x1b[35m[AI]\x1b[0m {act}");
                    } else if act.contains("tool.") {
                        println!("\x1b[33m[TOOL]\x1b[0m {act}");
                    } else if act.contains("policy.") {
                        println!("\x1b[34m[POLICY]\x1b[0m {act}");
                    } else {
                        println!("\x1b[90m[STEP]\x1b[0m {act}");
                    }
                }
                for (level, msg) in &report.logs {
                    let col = match level.as_str() {
                        "error" => "\x1b[31m",
                        "warn" => "\x1b[33m",
                        _ => "\x1b[36m",
                    };
                    println!("{col}[{}]{col} {msg}\x1b[0m", level.to_uppercase());
                }
                if let Some(ref v) = report.verdict {
                    match v {
                        crate::sfs::vm::Verdict::Allow(r) => {
                            println!("\x1b[1m[VERDICT]\x1b[0m \x1b[32mALLOW\x1b[0m {r}");
                        }
                        crate::sfs::vm::Verdict::Reject(r) => {
                            println!("\x1b[1m[VERDICT]\x1b[0m \x1b[31mREJECT\x1b[0m {r}");
                        }
                    }
                } else if report.value != "nothing" && !report.value.is_empty() {
                    println!("\x1b[32m[RESULT]\x1b[0m {}", report.value);
                }
                println!(
                    "\x1b[90m{} instructions · {} fuel used\x1b[0m",
                    report.instructions, report.fuel_used
                );
            }
            Err(e) => {
                failures += 1;
                eprintln!("\x1b[31m[ERROR]\x1b[0m Cycle #{cycles} failed: {} — {}", e.code, e.message);
            }
        }

        // Sleep interval in 50ms chunks checking RUNNING
        let sleep_until = Instant::now() + Duration::from_millis(interval_ms);
        while Instant::now() < sleep_until && RUNNING.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    let uptime = format_duration(start_time.elapsed());
    println!();
    println!("\x1b[35m[DAEMON]\x1b[0m Shutdown signal received. Stopping supervisor loop...");
    println!(
        "\x1b[32m[SUMMARY]\x1b[0m Completed {cycles} cycles ({successes} ok, {failures} failed) · Uptime: {uptime}"
    );

    0
}
