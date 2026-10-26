// crates/sofuu-core/src/frontend/dev_server.rs
//
// AI-Native Frontend Dev Server & File Router in Rust.
// Serves unbundled ESM, transpiles JSX/TSX on-the-fly in RAM, and handles
// AI token streaming directly over HTTP/1.1 without external daemons.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use super::runtime::{HTML_SHELL, UI_JS_GZ};
use super::transpile::transpile;

static RELOAD_EPOCH: AtomicU64 = AtomicU64::new(1);

pub struct DevServerConfig {
    pub root_dir: PathBuf,
    pub host: String,
    pub port: u16,
    pub model_path: Option<PathBuf>,
}

pub fn start_dev_server(config: DevServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)?;
    let cfg = Arc::new(config);

    println!("\x1b[1;36m  ┌──────────────────────────────────────────────┐\x1b[0m");
    println!("\x1b[1;36m  │   ⚡ Sofuu AI Full-Stack App Server Ready     │\x1b[0m");
    println!("\x1b[1;36m  │   Local URL: \x1b[1;32mhttp://{}:{}\x1b[1;36m             │\x1b[0m", cfg.host, cfg.port);
    println!("\x1b[1;36m  │   Mode: Native Unbundled ESM (0ms boot)      │\x1b[0m");
    println!("\x1b[1;36m  └──────────────────────────────────────────────┘\x1b[0m\n");

    // Start background file watcher for instant live-reload
    let watch_dir = cfg.root_dir.clone();
    thread::spawn(move || {
        let mut last_max_mtime = scan_dir_latest_mtime(&watch_dir);
        loop {
            thread::sleep(std::time::Duration::from_millis(400));
            let current_max = scan_dir_latest_mtime(&watch_dir);
            if current_max > last_max_mtime {
                last_max_mtime = current_max;
                RELOAD_EPOCH.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let cfg_clone = Arc::clone(&cfg);
                thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &cfg_clone) {
                        eprintln!("[sofuu:dev_server] Connection error: {}", e);
                    }
                });
            }
            Err(e) => {
                eprintln!("[sofuu:dev_server] Accept error: {}", e);
            }
        }
    }

    Ok(())
}

fn scan_dir_latest_mtime(dir: &Path) -> std::time::SystemTime {
    let mut latest = std::time::SystemTime::UNIX_EPOCH;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            if path.is_dir() {
                let sub_latest = scan_dir_latest_mtime(&path);
                if sub_latest > latest {
                    latest = sub_latest;
                }
            } else if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if mtime > latest {
                        latest = mtime;
                    }
                }
            }
        }
    }
    latest
}

fn handle_connection(mut stream: TcpStream, cfg: &DevServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;

    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }

    let method = parts[0].to_string();
    let path = parts[1].split('?').next().unwrap_or("/").to_string();

    // Read remaining headers
    let mut content_length = 0usize;
    loop {
        line.clear();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        let lower = line.to_lowercase();
        if lower.starts_with("content-length:") {
            if let Some(val) = lower.split(':').nth(1) {
                content_length = val.trim().parse().unwrap_or(0);
            }
        }
    }

    // Read body if POST
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    // ── Route Dispatch ──────────────────────────────────────────────

    // 0. CORS Preflight
    if method == "OPTIONS" {
        let cors_headers = "HTTP/1.1 204 No Content\r\n\
            Access-Control-Allow-Origin: *\r\n\
            Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
            Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
            \r\n";
        stream.write_all(cors_headers.as_bytes())?;
        return Ok(());
    }

    // 1. Embedded Micro-Runtime: `/_sofuu/ui.js`, `/_sofuu/router.js`, `/_sofuu/ai.js`, `/_sofuu/store.js`
    if path == "/_sofuu/ui.js" || path == "/_sofuu/router.js" || path == "/_sofuu/ai.js" || path == "/_sofuu/store.js" {
        return send_response(
            &mut stream,
            200,
            "application/javascript; charset=utf-8\r\nContent-Encoding: gzip",
            UI_JS_GZ,
        );
    }

    // 1b. Dev Live Reload Stream: `/_sofuu/live`
    if path == "/_sofuu/live" {
        return handle_live_reload(&mut stream);
    }

    // 2. AI Chat Stream: `POST /api/chat`
    if method == "POST" && path == "/api/chat" {
        return handle_ai_stream(&mut stream, &body, cfg);
    }

    // 2b. AI Completion Stream: `POST /api/completion`
    if method == "POST" && path == "/api/completion" {
        return handle_ai_completion(&mut stream, &body, cfg);
    }

    // 3. HTML Shell Routes: `GET /` or subpage routes (`/chat`, `/about`)
    let clean_path = path.trim_start_matches('/');
    if (method == "GET" || method == "HEAD") && (path == "/" || (!clean_path.contains('.') && !cfg.root_dir.join(clean_path).is_file())) {
        let entry_route = detect_entry_route_for_path(&cfg.root_dir, clean_path);
        let html = HTML_SHELL.replace("{{ENTRY_ROUTE}}", &entry_route);
        return send_response(&mut stream, 200, "text/html; charset=utf-8", html.as_bytes());
    }

    // 4. Source Files (Transpiled on-the-fly): `.jsx`, `.tsx`, `.js`, `.ts`
    if let Some(target_file) = resolve_file_candidate(&cfg.root_dir, clean_path) {
        let ext = target_file.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext == "jsx" || ext == "tsx" || ext == "ts" || ext == "js" {
            let src = fs::read_to_string(&target_file)?;
            let is_ts = ext == "ts" || ext == "tsx";
            match transpile(&src, is_ts) {
                Ok(transpiled) => {
                    return send_response(&mut stream, 200, "application/javascript; charset=utf-8", transpiled.as_bytes());
                }
                Err(err) => {
                    let err_js = format!("console.error('Sofuu Transpile Error: {}');", err.replace('\'', "\\'"));
                    return send_response(&mut stream, 500, "application/javascript; charset=utf-8", err_js.as_bytes());
                }
            }
        }

        // Static files (CSS, images, etc.)
        let mime = get_mime_type(ext);
        let bytes = fs::read(&target_file)?;
        return send_response(&mut stream, 200, mime, &bytes);
    }

    // Also check `public/` folder for static assets
    let public_file = cfg.root_dir.join("public").join(clean_path);
    if public_file.exists() && public_file.is_file() {
        let ext = public_file.extension().and_then(|s| s.to_str()).unwrap_or("");
        let mime = get_mime_type(ext);
        let bytes = fs::read(&public_file)?;
        return send_response(&mut stream, 200, mime, &bytes);
    }

    // 404 Not Found
    send_response(&mut stream, 404, "text/plain", b"404 Not Found")
}

fn handle_live_reload(stream: &mut TcpStream) -> Result<(), Box<dyn std::error::Error>> {
    let headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: keep-alive\r\n\
        Access-Control-Allow-Origin: *\r\n\
        \r\n";
    stream.write_all(headers.as_bytes())?;
    stream.flush()?;

    let mut current_epoch = RELOAD_EPOCH.load(Ordering::Relaxed);
    loop {
        thread::sleep(std::time::Duration::from_millis(350));
        let latest = RELOAD_EPOCH.load(Ordering::Relaxed);
        if latest > current_epoch {
            current_epoch = latest;
            if stream.write_all(b"data: reload\n\n").is_err() || stream.flush().is_err() {
                break;
            }
        } else {
            // Heartbeat comment to keep connection alive
            if stream.write_all(b": ping\n\n").is_err() || stream.flush().is_err() {
                break;
            }
        }
    }
    Ok(())
}

fn resolve_file_candidate(root: &Path, clean_path: &str) -> Option<PathBuf> {
    let direct = root.join(clean_path);
    if direct.exists() && direct.is_file() {
        return Some(direct);
    }

    // Try extensionless candidates (e.g. `import Nav from "./Nav"`)
    let exts = ["jsx", "tsx", "js", "ts", "json", "css", "svg"];
    for ext in &exts {
        let candidate = root.join(format!("{}.{}", clean_path, ext));
        if candidate.exists() && candidate.is_file() {
            return Some(candidate);
        }
    }

    // Try directory index candidates (e.g. `import Comp from "./components"`)
    let index_exts = ["index.jsx", "index.tsx", "index.js", "index.ts"];
    for index_ext in &index_exts {
        let candidate = direct.join(index_ext);
        if candidate.exists() && candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn detect_entry_route_for_path(root: &Path, clean_path: &str) -> String {
    if !clean_path.is_empty() {
        let candidates = [
            format!("app/{}/page.jsx", clean_path),
            format!("app/{}/page.tsx", clean_path),
            format!("app/{}/page.js", clean_path),
            format!("pages/{}.jsx", clean_path),
            format!("pages/{}.tsx", clean_path),
            format!("pages/{}.js", clean_path),
        ];
        for c in &candidates {
            if root.join(c).exists() {
                return format!("/{}", c);
            }
        }
    }
    detect_entry_route(root)
}

fn detect_entry_route(root: &Path) -> String {
    let candidates = [
        "app/page.jsx",
        "app/page.tsx",
        "app/page.js",
        "pages/index.jsx",
        "pages/index.tsx",
        "pages/index.js",
        "src/App.jsx",
        "src/App.tsx",
    ];
    for c in &candidates {
        if root.join(c).exists() {
            return format!("/{}", c);
        }
    }
    "/app/page.jsx".to_string()
}

fn handle_ai_stream(stream: &mut TcpStream, body: &[u8], _cfg: &DevServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let response_headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: keep-alive\r\n\
        Access-Control-Allow-Origin: *\r\n\
        \r\n";
    stream.write_all(response_headers.as_bytes())?;
    stream.flush()?;

    let query = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(json) => {
            json.get("messages")
                .and_then(|m| m.as_array())
                .and_then(|arr| arr.last())
                .and_then(|last| last.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("Hello")
                .to_string()
        }
        Err(_) => "Hello".to_string(),
    };

    let sample_response = format!(
        "Hello! I am running completely inside Sofuu's single native binary with zero background daemons. You asked: \"{}\". The frontend and inference engine share the same memory space for instant sub-millisecond response!",
        query
    );

    let words: Vec<&str> = sample_response.split(' ').collect();
    for word in words {
        let payload = format!("data: {}\n\n", serde_json::json!({ "delta": format!("{} ", word) }));
        stream.write_all(payload.as_bytes())?;
        stream.flush()?;
        thread::sleep(std::time::Duration::from_millis(30));
    }

    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
}

fn handle_ai_completion(stream: &mut TcpStream, body: &[u8], _cfg: &DevServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let response_headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: keep-alive\r\n\
        Access-Control-Allow-Origin: *\r\n\
        \r\n";
    stream.write_all(response_headers.as_bytes())?;
    stream.flush()?;

    let prompt = match serde_json::from_slice::<serde_json::Value>(body) {
        Ok(json) => json.get("prompt").and_then(|p| p.as_str()).unwrap_or("Hello").to_string(),
        Err(_) => "Hello".to_string(),
    };

    let sample_response = format!("Completion for: \"{}\"", prompt);
    for word in sample_response.split(' ') {
        let payload = format!("data: {}\n\n", serde_json::json!({ "delta": format!("{} ", word) }));
        stream.write_all(payload.as_bytes())?;
        stream.flush()?;
        thread::sleep(std::time::Duration::from_millis(30));
    }

    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
}

fn send_response(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let status_line = match status {
        200 => "200 OK",
        404 => "404 Not Found",
        500 => "500 Internal Server Error",
        _ => "200 OK",
    };

    let headers = format!(
        "HTTP/1.1 {}\r\n\
        Content-Type: {}\r\n\
        Content-Length: {}\r\n\
        Access-Control-Allow-Origin: *\r\n\
        Connection: close\r\n\
        \r\n",
        status_line,
        content_type,
        body.len()
    );

    stream.write_all(headers.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn get_mime_type(ext: &str) -> &'static str {
    match ext {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
}
