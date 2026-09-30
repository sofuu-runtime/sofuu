// sofuu-desktop — Tauri backend (PLAN-DESKTOP D).
//
// The app shell: one window, a real NSMenu, and an engine worker thread
// that owns the single in-process SofuuRuntime. TypeScript is strictly the
// frontend UI; everything the runtime touches lives here in Rust.

mod commands;
mod config;
mod engine;
mod host;
mod keychain;
mod sessions;

use std::sync::Mutex;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{Emitter, Manager, RunEvent};

/// The engine thread's JoinHandle, parked for the exit path.
struct EngineThread(Mutex<Option<std::thread::JoinHandle<()>>>);

/// Menu item ids (NSMenu → the same actions the UI buttons call).
const MENU_NEW_CHAT: &str = "menu-new-chat";
const MENU_SETTINGS: &str = "menu-settings";
const MENU_STOP: &str = "menu-stop";
const MENU_COMPACT: &str = "menu-compact";
const MENU_ZOOM_IN: &str = "menu-zoom-in";
const MENU_ZOOM_OUT: &str = "menu-zoom-out";
const MENU_ZOOM_RESET: &str = "menu-zoom-reset";

fn build_menu(app: &tauri::AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let about = PredefinedMenuItem::about(app, Some("About Sofuu"), None)?;
    let settings = MenuItem::with_id(app, MENU_SETTINGS, "Settings…", true, Some("CmdOrCtrl+,"))?;
    let quit = PredefinedMenuItem::quit(app, Some("Quit Sofuu"))?;
    let app_menu = Submenu::with_items(
        app,
        "Sofuu",
        true,
        &[
            &about,
            &PredefinedMenuItem::separator(app)?,
            &settings,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;

    let new_chat = MenuItem::with_id(app, MENU_NEW_CHAT, "New Chat", true, Some("CmdOrCtrl+N"))?;
    let file_menu = Submenu::with_items(app, "File", true, &[&new_chat])?;

    let edit_menu = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &PredefinedMenuItem::undo(app, None)?,
            &PredefinedMenuItem::redo(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
        ],
    )?;

    let stop = MenuItem::with_id(app, MENU_STOP, "Stop", true, Some("CmdOrCtrl+."))?;
    let compact = MenuItem::with_id(app, MENU_COMPACT, "Compact Session", true, Some("CmdOrCtrl+K"))?;
    let session_menu = Submenu::with_items(app, "Session", true, &[&stop, &compact])?;

    // Standard macOS zoom accelerators; the frontend owns the zoom level
    // (persisted) and applies it as CSS zoom, so these route back to it.
    let zoom_in = MenuItem::with_id(app, MENU_ZOOM_IN, "Zoom In", true, Some("CmdOrCtrl+="))?;
    let zoom_out = MenuItem::with_id(app, MENU_ZOOM_OUT, "Zoom Out", true, Some("CmdOrCtrl+-"))?;
    let zoom_reset = MenuItem::with_id(app, MENU_ZOOM_RESET, "Actual Size", true, Some("CmdOrCtrl+0"))?;
    let view_menu = Submenu::with_items(
        app,
        "View",
        true,
        &[&zoom_in, &zoom_out, &zoom_reset],
    )?;

    let window_menu = Submenu::with_items(
        app,
        "Window",
        true,
        &[
            &PredefinedMenuItem::minimize(app, None)?,
            &PredefinedMenuItem::maximize(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::close_window(app, None)?,
        ],
    )?;

    Menu::with_items(
        app,
        &[&app_menu, &file_menu, &edit_menu, &view_menu, &session_menu, &window_menu],
    )
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        // Boot diagnostics: page-load lifecycle → stderr.
        .on_page_load(|webview, payload| {
            let url = webview.url().map(|u| u.to_string()).unwrap_or_default();
            eprintln!("[webview] load event={:?} url={}", payload.event(), url);
        })
        .setup(|app| {
            let handle = app.handle().clone();
            host::set_app_handle(handle.clone());

            // The engine worker boots off the main thread; the UI gets a
            // boot_error event if it fails (first launch without chat.js is
            // reported this way too).
            let project = config::project_dir();
            let (engine, thread) = engine::spawn(project);
            app.manage(engine.clone());
            app.manage(EngineThread(Mutex::new(Some(thread))));

            let boot_reporter = handle.clone();
            std::thread::Builder::new()
                .name("sofuu-boot-watch".into())
                .spawn(move || {
                    match engine.wait_ready(Duration::from_secs(60)) {
                        Ok(()) => {
                            let _ = boot_reporter.emit(
                                host::EVENT_CHANNEL,
                                serde_json::json!({ "kind": "engine_ready" }),
                            );
                        }
                        Err(e) => {
                            let _ = boot_reporter.emit(
                                host::EVENT_CHANNEL,
                                serde_json::json!({ "kind": "boot_error", "payload": { "message": e } }),
                            );
                        }
                    }
                })
                .ok();

            // Real NSMenu, wired to the same actions the UI calls.
            let menu = build_menu(app.handle())?;
            app.set_menu(menu)?;

            // Liquid-glass sidebar: a native vibrancy layer behind the
            // (transparent) webview. The CSS sidebar paints a translucent
            // warm tint over it, so only that column shows the desktop.
            #[cfg(target_os = "macos")]
            {
                use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial, NSVisualEffectState};
                if let Some(win) = app.get_webview_window("main") {
                    let _ = apply_vibrancy(
                        &win,
                        // UnderWindowBackground reads much clearer than the
                        // grey Sidebar film; the CSS tint keeps it warm.
                        NSVisualEffectMaterial::UnderWindowBackground,
                        Some(NSVisualEffectState::Active),
                        None,
                    );
                }
            }

            // Windows: the same glass effect via Mica (Win11) with an
            // acrylic fallback (Win10 1809+). Older systems keep a plain
            // opaque window — tauri.conf's transparent flag degrades fine.
            #[cfg(target_os = "windows")]
            {
                use window_vibrancy::{apply_acrylic, apply_mica};
                if let Some(win) = app.get_webview_window("main") {
                    // dark: None = follow the system preference (the theme
                    // effect re-applies live via Window::setTheme anyway).
                    if apply_mica(&win, None).is_err() {
                        let _ = apply_acrylic(&win, None);
                    }
                }
            }
            Ok(())
        })
        .on_menu_event(|app, event| match event.id().as_ref() {
            MENU_NEW_CHAT => {
                let _ = app.emit("menu://action", "new-chat");
            }
            MENU_SETTINGS => {
                let _ = app.emit("menu://action", "settings");
            }
            MENU_STOP => {
                // Mid-turn safe: the poke wakes the blocked engine.
                sofuu_core::rt::host_poke::host_poke_send(r#"{"type":"cancel"}"#);
                let _ = app.emit("menu://action", "stop");
            }
            MENU_COMPACT => {
                let _ = app.emit("menu://action", "compact");
            }
            MENU_ZOOM_IN => {
                let _ = app.emit("menu://action", "zoom-in");
            }
            MENU_ZOOM_OUT => {
                let _ = app.emit("menu://action", "zoom-out");
            }
            MENU_ZOOM_RESET => {
                let _ = app.emit("menu://action", "zoom-reset");
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            commands::send_turn,
            commands::cancel_turn,
            commands::resolve_approval,
            commands::list_sessions,
            commands::session_turns,
            commands::new_session,
            commands::resume_session,
            commands::set_permissions,
            commands::chat_state,
            commands::usage_report,
            commands::compact,
            commands::list_models,
            commands::refresh_model_cache,
            commands::get_config,
            commands::update_config,
            commands::get_desktop_state,
            commands::set_project_dir,
            commands::pick_project_dir,
            commands::pick_files,
            commands::keychain_has,
            commands::keychain_set,
            commands::keychain_delete,
            commands::remove_provider,
            commands::frontend_log,
            commands::delete_session,
            commands::clear_all_sessions,
            commands::list_tools,
            commands::list_agents,
            commands::brain_remember,
            commands::brain_why,
            commands::ml_info,
            commands::context_dump,
        ])
        .build(tauri::generate_context!())
        .expect("failed to build sofuu-desktop")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = &event {
                // Drop the runtime on its own thread before the process
                // exits (P0-7 teardown discipline), bounded so quit never
                // hangs on a wedged engine.
                if let Some(engine) = app_handle.try_state::<engine::EngineHandle>() {
                    engine.shutdown();
                }
                if let Some(t) = app_handle.try_state::<EngineThread>() {
                    if let Ok(mut guard) = t.0.lock() {
                        if let Some(thread) = guard.take() {
                            let _ = thread.join();
                        }
                    }
                }
            }
        });
}
