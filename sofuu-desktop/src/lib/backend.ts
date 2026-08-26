// lib/backend.ts — the typed bridge to the Tauri backend (PLAN-DESKTOP E).
//
// Every function maps 1:1 to a #[tauri::command] in src-tauri/commands.rs.
// Outside a Tauri webview (plain `npm run dev` in a browser) the calls fail
// cleanly so the UI can still be developed against mock data.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { DesktopState, SessionInfo, SofuuConfig } from "./events";

export const EVENT_CHANNEL = "sofuu://event";
export const MENU_CHANNEL = "menu://action";

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

// ── Turns ─────────────────────────────────────────────────────────

export function sendTurn(text: string, opts: Record<string, unknown> = {}): Promise<{ ok: boolean }> {
  return invoke("send_turn", { text, opts });
}

export function cancelTurn(): Promise<{ ok: boolean }> {
  return invoke("cancel_turn");
}

export function resolveApproval(id: string, allow: boolean, always = false): Promise<{ ok: boolean }> {
  return invoke("resolve_approval", { id, allow, always });
}

// ── Sessions ──────────────────────────────────────────────────────

export function listSessions(): Promise<SessionInfo[]> {
  return invoke("list_sessions");
}

export function sessionTurns(id: string): Promise<Array<{ prompt: string; answer: string }>> {
  return invoke("session_turns", { id });
}

export function newSession(): Promise<{ ok?: boolean; sessionId?: string; error?: string }> {
  return invoke("new_session");
}

export function resumeSession(id: string): Promise<{ ok?: boolean; sessionId?: string; error?: string }> {
  return invoke("resume_session", { id });
}

export function setPermissions(profile: string): Promise<{ ok?: boolean; permissions?: string; error?: string }> {
  return invoke("set_permissions", { profile });
}

export function chatState(): Promise<unknown> {
  return invoke("chat_state");
}

export function compactSession(): Promise<{ ok?: boolean; error?: string }> {
  return invoke("compact");
}

// ── Config ────────────────────────────────────────────────────────

export function getConfig(): Promise<SofuuConfig> {
  return invoke("get_config");
}

export function updateConfig(patch: Record<string, unknown>): Promise<SofuuConfig> {
  return invoke("update_config", { patch });
}

export function getDesktopState(): Promise<DesktopState> {
  return invoke("get_desktop_state");
}

export function setProjectDir(
  path: string
): Promise<{ ok?: boolean; sessionId?: string; project?: string; error?: string }> {
  return invoke("set_project_dir", { path });
}

export function pickProjectDir(): Promise<string | null> {
  return invoke("pick_project_dir");
}

/** Native multi-file picker (composer attach → @file mentions). */
export function pickFiles(): Promise<string[] | null> {
  return invoke("pick_files");
}

// ── Keychain ──────────────────────────────────────────────────────

export function keychainHas(provider: string): Promise<{ has_key: boolean }> {
  return invoke("keychain_has", { provider });
}

export function keychainSet(provider: string, key: string): Promise<{ ok: boolean }> {
  return invoke("keychain_set", { provider, key });
}

export function keychainDelete(provider: string): Promise<{ ok: boolean }> {
  return invoke("keychain_delete", { provider });
}

// ── Subscriptions ─────────────────────────────────────────────────

export function onEngineEvent(cb: (payload: Record<string, unknown>) => void): Promise<UnlistenFn> {
  return listen<Record<string, unknown>>(EVENT_CHANNEL, (e) => cb(e.payload));
}

export function onMenuAction(cb: (action: string) => void): Promise<UnlistenFn> {
  return listen<string>(MENU_CHANNEL, (e) => cb(e.payload));
}
