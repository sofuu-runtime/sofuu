// lib/backend.ts — the typed bridge to the Tauri backend (PLAN-DESKTOP E).
//
// Every function maps 1:1 to a #[tauri::command] in src-tauri/commands.rs.
// Outside a Tauri webview (plain `npm run dev` in a browser) the calls fail
// cleanly so the UI can still be developed against mock data.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { DesktopState, SessionInfo, SofuuConfig } from "./events";

/** The subset of chat.state() the Settings → General pane renders. */
export interface EngineState {
  ok?: boolean;
  project?: string | null;
  sessionId?: string | null;
  model?: string;
  provider?: string;
  effort?: string;
  streaming?: boolean;
  historyTurns?: number;
  contextTokens?: number;
  sessionSpendUsd?: number;
  permissionProfile?: string;
  pendingApprovals?: number;
}

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
  // Normalize defensively: the engine reply may arrive as {sessions:[…]}
  // or a bare array — a malformed shape must never reach a .map() and
  // blank the app (the TopBar crash class).
  return invoke("list_sessions")
    .then((raw) => {
      const arr = Array.isArray(raw)
        ? raw
        : (raw as { sessions?: unknown })?.sessions;
      return Array.isArray(arr) ? (arr as SessionInfo[]) : [];
    })
    .catch(() => [] as SessionInfo[]);
}

export function sessionTurns(id: string): Promise<Array<{ prompt: string; answer: string }>> {
  return invoke("session_turns", { id });
}

export function newSession(): Promise<{ ok?: boolean; sessionId?: string; error?: string }> {
  return invoke("new_session");
}

export function deleteSession(
  id: string
): Promise<{ ok?: boolean; wasActive?: boolean; error?: string }> {
  return invoke("delete_session", { id });
}

export function clearSessions(): Promise<{ ok?: boolean; cleared?: number; error?: string }> {
  return invoke("clear_all_sessions");
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

/** Cached model list for a provider — populated by the background
 *  refresh (never fetched at click time). `cached:false` means the
 *  prefetch has not reached this provider yet. */
export function listModels(name: string): Promise<{ ok?: boolean; models?: string[]; error?: string; note?: string; cached?: boolean; at?: number }> {
  return invoke("list_models", { name });
}

/** Fetch + cache the model lists of ALL providers in the engine (keys
 *  never reach the WebView). Call at start and on a timer. */
export function refreshModelCache(): Promise<{ ok?: boolean; error?: string; providers?: number }> {
  return invoke("refresh_model_cache");
}

export function compactSession(): Promise<{ ok?: boolean; error?: string }> {
  return invoke("compact");
}

// ── Parity surface (CLI slash parity) ─────────────────────────────

export function listTools(): Promise<{ ok?: boolean; builtin?: string[]; mcp?: unknown; error?: string }> {
  return invoke("list_tools");
}

export function listAgents(): Promise<{ ok?: boolean; agents?: unknown; error?: string }> {
  return invoke("list_agents");
}

export function brainRemember(fact: string): Promise<{ ok?: boolean; error?: string }> {
  return invoke("brain_remember", { fact });
}

export function brainWhy(): Promise<{ ok?: boolean; lastRecall?: unknown; recall?: unknown; error?: string }> {
  return invoke("brain_why");
}

export function mlInfo(): Promise<{ ok?: boolean; info?: unknown; error?: string }> {
  return invoke("ml_info");
}

export function contextDump(): Promise<{ ok?: boolean; state?: unknown; error?: string }> {
  return invoke("context_dump");
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

/** Delete a provider end-to-end: config entry, Keychain key, in-memory
 *  engine key, cached model list. The active provider is refused
 *  (switch first). Returns the redacted config for the caller to adopt. */
export function removeProvider(provider: string): Promise<SofuuConfig> {
  return invoke("remove_provider", { provider });
}

/** Live engine state (chat.state()): session id, tokens, spend,
 *  permission profile — used by Settings → General to show what the
 *  engine is actually doing right now. Alias of chatState() with a typed
 *  shape (both invoke the same `chat_state` command; kept separate so
 *  existing call sites don't churn). */
export function engineState(): Promise<EngineState | null> {
  return chatState()
    .then((raw) => (raw && typeof raw === "object" ? (raw as EngineState) : null))
    .catch(() => null);
}

// ── Usage (Settings → Usage) ──────────────────────────────────────

export interface UsageModelRow {
  model: string;
  pt: number;
  ct: number;
  cr: number;
  cw: number;
  turns: number;
  cost: number;
}

export interface UsageDayRow {
  day: string; // YYYY-MM-DD
  pt: number;
  ct: number;
  turns: number;
}

export interface UsageSessionRow {
  id: string;
  model: string;
  task: string | null;
  started_at: number;
  last_seen: number;
  pt: number;
  ct: number;
  turns: number;
  cost: number;
}

export interface UsageReport {
  ok: boolean;
  project: string | null;
  models: UsageModelRow[];
  days: UsageDayRow[];
  sessions: UsageSessionRow[];
  totals: {
    pt: number;
    ct: number;
    cr: number;
    cw: number;
    turns: number;
    cost: number;
    sessions: number;
  };
}

/** Aggregated per-turn usage across every session in the workspace —
 *  computed inside the engine from the encrypted session store. */
export function usageReport(): Promise<UsageReport | null> {
  return invoke("usage_report")
    .then((raw) => (raw && typeof raw === "object" && (raw as UsageReport).ok ? (raw as UsageReport) : null))
    .catch(() => null);
}

// ── Subscriptions ─────────────────────────────────────────────────

export function onEngineEvent(cb: (payload: Record<string, unknown>) => void): Promise<UnlistenFn> {
  return listen<Record<string, unknown>>(EVENT_CHANNEL, (e) => cb(e.payload));
}

export function onMenuAction(cb: (action: string) => void): Promise<UnlistenFn> {
  return listen<string>(MENU_CHANNEL, (e) => cb(e.payload));
}
