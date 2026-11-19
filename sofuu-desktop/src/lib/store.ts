// lib/store.ts — app state: folds the engine event stream into UI state.

import { useCallback, useEffect, useReducer, useRef } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import * as backend from "./backend";
import {
  MAX_SESSIONS,
  parseEvent,
  type Approval,
  type ChatMessage,
  type CtxMeter,
  type IncomingEvent,
  type SessionInfo,
  type SofuuConfig,
  type StepEvent,
  type TimelineEntry,
  type Usage,
} from "./events";
import { formatDur, mlgateTitle, toolRow } from "./timeline";
import { SLASH_COMMANDS, slashSignature } from "./commands";

export interface AppState {
  engineReady: boolean;
  bootError: string | null;
  sessions: SessionInfo[];
  /** Why the session list could not be read (engine unreachable, registry
   *  unreadable). Set while `sessions` holds whatever the last successful
   *  read produced — an empty list must never stand for a failed read. */
  sessionsError: string | null;
  /** Where the session list came from: the engine's registry, or the
   *  on-disk registry mirror (the engine did not answer). Null while the
   *  list is the engine's own. */
  sessionsSource: string | null;
  /** Id of the session currently shown in the chat pane, or null. */
  activeSessionId: string | null;
  messages: ChatMessage[];
  /** Transcript per session id — parked when switching pills so each
   *  session keeps its own page content. */
  messageCache: Record<string, ChatMessage[]>;
  streaming: boolean;
  approvals: Approval[];
  config: SofuuConfig | null;
  projectDir: string | null;
  /** Known workspace folder paths (sidebar), newest first. */
  workspaces: string[];
  /** Tool permission profile: full / edit / plan / prompt (ask). */
  permissions: PermissionProfile;
  /** Why the profile in force is not the one the shell asked for — the
   *  engine refused a permission change, so the shown profile is not
   *  necessarily the one the runtime enforces. */
  permissionsError: string | null;
  settingsOpen: boolean;
  /** Settings tab the modal opens on (model picker jumps to Providers). */
  settingsTab: string;
  /** Chat body font size in px (Appearance → Chat font size). */
  chatFontSize: number;
  /** Interface zoom factor (⌘+/⌘-/⌘0, View menu, Appearance). */
  zoom: number;
  /** Composer mode: agent = tools enabled, chat = pure conversation. */
  mode: "agent" | "chat";
  /** Read-only transcript being viewed (session history), if any. */
  preview: SessionPreview | null;
  /** Sidebar collapsed to the icon rail (⌘B). */
  sidebarCollapsed: boolean;
  /** Appearance → procedural cloud background toggle (off = flat gradient). */
  cloudBg: boolean;
  /** Appearance → theme preference. "system" follows macOS live. */
  theme: ThemePref;
  /** Resolved dark flag (theme pref + macOS), updated live so the cloud
   *  background and any non-CSS consumers can react to System flips. */
  resolvedDark: boolean;
  /** Live phase label for the streaming turn, mirroring the TUI's
   *  phaseForStep mapping ("Thinking", "Web searching", "Coding", …).
   *  Null = the turn has not reached a labelled phase yet. */
  phase: string | null;
  /** Live context-window meter for the composer ring: used tokens vs the
   * resolved window (engine 'ctx' events — first-request calibrated).
   * Null until the engine reports, so the ring stays hidden rather than
   * showing a made-up number. BELONGS TO THE ACTIVE SESSION: parked and
   * restored with the transcript (see ctxCache) — never shared across
   * sessions or workspaces. */
  ctx: CtxMeter | null;
  /** The meter per session id, same discipline as messageCache: parked
   * on switch, restored on return, dropped when the session is deleted.
   * ctx events carry the owning sid so a late event from a backgrounded
   * turn updates the right entry, not the visible one. */
  ctxCache: Record<string, CtxMeter>;
}

/** A past session's transcript loaded for read-only viewing. */
export interface SessionPreview {
  id: string;
  label: string;
  turns: Array<{ prompt: string; answer: string }>;
}

function storedZoom(): number {
  try {
    const n = Number(localStorage.getItem("sofuu.zoom"));
    if (n >= 0.5 && n <= 2) return n;
  } catch {
    /* no localStorage */
  }
  return 1;
}

function storedMode(): "agent" | "chat" {
  try {
    if (localStorage.getItem("sofuu.mode") === "chat") return "chat";
  } catch {
    /* no localStorage */
  }
  return "agent";
}

function storedSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem("sofuu.sidebar") === "hidden";
  } catch {
    /* no localStorage */
  }
  return false;
}

export type PermissionProfile = "full" | "edit" | "plan" | "prompt";

function storedPermissions(): PermissionProfile {
  try {
    const p = localStorage.getItem("sofuu.permissions");
    if (p === "full" || p === "edit" || p === "plan") return p;
  } catch {
    /* no localStorage */
  }
  return "prompt";
}

/** Appearance → cloud background (off = flat gradient). Defaults on. */
function storedCloudBg(): boolean {
  try {
    return localStorage.getItem("sofuu.cloudBg") !== "off";
  } catch {
    /* no localStorage */
  }
  return true;
}

export type ThemePref = "light" | "dark" | "system";

/** Appearance → theme. Defaults to following macOS. */
function storedTheme(): ThemePref {
  try {
    const t = localStorage.getItem("sofuu.theme");
    if (t === "light" || t === "dark" || t === "system") return t;
  } catch {
    /* no localStorage */
  }
  return "system";
}

/** Known workspaces (folder paths picked in the sidebar), newest first. */
function storedWorkspaces(): string[] {
  try {
    const raw = localStorage.getItem("sofuu.workspaces");
    const arr = raw ? JSON.parse(raw) : null;
    if (Array.isArray(arr)) return arr.filter((p): p is string => typeof p === "string");
  } catch {
    /* no localStorage */
  }
  return [];
}

const initialState: AppState = {
  engineReady: false,
  bootError: null,
  sessions: [],
  sessionsError: null,
  sessionsSource: null,
  activeSessionId: null,
  messages: [],
  messageCache: {},
  streaming: false,
  approvals: [],
  config: null,
  projectDir: null,
  workspaces: storedWorkspaces(),
  permissions: storedPermissions(),
  permissionsError: null,
  settingsOpen: false,
  settingsTab: "general",
  chatFontSize: 15,
  zoom: storedZoom(),
  mode: storedMode(),
  preview: null,
  sidebarCollapsed: storedSidebarCollapsed(),
  cloudBg: storedCloudBg(),
  theme: storedTheme(),
  resolvedDark: storedTheme() === "dark" ||
    (storedTheme() === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches),
  phase: null,
  ctx: null,
  ctxCache: {},
};

type Action =
  | { type: "event"; event: IncomingEvent }
  | { type: "user_submitted"; text: string }
  | { type: "system_row"; text: string }
  | { type: "delete_session"; id: string }
  | { type: "sessions"; sessions: SessionInfo[]; projectDir?: string | null; error?: string | null; source?: string | null }
  | { type: "workspace_switch"; path: string | null }
  | { type: "sessions_cleared" }
  | { type: "add_session"; session: SessionInfo }
  | { type: "active_session"; id: string | null }
  | { type: "config"; config: SofuuConfig }
  | { type: "project_dir"; path: string | null }
  | { type: "workspaces"; workspaces: string[] }
  | { type: "permissions"; permissions: PermissionProfile }
  | { type: "permissions_error"; error: string | null }
  | { type: "settings_open"; open: boolean; tab?: string }
  | { type: "chat_font_size"; size: number }
  | { type: "zoom"; zoom: number }
  | { type: "zoom_delta"; delta: number }
  | { type: "mode"; mode: "agent" | "chat" }
  | { type: "preview"; preview: SessionPreview | null }
  | { type: "sidebar_collapsed"; collapsed: boolean }
  | { type: "cloud_bg"; on: boolean }
  | { type: "theme"; theme: ThemePref }
  | { type: "resolved_dark"; dark: boolean }
  | { type: "new_chat" };

let seq = 0;
function nextId(prefix: string): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

/** Optimistic pill row for a session the engine just created.
 *
 *  The model/provider are left EMPTY on purpose: the engine has not
 *  reported which model serves this session yet (it can differ from the
 *  configured one after a failover), and the config's model is not the
 *  session's model. The next real session list fills them in. */
function makeSessionInfo(id: string, cwd: string): SessionInfo {
  const now = Math.floor(Date.now() / 1000);
  return {
    id,
    pid: 0,
    host: "local",
    cwd,
    model: "",
    provider: "",
    started_at: now,
    last_seen: now,
    task: "",
    ended: false,
  };
}

/** The assistant message currently receiving stream data, if any. */
function currentAssistant(messages: ChatMessage[]): ChatMessage | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") return messages[i];
  }
  return null;
}

/** Live phase label, kept in lockstep with the TUI's phaseForStep
 *  (chat.rs): plan/think/recall → Thinking, tool/delegate name-matched
 *  → Web searching / Debugging / Coding / Auditing / Working, and null
 *  means "keep the current phase" (tool_result: the phase that started
 *  the work keeps showing until the next step arrives). */
/** The engine clips tool args to 200 chars (JSON.stringify), so the
 *  string often isn't valid JSON — parse defensively, null on failure. */
function parseArgs(raw: unknown): Record<string, unknown> | null {
  if (typeof raw !== "string" || !raw) return null;
  try {
    const v = JSON.parse(raw);
    return v && typeof v === "object" ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function phaseForStep(kind: string, payload?: Record<string, unknown>): string | null {
  const p = payload ?? {};
  if (kind === "plan" || kind === "think" || kind === "recall") return "Thinking";
  if (kind === "tool" || kind === "delegate") {
    const name = String(p.name ?? p.agent ?? p.tool ?? p.action ?? "");
    if (/web_search|web_open|search/i.test(name)) return "Web searching";
    if (/debug|trace|error|fix|diagnos/i.test(name)) return "Debugging";
    if (/write|create|edit|implement|refactor|generate/i.test(name) && !/debug|test/i.test(name))
      return "Coding";
    if (/read|list|scan|grep|inspect|review|audit|test|lint|check|verify/i.test(name))
      return "Auditing";
    return "Working";
  }
  if (kind === "tool_result") return null; /* keep the preceding phase */
  if (kind === "compact") return "Compressing";
  if (kind === "answer_delta") return "Working";
  if (kind.startsWith("rlm")) return "Working";
  return null;
}

function stepToTimeline(step: StepEvent): TimelineEntry | null {
  const p = step.payload ?? {};
  switch (step.kind) {
    case "tool": {
      const name = String(p.tool ?? p.name ?? "tool");
      // delegate/todo_write have richer sibling events (delegate → agent
      // row with the task; todo_write result → live checklist row); a
      // tool row here would just duplicate them.
      if (name === "delegate" || name === "todo_write") return null;
      const entry = toolRow(name, parseArgs(p.args));
      return {
        id: nextId("tl"),
        kind: "tool",
        tool: name,
        title: entry.title,
        detail: entry.detail,
        t: step.t,
      };
    }
    case "tool_result":
      // todo_write (P2): the checklist rides the payload — a live
      // checklist row instead of an opaque "checklist updated" line.
      // (Every other result merges into its call row in applyStep —
      // this path is only the no-match fallback: system notices,
      // results after a reload.)
      if (p.name === "todo_write" && Array.isArray(p.todos)) {
        const todos = p.todos as Array<{ content: string; status: string }>;
        const done = todos.filter((t) => t.status === "done").length;
        return {
          id: nextId("tl"),
          kind: "todo",
          title: "Checklist",
          detail: JSON.stringify(todos),
          badge: `· ${done}/${todos.length} done`,
          t: step.t,
        };
      }
      {
        const name = String(p.tool ?? p.name ?? "");
        const resultText =
          typeof p.result === "string"
            ? p.result.slice(0, 400)
            : typeof p.summary === "string"
              ? p.summary
              : undefined;
        // The engine's system notices (retries, approvals, errors) ride
        // tool_result with name:"system" — those read as "Notice", not a
        // tool call.
        if (name === "system") {
          return {
            id: nextId("tl"),
            kind: "tool_result",
            tool: "system",
            title: "Notice",
            result: resultText,
            ok: p.error == null,
            t: step.t,
          };
        }
        return {
          id: nextId("tl"),
          kind: "tool_result",
          tool: name || undefined,
          title: name || "result",
          result: resultText,
          ok: p.error == null,
          t: step.t,
        };
      }
    case "delegate":
      return { id: nextId("tl"), kind: "delegate", tool: "delegate", title: String(p.agent ?? "sub-agent"), detail: typeof p.task === "string" ? p.task : undefined, t: step.t };
    case "recall": {
      // One line, count on the row itself: "Recalled 15 memories".
      const count = Number(p.count ?? 0);
      const bits: string[] = [];
      if (count > 0) bits.push(`${count.toLocaleString()} ${count === 1 ? "memory" : "memories"}`);
      if (p.scope === "agent") bits.push("private");
      else if (p.scope === "shared") bits.push("shared brain");
      if (Number(p.dropped ?? 0) > 0) bits.push(`${Number(p.dropped)} dropped`);
      return {
        id: nextId("tl"),
        kind: "recall",
        title: "Recalled memories",
        badge: bits.length ? `· ${bits.join(" · ")}` : undefined,
        t: step.t,
      };
    }
    case "plan":
      // Dropped: the plan's only payload is "Step N" — a row that says
      // nothing the work rows don't already show. It cost a line per
      // step (the old "Plan / bash / Plan / bash" spam).
      return null;
    case "warn": {
      const msg = typeof p.message === "string" ? p.message : "";
      // Compaction is not a warning — it's the context gate doing its
      // job; give it its own honest row. The ML junk-drop and the
      // summarizer fold are two different compactions — distinct titles
      // so one turn never shows two identical rows.
      if (/^ml-compaction/.test(msg)) {
        const freed = /freed ~([\d,]+) tk/.exec(msg)?.[1];
        return {
          id: nextId("tl"),
          kind: "warn",
          title: "Context pruned",
          badge: freed ? `· ~${freed} tk freed` : undefined,
          detail: msg,
          t: step.t,
        };
      }
      if (/compact/i.test(msg)) {
        return { id: nextId("tl"), kind: "warn", title: "Context compacted", detail: msg, t: step.t };
      }
      return { id: nextId("tl"), kind: "warn", title: "Warning", detail: msg || undefined, t: step.t };
    }
    case "consolidated": {
      const clusters = Number(p.clusters ?? 0);
      return {
        id: nextId("tl"),
        kind: "consolidated",
        title: "Memory consolidated",
        badge: clusters > 0 ? `· ${clusters} ${clusters === 1 ? "cluster" : "clusters"}` : undefined,
        t: step.t,
      };
    }
    case "allocgate": {
      // The alloc gate's visible decisions: fit ratio, clamps, learning.
      const bits: string[] = [];
      if (p.fit) bits.push(String(p.fit));
      if (p.actions) bits.push(String(p.actions));
      if (p.action === "learned_limit") {
        bits.push(`learned ${String(p.kind ?? "limit")} for ${String(p.model ?? "model")}`);
      }
      return {
        id: nextId("tl"),
        kind: "allocgate",
        title: "Context gate",
        detail: bits.join(" — ") || undefined,
        t: step.t,
      };
    }
    case "mlgate": {
      // Each ML rule gets its own row + icon (freshness/relevance/loop/
      // supervisor) — the models are visible activity, not one blob.
      const rule = String(p.rule ?? "");
      return {
        id: nextId("tl"),
        kind: "mlgate",
        tool: rule || undefined,
        title: mlgateTitle(rule),
        detail: typeof p.nudge === "string" ? p.nudge : rule || undefined,
        t: step.t,
      };
    }
    default:
      if (step.kind.startsWith("rlm")) {
        // rlm payloads vary by subtype: trace events carry a bare
        // (already-clipped) repr string; route/done/fallback carry
        // objects with their own fields.
        const raw = step.payload;
        let detail: string | undefined;
        if (typeof raw === "string") {
          detail = raw;
        } else if (raw && typeof raw === "object") {
          const o = raw as Record<string, unknown>;
          if (step.kind === "rlm:route") {
            detail = `routed to RLM (${String(o.ctxTokens ?? "?")}/${String(o.windowTokens ?? "?")} tk)`;
          } else if (step.kind === "rlm:done") {
            const bits: string[] = [];
            if (Number(o.calls ?? 0) > 0) bits.push(`${o.calls} LLM calls`);
            if (Number(o.toolCalls ?? 0) > 0) bits.push(`${o.toolCalls} tool calls`);
            if (Number(o.rounds ?? 0) > 0) bits.push(`${o.rounds} rounds`);
            detail = bits.join(" · ") || undefined;
          } else if (step.kind === "rlm:fallback") {
            detail = typeof o.error === "string" ? o.error : undefined;
          }
        }
        return {
          id: nextId("tl"),
          kind: "rlm",
          tool: step.kind,
          title: "RLM run",
          detail,
          t: step.t,
        };
      }
      return null;
  }
}

function applyStep(state: AppState, step: StepEvent): AppState {
  const assistant = currentAssistant(state.messages);
  if (!assistant) return state;
  const phase = phaseForStep(step.kind, step.payload);

  // answer_delta: append raw text to the streaming answer.
  if (step.kind === "answer_delta") {
    const text = String(step.payload?.text ?? "");
    if (!text) return { ...state, phase: phase ?? state.phase };
    return patchAssistant(
      state,
      (m) => ({ ...m, content: m.content + text }),
      { phase: phase ?? state.phase }
    );
  }
  // think: reasoning renders as timeline rows — "Thought for 1.2s" with
  // the full text one click away — not a separate transcript pane. A
  // run of consecutive think chunks is ONE row (duration = last chunk
  // t − first chunk t, both ms since run start); the next non-think
  // step starts a fresh row for the next reasoning segment.
  if (step.kind === "think") {
    const text = String(step.payload?.text ?? "");
    if (!text) return { ...state, phase: phase ?? state.phase };
    return patchAssistant(
      state,
      (m) => {
        const timeline = m.timeline.slice();
        const last = timeline[timeline.length - 1];
        if (last && last.kind === "think" && last.runId === step.runId) {
          const started = typeof last.t === "number" ? last.t : step.t;
          const durationMs =
            typeof step.t === "number" && typeof started === "number"
              ? Math.max(0, step.t - started)
              : undefined;
          timeline[timeline.length - 1] = {
            ...last,
            detail: (last.detail ?? "") + text,
            durationMs,
            badge: durationMs != null && durationMs > 0 ? `· ${formatDur(durationMs)}` : last.badge,
          };
          return { ...m, timeline };
        }
        const entry: TimelineEntry = {
          id: nextId("tl"),
          kind: "think",
          runId: step.runId,
          title: "Thought",
          detail: text,
          t: step.t,
        };
        return { ...m, timeline: [...timeline, entry] };
      },
      { phase: phase ?? state.phase }
    );
  }

  // One row per tool call: a result merges into its pending call row
  // ("Read AGENTS.md" + ✓ + result text under the same row) instead of
  // rendering a second row for the same tool. Delegate results merge
  // into the delegate row (the call row is suppressed; the title there
  // is the child agent's name). Without a match (system notices,
  // results after a reload) the standalone tool_result row still
  // renders.
  if (step.kind === "tool_result") {
    const rp = step.payload ?? {};
    const isChecklist = rp.name === "todo_write" && Array.isArray(rp.todos);
    if (!isChecklist) {
      const ok = rp.error == null;
      const result =
        typeof rp.result === "string"
          ? rp.result.slice(0, 400)
          : typeof rp.summary === "string"
            ? rp.summary
            : undefined;
      const name = String(rp.name ?? "");
      return patchAssistant(
        state,
        (m) => {
          for (let i = 0; i < m.timeline.length; i++) {
            const e = m.timeline[i];
            const match =
              name === "delegate"
                ? e.kind === "delegate" && e.ok === undefined
                : (e.kind === "tool" || e.kind === "tool_result") &&
                  e.ok === undefined &&
                  (e.tool ? e.tool === name : e.title === name);
            if (!match) continue;
            const timeline = m.timeline.slice();
            timeline[i] = { ...e, ok, result: result ?? e.result };
            return { ...m, timeline };
          }
          const fallback = stepToTimeline(step);
          return fallback ? { ...m, timeline: [...m.timeline, fallback] } : m;
        },
        { phase: phase ?? state.phase }
      );
    }
  }

  // compact: the auto-compaction lifecycle. start → ONE live row (the
  // "Compressing" sweep — compaction runs at the very end of a turn,
  // after `done`, so the animation must live in the timeline, not the
  // phase line); the completion warn stashes onto that row as its
  // detail instead of a second row; end settles it to "Context
  // compacted · N tk saved" — or removes it when nothing folded
  // (ok:false): a no-op compaction leaves no trace.
  if (step.kind === "compact") {
    const stage = String(step.payload?.stage ?? "");
    return patchAssistant(
      state,
      (m) => {
        const timeline = m.timeline.slice();
        const idx = timeline.findIndex((e) => e.kind === "compact" && e.live);
        if (stage === "start") {
          if (idx >= 0) return m; // one pending row at a time
          timeline.push({
            id: nextId("tl"),
            kind: "compact",
            title: "Compressing",
            live: true,
            t: step.t,
          });
          return { ...m, timeline };
        }
        // stage "end"
        if (idx < 0) return m;
        if (step.payload?.ok === false) {
          timeline.splice(idx, 1);
          return { ...m, timeline };
        }
        const saved = /([\d,]+) tk saved/.exec(timeline[idx].detail ?? "")?.[1];
        timeline[idx] = {
          ...timeline[idx],
          live: false,
          title: "Context compacted",
          badge: saved ? `· ${saved} tk saved` : undefined,
        };
        return { ...m, timeline };
      },
      { phase: phase ?? state.phase }
    );
  }

  // While a compact row is live, the compaction warn is that row's
  // completion note ("auto-compacted → summary (N tk saved…)" — the
  // end step turns it into the badge), not a second row. Warns with no
  // pending row (ml-compaction's instant junk-drop, history trim, or
  // events after a reload) fall through to the normal warn row.
  if (step.kind === "warn" && /compact/i.test(String(step.payload?.message ?? ""))) {
    const pending = assistant.timeline.some((e) => e.kind === "compact" && e.live);
    if (pending) {
      const msg = String(step.payload!.message);
      return patchAssistant(
        state,
        (m) => {
          const timeline = m.timeline.slice();
          const idx = timeline.findIndex((e) => e.kind === "compact" && e.live);
          if (idx >= 0) timeline[idx] = { ...timeline[idx], detail: msg };
          return { ...m, timeline };
        },
        { phase: phase ?? state.phase }
      );
    }
  }

  const entry = stepToTimeline(step);
  if (!entry) return { ...state, phase: phase ?? state.phase };
  return patchAssistant(
    state,
    (m) => ({ ...m, timeline: [...m.timeline, { ...entry, runId: step.runId }] }),
    { phase: phase ?? state.phase }
  );
}

function patchAssistant(
  state: AppState,
  fn: (m: ChatMessage) => ChatMessage,
  extra: Partial<AppState> = {}
): AppState {
  const messages = [...state.messages];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      messages[i] = fn(messages[i]);
      break;
    }
  }
  return { ...state, ...extra, messages };
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "event": {
      const e = action.event;
      switch (e.type) {
        case "engine_ready":
          return { ...state, engineReady: true, bootError: null };
        case "boot_error":
          return { ...state, engineReady: false, bootError: e.message };
        case "turn_started":
          return state; // the assistant bubble is created on user_submitted
        case "turn_finished":
          return { ...state, streaming: false, phase: null };
        case "turn_error":
          return {
            ...state,
            streaming: false,
            phase: null,
            ...patchAssistantStatus(state, "error", e.message),
          };
        case "approval_request":
          return { ...state, approvals: [...state.approvals, e.approval] };
        case "approval_resolved":
          return { ...state, approvals: state.approvals.filter((a) => a.id !== e.id) };
        case "ctx": {
          // Attribute the meter to its owning session. The engine tags
          // every ctx event with the session it belongs to; without a tag
          // (older engine), fall back to the active session. Either way
          // the meter only lands on the VISIBLE ring when the owner IS
          // the active session — a backgrounded session's event just
          // updates its parked cache entry.
          const owner = e.ctx.sid ?? state.activeSessionId;
          if (!owner) return state;
          const cache = { ...state.ctxCache, [owner]: e.ctx };
          return owner === state.activeSessionId
            ? { ...state, ctx: e.ctx, ctxCache: cache }
            : { ...state, ctxCache: cache };
        }
        case "done":
          return {
            ...state,
            streaming: false,
            phase: null,
            ...patchAssistantDone(state, e.usage),
          };
        case "step":
          return applyStep(state, e.step);
        case "unknown":
          return state;
      }
      return state;
    }
    case "system_row": {
      const row: ChatMessage = {
        id: nextId("m"),
        role: "system",
        content: action.text,
        timeline: [],
        status: "done",
      };
      return { ...state, messages: [...state.messages, row] };
    }
    case "user_submitted": {
      const user: ChatMessage = {
        id: nextId("m"),
        role: "user",
        content: action.text,
        timeline: [],
        status: "done",
      };
      const assistant: ChatMessage = {
        id: nextId("m"),
        role: "assistant",
        content: "",
        timeline: [],
        status: "streaming",
      };
      return {
        ...state,
        messages: [...state.messages, user, assistant],
        streaming: true,
        // The TUI starts every turn in "Thinking" — setup runs first, and
        // the labelled phase takes over as soon as a step names one.
        phase: "Thinking",
      };
    }
    case "sessions": {
      // A FAILED read keeps the last known list and records why: an empty
      // strip must never be presented as "this workspace has no chats".
      if (action.error) {
        return { ...state, sessionsError: action.error, sessionsSource: action.source ?? null };
      }
      // Workspace tag: a fetch issued for a folder other than the one
      // currently shown raced a workspace switch (the engine serializes
      // queries, so an in-flight listSessions can execute on either side
      // of setProjectDir) — its reply belongs to the OLD strip. Drop it.
      if (action.projectDir && state.projectDir && action.projectDir !== state.projectDir) {
        return state;
      }
      // MERGE, never clobber: the fetched list may be STALE — it was read
      // on the engine thread before a just-minted session's registry write
      // landed, and clobbering here parked the live view mid-turn (the
      // "first hi vanishes" bug). Union by id: fetched entries win on
      // update, entries only in current state (newer than the fetch) stay.
      const fetched = Array.isArray(action.sessions) ? action.sessions : [];
      const byId = new Map(fetched.map((s) => [s.id, s]));
      for (const existing of state.sessions) {
        if (!byId.has(existing.id)) byId.set(existing.id, existing);
      }
      const list = Array.from(byId.values()).sort((a, b) => (b.last_seen || 0) - (a.last_seen || 0));
      // While a turn streams, NEVER park the live view: keep the active
      // id and the in-flight messages regardless of list contents.
      if (state.streaming) {
        return { ...state, sessions: list, sessionsError: null, sessionsSource: action.source ?? null };
      }
      const nextActive =
        state.activeSessionId && list.some((s) => s.id === state.activeSessionId)
          ? state.activeSessionId
          : list[0]?.id ?? null;
      if (nextActive === state.activeSessionId) {
        return { ...state, sessions: list, sessionsError: null, sessionsSource: action.source ?? null };
      }
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      const ctxCache = { ...state.ctxCache };
      if (state.activeSessionId && state.ctx) ctxCache[state.activeSessionId] = state.ctx;
      return {
        ...state,
        sessions: list,
        sessionsError: null,
        sessionsSource: action.source ?? null,
        activeSessionId: nextActive,
        messages: nextActive ? cache[nextActive] ?? [] : [],
        approvals: [],
        preview: null,
        messageCache: cache,
        ctxCache,
        ctx: nextActive ? ctxCache[nextActive] ?? null : null,
      };
    }
    case "delete_session": {
      const wasActive = state.activeSessionId === action.id;
      const cache = { ...state.messageCache };
      delete cache[action.id];
      const ctxCache = { ...state.ctxCache };
      delete ctxCache[action.id];
      const rest = state.sessions.filter((s) => s.id !== action.id);
      return {
        ...state,
        sessions: rest,
        activeSessionId: wasActive ? null : state.activeSessionId,
        messages: wasActive ? [] : state.messages,
        preview: wasActive ? null : state.preview,
        messageCache: cache,
        ctxCache,
        ctx: wasActive ? null : state.ctx,
      };
    }
    case "add_session":
      // Newest first — the engine's list_sessions sorts by last_seen desc.
      return { ...state, sessions: [action.session, ...state.sessions] };
    case "active_session": {
      // Park the old page's transcript, restore the new one (empty if the
      // session never had a transcript in this window). A streaming page
      // is never parked — the live turn belongs to the view until done.
      // The ctx meter moves the same way: each session's ring state is
      // its own, never the folder's or the window's.
      if (action.id === state.activeSessionId) return state;
      if (state.streaming) {
        const streamingCache = { ...state.ctxCache };
        if (state.activeSessionId && state.ctx) streamingCache[state.activeSessionId] = state.ctx;
        return {
          ...state,
          activeSessionId: action.id,
          ctx: action.id ? streamingCache[action.id] ?? null : null,
          ctxCache: streamingCache,
        };
      }
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      const ctxCache = { ...state.ctxCache };
      if (state.activeSessionId && state.ctx) ctxCache[state.activeSessionId] = state.ctx;
      return {
        ...state,
        activeSessionId: action.id,
        messages: action.id ? cache[action.id] ?? [] : [],
        approvals: [],
        preview: null,
        messageCache: cache,
        ctxCache,
        ctx: action.id ? ctxCache[action.id] ?? null : null,
      };
    }
    case "config":
      return { ...state, config: action.config };
    case "project_dir":
      return { ...state, projectDir: action.path };
    case "workspaces":
      return { ...state, workspaces: action.workspaces };
    case "permissions":
      return { ...state, permissions: action.permissions, permissionsError: null };
    case "permissions_error":
      return { ...state, permissionsError: action.error };
    case "settings_open":
      return { ...state, settingsOpen: action.open, settingsTab: action.tab ?? state.settingsTab };
    case "chat_font_size":
      return { ...state, chatFontSize: Math.min(24, Math.max(14, action.size)) };
    case "zoom":
      return { ...state, zoom: Math.round(Math.min(2, Math.max(0.5, action.zoom)) * 10) / 10 };
    case "zoom_delta":
      return {
        ...state,
        zoom: Math.round(Math.min(2, Math.max(0.5, state.zoom + action.delta)) * 10) / 10,
      };
    case "mode":
      return { ...state, mode: action.mode };
    case "preview":
      return { ...state, preview: action.preview };
    case "sidebar_collapsed":
      return { ...state, sidebarCollapsed: action.collapsed };
    case "cloud_bg":
      return { ...state, cloudBg: action.on };
    case "theme":
      return { ...state, theme: action.theme };
    case "resolved_dark":
      return state.resolvedDark === action.dark ? state : { ...state, resolvedDark: action.dark };
    case "new_chat": {
      // Park the outgoing page and clear the active id so the incoming
      // active_session action cannot re-park an empty page over it.
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      const ctxCache = { ...state.ctxCache };
      if (state.activeSessionId && state.ctx) ctxCache[state.activeSessionId] = state.ctx;
      return {
        ...state,
        activeSessionId: null,
        messages: [],
        approvals: [],
        preview: null,
        messageCache: cache,
        ctxCache,
        // Fresh chat = empty history; its own meter starts empty, and the
        // old session's meter stays parked in ctxCache for when the user
        // returns to that pill.
        ctx: null,
      };
    }
    case "workspace_switch": {
      // Folder change: the strip, pills and transcripts belong to the OLD
      // workspace — reset them so another folder's sessions can never
      // merge into this view (the "49 sessions" bug). messageCache stays:
      // session ids are globally unique, so cached pages survive going
      // back and forth. Same for the meters: the outgoing session's ctx
      // parks into ctxCache, the folder's view starts clean, and a
      // restored session brings its own meter back.
      const wsCtxCache = { ...state.ctxCache };
      if (state.activeSessionId && state.ctx) wsCtxCache[state.activeSessionId] = state.ctx;
      return {
        ...state,
        projectDir: action.path,
        sessions: [],
        sessionsError: null,
        sessionsSource: null,
        activeSessionId: null,
        messages: [],
        approvals: [],
        preview: null,
        ctxCache: wsCtxCache,
        ctx: null,
      };
    }
    case "sessions_cleared": {
      // "Delete all chats in this folder": drop every pill + transcript.
      const cache = { ...state.messageCache };
      for (const s of state.sessions) delete cache[s.id];
      const ctxCache = { ...state.ctxCache };
      for (const s of state.sessions) delete ctxCache[s.id];
      return {
        ...state,
        sessions: [],
        sessionsError: null,
        sessionsSource: null,
        activeSessionId: null,
        messages: [],
        approvals: [],
        preview: null,
        messageCache: cache,
        ctxCache,
        ctx: null,
      };
    }
  }
}

function patchAssistantStatus(state: AppState, status: ChatMessage["status"], error?: string): Partial<AppState> {
  const messages = [...state.messages];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      messages[i] = { ...messages[i], status, error };
      break;
    }
  }
  return { messages };
}

function patchAssistantDone(state: AppState, usage?: Usage): Partial<AppState> {
  const messages = [...state.messages];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      messages[i] = { ...messages[i], status: "done", usage };
      break;
    }
  }
  return { messages };
}

/** The app store hook: state + the side-effecting action helpers. */
export function useAppStore() {
  const [state, dispatch] = useReducer(reducer, initialState);

  // The workspace a fetch was issued FOR — stamped onto listSessions
  // replies so the reducer can drop ones that raced a folder switch.
  // A ref, not state: the fetch sites run in callbacks holding stale
  // closures.
  const projectDirRef = useRef<string | null>(state.projectDir);
  useEffect(() => {
    projectDirRef.current = state.projectDir;
  }, [state.projectDir]);

  // Engine event stream.
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    if (backend.isTauri()) {
      backend
        .onEngineEvent((raw) => {
          dispatch({ type: "event", event: parseEvent(raw) });
          // After a turn completes, refresh the session strip once — the
          // turn may have minted its session (first send in an empty
          // workspace) and the pill must appear without a stale-list race.
          const kind = typeof raw?.kind === "string" ? raw.kind : "";
          if (kind === "turn_finished" || kind === "done") {
            const ws = projectDirRef.current;
            backend.listSessions().then((r) =>
              dispatch({
                type: "sessions",
                sessions: r.sessions,
                projectDir: ws,
                error: r.error ?? null,
                source: r.source ?? null,
              })
            );
          }
        })
        .then((fn) => {
          if (cancelled) fn();
          else unlisten = fn;
        })
        .catch(() => {});
    }
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // Initial loads: config + desktop state immediately (cheap, no engine);
  // the session list and permission re-apply wait for engine_ready — they
  // are engine queries, and issuing them during boot eval just queues
  // behind the engine's init work (slow first paint for nothing).
  useEffect(() => {
    if (!backend.isTauri()) return;
    Promise.all([backend.getConfig(), backend.getDesktopState()])
      .then(([c, d]) => {
        if (c) dispatch({ type: "config", config: c });
        dispatch({ type: "project_dir", path: d?.project_dir ?? null });
      })
      .catch(() => {});
  }, []);

  // When the engine reports ready: load the workspace's sessions and
  // re-apply the persisted permission profile. The engine attached the
  // workspace's newest session during its own boot; the frontend only
  // tracked the folder name — activate that newest pill so a first send
  // appends to it instead of minting a duplicate "session-2" next to it.
  useEffect(() => {
    if (!state.engineReady) return;
    const ws = projectDirRef.current;
    backend.listSessions().then((r) => {
      dispatch({
        type: "sessions",
        sessions: r.sessions,
        projectDir: ws,
        error: r.error ?? null,
        source: r.source ?? null,
      });
      if (r.sessions.length) {
        dispatch({ type: "active_session", id: r.sessions[0].id });
        backend.resumeSession(r.sessions[0].id).catch(() => {});
      }
    });
    backend
      .setPermissions(storedPermissions())
      .then((r) => {
        // The engine's answer is the profile in force — show that, not the
        // one that was persisted.
        if (r?.ok && r.permissions) {
          dispatch({ type: "permissions", permissions: r.permissions as PermissionProfile });
          return;
        }
        // The engine did not take the persisted profile: the shell must
        // not keep showing a policy the runtime is not enforcing.
        dispatch({
          type: "permissions_error",
          error: `the engine refused the persisted permission profile${r?.error ? `: ${r.error}` : ""}`,
        });
      })
      .catch((e) => {
        dispatch({
          type: "permissions_error",
          error: `the engine did not confirm the permission profile: ${String(e)}`,
        });
      });
  }, [state.engineReady]);

  // Interface zoom (⌘+/⌘-/⌘0): CSS zoom on the root scales the whole UI,
  // persisted so relaunch keeps the size the user picked.
  useEffect(() => {
    document.documentElement.style.zoom = String(state.zoom);
    try {
      localStorage.setItem("sofuu.zoom", String(state.zoom));
    } catch {
      /* no localStorage */
    }
  }, [state.zoom]);

  // Workspaces + permission profile persist across launches.
  useEffect(() => {
    try {
      localStorage.setItem("sofuu.workspaces", JSON.stringify(state.workspaces));
    } catch {
      /* no localStorage */
    }
  }, [state.workspaces]);

  useEffect(() => {
    try {
      localStorage.setItem("sofuu.permissions", state.permissions);
    } catch {
      /* no localStorage */
    }
  }, [state.permissions]);

  // Cloud background preference persists across launches.
  useEffect(() => {
    try {
      localStorage.setItem("sofuu.cloudBg", state.cloudBg ? "on" : "off");
    } catch {
      /* no localStorage */
    }
  }, [state.cloudBg]);

  // Theme: resolve the preference against macOS, paint data-theme +
  // color-scheme on the root, and mirror it onto the NSWindow appearance
  // so the native vibrancy material and traffic lights follow (null =
  // follow the system). While on "system", re-resolve live when macOS
  // switches appearance.
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  useEffect(() => {
    try {
      localStorage.setItem("sofuu.theme", state.theme);
    } catch {
      /* no localStorage */
    }
  }, [state.theme]);

  useEffect(() => {
    const apply = () => {
      const dark = state.theme === "dark" || (state.theme === "system" && systemDark.matches);
      const root = document.documentElement;
      if (dark) root.dataset.theme = "dark";
      else delete root.dataset.theme;
      root.style.colorScheme = dark ? "dark" : "light";
      dispatch({ type: "resolved_dark", dark });
      if (backend.isTauri()) {
        getCurrentWindow()
          .setTheme(state.theme === "system" ? null : state.theme)
          .catch(() => {});
      }
    };
    apply();
    if (state.theme !== "system") return;
    systemDark.addEventListener("change", apply);
    return () => systemDark.removeEventListener("change", apply);
  }, [state.theme, systemDark]);

  // Slash commands (P2): the TUI's command surface, desktop edition. Parsed
  // BEFORE a turn is sent — an unknown command errors locally instead of
  // reaching the model as literal text. All engine calls ride the existing
  // backend API; output lands as a dim system row in the timeline.
  const systemRow = useCallback((text: string) => {
    dispatch({ type: "system_row", text });
  }, []);

  const cancel = useCallback(() => {
    backend.cancelTurn().catch(() => {});
  }, []);

  const approve = useCallback((id: string, allow: boolean, always = false) => {
    backend.resolveApproval(id, allow, always).catch(() => {});
    dispatch({ type: "event", event: { type: "approval_resolved", id } });
  }, []);

  const refreshSessions = useCallback(() => {
    const ws = projectDirRef.current;
    backend.listSessions().then((r) =>
      dispatch({
        type: "sessions",
        sessions: r.sessions,
        projectDir: ws,
        error: r.error ?? null,
      })
    );
  }, []);

  const newChat = useCallback(() => {
    // Session cap (MAX_SESSIONS per workspace): the topbar + is disabled
    // at the cap; this covers ⌘N / /new — and says WHY instead of a
    // silent no-op.
    if (state.sessions.length >= MAX_SESSIONS) {
      dispatch({
        type: "system_row",
        text: `Chat limit reached (${MAX_SESSIONS} per workspace) — delete a chat (the x on a pill) to add a new one.`,
      });
      return;
    }
    backend
      .newSession()
      .then((res) => {
        if (res?.sessionId) {
          const session = makeSessionInfo(res.sessionId, state.projectDir ?? "");
          dispatch({ type: "add_session", session });
          dispatch({ type: "active_session", id: res.sessionId });
        } else {
          dispatch({
            type: "system_row",
            text: `New chat failed${res?.error ? `: ${res.error}` : " — the engine returned no session id"}`,
          });
        }
      })
      .catch((e) => {
        dispatch({ type: "system_row", text: `New chat failed: ${String(e)}` });
      });
    dispatch({ type: "new_chat" });
  }, [state.projectDir, state.sessions]);

  const handleCommand = useCallback(
    async (raw: string) => {
      const m = /^\/(\w+)(?:\s+([\s\S]*))?$/.exec(raw.trim());
      const cmd = (m?.[1] ?? "help").toLowerCase();
      const arg = (m?.[2] ?? "").trim();
      // Derived from the shared registry — the "/" palette and /help can
      // never drift apart.
      const sigs = SLASH_COMMANDS.map(slashSignature);
      const sigW = Math.max(...sigs.map((s) => s.length)) + 2;
      const HELP = sigs
        .map((s, i) => s.padEnd(sigW) + "— " + SLASH_COMMANDS[i].desc)
        .join("\n");
      switch (cmd) {
        case "help":
        case "commands":
          systemRow(HELP);
          return;
        case "new":
        case "clear":
          newChat();
          return;
        case "compact": {
          systemRow("Compacting…");
          try {
            const r = await backend.compactSession();
            systemRow(r?.ok ? "Compacted." : "Compact failed" + (r?.error ? `: ${r.error}` : ""));
          } catch (e) {
            systemRow(`Compact failed: ${String(e)}`);
          }
          return;
        }
        case "ctx": {
          if (arg) {
            const n = Number(arg.replace(/[^0-9]/g, ""));
            if (Number.isFinite(n) && n >= 1000 && n <= 1000000) {
              try {
                // The backend answers with the PERSISTED config — report
                // what it now holds, not what was asked for.
                const c = (await backend.updateConfig({ ctx_window: n })) as Record<string, unknown>;
                const got = Number(c?.ctx_window);
                systemRow(
                  got === n
                    ? `Context window → ${n.toLocaleString()} tokens.`
                    : `Context window is ${got.toLocaleString()} tokens — the config did not take ${n.toLocaleString()}.`
                );
              } catch (e) {
                systemRow(`ctx: ${String(e)}`);
              }
            } else {
              systemRow("Usage: /ctx [tokens] (1000–1000000). Bare /ctx shows live usage.");
            }
            return;
          }
          const c = state.ctx;
          systemRow(
            c && c.window > 0
              ? `Context ${c.used.toLocaleString()} / ${c.window.toLocaleString()} tokens (${Math.round((c.used / c.window) * 100)}%) — ${c.source}${c.clampedConfig ? " (config clamped to the model's real limit)" : ""}`
              : "No context meter yet — send a message first."
          );
          return;
        }
        case "maxout": {
          if (!arg) {
            const c = state.config as unknown as Record<string, unknown> | null;
            const v = c?.["max_output"];
            systemRow(typeof v === "number" && v > 0 ? `Max output: ${v.toLocaleString()} tokens.` : "Max output: model default (set with /maxout <tokens>).");
            return;
          }
          const n = Number(arg.replace(/[^0-9]/g, ""));
          if (Number.isFinite(n) && n >= 256 && n <= 384000) {
            try {
              const c = (await backend.updateConfig({ max_output: n })) as Record<string, unknown>;
              const got = Number(c?.max_output);
              systemRow(
                got === n
                  ? `Max output → ${n.toLocaleString()} tokens.`
                  : `Max output is ${got.toLocaleString()} tokens — the config did not take ${n.toLocaleString()}.`
              );
            } catch (e) {
              systemRow(`maxout: ${String(e)}`);
            }
          } else {
            systemRow("Usage: /maxout <tokens> (256–384000).");
          }
          return;
        }
        case "model": {
          if (!arg) {
            const c = state.config;
            systemRow(c ? `${c.provider || "(no provider)"} · ${c.model || "(no model)"}` : "Config not loaded yet.");
            return;
          }
          try {
            const c = (await backend.updateConfig({ model: arg })) as Record<string, unknown>;
            const got = String(c?.model ?? "");
            systemRow(
              got === arg
                ? `Model → ${arg}.`
                : `Model is ${got || "(none)"} — the config did not take ${arg}.`
            );
          } catch (e) {
            systemRow(`model: ${String(e)}`);
          }
          return;
        }
        case "provider": {
          if (!arg) {
            const c = state.config;
            systemRow(c ? `Provider: ${c.provider || "(none)"} · model ${c.model || "(none)"}` : "Config not loaded yet.");
            return;
          }
          try {
            const c = (await backend.updateConfig({ provider: arg, active: arg })) as Record<string, unknown>;
            const got = String(c?.active ?? c?.provider ?? "");
            systemRow(
              got === arg
                ? `Provider → ${arg}. Pick a model in Settings → Models if turns fail.`
                : `Provider is ${got || "(none)"} — the config did not take ${arg}.`
            );
          } catch (e) {
            systemRow(`provider: ${String(e)}`);
          }
          return;
        }
        case "effort": {
          if (!arg) {
            const c = state.config as unknown as Record<string, unknown> | null;
            systemRow(`Effort: ${String(c?.["effort"] ?? "off")} (low|medium|high|max|off).`);
            return;
          }
          const lvl = arg.toLowerCase();
          if (!["low", "medium", "high", "max", "off"].includes(lvl)) {
            systemRow("Usage: /effort low|medium|high|max|off.");
            return;
          }
          try {
            const c = (await backend.updateConfig({ effort: lvl })) as Record<string, unknown>;
            const got = String(c?.effort ?? "");
            systemRow(
              got === lvl
                ? `Effort → ${lvl}.`
                : `Effort is ${got || "off"} — the config did not take ${lvl}.`
            );
          } catch (e) {
            systemRow(`effort: ${String(e)}`);
          }
          return;
        }
        case "brain": {
          if (!arg) {
            const c = state.config as unknown as Record<string, unknown> | null;
            systemRow(`Brain: ${c?.["brain"] ? "on" : "off"} — memories persist across sessions.`);
            return;
          }
          const on = /^(on|1|true|yes)$/i.test(arg);
          const off = /^(off|0|false|no)$/i.test(arg);
          if (!on && !off) {
            systemRow("Usage: /brain [on|off].");
            return;
          }
          try {
            const c = (await backend.updateConfig({ brain: on })) as Record<string, unknown>;
            const got = !!c?.brain;
            systemRow(
              got === on
                ? `Brain ${on ? "on — memories persist across sessions" : "off"}.`
                : `Brain is ${got ? "on" : "off"} — the config did not take the change.`
            );
          } catch (e) {
            systemRow(`brain: ${String(e)}`);
          }
          return;
        }
        case "remember": {
          if (!arg) {
            systemRow("Usage: /remember <fact> — pin a fact to the brain.");
            return;
          }
          try {
            const r = await backend.brainRemember(arg);
            systemRow(r?.ok ? "Remembered." : `Remember failed${r?.error ? `: ${r.error}` : ""}`);
          } catch (e) {
            systemRow(`remember: ${String(e)}`);
          }
          return;
        }
        case "why": {
          try {
            const r = await backend.brainWhy();
            if (!r?.ok) {
              systemRow(`why: ${r?.error ?? "unavailable"}`);
            } else if (!r.lastRecall && !r.recall) {
              systemRow("No recalled memories shaped the last answer.");
            } else {
              systemRow(`Last recall: ${JSON.stringify(r.lastRecall ?? r.recall)}`);
            }
          } catch (e) {
            systemRow(`why: ${String(e)}`);
          }
          return;
        }
        case "tools": {
          try {
            const r = await backend.listTools();
            if (!r?.ok) systemRow(`tools: ${r?.error ?? "unavailable"}`);
            else {
              const b = Array.isArray(r.builtin) ? r.builtin.join(", ") : "—";
              // An unreported MCP surface is "unknown", not an empty list.
              const mcp = r.mcp == null ? (r.mcpNote ?? "not reported by the engine") : JSON.stringify(r.mcp);
              systemRow(`Built-in: ${b}\nMCP: ${mcp}`);
            }
          } catch (e) {
            systemRow(`tools: ${String(e)}`);
          }
          return;
        }
        case "agents": {
          try {
            const r = await backend.listAgents();
            systemRow(r?.ok ? `Agents: ${JSON.stringify(r.agents ?? [])}` : `agents: ${r?.error ?? "unavailable"}`);
          } catch (e) {
            systemRow(`agents: ${String(e)}`);
          }
          return;
        }
        case "rlm": {
          if (!arg) {
            const c = state.config as unknown as Record<string, unknown> | null;
            systemRow(`RLM: ${String(c?.["rlm"] ?? "off")} (on|off|auto).`);
            return;
          }
          const v = arg.toLowerCase();
          if (!["on", "off", "auto"].includes(v)) {
            systemRow("Usage: /rlm on|off|auto.");
            return;
          }
          try {
            const c = (await backend.updateConfig({ rlm: v })) as Record<string, unknown>;
            const got = String(c?.rlm ?? "");
            systemRow(
              got === v
                ? `RLM → ${v}.`
                : `RLM is ${got || "off"} — the config did not take ${v}.`
            );
          } catch (e) {
            systemRow(`rlm: ${String(e)}`);
          }
          return;
        }
        case "ml": {
          const sub = arg.toLowerCase();
          if (!arg || sub === "info") {
            try {
              const r = await backend.mlInfo();
              systemRow(r?.ok ? `ML gates: ${JSON.stringify(r.info ?? {})}` : `ml: ${r?.error ?? "unavailable"}`);
            } catch (e) {
              systemRow(`ml: ${String(e)}`);
            }
            return;
          }
          if (sub === "on" || sub === "off") {
            try {
              const want = sub === "on";
              const c = (await backend.updateConfig({ ml: want })) as Record<string, unknown>;
              const got = !!c?.ml;
              systemRow(
                got === want
                  ? `ML gates ${sub}.`
                  : `ML gates are ${got ? "on" : "off"} — the config did not take the change.`
              );
            } catch (e) {
              systemRow(`ml: ${String(e)}`);
            }
            return;
          }
          systemRow("Usage: /ml [on|off|info] — learn/adopt/discard/reset stay CLI-only for now.");
          return;
        }
        case "sessions": {
          const r = await backend.listSessions();
          // A list served by the on-disk registry mirror is NOT the same
          // as an engine list: say which one, and why the engine was
          // bypassed.
          const srcNote = r.source ? ` (${r.source === "registry-file" ? "from the on-disk registry" : r.source}${r.error ? ` — the engine did not answer: ${r.error}` : ""})` : "";
          systemRow(
            r.error && !r.sessions.length
              ? `sessions unavailable: ${r.error}`
              : r.sessions.length
                ? r.sessions.map((s) => `${s.id} · ${s.model || "?"} · ${s.ended ? "ended" : "live"}`).join("\n") + srcNote
                : `No sessions in this workspace yet.${srcNote}`
          );
          return;
        }
        case "context": {
          try {
            const r = await backend.contextDump();
            systemRow(r?.ok ? `Context: ${JSON.stringify(r.state ?? {}).slice(0, 2000)}` : `context: ${r?.error ?? "unavailable"}`);
          } catch (e) {
            systemRow(`context: ${String(e)}`);
          }
          return;
        }
        // CLI-only surfaces: honest pointer instead of "unknown".
        case "watch":
        case "hooks":
        case "ghost":
        case "serve":
        case "outputs":
        case "verify":
        case "share":
        case "import":
        case "work":
        case "done":
        case "note":
        case "notify":
        case "sync":
        case "at":
          systemRow(`/${cmd} is CLI-only in this version — use the terminal chat for it. Parity tracked for a later release.`);
          return;
        case "cost": {
          try {
            const st = (await backend.chatState()) as Record<string, unknown>;
            // An unreported number is UNKNOWN, not zero: "0.0000 USD" reads
            // as a measured zero spend.
            const spend = typeof st.sessionSpendUsd === "number" ? `$${st.sessionSpendUsd.toFixed(4)}` : "unknown";
            const tokens = typeof st.sessionTokens === "number" ? `${st.sessionTokens.toLocaleString()} tokens` : "unknown";
            systemRow(`Session: ${spend} USD · ${tokens}`);
          } catch (e) {
            systemRow(`cost: ${String(e)}`);
          }
          return;
        }
        case "resume": {
          if (!arg) {
            const r = await backend.listSessions();
            systemRow(
              r.error
                ? `sessions unavailable: ${r.error}`
                : r.sessions.length
                  ? r.sessions.slice(0, 10).map((s) => `${s.id} · ${s.model || "?"} · ${s.ended ? "ended" : "live"}`).join("\n") +
                    "\n/resume <id> to open one"
                  : "No sessions in this workspace yet."
            );
            return;
          }
          try {
            const r = await backend.resumeSession(arg);
            systemRow(r?.ok ? `Resumed ${arg}.` : `Resume failed${r?.error ? `: ${r.error}` : ""}`);
          } catch (e) {
            systemRow(`Resume failed: ${String(e)}`);
          }
          return;
        }
        case "permissions": {
          if (arg) {
            try {
              const r = await backend.setPermissions(arg);
              systemRow(r?.ok ? `Permissions → ${r.permissions}` : `Failed${r?.error ? `: ${r.error}` : ""}`);
            } catch (e) {
              systemRow(`permissions: ${String(e)}`);
            }
          } else {
            systemRow(`Permissions: ${state.permissions} (full | edit | plan | prompt)`);
          }
          return;
        }
        default:
          systemRow(`Unknown command /${cmd} — /help lists what exists.`);
      }
    },
    [systemRow, newChat, state.ctx, state.config, state.permissions]
  );

  const submit = useCallback(
    async (text: string, images?: string[]) => {
      const trimmed = text.trim();
      if (!trimmed && !(images && images.length)) return;
      if (trimmed.startsWith("/")) {
        await handleCommand(trimmed);
        return;
      }
      dispatch({ type: "user_submitted", text: images && images.length && !trimmed ? "[image]" : trimmed });
      try {
        // Never mint on submit while pills exist (user rule: sessions come
        // from the + button, or the app's own first-session mint). If the
        // UI somehow has no active session, ADOPT the newest pill instead;
        // only a workspace with zero sessions mints here — the same
        // first-session case as folder open.
        let activeId = state.activeSessionId;
        if (!activeId && state.sessions.length > 0) {
          const newest = state.sessions[0];
          await backend.resumeSession(newest.id).catch(() => {});
          dispatch({ type: "active_session", id: newest.id });
          activeId = newest.id;
        } else if (!activeId) {
          const minted = await backend.newSession();
          if (minted?.sessionId) {
            activeId = minted.sessionId;
            dispatch({
              type: "add_session",
              session: makeSessionInfo(minted.sessionId, state.projectDir ?? ""),
            });
            dispatch({ type: "active_session", id: minted.sessionId });
          } else {
            dispatch({
              type: "system_row",
              text: `No session could be created${minted?.error ? `: ${minted.error}` : " — the engine returned no session id"}`,
            });
          }
        }
        // "Just chat" mode runs the turn without any tools attached.
        // Images (P2 multimodal) ride opts.images as data URLs.
        const opts = state.mode === "chat" ? { noTools: true } : {};
        if (images && images.length) Object.assign(opts, { images });
        await backend.sendTurn(trimmed, opts);
      } catch (err) {
        dispatch({
          type: "event",
          event: { type: "turn_error", message: String(err) },
        });
      }
    },
    [state.mode, handleCommand]
  );

  // Switch the engine to a known workspace and load its sessions. The
  // strip resets to empty IMMEDIATELY (optimistic) — the old folder's
  // pills must never show under the new one, and resetting first makes
  // any in-flight old-workspace fetch tag-drop on arrival.
  const openWorkspace = useCallback(
    (path: string) => {
      if (path === state.projectDir) {
        refreshSessions();
        return;
      }
      const prev = state.projectDir;
      projectDirRef.current = path;
      dispatch({ type: "workspace_switch", path });
      backend
        .setProjectDir(path)
        .then((res) => {
          // Folder rules (user): a switch NEVER creates a chat — except
          // the FIRST session of an empty workspace, which the app mints
          // itself so the strip is never empty. New sessions otherwise
          // come only from the + button. With pills present: activate what
          // the engine attached, or (attach failed) the newest pill and
          // point the engine at it — never park on null while pills exist.
          const attached: string | null = res?.sessionId ?? null;
          return backend.listSessions().then(async (r) => {
            dispatch({
              type: "sessions",
              sessions: r.sessions,
              projectDir: path,
              error: r.error ?? null,
              source: r.source ?? null,
            });
            if (attached && r.sessions.some((x) => x.id === attached)) {
              dispatch({ type: "active_session", id: attached });
              return;
            }
            if (!r.sessions.length) {
              const minted = await backend.newSession();
              if (minted?.sessionId) {
                dispatch({
                  type: "add_session",
                  session: makeSessionInfo(minted.sessionId, path),
                });
                dispatch({ type: "active_session", id: minted.sessionId });
              } else {
                dispatch({ type: "active_session", id: null });
              }
              return;
            }
            dispatch({ type: "active_session", id: r.sessions[0].id });
            backend.resumeSession(r.sessions[0].id).catch(() => {});
          });
        })
        .catch((e) => {
          // Engine refused the switch — revert so later fetch tags still
          // match the folder actually loaded, and say why.
          projectDirRef.current = prev;
          dispatch({ type: "workspace_switch", path: prev });
          dispatch({
            type: "system_row",
            text: `Workspace switch failed: ${String(e)}`,
          });
        });
    },
    [state.projectDir, refreshSessions]
  );

  // Delete a chat pill: engine detaches + drops the registry entry, the
  // Rust side removes the session folder. Active page parks to empty.
  // A refusal keeps the pill and says why — the chat still exists.
  const deleteSession = useCallback(
    (id: string) => {
      backend
        .deleteSession(id)
        .then((res) => {
          if (res?.ok === false) {
            dispatch({
              type: "system_row",
              text: `Delete failed${res?.error ? `: ${res.error}` : " — the engine refused"}`,
            });
            return;
          }
          dispatch({ type: "delete_session", id });
        })
        .catch((e) => {
          dispatch({ type: "system_row", text: `Delete failed: ${String(e)}` });
        });
    },
    []
  );

  const setActiveSession = useCallback((id: string) => {
    dispatch({ type: "active_session", id });
    // Point the engine at this session so the next submit appends to its
    // transcript instead of the newest session's.
    backend.resumeSession(id).catch(() => {});
  }, []);

  // Delete EVERY chat in the current workspace (history popover): the
  // engine empties the registry, Rust removes the session folders. The
  // active page parks to empty.
  const clearSessions = useCallback(() => {
    backend
      .clearSessions()
      .then((res) => {
        if (res?.ok === false) {
          dispatch({
            type: "system_row",
            text: `Clear failed${res?.error ? `: ${res.error}` : ""}`,
          });
          return;
        }
        dispatch({ type: "sessions_cleared" });
        const n = res?.cleared ?? 0;
        dispatch({
          type: "system_row",
          text: n
            ? `Deleted all ${n} chat${n === 1 ? "" : "s"} in this workspace.`
            : "No chats to delete in this workspace.",
        });
      })
      .catch((e) => {
        dispatch({ type: "system_row", text: `Clear failed: ${String(e)}` });
      });
  }, []);

  // Remove a folder from the sidebar list ONLY — the folder and its chats
  // stay on disk. Removing the active folder switches to the next
  // remaining one (if any).
  const removeWorkspace = useCallback(
    (path: string) => {
      const rest = state.workspaces.filter((p) => p !== path);
      dispatch({ type: "workspaces", workspaces: rest });
      if (path === state.projectDir && rest.length > 0) openWorkspace(rest[0]);
    },
    [state.workspaces, state.projectDir, openWorkspace]
  );

  const openSettings = useCallback(
    (open: boolean, tab?: string) => dispatch({ type: "settings_open", open, tab }),
    []
  );
  const setChatFontSize = useCallback((size: number) => dispatch({ type: "chat_font_size", size }), []);
  const setConfig = useCallback((config: SofuuConfig) => dispatch({ type: "config", config }), []);
  const setProjectDir = useCallback((path: string | null) => dispatch({ type: "project_dir", path }), []);

  const zoomBy = useCallback((delta: number) => dispatch({ type: "zoom_delta", delta }), []);
  const zoomReset = useCallback(() => dispatch({ type: "zoom", zoom: 1 }), []);
  const setZoom = useCallback((zoom: number) => dispatch({ type: "zoom", zoom }), []);

  const setMode = useCallback((mode: "agent" | "chat") => {
    dispatch({ type: "mode", mode });
    try {
      localStorage.setItem("sofuu.mode", mode);
    } catch {
      /* no localStorage */
    }
  }, []);

  // Register a folder in the sidebar (newest first, deduped).
  const addWorkspace = useCallback(
    (path: string) => {
      dispatch({
        type: "workspaces",
        workspaces: [path, ...state.workspaces.filter((p) => p !== path)],
      });
    },
    [state.workspaces]
  );

  // Tool permission profile: applied to the engine immediately and kept
  // locally so every launch starts with the same policy. The UI only
  // shows the profile the ENGINE accepted: a refused switch must not
  // leave the shell claiming a policy the runtime is not enforcing.
  const setPermissions = useCallback((profile: PermissionProfile) => {
    backend
      .setPermissions(profile)
      .then((r) => {
        if (r?.ok) {
          dispatch({ type: "permissions", permissions: (r.permissions as PermissionProfile) ?? profile });
          return;
        }
        dispatch({
          type: "permissions_error",
          error: `the engine refused the permission change${r?.error ? `: ${r.error}` : ""}`,
        });
        dispatch({
          type: "system_row",
          text: `Permissions unchanged${r?.error ? `: ${r.error}` : " — the engine refused the change"}`,
        });
      })
      .catch((e) => {
        dispatch({
          type: "permissions_error",
          error: `the engine did not confirm the permission profile: ${String(e)}`,
        });
        dispatch({
          type: "system_row",
          text: `Permissions unchanged: ${String(e)}`,
        });
      });
  }, []);

  // Open a past session as a read-only transcript (sidebar / ◷ history).
  // The .qtsq files are encrypted, so turns come through the engine.
  // A failed read is NOT an empty transcript: the preview says why.
  const openSession = useCallback(async (id: string, label: string) => {
    try {
      const turns = await backend.sessionTurns(id);
      if (!Array.isArray(turns)) {
        dispatch({
          type: "system_row",
          text: `Transcript unavailable for ${label} — the engine returned no turns.`,
        });
        return;
      }
      dispatch({ type: "preview", preview: { id, label, turns } });
    } catch (e) {
      dispatch({
        type: "system_row",
        text: `Transcript unavailable for ${label}: ${String(e)}`,
      });
    }
  }, []);
  const closePreview = useCallback(() => dispatch({ type: "preview", preview: null }), []);

  const toggleSidebar = useCallback(() => {
    dispatch({ type: "sidebar_collapsed", collapsed: !state.sidebarCollapsed });
    try {
      localStorage.setItem("sofuu.sidebar", state.sidebarCollapsed ? "shown" : "hidden");
    } catch {
      /* no localStorage */
    }
  }, [state.sidebarCollapsed]);

  const setCloudBg = useCallback((on: boolean) => dispatch({ type: "cloud_bg", on }), []);

  const setTheme = useCallback((theme: ThemePref) => dispatch({ type: "theme", theme }), []);

  return {
    state,
    submit,
    cancel,
    approve,
    refreshSessions,
    newChat,
    deleteSession,
    clearSessions,
    setActiveSession,
    openSettings,
    setChatFontSize,
    setConfig,
    setProjectDir,
    zoomBy,
    zoomReset,
    setZoom,
    setMode,
    addWorkspace,
    removeWorkspace,
    openWorkspace,
    setPermissions,
    setCloudBg,
    setTheme,
    openSession,
    closePreview,
    toggleSidebar,
  };
}

export type AppStore = ReturnType<typeof useAppStore>;
