// lib/store.ts — app state: folds the engine event stream into UI state.

import { useCallback, useEffect, useReducer } from "react";
import * as backend from "./backend";
import {
  parseEvent,
  type Approval,
  type ChatMessage,
  type IncomingEvent,
  type SessionInfo,
  type SofuuConfig,
  type StepEvent,
  type TimelineEntry,
  type Usage,
} from "./events";

export interface AppState {
  engineReady: boolean;
  bootError: string | null;
  sessions: SessionInfo[];
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
  settingsOpen: boolean;
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
  activeSessionId: null,
  messages: [],
  messageCache: {},
  streaming: false,
  approvals: [],
  config: null,
  projectDir: null,
  workspaces: storedWorkspaces(),
  permissions: storedPermissions(),
  settingsOpen: false,
  chatFontSize: 18,
  zoom: storedZoom(),
  mode: storedMode(),
  preview: null,
  sidebarCollapsed: storedSidebarCollapsed(),
};

type Action =
  | { type: "event"; event: IncomingEvent }
  | { type: "user_submitted"; text: string }
  | { type: "sessions"; sessions: SessionInfo[] }
  | { type: "add_session"; session: SessionInfo }
  | { type: "active_session"; id: string | null }
  | { type: "config"; config: SofuuConfig }
  | { type: "project_dir"; path: string | null }
  | { type: "workspaces"; workspaces: string[] }
  | { type: "permissions"; permissions: PermissionProfile }
  | { type: "settings_open"; open: boolean }
  | { type: "chat_font_size"; size: number }
  | { type: "zoom"; zoom: number }
  | { type: "zoom_delta"; delta: number }
  | { type: "mode"; mode: "agent" | "chat" }
  | { type: "preview"; preview: SessionPreview | null }
  | { type: "sidebar_collapsed"; collapsed: boolean }
  | { type: "new_chat" };

let seq = 0;
function nextId(prefix: string): string {
  seq += 1;
  return `${prefix}-${Date.now().toString(36)}-${seq}`;
}

/** Optimistic pill row for a session the engine just created. */
function makeSessionInfo(id: string, cwd: string, model: string, provider: string): SessionInfo {
  const now = Math.floor(Date.now() / 1000);
  return {
    id,
    pid: 0,
    host: "local",
    cwd,
    model,
    provider,
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

function stepToTimeline(step: StepEvent): TimelineEntry | null {
  const p = step.payload ?? {};
  switch (step.kind) {
    case "tool":
      return {
        id: nextId("tl"),
        kind: "tool",
        title: String(p.tool ?? p.name ?? "tool"),
        detail: typeof p.summary === "string" ? p.summary : undefined,
        t: step.t,
      };
    case "tool_result":
      return {
        id: nextId("tl"),
        kind: "tool_result",
        title: String(p.tool ?? p.name ?? "result"),
        detail: typeof p.summary === "string" ? p.summary : typeof p.text === "string" ? p.text.slice(0, 400) : undefined,
        ok: p.error == null,
        t: step.t,
      };
    case "delegate":
      return { id: nextId("tl"), kind: "delegate", title: String(p.agent ?? "sub-agent"), detail: typeof p.task === "string" ? p.task : undefined, t: step.t };
    case "recall":
      return { id: nextId("tl"), kind: "recall", title: "Memory recall", detail: typeof p.summary === "string" ? p.summary : undefined, t: step.t };
    case "plan":
      return { id: nextId("tl"), kind: "plan", title: "Plan", detail: typeof p.text === "string" ? p.text : undefined, t: step.t };
    case "warn":
      return { id: nextId("tl"), kind: "warn", title: "Warning", detail: typeof p.message === "string" ? p.message : undefined, t: step.t };
    case "consolidated":
      return { id: nextId("tl"), kind: "consolidated", title: "Context consolidated", t: step.t };
    default:
      if (step.kind.startsWith("rlm")) {
        return { id: nextId("tl"), kind: "rlm", title: "RLM", detail: typeof p.summary === "string" ? p.summary : undefined, t: step.t };
      }
      return null;
  }
}

function applyStep(state: AppState, step: StepEvent): AppState {
  const assistant = currentAssistant(state.messages);
  if (!assistant) return state;

  // answer_delta: append raw text to the streaming answer.
  if (step.kind === "answer_delta") {
    const text = String(step.payload?.text ?? "");
    if (!text) return state;
    return patchAssistant(state, (m) => ({ ...m, content: m.content + text }));
  }
  // think: accumulate reasoning text (collapsible in the UI).
  if (step.kind === "think") {
    const text = String(step.payload?.text ?? "");
    if (!text) return state;
    return patchAssistant(state, (m) => ({ ...m, thinking: (m.thinking ?? "") + text }));
  }

  const entry = stepToTimeline(step);
  if (!entry) return state;
  return patchAssistant(state, (m) => ({ ...m, timeline: [...m.timeline, entry] }));
}

function patchAssistant(state: AppState, fn: (m: ChatMessage) => ChatMessage): AppState {
  const messages = [...state.messages];
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      messages[i] = fn(messages[i]);
      break;
    }
  }
  return { ...state, messages };
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
          return { ...state, streaming: false };
        case "turn_error":
          return {
            ...state,
            streaming: false,
            ...patchAssistantStatus(state, "error", e.message),
          };
        case "approval_request":
          return { ...state, approvals: [...state.approvals, e.approval] };
        case "approval_resolved":
          return { ...state, approvals: state.approvals.filter((a) => a.id !== e.id) };
        case "done":
          return {
            ...state,
            streaming: false,
            ...patchAssistantDone(state, e.usage),
          };
        case "step":
          return applyStep(state, e.step);
        case "unknown":
          return state;
      }
      return state;
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
      };
    }
    case "sessions": {
      // Keep the active session when it is still in the new list; otherwise
      // (workspace switch, engine reset) park the outgoing page and restore
      // the new active session's transcript from the cache.
      const nextActive =
        state.activeSessionId && action.sessions.some((s) => s.id === state.activeSessionId)
          ? state.activeSessionId
          : action.sessions[0]?.id ?? null;
      if (nextActive === state.activeSessionId) return { ...state, sessions: action.sessions };
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      return {
        ...state,
        sessions: action.sessions,
        activeSessionId: nextActive,
        messages: nextActive ? cache[nextActive] ?? [] : [],
        approvals: [],
        preview: null,
        messageCache: cache,
      };
    }
    case "add_session":
      // Newest first — the engine's list_sessions sorts by last_seen desc.
      return { ...state, sessions: [action.session, ...state.sessions] };
    case "active_session": {
      // Park the old page's transcript, restore the new one (empty if the
      // session never had a transcript in this window).
      if (action.id === state.activeSessionId) return state;
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      return {
        ...state,
        activeSessionId: action.id,
        messages: action.id ? cache[action.id] ?? [] : [],
        approvals: [],
        preview: null,
        messageCache: cache,
      };
    }
    case "config":
      return { ...state, config: action.config };
    case "project_dir":
      return { ...state, projectDir: action.path };
    case "workspaces":
      return { ...state, workspaces: action.workspaces };
    case "permissions":
      return { ...state, permissions: action.permissions };
    case "settings_open":
      return { ...state, settingsOpen: action.open };
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
    case "new_chat": {
      // Park the outgoing page and clear the active id so the incoming
      // active_session action cannot re-park an empty page over it.
      const cache = { ...state.messageCache };
      if (state.activeSessionId) cache[state.activeSessionId] = state.messages;
      return {
        ...state,
        activeSessionId: null,
        messages: [],
        approvals: [],
        preview: null,
        messageCache: cache,
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

  // Engine event stream.
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    if (backend.isTauri()) {
      backend
        .onEngineEvent((raw) => dispatch({ type: "event", event: parseEvent(raw) }))
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

  // Initial loads: config, desktop state, sessions.
  useEffect(() => {
    if (!backend.isTauri()) return;
    backend.getConfig().then((c) => dispatch({ type: "config", config: c })).catch(() => {});
    backend
      .getDesktopState()
      .then((d) => dispatch({ type: "project_dir", path: d.project_dir ?? null }))
      .catch(() => {});
    backend.listSessions().then((s) => dispatch({ type: "sessions", sessions: s })).catch(() => {});
    // The engine always boots at the 'prompt' (ask) gate; re-apply the
    // persisted profile so the running policy matches the pill.
    backend.setPermissions(storedPermissions()).catch(() => {});
  }, []);

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

  const submit = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      dispatch({ type: "user_submitted", text: trimmed });
      try {
        // "Just chat" mode runs the turn without any tools attached.
        await backend.sendTurn(trimmed, state.mode === "chat" ? { noTools: true } : {});
      } catch (err) {
        dispatch({
          type: "event",
          event: { type: "turn_error", message: String(err) },
        });
      }
    },
    [state.mode]
  );

  const cancel = useCallback(() => {
    backend.cancelTurn().catch(() => {});
  }, []);

  const approve = useCallback((id: string, allow: boolean, always = false) => {
    backend.resolveApproval(id, allow, always).catch(() => {});
    dispatch({ type: "event", event: { type: "approval_resolved", id } });
  }, []);

  const refreshSessions = useCallback(() => {
    backend.listSessions().then((s) => dispatch({ type: "sessions", sessions: s })).catch(() => {});
  }, []);

  const newChat = useCallback(() => {
    backend
      .newSession()
      .then((res) => {
        if (res?.sessionId) {
          const session = makeSessionInfo(
            res.sessionId,
            state.projectDir ?? "",
            state.config?.model ?? "",
            state.config?.provider ?? ""
          );
          dispatch({ type: "add_session", session });
          dispatch({ type: "active_session", id: res.sessionId });
        }
      })
      .catch(() => {});
    dispatch({ type: "new_chat" });
  }, [state.projectDir, state.config?.model, state.config?.provider]);

  // Switch the engine to a known workspace and load its sessions. The
  // engine mints a fresh working session on setProject, so mirror its
  // pill as the active one — otherwise a typed turn would land in a
  // session the strip does not show.
  const openWorkspace = useCallback(
    (path: string) => {
      if (path === state.projectDir) {
        refreshSessions();
        return;
      }
      backend
        .setProjectDir(path)
        .then((res) => {
          if (res.ok === false || !res.sessionId) return;
          const sessionId: string = res.sessionId;
          dispatch({ type: "project_dir", path });
          return backend.listSessions().then((s) => {
            dispatch({ type: "sessions", sessions: s });
            if (s.some((x) => x.id === sessionId)) {
              dispatch({ type: "active_session", id: sessionId });
            } else {
              // Engine session not on disk yet (write failed?) — keep the
              // pill in the UI anyway so the page matches the engine.
              dispatch({
                type: "add_session",
                session: makeSessionInfo(
                  sessionId,
                  path,
                  state.config?.model ?? "",
                  state.config?.provider ?? ""
                ),
              });
              dispatch({ type: "active_session", id: sessionId });
            }
          });
        })
        .catch(() => {});
    },
    [state.projectDir, state.config?.model, state.config?.provider, refreshSessions]
  );

  const setActiveSession = useCallback((id: string) => {
    dispatch({ type: "active_session", id });
    // Point the engine at this session so the next submit appends to its
    // transcript instead of the newest session's.
    backend.resumeSession(id).catch(() => {});
  }, []);

  const openSettings = useCallback((open: boolean) => dispatch({ type: "settings_open", open }), []);
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
  // locally so every launch starts with the same policy.
  const setPermissions = useCallback((profile: PermissionProfile) => {
    dispatch({ type: "permissions", permissions: profile });
    backend.setPermissions(profile).catch(() => {});
  }, []);

  // Open a past session as a read-only transcript (sidebar / ◷ history).
  // The .qtsq files are encrypted, so turns come through the engine.
  const openSession = useCallback(async (id: string, label: string) => {
    try {
      const turns = await backend.sessionTurns(id);
      dispatch({
        type: "preview",
        preview: { id, label, turns: Array.isArray(turns) ? turns : [] },
      });
    } catch (e) {
      console.error(e);
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

  return {
    state,
    submit,
    cancel,
    approve,
    refreshSessions,
    newChat,
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
    openWorkspace,
    setPermissions,
    openSession,
    closePreview,
    toggleSidebar,
  };
}

export type AppStore = ReturnType<typeof useAppStore>;
