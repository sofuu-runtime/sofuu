// lib/events.ts — the event protocol (PLAN-DESKTOP E).
//
// Mirrors the JSON envelopes the engine emits on "sofuu://event":
// agent.js onStep events ({runId, name, depth, t, kind, payload}) plus the
// chat-level lifecycle events chat.js and the Tauri backend add. Keep this
// file and the Rust/JS emitters in lockstep.

/** One streaming step from the agent loop (agent.js onStep envelope). */
export interface StepEvent {
  kind: string;
  payload?: Record<string, unknown>;
  runId?: string;
  name?: string;
  depth?: number;
  t?: number;
}

/** A tool call waiting for the user's approval. */
export interface Approval {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  /** Read-only tools never ask; these are the write/exec ones. */
  risky: boolean;
}

/** One entry in the tool timeline under an assistant message. */
export interface TimelineEntry {
  id: string;
  kind:
    | "tool"
    | "tool_result"
    | "delegate"
    | "recall"
    | "plan"
    | "think"
    | "warn"
    | "consolidated"
    | "rlm";
  title: string;
  detail?: string;
  /** tool_result success flag, when known */
  ok?: boolean;
  t?: number;
}

export interface Usage {
  promptTokens?: number;
  completionTokens?: number;
  costUsd?: number;
  steps?: number;
}

/** A chat message rendered in the pane. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  timeline: TimelineEntry[];
  usage?: Usage;
  status: "streaming" | "done" | "error";
  error?: string;
}

/** Session row from the registry mirror (sessions.rs). */
export interface SessionInfo {
  id: string;
  pid: number;
  host: string;
  cwd: string;
  model: string;
  provider: string;
  started_at: number;
  last_seen: number;
  task: string | null;
  ended: boolean;
}

/** The shared ~/.sofuu/config.json, api keys redacted by the backend. */
export interface SofuuConfig {
  provider?: string;
  model?: string;
  effort?: string;
  brain?: boolean;
  ml?: boolean;
  sync?: boolean;
  rlm?: string;
  ctx_window?: number;
  max_output?: number;
  has_api_key?: boolean;
  providers?: Array<{
    name: string;
    endpoint: string;
    model: string;
    profile: string;
    has_api_key?: boolean;
  }>;
  active?: string;
  [key: string]: unknown;
}

export interface DesktopState {
  project_dir?: string;
}

/** Discriminate the incoming event stream. */
export type IncomingEvent =
  | { type: "engine_ready" }
  | { type: "boot_error"; message: string }
  | { type: "turn_started" }
  | { type: "turn_finished" }
  | { type: "turn_error"; message: string }
  | { type: "approval_request"; approval: Approval }
  | { type: "approval_resolved"; id: string }
  | { type: "step"; step: StepEvent }
  | { type: "done"; usage?: Usage }
  | { type: "unknown"; raw: Record<string, unknown> };

/** Normalize a raw payload into the discriminated union. */
export function parseEvent(raw: Record<string, unknown>): IncomingEvent {
  const kind = typeof raw.kind === "string" ? raw.kind : "";
  const payload = (raw.payload ?? {}) as Record<string, unknown>;
  switch (kind) {
    case "engine_ready":
      return { type: "engine_ready" };
    case "boot_error":
      return { type: "boot_error", message: String(payload.message ?? "boot failed") };
    case "turn_started":
    case "start":
      return { type: "turn_started" };
    case "turn_finished":
      return { type: "turn_finished" };
    case "turn_error":
      return { type: "turn_error", message: String(payload.message ?? "turn failed") };
    case "approval_request":
      return {
        type: "approval_request",
        approval: {
          id: String(payload.id ?? ""),
          tool: String(payload.tool ?? "tool"),
          args: (payload.args ?? {}) as Record<string, unknown>,
          risky: payload.risky !== false,
        },
      };
    case "approval_resolved":
      return { type: "approval_resolved", id: String(payload.id ?? "") };
    case "done":
    case "answer":
      return { type: "done", usage: payload.usage as Usage | undefined };
    default:
      // Everything else is an agent step the timeline/answer renderer eats.
      if (kind) {
        return { type: "step", step: raw as unknown as StepEvent };
      }
      return { type: "unknown", raw };
  }
}
