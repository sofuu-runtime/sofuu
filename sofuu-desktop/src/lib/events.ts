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
    | "compact"
    | "rlm"
    | "allocgate"
    | "mlgate"
    | "todo";
  title: string;
  detail?: string;
  /** Small suffix on the row itself (e.g. "· 15 memories") — keeps the
   *  entry one line instead of nesting a child row for its content. */
  badge?: string;
  /** Result text merged in from the tool_result step (one row per
   *  call: args + result expand under the same row). */
  result?: string;
  /** tool_result success flag, when known */
  ok?: boolean;
  /** Original engine tool name ("read_file"), for icon + run grouping —
   *  the title is the human row ("Read AGENTS.md"). */
  tool?: string;
  /** Agent run that produced this entry (agent.js runId) — consecutive
   *  think chunks only merge when they come from the same run. */
  runId?: string;
  /** Elapsed ms of a think segment (last chunk t − first chunk t; step
   *  t is ms since run start) — the row renders "Thought for Xs". */
  durationMs?: number;
  /** Compact rows only: true while the summarizer is folding the
   *  history (the row sweeps "Compressing"); false once settled. */
  live?: boolean;
  t?: number;
}

export interface Usage {
  promptTokens?: number;
  completionTokens?: number;
  costUsd?: number;
  steps?: number;
  /** The model that actually served the turn (truth after failover). */
  servedBy?: string;
}

/** A chat message rendered in the pane. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
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

/** Pill cap: at most this many sessions per workspace (the + disables at the cap). */
export const MAX_SESSIONS = 7;

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
  /** Per-session spend cap in USD — the engine refuses new turns past it. */
  budget_usd?: number;
  /** Brain embeddings + recall tuning (Settings → General → Memory). */
  embed_provider?: string;
  embed_model?: string;
  recall_min?: number;
  recall_budget?: number;
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
  | { type: "ctx"; ctx: CtxMeter }
  | { type: "done"; usage?: Usage }
  | { type: "unknown"; raw: Record<string, unknown> };

/** First-request section weights behind the ctx ring (token estimates —
 * the engine measures the wire request with the same estimator its fit
 * guard uses; the true total stays anchored to the provider's usage). */
export interface CtxSections {
  system: number;
  /** Meta context: date line, recall/notice blocks, shared layers. */
  meta: number;
  /** Everything conversational: history, task, tool transcripts. */
  messages: number;
  /** Built-in tool schemas (incl. delegate). */
  systemTools: number;
  /** MCP-served tool schemas. */
  mcpTools: number;
}

/** Live context-window meter (the composer ring): used tokens vs the
 * resolved window. `used` comes from the first LLM request's prompt (the
 * true context size) once a turn has run; the window from the same
 * evidence ladder the caps resolve uses. */
export interface CtxMeter {
  used: number;
  window: number;
  source: string;
  /** True when a config override (flat ctx_window/max_output) was clamped
   * down by real per-model evidence — the ring tooltip says why. */
  clampedConfig?: boolean;
  /** The model the window was resolved for (tooltip). */
  model?: string;
  /** First-request section breakdown (hover card). Absent until the
   * session's first LLM request has run. */
  sections?: CtxSections;
  /** cacheReadTokens / promptTokens of the latest turn's first request,
   * when the provider reported cache reads. */
  cacheHit?: number;
  /** Owning session id — the store keys meters per session, so an event
   * from a backgrounded session can't land on the visible one. */
  sid?: string;
}

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
    case "ctx": {
      const rs = payload.sections as Partial<CtxSections> | undefined;
      const sections: CtxSections | undefined = rs
        ? {
            system: Math.max(0, Number(rs.system ?? 0)) || 0,
            meta: Math.max(0, Number(rs.meta ?? 0)) || 0,
            messages: Math.max(0, Number(rs.messages ?? 0)) || 0,
            systemTools: Math.max(0, Number(rs.systemTools ?? 0)) || 0,
            mcpTools: Math.max(0, Number(rs.mcpTools ?? 0)) || 0,
          }
        : undefined;
      const hit = Number(payload.cacheHit);
      return {
        type: "ctx",
        ctx: {
          used: Math.max(0, Number(payload.used ?? 0)),
          window: Math.max(0, Number(payload.window ?? 0)),
          source: String(payload.source ?? ""),
          clampedConfig: payload.clampedConfig === true,
          model: payload.model ? String(payload.model) : undefined,
          sections,
          cacheHit: isFinite(hit) && hit > 0 ? Math.min(1, hit) : undefined,
          sid: payload.sid ? String(payload.sid) : undefined,
        },
      };
    }
    case "done":
    case "answer": {
      const usage = (payload.usage as Usage | undefined) ?? {};
      if (typeof payload.servedBy === "string" && payload.servedBy) {
        usage.servedBy = payload.servedBy;
      }
      return { type: "done", usage };
    }
    default:
      // Everything else is an agent step the timeline/answer renderer eats.
      if (kind) {
        return { type: "step", step: raw as unknown as StepEvent };
      }
      return { type: "unknown", raw };
  }
}
