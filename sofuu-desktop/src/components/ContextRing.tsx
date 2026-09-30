// ContextRing.tsx — the live context-window meter ring.
//
// Sits right of the context-layers button in the composer bar. An SVG
// circle that fills clockwise with how much of the model's real context
// window the session is using, in real time: the engine emits a 'ctx'
// event at boot and after every turn (first-request calibrated — the
// number is what the model actually received, not an estimate).
//
// The fill is a stroke-dashoffset animation, so transitions between
// turns tween smoothly instead of snapping. Color steps at 70% and 90%
// mirror the alloc gate's compaction thresholds (compactAt slack 0.70 →
// tight 0.50): blue while roomy, amber as pressure builds, red when the
// next turn is at risk.
//
// Hovering the ring opens a breakdown card: where the first request's
// tokens went (system prompt / meta context / messages / tool schemas,
// split built-in vs MCP), the cache-hit rate when the provider reported
// one, and the honesty notes (which model, where the window evidence
// came from, config-override clamps). Sections are estimates measured
// engine-side with the same estimator the fit guard uses; the total in
// the header is the real promptTokens.

import { useState } from "react";
import type { CtxMeter, CtxSections } from "../lib/events";

interface Props {
  ctx: CtxMeter | null;
}

const SIZE = 22;      // ring box (matches the 20px icon buttons + padding)
const STROKE = 2.5;
const R = (SIZE - STROKE) / 2;
const C = 2 * Math.PI * R;

function fmt(n: number): string {
  if (n >= 1e6) return `${Math.round(n / 1e5) / 10}M`;
  if (n >= 1000) {
    const k = n / 1000;
    return `${k >= 100 ? Math.round(k) : Math.round(k * 10) / 10}k`;
  }
  return String(Math.round(n));
}

/** One decimal, trailing ".0" stripped ("48.2%", "39%"). */
function pct1(x: number): string {
  return `${Math.round(x * 1000) / 10}%`;
}

const SECTIONS: Array<{ key: keyof CtxSections; label: string; color: string }> = [
  { key: "messages", label: "Messages", color: "#7c2d8a" },
  { key: "systemTools", label: "System tools", color: "#a44f9e" },
  { key: "mcpTools", label: "MCP tools", color: "#c07ab3" },
  { key: "system", label: "System prompt", color: "#d9a6c9" },
  { key: "meta", label: "Meta context", color: "#9aa7b8" },
];

export function ContextRing({ ctx }: Props) {
  const [hover, setHover] = useState(false);

  // No report yet (engine not booted / no window resolved): a flat empty
  // ring — never a made-up number.
  const used = ctx?.used ?? 0;
  const window = ctx?.window ?? 0;
  const pct = window > 0 ? Math.min(1, Math.max(0, used / window)) : 0;
  const cls = pct >= 0.9 ? " danger" : pct >= 0.7 ? " warn" : "";

  // Honest footnote: the model the window was resolved for, where the
  // window evidence came from, and when a config override was clamped.
  const src = ctx?.source || "";
  const srcText =
    src === "learned" ? "window learned from a provider limit"
    : src === "discovered" ? "window from the endpoint's model listing"
    : src === "registry" ? "window from the registry"
    : src === "config" ? "window from config"
    : src === "default" ? "conservative default window (model unknown)"
    : "";
  const notes: string[] = [];
  if (ctx?.model) notes.push(ctx.model);
  if (srcText) notes.push(srcText);
  if (ctx?.clampedConfig) notes.push("config override clamped to the model's real limit");

  const rows = ctx?.sections
    ? SECTIONS
        .map((s) => ({ ...s, tk: Math.max(0, ctx.sections?.[s.key] ?? 0) }))
        .filter((r) => r.tk > 0)
        .sort((a, b) => b.tk - a.tk)
    : [];
  const sum = rows.reduce((acc, r) => acc + r.tk, 0);

  return (
    <span
      className={"ctx-ring-wrap" + cls}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <span className={"ctx-ring" + cls} role="img" aria-label="Context window usage">
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
          {/* track */}
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            fill="none"
            strokeWidth={STROKE}
            className="ctx-ring-track"
          />
          {/* fill — rotates -90° so it starts at 12 o'clock, sweeps clockwise */}
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            fill="none"
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={C * (1 - pct)}
            className="ctx-ring-fill"
            transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
          />
        </svg>
      </span>

      {hover && (
        <div className="ctx-pop" role="tooltip">
          <div className="ctx-pop-head">
            <span className="ctx-pop-title">Context window</span>
            <span className="ctx-pop-nums">
              {window > 0 ? `${fmt(used)} / ${fmt(window)} (${pct1(pct)})` : "unknown"}
            </span>
          </div>

          {rows.length > 0 && sum > 0 ? (
            <>
              <div className="ctx-pop-bar">
                {rows.map((r) => (
                  <span
                    key={r.key}
                    style={{ flexGrow: r.tk / sum, flexBasis: 0, background: r.color }}
                  />
                ))}
              </div>
              {rows.map((r) => (
                <div key={r.key} className="ctx-pop-row">
                  <span className="ctx-dot" style={{ background: r.color }} />
                  <span className="ctx-label">{r.label}</span>
                  <span className="ctx-pct">{pct1(r.tk / sum)}</span>
                </div>
              ))}
            </>
          ) : (
            <div className="ctx-pop-empty">
              {window > 0
                ? "Send a message — the breakdown appears after the first turn."
                : "No context window resolved yet."}
            </div>
          )}

          {ctx?.cacheHit != null && ctx.cacheHit > 0 && (
            <div className="ctx-pop-foot">
              <span>Cache hit (this turn)</span>
              <span>{Math.round(ctx.cacheHit * 100)}%</span>
            </div>
          )}

          {notes.length > 0 && <div className="ctx-pop-note">{notes.join(" · ")}</div>}
        </div>
      )}
    </span>
  );
}
