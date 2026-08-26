// ToolTimeline.tsx — cards under an assistant message built from
// tool/tool_result/delegate/recall/plan/warn/rlm events (§4a card style).

import type { TimelineEntry } from "../lib/events";

function glyphOf(e: TimelineEntry): { g: string; cls: string } {
  switch (e.kind) {
    case "tool":
      return { g: "⚙", cls: "tool" };
    case "tool_result":
      return e.ok === false ? { g: "✕", cls: "err" } : { g: "✓", cls: "ok" };
    case "delegate":
      return { g: "⑂", cls: "tool" };
    case "recall":
      return { g: "✦", cls: "tool" };
    case "plan":
      return { g: "≡", cls: "tool" };
    case "warn":
      return { g: "△", cls: "err" };
    case "consolidated":
      return { g: "∴", cls: "tool" };
    default:
      return { g: "◦", cls: "tool" };
  }
}

export function ToolTimeline({ entries }: { entries: TimelineEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="timeline">
      {entries.map((e) => {
        const gl = glyphOf(e);
        return (
          <div className="tl-entry" key={e.id}>
            <div className="tl-head">
              <span className={`tl-glyph ${gl.cls}`}>{gl.g}</span>
              <span className="tl-title">{e.title}</span>
            </div>
            {e.detail && <div className="tl-detail">{e.detail}</div>}
          </div>
        );
      })}
    </div>
  );
}
