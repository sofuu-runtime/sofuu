// ToolTimeline.tsx — the assistant's activity area, one line per real
// action: verb-first rows ("Read AGENTS.md", "Ran git status"), mixed
// tool runs collapsed into a single summary row ("Searched 2 patterns,
// read 4 files ✓ › 6 tools"), per-activity icons, "Thought for 1.2s"
// reasoning rows. Nothing repeats: results merge into their call row,
// single-call runs render as the leaf itself.
//
// Icons render the user-supplied Tabler outline SVGs (icons.tsx), keyed
// by icon id at <Icon> below — one point. Compress carries no icon by
// design: its live row animates the label instead (CompactEntry).

import { useEffect, useState } from "react";
import type { TimelineEntry } from "../lib/events";
import {
  composeSummary,
  formatDur,
  iconIdOf,
  isMonoIcon,
  type IconKey,
} from "../lib/timeline";
import { ICONS } from "./icons";

export function Icon({ id }: { id: IconKey }) {
  return (
    <span className="tl-icon" data-icon={id} aria-hidden>
      {ICONS[id]}
    </span>
  );
}

/** The phase label with the TUI's animation: one dim character sweeps
 *  left→right over the label every 120ms and loops (chat.rs tickPhase).
 *  Lives here so the timeline's live rows can sweep too (CompactEntry);
 *  ChatPane imports it for the phase line. */
export function PhaseSweep({ label }: { label: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const t = setInterval(() => setFrame((f) => (f + 1) % label.length), 120);
    return () => clearInterval(t);
  }, [label]);
  return (
    <span className="phase-sweep">
      {label.split("").map((ch, i) => (
        <span key={i} className={i === frame ? "dim" : undefined}>
          {ch}
        </span>
      ))}
    </span>
  );
}

const PER_ROW = 3;

/** Status mark on a row: ✓ done, ✕ failed. */
function Status({ e }: { e: TimelineEntry }) {
  if (e.ok === true) return <span className="tl-status tl-pop">✓</span>;
  if (e.ok === false) return <span className="tl-status err tl-pop">✕</span>;
  return null;
}

/** The expandable body under a leaf row: detail (args/context), then
 *  the merged result text. */
function EntryDetail({ e }: { e: TimelineEntry }) {
  const parts: string[] = [];
  if (e.detail) parts.push(e.detail);
  if (e.result) parts.push(`→ ${e.result}`);
  const text = parts.join("\n");
  if (!text) return null;
  return <div className="tl-detail">{text}</div>;
}

/** One leaf row: icon + title + status + count + chevron (reference
 *  order). Click expands detail; rows with nothing to show don't. */
function Entry({ e }: { e: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  const icon = iconIdOf(e);
  const expandable = Boolean(e.detail || e.result);
  return (
    <div className="tl-entry">
      <button
        className={"tl-row" + (open ? " open" : "")}
        onClick={expandable ? () => setOpen((o) => !o) : undefined}
        style={expandable ? undefined : { cursor: "default" }}
      >
        <Icon id={icon} />
        <span className={"tl-cell-title" + (isMonoIcon(icon) ? " tl-mono" : "")}>
          {e.title}
        </span>
        <Status e={e} />
        {expandable && <span className="tl-chevron">›</span>}
        {e.badge && <span className="tl-badge">{e.badge}</span>}
      </button>
      {open && <EntryDetail e={e} />}
    </div>
  );
}

/** A run of consecutive tool calls → ONE row: "Searched 2 patterns, read
 *  4 files ✓ › · 6 tools". Opening it lists the per-call rows (first
 *  PER_ROW inline, "Show N more tools" reveals the rest). A single-call
 *  run IS the leaf — no wrapper, no duplicate. */
function RunGroup({ items }: { items: TimelineEntry[] }) {
  const [open, setOpen] = useState(false);
  const [all, setAll] = useState(false);
  if (items.length === 1) return <Entry e={items[0]} />;

  const bad = items.some((e) => e.ok === false);
  const good = !bad && items.every((e) => e.ok !== undefined);
  const shown = all ? items : items.slice(0, PER_ROW);
  const hidden = items.length - PER_ROW;

  return (
    <div className="tl-entry tl-run">
      <button className={"tl-row" + (open ? " open" : "")} onClick={() => setOpen((o) => !o)}>
        <Icon id={iconIdOf(items[0])} />
        <span className="tl-cell-title">{composeSummary(items)}</span>
        {bad ? (
          <span className="tl-status err tl-pop">✕</span>
        ) : good ? (
          <span className="tl-status tl-pop">✓</span>
        ) : null}
        <span className="tl-chevron">›</span>
        <span className="tl-badge">· {items.length} tools</span>
      </button>
      {open && (
        <div className="tl-children">
          {shown.map((e) => (
            <Entry key={e.id} e={e} />
          ))}
          {!all && hidden > 0 && (
            <button className="tl-more" onClick={() => setAll(true)}>
              Show {hidden} more tool{hidden === 1 ? "" : "s"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** Consecutive rlm events → one row ("RLM run · 2 LLM calls · 3 rounds"),
 *  per-event children on open. A single event renders as a plain leaf. */
function RlmGroup({ items }: { items: TimelineEntry[] }) {
  const [open, setOpen] = useState(false);
  if (items.length === 1) return <Entry e={items[0]} />;
  const bits = items
    .map((e) => e.detail)
    .filter((d): d is string => typeof d === "string" && d.length > 0);
  const summary = bits.length ? bits.join(" · ") : `${items.length} events`;
  const bad = items.some((e) => e.ok === false);
  return (
    <div className="tl-entry">
      <button className={"tl-row" + (open ? " open" : "")} onClick={() => setOpen((o) => !o)}>
        <Icon id="rlm" />
        <span className="tl-cell-title">{summary}</span>
        {bad && <span className="tl-status err tl-pop">✕</span>}
        <span className="tl-chevron">›</span>
        <span className="tl-badge">· {items.length} events</span>
      </button>
      {open && (
        <div className="tl-children">
          {items.map((e) => (
            <Entry key={e.id} e={e} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Thought row: "Thinking" live → "Thought · 1.2s" when the segment
 *  closes (duration = last think chunk t − first, ms since run start).
 *  Full reasoning expands with the tail-pinned, fade-masked body. */
function ThinkEntry({ e }: { e: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  const text = e.detail ?? "";
  const dur = e.durationMs;
  const done = dur != null && dur > 0;
  return (
    <div className="tl-entry tl-think">
      <button className={"tl-row" + (open ? " open" : "")} onClick={() => setOpen((o) => !o)}>
        <Icon id="think" />
        <span className="tl-cell-title">{done ? "Thought" : "Thinking"}</span>
        {done && <span className="tl-badge">· {formatDur(dur!)}</span>}
        <span className="tl-chevron">›</span>
      </button>
      {open && text && <div className="tl-detail tl-thought-body">{text}</div>}
    </div>
  );
}

/** todo_write checklist: the row + the live checklist itself, always
 *  visible (a checklist hidden behind a chevron defeats its purpose). */
function TodoEntry({ e }: { e: TimelineEntry }) {
  let todos: Array<{ content: string; status: string }> = [];
  try {
    const parsed = JSON.parse(e.detail ?? "[]");
    if (Array.isArray(parsed)) todos = parsed;
  } catch {
    /* no checklist body */
  }
  const glyph = (s: string) => (s === "done" ? "☑" : s === "doing" ? "◐" : "⬚");
  return (
    <div className="tl-entry">
      <button className="tl-row" style={{ cursor: "default" }}>
        <Icon id="checklist" />
        <span className="tl-cell-title">{e.title}</span>
        {e.badge && <span className="tl-badge">{e.badge}</span>}
      </button>
      {todos.length > 0 && (
        <div className="todo-list">
          {todos.map((t, i) => (
            <div key={i} className={`todo-row todo-${t.status}`}>
              <span className="todo-glyph">{glyph(t.status)}</span>
              <span>{t.content}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Compress row — the one icon-free activity (the user asked for text
 *  animation instead of a glyph): while the summarizer folds the history
 *  the label sweeps like the phase line ("Compressing"); when the fold
 *  lands it settles to "Context compacted · N tk saved", the engine's
 *  full note one click away. The empty icon slot keeps the row aligned
 *  with its neighbours. */
function CompactEntry({ e }: { e: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  if (e.live) {
    return (
      <div className="tl-entry">
        <button className="tl-row" style={{ cursor: "default" }}>
          <Icon id="compress" />
          <span className="tl-cell-title">
            <PhaseSweep label="Compressing" />
          </span>
        </button>
      </div>
    );
  }
  const expandable = Boolean(e.detail);
  return (
    <div className="tl-entry">
      <button
        className={"tl-row" + (open ? " open" : "")}
        onClick={expandable ? () => setOpen((o) => !o) : undefined}
        style={expandable ? undefined : { cursor: "default" }}
      >
        <Icon id="compress" />
        <span className="tl-cell-title">{e.title}</span>
        {expandable && <span className="tl-chevron">›</span>}
        {e.badge && <span className="tl-badge">{e.badge}</span>}
      </button>
      {open && <EntryDetail e={e} />}
    </div>
  );
}

interface Group {
  kind: "run" | "rlm" | "single";
  items: TimelineEntry[];
}

/** Collapse the flat entry list: consecutive tool-ish entries (tool,
 *  tool_result fallback, delegate — any mix) → one run group;
 *  consecutive rlm → one rlm group; system notices and every other
 *  kind stand alone. */
function groupOf(entries: TimelineEntry[]): Group[] {
  const groups: Group[] = [];
  const isRun = (e: TimelineEntry) =>
    (e.kind === "tool" || e.kind === "tool_result" || e.kind === "delegate") &&
    e.tool !== "system";
  for (const e of entries) {
    const last = groups[groups.length - 1];
    if (isRun(e) && last && last.kind === "run") {
      last.items.push(e);
      continue;
    }
    if (e.kind === "rlm" && last && last.kind === "rlm") {
      last.items.push(e);
      continue;
    }
    groups.push({
      kind: isRun(e) ? "run" : e.kind === "rlm" ? "rlm" : "single",
      items: [e],
    });
  }
  return groups;
}

export function ToolTimeline({ entries }: { entries: TimelineEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="timeline">
      {groupOf(entries).map((g, i) => {
        if (g.kind === "run") return <RunGroup key={i} items={g.items} />;
        if (g.kind === "rlm") return <RlmGroup key={i} items={g.items} />;
        const e = g.items[0];
        if (e.kind === "think") return <ThinkEntry key={i} e={e} />;
        if (e.kind === "todo") return <TodoEntry key={i} e={e} />;
        if (e.kind === "compact") return <CompactEntry key={i} e={e} />;
        return <Entry key={i} e={e} />;
      })}
    </div>
  );
}
