// timeline.ts — presentation helpers for the chat activity timeline:
// per-activity icon ids (the registry the user-supplied SVGs plug into),
// verb-first tool row titles ("Read AGENTS.md", "Ran git status"), and
// the aggregate run summary ("Searched 2 patterns, read 4 files").
// Pure logic — no JSX, so the renderer can stay a .tsx component file.

import type { TimelineEntry } from "./events";

/* ── Icon registry ───────────────────────────────────────────────
 * Every activity kind maps to ONE icon id here. The SVGs themselves
 * live in components/icons.tsx (the user-supplied Tabler outlines) —
 * this file stays pure logic, no JSX. */
export type IconKey =
  | "think" | "read" | "write" | "edit" | "grep" | "glob" | "folder"
  | "terminal" | "web" | "link" | "delegate" | "checklist" | "brain"
  | "compress" | "gauge" | "freshness" | "relevance" | "supervisor"
  | "loop" | "rlm" | "warn" | "notice" | "plan" | "dot";

/** Icons whose subject text renders in the mono face (commands, patterns). */
export function isMonoIcon(icon: IconKey): boolean {
  return icon === "terminal" || icon === "grep" || icon === "glob";
}

const TOOL_ICON: Record<string, IconKey> = {
  read_file: "read",
  write_file: "write",
  edit_file: "edit",
  grep: "grep",
  glob: "glob",
  list_dir: "folder",
  bash: "terminal",
  web_search: "web",
  web_open: "link",
  delegate: "delegate",
  todo_write: "checklist",
};

export function toolIcon(name: string): IconKey {
  return TOOL_ICON[name] ?? "dot";
}

export function iconIdOf(e: TimelineEntry): IconKey {
  switch (e.kind) {
    case "tool":
      return toolIcon(e.tool ?? e.title);
    case "tool_result":
      return e.title === "Notice" ? "notice" : toolIcon(e.tool ?? e.title);
    case "delegate":
      return "delegate";
    case "recall":
      return "brain";
    case "consolidated":
      return "compress";
    case "allocgate":
      return "gauge";
    case "mlgate":
      if (e.tool === "freshness") return "freshness";
      if (e.tool === "relevance") return "relevance";
      if (e.tool === "loop") return "loop";
      return "supervisor";
    case "rlm":
      return "rlm";
    case "todo":
      return "checklist";
    case "warn":
      return /compact|prun/i.test(e.title) ? "compress" : "warn";
    case "think":
      return "think";
    case "plan":
      return "plan";
    default:
      return "dot";
  }
}

/* ── Verb-first row titles ─────────────────────────────────────── */

function basename(path: string): string {
  const parts = String(path).split("/");
  return parts[parts.length - 1] || String(path);
}

function clip(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function firstLine(s: string): string {
  const nl = s.indexOf("\n");
  return nl >= 0 ? s.slice(0, nl) : s;
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return clip(url, 40);
  }
}

/** {title, detail} for one tool call. `args` is the JSON-parsed args the
 * engine clipped to 200 chars — may be null when the clip broke the JSON. */
export function toolRow(
  name: string,
  args: Record<string, unknown> | null
): { title: string; detail?: string } {
  const a = args ?? {};
  const str = (k: string) => (typeof a[k] === "string" && a[k] ? (a[k] as string) : "");
  switch (name) {
    case "read_file": {
      const path = str("path");
      if (!path) break;
      const range =
        a.start != null || a.end != null
          ? `lines ${a.start ?? 1}–${a.end ?? "end"}`
          : undefined;
      return { title: `Read ${basename(path)}`, detail: range };
    }
    case "write_file": {
      const path = str("path");
      if (!path) break;
      return { title: `Wrote ${basename(path)}` };
    }
    case "edit_file": {
      const path = str("path");
      if (!path) break;
      return { title: `Edited ${basename(path)}` };
    }
    case "grep": {
      const pattern = str("pattern");
      if (!pattern) break;
      const path = str("path");
      return {
        title: `Searched “${clip(pattern, 48)}”`,
        detail: path ? `${clip(pattern, 200)}\nin ${path}` : clip(pattern, 200),
      };
    }
    case "glob": {
      const pattern = str("pattern");
      if (!pattern) break;
      const path = str("path");
      return {
        title: `Matched “${clip(pattern, 48)}”`,
        detail: path ? `${clip(pattern, 200)}\nin ${path}` : clip(pattern, 200),
      };
    }
    case "list_dir": {
      const path = str("path");
      return { title: `Listed ${path ? basename(path) : "project"}` };
    }
    case "bash": {
      const cmd = str("command");
      if (!cmd) break;
      return { title: `Ran ${clip(firstLine(cmd), 72)}`, detail: cmd };
    }
    case "web_search": {
      const query = str("query");
      if (!query) break;
      return { title: `Searched the web for “${clip(query, 48)}”` };
    }
    case "web_open": {
      const url = str("url");
      if (!url) break;
      return { title: `Opened ${hostOf(url)}`, detail: url };
    }
  }
  // Built-in tool with unparseable args, or a dynamic (MCP) tool: the
  // raw name is the honest row.
  return { title: name };
}

/* ── Run summaries ─────────────────────────────────────────────── */

/** [verb, singular noun, plural noun] per known tool. Unknown tools
 * fall back to "Ran <name>". */
const TOOL_RUN: Record<string, [string, string, string]> = {
  read_file: ["Read", "file", "files"],
  write_file: ["Wrote", "file", "files"],
  edit_file: ["Edited", "file", "files"],
  grep: ["Searched", "pattern", "patterns"],
  glob: ["Matched", "pattern", "patterns"],
  list_dir: ["Listed", "folder", "folders"],
  bash: ["Ran", "command", "commands"],
  web_search: ["Searched the web for", "query", "queries"],
  web_open: ["Opened", "page", "pages"],
  delegate: ["Delegated", "task", "tasks"],
};

function runClause(e: TimelineEntry, count: number): string {
  const name = e.tool ?? e.title;
  const run = TOOL_RUN[name];
  if (!run) return count > 1 ? `Ran ${name} ×${count}` : `Ran ${name}`;
  const [verb, one, many] = run;
  return count > 1 ? `${verb} ${count} ${many}` : `${verb} 1 ${one}`;
}

/** Aggregate row title for a consecutive run of tool calls, in order of
 * first appearance, capped at 3 clauses ("+2 more"). */
export function composeSummary(items: TimelineEntry[]): string {
  const order: string[] = [];
  const counts = new Map<string, number>();
  for (const e of items) {
    const key = e.tool ?? e.title;
    if (!counts.has(key)) order.push(key);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const clauses = order.map((key) => runClause(items.find((e) => (e.tool ?? e.title) === key)!, counts.get(key)!));
  const shown = clauses.slice(0, 3);
  const rest = clauses.length - shown.length;
  return shown.join(", ") + (rest > 0 ? `, +${rest} more` : "");
}

/* ── Durations & gate titles ───────────────────────────────────── */

export function formatDur(ms: number): string {
  if (ms < 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 1000)}s`;
}

export function mlgateTitle(rule: string): string {
  if (rule === "freshness") return "Freshness gate";
  if (rule === "relevance") return "Relevance gate";
  if (rule === "loop") return "Loop gate";
  return "Supervisor";
}
