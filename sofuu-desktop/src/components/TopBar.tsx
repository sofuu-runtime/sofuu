// TopBar.tsx — ref3 tab bar: horizontal strip of session pills (active
// highlighted). Pills switch between sessions of the active workspace;
// the right-side + creates a new one. 64px tall, floats under the
// traffic lights. The bar doubles as the window drag region.

import { useState } from "react";
import { MAX_SESSIONS, type SessionInfo } from "../lib/events";
import { Icon } from "./Icon";
import { Popover, PopoverItem } from "./Popover";

interface Props {
  /** Id of the session currently in the chat pane, or null when empty. */
  activeId: string | null;
  /** Read-only transcript being viewed, if any (◷ history). */
  previewLabel: string | null;
  sessions: SessionInfo[];
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onClearSessions: () => void;
  onExitPreview: () => void;
}

function labelOf(s: SessionInfo, n: number): string {
  return s.task || `Chat ${n}`;
}

function ageOf(s: SessionInfo): string {
  const mins = Math.max(0, Math.round((Date.now() / 1000 - s.last_seen) / 60));
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

/** One session pill. Delete = inline two-step confirm (no window.confirm —
 *  Tauri's dialog ACL blocks it): first click arms the ×, second click
 *  deletes, or it disarms after 2.5s. Untitled chats skip the arm step. */
function Pill({
  s,
  n,
  isActive,
  onSelect,
  onDelete,
}: {
  s: SessionInfo;
  n: number;
  isActive: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [armed, setArmed] = useState(false);
  return (
    <span
      className={"topbar-tab" + (isActive ? " active" : "") + (armed ? " armed" : "")}
      role="tab"
      aria-selected={isActive}
      title={s.task || s.id}
      onClick={() => onSelect(s.id)}
    >
      <Icon name="spark" size={16} className="topbar-spark" /> {labelOf(s, n)}
      <button
        className="topbar-tab-close"
        title={armed ? "Click again to delete" : "Delete chat"}
        onClick={(e) => {
          e.stopPropagation();
          if (armed || !s.task) {
            onDelete(s.id);
          } else {
            setArmed(true);
            setTimeout(() => setArmed(false), 2500);
          }
        }}
      >
        <Icon name="close" size={12} />
      </button>
    </span>
  );
}

export function TopBar(props: Props) {
  const [historyOpen, setHistoryOpen] = useState(false);
  // Clear-all uses the same two-step armed confirm as the pill × (2.5s).
  const [clearArmed, setClearArmed] = useState(false);
  // Pill numbers follow creation order (oldest = Session - 1); ties broken
  // by id so the map is deterministic across re-renders.
  const nums = new Map(
    [...props.sessions]
      .sort((a, b) => a.started_at - b.started_at || a.id.localeCompare(b.id))
      .map((s, i) => [s.id, i + 1])
  );
  const atCap = props.sessions.length >= MAX_SESSIONS;

  return (
    <div className="topbar" data-tauri-drag-region>
      <div className="topbar-tabs">
        {props.previewLabel ? (
          <span className="topbar-tab preview">
            <Icon name="clock" size={16} className="topbar-spark" /> {props.previewLabel}
            <button className="topbar-tab-close" title="Back to active chat" onClick={props.onExitPreview}>
              <Icon name="close" size={12} />
            </button>
          </span>
        ) : (
          (() => {
            // The strip shows at most MAX_SESSIONS pills (newest first).
            // A legacy workspace may hold more — the overflow lives in the
            // clock (history) popover. The ACTIVE pill is always visible:
            // if it falls outside the newest window, it swaps in for the
            // oldest shown entry.
            const sorted = [...props.sessions].sort(
              (a, b) => (b.last_seen || 0) - (a.last_seen || 0)
            );
            let visible = sorted.slice(0, MAX_SESSIONS);
            if (props.activeId && !visible.some((s) => s.id === props.activeId)) {
              const act = props.sessions.find((s) => s.id === props.activeId);
              if (act) visible = [act, ...visible.slice(0, MAX_SESSIONS - 1)];
            }
            return visible.map((s) => {
              const isActive = s.id === props.activeId;
              return (
                <Pill
                  key={s.id}
                  s={s}
                  n={nums.get(s.id) ?? 0}
                  isActive={isActive}
                  onSelect={props.onSelectSession}
                  onDelete={props.onDeleteSession}
                />
              );
            });
          })()
        )}
      </div>

      <div className="topbar-right">
        <button
          className="icon-btn"
          title={atCap ? `Chat limit reached (${MAX_SESSIONS}) — close or delete one first` : "New chat (⌘N)"}
          onClick={props.onNewChat}
          disabled={atCap}
        >
          <Icon name="plus" />
        </button>
        <span className="pop-anchor">
          <button
            className={"icon-btn" + (historyOpen ? " active" : "")}
            title="Chat history"
            onClick={() => setHistoryOpen((v) => !v)}
          >
            <Icon name="clock" />
          </button>
          <Popover open={historyOpen} onClose={() => setHistoryOpen(false)} align="right">
            <div className="popover-title">Chat history</div>
            {props.sessions.length === 0 && <div className="popover-empty">No chats yet</div>}
            {props.sessions.map((s) => (
              <PopoverItem
                key={s.id}
                label={labelOf(s, nums.get(s.id) ?? 0)}
                hint={ageOf(s)}
                onClick={() => {
                  setHistoryOpen(false);
                  props.onSelectSession(s.id);
                }}
              />
            ))}
            {props.sessions.length > 0 && (
              <>
                <div className="popover-sep" />
                <button
                  className={"popover-item danger" + (clearArmed ? " armed" : "")}
                  title="Permanently delete all chats in this workspace"
                  onClick={() => {
                    if (clearArmed) {
                      setClearArmed(false);
                      setHistoryOpen(false);
                      props.onClearSessions();
                    } else {
                      setClearArmed(true);
                      setTimeout(() => setClearArmed(false), 2500);
                    }
                  }}
                >
                  <span>{clearArmed ? "Click again to confirm" : "Delete all chats in workspace"}</span>
                </button>
              </>
            )}
          </Popover>
        </span>
      </div>
    </div>
  );
}
