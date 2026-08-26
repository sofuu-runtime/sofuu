// TopBar.tsx — ref3 tab bar: horizontal strip of session pills (active
// highlighted). Pills switch between sessions of the active workspace;
// the right-side + creates a new one. 64px tall, floats under the
// traffic lights. The bar doubles as the window drag region.

import { useState } from "react";
import type { SessionInfo } from "../lib/events";
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
  onExitPreview: () => void;
}

function labelOf(s: SessionInfo): string {
  return s.task || "New chat";
}

function ageOf(s: SessionInfo): string {
  const mins = Math.max(0, Math.round((Date.now() / 1000 - s.last_seen) / 60));
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

export function TopBar(props: Props) {
  const [historyOpen, setHistoryOpen] = useState(false);

  return (
    <div className="topbar" data-tauri-drag-region>
      <div className="topbar-tabs">
        {props.previewLabel ? (
          <span className="topbar-tab preview">
            <Icon name="clock" size={16} className="topbar-spark" /> {props.previewLabel}
            <button className="topbar-tab-close" title="Back to the current chat" onClick={props.onExitPreview}>
              <Icon name="close" size={12} />
            </button>
          </span>
        ) : (
          props.sessions.map((s) => {
            const isActive = s.id === props.activeId;
            return (
              <button
                key={s.id}
                className={"topbar-tab" + (isActive ? " active" : "")}
                onClick={() => props.onSelectSession(s.id)}
                title={s.task || s.id}
              >
                <Icon name="spark" size={16} className="topbar-spark" /> {labelOf(s)}
              </button>
            );
          })
        )}
      </div>

      <div className="topbar-right">
        <button className="icon-btn" title="New chat (⌘N)" onClick={props.onNewChat}>
          <Icon name="plus" />
        </button>
        <span className="pop-anchor">
          <button
            className={"icon-btn" + (historyOpen ? " active" : "")}
            title="Session history"
            onClick={() => setHistoryOpen((v) => !v)}
          >
            <Icon name="clock" />
          </button>
          <Popover open={historyOpen} onClose={() => setHistoryOpen(false)} align="right">
            <div className="popover-title">Session history</div>
            {props.sessions.length === 0 && <div className="popover-empty">No sessions yet</div>}
            {props.sessions.map((s) => (
              <PopoverItem
                key={s.id}
                label={labelOf(s)}
                hint={ageOf(s)}
                onClick={() => {
                  setHistoryOpen(false);
                  props.onSelectSession(s.id);
                }}
              />
            ))}
          </Popover>
        </span>
      </div>
    </div>
  );
}
