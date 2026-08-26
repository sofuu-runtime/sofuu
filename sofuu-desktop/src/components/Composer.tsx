// Composer.tsx — the ref1/ref3 composer card: textarea on top, bottom bar
// with model/effort pills + icons left and the split send pill right.
// Home variant shows the mode pill below; chat variant shows the ⌘L hint.
//
// Every control is live: the model/effort pills patch the shared config,
// ⎘ attaches files as @mentions (chat.js expands them), ≣ toggles the
// context layers, and the mode menu switches between agent (tools) and
// "just chat" (no tools) turns.

import { useEffect, useRef, useState } from "react";
import * as backend from "../lib/backend";
import type { SofuuConfig } from "../lib/events";
import type { PermissionProfile } from "../lib/store";
import { Icon } from "./Icon";
import { Popover, PopoverItem } from "./Popover";

const EFFORTS = ["off", "low", "medium", "high", "max"];

const PERMISSIONS: Array<{ id: PermissionProfile; label: string; hint: string }> = [
  { id: "full", label: "Full access", hint: "all tools auto-run" },
  { id: "edit", label: "Edit only", hint: "read + file edits" },
  { id: "plan", label: "Plan mode", hint: "no tool access" },
  { id: "prompt", label: "Ask me", hint: "approve every tool" },
];

interface Props {
  variant: "home" | "chat";
  config: SofuuConfig | null;
  streaming: boolean;
  mode: "agent" | "chat";
  permissions: PermissionProfile;
  onSubmit: (text: string) => void;
  onCancel: () => void;
  onModeChange: (mode: "agent" | "chat") => void;
  onPermissionsChange: (profile: PermissionProfile) => void;
  onConfigPatch: (patch: Record<string, unknown>) => void;
}

type Pop = "model" | "effort" | "layers" | "mode" | "mode-caret" | "permissions" | null;

export function Composer(props: Props) {
  const { variant, config, streaming, mode, permissions, onSubmit, onCancel, onModeChange, onPermissionsChange, onConfigPatch } = props;
  const [text, setText] = useState("");
  const [pop, setPop] = useState<Pop>(null);
  const ref = useRef<HTMLTextAreaElement>(null);

  const togglePop = (which: Pop) => setPop((p) => (p === which ? null : which));

  // ⌘L focuses the composer (ref3 hint).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "l") {
        e.preventDefault();
        ref.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Auto-grow the textarea within its cap.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 220) + "px";
  }, [text]);

  const submit = () => {
    if (streaming || !text.trim()) return;
    onSubmit(text);
    setText("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  // Attach: native file picker → splice @path mentions at the cursor.
  const attach = async () => {
    try {
      const paths = await backend.pickFiles();
      if (!paths || paths.length === 0) return;
      const mention = paths.map((p) => `@${p}`).join(" ") + " ";
      const el = ref.current;
      if (el) {
        const pos = el.selectionStart ?? text.length;
        const next = text.slice(0, pos) + mention + text.slice(pos);
        setText(next);
        requestAnimationFrame(() => {
          el.focus();
          const p = pos + mention.length;
          el.setSelectionRange(p, p);
        });
      } else {
        setText(text + mention);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const model = config?.model || "Choose model";
  const effort = config?.effort || "high";
  const effortLabel = effort === "off" ? "Off" : effort.charAt(0).toUpperCase() + effort.slice(1);
  const providers = config?.providers ?? [];

  const placeholder =
    variant === "home"
      ? "Describe what you want to build"
      : "Ask to make changes, @mention files, run /commands";

  const modeMenu = (which: Pop) => (
    <Popover open={pop === which} onClose={() => setPop(null)}>
      <div className="popover-title">Turn mode</div>
      <PopoverItem
        label="Agent"
        hint="tools enabled"
        selected={mode === "agent"}
        onClick={() => {
          onModeChange("agent");
          setPop(null);
        }}
      />
      <PopoverItem
        label="Just chat"
        hint="no tools"
        selected={mode === "chat"}
        onClick={() => {
          onModeChange("chat");
          setPop(null);
        }}
      />
    </Popover>
  );

  const permissionsMenu = (
    <Popover open={pop === "permissions"} onClose={() => setPop(null)}>
      <div className="popover-title">Tool permissions</div>
      {PERMISSIONS.map((p) => (
        <PopoverItem
          key={p.id}
          label={p.label}
          hint={p.hint}
          selected={permissions === p.id}
          onClick={() => {
            onPermissionsChange(p.id);
            setPop(null);
          }}
        />
      ))}
    </Popover>
  );

  const permissionsLabel =
    PERMISSIONS.find((p) => p.id === permissions)?.label ?? "Ask me";

  return (
    <div className={"composer-wrap" + (variant === "home" ? " home" : "")}>
      <div className="composer">
        <textarea
          ref={ref}
          rows={2}
          placeholder={placeholder}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="composer-bar">
          <span className="pop-anchor">
            <button className="pill" title="Model" onClick={() => togglePop("model")}>
              <Icon name="spark" size={16} className="spark" />
              {model}
              <Icon name="caret" size={12} className="caret" />
            </button>
            <Popover open={pop === "model"} onClose={() => setPop(null)}>
              <div className="popover-title">Model</div>
              {providers.length === 0 && (
                <div className="popover-empty">Add providers in Settings</div>
              )}
              {providers.map((p) => (
                <PopoverItem
                  key={p.name}
                  label={p.model || p.name}
                  hint={p.name}
                  selected={p.model === config?.model && p.name === config?.active}
                  onClick={() => {
                    onConfigPatch({ active: p.name, model: p.model });
                    setPop(null);
                  }}
                />
              ))}
            </Popover>
          </span>

          <span className="pop-anchor">
            <button className="pill" title="Thinking effort" onClick={() => togglePop("effort")}>
              {effortLabel}
              <Icon name="caret" size={12} className="caret" />
            </button>
            <Popover open={pop === "effort"} onClose={() => setPop(null)}>
              <div className="popover-title">Thinking effort</div>
              {EFFORTS.map((e) => (
                <PopoverItem
                  key={e}
                  label={e === "off" ? "Off" : e.charAt(0).toUpperCase() + e.slice(1)}
                  selected={(config?.effort || "high") === e}
                  onClick={() => {
                    onConfigPatch({ effort: e === "off" ? "" : e });
                    setPop(null);
                  }}
                />
              ))}
            </Popover>
          </span>

          <div className="composer-icons">
            <button className="icon-btn" title="Attach files (@mention)" onClick={attach}>
              <Icon name="attach" />
            </button>
            <span className="pop-anchor">
              <button
                className={"icon-btn" + (pop === "layers" ? " active" : "")}
                title="Context layers"
                onClick={() => togglePop("layers")}
              >
                <Icon name="layers" />
              </button>
              <Popover open={pop === "layers"} onClose={() => setPop(null)}>
                <div className="popover-title">Context layers</div>
                {(
                  [
                    ["brain", "Memory (brain)", !!config?.brain],
                    ["sync", "Session mesh", config?.sync !== false],
                    ["ml", "ML context gates", config?.ml !== false],
                  ] as Array<[string, string, boolean]>
                ).map(([key, label, on]) => (
                  <button
                    key={key}
                    className="popover-row"
                    onClick={() => onConfigPatch({ [key]: !on })}
                  >
                    <span>{label}</span>
                    <span className={"toggle mini" + (on ? " on" : "")}>
                      <span className="knob" />
                    </span>
                  </button>
                ))}
              </Popover>
            </span>
          </div>

          {streaming ? (
            <button className="send-arrow" title="Stop (⌘.)" onClick={onCancel}>
              <Icon name="stop" />
            </button>
          ) : variant === "home" ? (
            <span className="send-split-wrap pop-anchor">
              <div className="send-split">
                <button className="send-main" onClick={submit}>
                  <Icon name="send" size={16} /> {mode === "chat" ? "Just chat" : "New Workspace"}
                </button>
                <button className="send-caret" title="Turn mode" onClick={() => togglePop("mode-caret")}>
                  <Icon name="caret" size={12} />
                </button>
              </div>
              {modeMenu("mode-caret")}
            </span>
          ) : (
            <>
              <span className="composer-hint">⌘L to focus</span>
              <button className="send-arrow" title="Send" onClick={submit} disabled={!text.trim()}>
                <Icon name="send" />
              </button>
            </>
          )}
        </div>
      </div>

      {variant === "home" && (
        <div className="composer-below">
          <span className="pop-anchor">
            <button className="mode-pill" title="Turn mode" onClick={() => togglePop("mode")}>
              <Icon name={mode === "chat" ? "circle-fill" : "circle"} size={12} />
              {mode === "chat" ? "Just chat" : "Agent"}
              <Icon name="caret" size={12} />
            </button>
            {modeMenu("mode")}
          </span>
          <span className="pop-anchor">
            <button
              className="mode-pill"
              title="Tool permissions"
              onClick={() => togglePop("permissions")}
            >
              <Icon name="shield" size={12} />
              {permissionsLabel}
              <Icon name="caret" size={12} />
            </button>
            {permissionsMenu}
          </span>
        </div>
      )}
    </div>
  );
}
