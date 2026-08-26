// SettingsModal.tsx — ref2/ref4: centered 860×520 modal, left nav
// (General/Appearance/Models/Providers/Shortcuts), right pane rows with
// toggles/dropdowns/steppers.

import { useEffect, useState } from "react";
import * as backend from "../lib/backend";
import type { SofuuConfig } from "../lib/events";
import { SHORTCUTS, SHORTCUT_GROUPS } from "../lib/shortcuts";
import { Icon } from "./Icon";

interface Props {
  config: SofuuConfig | null;
  projectDir: string | null;
  chatFontSize: number;
  zoom: number;
  onChatFontSize: (n: number) => void;
  onZoomBy: (delta: number) => void;
  onZoomReset: () => void;
  onConfigChanged: (c: SofuuConfig) => void;
  onProjectDirChanged: (p: string | null) => void;
  onClose: () => void;
}

const NAV = [
  { id: "general", label: "General" },
  { id: "appearance", label: "Appearance" },
  { id: "models", label: "Models" },
  { id: "providers", label: "Providers" },
  { id: "shortcuts", label: "Shortcuts" },
];

const EFFORTS = ["off", "low", "medium", "high", "max"];

export function SettingsModal(props: Props) {
  const [tab, setTab] = useState("general");

  // Esc closes (Mac convention).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") props.onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [props]);

  return (
    <div className="modal-backdrop" onClick={props.onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header" data-tauri-drag-region>
          <span className="modal-title">Settings</span>
          <button className="modal-close" onClick={props.onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body">
          <nav className="modal-nav">
            {NAV.map((n) => (
              <button
                key={n.id}
                className={tab === n.id ? "active" : ""}
                onClick={() => setTab(n.id)}
              >
                {n.label}
              </button>
            ))}
          </nav>
          <div className="modal-pane">
            {tab === "general" && <GeneralPane {...props} />}
            {tab === "appearance" && <AppearancePane {...props} />}
            {tab === "models" && <ModelsPane {...props} />}
            {tab === "providers" && <ProvidersPane {...props} />}
            {tab === "shortcuts" && <ShortcutsPane />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Rows ──────────────────────────────────────────────────────────

function Row({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="settings-row">
      <div>
        <div className="settings-label">{label}</div>
        {desc && <div className="settings-desc">{desc}</div>}
      </div>
      {children}
    </div>
  );
}

function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      className={"toggle" + (on ? " on" : "")}
      onClick={() => onChange(!on)}
      role="switch"
      aria-checked={on}
    >
      <span className="knob" />
    </button>
  );
}

// ── General ───────────────────────────────────────────────────────

function GeneralPane({ config, projectDir, onConfigChanged, onProjectDirChanged }: Props) {
  const patch = async (p: Record<string, unknown>) => {
    try {
      const next = await backend.updateConfig(p);
      onConfigChanged(next);
    } catch (e) {
      console.error(e);
    }
  };

  const pickProject = async () => {
    try {
      const path = await backend.pickProjectDir();
      if (!path) return;
      const res = await backend.setProjectDir(path);
      if (res.ok !== false) onProjectDirChanged(path);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <>
      <Row label="Project folder" desc="Tool jail + session root. The engine runs inside it.">
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="field mono" style={{ maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis" }}>
            {projectDir ?? "not set"}
          </span>
          <button className="btn" onClick={pickProject}>Choose…</button>
        </div>
      </Row>
      <Row label="Memory (brain)" desc="Recall + store long-term memory with the chat.">
        <Toggle on={!!config?.brain} onChange={(v) => patch({ brain: v })} />
      </Row>
      <Row label="ML context gates" desc="Date fix + repeat-call/re-read nudges.">
        <Toggle on={config?.ml !== false} onChange={(v) => patch({ ml: v })} />
      </Row>
      <Row label="Session mesh" desc="See other sessions' tasks and notices on this project.">
        <Toggle on={config?.sync !== false} onChange={(v) => patch({ sync: v })} />
      </Row>
    </>
  );
}

// ── Appearance ────────────────────────────────────────────────────

function AppearancePane({ chatFontSize, onChatFontSize, zoom, onZoomBy, onZoomReset }: Props) {
  return (
    <>
      <Row label="Theme" desc="Dark and System arrive in v2.">
        <div className="segmented">
          <button className="active">☀ Light</button>
          <button disabled>☾ Dark</button>
          <button disabled>☐ System</button>
        </div>
      </Row>
      <Row label="Interface zoom" desc="Size of the whole UI (⌘+ / ⌘- / ⌘0).">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="stepper">
            <button onClick={() => onZoomBy(-0.1)} aria-label="Zoom out">−</button>
            <span className="value">{Math.round(zoom * 100)}%</span>
            <button onClick={() => onZoomBy(0.1)} aria-label="Zoom in">+</button>
          </div>
          {zoom !== 1 && (
            <button className="btn" onClick={onZoomReset}>
              Reset
            </button>
          )}
        </div>
      </Row>
      <Row label="Chat font size" desc="Assistant message body size.">
        <div className="stepper">
          <button onClick={() => onChatFontSize(chatFontSize - 1)} aria-label="Smaller">−</button>
          <span className="value">{chatFontSize}px</span>
          <button onClick={() => onChatFontSize(chatFontSize + 1)} aria-label="Larger">+</button>
        </div>
      </Row>
      <Row label="Interface font" desc="UI typeface.">
        <span className="field" style={{ minWidth: 220 }}>JetBrains Mono</span>
      </Row>
      <Row label="Code font" desc="Terminal + code blocks.">
        <span className="field mono" style={{ minWidth: 220 }}>JetBrains Mono</span>
      </Row>
    </>
  );
}

// ── Shortcuts ─────────────────────────────────────────────────────

function ShortcutsPane() {
  return (
    <>
      {SHORTCUT_GROUPS.map((group) => (
        <div key={group}>
          <div className="shortcut-group">{group}</div>
          {SHORTCUTS.filter((s) => s.group === group).map((s) => (
            <div className="settings-row" key={s.label}>
              <div className="settings-label">{s.label}</div>
              <div className="shortcut-keys">
                {s.keys.map((k, i) => (
                  <kbd className="kbd" key={i}>
                    {k}
                  </kbd>
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}
    </>
  );
}

// ── Models ────────────────────────────────────────────────────────

function ModelsPane({ config, onConfigChanged }: Props) {
  const patch = async (p: Record<string, unknown>) => {
    try {
      onConfigChanged(await backend.updateConfig(p));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <>
      <Row label="Model" desc="Default model for new turns.">
        <input
          className="field mono"
          style={{ minWidth: 260 }}
          defaultValue={config?.model ?? ""}
          onBlur={(e) => {
            if (e.target.value !== (config?.model ?? "")) patch({ model: e.target.value });
          }}
        />
      </Row>
      <Row label="Thinking effort" desc="Reasoning budget per turn (off = provider default).">
        <select
          className="field"
          value={config?.effort || "high"}
          onChange={(e) => patch({ effort: e.target.value === "off" ? "" : e.target.value })}
        >
          {EFFORTS.map((e) => (
            <option key={e} value={e}>
              {e === "off" ? "off" : e.charAt(0).toUpperCase() + e.slice(1)}
            </option>
          ))}
        </select>
      </Row>
      <Row label="Context window" desc="Tokens; 0 = model default (max 1M).">
        <input
          className="field mono"
          style={{ width: 140 }}
          defaultValue={config?.ctx_window ?? 0}
          onBlur={(e) => {
            const n = Number(e.target.value) || 0;
            if (n !== (config?.ctx_window ?? 0)) patch({ ctx_window: n });
          }}
        />
      </Row>
      <Row label="Max output tokens" desc="Per response; 0 = model default (cap 384k).">
        <input
          className="field mono"
          style={{ width: 140 }}
          defaultValue={config?.max_output ?? 0}
          onBlur={(e) => {
            const n = Number(e.target.value) || 0;
            if (n !== (config?.max_output ?? 0)) patch({ max_output: n });
          }}
        />
      </Row>
    </>
  );
}

// ── Providers ─────────────────────────────────────────────────────

function ProvidersPane({ config, onConfigChanged }: Props) {
  const [keyDraft, setKeyDraft] = useState<Record<string, string>>({});
  const providers = config?.providers ?? [];

  const select = async (name: string) => {
    try {
      onConfigChanged(await backend.updateConfig({ active: name }));
    } catch (e) {
      console.error(e);
    }
  };

  const saveKey = async (name: string) => {
    const key = (keyDraft[name] ?? "").trim();
    if (!key) return;
    try {
      await backend.keychainSet(name, key);
      setKeyDraft((d) => ({ ...d, [name]: "" }));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <>
      {providers.length === 0 && (
        <div style={{ padding: "16px 0", color: "var(--text-muted)", fontSize: 14 }}>
          No providers configured yet. Add one in the CLI (<span className="field mono" style={{ padding: "2px 6px" }}>/provider</span>) or add entries here in v2.
        </div>
      )}
      {providers.map((p) => (
        <div key={p.name}>
          <div className="provider-row">
            <span className="provider-name">{p.name}</span>
            {p.name === config?.active && <span className="provider-active">active</span>}
            <span className="provider-meta" style={{ marginLeft: "auto" }}>
              {p.model || "no model"} {p.endpoint ? `· ${p.endpoint}` : ""}
            </span>
            {p.name !== config?.active && (
              <button className="btn" onClick={() => select(p.name)}>Use</button>
            )}
          </div>
          <div className="settings-row" style={{ paddingTop: 8 }}>
            <div>
              <div className="settings-desc">
                API key stored in the macOS Keychain{p.has_api_key ? " (one is saved)" : ""}.
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <input
                className="field mono"
                type="password"
                placeholder={p.has_api_key ? "••••••••" : "paste key"}
                value={keyDraft[p.name] ?? ""}
                onChange={(e) => setKeyDraft((d) => ({ ...d, [p.name]: e.target.value }))}
                style={{ minWidth: 220 }}
              />
              <button className="btn" onClick={() => saveKey(p.name)} disabled={!(keyDraft[p.name] ?? "").trim()}>
                Save
              </button>
            </div>
          </div>
        </div>
      ))}
    </>
  );
}
