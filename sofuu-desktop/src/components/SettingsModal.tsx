// SettingsModal.tsx — every control here is wired: toggles/selects patch
// ~/.sofuu/config.json through update_config (the Rust side whitelists
// keys and merges providers by name), permission profiles ride
// set_permissions (engine-enforced + persisted), provider keys ride the
// macOS Keychain, provider removal rides remove_provider, and the Engine
// card shows live chat.state() values, and the Theme control re-colors the
// whole app live (see store.ts — data-theme + native window setTheme).

import { useCallback, useEffect, useState } from "react";
import * as backend from "../lib/backend";
import type { EngineState } from "../lib/backend";
import type { SofuuConfig } from "../lib/events";
import type { PermissionProfile, ThemePref } from "../lib/store";
import { SHORTCUTS, SHORTCUT_GROUPS } from "../lib/shortcuts";
import { Icon } from "./Icon";

interface Props {
  config: SofuuConfig | null;
  /** Tab the modal opens on ("providers" when the model picker asks to
   *  add one). The modal mounts fresh per open, so this only seeds state. */
  initialTab?: string;
  projectDir: string | null;
  chatFontSize: number;
  zoom: number;
  /** Procedural cloud background (Appearance). */
  cloudBg: boolean;
  /** App-wide theme preference — "system" follows macOS live. */
  theme: ThemePref;
  /** Tool permission profile (engine-enforced; persisted). */
  permissions: PermissionProfile;
  onChatFontSize: (n: number) => void;
  onZoomBy: (delta: number) => void;
  onZoomReset: () => void;
  onCloudBg: (v: boolean) => void;
  onTheme: (t: ThemePref) => void;
  onPermissionsChange: (p: PermissionProfile) => void;
  onConfigChanged: (c: SofuuConfig) => void;
  onProjectDirChanged: (p: string | null) => void;
  onClose: () => void;
}

const NAV = [
  { id: "general", label: "General" },
  { id: "appearance", label: "Appearance" },
  { id: "models", label: "Models" },
  { id: "providers", label: "Providers" },
  { id: "usage", label: "Usage" },
  { id: "shortcuts", label: "Shortcuts" },
];

const EFFORTS = ["off", "low", "medium", "high", "max"];

/** Real enforcement matrix — mirrors profileGates() in src/js/chat.js. */
const PERMISSION_PROFILES: Array<{ id: PermissionProfile; label: string; desc: string }> = [
  { id: "full", label: "Full access", desc: "Every tool runs without asking — shell included." },
  { id: "edit", label: "Edit only", desc: "Reads, search and jailed file edits run; bash and MCP tools are blocked." },
  { id: "plan", label: "Plan mode", desc: "Read-only tools only — the model plans and answers, changes nothing." },
  { id: "prompt", label: "Ask me", desc: "Every non-read-only tool waits for your approval." },
];

export function SettingsModal(props: Props) {
  const [tab, setTab] = useState(props.initialTab ?? "general");

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
            {tab === "usage" && <UsagePane {...props} />}
            {tab === "shortcuts" && <ShortcutsPane />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Shared bits ───────────────────────────────────────────────────

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

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div className="settings-section">{children}</div>;
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

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: Array<{ id: T; label: string; title?: string }>;
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button
          key={o.id}
          className={value === o.id ? "active" : ""}
          title={o.title}
          onClick={() => onChange(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Numeric field that commits on blur/Enter and reverts on invalid input.
 *  Empty text parses as 0 — the config's "use the built-in default". */
function NumberField({
  value,
  onCommit,
  width = 140,
  step,
  suffix,
}: {
  value: number;
  onCommit: (n: number) => void;
  width?: number;
  step?: string;
  suffix?: string;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const commit = () => {
    const t = draft.trim();
    const n = t === "" ? 0 : Number(t);
    if (Number.isFinite(n) && n >= 0 && n !== value) onCommit(n);
    else setDraft(String(value));
  };
  return (
    <div className="numfield">
      <input
        className="field mono"
        inputMode="decimal"
        style={{ width }}
        value={draft}
        step={step}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
      />
      {suffix && <span className="numfield-suffix">{suffix}</span>}
    </div>
  );
}

function fmtTk(n: number | undefined): string {
  const v = n ?? 0;
  return v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : String(v);
}

// ── General ───────────────────────────────────────────────────────

function GeneralPane(props: Props) {
  const { config, projectDir, permissions, onPermissionsChange, onConfigChanged, onProjectDirChanged } = props;
  const [engine, setEngine] = useState<EngineState | null>(null);
  const pullEngine = useCallback(() => {
    backend.engineState().then(setEngine);
  }, []);
  useEffect(pullEngine, [pullEngine]);

  const patch = useCallback(
    async (p: Record<string, unknown>) => {
      try {
        onConfigChanged(await backend.updateConfig(p));
        pullEngine();
      } catch (e) {
        console.error(e);
      }
    },
    [onConfigChanged, pullEngine]
  );

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

  const rlm = config?.rlm === "on" ? "on" : config?.rlm === "auto" ? "auto" : "off";
  const profileDesc =
    PERMISSION_PROFILES.find((p) => p.id === (permissions as PermissionProfile))?.desc ?? "";

  return (
    <>
      <SectionTitle>Engine</SectionTitle>
      {engine ? (
        <div className="engine-card">
          <div className="engine-grid">
            <div className="engine-cell">
              <span className="engine-key">Model</span>
              <span className="engine-val mono">{engine.model || "None"}</span>
            </div>
            <div className="engine-cell">
              <span className="engine-key">Provider</span>
              <span className="engine-val">{engine.provider || "None"}</span>
            </div>
            <div className="engine-cell">
              <span className="engine-key">Session</span>
              <span className="engine-val mono" title={engine.sessionId ?? undefined}>
                {engine.sessionId ? engine.sessionId.slice(0, 10) + "…" : "None yet"}
              </span>
            </div>
            <div className="engine-cell">
              <span className="engine-key">History</span>
              <span className="engine-val">{engine.historyTurns ?? 0} turns</span>
            </div>
            <div className="engine-cell">
              <span className="engine-key">Context</span>
              <span className="engine-val">{fmtTk(engine.contextTokens)} tk</span>
            </div>
            <div className="engine-cell">
              <span className="engine-key">Session spend</span>
              <span className="engine-val">${(engine.sessionSpendUsd ?? 0).toFixed(4)}</span>
            </div>
          </div>
          {(engine.streaming || (engine.pendingApprovals ?? 0) > 0) && (
            <div className="engine-live">
              {engine.streaming && <span className="engine-badge">Generating…</span>}
              {(engine.pendingApprovals ?? 0) > 0 && (
                <span className="engine-badge warn">{engine.pendingApprovals} awaiting approval</span>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="settings-desc" style={{ padding: "4px 0 10px" }}>
          Engine not connected (dev mode outside Tauri shows no live state).
        </div>
      )}

      <SectionTitle>Workspace</SectionTitle>
      <Row label="Project folder" desc="Workspace root and per-project storage (.sofuu/ — sessions, memory, settings). The engine operates inside this directory.">
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="field mono" style={{ maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis" }}>
            {projectDir ?? "not set"}
          </span>
          <button className="btn" onClick={pickProject}>Choose…</button>
        </div>
      </Row>

      <SectionTitle>Permissions</SectionTitle>
      <Row label="Tool permission profile" desc={profileDesc + " Applied live in the engine and re-applied on launch."}>
        <Segmented<PermissionProfile>
          value={permissions}
          options={PERMISSION_PROFILES.map((p) => ({ id: p.id, label: p.label, title: p.desc }))}
          onChange={onPermissionsChange}
        />
      </Row>

      <SectionTitle>Memory</SectionTitle>
      <Row label="Memory (brain)" desc="Recall saved project memories into each turn and let the model store new ones (.sofuu/brain/).">
        <Toggle on={!!config?.brain} onChange={(v) => patch({ brain: v })} />
      </Row>
      <Row label="Recall similarity floor" desc="Minimum match score for a memory to be recalled. 0 = built-in 0.30 (unrelated text scores ≈0.27, paraphrases >0.35 on the bundled embedder).">
        <NumberField
          value={config?.recall_min ?? 0}
          step="0.01"
          width={110}
          onCommit={(n) => patch({ recall_min: n })}
        />
      </Row>
      <Row label="Recall token budget" desc="Max tokens of memories injected per turn. 0 = auto: ≈2% of the context window (1k–16k).">
        <NumberField
          value={config?.recall_budget ?? 0}
          width={110}
          suffix="tk"
          onCommit={(n) => patch({ recall_budget: n })}
        />
      </Row>
      <Row label="Embeddings" desc="Optional OpenAI-compatible provider + model for memory embeddings. Blank = the bundled offline embedder (no network).">
        <div style={{ display: "flex", gap: 8 }}>
          <input
            className="field mono"
            style={{ width: 150 }}
            placeholder="provider"
            defaultValue={config?.embed_provider ?? ""}
            onBlur={(e) => {
              if (e.target.value !== (config?.embed_provider ?? "")) patch({ embed_provider: e.target.value });
            }}
          />
          <input
            className="field mono"
            style={{ width: 150 }}
            placeholder="model"
            defaultValue={config?.embed_model ?? ""}
            onBlur={(e) => {
              if (e.target.value !== (config?.embed_model ?? "")) patch({ embed_model: e.target.value });
            }}
          />
        </div>
      </Row>

      <SectionTitle>Reasoning</SectionTitle>
      <Row label="Smart context filters" desc="On-device neural filters (freshness, compaction, relevance, allocation) that optimize context efficiency without blocking tools.">
        <Toggle on={config?.ml !== false} onChange={(v) => patch({ ml: v })} />
      </Row>
      <Row label="RLM sandbox" desc="Route very long-context turns through the local QuickJS reasoning sandbox. Auto = the engine decides when it helps.">
        <Segmented
          value={rlm}
          options={[
            { id: "off", label: "Off", title: "Never use the sandbox" },
            { id: "auto", label: "Auto", title: "Sandbox only turns the engine estimates need it" },
            { id: "on", label: "Always", title: "Route every turn through the sandbox" },
          ]}
          onChange={(v) => patch({ rlm: v })}
        />
      </Row>
      <Row label="Session mesh" desc="Sessions persist to the shared .sofuu store and sync with the CLI/TUI automatically — always on, by design.">
        <span className="field" style={{ minWidth: 120, textAlign: "center" }}>Built-in</span>
      </Row>
    </>
  );
}

// ── Appearance ────────────────────────────────────────────────────

function AppearancePane(props: Props) {
  const { chatFontSize, onChatFontSize, zoom, onZoomBy, onZoomReset, cloudBg, onCloudBg, theme, onTheme } = props;
  return (
    <>
      <Row label="Theme" desc="Dark re-colors the whole app — sidebar glass, chat, popovers and the cloud background. System follows macOS and switches live.">
        <Segmented
          value={theme}
          options={[
            { id: "light", label: "Light" },
            { id: "dark", label: "Dark" },
            { id: "system", label: "System" },
          ]}
          onChange={(id) => onTheme(id as ThemePref)}
        />
      </Row>
      <Row label="Interface zoom" desc="Scales the whole UI, 50%–200% (⌘+ / ⌘- / ⌘0 work anywhere).">
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
      <Row label="Chat font size" desc="Assistant message body size, 14–24 px.">
        <div className="stepper">
          <button
            onClick={() => onChatFontSize(chatFontSize - 1)}
            disabled={chatFontSize <= 14}
            aria-label="Smaller"
          >
            −
          </button>
          <span className="value">{chatFontSize}px</span>
          <button
            onClick={() => onChatFontSize(chatFontSize + 1)}
            disabled={chatFontSize >= 24}
            aria-label="Larger"
          >
            +
          </button>
        </div>
      </Row>
      <Row label="Cloud background" desc="Animated procedural clouds behind the interface (WebGL). Off = flat gradient.">
        <Toggle on={cloudBg} onChange={onCloudBg} />
      </Row>
    </>
  );
}

// ── Models ────────────────────────────────────────────────────────

function ModelsPane({ config, onConfigChanged, projectDir }: Props) {
  const patch = useCallback(
    async (p: Record<string, unknown>) => {
      try {
        onConfigChanged(await backend.updateConfig(p));
      } catch (e) {
        console.error(e);
      }
    },
    [onConfigChanged]
  );

  return (
    <>
      <Row label="Active provider" desc="Set on the Providers tab. New turns go to it.">
        <span className="field" style={{ minWidth: 200 }}>{config?.provider || "none"}</span>
      </Row>
      <Row label="Model" desc="Model id sent verbatim to the active provider. The composer's model menu lists the provider's models and writes the pick here.">
        <input
          className="field mono"
          style={{ minWidth: 260 }}
          defaultValue={config?.model ?? ""}
          onBlur={(e) => {
            if (e.target.value !== (config?.model ?? "")) patch({ model: e.target.value });
          }}
        />
      </Row>
      <Row label="Thinking effort" desc="Reasoning effort sent with each request; off omits the parameter. Models that rejected thinking are remembered and never receive it again.">
        <select
          className="field"
          value={config?.effort === "" ? "off" : config?.effort || "high"}
          onChange={(e) => patch({ effort: e.target.value === "off" ? "" : e.target.value })}
        >
          {EFFORTS.map((e) => (
            <option key={e} value={e}>
              {e === "off" ? "off" : e.charAt(0).toUpperCase() + e.slice(1)}
            </option>
          ))}
        </select>
      </Row>
      <Row label="Context window" desc="Tokens. 0 = the engine resolves it live: boot report → learned model limits → discovered caps → 131k default. A manual override is clamped to the model's real limit.">
        <NumberField
          value={config?.ctx_window ?? 0}
          width={130}
          suffix="tk"
          onCommit={(n) => patch({ ctx_window: n })}
        />
      </Row>
      <Row label="Max output tokens" desc="Sent as max_tokens; a too-large value is clamped to what the model accepted before. 0 = provider default (a global override here is what 400'd small models — leave 0 unless you know the limit).">
        <NumberField
          value={config?.max_output ?? 0}
          width={130}
          suffix="tk"
          onCommit={(n) => patch({ max_output: n })}
        />
      </Row>
      <Row label="Session spend cap" desc="New turns are refused once the session's priced spend (pricing-table estimate) reaches this. 0 = no cap.">
        <NumberField
          value={config?.budget_usd ?? 0}
          width={110}
          suffix="USD"
          onCommit={(n) => patch({ budget_usd: n })}
        />
      </Row>
      <div className="settings-desc" style={{ padding: "10px 0" }}>
        Workspace store: <span className="mono">{projectDir ?? "not set"}</span>
      </div>
    </>
  );
}

// ── Providers ─────────────────────────────────────────────────────

/** Keychain status is separate from config's has_api_key (which only
 *  reflects CLI-written keys in config.json) — query it per provider. */
function useKeychainStatus(names: string[]) {
  const [status, setStatus] = useState<Record<string, boolean>>({});
  const namesKey = names.join("|");
  useEffect(() => {
    let alive = true;
    for (const n of namesKey.split("|").filter(Boolean)) {
      backend
        .keychainHas(n)
        .then((r) => {
          if (alive) setStatus((s) => ({ ...s, [n]: !!r.has_key }));
        })
        .catch(() => {});
    }
    return () => {
      alive = false;
    };
  }, [namesKey]);
  return [status, setStatus] as const;
}

function ProvidersPane({ config, onConfigChanged }: Props) {
  const providers = config?.providers ?? [];
  const active = config?.active || config?.provider || "";
  const [keyStatus, setKeyStatus] = useKeychainStatus(providers.map((p) => p.name));
  const [draft, setDraft] = useState({ name: "", endpoint: "", profile: "openai", model: "" });
  const [draftKey, setDraftKey] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  const patch = useCallback(
    async (p: Record<string, unknown>) => {
      try {
        onConfigChanged(await backend.updateConfig(p));
      } catch (e) {
        console.error(e);
      }
    },
    [onConfigChanged]
  );

  /** Partial per-provider patch: the Rust merge updates only the named
   *  fields and never touches api_key. Re-send `active` so the flat
   *  mirror re-syncs when editing the active provider's entry. */
  const patchProvider = useCallback(
    (name: string, fields: Record<string, unknown>) => {
      const p: Record<string, unknown> = { providers: [{ name, ...fields }] };
      if (name === active) p.active = name;
      patch(p);
    },
    [active, patch]
  );

  const saveKey = async (name: string, key: string) => {
    try {
      await backend.keychainSet(name, key);
      setKeyStatus((s) => ({ ...s, [name]: true }));
      setNotice(`Key for ${name} saved to the Keychain.`);
    } catch (e) {
      console.error(e);
    }
  };

  const removeKey = async (name: string) => {
    try {
      await backend.keychainDelete(name);
      setKeyStatus((s) => ({ ...s, [name]: false }));
      setNotice(`Keychain entry for ${name} removed.`);
    } catch (e) {
      console.error(e);
    }
  };

  const deleteProvider = async (name: string) => {
    if (!window.confirm(`Remove provider "${name}" and its stored Keychain credentials?`)) return;
    try {
      onConfigChanged(await backend.removeProvider(name));
      setNotice(`Provider ${name} removed.`);
    } catch (e) {
      setNotice(String(e));
    }
  };

  const addProvider = async () => {
    const name = draft.name.trim();
    const endpoint = draft.endpoint.trim();
    if (!name || !endpoint) return;
    if (!/^https?:\/\//i.test(endpoint)) {
      setNotice("Base URL must start with http:// or https://");
      return;
    }
    const key = draftKey.trim();
    try {
      const next = [
        ...providers,
        { name, endpoint, profile: draft.profile, model: draft.model.trim(), api_key: "" },
      ];
      const p: Record<string, unknown> = { providers: next, active: name };
      if (draft.model.trim()) p.model = draft.model.trim();
      onConfigChanged(await backend.updateConfig(p));
      if (key) {
        await backend.keychainSet(name, key);
        setKeyStatus((s) => ({ ...s, [name]: true }));
      }
      setDraft({ name: "", endpoint: "", profile: "openai", model: "" });
      setDraftKey("");
      setNotice(`Provider ${name} added and set active.`);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <>
      {notice && <div className="settings-notice">{notice}</div>}
      {providers.length === 0 && (
        <div className="settings-desc" style={{ padding: "8px 0 12px" }}>
          No providers yet — add one below. Nothing sends until a provider is active and has a
          model.
        </div>
      )}
      {providers.map((p) => {
        const keySaved = keyStatus[p.name] ?? false;
        const hasKey = keySaved || !!p.has_api_key;
        const isActive = p.name === active;
        return (
          <div key={p.name} className="provider-card">
            <div className="provider-head">
              <span className="provider-name">{p.name}</span>
              {isActive && <span className="provider-active">active</span>}
              <span className="provider-meta">
                {p.profile}
                {hasKey ? " · key set" : " · no key"}
              </span>
              <div className="provider-actions">
                {!isActive && (
                  <button className="btn" onClick={() => patch({ active: p.name })}>
                    Use
                  </button>
                )}
                <button className="btn danger" onClick={() => deleteProvider(p.name)} title="Remove this provider, its Keychain key and cached model list">
                  Remove
                </button>
              </div>
            </div>
            <div className="provider-endpoint">{p.endpoint}</div>
            <div className="provider-grid">
              <div className="provider-field">
                <span className="provider-field-label">Model</span>
                <input
                  className="field mono"
                  placeholder="model id sent verbatim"
                  defaultValue={p.model ?? ""}
                  onBlur={(e) => {
                    const v = e.target.value.trim();
                    if (v !== (p.model ?? "")) patchProvider(p.name, { model: v });
                  }}
                />
              </div>
              <div className="provider-field" style={{ flex: "2 1 260px" }}>
                <span className="provider-field-label">API key</span>
                <div className="provider-key-row">
                  <input
                    className="field mono"
                    type="password"
                    placeholder={hasKey ? "replace key" : "paste API key"}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        const v = (e.target as HTMLInputElement).value.trim();
                        if (v) {
                          saveKey(p.name, v);
                          (e.target as HTMLInputElement).value = "";
                        }
                      }
                    }}
                  />
                  <button
                    className="btn"
                    onClick={(e) => {
                      const input = (e.currentTarget.parentElement?.querySelector("input") ?? null) as HTMLInputElement | null;
                      const v = (input?.value ?? "").trim();
                      if (v && input) {
                        saveKey(p.name, v);
                        input.value = "";
                      }
                    }}
                  >
                    Save key
                  </button>
                  {hasKey && (
                    <button className="btn" onClick={() => removeKey(p.name)} title="Delete the Keychain entry for this provider">
                      Remove key
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })}
      <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
        <div className="settings-desc" style={{ marginBottom: 10 }}>
          Add a provider — it appears in the model picker, where its model list loads in the
          background (launch + every 5 min). Keys go to the macOS Keychain, never the config file.
        </div>
        <div className="provider-grid" style={{ paddingTop: 2 }}>
          <div className="provider-field" style={{ flex: "1 1 150px", minWidth: 140 }}>
            <span className="provider-field-label">Name</span>
            <input
              className="field"
              placeholder="e.g. OpenAI"
              value={draft.name}
              onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
            />
          </div>
          <div className="provider-field" style={{ flex: "2 1 240px" }}>
            <span className="provider-field-label">Base URL (API root)</span>
            <input
              className="field mono"
              placeholder="https://api.openai.com/v1"
              value={draft.endpoint}
              onChange={(e) => setDraft((d) => ({ ...d, endpoint: e.target.value }))}
            />
          </div>
          <div className="provider-field" style={{ flex: "1 1 160px", minWidth: 150 }}>
            <span className="provider-field-label">API format</span>
            <select
              className="field"
              value={draft.profile}
              onChange={(e) => setDraft((d) => ({ ...d, profile: e.target.value }))}
            >
              <option value="openai">OpenAI-compatible</option>
              <option value="anthropic">Anthropic</option>
              <option value="local">Local (Ollama)</option>
            </select>
          </div>
          <div className="provider-field" style={{ flex: "1 1 160px", minWidth: 150 }}>
            <span className="provider-field-label">Model (optional)</span>
            <input
              className="field mono"
              placeholder="e.g. my-model"
              value={draft.model}
              onChange={(e) => setDraft((d) => ({ ...d, model: e.target.value }))}
            />
          </div>
          <div className="provider-field" style={{ flex: "1 1 180px", minWidth: 170 }}>
            <span className="provider-field-label">API key (optional)</span>
            <input
              className="field mono"
              type="password"
              placeholder="stored in Keychain"
              value={draftKey}
              onChange={(e) => setDraftKey(e.target.value)}
            />
          </div>
        </div>
        <button
          className="btn"
          disabled={!draft.name.trim() || !draft.endpoint.trim()}
          onClick={addProvider}
        >
          Add provider
        </button>
      </div>
    </>
  );
}

// ── Usage ─────────────────────────────────────────────────────────

/** Model colors — one slot per model, wrapped around. CSS vars so the
 *  dark theme re-maps them (see styles.css --usage-c0..7). */
function usageColor(i: number): string {
  return `var(--usage-c${i % 8})`;
}

function fmtCost(c: number): string {
  if (c <= 0) return "$0";
  return c < 1 ? `$${c.toFixed(4)}` : `$${c.toFixed(2)}`;
}

/** Full token counts for stat cards (12,483) — compact fmtTk is for rows. */
function fmtTkFull(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

function fmtDay(day: string): string {
  const [y, m, d] = day.split("-").map(Number);
  if (!y || !m || !d) return day;
  return new Date(y, m - 1, d).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function fmtAgo(ts: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000) - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** Last-14-day stacked bar chart (prompt under output), zero-filled.
 *  Pure SVG — no chart library, scales with the pane. */
function UsageChart({ days }: { days: backend.UsageDayRow[] }) {
  const N = 14;
  const byDay = new Map(days.map((d) => [d.day, d]));
  const buckets: Array<backend.UsageDayRow | null> = [];
  const today = new Date();
  for (let i = N - 1; i >= 0; i--) {
    const d = new Date(today.getFullYear(), today.getMonth(), today.getDate() - i);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    buckets.push(byDay.get(key) ?? null);
  }
  const W = 700, H = 128, TOP = 6;
  const max = Math.max(1000, ...buckets.map((b) => (b ? b.pt + b.ct : 0)));
  const bw = W / N;
  return (
    <div>
      <svg className="usage-chart" viewBox={`0 0 ${W} ${H + TOP + 2}`} preserveAspectRatio="none" role="img">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            className="usage-grid"
            x1={0}
            x2={W}
            y1={TOP + H * (1 - f)}
            y2={TOP + H * (1 - f)}
          />
        ))}
        {buckets.map((b, i) => {
          const x = i * bw + bw * 0.18;
          const w = Math.max(2, bw * 0.64);
          const hp = b ? (b.pt / max) * H : 0;
          const hc = b ? (b.ct / max) * H : 0;
          const tot = b ? b.pt + b.ct : 0;
          const label = `${fmtDay(b?.day ?? "")}${b ? ` — ${fmtTkFull(tot)} tokens · ${b.turns} turn${b.turns === 1 ? "" : "s"}` : " — no turns"}`;
          return (
            <g key={i}>
              <title>{label}</title>
              {b && tot > 0 ? (
                <>
                  <rect className="usage-bar-pt" x={x} y={TOP + H - hp} width={w} height={Math.max(hp, 1)} rx={2} />
                  {hc > 0 && (
                    <rect className="usage-bar-ct" x={x} y={TOP + H - hp - hc} width={w} height={hc} rx={2} />
                  )}
                </>
              ) : (
                <rect className="usage-bar-zero" x={x} y={TOP + H - 2} width={w} height={2} />
              )}
            </g>
          );
        })}
      </svg>
      <div className="usage-chart-days">
        {buckets.map((b, i) => (
          <span key={i}>{i % 3 === 0 || i === N - 1 ? fmtDay(b?.day ?? "") : ""}</span>
        ))}
      </div>
    </div>
  );
}

function UsagePane({ projectDir }: Props) {
  const [report, setReport] = useState<backend.UsageReport | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const pull = useCallback(() => {
    backend
      .usageReport()
      .then((r) => {
        if (r) setReport(r);
        else setUnavailable(true);
      })
      .catch(() => setUnavailable(true));
  }, []);
  useEffect(() => {
    pull();
  }, [pull]);
  // Refresh while the pane is open — usage records land after each turn.
  useEffect(() => {
    const iv = setInterval(pull, 20000);
    return () => clearInterval(iv);
  }, [pull]);

  if (unavailable) {
    return (
      <div className="settings-desc" style={{ padding: "8px 0" }}>
        Engine not connected — usage is computed by the engine from the workspace session store.
      </div>
    );
  }
  if (!report) {
    return <div className="settings-desc" style={{ padding: "8px 0" }}>Loading usage…</div>;
  }
  const t = report.totals;
  if (t.turns === 0) {
    return (
      <div className="usage-empty">
        <div className="usage-empty-title">No usage recorded yet</div>
        <div className="usage-empty-desc">
          Every turn the engine serves in this workspace logs its real token counts, cache
          hits and priced spend here. Send a message to start tracking.
        </div>
      </div>
    );
  }

  const stats: Array<{ label: string; value: string; title?: string }> = [
    { label: "Total tokens", value: fmtTk(t.pt + t.ct), title: `${fmtTkFull(t.pt + t.ct)} tokens — ↑${fmtTkFull(t.pt)} prompt · ↓${fmtTkFull(t.ct)} output` },
    { label: "Turns", value: String(t.turns), title: "Engine turns that reported usage" },
    { label: "Sessions", value: String(t.sessions), title: "Chats with recorded usage in this workspace" },
    { label: "Cache read", value: fmtTk(t.cr), title: `${fmtTkFull(t.cr)} prompt tokens served from the provider's prefix cache` },
    { label: "Est. spend", value: fmtCost(t.cost), title: "Pricing-table estimate, not a bill" },
  ];

  return (
    <>
      <div className="usage-stats">
        {stats.map((s) => (
          <div className="usage-stat" key={s.label} title={s.title}>
            <span className="usage-stat-val">{s.value}</span>
            <span className="usage-stat-label">{s.label}</span>
          </div>
        ))}
      </div>

      <SectionTitle>Daily activity</SectionTitle>
      <div className="usage-card">
        <div className="usage-legend">
          <span><i className="usage-swatch swatch-pt" /> prompt</span>
          <span><i className="usage-swatch swatch-ct" /> output</span>
        </div>
        <UsageChart days={report.days} />
      </div>

      {report.models.length > 0 && (
        <>
          <SectionTitle>Models</SectionTitle>
          <div className="usage-card">
            {report.models.map((m, i) => {
              const total = t.pt + t.ct;
              const share = total > 0 ? (m.pt + m.ct) / total : 0;
              return (
                <div className="usage-model" key={m.model}>
                  <span className="usage-dot" style={{ background: usageColor(i) }} />
                  <div className="usage-model-main">
                    <div className="usage-model-top">
                      <span className="mono usage-model-name" title={m.model}>{m.model}</span>
                      <span className="usage-model-share">{(share * 100).toFixed(1)}%</span>
                    </div>
                    <div className="usage-model-bar">
                      <span style={{ width: `${Math.max(share * 100, 1.5)}%`, background: usageColor(i) }} />
                    </div>
                    <div className="usage-model-meta">
                      {m.turns} turn{m.turns === 1 ? "" : "s"}
                      {" · "}↑{fmtTk(m.pt)} ↓{fmtTk(m.ct)}
                      {m.cr > 0 && ` · cache ${fmtTk(m.cr)}`}
                      {m.cost > 0 && ` · ${fmtCost(m.cost)}`}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {report.sessions.length > 0 && (
        <>
          <SectionTitle>Sessions</SectionTitle>
          <div className="usage-card">
            {report.sessions.slice(0, 8).map((s) => (
              <div className="usage-session" key={s.id}>
                <span className="mono usage-session-id" title={s.id}>{s.id.slice(0, 8)}</span>
                <span className="mono usage-session-model">{s.model || "—"}</span>
                <span className="usage-session-when">{fmtAgo(s.last_seen)}</span>
                <span className="usage-session-nums">
                  {s.turns} turn{s.turns === 1 ? "" : "s"} · {fmtTk(s.pt + s.ct)} tk
                  {s.cost > 0 && ` · ${fmtCost(s.cost)}`}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="settings-desc" style={{ padding: "12px 0 4px" }}>
        Real counts from every turn the engine served in this workspace
        {report.days.length > 0 && <> since {fmtDay(report.days[0].day)}</>}. Spend is the
        pricing-table estimate. Store: <span className="mono">{projectDir ?? "not set"}</span>
      </div>
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
