// Composer.tsx — the composer card: textarea on top, bottom bar with
// model/effort pills + icons left and the split send pill right.
// Home variant shows the mode pill below; chat variant hides it.
//
// Every control is live: the model/effort pills patch the shared config,
// ⎘ attaches files as @mentions (chat.js expands them), ≣ toggles the
// context layers, and the mode menu switches between agent (tools) and
// "just chat" (no tools) turns.

import { useEffect, useMemo, useRef, useState } from "react";
import * as backend from "../lib/backend";
import { SLASH_COMMANDS, type SlashCommand } from "../lib/commands";
import type { CtxMeter, SofuuConfig } from "../lib/events";
import type { PermissionProfile } from "../lib/store";
import { ContextRing } from "./ContextRing";
import { Icon } from "./Icon";
import { Popover, PopoverItem } from "./Popover";

const EFFORTS = ["off", "low", "medium", "high", "max"];

const PERMISSIONS: Array<{ id: PermissionProfile; label: string; hint: string }> = [
  { id: "full", label: "Full access", hint: "All tools run automatically" },
  { id: "edit", label: "Edit only", hint: "Read files & apply edits" },
  { id: "plan", label: "Plan mode", hint: "Plan & analyze, no changes" },
  { id: "prompt", label: "Ask me", hint: "Approve each tool execution" },
];

interface Props {
  variant: "home" | "chat";
  config: SofuuConfig | null;
  streaming: boolean;
  mode: "agent" | "chat";
  permissions: PermissionProfile;
  /** Live context-window meter (engine 'ctx' events) — the ring. */
  ctx: CtxMeter | null;
  onSubmit: (text: string, images?: string[]) => void;
  onCancel: () => void;
  onModeChange: (mode: "agent" | "chat") => void;
  onPermissionsChange: (profile: PermissionProfile) => void;
  onConfigPatch: (patch: Record<string, unknown>) => void;
  /** Open Settings (the model picker's "add a provider" escape hatch). */
  onOpenSettings: () => void;
}

type Pop = "model" | "effort" | "layers" | "mode" | "mode-caret" | "permissions" | null;

/** Cached model list state for the picker's second stage. */
interface ModelFetch {
  prov: string;
  loading: boolean;
  models: string[];
  error: string | null;
  note: string | null;
}

const EMPTY_FETCH: ModelFetch = { prov: "", loading: false, models: [], error: null, note: null };

export function Composer(props: Props) {
  const { variant, config, streaming, mode, permissions, ctx, onSubmit, onCancel, onModeChange, onPermissionsChange, onConfigPatch, onOpenSettings } = props;
  const [text, setText] = useState("");
  const [pop, setPop] = useState<Pop>(null);
  const ref = useRef<HTMLTextAreaElement>(null);

  // Slash-command palette: shows while the text is a bare "/query" token.
  // Escape dismisses until the text changes; arrows move the highlight,
  // Enter runs the highlighted command, Tab completes it into the input.
  const [slashClosed, setSlashClosed] = useState(false);
  const [slashIdx, setSlashIdx] = useState(0);
  const activeCmdRef = useRef<HTMLButtonElement>(null);
  const slashQuery = text.startsWith("/") ? text.slice(1).trim().toLowerCase() : null;
  const slashMatches = useMemo(() => {
    if (slashQuery === null) return [];
    return SLASH_COMMANDS.filter(
      (c) => !slashQuery || c.name.includes(slashQuery) || c.desc.toLowerCase().includes(slashQuery)
    );
  }, [slashQuery]);
  const slashOpen = slashQuery !== null && !slashClosed;

  useEffect(() => {
    activeCmdRef.current?.scrollIntoView({ block: "nearest" });
  }, [slashIdx, slashOpen]);

  const runCommand = (c: SlashCommand) => {
    if (streaming) return;
    onSubmit("/" + c.name);
    setText("");
    setSlashClosed(true);
    setSlashIdx(0);
    ref.current?.focus();
  };

  // Model picker: stage 1 (null) lists providers, a name shows that
  // provider's live models.
  const [modelProv, setModelProv] = useState<string | null>(null);
  const [modelFetch, setModelFetch] = useState<ModelFetch>(EMPTY_FETCH);
  const [manualModel, setManualModel] = useState("");

  const resetModelPop = () => {
    setModelProv(null);
    setModelFetch(EMPTY_FETCH);
    setManualModel("");
  };
  const closeModelPop = () => {
    setPop(null);
    resetModelPop();
  };

  const openProviderModels = async (name: string) => {
    setModelProv(name);
    setModelFetch({ prov: name, loading: false, models: [], error: null, note: null });
    // The list is pre-cached in the background — the click itself reads the
    // cache and never waits on the network. If the prefetch hasn't reached
    // this provider yet, show a light spinner and re-read a few times.
    const apply = (r: { models?: string[]; ok?: boolean; error?: string; note?: string }) =>
      setModelFetch((cur) =>
        cur.prov === name
          ? {
              prov: name,
              loading: false,
              models: r.models ?? [],
              error: r.ok === false ? r.error ?? "unreachable" : null,
              note: r.note ?? null,
            }
          : cur
      );
    try {
      const r = await backend.listModels(name);
      if (r.cached) {
        apply(r);
        return;
      }
      // no cache yet — the boot/interval refresh is presumably in flight
      setModelFetch({ prov: name, loading: true, models: [], error: null, note: null });
      const poll = (tries: number) => {
        if (tries <= 0) return;
        setTimeout(async () => {
          try {
            const r2 = await backend.listModels(name);
            if (r2.cached) apply(r2);
            else poll(tries - 1);
          } catch {
            poll(tries - 1);
          }
        }, 2000);
      };
      poll(4);
    } catch (e) {
      setModelFetch((cur) =>
        cur.prov === name ? { prov: name, loading: false, models: [], error: String(e), note: null } : cur
      );
    }
  };

  /** Choosing a model also writes it into the provider's entry — the
   *  flat config mirror is synced FROM the active entry, so patching only
   *  {model} would clobber it back to the entry's old value. */
  const pickModel = (prov: string, model: string) => {
    const nextProviders = (config?.providers ?? []).map((p) =>
      p.name === prov ? { ...p, model } : p
    );
    onConfigPatch({ providers: nextProviders, active: prov, model });
    closeModelPop();
  };

  const togglePop = (which: Pop) => setPop((p) => (p === which ? null : which));

  // Auto-grow the textarea within its cap.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 220) + "px";
  }, [text]);

  // Image attachments (P2 multimodal): paste or drag images into the
  // composer; they ride the turn as data URLs. No vision-flag gate — the
  // provider's own error surfaces honestly for non-vision models.
  const [images, setImages] = useState<Array<{ name: string; url: string }>>([]);
  const addImageFiles = (files: FileList | File[]) => {
    const list = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;
    for (const f of list.slice(0, 4)) {
      if (f.size > 6 * 1024 * 1024) continue; // ~6 MB cap per image
      const reader = new FileReader();
      reader.onload = () =>
        setImages((cur) =>
          cur.length >= 8 ? cur : [...cur, { name: f.name, url: String(reader.result) }]
        );
      reader.readAsDataURL(f);
    }
  };
  const onPaste = (e: React.ClipboardEvent) => {
    const files = e.clipboardData?.files;
    if (files && files.length && Array.from(files).some((f) => f.type.startsWith("image/"))) {
      e.preventDefault();
      addImageFiles(files);
    }
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer?.files?.length) addImageFiles(e.dataTransfer.files);
  };

  const submit = () => {
    if (streaming || (!text.trim() && images.length === 0)) return;
    onSubmit(text, images.length ? images.map((i) => i.url) : undefined);
    setText("");
    setImages([]);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (slashOpen) {
      const n = slashMatches.length;
      if (e.key === "ArrowDown" && n > 0) {
        e.preventDefault();
        setSlashIdx((i) => (i + 1) % n);
        return;
      }
      if (e.key === "ArrowUp" && n > 0) {
        e.preventDefault();
        setSlashIdx((i) => (i - 1 + n) % n);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setSlashClosed(true);
        return;
      }
      const c = slashMatches[slashIdx];
      if (e.key === "Tab" && c) {
        e.preventDefault();
        setText("/" + c.name + " ");
        setSlashClosed(true);
        return;
      }
      if (e.key === "Enter" && !e.shiftKey && c) {
        e.preventDefault();
        runCommand(c);
        return;
      }
    }
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

  const providers = config?.providers ?? [];
  const model = config?.model || (providers.length ? "Choose model" : "No model selected");
  /* "" is the stored form of Off (engine sends no effort param when falsy);
   * only a genuinely unset value falls back to the High default. */
  const effort = config?.effort === "" ? "off" : config?.effort || "high";
  const effortLabel = effort === "off" ? "Off" : effort.charAt(0).toUpperCase() + effort.slice(1);
  const rlmState = config?.rlm === "on" ? "on" : config?.rlm === "auto" ? "auto" : "off";

  const placeholder =
    variant === "home"
      ? "What would you like to build or explore?"
      : "Ask a follow-up, @mention files, or type / for commands...";

  const modeMenu = (which: Pop) => (
    <Popover open={pop === which} onClose={() => setPop(null)}>
      <div className="popover-title">Turn mode</div>
      <PopoverItem
        label="Agent"
        hint="Autonomous tools & file edits"
        selected={mode === "agent"}
        onClick={() => {
          onModeChange("agent");
          setPop(null);
        }}
      />
      <PopoverItem
        label="Chat only"
        hint="Conversation only, no tools"
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
      <div className="composer" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
        {images.length > 0 && (
          <div className="composer-attachments">
            {images.map((img, i) => (
              <span key={i} className="attachment-chip">
                <img src={img.url} alt={img.name} />
                <button
                  className="attachment-remove"
                  title="Remove"
                  onClick={() => setImages((cur) => cur.filter((_, k) => k !== i))}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <textarea
          ref={ref}
          rows={2}
          placeholder={placeholder}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setSlashClosed(false);
            setSlashIdx(0);
          }}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
        />
        {slashOpen && (
          <div className="slash-palette">
            <div className="slash-list" role="listbox" aria-label="Slash commands">
              {slashMatches.map((c, i) => (
                <button
                  key={c.name}
                  ref={i === slashIdx ? activeCmdRef : undefined}
                  role="option"
                  aria-selected={i === slashIdx}
                  className={"slash-item" + (i === slashIdx ? " active" : "")}
                  onMouseEnter={() => setSlashIdx(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => runCommand(c)}
                >
                  <span className="slash-name">/{c.name}</span>
                  {c.arg && <span className="slash-arg">{c.arg}</span>}
                  <span className="slash-desc">{c.desc}</span>
                </button>
              ))}
              {slashMatches.length === 0 && (
                <div className="slash-empty">No matching commands</div>
              )}
            </div>
            <div className="slash-foot">
              <span>↑↓ navigate</span>
              <span>↵ select</span>
              <span>tab complete</span>
              <span>esc dismiss</span>
            </div>
          </div>
        )}
        <div className="composer-bar">
          <span className="pop-anchor">
            <button className="pill" title="Model" onClick={() => togglePop("model")}>
              <Icon name="spark" size={16} className="spark" />
              {model}
              <Icon name="caret" size={12} className="caret" />
            </button>
            <Popover open={pop === "model"} onClose={closeModelPop} place="up" className="pop-model">
              <div className="popover-title">Model</div>
              {modelProv === null && providers.length > 0 && (
                <div className="popover-hint" style={{ padding: "0 10px 8px" }}>
                  Select a provider to view models
                </div>
              )}
              {modelProv === null ? (
                providers.length === 0 ? (
                  <>
                    <div className="popover-empty">
                      No model selected — connect a provider first.
                    </div>
                    <button
                      className="popover-item"
                      onClick={() => {
                        setPop(null);
                        onOpenSettings();
                      }}
                    >
                      <span>＋ Add provider</span>
                    </button>
                  </>
                ) : (
                  providers.map((p) => (
                    <PopoverItem
                      key={p.name}
                      label={p.name}
                      selected={p.name === config?.active}
                      onClick={() => openProviderModels(p.name)}
                    />
                  ))
                )
              ) : (
                <ModelList
                  prov={modelProv}
                  fetch={modelFetch}
                  savedModel={config?.providers?.find((p) => p.name === modelProv)?.model ?? ""}
                  active={config?.active ?? ""}
                  manual={manualModel}
                  setManual={setManualModel}
                  onBack={() => setModelProv(null)}
                  onPick={pickModel}
                />
              )}
            </Popover>
          </span>

          <span className="pop-anchor">
            <button className="pill" title="Thinking effort" onClick={() => togglePop("effort")}>
              {effortLabel}
              <Icon name="caret" size={12} className="caret" />
            </button>
            <Popover open={pop === "effort"} onClose={() => setPop(null)} place="up">
              <div className="popover-title">Thinking effort</div>
              {EFFORTS.map((e) => (
                <PopoverItem
                  key={e}
                  label={e === "off" ? "Off" : e.charAt(0).toUpperCase() + e.slice(1)}
                  selected={effort === e}
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
              <Popover open={pop === "layers"} onClose={() => setPop(null)} place="up" className="pop-layers">
                <div className="popover-title">Context layers</div>
                {(
                  [
                    ["brain", "Project memory", !!config?.brain, "Recall relevant saved memories"],
                    ["ml", "Smart context filters", config?.ml !== false, "Adaptive context relevance & compaction"],
                  ] as Array<[string, string, boolean, string]>
                ).map(([key, label, on, hint]) => (
                  <button
                    key={key}
                    className="popover-row layers-row"
                    title={
                      key === "brain"
                        ? "Project memory: recalls saved memories into the conversation context."
                        : "Smart context filters: freshness, compaction, relevance, and token allocation gates."
                    }
                    onClick={() => onConfigPatch({ [key]: !on })}
                  >
                    <span className="layers-main">
                      <span className="layers-label">{label}</span>
                      <span className={"toggle mini" + (on ? " on" : "")}>
                        <span className="knob" />
                      </span>
                    </span>
                    <span className="layers-hint">{hint}</span>
                  </button>
                ))}
                <button
                  className="popover-row layers-row"
                  title="RLM: route complex long-context turns through the local reasoning sandbox."
                  onClick={() => onConfigPatch({ rlm: rlmState === "on" ? "off" : "on" })}
                >
                  <span className="layers-main">
                    <span className="layers-label">Reasoning sandbox</span>
                    <span className={"toggle mini" + (rlmState === "on" ? " on" : "")}>
                      <span className="knob" />
                    </span>
                  </span>
                  <span className="layers-hint">{rlmState === "on" ? "on" : rlmState === "auto" ? "auto" : "off"}</span>
                </button>
              </Popover>
            </span>
            {/* Live context-window fill — right of the layers button. */}
            <ContextRing ctx={ctx} />
          </div>

          {streaming ? (
            <button className="send-arrow" title="Stop (⌘.)" onClick={onCancel}>
              <Icon name="stop" />
            </button>
          ) : variant === "home" ? (
            <span className="send-split-wrap pop-anchor">
              <div className="send-split">
                <button className="send-main" title="Send" onClick={submit}>
                  <Icon name="send" size={16} />
                </button>
                <button className="send-caret" title="Turn mode" onClick={() => togglePop("mode-caret")}>
                  <Icon name="caret" size={12} />
                </button>
              </div>
              {modeMenu("mode-caret")}
            </span>
          ) : (
            <button
              className="send-arrow"
              title="Send"
              onClick={submit}
              disabled={!text.trim() && images.length === 0}
            >
              <Icon name="send" />
            </button>
          )}
        </div>
      </div>

      {variant === "home" && (
        <div className="composer-below">
          <span className="pop-anchor">
            <button className="mode-pill" title="Turn mode" onClick={() => togglePop("mode")}>
              <Icon name={mode === "chat" ? "circle-fill" : "circle"} size={12} />
              {mode === "chat" ? "Chat only" : "Agent"}
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

/** Second stage of the model picker: one provider's models. Live list
 *  first (engine fetch), the provider's saved model flagged when it is
 *  not in the list, and manual entry when the endpoint yielded nothing
 *  (Anthropic has no public list; unreachable/custom endpoints). */
function ModelList({
  prov,
  fetch,
  savedModel,
  active,
  manual,
  setManual,
  onBack,
  onPick,
}: {
  prov: string;
  fetch: ModelFetch;
  savedModel: string;
  active: string;
  manual: string;
  setManual: (m: string) => void;
  onBack: () => void;
  onPick: (prov: string, model: string) => void;
}) {
  const saved = savedModel.trim();
  const isActive = prov === active;

  // Saved model first (marked), then the cached live list, deduped.
  const rows: Array<{ model: string; hint?: string }> = [];
  if (saved) rows.push({ model: saved, hint: isActive ? "current" : "saved" });
  for (const m of fetch.models) {
    if (!rows.some((r) => r.model === m)) rows.push({ model: m });
  }
  const listEmpty = fetch.models.length === 0;

  // Search filters by case-insensitive substring.
  const [q, setQ] = useState("");
  const query = q.trim().toLowerCase();
  const filtered = rows.filter((r) => !query || r.model.toLowerCase().includes(query));

  return (
    <>
      <button className="popover-item back-row" onClick={onBack}>
        <span>‹ Providers</span>
      </button>
      <div className="popover-title">
        {prov}
        {listEmpty && !fetch.loading
          ? " · no model list"
          : query
          ? ` · ${filtered.length} of ${rows.length}`
          : ` · ${rows.length} models`}
      </div>
      {!fetch.loading && rows.length > 0 && (
        <input
          className="field mono model-search"
          placeholder="Search models…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      )}
      {fetch.loading ? (
        <div className="popover-empty">Loading models…</div>
      ) : (
        <>
          {filtered.map((r) => {
            const isSel = r.model === saved && isActive;
            return (
              <button
                key={r.model}
                className={"popover-item model-item" + (isSel ? " selected" : "")}
                onClick={() => onPick(prov, r.model)}
              >
                <span className="model-name">{r.model}</span>
                {r.hint && <span className="popover-hint">{r.hint}</span>}
                {isSel && <span className="popover-check">✓</span>}
              </button>
            );
          })}
          {fetch.error && <div className="popover-empty">{fetch.error}</div>}
          {rows.length > 0 && filtered.length === 0 && (
            <div className="popover-empty">No model matches "{q.trim()}"</div>
          )}
          {listEmpty && (
            <form
              className="model-manual"
              onSubmit={(e) => {
                e.preventDefault();
                if (manual.trim()) onPick(prov, manual.trim());
              }}
            >
              <input
                className="field mono"
                placeholder="Type a model name…"
                value={manual}
                onChange={(e) => setManual(e.target.value)}
                autoFocus
              />
              <button className="btn" type="submit" disabled={!manual.trim()}>
                Use
              </button>
            </form>
          )}
        </>
      )}
    </>
  );
}
