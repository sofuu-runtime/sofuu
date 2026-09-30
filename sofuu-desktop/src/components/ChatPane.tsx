// ChatPane.tsx — the message list: user bubbles, assistant markdown,
// live activity area (animated phase label + streaming thinking text +
// tool timeline), usage footer, streaming cursor. Also renders read-only
// transcripts when a past session is being viewed.

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../lib/events";
import type { SessionPreview } from "../lib/store";
import { ToolTimeline, PhaseSweep, Icon } from "./ToolTimeline";

/** Allow only safe link targets in model-rendered markdown. javascript:,
 *  data:, file:, vbscript: never render as clickable (XSS → Tauri IPC).
 *  https/http/mailto pass through; anything else renders as plain text. */
function safeUrl(url: string): string | undefined {
  const low = url.trim().toLowerCase();
  if (
    low.startsWith("https://") ||
    low.startsWith("http://") ||
    low.startsWith("mailto:")
  ) {
    return url;
  }
  return undefined;
}

/** Thinking block, in the reference-app style: the header row is always
 *  there, but the reasoning text only appears when the user clicks it.
 *  Opening pins the view to the END of the text (terminal tail); hovering
 *  the open text enables scrolling so the user can read the whole process;
 *  leaving collapses the view back to the tail. Clicking the header again
 *  hides the text. While streaming, the open view auto-follows the end. */
function ThinkingBlock({ text, live }: { text: string; live?: boolean }) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(false);
  const [hover, setHover] = useState(false);
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  // Long text scrolls inside the cap; short text just sits there.
  const scrollable = text.length > 600;

  const pinToEnd = () => {
    const el = bodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  // Keep the END of the text in view while open: right after expanding,
  // on every new chunk while streaming, and again on mouse-leave (the
  // reference-app "collapse back to the tail"). Hovering the body stops
  // the pinning so the user can scroll freely.
  useEffect(() => {
    if (open && !hover) pinToEnd();
  }, [text, hover, open]);

  return (
    <div className="thinking">
      <button
        className={"tl-row" + (open ? " open" : "") + (scrollable ? " scrollable" : "")}
        onClick={() => setOpen((o) => !o)}
        title={open ? "Click to hide the thinking process" : "Click to show the thinking process"}
      >
        <span className="tl-chevron">›</span>
        <Icon id="think" />
        <span className="tl-cell-title">{live ? "Thinking" : "Thought"}</span>
        <span className="tl-badge">
          {live && "· streaming"}
          {!live && words > 0 && `· ${words} ${words === 1 ? "word" : "words"}`}
        </span>
      </button>
      {open && (
        <div
          ref={bodyRef}
          className={"thinking-body" + (scrollable ? " scrollable" : "") + (hover ? " hover" : "")}
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
        >
          {text}
        </div>
      )}
    </div>
  );
}

function usageLine(m: ChatMessage): string | null {
  const u = m.usage;
  if (!u) return null;
  const bits: string[] = [];
  if (u.servedBy) bits.push(u.servedBy);
  if (u.promptTokens != null) bits.push(`${u.promptTokens.toLocaleString()} in`);
  if (u.completionTokens != null) bits.push(`${u.completionTokens.toLocaleString()} out`);
  if (u.costUsd != null && u.costUsd > 0) bits.push(`$${u.costUsd.toFixed(4)}`);
  if (u.steps != null && u.steps > 0) bits.push(`${u.steps} steps`);
  return bits.length ? bits.join(" · ") : null;
}

/** Read-only view of a past session's transcript (◷ history / sidebar). */
function PreviewPane({ preview, chatFontSize }: { preview: SessionPreview; chatFontSize: number }) {
  return (
    <div className="chat-pane" style={{ ["--chat-font-size" as string]: `${chatFontSize}px` }}>
      <div className="preview-note">
        Read-only transcript · {preview.label} · {preview.turns.length}{" "}
        {preview.turns.length === 1 ? "turn" : "turns"}
      </div>
      {preview.turns.length === 0 && (
        <div className="preview-note">No messages recorded in this chat.</div>
      )}
      {preview.turns.map((t, i) => (
        <div key={i} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="msg msg-user">
            <div className="bubble">{t.prompt}</div>
          </div>
          {t.answer && (
            <div className="msg msg-assistant">
              <div className="md-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={safeUrl}>{t.answer}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function ChatPane({
  messages,
  chatFontSize,
  phase,
  preview,
}: {
  messages: ChatMessage[];
  chatFontSize: number;
  phase: string | null;
  preview?: SessionPreview | null;
}) {
  if (preview) return <PreviewPane preview={preview} chatFontSize={chatFontSize} />;
  return (
    <div className="chat-pane" style={{ ["--chat-font-size" as string]: `${chatFontSize}px` }}>
      {messages.map((m) =>
        m.role === "system" ? (
          // Slash-command output and engine notices: dim, compact, plain.
          <div className="msg msg-system" key={m.id}>
            {m.content.split("\n").map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </div>
        ) : m.role === "user" ? (
          <div className="msg msg-user" key={m.id}>
            <div className="bubble">{m.content}</div>
          </div>
        ) : (
          <div className="msg msg-assistant" key={m.id}>
            {m.status === "streaming" && phase && (
              <div className="phase-line">
                <PhaseSweep label={phase} />
              </div>
            )}
            {/* Legacy only: new turns render reasoning as "Thought"
                timeline rows (ToolTimeline); this block survives for
                cached transcripts that predate that change. */}
            {m.thinking && <ThinkingBlock text={m.thinking} live={m.status === "streaming"} />}

            <ToolTimeline entries={m.timeline} />

            {(m.content || m.status === "streaming") && (
              <div className="md-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={safeUrl}>{m.content}</ReactMarkdown>
                {m.status === "streaming" && <span className="cursor" />}
              </div>
            )}

            {m.status === "error" && m.error && <div className="msg-error">{m.error}</div>}

            {m.status === "done" && usageLine(m) && <div className="msg-usage">{usageLine(m)}</div>}
          </div>
        )
      )}
    </div>
  );
}
