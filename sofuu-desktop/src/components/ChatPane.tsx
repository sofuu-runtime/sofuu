// ChatPane.tsx — the message list: user bubbles, assistant markdown with
// collapsible thinking, tool timeline, usage footer, streaming cursor.
// Also renders read-only transcripts when a past session is being viewed.

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../lib/events";
import type { SessionPreview } from "../lib/store";
import { ToolTimeline } from "./ToolTimeline";

function usageLine(m: ChatMessage): string | null {
  const u = m.usage;
  if (!u) return null;
  const bits: string[] = [];
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
        <div className="preview-note">This session has no recorded turns.</div>
      )}
      {preview.turns.map((t, i) => (
        <div key={i} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div className="msg msg-user">
            <div className="bubble">{t.prompt}</div>
          </div>
          {t.answer && (
            <div className="msg msg-assistant">
              <div className="md-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.answer}</ReactMarkdown>
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
  preview,
}: {
  messages: ChatMessage[];
  chatFontSize: number;
  preview?: SessionPreview | null;
}) {
  if (preview) return <PreviewPane preview={preview} chatFontSize={chatFontSize} />;
  return (
    <div className="chat-pane" style={{ ["--chat-font-size" as string]: `${chatFontSize}px` }}>
      {messages.map((m) =>
        m.role === "user" ? (
          <div className="msg msg-user" key={m.id}>
            <div className="bubble">{m.content}</div>
          </div>
        ) : (
          <div className="msg msg-assistant" key={m.id}>
            {m.thinking && (
              <details className="thinking">
                <summary>
                  <span style={{ fontSize: 12 }}>▸</span> Thinking
                </summary>
                <div className="thinking-body">{m.thinking}</div>
              </details>
            )}

            <ToolTimeline entries={m.timeline} />

            {(m.content || m.status === "streaming") && (
              <div className="md-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
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
