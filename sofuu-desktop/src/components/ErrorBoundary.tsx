// ErrorBoundary.tsx — last-resort crash guard for the React tree. A render
// error used to unmount everything into a blank window (the TopBar
// sessions.map crash class); with this, the failure is VISIBLE — message,
// stack, reload — instead of a silently dead app.
import React from "react";

interface State {
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Also reaches the Rust stderr log via errlog's window hook? No —
    // React catches render errors before window.onerror fires in some
    // paths, so report explicitly.
    try {
      import("@tauri-apps/api/core").then(({ invoke }) =>
        invoke("frontend_log", {
          msg: "render crash: " + String(error?.message || error) +
            " | " + String(info.componentStack || "").slice(0, 900),
        }).catch(() => {})
      );
    } catch {
      /* never throw from the boundary */
    }
  }

  render(): React.ReactNode {
    if (this.state.error) {
      return (
        <div
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            padding: 32,
            fontFamily: "ui-monospace, monospace",
            fontSize: 13,
            color: "#b3362b",
            textAlign: "center",
          }}
        >
          <div style={{ fontWeight: 600, color: "var(--text, #222)" }}>
            Sofuu hit a rendering error
          </div>
          <div style={{ maxWidth: 720, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {String(this.state.error?.message || this.state.error)}
          </div>
          <button
            className="btn btn-primary"
            onClick={() => {
              this.setState({ error: null });
              location.reload();
            }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
