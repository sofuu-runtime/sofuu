// errlog.ts — forward webview JS errors to the Rust stderr log.
// Release builds have no visible console; without this a single render
// error shows up as a permanently blank window. Imported FIRST in
// main.tsx so the hook exists before any other module code runs.
import { invoke } from "@tauri-apps/api/core";

function report(kind: string, msg: unknown): void {
  try {
    const err = msg as {
      message?: string;
      filename?: string;
      lineno?: number;
      error?: Error;
    };
    const line =
      kind +
      ": " +
      String((err && (err.message as string)) || msg).slice(0, 500) +
      " @ " +
      String((err && err.filename) || "") +
      ":" +
      String((err && err.lineno) || 0) +
      " | stack: " +
      String((err && err.error && err.error.stack) || "").slice(0, 1200);
    invoke("frontend_log", { msg: line }).catch(() => {});
  } catch {
    /* never let the reporter throw */
  }
}

window.addEventListener("error", (e) => report("error", e));
window.addEventListener("unhandledrejection", (e) =>
  report("rejection", e.reason)
);
