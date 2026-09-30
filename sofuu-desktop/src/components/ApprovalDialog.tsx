// ApprovalDialog.tsx — the permission gate UI (PLAN-DESKTOP C/D): a
// floating card above the composer for each pending risky tool call.
// Allow / Deny / Always-allow-this-session resolve the promise in chat.js
// through the poke channel.
//
// P2: write_file/edit_file get a real red/green diff (DiffView) instead
// of escaped-string JSON — the point of a gate is seeing what you approve.

import type { Approval } from "../lib/events";
import { DiffView } from "./DiffView";

interface Props {
  approval: Approval;
  onResolve: (id: string, allow: boolean, always: boolean) => void;
}

function formatArgs(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

/** A short one-line summary of which file the edit touches. */
function pathOf(args: Record<string, unknown>): string | null {
  const p = args.path ?? args.file ?? args.file_path;
  return typeof p === "string" && p ? p : null;
}

function DiffSection({ approval }: { approval: Approval }) {
  const a = approval.args;
  const path = pathOf(a);
  if (approval.tool === "edit_file" &&
      typeof a.old_string === "string" && typeof a.new_string === "string") {
    return (
      <>
        {path && <div className="approval-path">{path}</div>}
        <DiffView oldText={a.old_string} newText={a.new_string} />
        {a.replace_all === true && <div className="approval-note">Replaces all occurrences</div>}
      </>
    );
  }
  if (approval.tool === "write_file" && typeof a.content === "string") {
    return (
      <>
        {path && <div className="approval-path">{path}</div>}
        <DiffView newText={a.content} />
      </>
    );
  }
  return null;
}

export function ApprovalDialog({ approval, onResolve }: Props) {
  const diffable =
    approval.tool === "edit_file" || approval.tool === "write_file";
  const isBash = approval.tool === "bash";
  const cmd =
    isBash && typeof approval.args.command === "string"
      ? approval.args.command
      : null;
  return (
    <div className="approval-float">
      <div className="approval-card">
        <div className="approval-title">
          Approve <span className="approval-tool">{approval.tool}</span>?
        </div>
        {diffable ? (
          <DiffSection approval={approval} />
        ) : cmd !== null ? (
          <pre className="approval-cmd">{cmd}</pre>
        ) : (
          <div className="approval-args">{formatArgs(approval.args)}</div>
        )}
        <div className="approval-actions">
          <button className="btn btn-danger" onClick={() => onResolve(approval.id, false, false)}>
            Deny
          </button>
          <button
            className="btn"
            onClick={() => onResolve(approval.id, true, true)}
            title={isBash ? "Auto-run this exact command for the rest of the session" : "Auto-approve this tool for the rest of the session"}
          >
            {isBash ? "Always allow this command" : "Always allow this session"}
          </button>
          <button className="btn btn-primary" onClick={() => onResolve(approval.id, true, false)}>
            Allow
          </button>
        </div>
      </div>
    </div>
  );
}
