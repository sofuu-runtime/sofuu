// ApprovalDialog.tsx — the permission gate UI (PLAN-DESKTOP C/D): a
// floating card above the composer for each pending risky tool call.
// Allow / Deny / Always-allow-this-session resolve the promise in chat.js
// through the poke channel.

import type { Approval } from "../lib/events";

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

export function ApprovalDialog({ approval, onResolve }: Props) {
  return (
    <div className="approval-float">
      <div className="approval-card">
        <div className="approval-title">
          Approve <span className="approval-tool">{approval.tool}</span> ?
        </div>
        <div className="approval-args">{formatArgs(approval.args)}</div>
        <div className="approval-actions">
          <button className="btn btn-danger" onClick={() => onResolve(approval.id, false, false)}>
            Deny
          </button>
          <button className="btn" onClick={() => onResolve(approval.id, true, true)}>
            Always allow this session
          </button>
          <button className="btn btn-primary" onClick={() => onResolve(approval.id, true, false)}>
            Allow
          </button>
        </div>
      </div>
    </div>
  );
}
