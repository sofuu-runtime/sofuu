// DiffView.tsx — red/green line diff for write/edit approvals (P2).
// edit_file: old_string → new_string (removed − red, added + green).
// write_file: full content shown as added lines. Anything else falls back
// to the plain JSON args view in ApprovalDialog. A simple LCS keeps the
// diff honest without pulling a dependency.

const MAX_LINES = 400;

type DiffRow = { kind: "ctx" | "del" | "add"; text: string };

function lcsTable(a: string[], b: string[]): number[][] {
  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0)
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  return dp;
}

export function diffLines(oldText: string, newText: string): DiffRow[] {
  const a = oldText.split("\n");
  const b = newText.split("\n");
  if (a.length > MAX_LINES || b.length > MAX_LINES) {
    return [
      { kind: "del", text: `… ${a.length} old lines (too large to diff inline) …` },
      { kind: "add", text: `… ${b.length} new lines (too large to diff inline) …` },
    ];
  }
  const dp = lcsTable(a, b);
  const rows: DiffRow[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      rows.push({ kind: "ctx", text: a[i] });
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      rows.push({ kind: "del", text: a[i++] });
    } else {
      rows.push({ kind: "add", text: b[j++] });
    }
  }
  while (i < a.length) rows.push({ kind: "del", text: a[i++] });
  while (j < b.length) rows.push({ kind: "add", text: b[j++] });
  return rows;
}

interface Props {
  oldText?: string;
  newText?: string;
}

export function DiffView({ oldText, newText }: Props) {
  const rows = diffLines(oldText ?? "", newText ?? "");
  let line = 1;
  return (
    <div className="diff-view">
      {rows.map((r, i) => {
        const n = r.kind === "add" ? undefined : line++;
        const label = r.kind === "add" ? "+" : r.kind === "del" ? "−" : " ";
        return (
          <div key={i} className={`diff-row diff-${r.kind}`}>
            <span className="diff-sign">{label}</span>
            <span className="diff-lno">{n ?? ""}</span>
            <span className="diff-text">{r.text || " "}</span>
          </div>
        );
      })}
    </div>
  );
}
