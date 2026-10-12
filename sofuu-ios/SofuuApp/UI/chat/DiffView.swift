import SwiftUI

private let maxDiffLines = 400

struct DiffRow {
    let kind: String // ctx | del | add
    let text: String
}

/// LCS line diff (the desktop's DiffView algorithm, ported through Android).
func diffLines(_ oldText: String, _ newText: String) -> [DiffRow] {
    let a = oldText.components(separatedBy: "\n")
    let b = newText.components(separatedBy: "\n")
    if a.count > maxDiffLines || b.count > maxDiffLines {
        return [
            DiffRow(kind: "del", text: "… \(a.count) old lines (too large to diff inline) …"),
            DiffRow(kind: "add", text: "… \(b.count) new lines (too large to diff inline) …"),
        ]
    }
    // dp[i][j] = LCS length of a[i...] and b[j...]
    var dp = Array(repeating: Array(repeating: 0, count: b.count + 1), count: a.count + 1)
    for i in stride(from: a.count - 1, through: 0, by: -1) {
        for j in stride(from: b.count - 1, through: 0, by: -1) {
            dp[i][j] = a[i] == b[j] ? dp[i + 1][j + 1] + 1 : max(dp[i + 1][j], dp[i][j + 1])
        }
    }
    var rows: [DiffRow] = []
    var i = 0, j = 0
    while i < a.count && j < b.count {
        if a[i] == b[j] {
            rows.append(DiffRow(kind: "ctx", text: a[i])); i += 1; j += 1
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            rows.append(DiffRow(kind: "del", text: a[i])); i += 1
        } else {
            rows.append(DiffRow(kind: "add", text: b[j])); j += 1
        }
    }
    while i < a.count { rows.append(DiffRow(kind: "del", text: a[i])); i += 1 }
    while j < b.count { rows.append(DiffRow(kind: "add", text: b[j])); j += 1 }
    return rows
}

/// Red/green line diff for write/edit tool results (desktop parity).
struct DiffView: View {
    let oldText: String
    let newText: String

    @Environment(\.sofuuColors) private var c

    var body: some View {
        let rows = diffLines(oldText, newText)
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                row(r)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .frame(maxHeight: 320, alignment: .top)
        .clipped()
    }

    @ViewBuilder
    private func row(_ r: DiffRow) -> some View {
        let bg: Color = r.kind == "add" ? c.diffAddBg : (r.kind == "del" ? c.diffDelBg : .clear)
        let fg: Color = r.kind == "add" ? c.diffAddFg : (r.kind == "del" ? c.diffDelFg : c.textMuted)
        let marker = r.kind == "add" ? "+" : (r.kind == "del" ? "−" : " ")
        HStack(alignment: .top, spacing: 0) {
            Text(marker).frame(width: 16, alignment: .leading)
            Text(r.text.isEmpty ? " " : r.text)
        }
        .font(.system(size: 12, design: .monospaced))
        .foregroundStyle(fg)
        .padding(.vertical, 1)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(bg)
    }
}
