import SwiftUI

// MARK: - Usage (dashboard with the 14-day chart)

struct UsagePane: View {
    @ObservedObject var store: AppStore
    @Environment(\.sofuuColors) private var c

    @State private var usage: JSONObject?
    @State private var pollTimer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let totals = usage?.optObject("totals") {
                SectionCard(label: "TOTALS",
                            footer: "Per-turn usage across the workspace's sessions, computed inside the engine.") {
                    HStack(spacing: 10) {
                        statCard("Total tokens",
                                 "\(grouped(Int(totals.optDouble("pt")))) in\n\(grouped(Int(totals.optDouble("ct")))) out",
                                 highlight: Int(totals.optDouble("cr")) > 0 ? "cache \(grouped(Int(totals.optDouble("cr"))))" : nil)
                        statCard("Turns", grouped(Int(totals.optDouble("turns"))), highlight: nil)
                        statCard("Sessions", grouped(Int(totals.optDouble("sessions"))), highlight: nil)
                        statCard("Est. spend", String(format: "$%.4f", totals.optDouble("cost")), highlight: nil)
                    }
                }
            } else {
                SectionCard(label: "TOTALS") {
                    Text(usage == nil ? "Reading usage from the engine…" : "No usage yet — send a message first.")
                        .font(.system(size: 12)).foregroundStyle(c.textFaint)
                }
            }

            if let days = usage?.optArray("days") as? [JSONObject], !days.isEmpty {
                SectionCard(label: "LAST 14 DAYS") {
                    UsageChart(days: days)
                        .frame(height: 120)
                }
            }

            if let models = usage?.optArray("models") as? [JSONObject], !models.isEmpty {
                SectionCard(label: "MODELS") {
                    ForEach(Array(models.prefix(10).enumerated()), id: \.offset) { _, m in
                        mono("\(m.optString("model"))  ·  \(m.optInt("turns")) turns  ·  \(grouped(Int(m.optDouble("pt")))) in · \(grouped(Int(m.optDouble("ct")))) out  ·  $\(String(format: "%.4f", m.optDouble("cost")))")
                    }
                }
            }

            if let sess = usage?.optArray("sessions") as? [JSONObject], !sess.isEmpty {
                SectionCard(label: "SESSIONS") {
                    ForEach(Array(sess.prefix(8).enumerated()), id: \.offset) { _, s in
                        let name = s.optString("task").nilIfEmpty ?? s.optString("id")
                        mono("\(String(name.prefix(24)))  ·  \(s.optString("model"))  ·  \(s.optInt("turns")) turns  ·  $\(String(format: "%.4f", s.optDouble("cost")))")
                    }
                }
            }
        }
        .task { await load() }
        .onDisappear { pollTimer?.invalidate() }
        .onAppear {
            // The desktop polls every 20 s while the pane is open.
            pollTimer = Timer.scheduledTimer(withTimeInterval: 20, repeats: true) { _ in
                Task { @MainActor in await load() }
            }
        }
    }

    private func load() async {
        usage = await store.engine.usageReport()
    }

    private func statCard(_ label: String, _ value: String, highlight: String?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.system(size: 10, weight: .medium)).foregroundStyle(c.textFaint)
            Text(value)
                .font(.system(size: 14, design: .monospaced)).foregroundStyle(c.text)
                .lineSpacing(1)
            if let highlight {
                Text(highlight).font(.system(size: 10)).foregroundStyle(c.accent)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(c.bgCard, in: RoundedRectangle(cornerRadius: 9))
    }

    private func mono(_ s: String) -> some View {
        Text(s).font(.system(size: 11, design: .monospaced))
            .foregroundStyle(c.textMuted)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 2)
    }
}

/// The desktop Usage pane's 14-day stacked bar chart (prompt under output,
/// zero-filled days), hand-drawn — no Charts dependency.
private struct UsageChart: View {
    let days: [JSONObject]

    @Environment(\.sofuuColors) private var c

    /// Zero-filled 14-day window ending today. Engine days are keyed
    /// "YYYY-MM-DD".
    private var series: [(day: String, pt: Double, ct: Double)] {
        struct Day { var pt = 0.0; var ct = 0.0 }
        var byDay: [String: Day] = [:]
        for d in days {
            byDay[d.optString("day")] = Day(pt: d.optDouble("pt"), ct: d.optDouble("ct"))
        }
        var out: [(String, Double, Double)] = []
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        let cal = Calendar.current
        for i in stride(from: 13, through: 0, by: -1) {
            let date = cal.date(byAdding: .day, value: -i, to: cal.startOfDay(for: Date()))!
            let key = f.string(from: date)
            let d = byDay[key] ?? Day()
            out.append((key, d.pt, d.ct))
        }
        return out
    }

    var body: some View {
        let data = series
        let maxTotal = max(1, data.map { $0.pt + $0.ct }.max() ?? 1)
        let dayLabelsEvery = 3
        GeometryReader { geo in
            VStack(alignment: .leading, spacing: 4) {
                // Grid lines at 25/50/75%.
                ZStack(alignment: .bottomLeading) {
                    ForEach(1..<4) { i in
                        Rectangle().fill(c.separator)
                            .frame(height: 1)
                            .offset(y: -geo.size.height * CGFloat(i) / 4 - 18)
                    }
                    HStack(alignment: .bottom, spacing: 2) {
                        ForEach(Array(data.enumerated()), id: \.offset) { i, d in
                            VStack(spacing: 1) {
                                ZStack(alignment: .bottom) {
                                    VStack(spacing: 0) {
                                        Rectangle().fill(c.accent).frame(height: bar(d.ct, of: maxTotal, height: geo.size.height - 18))
                                        Rectangle().fill(c.accent.opacity(0.45)).frame(height: bar(d.pt, of: maxTotal, height: geo.size.height - 18))
                                    }
                                }
                                .frame(maxHeight: .infinity, alignment: .bottom)
                                Text(i % dayLabelsEvery == 0 ? String(d.day.suffix(2)) : " ")
                                    .font(.system(size: 8, design: .monospaced))
                                    .foregroundStyle(c.textFaint)
                                    .frame(height: 12)
                            }
                            .frame(maxWidth: .infinity)
                            .help(toolTip(d))
                        }
                    }
                }
                HStack(spacing: 12) {
                    swatch(c.accent, "completion tokens")
                    swatch(c.accent.opacity(0.45), "prompt tokens")
                    Spacer()
                }
                .font(.system(size: 9))
                .foregroundStyle(c.textFaint)
            }
        }
    }

    private func bar(_ v: Double, of maxTotal: Double, height: CGFloat) -> CGFloat {
        guard maxTotal > 0, v > 0 else { return 0 }
        return max(1, CGFloat(v / maxTotal) * height)
    }

    private func swatch(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 3) {
            Rectangle().fill(color).frame(width: 8, height: 8)
            Text(label)
        }
    }

    private func toolTip(_ d: (day: String, pt: Double, ct: Double)) -> String {
        "\(d.day)\n\(grouped(Int(d.pt))) in · \(grouped(Int(d.ct))) out"
    }
}

// MARK: - Shortcuts

/// Renders the shared registry — the single source of truth stays
/// App/Shortcuts.swift, in lockstep with the menu accelerators.
struct ShortcutsPane: View {
    @Environment(\.sofuuColors) private var c

    var body: some View {
        VStack(spacing: 14) {
            ForEach(SHORTCUT_GROUPS, id: \.self) { g in
                SectionCard(label: g.uppercased()) {
                    ForEach(SHORTCUTS.filter { $0.group == g }) { s in
                        HStack {
                            Text(s.label).font(.system(size: 13)).foregroundStyle(c.text)
                            Spacer()
                            HStack(spacing: 3) {
                                ForEach(s.keys, id: \.self) { k in
                                    Text(k)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundStyle(c.textMuted)
                                        .padding(.horizontal, 6).padding(.vertical, 2)
                                        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 4))
                                }
                            }
                        }
                        .padding(.vertical, 3)
                    }
                }
            }
        }
    }
}
