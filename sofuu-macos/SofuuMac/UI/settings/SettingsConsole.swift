import SwiftUI
import AppKit

/// The engine console: a live tail of the app's captured stderr — the Rust
/// engine's log channel plus app-side error lines (CrashLog). Read-only
/// ring of the last 2000 lines, refreshed 1×/s while open, mirrored on
/// disk at ~/Library/Application Support/SofuuMac/engine-console.log.
struct ConsolePane: View {
    let store: AppStore  // pane-signature symmetry (unused)

    @Environment(\.sofuuColors) private var c
    @State private var rows: [String] = []
    @State private var wrap = false
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 14) {
            SectionCard(label: "Engine console", footer:
                "Live stderr of the engine process and app errors. Also mirrored to "
                + "~/Library/Application Support/SofuuMac/engine-console.log") {
                HStack(spacing: 8) {
                    SmallButton(label: "Copy all", bg: c.bgRaised, fg: c.text) {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(rows.joined(separator: "\n"), forType: .string)
                    }
                    SmallButton(label: "Clear", bg: c.bgRaised, fg: c.text) {
                        EngineConsole.ring.clear(); rows = []
                    }
                    Spacer()
                    Toggle("Wrap lines", isOn: $wrap)
                        .toggleStyle(.checkbox)
                        .font(.system(size: 11))
                        .foregroundStyle(c.textMuted)
                    Text("\(rows.count) lines")
                        .font(.system(size: 11))
                        .foregroundStyle(c.textFaint)
                }
                .padding(.bottom, 8)

                if rows.isEmpty {
                    Text("Nothing yet — the console fills as the engine logs.")
                        .font(.system(size: 12))
                        .foregroundStyle(c.textFaint)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 40)
                } else {
                    ConsoleLogList(rows: rows, wrap: wrap)
                        .frame(height: 440)
                        .background(c.bgRaised, in: RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(c.separator, lineWidth: 1))
                }
            }
        }
        .onReceive(timer) { _ in
            let next = EngineConsole.ring.snapshot()
            if next != rows { rows = next }
        }
        .onAppear { rows = EngineConsole.ring.snapshot() }
    }
}

/// The log body. Reversed (newest at the top) so no scroll chasing; the
/// refresh keeps the scroll position stable.
private struct ConsoleLogList: View {
    let rows: [String]
    let wrap: Bool

    var body: some View {
        let reversed = rows.reversed()
        ScrollView(showsIndicators: true) {
            LazyVStack(alignment: .leading, spacing: 1) {
                ForEach(Array(reversed.enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.primary.opacity(0.85))
                        .lineLimit(wrap ? nil : 1)
                }
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
