import AppKit
import SwiftUI

/// Floating panels for hover tooltips, the completion list, and the
/// document outline. Non-activating so typing continues uninterrupted.
enum IDEHUD {
    static var hoverPanel: NSPanel?
    static var completionPanel: NSPanel?
    static var outlinePanel: NSPanel?

    // MARK: - Hover

    static func showHover(_ text: String, at anchor: NSRect) {
        // Size to the content: one line must not open a 160pt void.
        let lines = text.components(separatedBy: "\n").prefix(10)
        let contentHeight = CGFloat(lines.count) * 16 + 20
        let frame = hoveringRect(anchor: anchor, contentHeight: contentHeight)
        let view = NSHostingView(rootView:
            ScrollView { Text(text).font(.system(size: 12)) .frame(maxWidth: 420, alignment: .leading).padding(10) }
                .background(CompositedTheme.surface)
        )
        present(&hoverPanel, content: view, frame: frame)
    }

    static func hideHover() {
        hoverPanel?.orderOut(nil)
        hoverPanel = nil
    }

    // MARK: - Completion

    static func showCompletions(_ model: GutterModel, anchor: NSRect) {
        let frame = completionRect(anchor: anchor)
        let hosting = NSHostingView(rootView:
            CompletionList(model: model)
                .frame(width: 340, height: min(260, CGFloat((model.completions?.count ?? 1)) * 24 + 12))
                .background(CompositedTheme.surface)
        )
        present(&completionPanel, content: hosting, frame: frame)
    }

    static func hideCompletions() {
        completionPanel?.orderOut(nil)
        completionPanel = nil
    }

    // MARK: - Outline

    /// Toggle: visible → hide; hidden → show with the fresh nodes.
    @discardableResult
    static func showOutline(_ nodes: [LSPWire.OutlineNode], onPick: @escaping (LSPWire.OutlineNode) -> Void) -> Bool {
        if outlinePanel?.isVisible == true {
            hideOutline()
            return false
        }
        let hosting = NSHostingView(rootView:
            OutlineList(nodes: nodes, onPick: onPick)
                .frame(width: 320, height: 320)
                .background(CompositedTheme.surface)
        )
        let frame = NSRect(x: 60, y: 120, width: 320, height: 320)
        present(&outlinePanel, content: hosting, frame: frame)
        return true
    }

    static func hideOutline() {
        outlinePanel?.orderOut(nil)
        outlinePanel = nil
    }

    private static func present(_ panel: inout NSPanel?, content: NSView, frame: NSRect) {
        if let existing = panel {
            existing.setFrame(frame, display: true)
            existing.contentView = content
            existing.orderFrontRegardless()
            return
        }
        let p = NSPanel(
            contentRect: frame,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false)
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.level = .floating
        p.isMovableByWindowBackground = false
        p.contentView = content
        panel = p
        p.orderFrontRegardless()
    }

    // MARK: - Geometry

    private static func visibleScreenFrame() -> NSRect {
        NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1600, height: 900)
    }

    private static func hoveringRect(anchor: NSRect, contentHeight: CGFloat) -> NSRect {
        var frame = NSRect(x: anchor.minX, y: anchor.maxY + 4, width: 440, height: contentHeight)
        let screen = visibleScreenFrame()
        if frame.maxY > screen.maxY - 8 {
            frame = NSRect(x: anchor.minX, y: anchor.minY - contentHeight - 4, width: 440, height: contentHeight)
        }
        return frame
    }

    private static func completionRect(anchor: NSRect) -> NSRect {
        let height = 260.0
        var frame = NSRect(x: anchor.minX + 4, y: anchor.minY - height - 2, width: 340, height: height)
        let screen = visibleScreenFrame()
        if frame.minY < screen.minY + 8 {
            frame.origin.y = anchor.maxY + 4
        }
        return frame
    }
}

/// Light/dark surface for HUD content.
enum CompositedTheme {
    static var surface: Color {
        NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            ? Color(nsColor: NSColor(srgbRed: 0x2E / 255.0, green: 0x29 / 255.0, blue: 0x22 / 255.0, alpha: 1))
            : Color(nsColor: NSColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    }
    static var border: Color {
        NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            ? Color(nsColor: NSColor(srgbRed: 0x4A / 255.0, green: 0x43 / 255.0, blue: 0x36 / 255.0, alpha: 1))
            : Color(nsColor: NSColor(srgbRed: 0xE8 / 255.0, green: 0xE0 / 255.0, blue: 0xC9 / 255.0, alpha: 1))
    }
}

// MARK: - Views

struct CompletionList: View {
    @ObservedObject var model: GutterModel

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array((model.completions ?? []).enumerated()), id: \.offset) { index, item in
                HStack {
                    Text(item.label)
                        .font(.system(size: 12, design: .monospaced))
                        .lineLimit(1)
                    Spacer(minLength: 8)
                    if let detail = item.detail {
                        Text(detail)
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(index == model.completionIndex ? Color.accentColor.opacity(0.2) : Color.clear)
                .contentShape(Rectangle())
                .onTapGesture {
                    model.completionIndex = index
                    model.onCompletionCommit?()
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 6).stroke(CompositedTheme.border, lineWidth: 1)
        )
    }
}

struct OutlineList: View {
    let nodes: [LSPWire.OutlineNode]
    let onPick: (LSPWire.OutlineNode) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Document outline")
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(.secondary)
                .padding(8)
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(flatten(nodes).enumerated()), id: \.offset) { _, row in
                        Button {
                            onPick(row.node)
                        } label: {
                            HStack(spacing: 6) {
                                Text(row.depthLabel)
                                    .font(.system(size: 9))
                                    .foregroundColor(.secondary)
                                Text(row.node.name)
                                    .font(.system(size: 12, design: .monospaced))
                                    .lineLimit(1)
                                Spacer()
                                Text(String(row.node.line + 1))
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 6).stroke(CompositedTheme.border, lineWidth: 1)
        )
    }

    private func flatten(_ nodes: [LSPWire.OutlineNode], depth: Int = 0) -> [(node: LSPWire.OutlineNode, depthLabel: String)] {
        var out: [(LSPWire.OutlineNode, String)] = []
        for node in nodes {
            out.append((node, String(repeating: "·", count: depth)))
            out.append(contentsOf: flatten(node.children, depth: depth + 1))
        }
        return out
    }
}
