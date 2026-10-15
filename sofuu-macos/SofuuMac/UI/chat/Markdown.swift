import SwiftUI

/// A compact markdown renderer — enough parity with the desktop's
/// react-markdown (and the Android app's hand-rolled parser) for chat
/// answers: headings, bold/italic, inline code, fenced code blocks, lists
/// (bulleted + numbered), blockquotes, links, horizontal rules.
///
/// Only https/http/mailto links are clickable; anything else renders as
/// plain text (the same XSS discipline as the desktop's `safeUrl` — a
/// model-rendered link never becomes a live `javascript:` URL).
struct MarkdownBody: View {
    let text: String
    let fontSize: Int

    @Environment(\.sofuuColors) private var c

    var body: some View {
        let blocks = parseMdBlocks(text)
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                block(b)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func block(_ b: MdBlock) -> some View {
        switch b {
        case .code(let code, let lang):
            VStack(alignment: .leading, spacing: 0) {
                if !lang.isEmpty {
                    Text(lang).font(.system(size: 10, design: .monospaced)).foregroundStyle(c.textFaint)
                }
                ScrollView(.horizontal, showsIndicators: false) {
                    Text(code.trimmingCharacters(in: .whitespacesAndNewlines))
                        .font(.system(size: CGFloat(fontSize - 1), design: .monospaced))
                        .foregroundStyle(c.text)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(c.codeBlock, in: RoundedRectangle(cornerRadius: 8))
            .padding(.vertical, 6)

        case .heading(let t):
            Text(inlineMd(t, c))
                .font(.system(size: CGFloat(fontSize + 4), weight: .bold))
                .foregroundStyle(c.text)
                .padding(.top, 10).padding(.bottom, 4)

        case .paragraph(let t):
            Text(inlineMd(t, c))
                .font(.system(size: CGFloat(fontSize)))
                .lineSpacing(4)
                .foregroundStyle(c.text)
                .padding(.vertical, 2)

        case .bullet(let t):
            HStack(alignment: .top, spacing: 0) {
                Text("·  ").foregroundStyle(c.textMuted)
                Text(inlineMd(t, c)).foregroundStyle(c.text)
            }
            .font(.system(size: CGFloat(fontSize)))
            .padding(.vertical, 1)

        case .numbered(let idx, let t):
            HStack(alignment: .top, spacing: 0) {
                Text("\(idx).  ").foregroundStyle(c.textMuted)
                Text(inlineMd(t, c)).foregroundStyle(c.text)
            }
            .font(.system(size: CGFloat(fontSize)))
            .padding(.vertical, 1)

        case .quote(let t):
            Text(inlineMd(t, c))
                .font(.system(size: CGFloat(fontSize)))
                .foregroundStyle(c.textMuted)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(c.hoverVeil, in: RoundedRectangle(cornerRadius: 4))
                .padding(.leading, 8)
                .padding(.vertical, 2)

        case .rule:
            Rectangle().fill(c.separator).frame(height: 1).padding(.vertical, 8)
        }
    }
}

// MARK: - Inline markdown

/// Inline markdown via the platform parser, then link sanitizing.
func inlineMd(_ s: String, _ c: SofuuColors) -> AttributedString {
    let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    var a = (try? AttributedString(markdown: s, options: opts)) ?? AttributedString(s)

    // Collect first: mutating while iterating `runs` invalidates the view.
    var unsafeRanges: [Range<AttributedString.Index>] = []
    var codeRanges: [Range<AttributedString.Index>] = []
    for run in a.runs {
        if let url = run.link {
            let scheme = url.scheme?.lowercased() ?? ""
            if !(scheme == "https" || scheme == "http" || scheme == "mailto") {
                unsafeRanges.append(run.range)
            }
        }
        if run.inlinePresentationIntent?.contains(.code) == true {
            codeRanges.append(run.range)
        }
    }
    for r in unsafeRanges { a[r].link = nil }
    for r in codeRanges {
        a[r].font = .system(.body, design: .monospaced)
        a[r].backgroundColor = c.codeChip
    }
    return a
}

// MARK: - Block parsing

enum MdBlock {
    case paragraph(String)
    case heading(String)
    case code(String, String)
    case bullet(String)
    case numbered(Int, String)
    case quote(String)
    case rule
}

func parseMdBlocks(_ src: String) -> [MdBlock] {
    var blocks: [MdBlock] = []
    let lines = src.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
    var i = 0
    var para = ""
    var numbered = 0

    func flushPara() {
        if !para.isEmpty { blocks.append(.paragraph(para)) }
        para = ""
        numbered = 0
    }

    while i < lines.count {
        let trimmed = lines[i].trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("```") {
            flushPara()
            let lang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            var code = ""
            i += 1
            while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                code += lines[i] + "\n"
                i += 1
            }
            i += 1 // closing fence
            blocks.append(.code(code, lang))
        } else if trimmed.range(of: "^(-{3,}|\\*{3,}|_{3,})$", options: .regularExpression) != nil {
            flushPara()
            blocks.append(.rule)
        } else if trimmed.hasPrefix("#") {
            flushPara()
            blocks.append(.heading(trimmed.drop(while: { $0 == "#" }).trimmingCharacters(in: .whitespaces)))
        } else if trimmed.hasPrefix("> ") || trimmed == ">" {
            flushPara()
            var parts: [String] = []
            while i < lines.count {
                let t = lines[i].trimmingCharacters(in: .whitespaces)
                guard t.hasPrefix("> ") || t == ">" else { break }
                parts.append(String(t.dropFirst()).trimmingCharacters(in: .whitespaces))
                i += 1
            }
            blocks.append(.quote(parts.joined(separator: "\n")))
            continue
        } else if trimmed.range(of: "^[-*+] .+", options: .regularExpression) != nil {
            flushPara()
            blocks.append(.bullet(String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)))
        } else if trimmed.range(of: "^\\d+\\. .+", options: .regularExpression) != nil {
            flushPara()
            numbered += 1
            blocks.append(.numbered(numbered, trimmed.drop(while: { $0 != " " }).trimmingCharacters(in: .whitespaces)))
        } else if trimmed.isEmpty {
            flushPara()
        } else {
            if !para.isEmpty { para += " " }
            para += trimmed
        }
        i += 1
    }
    flushPara()
    return blocks
}
