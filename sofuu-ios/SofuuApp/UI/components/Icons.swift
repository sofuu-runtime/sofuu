import SwiftUI

/// Map the desktop's activity-icon ids onto SF Symbols. Same visual
/// vocabulary as the desktop's SVG registry and the Android app's Material
/// symbol map: read / write / edit / grep / glob / folder / terminal / web /
/// link / delegate / checklist / brain / compress / gauge / gates / rlm /
/// warn / think / dot.
func iconFor(_ id: String) -> String {
    switch id {
    case Timeline.iconRead: return "doc.text"
    case Timeline.iconWrite: return "doc.badge.plus"
    case Timeline.iconEdit: return "square.and.pencil"
    case Timeline.iconGrep: return "text.magnifyingglass"
    case Timeline.iconGlob: return "magnifyingglass.circle"
    case Timeline.iconFolder: return "folder"
    case Timeline.iconTerminal: return "terminal"
    case Timeline.iconWeb: return "globe"
    case Timeline.iconLink: return "link"
    case Timeline.iconDelegate: return "point.3.connected.trianglepath.dotted"
    case Timeline.iconChecklist: return "checklist"
    case Timeline.iconBrain: return "brain.head.profile"
    case Timeline.iconCompress: return "rectangle.compress.vertical"
    case Timeline.iconGauge: return "gauge.with.dots.needle.33percent"
    case Timeline.iconFreshness: return "snowflake"
    case Timeline.iconRelevance: return "line.3.horizontal.decrease"
    case Timeline.iconSupervisor: return "person.badge.shield.checkmark"
    case Timeline.iconLoop: return "arrow.triangle.2.circlepath"
    case Timeline.iconRLM: return "cpu"
    case Timeline.iconWarn: return "exclamationmark.triangle"
    case Timeline.iconNotice: return "info.circle"
    case Timeline.iconPlan: return "map"
    case Timeline.iconThink: return "brain"
    default: return "circle.fill"
    }
}

/// UI glyphs that aren't timeline rows.
enum Icons2 {
    static let plus = "plus"
    static let gear = "gearshape"
    static let menu = "line.3.horizontal"
    static let send = "arrow.up"
    static let stop = "stop.fill"
    static let spark = "sparkles"
    static let caret = "chevron.down"
    static let clock = "clock"
    static let close = "xmark"
    static let attach = "paperclip"
    static let layers = "square.3.layers.3d"
    static let shield = "shield"
    static let folder = "folder"
    static let sidebar = "sidebar.left"
    static let chat = "bubble.left"
    static let moreVert = "ellipsis"
    static let chevronDown = "chevron.down"
    static let check = "checkmark"
    static let search = "magnifyingglass"
    static let trash = "trash"
    static let image = "photo"
    static let mic = "mic"
}
