import Foundation

/// The keyboard-shortcut registry — the Swift port of the desktop's
/// `shortcuts.ts`. ONE list feeds the Settings → Shortcuts pane; the
/// pulldown menu accelerators (Menus.swift) and the composer's key monitor
/// (Composer.swift) implement it. Keep the three in lockstep.
struct ShortcutDef: Identifiable {
    let group: String
    let keys: [String]
    let label: String
    var id: String { group + keys.joined() + label }
}

let SHORTCUTS: [ShortcutDef] = [
    ShortcutDef(group: "General", keys: ["⌘", "N"], label: "New chat"),
    ShortcutDef(group: "General", keys: ["⌘", ","], label: "Open settings"),
    ShortcutDef(group: "General", keys: ["Esc"], label: "Close a dialog or popover"),
    ShortcutDef(group: "Turn", keys: ["↩"], label: "Send message"),
    ShortcutDef(group: "Turn", keys: ["⇧", "↩"], label: "New line in the composer"),
    ShortcutDef(group: "Turn", keys: ["⌘", "."], label: "Stop generating"),
    ShortcutDef(group: "Turn", keys: ["⌘", "K"], label: "Compact session context"),
    ShortcutDef(group: "View", keys: ["⌘", "B"], label: "Toggle sidebar"),
    ShortcutDef(group: "View", keys: ["⌘", "+"], label: "Zoom in"),
    ShortcutDef(group: "View", keys: ["⌘", "-"], label: "Zoom out"),
    ShortcutDef(group: "View", keys: ["⌘", "0"], label: "Actual size"),
]

let SHORTCUT_GROUPS = ["General", "Turn", "View"]
