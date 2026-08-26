// lib/shortcuts.ts — the single source of truth for keyboard shortcuts.
// The Settings → Shortcuts pane renders this list; the menu accelerators in
// src-tauri/lib.rs and the key handlers in App.tsx/Composer.tsx implement
// it. Keep the three in lockstep when adding bindings.

export interface ShortcutDef {
  group: string;
  keys: string[];
  label: string;
}

export const SHORTCUTS: ShortcutDef[] = [
  { group: "General", keys: ["⌘", "N"], label: "New chat" },
  { group: "General", keys: ["⌘", ","], label: "Open settings" },
  { group: "General", keys: ["⌘", "L"], label: "Focus the composer" },
  { group: "General", keys: ["Esc"], label: "Close a dialog or popover" },
  { group: "Turn", keys: ["↩"], label: "Send message" },
  { group: "Turn", keys: ["⇧", "↩"], label: "New line in the composer" },
  { group: "Turn", keys: ["⌘", "."], label: "Stop generating" },
  { group: "Turn", keys: ["⌘", "K"], label: "Compact session context" },
  { group: "View", keys: ["⌘", "B"], label: "Toggle sidebar" },
  { group: "View", keys: ["⌘", "+"], label: "Zoom in" },
  { group: "View", keys: ["⌘", "-"], label: "Zoom out" },
  { group: "View", keys: ["⌘", "0"], label: "Actual size" },
];

export const SHORTCUT_GROUPS = ["General", "Turn", "View"];
