import Foundation

/// Minimal string table — Phase 5's "i18n string-table-only" slice.
///
/// Deliberately small: ONE lookup function, an English table, and a
/// language preference (Settings ▸ Appearance). Adoption is incremental —
/// user-facing strings move into the table as parts of the app are
/// touched, so a future translation is adding a dictionary, not a
/// refactor. Untranslated keys fall back to English, and a missing key
/// falls back to the key itself (so the table can never blank the UI).
enum L10n {
    static var language: String {
        get { UserDefaults.standard.string(forKey: "appLanguage") ?? "en" }
        set { UserDefaults.standard.set(newValue, forKey: "appLanguage") }
    }

    static let tables: [String: [String: String]] = [
        "en": [
            // Menus (File group)
            "menu.newChat": "New Chat",
            "menu.saveFile": "Save File",
            "menu.findInFile": "Find in File",
            "menu.searchProject": "Search Project",
            "menu.renameSymbol": "Rename Symbol",
            "menu.findReferences": "Find References",
            "menu.findSymbol": "Find Symbol in Workspace",
            "menu.goToLine": "Go to Line",
            "menu.formatFile": "Format File",
            "menu.outline": "Document Outline",
            "menu.searchChat": "Search Chat",
            "menu.exportMarkdown": "Export Chat as Markdown…",
            // Session menu
            "menu.stop": "Stop",
            "menu.compact": "Compact Session",
            "menu.ideMode": "IDE Mode",
            "menu.exitIdeMode": "Exit IDE Mode",
            "menu.shortcuts": "Keyboard Shortcuts",
            "menu.restartEngine": "Restart Engine",
            // Composer
            "composer.placeholder": "Ask, @mention files, or type / for commands…",
            "composer.attach": "Attach files",
            "composer.voice": "Voice input",
            // Chat search
            "search.placeholder": "Search chat…",
            "search.noMatches": "No matches",
            "search.hint": "Search your messages and Sofuu's replies",
            "search.close": "Close (Esc)",
            "search.prev": "Previous match",
            "search.next": "Next match",
            // Git panel
            "git.title": "Git",
            "git.staged": "Staged",
            "git.unstaged": "Unstaged",
            "git.commit": "Commit",
            "git.messagePlaceholder": "Commit message…",
            "git.stage": "Stage",
            "git.unstage": "Unstage",
            "git.refresh": "Refresh",
            "git.notRepo": "Not a git repository",
            "git.clean": "Working tree is clean",
            // Onboarding
            "onboarding.title": "Sofuu",
            "onboarding.subtitle": "A local-first coding agent for macOS — chat with it, or hand it a folder and let it work in the built-in IDE.",
            "onboarding.getStarted": "Get Started",
        ],
    ]

    static func t(_ key: String) -> String {
        tables[language]?[key] ?? tables["en"]?[key] ?? key
    }
}
