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

    /// Display order for Settings ▸ Appearance. Codes match the table keys.
    static let languages: [(code: String, name: String)] = [
        ("en", "English"),
        ("es", "Español"),
        ("hi", "हिन्दी"),
    ]

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
            "menu.splitEditor": "Split Editor",
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
            // Split editor
            "split.open": "Open in Split",
            "split.close": "Close Split",
            "split.noFile": "Open a file first",
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
            "git.hunks": "Show hunks",
            "git.stageHunk": "Stage hunk",
            "git.unstageHunk": "Unstage hunk",
            "git.loadingHunks": "Loading hunks…",
            "git.noHunks": "No hunks",
            // Onboarding
            "onboarding.title": "Sofuu",
            "onboarding.subtitle": "A local-first coding agent for macOS — chat with it, or hand it a folder and let it work in the built-in IDE.",
            "onboarding.getStarted": "Get Started",
        ],
        "es": [
            // Menus (File group)
            "menu.newChat": "Nuevo chat",
            "menu.saveFile": "Guardar archivo",
            "menu.findInFile": "Buscar en el archivo",
            "menu.searchProject": "Buscar en el proyecto",
            "menu.renameSymbol": "Renombrar símbolo",
            "menu.findReferences": "Buscar referencias",
            "menu.findSymbol": "Buscar símbolo en el espacio de trabajo",
            "menu.goToLine": "Ir a la línea",
            "menu.formatFile": "Formatear archivo",
            "menu.outline": "Esquema del documento",
            "menu.searchChat": "Buscar en el chat",
            "menu.exportMarkdown": "Exportar chat como Markdown…",
            // Session menu
            "menu.stop": "Detener",
            "menu.compact": "Compactar sesión",
            "menu.ideMode": "Modo IDE",
            "menu.exitIdeMode": "Salir del modo IDE",
            "menu.splitEditor": "Dividir editor",
            "menu.shortcuts": "Atajos de teclado",
            "menu.restartEngine": "Reiniciar motor",
            // Composer
            "composer.placeholder": "Pregunta, menciona archivos con @ o escribe / para comandos…",
            "composer.attach": "Adjuntar archivos",
            "composer.voice": "Entrada de voz",
            // Chat search
            "search.placeholder": "Buscar en el chat…",
            "search.noMatches": "Sin coincidencias",
            "search.hint": "Busca en tus mensajes y en las respuestas de Sofuu",
            "search.close": "Cerrar (Esc)",
            "search.prev": "Coincidencia anterior",
            "search.next": "Coincidencia siguiente",
            // Split editor
            "split.open": "Abrir en la división",
            "split.close": "Cerrar división",
            "split.noFile": "Abre un archivo primero",
            // Git panel
            "git.title": "Git",
            "git.staged": "Preparado",
            "git.unstaged": "Sin preparar",
            "git.commit": "Confirmar",
            "git.messagePlaceholder": "Mensaje del commit…",
            "git.stage": "Preparar",
            "git.unstage": "Quitar",
            "git.refresh": "Actualizar",
            "git.notRepo": "No es un repositorio git",
            "git.clean": "El árbol de trabajo está limpio",
            "git.hunks": "Mostrar fragmentos",
            "git.stageHunk": "Preparar fragmento",
            "git.unstageHunk": "Quitar fragmento",
            "git.loadingHunks": "Cargando fragmentos…",
            "git.noHunks": "Sin fragmentos",
            // Onboarding
            "onboarding.title": "Sofuu",
            "onboarding.subtitle": "Un agente de programación local para macOS: conversa con él o dale una carpeta y deja que trabaje en el IDE integrado.",
            "onboarding.getStarted": "Comenzar",
        ],
        "hi": [
            // Menus (File group)
            "menu.newChat": "नया चैट",
            "menu.saveFile": "फ़ाइल सहेजें",
            "menu.findInFile": "फ़ाइल में खोजें",
            "menu.searchProject": "प्रोजेक्ट में खोजें",
            "menu.renameSymbol": "प्रतीक का नाम बदलें",
            "menu.findReferences": "संदर्भ खोजें",
            "menu.findSymbol": "वर्कस्पेस में प्रतीक खोजें",
            "menu.goToLine": "पंक्ति पर जाएँ",
            "menu.formatFile": "फ़ाइल फ़ॉर्मेट करें",
            "menu.outline": "दस्तावेज़ रूपरेखा",
            "menu.searchChat": "चैट खोजें",
            "menu.exportMarkdown": "चैट को Markdown के रूप में निर्यात करें…",
            // Session menu
            "menu.stop": "रोकें",
            "menu.compact": "सत्र संक्षिप्त करें",
            "menu.ideMode": "IDE मोड",
            "menu.exitIdeMode": "IDE मोड से बाहर",
            "menu.splitEditor": "एडिटर विभाजित करें",
            "menu.shortcuts": "कीबोर्ड शॉर्टकट",
            "menu.restartEngine": "इंजन पुनः आरंभ करें",
            // Composer
            "composer.placeholder": "पूछें, @ से फ़ाइलें जोड़ें, या / से कमांड लिखें…",
            "composer.attach": "फ़ाइलें जोड़ें",
            "composer.voice": "आवाज़ इनपुट",
            // Chat search
            "search.placeholder": "चैट में खोजें…",
            "search.noMatches": "कोई मिलान नहीं",
            "search.hint": "अपने संदेशों और Sofuu के उत्तरों में खोजें",
            "search.close": "बंद करें (Esc)",
            "search.prev": "पिछला मिलान",
            "search.next": "अगला मिलान",
            // Split editor
            "split.open": "विभाजन में खोलें",
            "split.close": "विभाजन बंद करें",
            "split.noFile": "पहले कोई फ़ाइल खोलें",
            // Git panel
            "git.title": "Git",
            "git.staged": "स्टेज किया गया",
            "git.unstaged": "स्टेज नहीं किया",
            "git.commit": "कमिट",
            "git.messagePlaceholder": "कमिट संदेश…",
            "git.stage": "स्टेज करें",
            "git.unstage": "स्टेज से हटाएँ",
            "git.refresh": "रीफ़्रेश",
            "git.notRepo": "यह git रिपॉज़िटरी नहीं है",
            "git.clean": "वर्किंग ट्री साफ़ है",
            "git.hunks": "हंक दिखाएँ",
            "git.stageHunk": "हंक स्टेज करें",
            "git.unstageHunk": "हंक हटाएँ",
            "git.loadingHunks": "हंक लोड हो रहे हैं…",
            "git.noHunks": "कोई हंक नहीं",
            // Onboarding
            "onboarding.title": "Sofuu",
            "onboarding.subtitle": "macOS के लिए एक लोकल-फ़र्स्ट कोडिंग एजेंट — इससे चैट करें, या एक फ़ोल्डर दें और अंतर्निहित IDE में काम करवाएँ।",
            "onboarding.getStarted": "शुरू करें",
        ],
    ]

    static func t(_ key: String) -> String {
        tables[language]?[key] ?? tables["en"]?[key] ?? key
    }
}
