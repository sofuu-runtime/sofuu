// App.tsx — the shell: sidebar + main column (topbar, content, composer),
// approval float, settings modal, menu-action + shortcut wiring.

import { useEffect } from "react";
import * as backend from "./lib/backend";
import { useAppStore } from "./lib/store";
import { ApprovalDialog } from "./components/ApprovalDialog";
import { ChatPane } from "./components/ChatPane";
import { CloudBackground } from "./components/CloudBackground";
import { Composer } from "./components/Composer";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";

export default function App() {
  const store = useAppStore();
  const { state } = store;

  // Prefetch every provider's model list in the background and keep it
  // fresh — the model picker reads this cache, so clicking a provider
  // never waits on the network (run at start, then every 5 minutes).
  useEffect(() => {
    if (!backend.isTauri()) return;
    const refresh = () => backend.refreshModelCache().catch(() => {});
    refresh();
    const t = window.setInterval(refresh, 5 * 60_000);
    return () => window.clearInterval(t);
  }, []);

  // NSMenu actions arrive as "menu://action" events (accelerators are
  // consumed by the native menu, so this is their only path in Tauri).
  useEffect(() => {
    if (!backend.isTauri()) return;
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    backend
      .onMenuAction((action) => {
        switch (action) {
          case "new-chat":
            store.newChat();
            break;
          case "settings":
            store.openSettings(true);
            break;
          case "stop":
            store.cancel();
            break;
          case "compact":
            backend.compactSession().catch(() => {});
            break;
          case "zoom-in":
            store.zoomBy(0.1);
            break;
          case "zoom-out":
            store.zoomBy(-0.1);
            break;
          case "zoom-reset":
            store.zoomReset();
            break;
        }
      })
      .then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [store]);

  // Keyboard fallback for browser dev (`npm run dev` outside Tauri, where
  // there is no NSMenu to own the accelerators). In the real app the menu
  // consumes those keys; ⌘L has no menu item, so the Composer handles it
  // directly in both worlds.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;
      const k = e.key.toLowerCase();
      // No NSMenu item owns ⌘B, so it works in both Tauri and the browser.
      if (k === "b") {
        e.preventDefault();
        store.toggleSidebar();
        return;
      }
      if (backend.isTauri()) return;
      if (k === "n") {
        e.preventDefault();
        store.newChat();
      } else if (k === ",") {
        e.preventDefault();
        store.openSettings(true);
      } else if (k === ".") {
        e.preventDefault();
        store.cancel();
      } else if (k === "k") {
        e.preventDefault();
        backend.compactSession().catch(() => {});
      } else if (k === "+" || k === "=") {
        e.preventDefault();
        store.zoomBy(0.1);
      } else if (k === "-") {
        e.preventDefault();
        store.zoomBy(-0.1);
      } else if (k === "0") {
        e.preventDefault();
        store.zoomReset();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [store]);

  const pickProject = async () => {
    try {
      const path = await backend.pickProjectDir();
      if (!path) return;
      store.addWorkspace(path);
      store.openWorkspace(path);
    } catch (e) {
      console.error(e);
    }
  };

  const patchConfig = async (patch: Record<string, unknown>) => {
    try {
      store.setConfig(await backend.updateConfig(patch));
    } catch (e) {
      console.error(e);
    }
  };

  const currentId = state.activeSessionId ?? state.sessions[0]?.id ?? null;
  const hasMessages = state.messages.length > 0;
  const preview = state.preview;

  // Topbar pill click: switch the live chat pane to that session.
  const switchSession = (id: string) => {
    store.setActiveSession(id);
    store.closePreview();
  };

  return (
    <div className={"app" + (state.sidebarCollapsed ? " sidebar-collapsed" : "")}>
      <Sidebar
        workspaces={state.workspaces}
        activePath={state.projectDir}
        onAddFolder={pickProject}
        onSelectWorkspace={store.openWorkspace}
        onRemoveWorkspace={store.removeWorkspace}
        onOpenSettings={() => store.openSettings(true)}
        collapsed={state.sidebarCollapsed}
        onToggle={store.toggleSidebar}
      />

      <div className="main">
        {state.cloudBg && <CloudBackground dark={state.resolvedDark} />}
        <TopBar
          activeId={currentId}
          previewLabel={preview ? preview.label : null}
          sessions={state.sessions}
          onNewChat={store.newChat}
          onSelectSession={switchSession}
          onDeleteSession={store.deleteSession}
          onClearSessions={store.clearSessions}
          onExitPreview={store.closePreview}
        />

        <div className="content">
          {state.bootError && <div className="banner error">{state.bootError}</div>}

          {preview ? (
            <ChatPane messages={[]} chatFontSize={state.chatFontSize} phase={null} preview={preview} />
          ) : !hasMessages ? (
            <div className="home-empty">
              <div className="headline">What should we work on?</div>
              {state.projectDir && (
                <div className="home-sub">Workspace: {shortProject(state.projectDir)}</div>
              )}
              <Composer
                variant="home"
                config={state.config}
                streaming={state.streaming}
                mode={state.mode}
                permissions={state.permissions}
                ctx={state.ctx}
                onSubmit={store.submit}
                onCancel={store.cancel}
                onModeChange={store.setMode}
                onPermissionsChange={store.setPermissions}
                onConfigPatch={patchConfig}
                onOpenSettings={() => store.openSettings(true, "providers")}
              />
            </div>
          ) : (
            <ChatPane messages={state.messages} chatFontSize={state.chatFontSize} phase={state.phase} />
          )}
        </div>

        {!preview && state.approvals.length > 0 && (
          <ApprovalDialog approval={state.approvals[0]} onResolve={store.approve} />
        )}

        {!preview && hasMessages && (
          <div style={{ padding: "0 0 20px", flexShrink: 0, position: "relative", zIndex: 1 }}>
            <Composer
              variant="chat"
              config={state.config}
              streaming={state.streaming}
              mode={state.mode}
              permissions={state.permissions}
              ctx={state.ctx}
              onSubmit={store.submit}
              onCancel={store.cancel}
              onModeChange={store.setMode}
              onPermissionsChange={store.setPermissions}
              onConfigPatch={patchConfig}
              onOpenSettings={() => store.openSettings(true, "providers")}
            />
          </div>
        )}
      </div>

      {state.settingsOpen && (
        <SettingsModal
          config={state.config}
          initialTab={state.settingsTab}
          projectDir={state.projectDir}
          chatFontSize={state.chatFontSize}
          zoom={state.zoom}
          cloudBg={state.cloudBg}
          theme={state.theme}
          onTheme={store.setTheme}
          onCloudBg={store.setCloudBg}
          permissions={state.permissions}
          onPermissionsChange={store.setPermissions}
          onChatFontSize={store.setChatFontSize}
          onZoomBy={store.zoomBy}
          onZoomReset={store.zoomReset}
          onConfigChanged={store.setConfig}
          onProjectDirChanged={(p) => {
            if (!p) return;
            store.addWorkspace(p);
            store.openWorkspace(p);
          }}
          onClose={() => store.openSettings(false)}
        />
      )}
    </div>
  );
}

function shortProject(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length ? `${parts.slice(0, -1).join("/")}/${parts[parts.length - 1]}` : path;
}
