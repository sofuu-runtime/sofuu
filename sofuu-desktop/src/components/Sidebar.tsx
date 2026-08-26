// Sidebar.tsx — ref1 left pane: Workspaces header with a + that adds a
// folder (same picker as Settings → project), the workspace list below,
// and the settings gear in the footer. 20% of the window; collapses to an
// 80px icon rail. The collapse toggle sits on the right border of the
// sidebar. Active workspace picks up the cream highlight (§4a).
//
// Sessions do NOT live here — the topbar strip holds the sessions of the
// active workspace; this pane is only about which folder you are in.

import { Icon } from "./Icon";

interface Props {
  /** Known workspace folder paths, newest first. */
  workspaces: string[];
  /** The workspace whose sessions are loaded (highlighted row). */
  activePath: string | null;
  onAddFolder: () => void;
  onSelectWorkspace: (path: string) => void;
  onOpenSettings: () => void;
  collapsed: boolean;
  onToggle: () => void;
}

function basename(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : path;
}

export function Sidebar({
  workspaces,
  activePath,
  onAddFolder,
  onSelectWorkspace,
  onOpenSettings,
  collapsed,
  onToggle,
}: Props) {
  return (
    <aside className={"sidebar" + (collapsed ? " collapsed" : "")} data-tauri-drag-region>
      <div className="sidebar-header">
        <span className="sidebar-title">Workspaces</span>
        <div className="sidebar-header-actions">
          <button
            className="icon-btn"
            title="Add folder (workspace)"
            aria-label="Add folder"
            onClick={onAddFolder}
          >
            <Icon name="plus" />
          </button>
        </div>
        <button
          className="icon-btn sidebar-edge-toggle"
          title={collapsed ? "Show sidebar (⌘B)" : "Hide sidebar (⌘B)"}
          aria-label={collapsed ? "Show sidebar" : "Hide sidebar"}
          aria-expanded={!collapsed}
          onClick={onToggle}
        >
          <Icon name="sidebar" />
        </button>
      </div>

      <div className="session-list">
        {workspaces.length === 0 && (
          <div className="sidebar-empty">
            Add a folder to begin
            <br />
            — or just pick one in Settings
          </div>
        )}
        {workspaces.map((path) => {
          const active = path === activePath;
          return (
            <button
              key={path}
              className={"session-item workspace-item" + (active ? " active" : "")}
              onClick={() => onSelectWorkspace(path)}
              title={path}
            >
              <Icon name="folder" size={16} className="workspace-icon" />
              <span className="session-label">{basename(path)}</span>
            </button>
          );
        })}
      </div>

      <div className="sidebar-footer">
        <button className="icon-btn" title="Settings (⌘,)" aria-label="Settings" onClick={onOpenSettings}>
          <Icon name="gear" />
        </button>
      </div>
    </aside>
  );
}
