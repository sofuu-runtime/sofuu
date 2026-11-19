// sessions.rs — read-only mirror of the session registry (PLAN-DESKTOP D).
//
// The session mesh stores a plaintext liveness index at
// `<project>/.sofuu/sessions/registry.json` (session.rs in sofuu-core).
// The desktop reads it DIRECTLY for the sidebar list — no engine
// round-trip, so the sidebar works even mid-turn. Turn history lives in
// QTSQ-encrypted files and is served through the engine (chat.js) instead.
//
// Once workstream B moves session.rs into the sofuu-core lib, this mirror
// can switch to the real types; the JSON shape is stable either way.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Mirror of session.rs's SessionInfo (field-for-field).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub pid: u64,
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub started_at: u64,
    #[serde(default)]
    pub last_seen: u64,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub ended: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Registry {
    #[serde(default)]
    sessions: Vec<SessionInfo>,
}

pub(crate) fn sessions_dir(project: &Path) -> PathBuf {
    project.join(".sofuu").join("sessions")
}

/// Public wrapper for commands.rs — the mesh root of a picked project.
pub(crate) fn mesh_root_for(project: &Path) -> PathBuf {
    mesh_root(project)
}

/// The session-mesh root for a picked project dir — must match what
/// chat.js/session.rs derive (session.rs::project_root): the git toplevel
/// when the dir lives inside a repo, else the dir itself. Without this a
/// sub-folder pick would read the registry at the wrong root (the mesh
/// writes it at the toplevel).
fn mesh_root(project: &Path) -> PathBuf {
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(project)
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let p = PathBuf::from(s.trim());
            if !p.as_os_str().is_empty() {
                return p;
            }
        }
    }
    project.to_path_buf()
}

/// List the project's sessions, newest activity first.
///
/// A MISSING registry file means the project has no chats yet — an empty
/// list, not an error. A registry that exists but cannot be read or parsed
/// IS an error: reporting "no sessions" there would hide a broken store
/// behind an empty sidebar.
pub fn list_sessions(project: &Path) -> Result<Vec<SessionInfo>, String> {
    let path = sessions_dir(&mesh_root(project)).join("registry.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("registry unreadable ({}): {e}", path.display())),
    };
    let Ok(reg) = serde_json::from_str::<Registry>(&text) else {
        return Err(format!("registry malformed ({})", path.display()));
    };
    let mut sessions = reg.sessions;
    sessions.sort_by(|a, b| b.last_seen.cmp(&a.last_seen));
    Ok(sessions)
}
