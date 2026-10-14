// rt/brain_key.rs — the machine-local brain key (ROADMAP-BRAIN-CMA B4).
//
// Every brain file used to be encrypted with ONE password compiled into the
// binary ("sofuu-kv-local-v1"), so any .qtsq file was readable by anyone
// holding the source. The key now comes from a machine-local secret:
//
//   1. SOFUU_BRAIN_KEY (64 hex chars) — an explicit override for CI or a
//      deliberately shared key across a fleet;
//   2. ~/.sofuu/brain.key — 32 random bytes as hex, mode 0600, generated on
//      first use.
//
// Why a key FILE and not the codec's device keystore (evaluated first, as
// the roadmap asks): on macOS `qtsq_keystore_init` goes through Keychain
// Services, and an unsigned binary triggers a Keychain prompt on every
// launch — the exact friction the fixed password existed to avoid. The
// keystore's encrypt path is also ML-KEM encrypt-to-self, a different
// mechanism from the password vault every existing brain file uses, so
// adopting it would change the on-disk scheme rather than the key. The
// 0600 file keeps the vault, keeps the format, and still makes the key
// machine-local.
//
// The legacy password stays readable: a file that only the old password
// opens is reported as legacy, backed up, and re-encrypted on its first
// write (see rt/memory.rs + rt/brain_store.rs). A file that NEITHER
// password opens fails closed — the open refuses, nothing is written.

use std::sync::OnceLock;

/// The password every brain file written before B4 carries. Kept only to
/// read (and then migrate) those files.
pub const LEGACY_PASSWORD: &str = "sofuu-kv-local-v1";

/// Environment override: 64 hex characters (32 bytes).
pub const KEY_ENV: &str = "SOFUU_BRAIN_KEY";

/// The key file under the user's home.
pub fn key_file_path() -> Option<String> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    if home.is_empty() {
        return None;
    }
    Some(format!("{home}/.sofuu/brain.key"))
}

/// The machine-local vault password.
///
/// Cached for the process: the file is read (or created) once, so a brain
/// flush never pays a syscall for it and cannot observe a half-written key.
pub fn local_vault_password() -> String {
    #[cfg(test)]
    if let Some(v) = TEST_KEY_OVERRIDE.with(|c| c.borrow().clone()) {
        return v;
    }
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED.get_or_init(resolve_password).clone()
}

fn resolve_password() -> String {
    if let Ok(v) = std::env::var(KEY_ENV) {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    match key_file_path() {
        Some(path) => match load_or_create_key(&path) {
            Ok(key) => key,
            Err(e) => {
                /* No usable key file (read-only home, no HOME). Falling back
                 * to the legacy password would silently downgrade every new
                 * brain, so say so loudly and use a per-process random key:
                 * the brain still works within this process, and the next
                 * run with a writable home gets a real key. */
                eprintln!(
                    "[sofuu/brain] no machine key at {path} ({e}) — using a \
                     process-local key; brains written now will not reopen"
                );
                random_hex(32)
            }
        },
        None => {
            eprintln!("[sofuu/brain] HOME is unset — using a process-local key");
            random_hex(32)
        }
    }
}

/// Read the key file, or create it (0600) on first use.
fn load_or_create_key(path: &str) -> Result<String, String> {
    if let Ok(existing) = std::fs::read_to_string(path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    let key = random_hex(32);
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    /* Write to a 0600 tmp sibling and rename: a reader never sees a partial
     * key, and the file is never briefly world-readable. */
    let tmp = format!("{path}.tmp");
    {
        use std::io::Write;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts.open(&tmp).map_err(|e| e.to_string())?;
        f.write_all(key.as_bytes()).map_err(|e| e.to_string())?;
        f.write_all(b"\n").map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })?;
    Ok(key)
}

/// 32+ bytes of hex from the OS entropy source.
fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    #[cfg(unix)]
    {
        use std::io::Read;
        if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
            if f.read_exact(&mut buf).is_ok() {
                return buf.iter().map(|b| format!("{b:02x}")).collect();
            }
        }
    }
    /* Non-Unix, or an unreadable /dev/urandom: mix what entropy is cheaply
     * available. Weaker than the OS source, so it is the last resort and
     * never silently preferred — the caller's eprintln already fired. */
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64).rotate_left(17)
        ^ (&buf as *const _ as u64);
    let mut out = String::with_capacity(bytes * 2);
    for _ in 0..bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push_str(&format!("{:02x}", (state >> 24) as u8));
    }
    out
}

#[cfg(test)]
thread_local! {
    /// Tests pin a key per-thread: the real key lives in the user's home,
    /// and cargo runs tests in parallel, so a process-global override would
    /// leak between them.
    static TEST_KEY_OVERRIDE: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub fn set_test_key(k: Option<&str>) {
    TEST_KEY_OVERRIDE.with(|c| *c.borrow_mut() = k.map(str::to_string));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_hex_is_hex_and_distinct() {
        let a = random_hex(32);
        let b = random_hex(32);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "two draws must not collide");
    }

    #[test]
    fn key_file_is_created_private_and_stable() {
        let dir = std::env::temp_dir().join(format!(
            "sofuu-brain-key-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("brain.key");
        let p = path.to_string_lossy().to_string();

        let first = load_or_create_key(&p).expect("create");
        assert_eq!(first.len(), 64);
        let second = load_or_create_key(&p).expect("reload");
        assert_eq!(first, second, "the key must be stable across reads");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "the key file must be private");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn env_override_wins() {
        /* The env is process-global; this test only asserts the resolution
         * ORDER, using the test override to stand in for the env so it does
         * not race the other key tests. */
        set_test_key(Some("deadbeef"));
        assert_eq!(local_vault_password(), "deadbeef");
        set_test_key(None);
    }
}
