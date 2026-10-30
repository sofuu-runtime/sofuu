// rt/brain_lock.rs — W2.1/W2.2: cross-process arbitration on a brain.
//
// Before this module NOTHING stopped two processes from writing the same
// brain. The store has a single-writer contract enforced only at flush time
// (`brain_flush_segmented` refuses when the manifest moved under it), which
// is a last line of defence, not a lock: both writers open, both remember,
// and the loser's records are stranded as unreferenced segment files that
// the age-guarded orphan sweep then DELETES. A reader had it worse — it
// could read a half-published store.
//
// So every open takes a `flock(2)` on `<brain>.lock`:
//
//   writable  → LOCK_EX|LOCK_NB, the `.lock` file created if absent, held
//               for the handle's lifetime and released on close/Drop.
//   readonly  → LOCK_SH|LOCK_NB. When `<brain>.lock` exists it is opened
//               READ-ONLY (a readonly open never creates or writes a file);
//               when it does not exist the SHARED lock is taken on the STORE
//               DIRECTORY instead — `flock` on a directory creates nothing,
//               which is the same trick the C engine used, and it is what
//               lets a readonly open work in a chmod 0555 directory.
//
// A lock file left behind by a dead process can never wedge a brain: `flock`
// is released by the KERNEL when the descriptor's last owner dies, so the
// stale file is just an empty file with an old pid in it. The pid is recorded
// for diagnostics only — the lock is the truth, never the pid (`acquire`
// never consults it). That also means a recycled pid cannot fake a holder.
//
// Nothing here reads or writes the segment files, so `<brain>.lock` lives
// BESIDE the brain and never inside `<brain>.d`: the store's own naming
// scheme (`seg-*.qtsq`) is the only thing segment scanning recognises, so the
// lock file cannot be mistaken for one.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// How a handle intends to use the brain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Writable: exclusive; a second writer is a hard refusal (E_LOCKED).
    Exclusive,
    /// Read-only: shared; concurrent readers are fine, a writer is not.
    Shared,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Exclusive => "exclusive writer",
            Mode::Shared => "shared reader",
        }
    }
}

/// Why the lock could not be taken. ONLY a real conflict is an error: every
/// other failure (no permission to create the file, a filesystem without
/// `flock`) degrades to an UNLOCKED handle with a loud warning and an
/// observable `degraded` reason — an open that could never write anyway must
/// not become a hard failure, and E_LOCKED must mean "someone else is here",
/// never "the disk said no".
#[derive(Clone, Debug)]
pub enum LockError {
    /// Another open file description holds a conflicting lock. This is the
    /// only "conflict" case: the caller maps it to the crate's E_LOCKED.
    Busy {
        path: String,
        mode: Mode,
        /// The pid the holder recorded, when it recorded one. Diagnostics
        /// only — the kernel's lock decided, not this number.
        holder_pid: Option<u32>,
    },
}
/// W2.1: ONE lock per (process, brain).
///
/// `flock` arbitrates between *open file descriptions*, not processes, so a
/// second `open()` of the same brain inside one process — a reopen after
/// `close()`, two handles in a test, a transient handle — would otherwise be
/// refused as "another writer holds it". That is wrong: the conflict W2.1
/// exists to stop is two PROCESSES interleaving writes. Handles in one
/// process therefore share a single kernel lock, refcounted here, and the
/// last one out releases it.
#[derive(Debug, Default)]
struct InProcessLock {
    file: Option<File>,
    refs: usize,
}

fn in_process_registry() -> &'static std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<std::sync::Mutex<InProcessLock>>>> {
    static REGISTRY: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<std::sync::Mutex<InProcessLock>>>>,
    > = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}


impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::Busy {
                path,
                mode,
                holder_pid,
            } => {
                write!(
                    f,
                    "{path} is already locked by another {} (pid {}) — two writers must not share a brain",
                    mode.label(),
                    holder_pid
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )
            }
        }
    }
}

/// An acquired lock, held for as long as the value lives.
#[derive(Debug)]
pub struct BrainLock {
    file: Option<File>,
    /// Set when this handle SHARES a lock another handle in the same process
    /// already holds. The registry below keeps the real `File` alive and
    /// counts references, so `flock`'s per-open-file-description semantics
    /// cannot make a process conflict with itself.
    shared: Option<std::sync::Arc<std::sync::Mutex<InProcessLock>>>,
    path: String,
    mode: Mode,
    /// `Some(reason)` when NO lock could be taken (see `LockError`): the
    /// handle is usable, but nothing is arbitrated.
    degraded: Option<String>,
}

impl BrainLock {
    /// The file (or directory) the lock was taken on.
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Why this handle holds no real lock, when it does not.
    pub fn degraded(&self) -> Option<&str> {
        self.degraded.as_deref()
    }

    /// Release the lock NOW instead of at Drop (a JS `close()` frees the
    /// handle's arbitration immediately, before the object is collected).
    pub fn release(&mut self) {
        if let Some(file) = self.file.take() {
            unlock(&file);
            /* Dropping the File here closes the descriptor, which would
             * release the flock anyway — the explicit unlock above just
             * makes the intent unambiguous. */
        }
        /* W2.1: a shared handle only releases the kernel lock when it is the
         * LAST handle for this path in the process. */
        if let Some(entry) = self.shared.take() {
            let mut reg = in_process_registry().lock().expect("lock registry");
            let mut e = entry.lock().expect("lock entry");
            e.refs = e.refs.saturating_sub(1);
            if e.refs == 0 {
                if let Some(file) = e.file.take() {
                    unlock(&file);
                }
                reg.remove(&self.path);
            }
        }
    }
}

impl Drop for BrainLock {
    fn drop(&mut self) {
        self.release();
    }
}

/// `flock` needs a unix target. On anything else the lock degrades to a
/// no-op (the same posture rt/events.rs takes; the brain store never
/// promised multi-process writers on Windows).
#[cfg(unix)]
fn try_lock(file: &File, mode: Mode) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let op = match mode {
        Mode::Exclusive => libc::LOCK_EX,
        Mode::Shared => libc::LOCK_SH,
    } | libc::LOCK_NB;
    // SAFETY: `file` owns the descriptor for the whole call.
    let rc = unsafe { libc::flock(file.as_raw_fd(), op) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock(file: &File) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: same open file description that took the lock.
    let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
}

#[cfg(not(unix))]
fn try_lock(_file: &File, _mode: Mode) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &File) {}

/// A "someone else holds it" errno, whatever the platform calls it.
fn is_busy(err: &std::io::Error) -> bool {
    if err.kind() == std::io::ErrorKind::WouldBlock {
        return true;
    }
    #[cfg(unix)]
    {
        if let Some(code) = err.raw_os_error() {
            return code == libc::EWOULDBLOCK || code == libc::EAGAIN || code == libc::EACCES;
        }
    }
    false
}

/// The lock file for `brain` — `<brain>.lock`, beside the brain, never
/// inside `<brain>.d`.
pub fn lock_path(brain: &str) -> String {
    format!("{brain}.lock")
}

/// The pid recorded in a lock file, when it holds one (diagnostics).
pub fn read_pid(path: &str) -> Option<u32> {
    let mut text = String::new();
    File::open(path).ok()?.read_to_string(&mut text).ok()?;
    text.trim().parse::<u32>().ok()
}

/// Take the lock for `brain`.
///
/// Exclusive creates `<brain>.lock` (and its directory) when absent and
/// records this process's pid once the lock is held. Shared never creates
/// anything: it opens an existing `.lock` read-only, or falls back to the
/// store DIRECTORY so a readonly open works in a directory it cannot write.
pub fn acquire(brain: &str, mode: Mode) -> Result<BrainLock, LockError> {
    let lock_file = lock_path(brain);
    /* Every failure OTHER than a conflict ends here as a degraded handle —
     * see LockError's docs. The reason is kept on the handle so stats() can
     * report "this brain is NOT arbitrated" instead of pretending. */
    let degrade = |path: String, err: String| -> Result<BrainLock, LockError> {
        eprintln!(
            "[sofuu/cma] W2.1 WARNING: cannot take the {} lock on {path} ({err}) — this handle runs UNLOCKED",
            mode.label()
        );
        Ok(BrainLock {
            file: None,
            shared: None,
            path,
            mode,
            degraded: Some(err),
        })
    };

    let (file, target, record_pid) = match mode {
        Mode::Exclusive => {
            if let Some(parent) = Path::new(&lock_file).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            let file = match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_file)
            {
                Ok(f) => f,
                Err(e) => return degrade(lock_file.clone(), e.to_string()),
            };
            (file, lock_file.clone(), true)
        }
        Mode::Shared => {
            if Path::new(&lock_file).exists() {
                /* READ-ONLY: a shared open must not touch the file's
                 * contents or its mtime. */
                let file = match OpenOptions::new().read(true).open(&lock_file) {
                    Ok(f) => f,
                    Err(e) => return degrade(lock_file.clone(), e.to_string()),
                };
                (file, lock_file.clone(), false)
            } else {
                /* No lock file (and creating one would be a write): take
                 * the SHARED lock on the store directory itself. `flock` on
                 * a directory needs no new file and no write permission. */
                let dir = Path::new(brain)
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| ".".to_string());
                let file = match File::open(&dir) {
                    Ok(f) => f,
                    Err(e) => return degrade(dir.clone(), e.to_string()),
                };
                (file, dir, false)
            }
        }
    };

    /* W2.1: is this brain already locked by a handle in THIS process? Then
     * share that lock. `flock` is per open-file-description, so without this
     * every in-process reopen (after close(), or a second handle in a test)
     * would be reported as "another writer holds it" and refused. */
    {
        let mut reg = in_process_registry().lock().expect("lock registry");
        if let Some(entry) = reg.get(&lock_file) {
            let upgraded = {
                let mut e = entry.lock().expect("lock entry");
                e.refs += 1;
                (e.refs, mode)
            };
            let _ = upgraded;
            return Ok(BrainLock {
                file: None,
                shared: Some(entry.clone()),
                path: target,
                mode,
                degraded: None,
            });
        }
    }

    if let Err(e) = try_lock(&file, mode) {
        if !is_busy(&e) {
            return degrade(target, e.to_string());
        }
        return Err(LockError::Busy {
            holder_pid: read_pid(&lock_file),
            path: target,
            mode,
        });
    }

    if record_pid {
        /* Informational only: the flock above is the arbitration. A file
         * holding a DEAD process's pid is therefore harmless — the kernel
         * released that process's lock when it died, and this write simply
         * replaces the stale number. */
        let mut f = &file;
        let _ = f.set_len(0);
        let _ = f.seek(SeekFrom::Start(0));
        let _ = write!(f, "{}\n", std::process::id());
        let _ = f.flush();
    }

    /* Publish the lock so a second handle in this process shares it rather
     * than colliding with it. The File lives inside the registry entry: the
     * kernel lock is released when the LAST handle for this path drops. */
    let entry = std::sync::Arc::new(std::sync::Mutex::new(InProcessLock {
        file: Some(file),
        refs: 1,
    }));
    in_process_registry()
        .lock()
        .expect("lock registry")
        .insert(lock_file, entry.clone());
    Ok(BrainLock {
        file: None,
        shared: Some(entry),
        path: target,
        mode,
        degraded: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("sofuu-brainlock-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn exclusive_blocks_a_second_writer_and_releases_on_drop() {
        let dir = tmpdir("excl");
        let brain = dir.join("brain.qtsq").to_string_lossy().to_string();
        let first = acquire(&brain, Mode::Exclusive).expect("first writer");
        assert_eq!(first.path(), lock_path(&brain));
        assert_eq!(read_pid(&lock_path(&brain)), Some(std::process::id()));

        /* A second handle in the SAME process shares the lock. `flock` is per
         * open file description, so treating that as a conflict would make
         * every reopen-after-close fail — the conflict W2.1 exists to stop is
         * two PROCESSES, and that is what the second half of this test proves. */
        let second = acquire(&brain, Mode::Exclusive)
            .expect("a second handle in one process shares the lock");
        assert_eq!(second.path(), first.path());
        drop(second);
        drop(first);

        /* An external holder is exactly "another open file description",
         * which is what a second PROCESS gets. Take the lock the way a rival
         * writer would and prove the arbitration still refuses us. */
        let rival = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path(&brain))
            .expect("rival opens the lock file");
        try_lock(&rival, Mode::Exclusive).expect("rival takes the lock");
        match acquire(&brain, Mode::Exclusive) {
            Err(LockError::Busy { .. }) => {}
            other => panic!("a rival writer must be refused, got {other:?}"),
        }
        /* A reader is refused too while a writer holds it. */
        assert!(matches!(
            acquire(&brain, Mode::Shared),
            Err(LockError::Busy { .. })
        ));

        unlock(&rival);
        acquire(&brain, Mode::Exclusive).expect("released when the rival lets go");
    }

    #[test]
    fn shared_readers_coexist_and_never_create_the_lock_file() {
        let dir = tmpdir("shared");
        let brain = dir.join("brain.qtsq").to_string_lossy().to_string();
        /* No writable open first: the .lock file must not be created. */
        let r1 = acquire(&brain, Mode::Shared).expect("reader 1");
        let r2 = acquire(&brain, Mode::Shared).expect("reader 2 (shared)");
        assert!(!Path::new(&lock_path(&brain)).exists());
        assert_eq!(r1.path(), dir.to_string_lossy(), "locked the directory");
        drop(r2);
        drop(r1);
        /* …and once a writer HAS created it, readers share it read-only. */
        drop(acquire(&brain, Mode::Exclusive).expect("writer"));
        let r3 = acquire(&brain, Mode::Shared).expect("reader 3");
        assert_eq!(r3.path(), lock_path(&brain));
    }

    #[test]
    fn a_lock_file_with_a_dead_pid_does_not_wedge_the_brain() {
        let dir = tmpdir("stale");
        let brain = dir.join("brain.qtsq").to_string_lossy().to_string();
        /* A pid that cannot be running: spawn a child, reap it, use its
         * (now certainly dead) pid. */
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .expect("spawn");
        let dead = child.id();
        let _ = child.wait();
        std::fs::write(lock_path(&brain), format!("{dead}\n")).unwrap();

        let lock = acquire(&brain, Mode::Exclusive).expect("a stale lock file must not block");
        assert_eq!(
            read_pid(&lock_path(&brain)),
            Some(std::process::id()),
            "the stale pid is replaced by the live holder"
        );
        drop(lock);
    }
}
