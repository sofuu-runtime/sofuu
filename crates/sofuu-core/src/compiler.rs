//! `sofuu compile` — turn an entry script into a standalone executable.
//!
//! The output is the running `sofuu` binary with an append-only payload and a
//! fixed 32-byte trailer. No external compiler, linker or archive tool is
//! involved: the payload is the entry's ESM graph bundled to one source, that
//! source compiled to QuickJS bytecode, and optionally a `.qtsq` brain.
//!
//! ```text
//! +------------------------------------------------+
//! | base `sofuu` executable (unchanged bytes)      |
//! +------------------------------------------------+
//! | payload container (header + name + bc + brain) |
//! +------------------------------------------------+
//! | 32-byte trailer (magic, version, flags, len,   |
//! |                  CRC32 of the payload)         |
//! +------------------------------------------------+
//! ```
//!
//! Two properties drive the design:
//!
//! * **Recompiling cannot nest payloads.** The base is whatever the running
//!   executable is *minus* an existing trailer, so `sofuu compile` run by a
//!   compiled agent produces the same artifact as running it by a plain CLI.
//! * **A broken payload is loud, never silent.** A trailer that proves a
//!   payload exists but whose CRC or container does not check out is an
//!   error: falling back to the CLI would run the wrong program.
//!
//! Bytecode is tied to the QuickJS build that produced it (atoms are interned
//! per build). That is safe here because the payload always sits on a copy of
//! the very binary that compiled it — the artifact is as portable as the
//! binary it carries, and no further.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use sofuu_ffi::qjs;

/// `SOFUU_BIN_V1` — exactly 12 bytes, so the trailer starts on a word
/// boundary. (The spec writes the magic with a trailing `\0` and calls it 12
/// bytes; `SOFUU_BIN_V1` is already 12, so the NUL is dropped rather than
/// making the trailer 33 bytes and breaking the fixed-width offsets.)
pub const TRAILER_MAGIC: &[u8; 12] = b"SOFUU_BIN_V1";
/// The trailer is a fixed 32 bytes, appended last, so boot is one seek.
pub const TRAILER_LEN: usize = 32;
/// Version of the trailer LAYOUT. A future layout change must bump it: an old
/// runtime reading a new artifact must refuse rather than guess.
pub const TRAILER_VERSION: u16 = 1;

/// Marker at the start of the payload body (behind the trailer).
const PAYLOAD_MAGIC: &[u8; 8] = b"SOFUUPK1";
const PAYLOAD_VERSION: u16 = 1;
/// magic(8) ver(2) flags(2) bc_len(4) brain_len(4) brain_dim(4) name_len(2) rsv(2)
const PAYLOAD_HEADER_LEN: usize = 28;

const FLAG_BC_GZIPPED: u16 = 1 << 0;

/* ------------------------------------------------------------------ */
/* Trailer                                                             */
/* ------------------------------------------------------------------ */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trailer {
    pub version: u16,
    pub flags: u16,
    pub payload_len: u64,
    pub crc32: u32,
}

impl Trailer {
    fn to_bytes(self) -> [u8; TRAILER_LEN] {
        let mut out = [0u8; TRAILER_LEN];
        out[..12].copy_from_slice(TRAILER_MAGIC);
        out[12..14].copy_from_slice(&self.version.to_le_bytes());
        out[14..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..24].copy_from_slice(&self.payload_len.to_le_bytes());
        out[24..28].copy_from_slice(&self.crc32.to_le_bytes());
        /* 28..32 reserved — zeros, so a future field can take them without
         * moving any offset a reader already knows. */
        out
    }

    /// `None` means "this is an ordinary binary", which is not an error. A
    /// matching magic with a malformed body is left to `detect_*`, which can
    /// tell the operator that their artifact is damaged.
    fn parse(bytes: &[u8]) -> Option<Trailer> {
        if bytes.len() < TRAILER_LEN || &bytes[..12] != TRAILER_MAGIC {
            return None;
        }
        Some(Trailer {
            version: u16::from_le_bytes([bytes[12], bytes[13]]),
            flags: u16::from_le_bytes([bytes[14], bytes[15]]),
            payload_len: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
            crc32: u32::from_le_bytes(bytes[24..28].try_into().ok()?),
        })
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    // flate2 already ships crc32fast for the gzip path; using its table keeps
    // the dependency count (and the binary) where the size cap wants them.
    let mut c = flate2::Crc::new();
    c.update(bytes);
    c.sum()
}

/* ------------------------------------------------------------------ */
/* Payload                                                             */
/* ------------------------------------------------------------------ */

/// A payload decoded from an executable.
#[derive(Clone, Debug)]
pub struct EmbeddedPayload {
    /// The entry's file name, for diagnostics ("agent.ts"). Not a path the
    /// payload can resolve later — the artifact is self-contained.
    pub entry: String,
    /// QuickJS bytecode exactly as stored (gzip-wrapped when
    /// [`EmbeddedPayload::gzipped`]).
    pub bytecode: Vec<u8>,
    pub gzipped: bool,
    pub brain: Option<Vec<u8>>,
    /// The dimension of `brain`, recorded at compile time — `sofuu.memory.open`
    /// needs it and the payload cannot ask a human.
    pub brain_dim: u32,
    /// CRC32 the trailer recorded, reused to name an extracted brain.
    pub crc32: u32,
}

/// The whole payload body: header, entry name, bytecode, brain.
fn encode_payload(
    entry: &str,
    bytecode: &[u8],
    gzipped: bool,
    brain: Option<&[u8]>,
    brain_dim: u32,
) -> Vec<u8> {
    let name = entry.as_bytes();
    let brain = brain.unwrap_or(&[]);
    let mut out = Vec::with_capacity(PAYLOAD_HEADER_LEN + name.len() + bytecode.len() + brain.len());
    out.extend_from_slice(PAYLOAD_MAGIC);
    out.extend_from_slice(&PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(if gzipped { FLAG_BC_GZIPPED } else { 0u16 }).to_le_bytes());
    out.extend_from_slice(&(bytecode.len() as u32).to_le_bytes());
    out.extend_from_slice(&(brain.len() as u32).to_le_bytes());
    out.extend_from_slice(&brain_dim.to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(name);
    out.extend_from_slice(bytecode);
    out.extend_from_slice(brain);
    out
}

/// Decode a payload body. Every length is bounds-checked against the body:
/// this runs on bytes that came off disk, on every boot.
fn decode_payload(p: &[u8], crc32: u32) -> Result<EmbeddedPayload, String> {
    if p.len() < PAYLOAD_HEADER_LEN || &p[..8] != PAYLOAD_MAGIC {
        return Err("ERR_BINARY_CORRUPTED: the embedded payload is not a Sofuu payload (bad magic)".to_string());
    }
    let ver = u16::from_le_bytes([p[8], p[9]]);
    if ver != PAYLOAD_VERSION {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: the embedded payload speaks format v{ver}; this runtime speaks v{PAYLOAD_VERSION} \
             — recompile the agent with this binary"
        ));
    }
    let flags = u16::from_le_bytes([p[10], p[11]]);
    let bc_len = u32::from_le_bytes(p[12..16].try_into().unwrap()) as usize;
    let brain_len = u32::from_le_bytes(p[16..20].try_into().unwrap()) as usize;
    let brain_dim = u32::from_le_bytes(p[20..24].try_into().unwrap());
    let name_len = u16::from_le_bytes([p[24], p[25]]) as usize;

    let name_start = PAYLOAD_HEADER_LEN;
    let bc_start = name_start
        .checked_add(name_len)
        .ok_or_else(|| "ERR_BINARY_CORRUPTED: the embedded payload has a corrupt header".to_string())?;
    let brain_start = bc_start
        .checked_add(bc_len)
        .ok_or_else(|| "ERR_BINARY_CORRUPTED: the embedded payload has a corrupt header".to_string())?;
    let end = brain_start
        .checked_add(brain_len)
        .ok_or_else(|| "ERR_BINARY_CORRUPTED: the embedded payload has a corrupt header".to_string())?;
    if end > p.len() {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: the embedded payload is truncated (header claims {end} bytes, {} present)",
            p.len()
        ));
    }
    let entry = String::from_utf8_lossy(&p[name_start..bc_start]).into_owned();
    let bytecode = p[bc_start..brain_start].to_vec();
    if bytecode.is_empty() {
        return Err("ERR_BINARY_CORRUPTED: the embedded payload carries no bytecode".to_string());
    }
    let brain = if brain_len > 0 {
        Some(p[brain_start..end].to_vec())
    } else {
        None
    };
    Ok(EmbeddedPayload {
        entry,
        bytecode,
        gzipped: flags & FLAG_BC_GZIPPED != 0,
        brain,
        brain_dim,
        crc32,
    })
}

/* ------------------------------------------------------------------ */
/* Assembly                                                            */
/* ------------------------------------------------------------------ */

/// Byte offset of the Mach-O code-signature blob (`LC_CODE_SIGNATURE.dataoff`).
///
/// codesign appends its blob after everything else, so on macOS the trailer is
/// NOT at EOF — it ends exactly where the blob begins. Parsing this is what
/// lets one artifact satisfy both the kernel's signature rules and the
/// payload's fixed tail. `None` for anything that is not 64-bit little-endian
/// Mach-O (ELF, PE, a test fixture).
fn macho_signature_offset(bytes: &[u8]) -> Option<u64> {
    const LC_CODE_SIGNATURE: u32 = 0x1d;
    const MH_MAGIC_64: u32 = 0xFEED_FACF;
    if bytes.len() < 32 {
        return None;
    }
    if u32::from_le_bytes(bytes[0..4].try_into().ok()?) != MH_MAGIC_64 {
        return None;
    }
    let ncmds = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let mut off = 32usize;
    for _ in 0..ncmds {
        if off + 8 > bytes.len() {
            return None;
        }
        let cmd = u32::from_le_bytes(bytes[off..off + 4].try_into().ok()?);
        let cmdsize = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().ok()?) as usize;
        if cmdsize < 8 || off + cmdsize > bytes.len() {
            return None;
        }
        if cmd == LC_CODE_SIGNATURE && cmdsize >= 16 {
            return Some(u32::from_le_bytes(bytes[off + 8..off + 12].try_into().ok()?) as u64);
        }
        off += cmdsize;
    }
    None
}

/// A trailer found in an image, at the offset where it starts.
#[derive(Clone, Copy, Debug)]
struct Found {
    trailer_at: usize,
    trailer: Trailer,
}

/// Find a trailer that ends at or just before `end`.
///
/// codesign aligns its blob to 16 bytes, so up to 15 bytes of padding sit
/// between our trailer and `LC_CODE_SIGNATURE.dataoff`. Scan the small window
/// backwards instead of assuming the trailer touches the blob.
fn trailer_ending_before(bytes: &[u8], end: usize) -> Option<Found> {
    let hi = end.checked_sub(TRAILER_LEN)?;
    let lo = end.saturating_sub(64);
    if hi > bytes.len() {
        return None;
    }
    for p in (lo..=hi).rev() {
        if p + TRAILER_LEN > bytes.len() {
            continue;
        }
        if &bytes[p..p + 12] == TRAILER_MAGIC {
            if let Some(trailer) = Trailer::parse(&bytes[p..p + TRAILER_LEN]) {
                return Some(Found {
                    trailer_at: p,
                    trailer,
                });
            }
        }
    }
    None
}

/// Find a trailer in an in-memory image: at (or within a few bytes of) EOF,
/// or — on a signed Mach-O — before the code-signature blob.
///
/// The tolerance at EOF matters for recompilation: `codesign
/// --remove-signature` leaves its own alignment padding after the trailer, so
/// "exactly the last 32 bytes" misses it and every recompile would bake the
/// previous payload into the new base (a ~1 KB leak per compile).
fn find_trailer(bytes: &[u8]) -> Option<Found> {
    if let Some(found) = trailer_ending_before(bytes, bytes.len()) {
        return Some(found);
    }
    let sig = macho_signature_offset(bytes)? as usize;
    if sig > bytes.len() {
        return None;
    }
    trailer_ending_before(bytes, sig)
}

/// Number of bytes a base binary occupies before any payload it may carry.
///
/// A valid trailer proves the bytes before it are an earlier payload; dropping
/// them is what makes recompilation idempotent instead of nesting agents
/// (base + payload1 + payload2 + …).
pub fn base_len(bytes: &[u8]) -> usize {
    match find_trailer(bytes) {
        Some(f) if f.trailer.payload_len <= f.trailer_at as u64 => {
            f.trailer_at - f.trailer.payload_len as usize
        }
        _ => bytes.len(),
    }
}

/// base ‖ payload ‖ trailer — the entire artifact, before any code signature.
pub fn assemble(base: &[u8], payload: &[u8]) -> Vec<u8> {
    let trailer = Trailer {
        version: TRAILER_VERSION,
        flags: 0,
        payload_len: payload.len() as u64,
        crc32: crc32(payload),
    };
    let mut out = Vec::with_capacity(base.len() + payload.len() + TRAILER_LEN);
    out.extend_from_slice(base);
    out.extend_from_slice(payload);
    out.extend_from_slice(&trailer.to_bytes());
    out
}

/// Decode the payload of an in-memory executable image.
pub fn embedded_payload(bytes: &[u8]) -> Result<Option<EmbeddedPayload>, String> {
    let Some(f) = find_trailer(bytes) else {
        return Ok(None);
    };
    let t = f.trailer;
    if t.version != TRAILER_VERSION {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: this executable carries trailer v{}; this runtime speaks v{TRAILER_VERSION} — \
             rebuild it with this binary",
            t.version
        ));
    }
    if t.payload_len > f.trailer_at as u64 {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: the embedded payload is truncated (trailer claims {} bytes, file holds {})",
            t.payload_len, f.trailer_at
        ));
    }
    let start = f.trailer_at - t.payload_len as usize;
    let body = &bytes[start..f.trailer_at];
    if crc32(body) != t.crc32 {
        return Err(
            "ERR_BINARY_CORRUPTED: the embedded payload is corrupt (CRC32 mismatch) — rebuild the agent".to_string(),
        );
    }
    decode_payload(body, t.crc32).map(Some)
}

/// Read this executable's payload, if it has one.
///
/// Reads the tail (or, on a signed Mach-O, the 32 bytes before the signature
/// blob) and then the payload — the fast path the plan budgets for, with no
/// full-file scan. `SOFUU_PLAIN` in the environment disables payload mode
/// (useful when a compiled agent has to be inspected or recompiled and its
/// program would otherwise take over).
pub fn detect_embedded_payload() -> Result<Option<EmbeddedPayload>, String> {
    if std::env::var_os("SOFUU_PLAIN").is_some_and(|v| !v.is_empty() && v != "0") {
        return Ok(None);
    }
    /* Not being able to read our own file means "cannot prove there is a
     * payload", which must not make every CLI command fail — only a payload
     * that IS there and does not check out is fatal. */
    let Ok(exe) = std::env::current_exe() else {
        return Ok(None);
    };
    let Ok(mut f) = std::fs::File::open(&exe) else {
        return Ok(None);
    };
    let Ok(md) = f.metadata() else {
        return Ok(None);
    };
    let len = md.len();
    if len < TRAILER_LEN as u64 {
        return Ok(None);
    }

    let tail_len = (len.min(64) as usize).max(1);
    let mut tail = vec![0u8; tail_len];
    if f.seek(SeekFrom::Start(len - tail_len as u64)).is_err()
        || f.read_exact(&mut tail).is_err()
    {
        return Ok(None);
    }
    let tail_base = len as usize - tail_len;
    let mut found = trailer_ending_before(&tail, tail.len()).map(|local| Found {
        trailer_at: tail_base + local.trailer_at,
        trailer: local.trailer,
    });

    if found.is_none() {
        /* A signed Mach-O: the trailer sits before the signature blob. Only
         * the load commands have to be read to learn where that is. */
        let head_len = len.min(64 * 1024) as usize;
        let mut head = vec![0u8; head_len];
        if f.seek(SeekFrom::Start(0)).is_err() || f.read_exact(&mut head).is_err() {
            return Ok(None);
        }
        if let Some(sig) = macho_signature_offset(&head) {
            if sig <= len && sig >= TRAILER_LEN as u64 {
                /* codesign pads its blob to a 16-byte boundary, so read the
                 * window before it and scan, exactly like the in-memory path. */
                let win_start = sig.saturating_sub(64);
                let win_len = (sig - win_start) as usize;
                let mut win = vec![0u8; win_len];
                f.seek(SeekFrom::Start(win_start))
                    .map_err(|e| format!("cannot seek {}: {e}", exe.display()))?;
                f.read_exact(&mut win)
                    .map_err(|e| format!("cannot read {}: {e}", exe.display()))?;
                if let Some(local) = trailer_ending_before(&win, win.len()) {
                    found = Some(Found {
                        trailer_at: win_start as usize + local.trailer_at,
                        trailer: local.trailer,
                    });
                }
            }
        }
    }

    let Some(found) = found else {
        return Ok(None);
    };
    let t = found.trailer;
    if t.version != TRAILER_VERSION {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: this executable carries trailer v{}; this runtime speaks v{TRAILER_VERSION} — \
             rebuild it with this binary",
            t.version
        ));
    }
    if t.payload_len > found.trailer_at as u64 {
        return Err(format!(
            "ERR_BINARY_CORRUPTED: the embedded payload is truncated (trailer claims {} bytes, file holds {})",
            t.payload_len, found.trailer_at
        ));
    }
    let start = found.trailer_at as u64 - t.payload_len;
    f.seek(SeekFrom::Start(start))
        .map_err(|e| format!("cannot seek {}: {e}", exe.display()))?;
    let mut body = vec![0u8; t.payload_len as usize];
    f.read_exact(&mut body)
        .map_err(|e| format!("cannot read the payload of {}: {e}", exe.display()))?;
    if crc32(&body) != t.crc32 {
        return Err(
            "ERR_BINARY_CORRUPTED: the embedded payload is corrupt (CRC32 mismatch) — rebuild the agent".to_string(),
        );
    }
    decode_payload(&body, t.crc32).map(Some)
}

/* ------------------------------------------------------------------ */
/* Brain extraction                                                    */
/* ------------------------------------------------------------------ */

#[derive(Clone, Debug)]
pub struct MountedBrain {
    pub path: String,
    pub dim: u32,
}

/// Where an embedded brain lives once it has been written out.
///
/// An existing file is never overwritten: the payload is a *seed*, and a
/// program that has been remembering things owns the file now. Replacing it
/// on every launch would silently throw those memories away.
fn seed_paths(payload: &EmbeddedPayload) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("SOFUU_EMBEDDED_BRAIN") {
        if !p.is_empty() {
            out.push(PathBuf::from(p));
            return out;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let stem = exe
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "agent".to_string());
        if let Some(dir) = exe.parent() {
            out.push(dir.join(format!("{stem}.brain.qtsq")));
        }
    }
    out.push(std::env::temp_dir().join(format!("sofuu-embedded-{:08x}.qtsq", payload.crc32)));
    out
}

/// Write the embedded brain beside the executable (or to a temp file when
/// that directory is read-only) and return where it landed.
pub fn materialize_brain(payload: &EmbeddedPayload) -> Result<Option<MountedBrain>, String> {
    let Some(bytes) = payload.brain.as_deref() else {
        return Ok(None);
    };
    let mut last_err = String::new();
    for path in seed_paths(payload) {
        if path.is_file() {
            /* Already seeded (and possibly grown since): use it as it is. */
            return Ok(Some(MountedBrain {
                path: path.to_string_lossy().into_owned(),
                dim: payload.brain_dim,
            }));
        }
        match unpack_brain(bytes, &path) {
            Ok(()) => {
                return Ok(Some(MountedBrain {
                    path: path.to_string_lossy().into_owned(),
                    dim: payload.brain_dim,
                }))
            }
            Err(e) => last_err = format!("{}: {e}", path.display()),
        }
    }
    Err(if last_err.is_empty() {
        "no writable location for the embedded brain".to_string()
    } else {
        last_err
    })
}

/* ------------------------------------------------------------------ */
/* Brain archive                                                       */
/* ------------------------------------------------------------------ */

/// A `.qtsq` store is a manifest file plus `<path>.d/` segments, so carrying
/// "a brain" means carrying both. The payload stores them in one flat archive;
/// the format is deliberately trivial (names, lengths, bytes) because its
/// reader runs before anything else on every launch.
const BRAIN_ARCHIVE_MAGIC: &[u8; 8] = b"SOFUBR01";
const BRAIN_ARCHIVE_VERSION: u16 = 1;

fn pack_brain(path: &str) -> Result<Vec<u8>, CompileError> {
    let manifest = std::fs::read(path)
        .map_err(|e| err("E_BRAIN", format!("cannot read {path}: {e}")))?;
    let mut files: Vec<(String, Vec<u8>)> = vec![("store".to_string(), manifest)];

    let dir = format!("{path}.d");
    match std::fs::read_dir(&dir) {
        Ok(rd) => {
            let mut paths: Vec<PathBuf> = rd
                .flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.path())
                .collect();
            paths.sort();
            for p in paths {
                let name = p
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let data = std::fs::read(&p)
                    .map_err(|e| err("E_BRAIN", format!("cannot read {}: {e}", p.display())))?;
                files.push((format!("d/{name}"), data));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => { /* flat v1 brain */ }
        Err(e) => return Err(err("E_BRAIN", format!("cannot read {dir}: {e}"))),
    }

    if files.len() > u16::MAX as usize {
        return Err(err("E_BRAIN", format!("{path}: too many segment files")));
    }
    let mut out = Vec::new();
    out.extend_from_slice(BRAIN_ARCHIVE_MAGIC);
    out.extend_from_slice(&BRAIN_ARCHIVE_VERSION.to_le_bytes());
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    for (name, data) in files {
        if name.len() > u16::MAX as usize || data.len() > u32::MAX as usize {
            return Err(err("E_BRAIN", format!("{path}: a segment is too large")));
        }
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&data);
    }
    Ok(out)
}

/// Write an archive back out as the store it came from.
///
/// Never overwrites: `store` is the manifest, `d/<name>` is one segment
/// (plain file names only — a payload cannot reach outside the target).
fn unpack_brain(bytes: &[u8], target: &Path) -> Result<(), String> {
    if bytes.len() < 12 || &bytes[..8] != BRAIN_ARCHIVE_MAGIC {
        return Err("the embedded brain is not a Sofuu brain archive".to_string());
    }
    let ver = u16::from_le_bytes([bytes[8], bytes[9]]);
    if ver != BRAIN_ARCHIVE_VERSION {
        return Err(format!("brain archive format v{ver} is not understood"));
    }
    let count = u16::from_le_bytes([bytes[10], bytes[11]]) as usize;
    let dir = PathBuf::from(format!("{}.d", target.display()));
    let mut off = 12usize;
    for _ in 0..count {
        if off + 6 > bytes.len() {
            return Err("the embedded brain archive is truncated".to_string());
        }
        let name_len = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize;
        let data_len =
            u32::from_le_bytes(bytes[off + 2..off + 6].try_into().unwrap()) as usize;
        off += 6;
        let name_end = off
            .checked_add(name_len)
            .ok_or_else(|| "the embedded brain archive is truncated".to_string())?;
        let data_end = name_end
            .checked_add(data_len)
            .ok_or_else(|| "the embedded brain archive is truncated".to_string())?;
        if data_end > bytes.len() {
            return Err("the embedded brain archive is truncated".to_string());
        }
        let name = std::str::from_utf8(&bytes[off..name_end])
            .map_err(|_| "the embedded brain archive has a non-UTF-8 name".to_string())?;
        let data = &bytes[name_end..data_end];
        off = data_end;

        let dest = if name == "store" {
            target.to_path_buf()
        } else if let Some(rest) = name.strip_prefix("d/") {
            if rest.is_empty() || rest.contains('/') || rest.contains('\\') || rest == ".." {
                return Err(format!("the embedded brain archive has an unsafe name '{name}'"));
            }
            dir.join(rest)
        } else {
            return Err(format!("the embedded brain archive has an unknown name '{name}'"));
        };
        if dest.is_file() {
            continue; /* a partially seeded store keeps what it has */
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest)
        {
            Ok(mut f) => {
                if let Err(e) = f.write_all(data).and_then(|_| f.sync_all()) {
                    let _ = std::fs::remove_file(&dest);
                    return Err(format!("cannot write {}: {e}", dest.display()));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("cannot write {}: {e}", dest.display())),
        }
    }
    Ok(())
}

/* ------------------------------------------------------------------ */
/* Compile                                                             */
/* ------------------------------------------------------------------ */

#[derive(Debug)]
pub struct CompileError {
    pub code: &'static str,
    pub message: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CompileError {}

fn err(code: &'static str, message: impl Into<String>) -> CompileError {
    CompileError {
        code,
        message: message.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CompileProfile {
    /// Lean agent base: smallest possible binary, excludes local GGUF inference & full CLI commands.
    #[default]
    Lean,
    /// Full base: includes local inference engine, robotics, and full CLI subcommands.
    Full,
    /// Micro base: ultra-lean MCP server / single-tool runtime.
    Micro,
}

impl CompileProfile {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lean => "lean",
            Self::Full => "full",
            Self::Micro => "micro",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    /// `-o` — defaults to the entry's stem in the current directory.
    pub output: Option<String>,
    /// `--embed-brain` — a `.qtsq` file to carry inside the artifact.
    pub embed_brain: Option<String>,
    /// `--no-strip` — keep the base binary's symbols.
    pub no_strip: bool,
    /// Compilation profile: Lean (default, smallest possible) vs Full.
    pub profile: CompileProfile,
    /// Custom base executable path to append the payload to.
    pub base: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CompileReport {
    pub output: String,
    pub total: u64,
    pub base: u64,
    pub payload: u64,
    pub bytecode_raw: u64,
    pub bytecode_stored: u64,
    pub gzipped: bool,
    pub brain: Option<u64>,
    pub signed: bool,
    pub stripped: bool,
    pub profile: &'static str,
    pub base_source: String,
}

fn resolve_base_binary(
    opts: &CompileOptions,
    self_path: &Path,
    self_bytes: &[u8],
) -> (Vec<u8>, &'static str, String) {
    if let Some(custom) = &opts.base {
        let p = Path::new(custom);
        if let Ok(bytes) = std::fs::read(p) {
            return (bytes, "custom", p.to_string_lossy().into_owned());
        }
    }

    if opts.profile == CompileProfile::Micro {
        let mut candidates: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running binary
        if let Some(parent) = self_path.parent() {
            candidates.push(parent.join("sofuu-micro"));
            candidates.push(parent.join("sofuu-mcp"));
        }
        // 2. target/release in workspace
        candidates.push(PathBuf::from("target/release/sofuu-micro"));
        candidates.push(PathBuf::from("target/release/sofuu-mcp"));
        // 3. Current working directory
        candidates.push(PathBuf::from("./sofuu-micro"));
        candidates.push(PathBuf::from("./sofuu-mcp"));
        // 4. ~/.sofuu/bin/sofuu-micro
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".sofuu/bin/sofuu-micro"));
        }
        // 5. Fallback candidates to lean agent base if micro not yet built
        if let Some(parent) = self_path.parent() {
            candidates.push(parent.join("sofuu-agent"));
        }
        candidates.push(PathBuf::from("target/release/sofuu-agent"));
        candidates.push(PathBuf::from("./sofuu-agent"));

        for candidate in candidates {
            if candidate.is_file() {
                if let Ok(bytes) = std::fs::read(&candidate) {
                    return (bytes, "micro", candidate.to_string_lossy().into_owned());
                }
            }
        }
    }

    if opts.profile == CompileProfile::Lean {
        // Search candidates for sofuu-agent lean runtime base
        let mut candidates: Vec<PathBuf> = Vec::new();

        // 1. Same directory as the running binary
        if let Some(parent) = self_path.parent() {
            candidates.push(parent.join("sofuu-agent"));
            candidates.push(parent.join("sofuu-lean"));
        }
        // 2. target/release in workspace
        candidates.push(PathBuf::from("target/release/sofuu-agent"));
        // 3. Current working directory
        candidates.push(PathBuf::from("./sofuu-agent"));
        // 4. ~/.sofuu/bin/sofuu-agent
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".sofuu/bin/sofuu-agent"));
        }

        for candidate in candidates {
            if candidate.is_file() {
                if let Ok(bytes) = std::fs::read(&candidate) {
                    return (bytes, "lean", candidate.to_string_lossy().into_owned());
                }
            }
        }
    }

    (self_bytes.to_vec(), opts.profile.as_str(), self_path.to_string_lossy().into_owned())
}

/// Compile `entry` into a standalone executable.
pub fn compile(entry: &str, opts: &CompileOptions) -> Result<CompileReport, CompileError> {
    let entry_path = Path::new(entry);
    if !entry_path.is_file() {
        return Err(err(
            "E_ENTRY",
            format!("{entry} is not a file (sofuu compile needs an entry .js/.ts)"),
        ));
    }

    let output = resolve_output(entry_path, opts.output.as_deref())?;
    let output_str = output.to_string_lossy().into_owned();

    let entry_bytes = std::fs::read(entry_path)
        .map_err(|e| err("E_ENTRY", format!("cannot read {entry}: {e}")))?;

    let (bytecode_stored, gzipped, bytecode_raw, brain, brain_dim) = if let Ok(Some(existing)) = embedded_payload(&entry_bytes) {
        let (br, dim) = if opts.embed_brain.is_some() {
            let (bytes, dim) = read_brain(opts.embed_brain.as_deref().unwrap())?;
            (Some(bytes), dim)
        } else {
            (existing.brain, existing.brain_dim)
        };
        let raw_len = existing.bytecode.len() as u64;
        (existing.bytecode, existing.gzipped, raw_len, br, dim)
    } else {
        /* 1. Bundle: one self-contained script, TypeScript stripped per module. */
        let source = if entry_path.extension().is_some_and(|ext| ext == "kaze") {
            generate_kaze_agent_bootstrap(entry_path, &entry_bytes)?
        } else {
            crate::bundler::bundle(entry_path, &crate::npm::npm_resolve)
                .map_err(|e| err("E_BUNDLE", format!("{entry}: {e}")))?
        };

        /* 2. Compile the bundle. A bare QuickJS runtime is enough: this only has
         *    to parse, so booting the engine here would buy nothing. */
        let bytecode = unsafe { compile_source(&source, entry) }
            .map_err(|m| err("E_COMPILE", m))?;
        let bytecode_raw = bytecode.len() as u64;

        /* 3. Store the bytecode compressed when that actually saves bytes.
         *    `eval_compiled_and_run` sniffs the gzip magic, and the driver
         *    snapshots use the same convention (`.bc.gz`). */
        let (bc_stored, gz) = maybe_gzip(&bytecode);

        /* 4. Optional brain. Refuse early rather than emit an artifact that
         *    cannot open it. */
        let (br, dim) = match opts.embed_brain.as_deref() {
            Some(b) => {
                let (bytes, dim) = read_brain(b)?;
                (Some(bytes), dim)
            }
            None => (None, 0),
        };
        (bc_stored, gz, bytecode_raw, br, dim)
    };

    /* 5. Base binary resolution:
     *    By default (CompileProfile::Lean), the compiler selects the leanest
     *    standalone agent base (`sofuu-agent`), dropping interactive TUI chat,
     *    developer CLI subcommands, and heavy inference engines.
     *    If `--full` / `--with-inference` is selected, or if a custom `--base`
     *    is provided, it uses that binary instead. */
    let self_path = std::env::current_exe()
        .map_err(|e| err("E_SELF", format!("cannot locate this executable: {e}")))?;
    let self_bytes = std::fs::read(&self_path)
        .map_err(|e| err("E_SELF", format!("cannot read {}: {e}", self_path.display())))?;

    let (base_source_bytes, profile_str, base_source_desc) =
        resolve_base_binary(opts, &self_path, &self_bytes);

    /* Refuse to clobber the binary we are running from: the rename would
     * succeed on Unix and leave the process mapped to a replaced inode. */
    if same_file(&output, &self_path) || opts.base.as_deref().is_some_and(|b| same_file(&output, Path::new(b))) {
        return Err(err(
            "E_OUTPUT",
            format!("{output_str} is the running or base binary — choose another output path"),
        ));
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                err("E_OUTPUT", format!("cannot create {}: {e}", parent.display()))
            })?;
        }
    }

    let tmp = format!("{output_str}.tmp-{}", std::process::id());
    let staged = match stage_base(&tmp, &base_source_bytes) {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };

    /* 6. Strip the base (release builds already are; a debug build shrinks a
     *    lot). Re-read afterwards because strip rewrites __LINKEDIT sizes. */
    let stripped = !opts.no_strip && strip_binary(&tmp);
    let base = if stripped {
        std::fs::read(&tmp)
            .map_err(|e| err("E_OUTPUT", format!("cannot re-read {tmp}: {e}")))?
    } else {
        staged
    };

    let payload = encode_payload(
        entry_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| entry.to_string())
            .as_str(),
        &bytecode_stored,
        gzipped,
        brain.as_deref(),
        brain_dim,
    );

    /* On Mach-O the load commands have to grow with the file, or the kernel
     * and codesign both reject the result. Patching the base BEFORE appending
     * keeps the helper's invariant simple: the last segment ends at EOF now
     * and must end at EOF after the payload lands. */
    #[cfg(target_os = "macos")]
    let mut base = base;
    #[cfg(target_os = "macos")]
    macho_extend_last_segment(&mut base, payload.len() as u64 + TRAILER_LEN as u64)
        .map_err(|e| err("E_SIGN", e))?;

    let artifact = assemble(&base, &payload);
    let mut f = std::fs::File::create(&tmp)
        .map_err(|e| err("E_OUTPUT", format!("cannot write {tmp}: {e}")))?;
    if let Err(e) = f.write_all(&artifact).and_then(|_| f.sync_all()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err("E_OUTPUT", format!("cannot write {tmp}: {e}")));
    }
    drop(f);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755)) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err("E_OUTPUT", format!("cannot chmod {tmp}: {e}")));
        }
    }

    #[cfg(target_os = "macos")]
    let signed = match adhoc_sign(&tmp) {
        Ok(signed) => signed,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    #[cfg(not(target_os = "macos"))]
    let signed = false;

    if let Err(e) = std::fs::rename(&tmp, &output) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err("E_OUTPUT", format!("cannot publish {output_str}: {e}")));
    }

    let total_on_disk = std::fs::metadata(&output)
        .map(|m| m.len())
        .unwrap_or(artifact.len() as u64);

    Ok(CompileReport {
        output: output_str,
        total: total_on_disk,
        base: base.len() as u64,
        payload: payload.len() as u64,
        bytecode_raw,
        bytecode_stored: bytecode_stored.len() as u64,
        gzipped,
        brain: brain.as_ref().map(|b| b.len() as u64),
        signed,
        stripped,
        profile: profile_str,
        base_source: base_source_desc,
    })
}

/// Write the running executable to `tmp` and return the base bytes a payload
/// can be appended to.
///
/// On macOS that means dropping the code signature (`codesign` rewrites the
/// load commands and truncates the blob) and then, if an earlier compile left
/// a payload, truncating it and shrinking __LINKEDIT back to where it was.
/// Elsewhere the trailer is at EOF to begin with and the payload is pure
/// trailing data.
#[cfg(target_os = "macos")]
fn stage_base(tmp: &str, self_bytes: &[u8]) -> Result<Vec<u8>, CompileError> {
    std::fs::write(tmp, self_bytes)
        .map_err(|e| err("E_OUTPUT", format!("cannot stage {tmp}: {e}")))?;
    /* Best effort: an unsigned base has nothing to remove, and that is not an
     * error. A base that IS signed but cannot be stripped fails loudly below
     * (the buried signature makes the final signing step reject the file). */
    let _ = std::process::Command::new("codesign")
        .args(["--remove-signature", tmp])
        .output();

    let mut staged = std::fs::read(tmp)
        .map_err(|e| err("E_OUTPUT", format!("cannot read {tmp}: {e}")))?;
    let n = base_len(&staged);
    if n < staged.len() {
        let dropped = (staged.len() - n) as u64;
        /* Shrink BEFORE truncating: the helper finds the segment by "its file
         * range ends at EOF", which is only true while the old payload is
         * still in the buffer. */
        macho_shrink_last_segment(&mut staged, dropped)
            .map_err(|e| err("E_SIGN", e))?;
        staged.truncate(n);
        std::fs::write(tmp, &staged)
            .map_err(|e| err("E_OUTPUT", format!("cannot stage {tmp}: {e}")))?;
    }
    Ok(staged)
}

#[cfg(not(target_os = "macos"))]
fn stage_base(tmp: &str, self_bytes: &[u8]) -> Result<Vec<u8>, CompileError> {
    /* ELF and PE both tolerate trailing data, so the base is simply the file
     * with any earlier payload truncated off; the temp file carries THAT so
     * the strip pass sees the real base. */
    let base = self_bytes[..base_len(self_bytes)].to_vec();
    std::fs::write(tmp, &base)
        .map_err(|e| err("E_OUTPUT", format!("cannot stage {tmp}: {e}")))?;
    Ok(base)
}

/// Locate the LC_SEGMENT_64 whose file range ends at EOF and return its
/// load-command offset. The last segment is always __LINKEDIT on a linked
/// executable, which is exactly the segment a trailing payload has to live in
/// once the file grows.
#[cfg(target_os = "macos")]
fn last_segment_command(bytes: &[u8]) -> Result<usize, String> {
    const LC_SEGMENT_64: u32 = 0x19;
    const MH_MAGIC_64: u32 = 0xFEED_FACF;
    if bytes.len() < 32 {
        return Err("not a Mach-O executable (too short)".to_string());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MH_MAGIC_64 {
        return Err(format!(
            "unsupported Mach-O magic {magic:#010x} — only 64-bit little-endian \
             executables can carry a payload"
        ));
    }
    let ncmds = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let eof = bytes.len() as u64;
    let mut off = 32usize;
    let mut found: Option<usize> = None;
    for _ in 0..ncmds {
        if off + 8 > bytes.len() {
            return Err("truncated Mach-O load commands".to_string());
        }
        let cmd = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        if cmdsize < 8 || off + cmdsize > bytes.len() {
            return Err("corrupt Mach-O load command size".to_string());
        }
        if cmd == LC_SEGMENT_64 && cmdsize >= 72 {
            let fileoff = u64::from_le_bytes(bytes[off + 40..off + 48].try_into().unwrap());
            let filesize = u64::from_le_bytes(bytes[off + 48..off + 56].try_into().unwrap());
            if fileoff.saturating_add(filesize) == eof {
                found = Some(off);
            }
        }
        off += cmdsize;
    }
    found.ok_or_else(|| "no Mach-O segment covers the end of the file".to_string())
}

#[cfg(target_os = "macos")]
fn macho_resize_last_segment(bytes: &mut [u8], delta: i64) -> Result<(), String> {
    let off = last_segment_command(bytes)?;
    let vmsize = u64::from_le_bytes(bytes[off + 32..off + 40].try_into().unwrap()) as i64;
    let filesize = u64::from_le_bytes(bytes[off + 48..off + 56].try_into().unwrap()) as i64;
    let new_vm = vmsize.checked_add(delta).filter(|v| *v >= 0);
    let new_fs = filesize.checked_add(delta).filter(|v| *v >= 0);
    let (Some(new_vm), Some(new_fs)) = (new_vm, new_fs) else {
        return Err("the payload would make the Mach-O segment sizes negative".to_string());
    };
    bytes[off + 32..off + 40].copy_from_slice(&(new_vm as u64).to_le_bytes());
    bytes[off + 48..off + 56].copy_from_slice(&(new_fs as u64).to_le_bytes());
    Ok(())
}

#[cfg(target_os = "macos")]
fn macho_extend_last_segment(bytes: &mut [u8], extra: u64) -> Result<(), String> {
    macho_resize_last_segment(bytes, extra as i64)
}

#[cfg(target_os = "macos")]
fn macho_shrink_last_segment(bytes: &mut [u8], dropped: u64) -> Result<(), String> {
    macho_resize_last_segment(bytes, -(dropped as i64))
}

fn resolve_output(entry: &Path, requested: Option<&str>) -> Result<PathBuf, CompileError> {
    let output = match requested {
        Some(o) if !o.is_empty() => PathBuf::from(o),
        Some(_) => return Err(err("E_USAGE", "-o requires a path")),
        None => {
            let stem = entry.file_stem().ok_or_else(|| {
                err(
                    "E_USAGE",
                    format!("cannot derive an output name from {}", entry.display()),
                )
            })?;
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(stem)
        }
    };
    if output.is_dir() {
        return Err(err(
            "E_OUTPUT",
            format!("{} is a directory", output.display()),
        ));
    }
    if same_file(&output, entry) {
        return Err(err(
            "E_OUTPUT",
            format!(
                "the output would overwrite the entry ({}) — pass -o",
                entry.display()
            ),
        ));
    }
    Ok(output)
}

fn same_file(a: &Path, b: &Path) -> bool {
    let ca = std::fs::canonicalize(a).ok();
    let cb = std::fs::canonicalize(b).ok();
    match (ca, cb) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

fn maybe_gzip(bytecode: &[u8]) -> (Vec<u8>, bool) {
    if bytecode.len() < 1024 {
        return (bytecode.to_vec(), false);
    }
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    if enc.write_all(bytecode).is_err() {
        return (bytecode.to_vec(), false);
    }
    match enc.finish() {
        Ok(z) if z.len() + 64 < bytecode.len() => (z, true),
        _ => (bytecode.to_vec(), false),
    }
}

/// Compile a script to QuickJS bytecode, reporting the parse error verbatim.
unsafe fn compile_source(src: &str, name: &str) -> Result<Vec<u8>, String> {
    use std::ffi::CString;
    let rt = unsafe { qjs::JS_NewRuntime() };
    if rt.is_null() {
        return Err("QuickJS runtime allocation failed".to_string());
    }
    let ctx = unsafe { qjs::JS_NewContext(rt) };
    if ctx.is_null() {
        unsafe { qjs::JS_FreeRuntime(rt) };
        return Err("QuickJS context allocation failed".to_string());
    }
    let result = (|| {
        let c_src = CString::new(src)
            .map_err(|_| "the bundle contains a NUL byte".to_string())?;
        let c_name = CString::new(name).unwrap_or_default();
        let fun = unsafe {
            qjs::JS_Eval(
                ctx,
                c_src.as_ptr(),
                src.len(),
                c_name.as_ptr(),
                qjs::JS_EVAL_TYPE_GLOBAL | qjs::JS_EVAL_FLAG_COMPILE_ONLY,
            )
        };
        if unsafe { qjs::is_exception(fun) } {
            let exc = unsafe { qjs::JS_GetException(ctx) };
            let p = unsafe { qjs::sofuu_js_to_cstring(ctx, exc) };
            let msg = if p.is_null() {
                "compile failed".to_string()
            } else {
                let m = unsafe { std::ffi::CStr::from_ptr(p) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { qjs::sofuu_js_free_cstring(ctx, p) };
                m
            };
            unsafe {
                qjs::sofuu_js_free_value(ctx, exc);
                qjs::sofuu_js_free_value(ctx, fun);
            }
            return Err(msg);
        }
        let mut out_len: usize = 0;
        let bc = unsafe { qjs::JS_WriteObject(ctx, &mut out_len, fun, qjs::JS_WRITE_OBJ_BYTECODE) };
        unsafe { qjs::sofuu_js_free_value(ctx, fun) };
        if bc.is_null() || out_len == 0 {
            return Err("QuickJS could not serialize the program".to_string());
        }
        let bytes = unsafe { std::slice::from_raw_parts(bc, out_len) }.to_vec();
        unsafe { qjs::js_free(ctx, bc as *mut std::ffi::c_void) };
        Ok(bytes)
    })();
    unsafe {
        qjs::JS_FreeContext(ctx);
        qjs::JS_FreeRuntime(rt);
    }
    result
}

/// Read the brain file and decide its vector dimension.
///
/// The dimension is part of the artifact because `sofuu.memory.open` requires
/// it and a compiled agent cannot be asked. Both store layouts are read: the
/// segmented manifest carries it in its embedding block, and a flat v1 brain
/// carries it in the `memories` tensor shape.
fn read_brain(path: &str) -> Result<(Vec<u8>, u32), CompileError> {
    let dim = brain_dim(path)?;
    /* The store is the manifest PLUS its `.d/` segments; both travel. */
    let archive = pack_brain(path)?;
    if archive.len() <= 12 {
        return Err(err("E_BRAIN", format!("{path} carries no records")));
    }
    Ok((archive, dim))
}

#[cfg(has_qtsq)]
fn brain_dim(path: &str) -> Result<u32, CompileError> {
    use crate::rt::brain_store as bs;
    unsafe {
        match bs::read_manifest_ex(path) {
            Ok(Some((m, _))) => {
                if let Some(d) = m.embedding.get("dimension").and_then(|v| v.as_u64()) {
                    if d > 0 && d <= u32::MAX as u64 {
                        return Ok(d as u32);
                    }
                }
            }
            /* A container this key cannot open is a real error, not an
             * "unknown shape": embedding it would only move the failure to
             * the artifact's first launch. */
            Err(rc) => {
                return Err(err(
                    "E_BRAIN",
                    format!(
                        "{path}: the container exists but this machine's brain key cannot open it \
                         (codec rc {rc})"
                    ),
                ))
            }
            Ok(None) => { /* flat v1 brain — read the tensor shape below */ }
        }
        if let Ok((c, _)) = bs::container_open_ex(path) {
            let tensor = bs::read_tensor_stream(c, bs::SEGMENT_TENSOR_STREAM);
            bs::container_free(c);
            if let Some((_, dims)) = tensor {
                if dims.len() >= 2 && dims[1] > 0 {
                    return Ok(dims[1]);
                }
            }
        }
    }
    Err(err(
        "E_BRAIN",
        format!("{path}: cannot determine the brain's vector dimension"),
    ))
}

#[cfg(not(has_qtsq))]
fn brain_dim(path: &str) -> Result<u32, CompileError> {
    Err(err(
        "E_BRAIN",
        format!(
            "--embed-brain {path}: this build has no QTSQ codec, so it could neither embed nor \
             open a brain — build with QTSQ first"
        ),
    ))
}

/* Not on Mach-O: `codesign --remove-signature` leaves __LINKEDIT's
 * bookkeeping in a state `strip` refuses ("link edit information does not fill
 * the __LINKEDIT segment"), and re-stripping a binary whose payload lives in
 * that very segment buys nothing — the release profile already strips. ELF and
 * PE keep the best-effort pass. */
#[cfg(all(unix, not(target_os = "macos")))]
fn strip_binary(path: &str) -> bool {
    std::process::Command::new("strip")
        .arg("--strip-unneeded")
        .arg(path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(all(unix, not(target_os = "macos"))))]
fn strip_binary(path: &str) -> bool {
    let _ = path;
    false
}

/// Ad-hoc signature, required after appending bytes to a Mach-O: the kernel
/// validates the signature over the whole file, and the payload invalidated
/// the one cargo produced.
#[cfg(target_os = "macos")]
fn adhoc_sign(path: &str) -> Result<bool, CompileError> {
    match std::process::Command::new("codesign")
        .args(["-s", "-", "-f", path])
        .output()
    {
        Ok(out) if out.status.success() => Ok(true),
        Ok(out) => Err(err(
            "E_SIGN",
            format!(
                "codesign failed on {path}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        )),
        Err(e) => Err(err(
            "E_SIGN",
            format!("cannot run codesign ({e}) — the artifact would be killed on launch"),
        )),
    }
}

fn generate_kaze_agent_bootstrap(entry_path: &Path, kaze_bytes: &[u8]) -> Result<String, CompileError> {
    let kaze_src = String::from_utf8(kaze_bytes.to_vec())
        .map_err(|e| err("E_KAZE_SRC", format!("{}: not valid UTF-8: {e}", entry_path.display())))?;

    crate::sfs::check_source(&kaze_src, &entry_path.to_string_lossy())
        .map_err(|e| err("E_KAZE_CHECK", e.render()))?;

    let file_stem = entry_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|&s| s != "agent" && s != "main")
        .or_else(|| {
            entry_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
        })
        .unwrap_or("agent");

    let declared_grants: Vec<String> = kaze_src
        .lines()
        .find(|l| l.trim_start().starts_with("needs:"))
        .map(|l| {
            l.trim_start()["needs:".len()..]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let grants_json = serde_json::to_string(&declared_grants).unwrap_or_else(|_| "[]".to_string());

    let parsed_prog = crate::domains::kaze::parser::parse(&kaze_src).ok();
    let default_entry = parsed_prog.as_ref().and_then(|p| {
        if p.item("main").is_some() {
            Some("main".to_string())
        } else {
            p.items
                .iter()
                .find(|i| matches!(i.kind, crate::domains::kaze::ast::ItemKind::Action | crate::domains::kaze::ast::ItemKind::Filter | crate::domains::kaze::ast::ItemKind::Check))
                .map(|i| i.name.clone())
        }
    });
    let default_entry_json = serde_json::to_string(&default_entry).unwrap_or_else(|_| "null".to_string());

    let mut routine_signatures: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    if let Some(ref p) = parsed_prog {
        for item in &p.items {
            routine_signatures.insert(item.name.clone(), item.params.clone());
        }
    }
    let routine_signatures_json = serde_json::to_string(&routine_signatures).unwrap_or_else(|_| "{}".to_string());

    let kaze_source_json = serde_json::to_string(&kaze_src).unwrap_or_else(|_| "\"\"".to_string());
    let kaze_name_json = serde_json::to_string(file_stem).unwrap_or_else(|_| "\"agent\"".to_string());

    let bootstrap = format!(r##"
(function() {{
    var kazeSource = {kaze_source_json};
    var kazeName = {kaze_name_json};
    var defaultGrants = {grants_json};
    var defaultEntry = {default_entry_json};
    var routineSignatures = {routine_signatures_json};

    var argv = (typeof process !== 'undefined' && process.argv) ? process.argv.slice(2) : [];
    if (argv.indexOf('--help') !== -1 || argv.indexOf('-h') !== -1) {{
        console.log("Usage: " + kazeName + " [options] [positional args...]\n\n" +
            "Autonomous AI Agent powered by Kaze & Sofuu.\n\n" +
            "Options:\n" +
            "  -c, --config <file>    Load configuration from JSON file (default: ./agent.config.json)\n" +
            "  --arg key=val          Pass named argument (parsed as JSON primitive if valid)\n" +
            "  --entry <name>         Routine to execute (default: " + (defaultEntry ? "'" + defaultEntry + "'" : "auto") + ")\n" +
            "  -d, --daemon           Run continuous 24/7 autonomous supervisor loop\n" +
            "  --every <interval>     Supervisor cycle interval (e.g. 500ms, 2s, 10s, 1m, 1h; default: 5s)\n" +
            "  --interval <interval>  Alias for --every\n" +
            "  --fuel <n>             Instruction budget limit (default: 200000)\n" +
            "  --cap <name>           Grant capability backend (e.g. drone=sim)\n" +
            "  -i, --interactive      Launch interactive Agent TUI shell\n" +
            "  -h, --help             Show this help message\n\n" +
            "Configuration Precedence (Production 12-Factor):\n" +
            "  1. CLI Arguments (--arg / positionals)\n" +
            "  2. Environment Variables (AGENT_<PARAM> / SOFUU_<PARAM>)\n" +
            "  3. Configuration File (--config <path> / ./agent.config.json)\n");
        return;
    }}

    var isInteractive = argv.indexOf('--interactive') !== -1 || argv.indexOf('-i') !== -1;
    var isDaemon = argv.indexOf('--daemon') !== -1 || argv.indexOf('-d') !== -1;
    var explicitDaemon = isDaemon;
    var intervalMs = 5000;
    var intervalRaw = '5s';
    var explicitInterval = false;
    var entryRoutine = defaultEntry;
    var explicitEntry = false;
    var fuel = 200000;
    var explicitFuel = false;
    var explicitConfigFile = null;
    var cliArgs = {{}};
    var args = {{}};
    var positionals = [];

    function parseInterval(s) {{
        if (!s) return 5000;
        s = ('' + s).trim().toLowerCase();
        if (s.endsWith('ms')) {{
            return Math.max(10, parseInt(s.slice(0, -2), 10) || 5000);
        }}
        if (s.endsWith('s')) {{
            return Math.max(100, Math.round((parseFloat(s.slice(0, -1)) || 5) * 1000));
        }}
        if (s.endsWith('m')) {{
            return Math.max(1000, Math.round((parseFloat(s.slice(0, -1)) || 1) * 60 * 1000));
        }}
        if (s.endsWith('h')) {{
            return Math.max(1000, Math.round((parseFloat(s.slice(0, -1)) || 1) * 3600 * 1000));
        }}
        if (s.endsWith('d')) {{
            return Math.max(1000, Math.round((parseFloat(s.slice(0, -1)) || 1) * 86400 * 1000));
        }}
        var num = parseFloat(s);
        if (!isNaN(num) && num > 0) {{
            return num >= 100 ? Math.round(num) : Math.round(num * 1000);
        }}
        return 5000;
    }}

    for (var i = 0; i < argv.length; i++) {{
        var a = argv[i];
        if (a === '--interactive' || a === '-i') continue;
        if (a === '--daemon' || a === '-d') {{
            isDaemon = true;
            explicitDaemon = true;
            continue;
        }}
        if ((a === '--every' || a === '--interval') && i + 1 < argv.length) {{
            intervalRaw = argv[++i];
            intervalMs = parseInterval(intervalRaw);
            isDaemon = true;
            explicitInterval = true;
            explicitDaemon = true;
        }} else if (a.startsWith('--every=') || a.startsWith('--interval=')) {{
            intervalRaw = a.slice(a.indexOf('=') + 1);
            intervalMs = parseInterval(intervalRaw);
            isDaemon = true;
            explicitInterval = true;
            explicitDaemon = true;
        }} else if ((a === '--config' || a === '-c') && i + 1 < argv.length) {{
            explicitConfigFile = argv[++i];
        }} else if (a.startsWith('--config=')) {{
            explicitConfigFile = a.slice(9);
        }} else if (a.startsWith('-c=')) {{
            explicitConfigFile = a.slice(3);
        }} else if (a === '--entry' && i + 1 < argv.length) {{
            entryRoutine = argv[++i];
            explicitEntry = true;
        }} else if (a.startsWith('--entry=')) {{
            entryRoutine = a.slice(8);
            explicitEntry = true;
        }} else if (a.startsWith('--cap')) {{
            var capVal = a.indexOf('=') !== -1 ? a.slice(a.indexOf('=') + 1) : argv[++i];
            if (capVal) defaultGrants.push(capVal.split('=')[0].trim());
        }} else if (a.startsWith('--arg')) {{
            var pair = a.indexOf('=') !== -1 ? a.slice(a.indexOf('=') + 1) : argv[++i];
            if (pair) {{
                var eq = pair.indexOf('=');
                if (eq > 0) {{
                    var k = pair.slice(0, eq).trim();
                    var v = pair.slice(eq + 1).trim();
                    try {{ cliArgs[k] = JSON.parse(v); }} catch (_) {{ cliArgs[k] = v; }}
                }}
            }}
        }} else if (a.startsWith('--fuel=') || (a === '--fuel' && i + 1 < argv.length)) {{
            fuel = parseInt(a.indexOf('=') !== -1 ? a.slice(7) : argv[++i], 10) || 200000;
            explicitFuel = true;
        }} else if (!a.startsWith('-')) {{
            positionals.push(a);
        }}
    }}

    // --- Production Configuration Pipeline (Tiered Precedence) ---
    // Tier 1: Config File (Base Configuration)
    var configFileToLoad = explicitConfigFile;
    if (!configFileToLoad && typeof sofuu !== 'undefined' && sofuu.fs && sofuu.fs.existsSync) {{
        var autoCandidates = [
            './agent.config.json',
            './' + kazeName + '.config.json',
            './config.json'
        ];
        for (var cIdx = 0; cIdx < autoCandidates.length; cIdx++) {{
            if (sofuu.fs.existsSync(autoCandidates[cIdx])) {{
                configFileToLoad = autoCandidates[cIdx];
                break;
            }}
        }}
    }}

    if (configFileToLoad) {{
        try {{
            if (typeof sofuu !== 'undefined' && sofuu.fs && sofuu.fs.readFileSync) {{
                var rawCfg = sofuu.fs.readFileSync(configFileToLoad, 'utf8');
                var cfg = JSON.parse(rawCfg);
                if (cfg && typeof cfg === 'object') {{
                    if (cfg.entry && !explicitEntry) entryRoutine = cfg.entry;
                    if (cfg.fuel && !explicitFuel) fuel = parseInt(cfg.fuel, 10) || fuel;
                    if ((cfg.every || cfg.interval) && !explicitInterval) {{
                        intervalRaw = '' + (cfg.every || cfg.interval);
                        intervalMs = parseInterval(intervalRaw);
                        if (!explicitDaemon) isDaemon = true;
                    }}
                    if (cfg.daemon !== undefined && !explicitDaemon) isDaemon = !!cfg.daemon;
                    if (Array.isArray(cfg.grants)) {{
                        for (var g = 0; g < cfg.grants.length; g++) {{
                            if (defaultGrants.indexOf(cfg.grants[g]) === -1) defaultGrants.push(cfg.grants[g]);
                        }}
                    }}
                    var cfgArgs = cfg.args && typeof cfg.args === 'object' ? cfg.args : cfg;
                    for (var k in cfgArgs) {{
                        if (k !== 'args' && k !== 'entry' && k !== 'fuel' && k !== 'interval' && k !== 'every' && k !== 'daemon' && k !== 'grants') {{
                            if (args[k] === undefined) {{
                                args[k] = cfgArgs[k];
                            }}
                        }}
                    }}
                }}
            }}
        }} catch (cfgErr) {{
            console.error("\x1b[31m[Config Error]\x1b[0m Failed to parse configuration file '" + configFileToLoad + "': " + (cfgErr && cfgErr.message ? cfgErr.message : cfgErr));
            if (typeof process !== 'undefined' && process.exit) process.exit(1);
            return;
        }}
    }}

    // Tier 2: Environment Variables (Container / Cloud Injection)
    if (typeof process !== 'undefined' && process.env) {{
        var env = process.env;
        if (env.AGENT_ENTRY && !explicitEntry) entryRoutine = env.AGENT_ENTRY;
        if (env.AGENT_FUEL && !explicitFuel) fuel = parseInt(env.AGENT_FUEL, 10) || fuel;
        if (env.AGENT_INTERVAL && !explicitInterval) {{
            intervalRaw = env.AGENT_INTERVAL;
            intervalMs = parseInterval(intervalRaw);
            if (!explicitDaemon) isDaemon = true;
        }}
        if ((env.AGENT_DAEMON === 'true' || env.AGENT_DAEMON === '1') && !explicitDaemon) isDaemon = true;

        for (var envKey in env) {{
            var normKey = null;
            if (envKey.startsWith('AGENT_')) {{
                normKey = envKey.slice(6).toLowerCase();
            }} else if (envKey.startsWith('SOFUU_')) {{
                normKey = envKey.slice(6).toLowerCase();
            }}
            if (normKey && normKey !== 'entry' && normKey !== 'fuel' && normKey !== 'interval' && normKey !== 'every' && normKey !== 'daemon') {{
                if (cliArgs[normKey] === undefined) {{
                    var rawEnv = env[envKey];
                    try {{ args[normKey] = JSON.parse(rawEnv); }} catch (_) {{ args[normKey] = rawEnv; }}
                }}
            }}
        }}
    }}

    // Tier 3: CLI Arguments (Highest Precedence)
    for (var cliKey in cliArgs) {{
        args[cliKey] = cliArgs[cliKey];
    }}

    function badge(type, text) {{
        var colorMap = {{
            'AGENT': '\x1b[36m',
            'DAEMON': '\x1b[35m',
            'CYCLE': '\x1b[36m',
            'SUMMARY': '\x1b[32m',
            'AI': '\x1b[35m',
            'TOOL': '\x1b[33m',
            'POLICY': '\x1b[34m',
            'MEMORY': '\x1b[36m',
            'STEP': '\x1b[90m',
            'VERDICT': '\x1b[1m',
            'RESULT': '\x1b[32m',
            'STATUS': '\x1b[90m',
            'INFO': '\x1b[36m',
            'WARN': '\x1b[33m',
            'ERROR': '\x1b[31m'
        }};
        var col = colorMap[type] || '\x1b[1m';
        console.log(col + '[' + type + ']\x1b[0m ' + text);
    }}

    function bindPositionals(targetArgs) {{
        if (positionals.length > 0) {{
            var combined = positionals.join(' ');
            if (targetArgs['query'] === undefined) targetArgs['query'] = combined;
            if (targetArgs['prompt'] === undefined) targetArgs['prompt'] = combined;
            if (targetArgs['input'] === undefined) targetArgs['input'] = combined;

            var expected = routineSignatures[entryRoutine] || [];
            for (var p = 0; p < positionals.length; p++) {{
                var pKey = '' + p;
                var rawPos = positionals[p];
                var parsedVal;
                try {{ parsedVal = JSON.parse(rawPos); }} catch (_) {{ parsedVal = rawPos; }}
                if (targetArgs[pKey] === undefined) {{
                    targetArgs[pKey] = parsedVal;
                }}
                if (p < expected.length) {{
                    var paramName = expected[p];
                    if (targetArgs[paramName] === undefined) {{
                        targetArgs[paramName] = parsedVal;
                    }}
                }}
            }}
        }}
    }}

    function validateParameters(targetArgs) {{
        var expected = routineSignatures[entryRoutine];
        if (!expected || expected.length === 0) return true;

        var missing = [];
        for (var idx = 0; idx < expected.length; idx++) {{
            var pName = expected[idx];
            if (targetArgs[pName] === undefined && targetArgs['' + idx] === undefined) {{
                missing.push(pName);
            }}
        }}
        if (missing.length > 0) {{
            console.error("\x1b[31m[Agent Config Error]\x1b[0m Routine '" + entryRoutine + "' requires parameter(s):");
            for (var m = 0; m < missing.length; m++) {{
                console.error("  • \x1b[1m" + missing[m] + "\x1b[0m");
            }}
            console.error("\nProvide configuration via any of the following:");
            console.error("  1. Config file:     " + kazeName + " --config <file.json>");
            console.error("  2. Named CLI flags: " + kazeName + " --arg " + missing[0] + "=<value> ...");
            console.error("  3. Environment:     export AGENT_" + missing[0].toUpperCase() + "=<value>");
            if (expected.length <= 6) {{
                console.error("  4. Positional args: " + kazeName + " <" + expected.join("> <") + ">");
            }}
            console.error("");
            if (typeof process !== 'undefined' && process.exit) process.exit(1);
            return false;
        }}
        return true;
    }}

    function runDaemon() {{
        bindPositionals(args);
        if (!validateParameters(args)) return;

        var cycles = 0;
        var successes = 0;
        var failures = 0;
        var startTime = Date.now();
        var stopping = false;
        var timerId = null;

        badge('DAEMON', "Starting 24/7 supervisor loop for " + kazeName);
        badge('DAEMON', "Interval: " + intervalRaw + " (" + intervalMs + "ms) · Fuel: " + fuel + " · PID: " + (typeof process !== 'undefined' ? process.pid : 'unknown'));
        badge('DAEMON', "Autonomous recovery active · Press Ctrl+C to stop gracefully.");

        function formatDuration(ms) {{
            var s = Math.floor(ms / 1000);
            var m = Math.floor(s / 60);
            var h = Math.floor(m / 60);
            var d = Math.floor(h / 24);
            if (d > 0) return d + "d " + (h % 24) + "h " + (m % 60) + "m";
            if (h > 0) return h + "h " + (m % 60) + "m " + (s % 60) + "s";
            if (m > 0) return m + "m " + (s % 60) + "s";
            return s + "s";
        }}

        function shutdown() {{
            if (stopping) return;
            stopping = true;
            if (timerId !== null) {{
                clearInterval(timerId);
                timerId = null;
            }}
            var uptime = formatDuration(Date.now() - startTime);
            console.log("");
            badge('DAEMON', "Shutdown signal received. Stopping supervisor loop...");
            badge('SUMMARY', "Cycles: " + cycles + " (" + successes + " succeeded, " + failures + " failed) · Uptime: " + uptime);
            if (typeof process !== 'undefined' && process.exit) {{
                process.exit(0);
            }}
        }}

        if (typeof process !== 'undefined' && process.on) {{
            process.on('SIGINT', shutdown);
            process.on('SIGTERM', shutdown);
        }}

        function cycle() {{
            if (stopping) return;
            cycles++;
            var ts = (new Date()).toISOString().replace('T', ' ').slice(0, 19);
            badge('CYCLE', "#" + cycles + " · " + ts);
            try {{
                var runArgs = Object.assign({{}}, args);
                runArgs['cycle'] = cycles;
                runArgs['timestamp'] = ts;

                var opts = {{ fuel: fuel, grants: defaultGrants }};
                if (entryRoutine) opts.entry = entryRoutine;

                var res = sofuu.kaze.run(kazeSource, runArgs, opts);
                if (!res.ok) {{
                    failures++;
                    badge('ERROR', "Cycle #" + cycles + " failed: " + (res.error || res.code));
                }} else {{
                    successes++;
                    if (res.actions && res.actions.length > 0) {{
                        for (var j = 0; j < res.actions.length; j++) {{
                            var act = res.actions[j];
                            if (act.indexOf('ai.') !== -1) badge('AI', act);
                            else if (act.indexOf('tool.') !== -1) badge('TOOL', act);
                            else if (act.indexOf('policy.') !== -1) badge('POLICY', act);
                            else if (act.indexOf('memory.') !== -1) badge('MEMORY', act);
                            else badge('STEP', act);
                        }}
                    }}
                    if (res.logs && res.logs.length > 0) {{
                        for (var k = 0; k < res.logs.length; k++) {{
                            var l = res.logs[k];
                            badge(l.level.toUpperCase(), l.message);
                        }}
                    }}
                    if (res.verdict) {{
                        if (res.verdict.allow) {{
                            badge('VERDICT', "\x1b[32mALLOW\x1b[0m " + res.verdict.reason);
                        }} else {{
                            badge('VERDICT', "\x1b[31mREJECT\x1b[0m " + res.verdict.reason);
                        }}
                    }} else if (res.value && res.value !== 'nothing') {{
                        badge('RESULT', res.value);
                    }}
                    console.log("\x1b[90m" + (res.instructions || 0) + " instructions · " + (res.fuelUsed || 0) + " fuel\x1b[0m");
                }}
            }} catch (ex) {{
                failures++;
                badge('ERROR', "Cycle #" + cycles + " exception: " + (ex && ex.message ? ex.message : ex));
            }}
        }}

        cycle();
        timerId = setInterval(cycle, intervalMs);
    }}

    function runHeadless() {{
        bindPositionals(args);
        if (!validateParameters(args)) return;

        var opts = {{ fuel: fuel, grants: defaultGrants }};
        if (entryRoutine) opts.entry = entryRoutine;

        var res = sofuu.kaze.run(kazeSource, args, opts);
        if (!res.ok) {{
            console.error("\x1b[31m[Agent Error]\x1b[0m " + (res.error || res.code));
            if (typeof process !== 'undefined' && process.exit) process.exit(1);
            return;
        }}

        badge('AGENT', kazeName + " :: " + res.entry);
        if (res.actions && res.actions.length > 0) {{
            for (var j = 0; j < res.actions.length; j++) {{
                var act = res.actions[j];
                if (act.indexOf('ai.') !== -1) badge('AI', act);
                else if (act.indexOf('tool.') !== -1) badge('TOOL', act);
                else if (act.indexOf('policy.') !== -1) badge('POLICY', act);
                else if (act.indexOf('memory.') !== -1) badge('MEMORY', act);
                else badge('STEP', act);
            }}
        }}
        if (res.logs && res.logs.length > 0) {{
            for (var k = 0; k < res.logs.length; k++) {{
                var l = res.logs[k];
                badge(l.level.toUpperCase(), l.message);
            }}
        }}
        if (res.verdict) {{
            if (res.verdict.allow) {{
                badge('VERDICT', "\x1b[32mALLOW\x1b[0m " + res.verdict.reason);
            }} else {{
                badge('VERDICT', "\x1b[31mREJECT\x1b[0m " + res.verdict.reason);
            }}
        }} else if (res.value && res.value !== 'nothing') {{
            badge('RESULT', res.value);
        }}
        console.log("\x1b[90m" + (res.instructions || 0) + " instructions · " + (res.fuelUsed || 0) + " fuel\x1b[0m");
    }}

    function launchInteractiveTUI() {{
        badge('AGENT', "Online: " + kazeName);
        badge('STATUS', "Ready for multi-turn task input. Type a prompt or 'exit'.");

        if (typeof sofuu !== 'undefined' && sofuu.agent && sofuu.agent.interact) {{
            var agentProxy = {{
                name: kazeName,
                run: async function(prompt) {{
                    var runArgs = Object.assign({{}}, args);
                    runArgs['prompt'] = prompt;
                    runArgs['query'] = prompt;
                    runArgs['input'] = prompt;
                    var opts = {{ fuel: fuel, grants: defaultGrants }};
                    if (entryRoutine) opts.entry = entryRoutine;
                    var res = sofuu.kaze.run(kazeSource, runArgs, opts);
                    if (!res.ok) throw new Error(res.error || res.code);
                    if (res.actions) {{
                        for (var a = 0; a < res.actions.length; a++) {{
                            var act = res.actions[a];
                            if (act.indexOf('ai.') !== -1) badge('AI', act);
                            else if (act.indexOf('tool.') !== -1) badge('TOOL', act);
                            else if (act.indexOf('policy.') !== -1) badge('POLICY', act);
                        }}
                    }}
                    if (res.verdict) {{
                        return (res.verdict.allow ? "VERDICT ALLOW: " : "VERDICT REJECT: ") + res.verdict.reason;
                    }}
                    return res.value;
                }}
            }};
            sofuu.agent.interact(agentProxy, {{ title: kazeName }}).catch(function(err) {{
                console.error(err);
            }});
        }} else {{
            runHeadless();
        }}
    }}

    var isTTY = typeof process !== 'undefined' && process.stdout && process.stdout.isTTY;
    if (isDaemon) {{
        runDaemon();
    }} else if (isInteractive || (positionals.length === 0 && Object.keys(args).length === 0 && isTTY)) {{
        launchInteractiveTUI();
    }} else {{
        runHeadless();
    }}
}})();
"##);

    Ok(bootstrap)
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_base(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn trailer_roundtrips_at_a_fixed_width() {
        let t = Trailer {
            version: TRAILER_VERSION,
            flags: 7,
            payload_len: 1234,
            crc32: 0xDEADBEEF,
        };
        let bytes = t.to_bytes();
        assert_eq!(bytes.len(), 32);
        assert_eq!(Trailer::parse(&bytes), Some(t));
        /* The magic doubles as the "is this an artifact?" test. */
        assert_eq!(Trailer::parse(&[0u8; 32]), None);
    }

    #[test]
    fn assemble_puts_the_payload_between_base_and_trailer() {
        let base = fake_base(100);
        let payload = encode_payload("agent.ts", b"bytecode", false, None, 0);
        let img = assemble(&base, &payload);
        assert_eq!(&img[..100], &base[..]);
        assert_eq!(&img[100..100 + payload.len()], &payload[..]);
        assert_eq!(img.len(), 100 + payload.len() + 32);

        let got = embedded_payload(&img).unwrap().expect("a payload");
        assert_eq!(got.entry, "agent.ts");
        assert_eq!(got.bytecode, b"bytecode");
        assert!(got.brain.is_none());
    }

    #[test]
    fn embedded_payload_roundtrips_a_brain_and_its_dimension() {
        let base = fake_base(64);
        let payload = encode_payload("bot.js", b"bc", true, Some(b"BRAINBYTES"), 768);
        let img = assemble(&base, &payload);
        let got = embedded_payload(&img).unwrap().unwrap();
        assert!(got.gzipped);
        assert_eq!(got.brain.as_deref(), Some(&b"BRAINBYTES"[..]));
        assert_eq!(got.brain_dim, 768);
    }

    #[test]
    fn a_plain_binary_is_not_an_artifact_and_is_not_an_error() {
        assert!(embedded_payload(&fake_base(200)).unwrap().is_none());
        assert!(embedded_payload(&[]).unwrap().is_none());
    }

    #[test]
    fn a_corrupt_payload_is_refused_not_ignored() {
        let base = fake_base(64);
        let payload = encode_payload("x.js", b"bc", false, None, 0);
        let mut img = assemble(&base, &payload);
        /* Flip one payload byte: the trailer no longer matches. */
        let at = img.len() - TRAILER_LEN - 3;
        img[at] ^= 0xFF;
        let e = embedded_payload(&img).unwrap_err();
        assert!(e.contains("CRC32"), "{e}");
    }

    #[test]
    fn a_truncated_artifact_is_refused_not_ignored() {
        let base = fake_base(64);
        let payload = encode_payload("x.js", b"bc", false, None, 0);
        let mut img = assemble(&base, &payload);
        /* Move the trailer's payload length past the start of the file. */
        let n = img.len();
        let bogus = (n as u64 + 1).to_le_bytes();
        img[n - 16..n - 8].copy_from_slice(&bogus);
        let e = embedded_payload(&img).unwrap_err();
        assert!(e.contains("truncated"), "{e}");
    }

    #[test]
    fn a_future_trailer_version_is_refused_with_an_explanation() {
        let base = fake_base(64);
        let payload = encode_payload("x.js", b"bc", false, None, 0);
        let mut img = assemble(&base, &payload);
        let n = img.len();
        img[n - 20..n - 18].copy_from_slice(&99u16.to_le_bytes());
        let e = embedded_payload(&img).unwrap_err();
        assert!(e.contains("v99") && e.contains("rebuild"), "{e}");
    }

    #[test]
    fn a_payload_with_a_bogus_length_is_refused() {
        /* bc_len larger than the body must not panic or read out of bounds. */
        let mut p = encode_payload("x.js", b"bc", false, None, 0);
        p[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let e = decode_payload(&p, 0).unwrap_err();
        assert!(e.contains("truncated") || e.contains("corrupt"), "{e}");
    }

    #[test]
    fn recompiling_drops_an_earlier_payload_instead_of_nesting() {
        let base = fake_base(300);
        let p1 = encode_payload("one.js", b"first", false, None, 0);
        let once = assemble(&base, &p1);
        assert_eq!(base_len(&once), 300);

        let p2 = encode_payload("two.js", b"second", false, None, 0);
        let twice = assemble(&once[..base_len(&once)], &p2);
        assert_eq!(base_len(&twice), 300);
        assert_eq!(twice.len(), 300 + p2.len() + 32);
        let got = embedded_payload(&twice).unwrap().unwrap();
        assert_eq!(got.entry, "two.js");
        assert_eq!(got.bytecode, b"second");
    }

    #[test]
    fn a_signed_macho_trailer_is_found_before_the_signature_blob() {
        /* The shape codesign leaves behind: header, LC_CODE_SIGNATURE, base,
         * payload, trailer, signature. The trailer is NOT at EOF, so a naive
         * tail read would report "not an artifact". */
        let base = fake_base(64);
        let payload = encode_payload("mac.js", b"macbc", false, None, 0);
        let header_len = 32 + 16;
        /* codesign pads to 16 bytes: reproduce that, or the test would only
         * prove the easy (aligned) case. */
        let padding = 8usize;
        let sig_off = header_len + base.len() + payload.len() + TRAILER_LEN + padding;

        let mut img = vec![0u8; 32];
        img[0..4].copy_from_slice(&0xFEED_FACFu32.to_le_bytes());
        img[4..8].copy_from_slice(&0x0100_000Cu32.to_le_bytes()); // CPU_TYPE_ARM64
        img[16..20].copy_from_slice(&1u32.to_le_bytes()); // ncmds
        img[20..24].copy_from_slice(&16u32.to_le_bytes()); // sizeofcmds
        img.extend_from_slice(&0x1du32.to_le_bytes()); // LC_CODE_SIGNATURE
        img.extend_from_slice(&16u32.to_le_bytes());
        img.extend_from_slice(&(sig_off as u32).to_le_bytes());
        img.extend_from_slice(&64u32.to_le_bytes());
        img.extend_from_slice(&base);
        let mut img = assemble(&img, &payload);
        img.extend_from_slice(&[0x00; 8]); // alignment padding
        img.extend_from_slice(&[0xAA; 64]); // the signature blob

        assert_eq!(base_len(&img), header_len + base.len());
        let got = embedded_payload(&img).unwrap().unwrap();
        assert_eq!(got.entry, "mac.js");
        assert_eq!(got.bytecode, b"macbc");
    }

    #[test]
    fn a_brain_archive_roundtrips_the_manifest_and_its_segments() {
        let dir = std::env::temp_dir().join(format!("sofuu-brain-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("b.qtsq.d")).unwrap();
        std::fs::write(dir.join("b.qtsq"), b"MANIFEST").unwrap();
        std::fs::write(dir.join("b.qtsq.d/seg-000001-000001.qtsq"), b"SEGMENT").unwrap();
        std::fs::write(dir.join("b.qtsq.d/graph-000001.bin"), b"GRAPH").unwrap();

        let src = dir.join("b.qtsq").to_string_lossy().into_owned();
        let archive = pack_brain(&src).unwrap();
        let out = dir.join("out").join("b.qtsq");
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        unpack_brain(&archive, &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"MANIFEST");
        assert_eq!(
            std::fs::read(format!("{}.d/seg-000001-000001.qtsq", out.display())).unwrap(),
            b"SEGMENT"
        );
        assert_eq!(
            std::fs::read(format!("{}.d/graph-000001.bin", out.display())).unwrap(),
            b"GRAPH"
        );
        /* Re-seeding an existing store must be a no-op, not a clobber. */
        unpack_brain(&archive, &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"MANIFEST");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_archive_name_cannot_escape_the_target() {
        let mut archive = Vec::new();
        archive.extend_from_slice(BRAIN_ARCHIVE_MAGIC);
        archive.extend_from_slice(&BRAIN_ARCHIVE_VERSION.to_le_bytes());
        archive.extend_from_slice(&1u16.to_le_bytes());
        let name = b"d/../evil";
        archive.extend_from_slice(&(name.len() as u16).to_le_bytes());
        archive.extend_from_slice(&1u32.to_le_bytes());
        archive.extend_from_slice(name);
        archive.push(b'x');
        let out = std::env::temp_dir().join("sofuu-evil/b.qtsq");
        let e = unpack_brain(&archive, &out).unwrap_err();
        assert!(e.contains("unsafe"), "{e}");
    }

    #[test]
    fn a_trailer_with_codesign_padding_after_it_is_still_the_trailer() {
        /* The state `codesign --remove-signature` leaves behind: base,
         * payload, trailer, a few bytes of alignment padding, no signature.
         * Missing this made every recompile bake the old payload into the new
         * base (~1 KB growth per compile). */
        let base = fake_base(64);
        let payload = encode_payload("p.js", b"pbc", false, None, 0);
        let mut img = assemble(&base, &payload);
        img.extend_from_slice(&[0u8; 6]);
        assert_eq!(base_len(&img), base.len());
        assert_eq!(embedded_payload(&img).unwrap().unwrap().entry, "p.js");
    }

    #[test]
    fn output_naming_defaults_to_the_entry_stem() {
        let out = resolve_output(Path::new("dir/agent.ts"), None).unwrap();
        assert_eq!(out.file_name().unwrap(), "agent");
        /* `-o` is taken verbatim (relative paths resolve against cwd). */
        let out = resolve_output(Path::new("dir/agent.ts"), Some("dist/bot")).unwrap();
        assert_eq!(out, PathBuf::from("dist/bot"));
        assert!(resolve_output(Path::new("a"), Some("")).is_err());
    }

    #[test]
    fn gzip_is_only_used_when_it_wins() {
        let small = b"tiny".to_vec();
        let (stored, gz) = maybe_gzip(&small);
        assert_eq!(stored, small);
        assert!(!gz);

        let big: Vec<u8> = (0..100_000).map(|i| b"abcabcab"[i % 8]).collect();
        let (stored, gz) = maybe_gzip(&big);
        assert!(gz, "a highly repetitive 100 KB program must compress");
        assert!(stored.len() < big.len() / 2);
    }

    #[test]
    fn compile_source_reports_the_parse_error() {
        let e = unsafe { compile_source("function (", "bad.js") }.unwrap_err();
        assert!(!e.is_empty());
    }
}
