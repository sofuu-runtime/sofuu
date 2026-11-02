// sfs/arena.rs — the linear bump arena.
//
// PLAN-KAZE-LANGUAGE §4/§1.4: "Memory is allocated sequentially in a linear
// arena per run and deallocated wholesale in a single operation
// (free_wholesale). There is no garbage collector, zero stop-the-world pauses,
// and zero heap fragmentation."
//
// Implementation: one `Vec<u8>` per run. String bytes are appended, never
// moved, and only the offset/length pair survives. Dropping the arena (or
// calling `free_wholesale`) hands the entire run's memory back at once, which
// is the property a 10 Hz vehicle controller needs: bounded, O(1), and
// impossible to leak across runs.

use crate::sfs::value::Value;

/// A slice of the arena. `off`/`len` are byte offsets, valid only while the
/// arena that produced them is alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrRef {
    pub off: u32,
    pub len: u32,
}

impl StrRef {
    pub const EMPTY: StrRef = StrRef { off: 0, len: 0 };
}

pub struct Arena {
    bytes: Vec<u8>,
    /// Bytes ever written; the fuel-equivalent of memory pressure.
    high_water: usize,
    limit: usize,
}

impl Arena {
    /// `limit` bounds one run's string bytes. Exceeding it is a typed error,
    /// never a silent truncation (reject-never-clamp).
    pub fn new(limit: usize) -> Self {
        Arena { bytes: Vec::new(), high_water: 0, limit }
    }

    pub fn alloc(&mut self, s: &str) -> Result<StrRef, String> {
        let len = s.len();
        if len > u32::MAX as usize {
            return Err("text is too long for one value".to_string());
        }
        let end = self.bytes.len().saturating_add(len);
        if end > self.limit {
            return Err(format!(
                "E_KAZE_MEMORY: the run's arena limit of {} bytes was exceeded",
                self.limit
            ));
        }
        let off = self.bytes.len();
        self.bytes.extend_from_slice(s.as_bytes());
        if end > self.high_water {
            self.high_water = end;
        }
        Ok(StrRef { off: off as u32, len: len as u32 })
    }

    pub fn get(&self, r: StrRef) -> &str {
        let start = r.off as usize;
        let end = start.saturating_add(r.len as usize);
        match self.bytes.get(start..end) {
            Some(b) => std::str::from_utf8(b).unwrap_or(""),
            None => "",
        }
    }

    pub fn used(&self) -> usize {
        self.high_water
    }

    /// The one deallocation: the whole run goes at once.
    pub fn free_wholesale(&mut self) {
        self.bytes.clear();
        self.high_water = 0;
    }
}

/// Structural equality across the two value kinds that can hold strings.
pub fn values_equal(a: Value, b: Value, arena: &Arena) -> bool {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => arena.get(x) == arena.get(y),
        (Value::Arr(x), Value::Arr(y)) => x == y,
        (Value::Map(x), Value::Map(y)) => x == y,
        _ => a == b,
    }
}

/// Render a number the way the DSL prints it: integers without a trailing
/// `.0`, everything else in Rust's shortest round-trip form.
pub fn fmt_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}
