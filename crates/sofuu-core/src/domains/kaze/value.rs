// sfs/value.rs — Kaze runtime values.
//
// PLAN-KAZE-LANGUAGE §4: the VM's tagged union. Strings live in the run's
// linear arena (`arena.rs`); arrays and maps live in the heap side tables
// (`Heap` below), which are dropped wholesale when the run ends. Nothing here
// owns a pointer into freed memory, so there is no GC and no unsafe code.

use crate::sfs::arena::StrRef;

/// A Kaze value. 16 bytes; `Copy` so the VM stack is a plain `Vec<Value>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// No value: an action that returns nothing, a `let` with no initializer.
    Unit,
    Num(f64),
    Bool(bool),
    Str(StrRef),
    /// Index into `Heap::arrays`.
    Arr(u32),
    /// Index into `Heap::maps`.
    Map(u32),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Unit => "nothing",
            Value::Num(_) => "number",
            Value::Bool(_) => "true/false",
            Value::Str(_) => "text",
            Value::Arr(_) => "list",
            Value::Map(_) => "record",
        }
    }

    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Unit => false,
            Value::Bool(b) => *b,
            Value::Num(n) => *n != 0.0,
            Value::Str(s) => s.len > 0,
            Value::Arr(_) | Value::Map(_) => true,
        }
    }
}

/// The collection heap. `arrays` and `maps` are cleared by `free_wholesale`;
/// an index is never reused within a run, so a stale index cannot alias.
///
/// The heap is **metered** (PLAN-AUDIT-REMEDIATION §3 Task 1.4). Fuel bounds
/// instructions, not bytes: without a second meter, a loop that allocates a
/// list every iteration is unbounded work at bounded fuel (measured: 300k
/// iterations reached 114 MB against a documented 256 KB arena). Every push
/// charges the bytes it is about to retain, and a push that would cross the
/// budget is refused — rejected, never clamped, exactly as H-4 refuses an
/// out-of-range altitude.
pub struct Heap {
    pub arrays: Vec<Vec<Value>>,
    pub maps: Vec<Vec<(Value, Value)>>,
    used_bytes: usize,
    limit: usize,
}

impl Default for Heap {
    /// A bare `Heap` gets the default budget, never an unlimited one.
    fn default() -> Self {
        Heap::new(Heap::DEFAULT_LIMIT)
    }
}

impl Heap {
    /// 4 MiB — about 16x the string arena. Generous for any hardware routine,
    /// small enough that several concurrent scripts still fit a small device.
    pub const DEFAULT_LIMIT: usize = 4 * 1024 * 1024;

    pub fn new(limit: usize) -> Self {
        Heap { arrays: Vec::new(), maps: Vec::new(), used_bytes: 0, limit }
    }

    /// A read-only heap for rendering a value that owns no collections.
    /// Its budget is zero, so it can never be pushed into by accident.
    pub fn empty() -> Self {
        Heap::new(0)
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Charge the bytes a collection will retain before it is retained.
    fn charge(&mut self, bytes: usize) -> Result<(), String> {
        let next = self.used_bytes.saturating_add(bytes);
        if next > self.limit {
            return Err(format!(
                "the run's collection budget of {} bytes was exceeded (arrays and maps are                  rejected, never clamped)",
                self.limit
            ));
        }
        self.used_bytes = next;
        Ok(())
    }

    pub fn push_array(&mut self, items: Vec<Value>) -> Result<Value, String> {
        let cost = items
            .len()
            .saturating_mul(std::mem::size_of::<Value>())
            .saturating_add(std::mem::size_of::<Vec<Value>>());
        self.charge(cost)?;
        let idx = self.arrays.len() as u32;
        self.arrays.push(items);
        Ok(Value::Arr(idx))
    }

    pub fn push_map(&mut self, entries: Vec<(Value, Value)>) -> Result<Value, String> {
        let cost = entries
            .len()
            .saturating_mul(2 * std::mem::size_of::<Value>())
            .saturating_add(std::mem::size_of::<Vec<Value>>());
        self.charge(cost)?;
        let idx = self.maps.len() as u32;
        self.maps.push(entries);
        Ok(Value::Map(idx))
    }

    pub fn array(&self, idx: u32) -> Option<&Vec<Value>> {
        self.arrays.get(idx as usize)
    }

    pub fn map(&self, idx: u32) -> Option<&Vec<(Value, Value)>> {
        self.maps.get(idx as usize)
    }

    /// Map lookup by key (numbers and strings only; first match wins).
    pub fn map_get(&self, idx: u32, key: Value, arena: &crate::sfs::arena::Arena) -> Option<Value> {
        let entries = self.maps.get(idx as usize)?;
        for (k, v) in entries {
            if crate::sfs::arena::values_equal(*k, key, arena, self) {
                return Some(*v);
            }
        }
        None
    }

    /// One operation, zero fragmentation — the whole run's collections go.
    /// The meter is part of that: a wholesale free that left the budget spent
    /// would refuse every allocation made afterwards.
    pub fn free_wholesale(&mut self) {
        self.arrays.clear();
        self.maps.clear();
        self.used_bytes = 0;
    }

    pub fn used(&self) -> usize {
        self.arrays.len() + self.maps.len()
    }
}
