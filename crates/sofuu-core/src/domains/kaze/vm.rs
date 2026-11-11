// sfs/vm.rs — the bytecode interpreter (M-K3).
//
// PLAN-KAZE-LANGUAGE §1.4/§1.5/§4. Two guarantees are enforced here and
// nowhere else:
//
//   * Fuel. Every instruction costs one unit. At zero the run stops with
//     E_TIMEOUT (fuel) — not a panic, not a hang, not a thread to kill. `FUEL_TICK` at
//     each loop head also polls the host deadline (rt/deadline.rs), so a run
//     started from JS obeys the same interrupt as any other host call.
//   * One side-effect path. CALL_CAP is the only instruction that can reach
//     the outside world, and it always asks rt/hardware.rs first. A Kaze
//     program cannot open a socket because there is no instruction for it.
//
// Arithmetic traps instead of wrapping: a non-finite result is E_KAZE_OVERFLOW,
// a division by zero is E_KAZE_DIV0, and a bad index is E_KAZE_BOUNDS. In a
// controller, a silently-wrong number is worse than a stopped run.

use crate::sfs::arena::{fmt_num, Arena};
use crate::sfs::caps::{self, CapHost};
use crate::sfs::isa::{field_a, opcode, Const, ItemCode, Module, Op};
use crate::sfs::value::{Heap, Value};

/// How much work one run may do before it is stopped.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub fuel: u64,
    pub arena_bytes: usize,
    /// Budget for arrays and maps. Two meters, not one: fuel bounds work,
    /// this bounds retained bytes.
    pub heap_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        // 200k instructions is a few milliseconds of work and far more than any
        // hardware routine needs; the point is that the bound exists.
        Limits { fuel: 200_000, arena_bytes: 256 * 1024, heap_bytes: Heap::DEFAULT_LIMIT }
    }
}

#[derive(Debug, Clone)]
pub struct VmError {
    pub code: &'static str,
    pub msg: String,
    pub line: u32,
}

impl VmError {
    fn new(code: &'static str, msg: impl Into<String>) -> Self {
        VmError { code, msg: msg.into(), line: 0 }
    }
}

/// What a `filter` decided.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Allow(String),
    Reject(String),
}

pub struct Vm<'a> {
    pub module: &'a Module,
    pub arena: Arena,
    pub heap: Heap,
    pub host: &'a mut dyn CapHost,
    declared: &'a [String],
    grants: &'a [String],
    policy: Option<&'a serde_json::Value>,
    runtime_policy: Option<&'a serde_json::Value>,
    fuel_left: u64,
    pub fuel_used: u64,
    pub instructions: u64,
    stack: Vec<Value>,
    locals: Vec<Value>,
    verdict: Option<Verdict>,
}

/// The stack is a fixed-size vector; a script cannot grow it without bound.
const MAX_STACK: usize = 256;

impl<'a> Vm<'a> {
    pub fn new(
        module: &'a Module,
        host: &'a mut dyn CapHost,
        declared: &'a [String],
        grants: &'a [String],
        policy: Option<&'a serde_json::Value>,
        limits: Limits,
    ) -> Self {
        Vm {
            module,
            arena: Arena::new(limits.arena_bytes),
            heap: Heap::new(limits.heap_bytes),
            host,
            declared,
            grants,
            policy,
            runtime_policy: None,
            fuel_left: limits.fuel,
            fuel_used: 0,
            instructions: 0,
            stack: Vec::with_capacity(32),
            locals: Vec::new(),
            verdict: None,
        }
    }

    /// Add the embedding runtime's restrictions without replacing the
    /// Kaze-specific policy required for every capability call.
    pub fn set_runtime_policy(&mut self, policy: Option<&'a serde_json::Value>) {
        self.runtime_policy = policy;
    }

    /// Run one item to completion. `args` bind to `item.params` by position.
    pub fn run_item(&mut self, item: &ItemCode, args: &[Value]) -> Result<Value, VmError> {
        self.stack.clear();
        self.locals = vec![Value::Unit; item.slots.max(item.params.len() as u8) as usize];
        for (i, a) in args.iter().enumerate().take(self.locals.len()) {
            self.locals[i] = *a;
        }
        self.verdict = None;

        let code = &item.code;
        let mut pc = 0usize;
        while pc < code.len() {
            let w = code[pc];
            let current_line = item.line_map.get(pc).copied()
                .filter(|line| *line > 0).unwrap_or(item.line);
            let Some(op) = Op::from_u8(opcode(w)) else {
                return Err(VmError::new("E_KAZE_BADCODE", "this module has an unknown instruction"));
            };
            pc += 1;
            // A loop head carries its own source line in the immediate word, so
            // an exhausted budget blames the loop rather than the routine.
            let step_line = if matches!(op, Op::FuelTick) {
                code.get(pc).copied().unwrap_or(current_line)
            } else {
                current_line
            };
            self.charge(step_line)?;
            match op {
                Op::Const => {
                    let imm = self.imm(code, pc, current_line)?;
                    pc += 1;
                    let v = self.constant(imm as usize, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::LoadLocal => {
                    let slot = field_a(w) as usize;
                    let v = *self
                        .locals
                        .get(slot)
                        .ok_or_else(|| VmError::new("E_KAZE_BADCODE", "LOAD from a missing slot"))?;
                    self.push(v, current_line)?;
                }
                Op::StoreLocal => {
                    let slot = field_a(w) as usize;
                    let v = self.pop(current_line)?;
                    if let Some(dst) = self.locals.get_mut(slot) {
                        *dst = v;
                    } else {
                        return Err(VmError::new("E_KAZE_BADCODE", "STORE to a missing slot"));
                    }
                }
                Op::NewArr => {
                    let n = field_a(w) as usize;
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.pop(current_line)?);
                    }
                    items.reverse();
                    let v = self
                        .heap
                        .push_array(items)
                        .map_err(|msg| VmError { code: "E_KAZE_MEMORY", msg, line: current_line })?;
                    self.push(v, current_line)?;
                }
                Op::GetIndex => {
                    let idx = self.pop(current_line)?;
                    let coll = self.pop(current_line)?;
                    let v = self.index(coll, idx, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Len => {
                    let v = self.pop(current_line)?;
                    let n = self.length(v, current_line)?;
                    self.push(Value::Num(n), current_line)?;
                }
                Op::GetField => {
                    let field = self.pop(current_line)?;
                    let base = self.pop(current_line)?;
                    let v = self.field(base, field, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Jeq | Op::Jne => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc += 1;
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let eq = crate::sfs::arena::values_equal(a, b, &self.arena, &self.heap);
                    let taken = if op == Op::Jeq { eq } else { !eq };
                    if taken {
                        pc = jump(pc, imm, code.len(), current_line)?;
                    }
                }
                Op::Pop => {
                    self.pop(current_line)?;
                }
                Op::Add => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let v = self.add(a, b, current_line)?;
                    self.push(v, current_line)?;
                }
                Op::Sub => self.arith(Op::Sub, current_line)?,
                Op::Mul => self.arith(Op::Mul, current_line)?,
                Op::Div => self.arith(Op::Div, current_line)?,
                Op::Mod => self.arith(Op::Mod, current_line)?,
                Op::Neg => {
                    let v = self.pop(current_line)?;
                    match v {
                        Value::Num(n) => self.push(Value::Num(-n), current_line)?,
                        other => {
                            return Err(VmError::new(
                                "E_KAZE_TYPE",
                                format!("cannot negate {}", other.type_name()),
                            ))
                        }
                    }
                }
                Op::CmpLt | Op::CmpLe | Op::CmpGt | Op::CmpGe => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let v = self.compare(op, a, b, current_line)?;
                    self.push(Value::Bool(v), current_line)?;
                }
                Op::CmpEq | Op::CmpNe => {
                    let b = self.pop(current_line)?;
                    let a = self.pop(current_line)?;
                    let eq = crate::sfs::arena::values_equal(a, b, &self.arena, &self.heap);
                    self.push(Value::Bool(if op == Op::CmpEq { eq } else { !eq }), current_line)?;
                }
                Op::Not => {
                    let v = self.pop(current_line)?;
                    self.push(Value::Bool(!v.is_truthy()), current_line)?;
                }
                Op::Jmp => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc = jump(pc + 1, imm, code.len(), current_line)?;
                }
                Op::JmpIfFalse | Op::JmpIfTrue => {
                    let imm = self.imm(code, pc, current_line)? as i32;
                    pc += 1;
                    let v = self.pop(current_line)?;
                    let taken = if op == Op::JmpIfTrue { v.is_truthy() } else { !v.is_truthy() };
                    if taken {
                        pc = jump(pc, imm, code.len(), current_line)?;
                    }
                }
                Op::CallCap => {
                    let imm = self.imm(code, pc, current_line)? as usize;
                    pc += 1;
                    self.call_cap(imm, field_a(w) as usize, current_line)?;
                }
                Op::PolicyStop => {
                    let v = self.pop(current_line)?;
                    return Ok(v);
                }
                Op::Ret => {
                    let v = self.pop(current_line)?;
                    return Ok(v);
                }
                Op::RetUnit => return Ok(Value::Unit),
                Op::FuelTick => {
                    // The pre-charge above already paid for this tick; the
                    // immediate only names the loop for error reporting.
                    let line = self.imm(code, pc, current_line)?;
                    pc += 1;
                    // A JS host may have armed a deadline for this whole call.
                    if crate::rt::deadline::poll_expired() != 0 {
                        // rt/deadline.rs already recorded the firing; ask it for
                        // the report so a host sees one story, not two.
                        let detail = crate::rt::deadline::classify_interrupted("interrupted")
                            .unwrap_or_else(|| {
                                "E_TIMEOUT: the call was interrupted after its deadline elapsed"
                                    .to_string()
                            });
                        let msg = detail
                            .strip_prefix("E_TIMEOUT: ")
                            .unwrap_or(detail.as_str())
                            .to_string();
                        return Err(VmError { code: "E_TIMEOUT", msg, line });
                    }
                }
                Op::Trap => {
                    let imm = self.imm(code, pc, current_line)? as usize;
                    let msg = match self.module.consts.get(imm) {
                        Some(Const::Str(s)) => s.clone(),
                        _ => "the script stopped".to_string(),
                    };
                    return Err(VmError { code: "E_KAZE_TRAP", msg, line: current_line });
                }
                Op::Halt => return Ok(Value::Unit),
            }
        }
        Ok(Value::Unit)
    }

    /// Take the verdict the last `run_item` produced, if it was a `filter`.
    pub fn take_verdict(&mut self) -> Option<Verdict> {
        self.verdict.take()
    }

    pub fn fuel_left(&self) -> u64 {
        self.fuel_left
    }

    // ── helpers ──────────────────────────────────────────────────────
    fn charge(&mut self, line: u32) -> Result<(), VmError> {
        self.instructions += 1;
        self.fuel_used += 1;
        if self.fuel_left == 0 {
            // PLAN-KAZE §1.5/§4: a runaway script halts uncatchably and comes
            // back as a typed E_TIMEOUT. The message names the cause, because
            // "out of fuel" and "the host's deadline fired" need different
            // fixes even though both are timeouts to the caller.
            return Err(VmError {
                code: "E_TIMEOUT",
                msg: "this run used all of its fuel and was stopped (fuel, not wall time); \
                      a loop that never ends or too much work in one tick is the usual cause"
                    .to_string(),
                line,
            });
        }
        self.fuel_left -= 1;
        Ok(())
    }

    fn imm(&self, code: &[u32], pc: usize, line: u32) -> Result<u32, VmError> {
        code.get(pc)
            .copied()
            .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "truncated instruction".into(), line })
    }

    fn push(&mut self, v: Value, line: u32) -> Result<(), VmError> {
        if self.stack.len() >= MAX_STACK {
            return Err(VmError {
                code: "E_KAZE_STACK",
                msg: "this expression nests too deeply; split it into smaller steps".to_string(),
                line,
            });
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self, line: u32) -> Result<Value, VmError> {
        self.stack.pop().ok_or_else(|| VmError {
            code: "E_KAZE_BADCODE",
            msg: "the value stack underflowed".to_string(),
            line,
        })
    }

    fn constant(&mut self, idx: usize, line: u32) -> Result<Value, VmError> {
        match self.module.consts.get(idx) {
            Some(Const::Num(n)) => Ok(Value::Num(*n)),
            Some(Const::Bool(b)) => Ok(Value::Bool(*b)),
            Some(Const::Str(s)) => {
                let text = s.clone();
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            None => Err(VmError { code: "E_KAZE_BADCODE", msg: "missing constant".into(), line }),
        }
    }

    fn add(&mut self, a: Value, b: Value, line: u32) -> Result<Value, VmError> {
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => {
                let sum = x + y;
                if !sum.is_finite() {
                    return Err(overflow(line));
                }
                Ok(Value::Num(sum))
            }
            (Value::Str(x), Value::Str(y)) => {
                let text = format!("{}{}", self.arena.get(x), self.arena.get(y));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (Value::Str(x), other) => {
                let text = format!("{}{}", self.arena.get(x), caps::render_value(other, &self.arena, &self.heap));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (other, Value::Str(y)) => {
                let text = format!("{}{}", caps::render_value(other, &self.arena, &self.heap), self.arena.get(y));
                let r = self.arena.alloc(&text).map_err(|e| VmError {
                    code: "E_KAZE_MEMORY",
                    msg: e,
                    line,
                })?;
                Ok(Value::Str(r))
            }
            (a, b) => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("cannot add {} to {}", a.type_name(), b.type_name()),
                line,
            }),
        }
    }

    fn arith(&mut self, op: Op, line: u32) -> Result<(), VmError> {
        let b = self.pop(line)?;
        let a = self.pop(line)?;
        let (x, y) = match (a, b) {
            (Value::Num(x), Value::Num(y)) => (x, y),
            (a, b) => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("cannot do arithmetic on {} and {}", a.type_name(), b.type_name()),
                    line,
                })
            }
        };
        let v = match op {
            Op::Sub => x - y,
            Op::Mul => x * y,
            Op::Div => {
                if y == 0.0 {
                    return Err(divide_by_zero(line));
                }
                x / y
            }
            Op::Mod => {
                if y == 0.0 {
                    return Err(divide_by_zero(line));
                }
                x % y
            }
            _ => unreachable!(),
        };
        if !v.is_finite() {
            return Err(overflow(line));
        }
        self.push(Value::Num(v), line)
    }

    fn compare(&self, op: Op, a: Value, b: Value, line: u32) -> Result<bool, VmError> {
        let ord = match (a, b) {
            (Value::Num(x), Value::Num(y)) => x.partial_cmp(&y),
            (Value::Str(x), Value::Str(y)) => self.arena.get(x).partial_cmp(self.arena.get(y)),
            (a, b) => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("cannot compare {} with {}", a.type_name(), b.type_name()),
                    line,
                })
            }
        };
        let ord = ord.ok_or_else(|| VmError {
            code: "E_KAZE_TYPE",
            msg: "this comparison has no answer (a value is not a number)".to_string(),
            line,
        })?;
        Ok(match op {
            Op::CmpLt => ord.is_lt(),
            Op::CmpLe => ord.is_le(),
            Op::CmpGt => ord.is_gt(),
            Op::CmpGe => ord.is_ge(),
            _ => unreachable!(),
        })
    }

    /// `record.field`. A missing field reads as `nothing` — an optional
    /// record field is normal data, not an error — but a field read on a
    /// non-record is a type error, because that is a bug in the script.
    fn field(&mut self, base: Value, field: Value, line: u32) -> Result<Value, VmError> {
        let name = match field {
            Value::Str(s) => self.arena.get(s).to_string(),
            other => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("a field name must be text, not {}", other.type_name()),
                    line,
                })
            }
        };
        match base {
            Value::Map(m) => {
                let key = self
                    .arena
                    .alloc(&name)
                    .map_err(|e| VmError { code: "E_KAZE_MEMORY", msg: e, line })?;
                Ok(self.heap.map_get(m, Value::Str(key), &self.arena).unwrap_or(Value::Unit))
            }
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("'.{}' reads a field, but the value is {}", name, other.type_name()),
                line,
            }),
        }
    }

    fn length(&self, v: Value, line: u32) -> Result<f64, VmError> {
        match v {
            Value::Str(s) => Ok(self.arena.get(s).chars().count() as f64),
            Value::Arr(i) => Ok(self.heap.array(i).map(|a| a.len()).unwrap_or(0) as f64),
            Value::Map(i) => Ok(self.heap.map(i).map(|m| m.len()).unwrap_or(0) as f64),
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("'length of' does not apply to {}", other.type_name()),
                line,
            }),
        }
    }

    fn index(&mut self, coll: Value, idx: Value, line: u32) -> Result<Value, VmError> {
        let i = match idx {
            Value::Num(n) => n,
            other => {
                return Err(VmError {
                    code: "E_KAZE_TYPE",
                    msg: format!("a position must be a number, not {}", other.type_name()),
                    line,
                })
            }
        };
        if i < 0.0 || i.fract() != 0.0 {
            return Err(VmError {
                code: "E_KAZE_BOUNDS",
                msg: format!("{} is not a valid position", fmt_num(i)),
                line,
            });
        }
        let i = i as usize;
        match coll {
            Value::Arr(a) => self
                .heap
                .array(a)
                .and_then(|items| items.get(i).copied())
                .ok_or_else(|| VmError {
                    code: "E_KAZE_BOUNDS",
                    msg: format!("position {} is past the end of the list", i),
                    line,
                }),
            Value::Str(s) => {
                let text: Vec<char> = self.arena.get(s).chars().collect();
                let Some(ch) = text.get(i).copied() else {
                    return Err(VmError {
                        code: "E_KAZE_BOUNDS",
                        msg: format!("position {} is past the end of the text", i),
                        line,
                    });
                };
                let one = ch.to_string();
                let r = self
                    .arena
                    .alloc(&one)
                    .map_err(|e| VmError { code: "E_KAZE_MEMORY", msg: e, line })?;
                Ok(Value::Str(r))
            }
            other => Err(VmError {
                code: "E_KAZE_TYPE",
                msg: format!("[ ] does not apply to {}", other.type_name()),
                line,
            }),
        }
    }

    fn call_cap(&mut self, cap_idx: usize, argc: usize, line: u32) -> Result<(), VmError> {
        let cap = self
            .module
            .caps
            .get(cap_idx)
            .cloned()
            .ok_or_else(|| VmError { code: "E_KAZE_BADCODE", msg: "missing capability".into(), line })?;
        let mut args = vec![Value::Unit; argc];
        for i in (0..argc).rev() {
            args[i] = self.pop(line)?;
        }

        // The R7 gate runs for EVERY call, including the in-process families.
        crate::rt::hardware::authorise(
            self.declared,
            self.grants,
            self.policy,
            &cap.family,
            &cap.op,
        )
        .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;
        if let Some(policy) = self.runtime_policy {
            crate::rt::hardware::authorise_runtime_policy(policy, &cap.family, &cap.op)
                .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;
        }
        crate::rt::hardware::validate_arguments(&cap.family, &cap.op, &args, &self.arena)
            .map_err(|d| VmError { code: d.code, msg: format!("{} ({})", d.reason, d.rule), line })?;

        // Verdicts are control flow, not effects: the VM owns them, so a host
        // backend cannot fake an approval.
        if cap.family == "policy" && cap.op == "allow" {
            let v = args.first().copied().unwrap_or(Value::Unit);
            self.verdict = Some(Verdict::Allow(render(&v, &self.arena, &self.heap)));
            self.push(v, line)?;
            return Ok(());
        }
        if cap.family == "policy" && cap.op == "reject" {
            let reason = match args.first() {
                Some(Value::Str(s)) => self.arena.get(*s).to_string(),
                other => other.map(|v| render(v, &self.arena, &self.heap)).unwrap_or_default(),
            };
            self.verdict = Some(Verdict::Reject(reason));
            self.push(Value::Unit, line)?;
            return Ok(());
        }

        let result = self
            .host
            .call(&cap.family, &cap.op, &args, &mut self.arena, &mut self.heap)
            .map_err(|msg| {
                // A live backend may name its own code (`E_KAZE_LINK_TIMEOUT: …`),
                // so the report says what actually went wrong instead of a
                // generic host failure. Only codes the engine knows survive;
                // anything else stays `E_KAZE_HOST`.
                let (code, msg) = split_host_code(msg);
                VmError { code, msg, line }
            })?;
        self.push(result, line)
    }

}

/// Codes a capability host may raise. The list is closed on purpose: a host
/// cannot invent an error the CLI has never been taught to explain.
const HOST_CODES: &[&str] = &[
    "E_CAP_MISSING",
    "E_CAP_DENIED",
    "E_KAZE_MEMORY",
    "E_KAZE_RANGE",
    "E_KAZE_LINK",
    "E_KAZE_LINK_TIMEOUT",
    "E_COMMAND_REJECTED",
    "E_ARMING_DENIED",
    "E_SAFETY_VIOLATION",
];

fn split_host_code(msg: String) -> (&'static str, String) {
    if let Some((head, rest)) = msg.split_once(':') {
        for code in HOST_CODES {
            if head == *code {
                return (code, rest.trim_start().to_string());
            }
        }
    }
    ("E_KAZE_HOST", msg)
}

fn jump(pc: usize, imm: i32, len: usize, line: u32) -> Result<usize, VmError> {
    let target = pc as i64 + imm as i64;
    if target < 0 || target as usize > len {
        return Err(VmError { code: "E_KAZE_BADCODE", msg: "a jump left the code".into(), line });
    }
    Ok(target as usize)
}

fn overflow(line: u32) -> VmError {
    VmError {
        code: "E_KAZE_OVERFLOW",
        msg: "this calculation overflowed; the run stopped instead of continuing with a wrong \
              number"
            .to_string(),
        line,
    }
}

fn divide_by_zero(line: u32) -> VmError {
    VmError { code: "E_KAZE_DIV0", msg: "dividing by zero".to_string(), line }
}

fn render(v: &Value, arena: &Arena, heap: &Heap) -> String {
    crate::sfs::caps::render_value(*v, arena, heap)
}
