// sfs/compiler.rs — AST → Kaze bytecode (M-K3).
//
// PLAN-KAZE-LANGUAGE §4/§5. One pass, one stack, no optimisation: the goal is
// an artefact that is small, auditable and byte-identical for the same source.
// Two properties are load-bearing:
//
//   1. Determinism. Constants and capabilities are deduplicated by linear
//      scan, never by hashing, so the same script compiles to the same bytes
//      on every machine and every run (§2 "bit-exact execution trace").
//   2. Explicit side effects. The only instruction that can leave the VM is
//      CALL_CAP, and every CALL_CAP names an entry in the module's capability
//      table, which the R7 gate re-checks at run time.

use crate::sfs::ast::*;
use crate::sfs::caps;
use crate::sfs::isa::{word, CapRef, Const, ItemCode, ItemTag, Module, Op};

#[derive(Debug, Clone)]
pub struct CompileError {
    pub line: u32,
    pub msg: String,
}

impl CompileError {
    pub fn render(&self, source_name: &str) -> String {
        format!("{}:{}: {}", source_name, self.line, self.msg)
    }
}

fn e<T>(line: u32, msg: impl Into<String>) -> Result<T, CompileError> {
    Err(CompileError { line, msg: msg.into() })
}

const MAX_SLOTS: u8 = 255;

pub fn compile(program: &Program) -> Result<Module, CompileError> {
    let mut module = Module {
        needs: program.needs.clone(),
        ..Default::default()
    };
    for item in &program.items {
        let mut cx = Cx {
            consts: &mut module.consts,
            caps: &mut module.caps,
            code: Vec::new(),
            line_map: Vec::new(),
            current_line: item.line,
            scopes: vec![Vec::new()],
            next_slot: 0,
            item_name: item.name.clone(),
        };
        for p in &item.params {
            cx.bind(p)?;
        }
        match &item.kind {
            ItemKind::When(cond) => {
                // `when` compiles the predicate into the item itself: the host
                // polls it, and the body runs only when it is true.
                cx.expr(cond)?;
                let skip = cx.emit_wide(Op::JmpIfFalse, 0, 0);
                cx.stmts(&item.body)?;
                cx.emit(Op::RetUnit, 0, 0, 0);
                cx.patch(skip);
                cx.emit(Op::RetUnit, 0, 0, 0);
            }
            _ => {
                cx.stmts(&item.body)?;
                cx.emit(Op::RetUnit, 0, 0, 0);
            }
        }
        let (tag, period) = match &item.kind {
            ItemKind::Action => (ItemTag::Action, 0.0),
            ItemKind::Check => (ItemTag::Check, 0.0),
            ItemKind::Filter => (ItemTag::Filter, 0.0),
            ItemKind::When(_) => (ItemTag::When, 0.0),
            ItemKind::Every(ms) => (ItemTag::Every, *ms),
        };
        let slots = cx.next_slot.max(item.params.len() as u8);
        module.items.push(ItemCode {
            name: item.name.clone(),
            tag,
            params: item.params.clone(),
            code: std::mem::take(&mut cx.code),
            line: item.line,
            line_map: std::mem::take(&mut cx.line_map),
            period_ms: period,
            slots,
        });
    }
    Ok(module)
}

struct Cx<'a> {
    consts: &'a mut Vec<Const>,
    caps: &'a mut Vec<CapRef>,
    code: Vec<u32>,
    line_map: Vec<u32>,
    current_line: u32,
    scopes: Vec<Vec<(String, u8)>>,
    next_slot: u8,
    item_name: String,
}

impl<'a> Cx<'a> {
    fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), CompileError> {
        self.scopes.push(Vec::new());
        for stmt in stmts {
            self.stmt(stmt)?;
        }
        self.scopes.pop();
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<(), CompileError> {
        let previous_line = self.current_line;
        if stmt.line() > 0 { self.current_line = stmt.line(); }
        match stmt {
            Stmt::Let(name, value, _line) => {
                self.expr(value)?;
                let slot = self.bind(name)?;
                self.emit(Op::StoreLocal, slot, 0, 0);
            }
            Stmt::Set(name, value, line) => {
                self.expr(value)?;
                let Some(slot) = self.resolve(name) else {
                    return e(*line, format!("'{}' is not defined", name));
                };
                self.emit(Op::StoreLocal, slot, 0, 0);
            }
            Stmt::Eval(expr, _line) => {
                self.expr(expr)?;
                match expr {
                    Expr::Call { recv, op, .. }
                        if caps::family_of(recv) == "policy"
                            && (op == "allow" || op == "reject") =>
                    {
                        self.emit(Op::PolicyStop, 0, 0, 0);
                    }
                    _ => {
                        self.emit(Op::Pop, 0, 0, 0);
                    }
                }
            }
            Stmt::If { cond, then, otherwise, .. } => {
                let to_else = self.jump_if_false(cond)?;
                self.stmts(then)?;
                if otherwise.is_empty() {
                    self.patch(to_else);
                } else {
                    let to_end = self.emit_wide(Op::Jmp, 0, 0);
                    self.patch(to_else);
                    self.stmts(otherwise)?;
                    self.patch(to_end);
                }
            }
            Stmt::ForEach { var, list, body, line } => {
                // Three hidden slots: the list, the index, the loop variable.
                self.expr(list)?;
                let list_slot = self.temp()?;
                self.emit(Op::StoreLocal, list_slot, 0, 0);
                let zero = self.push_const(Const::Num(0.0))?;
                self.emit_wide(Op::Const, 0, zero);
                let idx_slot = self.temp()?;
                self.emit(Op::StoreLocal, idx_slot, 0, 0);

                // The loop head carries its own source line, so an exhausted
                // budget points at the loop instead of at the routine.
                let head = self.code.len();
                self.emit_wide(Op::FuelTick, 0, *line);
                self.emit(Op::LoadLocal, list_slot, 0, 0);
                self.emit(Op::Len, 0, 0, 0);
                self.emit(Op::LoadLocal, idx_slot, 0, 0);
                self.emit(Op::CmpLe, 0, 0, 0);
                let to_end = self.emit_wide(Op::JmpIfTrue, 0, 0);

                self.scopes.push(Vec::new());
                let var_slot = self.bind(var)?;
                self.emit(Op::LoadLocal, list_slot, 0, 0);
                self.emit(Op::LoadLocal, idx_slot, 0, 0);
                self.emit(Op::GetIndex, 0, 0, 0);
                self.emit(Op::StoreLocal, var_slot, 0, 0);
                for s in body {
                    self.stmt(s)?;
                }
                self.scopes.pop();

                self.emit(Op::LoadLocal, idx_slot, 0, 0);
                let one = self.push_const(Const::Num(1.0))?;
                self.emit_wide(Op::Const, 0, one);
                self.emit(Op::Add, 0, 0, 0);
                self.emit(Op::StoreLocal, idx_slot, 0, 0);
                self.jump_back_to(head);
                self.patch(to_end);
            }
            Stmt::While { cond, body, line } => {
                let head = self.code.len();
                self.emit_wide(Op::FuelTick, 0, *line);
                let to_end = self.jump_if_false(cond)?;
                for s in body {
                    self.stmt(s)?;
                }
                self.jump_back_to(head);
                self.patch(to_end);
            }
            Stmt::Return(value, _line) => match value {
                Some(v) => {
                    self.expr(v)?;
                    self.emit(Op::Ret, 0, 0, 0);
                }
                None => {
                    self.emit(Op::RetUnit, 0, 0, 0);
                }
            },
        }
        self.current_line = previous_line;
        Ok(())
    }

    /// Branch when `cond` is false. `a == b` and `a != b` compile to the
    /// fused compare-and-branch opcodes: one instruction fewer per decision and
    /// no boolean materialised on the stack.
    fn jump_if_false(&mut self, cond: &Expr) -> Result<usize, CompileError> {
        if let Expr::Bin(op @ (BinOp::Eq | BinOp::Ne), left, right) = cond {
            self.expr(left)?;
            self.expr(right)?;
            // Jump to the else-branch when the comparison does NOT hold.
            let fused = if *op == BinOp::Eq { Op::Jne } else { Op::Jeq };
            return Ok(self.emit_wide(fused, 0, 0));
        }
        self.expr(cond)?;
        Ok(self.emit_wide(Op::JmpIfFalse, 0, 0))
    }

    fn expr(&mut self, expr: &Expr) -> Result<(), CompileError> {
        match expr {
            Expr::Num(n, _) => {
                let idx = self.push_const(Const::Num(*n))?;
                self.emit_wide(Op::Const, 0, idx);
            }
            Expr::Str(s) => {
                let idx = self.push_const(Const::Str(s.clone()))?;
                self.emit_wide(Op::Const, 0, idx);
            }
            Expr::Bool(b) => {
                let idx = self.push_const(Const::Bool(*b))?;
                self.emit_wide(Op::Const, 0, idx);
            }
            Expr::List(items) => {
                if items.len() > 255 {
                    return e(0, "this list has more than 255 items");
                }
                for i in items {
                    self.expr(i)?;
                }
                self.emit(Op::NewArr, items.len() as u8, 0, 0);
            }
            Expr::Name(name) => {
                match self.resolve(name) {
                    Some(slot) => {
                        self.emit(Op::LoadLocal, slot, 0, 0);
                    }
                    None => {
                        return e(0, format!("'{}' is not defined in '{}'", name, self.item_name))
                    }
                }
            }
            Expr::Ambient { field, family } => {
                let idx = self.push_const(Const::Str(field.clone()))?;
                self.emit_wide(Op::Const, 0, idx);
                let cap = self.push_cap(family, "telemetry", 1)?;
                self.emit_wide(Op::CallCap, 1, cap);
            }
            Expr::Call { recv, op, args, line } => {
                if args.len() > 255 {
                    return e(*line, format!("{}.{} takes too many values", recv, op));
                }
                for a in args {
                    self.expr(a)?;
                }
                let family = caps::family_of(recv);
                let cap = self.push_cap(family, op, args.len() as u8)?;
                self.emit_wide(Op::CallCap, args.len() as u8, cap);
            }
            Expr::Not(inner) => {
                self.expr(inner)?;
                self.emit(Op::Not, 0, 0, 0);
            }
            Expr::Len(inner) => {
                self.expr(inner)?;
                self.emit(Op::Len, 0, 0, 0);
            }
            Expr::Index(list, idx) => {
                self.expr(list)?;
                self.expr(idx)?;
                self.emit(Op::GetIndex, 0, 0, 0);
            }
            Expr::Field { base, field, line } => {
                match self.resolve(base) {
                    Some(slot) => {
                        self.emit(Op::LoadLocal, slot, 0, 0);
                    }
                    None => return e(*line, format!("'{}' is not a value here", base)),
                }
                let idx = self.push_const(Const::Str(field.clone()))?;
                self.emit_wide(Op::Const, 0, idx);
                self.emit(Op::GetField, 0, 0, 0);
            }
            Expr::Bin(op, left, right) => match op {
                BinOp::And => {
                    self.expr(left)?;
                    let to_false = self.emit_wide(Op::JmpIfFalse, 0, 0);
                    self.expr(right)?;
                    let to_end = self.emit_wide(Op::Jmp, 0, 0);
                    self.patch(to_false);
                    let f = self.push_const(Const::Bool(false))?;
                    self.emit_wide(Op::Const, 0, f);
                    self.patch(to_end);
                }
                BinOp::Or => {
                    self.expr(left)?;
                    let to_rhs = self.emit_wide(Op::JmpIfFalse, 0, 0);
                    let t = self.push_const(Const::Bool(true))?;
                    self.emit_wide(Op::Const, 0, t);
                    let to_end = self.emit_wide(Op::Jmp, 0, 0);
                    self.patch(to_rhs);
                    self.expr(right)?;
                    self.patch(to_end);
                }
                _ => {
                    self.expr(left)?;
                    self.expr(right)?;
                    let opcode = match op {
                        BinOp::Add => Op::Add,
                        BinOp::Sub => Op::Sub,
                        BinOp::Mul => Op::Mul,
                        BinOp::Div => Op::Div,
                        BinOp::Mod => Op::Mod,
                        BinOp::Lt => Op::CmpLt,
                        BinOp::Le => Op::CmpLe,
                        BinOp::Gt => Op::CmpGt,
                        BinOp::Ge => Op::CmpGe,
                        BinOp::Eq => Op::CmpEq,
                        BinOp::Ne => Op::CmpNe,
                        BinOp::And | BinOp::Or => unreachable!(),
                    };
                    self.emit(opcode, 0, 0, 0);
                }
            },
        }
        Ok(())
    }

    // ── emission ─────────────────────────────────────────────────────
    fn emit(&mut self, op: Op, a: u8, b: u8, c: u8) -> usize {
        let at = self.code.len();
        self.code.push(word(op, a, b, c));
        self.line_map.push(self.current_line);
        at
    }

    fn emit_wide(&mut self, op: Op, a: u8, imm: u32) -> usize {
        let at = self.code.len();
        self.code.push(word(op, a, 0, 0));
        self.code.push(imm);
        self.line_map.push(self.current_line);
        self.line_map.push(self.current_line);
        at
    }

    /// Point a forward jump emitted at `at` at the current end of code.
    fn patch(&mut self, at: usize) {
        let rel = self.code.len() as i64 - (at as i64 + 2);
        if let Some(slot) = self.code.get_mut(at + 1) {
            *slot = rel as u32;
        }
    }

    /// Emit a backward jump to `head`.
    fn jump_back_to(&mut self, head: usize) {
        let at = self.code.len();
        let rel = head as i64 - (at as i64 + 2);
        self.emit_wide(Op::Jmp, 0, rel as u32);
    }

    fn push_const(&mut self, c: Const) -> Result<u32, CompileError> {
        for (i, existing) in self.consts.iter().enumerate() {
            let same = match (existing, &c) {
                (Const::Num(a), Const::Num(b)) => a.to_bits() == b.to_bits(),
                (Const::Str(a), Const::Str(b)) => a == b,
                (Const::Bool(a), Const::Bool(b)) => a == b,
                _ => false,
            };
            if same {
                return Ok(i as u32);
            }
        }
        self.consts.push(c);
        Ok(self.consts.len() as u32 - 1)
    }

    fn push_cap(&mut self, family: &str, op: &str, argc: u8) -> Result<u32, CompileError> {
        for (i, existing) in self.caps.iter().enumerate() {
            if existing.family == family && existing.op == op && existing.argc == argc {
                return Ok(i as u32);
            }
        }
        self.caps.push(CapRef { family: family.to_string(), op: op.to_string(), argc });
        Ok(self.caps.len() as u32 - 1)
    }

    // ── slots ────────────────────────────────────────────────────────
    fn bind(&mut self, name: &str) -> Result<u8, CompileError> {
        let slot = self.temp()?;
        if let Some(top) = self.scopes.last_mut() {
            top.push((name.to_string(), slot));
        }
        Ok(slot)
    }

    fn temp(&mut self) -> Result<u8, CompileError> {
        if self.next_slot >= MAX_SLOTS {
            return e(0, "this block uses more than 255 values; split it into actions");
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        Ok(slot)
    }

    fn resolve(&self, name: &str) -> Option<u8> {
        for scope in self.scopes.iter().rev() {
            for (n, slot) in scope.iter().rev() {
                if n == name {
                    return Some(*slot);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfs::isa::opcode;
    use crate::sfs::parser::parse;

    fn compiles(src: &str) -> Module {
        compile(&parse(src).expect("parse")).expect("compile")
    }

    #[test]
    fn the_patrol_example_compiles_to_a_call_per_verb() {
        let src = "needs: drone, log\n\naction patrol(points):\n    log \"go\"\n    arm drone\n    takeoff to 15 meters\n    return to launch\n";
        let m = compiles(src);
        let item = m.item("patrol").expect("patrol");
        let caps: Vec<String> = m.caps.iter().map(|c| format!("{}.{}", c.family, c.op)).collect();
        assert!(caps.contains(&"drone.arm".to_string()), "{:?}", caps);
        assert!(caps.contains(&"drone.takeoff".to_string()), "{:?}", caps);
        assert!(caps.contains(&"drone.rtl".to_string()), "{:?}", caps);
        assert!(caps.contains(&"log.info".to_string()), "{:?}", caps);
        assert_eq!(item.params, vec!["points".to_string()]);
    }

    #[test]
    fn compilation_is_byte_identical() {
        let src = "needs: rover, lidar, log\n\nevery 100ms:\n    let front = lidar.front_distance()\n    if front < 0.5 meters:\n        stop rover\n    otherwise:\n        drive forward at 1.2 m/s\n";
        let a = crate::sfs::isa::serialize(&compiles(src));
        let b = crate::sfs::isa::serialize(&compiles(src));
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn equality_branches_fuse_into_one_instruction() {
        let m = compiles(
            "needs: log\naction a(x, y):\n    if x == y:\n        log \"same\"\n    if x != y:\n        log \"different\"\n",
        );
        let item = m.item("a").unwrap();
        let ops: Vec<u8> = item.code.iter().map(|w| opcode(*w)).collect();
        assert!(ops.contains(&(Op::Jne as u8)), "== should branch on JNE: {:?}", ops);
        assert!(ops.contains(&(Op::Jeq as u8)), "!= should branch on JEQ: {:?}", ops);
        assert!(
            !ops.contains(&(Op::CmpEq as u8)),
            "a fused branch must not also materialise a boolean: {:?}",
            ops
        );
    }

    #[test]
    fn constants_are_deduplicated() {
        let m = compiles("needs: log\naction a():\n    log \"same\"\n    log \"same\"\n");
        assert_eq!(m.consts.len(), 1);
    }

    #[test]
    fn a_loop_emits_a_fuel_checkpoint_and_a_back_edge() {
        let m = compiles("needs: log\naction a(items):\n    for each i in items:\n        log \"x\"\n");
        let item = m.item("a").unwrap();
        let ops: Vec<u8> = item.code.iter().map(|w| opcode(*w)).collect();
        assert!(ops.contains(&(Op::FuelTick as u8)));
        assert!(ops.contains(&(Op::Jmp as u8)));
    }

    #[test]
    fn a_while_loop_is_bounded_by_fuel_not_by_the_compiler() {
        let m = compiles("needs: log\naction a():\n    while true:\n        log \"spin\"\n");
        assert!(m.item("a").unwrap().code.iter().any(|w| opcode(*w) == Op::FuelTick as u8));
    }
}
