// sfs/checker.rs — the lightweight checker (M-K2).
//
// PLAN-KAZE-LANGUAGE §5 (M-K2): "Capability verification, type inference,
// exhaustiveness checks." The checker is what makes the safety story true
// before a single instruction runs:
//
//   * every capability family used is declared in `needs:` (H-1, early);
//   * every operation is a real v0.1 operation with the right argument count;
//   * every name resolves to a binding (no accidental ambient state);
//   * a comparison between `meters` and `m/s` is refused;
//   * a `filter` ends in `allow` or `reject` on every path — a guardrail that
//     can fall off the end of the function is not a guardrail.
//
// It is deliberately not a full type system: it knows numbers/units, text,
// true-or-false, lists and "unknown" (what a capability returns). Unknown is
// permissive, so a host that returns a new shape does not need a language
// change. Every error carries a line and a suggestion.

use crate::sfs::ast::*;
use crate::sfs::caps;

#[derive(Debug, Clone)]
pub struct CheckError {
    pub line: u32,
    pub msg: String,
}

impl CheckError {
    pub fn render(&self, source_name: &str) -> String {
        format!("{}:{}: {}", source_name, self.line, self.msg)
    }
}

fn e(line: u32, msg: impl Into<String>) -> CheckError {
    CheckError { line, msg: msg.into() }
}

/// What the checker can prove about a value. `Any` is "the host decides".
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ty {
    Num(Option<Unit>),
    Str,
    Bool,
    List,
    Unit,
    Any,
}

impl Ty {
    fn label(self) -> &'static str {
        match self {
            Ty::Num(Some(u)) => u.label(),
            Ty::Num(None) => "a number",
            Ty::Str => "text",
            Ty::Bool => "true or false",
            Ty::List => "a list",
            Ty::Unit => "nothing",
            Ty::Any => "a host value",
        }
    }
}

/// The declared rows, so `filter` bodies can be checked for a verdict.
pub fn check(p: &Program) -> Result<(), Vec<CheckError>> {
    let mut errors: Vec<CheckError> = Vec::new();
    let declared: Vec<&str> = p.needs.iter().map(|s| s.as_str()).collect();

    for need in &p.needs {
        let family = caps::family_of(need);
        if need != family && !caps::FAMILIES.contains(&family) {
            errors.push(e(
                1,
                format!("'{}' does not name a capability family or an instance", need),
            ));
        }
    }

    for item in &p.items {
        let mut scope: Vec<(String, Ty)> = Vec::new();
        for p in &item.params {
            scope.push((p.clone(), Ty::Any));
        }
        if let ItemKind::When(cond) = &item.kind {
            let t = infer(cond, &mut scope, &declared, &mut errors);
            if !matches!(t, Ty::Bool | Ty::Any) {
                errors.push(e(
                    item.line,
                    format!("a 'when' condition must be true or false, but this is {}", t.label()),
                ));
            }
        }
        check_block(&item.body, &mut scope, &declared, &mut errors);
        if matches!(item.kind, ItemKind::Filter) && !terminates(&item.body) {
            errors.push(e(
                item.line,
                format!(
                    "filter '{}' can reach its end without allowing or rejecting; end every \
                     path with 'allow …' or 'reject with …'",
                    item.name
                ),
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn check_block(
    stmts: &[Stmt],
    scope: &mut Vec<(String, Ty)>,
    declared: &[&str],
    errors: &mut Vec<CheckError>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Let(name, value, line) => {
                let t = infer(value, scope, declared, errors);
                if let Some(slot) = scope.iter_mut().find(|(n, _)| n == name) {
                    slot.1 = t;
                } else {
                    scope.push((name.clone(), t));
                }
                let _ = line;
            }
            Stmt::Set(name, value, line) => {
                infer(value, scope, declared, errors);
                if !scope.iter().any(|(n, _)| n == name) {
                    errors.push(e(*line, format!("'{}' is not defined", name)));
                }
            }
            Stmt::Eval(expr, _) => {
                infer(expr, scope, declared, errors);
            }
            Stmt::If { cond, then, otherwise, line } => {
                let t = infer(cond, scope, declared, errors);
                if !matches!(t, Ty::Bool | Ty::Any) {
                    errors.push(e(
                        *line,
                        format!("an 'if' condition must be true or false, but this is {}", t.label()),
                    ));
                }
                let mut inner = scope.clone();
                check_block(then, &mut inner, declared, errors);
                let mut inner2 = scope.clone();
                check_block(otherwise, &mut inner2, declared, errors);
            }
            Stmt::ForEach { var, list, body, line } => {
                let t = infer(list, scope, declared, errors);
                if !matches!(t, Ty::List | Ty::Any) {
                    errors.push(e(
                        *line,
                        format!("'for each' walks a list, but '{}' is {}", expr_label(list), t.label()),
                    ));
                }
                let mut inner = scope.clone();
                inner.push((var.clone(), Ty::Any));
                check_block(body, &mut inner, declared, errors);
            }
            Stmt::While { cond, body, line } => {
                let t = infer(cond, scope, declared, errors);
                if !matches!(t, Ty::Bool | Ty::Any) {
                    errors.push(e(
                        *line,
                        format!("a 'while' condition must be true or false, but this is {}", t.label()),
                    ));
                }
                let mut inner = scope.clone();
                check_block(body, &mut inner, declared, errors);
            }
            Stmt::Return(value, _) => {
                if let Some(v) = value {
                    infer(v, scope, declared, errors);
                }
            }
        }
    }
}

fn expr_label(e: &Expr) -> String {
    match e {
        Expr::Name(n) => n.clone(),
        Expr::Ambient { field, .. } => field.clone(),
        Expr::Call { recv, op, .. } => format!("{}.{}", recv, op),
        Expr::Field { base, field, .. } => format!("{}.{}", base, field),
        other => format!("{:?}", std::mem::discriminant(other)),
    }
}

fn infer(
    expr: &Expr,
    scope: &mut Vec<(String, Ty)>,
    declared: &[&str],
    errors: &mut Vec<CheckError>,
) -> Ty {
    match expr {
        Expr::Num(_, unit) => Ty::Num(if *unit == Unit::None { None } else { Some(*unit) }),
        Expr::Str(_) => Ty::Str,
        Expr::Bool(_) => Ty::Bool,
        Expr::List(items) => {
            for i in items {
                infer(i, scope, declared, errors);
            }
            Ty::List
        }
        Expr::Name(name) => match scope.iter().rev().find(|(n, _)| n == name) {
            Some((_, t)) => *t,
            None => {
                errors.push(e(
                    0,
                    format!(
                        "'{}' is not defined here; add it as a parameter, a 'let', or a capability value",
                        name
                    ),
                ));
                Ty::Any
            }
        },
        Expr::Ambient { field, family } => {
            if !declared.iter().any(|d| caps::family_of(d) == *family) {
                errors.push(e(
                    0,
                    format!("'{}' is read from '{}', which this script does not declare", field, family),
                ));
            }
            Ty::Any
        }
        Expr::Field { base, line, .. } => {
            if !scope.iter().rev().any(|(n, _)| n == base) {
                errors.push(e(
                    *line,
                    format!("'.{}' reads a field of '{}', which is not a value here", "", base),
                ));
            }
            Ty::Any
        }
        Expr::Not(inner) => {
            let t = infer(inner, scope, declared, errors);
            if !matches!(t, Ty::Bool | Ty::Any) {
                errors.push(e(0, format!("'not' needs true or false, but this is {}", t.label())));
            }
            Ty::Bool
        }
        Expr::Len(inner) => {
            let t = infer(inner, scope, declared, errors);
            if !matches!(t, Ty::Str | Ty::List | Ty::Any) {
                errors.push(e(0, format!("'length of' needs text or a list, but this is {}", t.label())));
            }
            Ty::Num(None)
        }
        Expr::Index(list, idx) => {
            let lt = infer(list, scope, declared, errors);
            let it = infer(idx, scope, declared, errors);
            if !matches!(lt, Ty::List | Ty::Str | Ty::Any) {
                errors.push(e(0, format!("[ ] needs a list or text, but this is {}", lt.label())));
            }
            if !matches!(it, Ty::Num(_) | Ty::Any) {
                errors.push(e(0, format!("a position must be a number, but this is {}", it.label())));
            }
            Ty::Any
        }
        Expr::Call { recv, op, args, line } => {
            let family = caps::family_of(recv);
            if !declared.iter().any(|d| *d == recv || caps::family_of(d) == family) {
                errors.push(e(
                    *line,
                    format!(
                        "'{}' is not declared. Add '{}' to the script's first line: needs: {}",
                        family, family, family
                    ),
                ));
            }
            let arg_tys: Vec<Ty> = args.iter().map(|a| infer(a, scope, declared, errors)).collect();
            match caps::op_arity(family, op) {
                None => errors.push(e(
                    *line,
                    format!(
                        "{}.{} is not something a script can ask for. Known {} operations: {}",
                        family,
                        op,
                        family,
                        known_ops(family)
                    ),
                )),
                Some((min, max)) => {
                    if arg_tys.len() < min || arg_tys.len() > max {
                        let want = if min == max {
                            format!("{}", min)
                        } else {
                            format!("{} to {}", min, max)
                        };
                        errors.push(e(
                            *line,
                            format!(
                                "{}.{} takes {} value(s), but {} were given",
                                family,
                                op,
                                want,
                                arg_tys.len()
                            ),
                        ));
                    }
                }
            }
            capability_result(family, op)
        }
        Expr::Bin(op, left, right) => {
            let lt = infer(left, scope, declared, errors);
            let rt = infer(right, scope, declared, errors);
            bin_result(*op, lt, rt)
        }
    }
}

fn known_ops(family: &str) -> String {
    let mut names: Vec<&str> = Vec::new();
    for op in [
        "arm", "takeoff", "land", "rtl", "hold", "goto", "stop", "telemetry", "forward", "backward",
        "turn_left", "turn_right", "temperature", "humidity", "distance", "front_distance", "read",
        "set_relay", "set_duty", "pulse", "read_gpio", "save", "recall", "detects", "allow", "reject", "info",
        "warn", "error", "debug", "send", "prompt", "complete", "classify", "call", "run",
    ] {
        if caps::known_op(family, op) {
            names.push(op);
        }
    }
    names.join(", ")
}

fn capability_result(family: &str, op: &str) -> Ty {
    match (family, op) {
        ("ai", "prompt") | ("ai", "complete") | ("ai", "classify") => Ty::Str,
        ("tool", "call") | ("tool", "run") => Ty::Str,
        ("policy", "detects") | ("policy", "detect") => Ty::Bool,
        ("memory", "recall") => Ty::List,
        ("sensor", "temperature")
        | ("sensor", "humidity")
        | ("sensor", "distance")
        | ("sensor", "front_distance")
        | ("sensor", "read")
        | ("device", "read_gpio") => Ty::Num(None),
        ("log", _) | ("alert", _) | ("memory", "save") | ("policy", "reject") | ("drone", "arm")
        | ("drone", "takeoff") | ("drone", "land") | ("drone", "rtl") | ("drone", "hold")
        | ("drone", "goto") | ("drone", "stop") | ("rover", "forward") | ("rover", "backward")
        | ("rover", "turn_left") | ("rover", "turn_right") | ("rover", "stop")
        | ("device", "set_relay") | ("device", "set_duty") | ("device", "pulse") => Ty::Unit,
        _ => Ty::Any,
    }
}

fn bin_result(op: BinOp, lt: Ty, rt: Ty) -> Ty {
    use BinOp::*;
    match op {
        Add | Sub | Mul | Div | Mod => {
            if (lt == Ty::Str || rt == Ty::Str) && op == Add {
                return Ty::Str;
            }
            let unit = match (lt, rt) {
                (Ty::Num(a), Ty::Num(b)) => match (a, b) {
                    (Some(x), Some(y)) if x != y => {
                        return Ty::Num(Some(x));
                    }
                    (Some(x), _) => Some(x),
                    (_, Some(y)) => Some(y),
                    _ => None,
                },
                (Ty::Num(a), _) => a,
                (_, Ty::Num(b)) => b,
                _ => None,
            };
            Ty::Num(unit)
        }
        Lt | Le | Gt | Ge | Eq | Ne => Ty::Bool,
        And | Or => Ty::Bool,
    }
}

/// Does every path through this body end in `allow`, `reject` or `return`?
fn terminates(stmts: &[Stmt]) -> bool {
    let Some(last) = stmts.last() else { return false };
    match last {
        Stmt::Return(..) => true,
        Stmt::Eval(Expr::Call { recv, op, .. }, _) => {
            caps::family_of(recv) == "policy" && (op == "allow" || op == "reject")
        }
        Stmt::If { then, otherwise, .. } => {
            !otherwise.is_empty() && terminates(then) && terminates(otherwise)
        }
        _ => false,
    }
}

/// Unit-compatibility errors that `bin_result` cannot report on its own: a
/// comparison between two concrete, different units is almost always a bug in
/// a hardware script, so it is refused rather than guessed at.
pub fn unit_conflict(left: &Expr, right: &Expr) -> Option<(Unit, Unit)> {
    let unit_of = |x: &Expr| match x {
        Expr::Num(_, Unit::None) => None,
        Expr::Num(_, u) => Some(*u),
        _ => None,
    };
    match (unit_of(left), unit_of(right)) {
        (Some(a), Some(b)) if a != b => Some((a, b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfs::parser::parse;

    fn check_src(src: &str) -> Result<(), Vec<CheckError>> {
        check(&parse(src).expect("parse"))
    }

    fn first_error(src: &str) -> String {
        match check_src(src) {
            Ok(()) => panic!("expected a check error"),
            Err(errs) => errs[0].msg.clone(),
        }
    }

    #[test]
    fn the_four_plan_examples_typecheck() {
        for src in [
            "needs: drone, log\n\naction patrol(points):\n    log \"go\"\n    arm drone\n    takeoff to 15 meters\n    for each point in points:\n        fly to point at 15 meters\n    return to launch\n\nwhen battery < 20%:\n    log warning \"low\"\n    return to launch\n",
            "needs: rover, lidar, log\n\nevery 100ms:\n    let front = lidar.front_distance()\n    if front < 0.5 meters:\n        log warning \"obstacle\"\n        stop rover\n        turn right by 90 degrees\n    otherwise:\n        drive forward at 1.2 m/s\n",
            "needs: sensor, fan, mister, alert\n\ncheck climate(t, h):\n    if sensor.temperature() > t + 2:\n        turn on fan\n        send alert \"cooling\"\n    otherwise:\n        turn off fan\n    if sensor.humidity() < h:\n        pulse mister for 5 seconds\n",
            "needs: memory, policy\n\nfilter user_prompt(prompt):\n    if length of prompt > 2000:\n        reject with \"too long\"\n    if policy detects secrets in prompt:\n        reject with \"secrets\"\n    remember prompt as \"last\"\n    allow prompt\n",
        ] {
            check_src(src).unwrap_or_else(|e| panic!("{}", e[0].msg));
        }
    }

    #[test]
    fn an_undeclared_capability_is_named_with_the_fix() {
        let msg = first_error("needs: log\naction a():\n    arm drone\n");
        assert!(msg.contains("not declared"), "{}", msg);
        assert!(msg.contains("needs: drone"), "{}", msg);
    }

    #[test]
    fn an_unknown_operation_lists_the_real_ones() {
        let msg = first_error("needs: drone\naction a():\n    drone.flyaway()\n");
        assert!(msg.contains("not something a script can ask for"), "{}", msg);
        assert!(msg.contains("takeoff"), "{}", msg);
    }

    #[test]
    fn an_unknown_name_is_refused() {
        let msg = first_error("needs: log\naction a():\n    log missing_name\n");
        assert!(msg.contains("not defined"), "{}", msg);
    }

    #[test]
    fn a_filter_that_can_fall_off_the_end_is_refused() {
        let msg = first_error("needs: policy\n\nfilter f(p):\n    if 1 < 2:\n        allow p\n");
        assert!(msg.contains("without allowing or rejecting"), "{}", msg);
    }

    #[test]
    fn a_filter_with_both_branches_is_accepted() {
        check_src("needs: policy\n\nfilter f(p):\n    if 1 < 2:\n        allow p\n    otherwise:\n        reject with \"no\"\n")
            .expect("balanced filter");
    }

    #[test]
    fn wrong_arity_is_reported() {
        let msg = first_error("needs: drone\naction a():\n    drone.takeoff()\n");
        assert!(msg.contains("takes 1 value"), "{}", msg);
    }

    #[test]
    fn a_non_boolean_condition_is_refused() {
        let msg = first_error("needs: log\naction a():\n    if 5:\n        log \"x\"\n");
        assert!(msg.contains("must be true or false"), "{}", msg);
    }

    #[test]
    fn a_dotted_receiver_must_resolve_to_a_declared_family() {
        // `lidar` is an instance of `sensor`, so declaring the instance is enough…
        check_src("needs: lidar\naction a():\n    let d = lidar.front_distance()\n").expect("alias");
        // …but using a family nobody declared is refused.
        let msg = first_error("needs: drone\naction a():\n    lidar.front_distance()\n");
        assert!(msg.contains("not declared"), "{}", msg);
        assert!(msg.contains("needs: sensor"), "{}", msg);
    }
}
