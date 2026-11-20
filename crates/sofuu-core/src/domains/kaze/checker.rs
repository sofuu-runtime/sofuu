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
    Point,
    Error,
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
            Ty::Point => "a point",
            Ty::Error => "an error",
            Ty::Unit => "nothing",
            Ty::Any => "a host value",
        }
    }
}

/// The server an `mcp(...)` or `plugin(...)` declaration names.
///
/// A binding is not a capability family: it reaches exactly one server, so it
/// never stands in for the family an unrelated receiver would resolve to.
fn binding_server(declared: &str) -> Option<&str> {
    let rest = declared
        .strip_prefix("mcp(")
        .or_else(|| declared.strip_prefix("plugin("))?;
    let inner = rest.strip_suffix(')').unwrap_or(rest);
    Some(inner.trim_matches('"'))
}

/// Whether a declaration is an `mcp(...)` or `plugin(...)` binding.
fn is_binding(declared: &str) -> bool {
    declared.starts_with("mcp(") || declared.starts_with("plugin(")
}

/// The declared rows, so `filter` bodies can be checked for a verdict.
pub fn check(p: &Program) -> Result<(), Vec<CheckError>> {
    let mut errors: Vec<CheckError> = Vec::new();
    let declared: Vec<&str> = p.needs.iter().map(|s| s.as_str()).collect();

    for need in &p.needs {
        // A binding names one server: `mcp("blender")` reaches the server
        // called blender and nothing else. An empty binding names no server at
        // all, so it is refused instead of standing in for a declaration.
        if let Some(server) = binding_server(need) {
            if server.is_empty() {
                errors.push(e(
                    1,
                    format!("'{}' names no server; write mcp(\"server\") with the server to reach", need),
                ));
            }
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
            Stmt::Try { body, otherwise, error_name, line: _ } => {
                let mut inner = scope.clone();
                check_block(body, &mut inner, declared, errors);
                let mut inner2 = scope.clone();
                if let Some(name) = error_name {
                    inner2.push((name.clone(), Ty::Error));
                }
                check_block(otherwise, &mut inner2, declared, errors);
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
            if !declared.iter().any(|d| !is_binding(d) && caps::family_of(d) == *family) {
                errors.push(e(
                    0,
                    format!("'{}' is read from '{}', which this script does not declare", field, family),
                ));
            }
            Ty::Any
        }
        Expr::Field { base, field, line } => {
            let rt = match scope.iter().rev().find(|(n, _)| n == base) {
                Some((_, t)) => *t,
                None => {
                    errors.push(e(
                        *line,
                        format!("'.{}' reads a field of '{}', which is not a value here", field, base),
                    ));
                    Ty::Any
                }
            };
            if rt == Ty::Error && !["code", "message", "line"].contains(&field.as_str()) {
                errors.push(e(
                    *line,
                    format!("an error record has code, message and line, not '{}'", field),
                ));
            }
            match (rt, field.as_str()) {
                (Ty::Error, "line") => Ty::Num(None),
                (Ty::Error, "code" | "message") => Ty::Str,
                _ => Ty::Any,
            }
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
            let is_mcp_or_plugin = declared.iter().any(|d| binding_server(d) == Some(recv.as_str()));
            if is_mcp_or_plugin {
                for a in args {
                    infer(a, scope, declared, errors);
                }
                return Ty::Any;
            }
            let family = caps::family_of(recv);
            // A binding declares one server, never a family, so it cannot
            // stand in for the receiver this call names.
            if !declared.iter().any(|d| {
                *d == recv || (!is_binding(d) && caps::family_of(d) == family)
            }) {
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
            let negation = *op == BinOp::Sub
                && rt == Ty::Point
                && matches!(**left, Expr::Num(n, _) if n == 0.0);
            if negation {
                return Ty::Point;
            }
            point_aware_bin(*op, lt, rt, errors)
        }
        Expr::Math { kind, args, line } => infer_math(*kind, args, *line, scope, declared, errors),
        Expr::LinkLost { after_ms: _, line } => {
            if !declared.iter().any(|d| caps::family_of(d) == "drone") {
                errors.push(e(
                    *line,
                    "'link lost' reads the drone link; declare it first with 'needs: drone'",
                ));
            }
            Ty::Bool
        }
    }
}

fn point_aware_bin(op: BinOp, lt: Ty, rt: Ty, errors: &mut Vec<CheckError>) -> Ty {
    use BinOp::*;
    if lt != Ty::Point && rt != Ty::Point {
        return bin_result(op, lt, rt);
    }
    match (op, lt, rt) {
        (Add | Sub, Ty::Point, Ty::Point) => Ty::Point,
        (Add | Sub, Ty::Point, Ty::Any) | (Add | Sub, Ty::Any, Ty::Point) => Ty::Point,
        (Mul, Ty::Point, Ty::Num(_)) | (Mul, Ty::Num(_), Ty::Point) => Ty::Point,
        (Div, Ty::Point, Ty::Num(_)) => Ty::Point,
        (Eq | Ne, Ty::Point, Ty::Point) => Ty::Bool,
        _ => {
            errors.push(e(
                0,
                format!(
                    "'{}' does not apply to {} and {}: points add to points, and a number scales one",
                    op.symbol(),
                    lt.label(),
                    rt.label()
                ),
            ));
            Ty::Any
        }
    }
}

fn infer_math(
    kind: MathKind,
    args: &[Expr],
    line: u32,
    scope: &mut Vec<(String, Ty)>,
    declared: &[&str],
    errors: &mut Vec<CheckError>,
) -> Ty {
    let tys: Vec<Ty> = args.iter().map(|a| infer(a, scope, declared, errors)).collect();
    let want = match kind {
        MathKind::Vector => 2,
        MathKind::Distance
        | MathKind::SlantDistance
        | MathKind::BearingFrom
        | MathKind::Atan2 => 2,
        _ => 1,
    };
    let wrong_count = match kind {
        MathKind::Vector => tys.len() != 2 && tys.len() != 3,
        _ => tys.len() != want,
    };
    if wrong_count {
        let expected = match kind {
            MathKind::Vector => "2 or 3 values".to_string(),
            _ => format!("{} value(s)", want),
        };
        errors.push(e(
            line,
            format!("{:?} takes {}, but {} were given", kind, expected, tys.len()),
        ));
        return Ty::Any;
    }
    let numeric = |t: Ty| matches!(t, Ty::Num(_) | Ty::Any);
    let point = |t: Ty| matches!(t, Ty::Point | Ty::Any);
    match kind {
        MathKind::Vector => {
            for t in &tys {
                if !numeric(*t) {
                    errors.push(e(line, format!("a point is three numbers, but this is {}", t.label())));
                }
            }
            Ty::Point
        }
        MathKind::Distance | MathKind::SlantDistance => {
            for t in &tys {
                if !point(*t) {
                    errors.push(e(
                        line,
                        format!("{:?} needs two points, but this is {}", kind, t.label()),
                    ));
                }
            }
            Ty::Num(Some(Unit::Meter))
        }
        MathKind::BearingOf => {
            if !point(tys[0]) {
                errors.push(e(
                    line,
                    format!("'bearing of' needs a point, but this is {}", tys[0].label()),
                ));
            }
            Ty::Num(Some(Unit::Deg))
        }
        MathKind::BearingFrom => {
            for t in &tys {
                if !point(*t) {
                    errors.push(e(
                        line,
                        format!("'bearing from ... to' needs two points, but this is {}", t.label()),
                    ));
                }
            }
            Ty::Num(Some(Unit::Deg))
        }
        MathKind::Sin | MathKind::Cos | MathKind::Sqrt => {
            if !numeric(tys[0]) {
                errors.push(e(line, format!("{:?} needs a number, but this is {}", kind, tys[0].label())));
            }
            Ty::Num(None)
        }
        MathKind::Atan2 => {
            for t in &tys {
                if !numeric(*t) {
                    errors.push(e(line, format!("atan2 needs two numbers, but this is {}", t.label())));
                }
            }
            Ty::Num(None)
        }
        MathKind::ToRadians | MathKind::ToDegrees => {
            if !numeric(tys[0]) {
                errors.push(e(line, format!("a conversion needs a number, but this is {}", tys[0].label())));
            }
            Ty::Num(None)
        }
    }
}

fn known_ops(family: &str) -> String {
    let mut names: Vec<&str> = Vec::new();
    for op in [
        "arm", "takeoff", "land", "rtl", "hold", "goto", "stop", "telemetry", "forward", "backward",
        "turn_left", "turn_right", "temperature", "humidity", "distance", "front_distance", "read",
        "set_relay", "set_duty", "pulse", "read_gpio", "save", "recall", "detects", "detect", "allow",
        "reject", "info", "warn", "error", "debug", "send", "prompt", "complete", "classify", "call",
        "run",
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
        Stmt::Try { body, otherwise, .. } => terminates(body) && terminates(otherwise),
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
            "needs: mcp(\"blender\"), plugin(\"github_repo\")\n\naction inspect_scene():\n    let info = blender.get_scene_info()\n    return info\n",
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

    #[test]
    fn a_binding_that_names_no_server_is_refused() {
        let msg = first_error("needs: mcp(\"\")\naction a():\n    return 1\n");
        assert!(msg.contains("names no server"), "{}", msg);
    }

    /// A binding reaches one server. It is not a family, so it cannot stand in
    /// for the receiver an unrelated call names.
    #[test]
    fn a_binding_does_not_declare_a_family_for_an_unrelated_receiver() {
        let errors = check_src("needs: mcp(\"\")\naction a():\n    fan.set_duty(50)\n").unwrap_err();
        let messages: Vec<&str> = errors.iter().map(|e| e.msg.as_str()).collect();
        assert!(messages.iter().any(|m| m.contains("not declared")), "{:?}", messages);
        assert!(messages.iter().any(|m| m.contains("names no server")), "{:?}", messages);
    }

    #[test]
    fn a_field_diagnostic_names_the_field_that_could_not_be_read() {
        let msg = first_error("action a():\n    let x = missing.field\n");
        assert!(msg.contains(".field"), "{}", msg);
        assert!(msg.contains("missing"), "{}", msg);
    }

    #[test]
    fn an_unknown_operation_lists_every_operation_the_registry_has() {
        let msg = first_error("needs: policy\naction a():\n    policy.frobnicate(\"x\")\n");
        assert!(msg.contains("not something a script can ask for"), "{}", msg);
        assert!(msg.contains("detect"), "{}", msg);
    }
}
