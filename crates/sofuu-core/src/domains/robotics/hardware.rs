// rt/hardware.rs — the R7 hardware safety firewall for Kaze capabilities.
//
// PLAN-KAZE-LANGUAGE §1.3 / §3.3.2 and RULES.md §1.7: "Kaze code has zero
// built-in I/O. It can only execute physical or ambient side-effects through
// explicitly declared capabilities (`needs: drone, sensor`) which are strictly
// intercepted and validated by Sofuu's R7 hardware safety firewall
// (rt/hardware.rs) with reject-never-clamp semantics."
//
// Three gates, all structural — a capability can never skip one, because the
// VM's only side-effect path is CALL_CAP and this module is what that path
// consults:
//
//   H-1  the family is declared in the script's `needs:`          E_CAP_UNDECLARED
//   H-2  an effector family has an explicit host backend grant     E_CAP_MISSING
//   H-3  the host policy does not deny this exact operation        E_CAP_DENIED
//   H-4  every numeric argument is inside its physical range       E_KAZE_RANGE
//
// "Reject never clamp": a takeoff to 900 m does not quietly become 500 m. The
// call fails, the message says what the limits are, and the script decides.
//
// In-process families (log, memory, policy) need a declaration but no backend
// grant: declaring them is the whole opt-in, and they cannot move a machine.

use serde_json::Value as Json;
use serde_json::json;

use crate::sfs::arena::Arena;
use crate::sfs::value::Value;

/// Families that touch a machine or a human's attention. A host must grant
/// each one explicitly (`--cap drone=sim`, `caps: { drone: 'sim' }`).
pub const EFFECTOR_FAMILIES: &[&str] = &["drone", "rover", "sensor", "device", "alert"];

pub fn needs_backend_grant(family: &str) -> bool {
    EFFECTOR_FAMILIES.contains(&family)
}

#[derive(Debug, Clone)]
pub struct Denial {
    pub code: &'static str,
    pub rule: &'static str,
    pub reason: String,
}

impl Denial {
    fn new(code: &'static str, rule: &'static str, reason: String) -> Self {
        Denial { code, rule, reason }
    }
}

/// H-1 + H-2 + H-3 for one capability call.
pub fn authorise(
    declared: &[String],
    grants: &[String],
    policy: Option<&Json>,
    family: &str,
    op: &str,
) -> Result<(), Denial> {
    // Declared resources resolve to families here exactly as the checker
    // resolved them: `needs: lidar` declares the `sensor` family, so the gate
    // and the front end can never disagree about whether a call is declared.
    let declared_families = || declared.iter().any(|d| crate::sfs::caps::family_of(d) == family);
    if !declared_families() {
        return Err(Denial::new(
            "E_CAP_UNDECLARED",
            "H-1/needs",
            format!(
                "'{}' is not declared. Add it to the script's first line: needs: {}",
                family, family
            ),
        ));
    }
    if needs_backend_grant(family)
        && !grants.iter().any(|g| crate::sfs::caps::family_of(g) == family)
    {
        return Err(Denial::new(
            "E_CAP_MISSING",
            "H-2/backend",
            format!(
                "the host did not grant a backend for '{}'. Grant one explicitly, e.g. --cap {}=sim",
                family, family
            ),
        ));
    }
    if let Some(p) = policy {
        let req = json!({ "op": "tool.run", "tool": format!("kaze.{}.{}", family, op) });
        let decision = crate::rt::policy::evaluate(&req, p);
        evaluate_verdict(&decision, family, op)?;
    }
    Ok(())
}

pub fn evaluate_verdict(decision: &serde_json::Value, family: &str, op: &str) -> Result<(), Denial> {
    let verdict = decision.get("decision").and_then(|v| v.as_str()).unwrap_or("");
    match verdict {
        "allow" => Ok(()),
        "deny" => {
            let rule = decision.get("rule").and_then(|v| v.as_str()).unwrap_or("R7.6");
            let reason = decision
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("the host policy denies this capability");
            Err(Denial::new("E_CAP_DENIED", "H-3/policy", format!("{} ({})", reason, rule)))
        }
        "ask" => {
            Err(Denial::new(
                "E_CAP_APPROVAL_REQUIRED",
                "H-3/approval",
                format!(
                    "the host policy requires approval for kaze.{}.{}; a control loop has no approver, so the call is refused",
                    family, op
                ),
            ))
        }
        other => {
            Err(Denial::new(
                "E_CAP_DENIED",
                "H-3/policy",
                format!(
                    "the policy returned an unreadable decision ({}); refusing rather than permitting",
                    if other.is_empty() { "<missing>" } else { other }
                ),
            ))
        }
    }
}

fn num_arg(args: &[Value], idx: usize) -> Option<f64> {
    args.get(idx).and_then(|v| v.as_num())
}

fn range(family: &str, op: &str, idx: usize, args: &[Value], lo: f64, hi: f64, unit: &str) -> Result<(), Denial> {
    let Some(n) = num_arg(args, idx) else {
        return Err(Denial::new(
            "E_KAZE_RANGE",
            "H-4/range",
            format!(
                "{}.{} needs a numeric {} as argument {}, but got none (rejected, not clamped)",
                family, op, unit, idx + 1
            ),
        ));
    };
    if !n.is_finite() || n < lo || n > hi {
        return Err(Denial::new(
            "E_KAZE_RANGE",
            "H-4/range",
            format!(
                "{}.{} got {} {}, which is outside the permitted range {} to {} {} (rejected, not clamped)",
                family,
                op,
                crate::sfs::arena::fmt_num(n),
                unit,
                crate::sfs::arena::fmt_num(lo),
                crate::sfs::arena::fmt_num(hi),
                unit
            ),
        ));
    }
    Ok(())
}

/// H-4: physical argument ranges. Out-of-range values are refused, never
/// adjusted — a machine that flew to the clamped altitude instead of the one
/// in the script is a machine acting on a lie.
pub fn validate_arguments(family: &str, op: &str, args: &[Value], _arena: &Arena) -> Result<(), Denial> {
    match (family, op) {
        ("drone", "takeoff") => range(family, op, 0, args, 0.0, 500.0, "meters")?,
        ("drone", "goto") => {
            // goto(lat, lon, alt) -> altitude at index 2 (sfs/caps.rs:357)
            // goto(lat, alt)      -> altitude at index 1 (fly to X at Y)
            let idx = if args.len() >= 3 { 2 } else { 1 };
            range(family, op, idx, args, 0.0, 500.0, "meters")?
        }
        ("rover", "forward") | ("rover", "backward") => range(family, op, 0, args, 0.0, 10.0, "m/s")?,
        ("rover", "turn_left") | ("rover", "turn_right") => range(family, op, 0, args, 0.0, 360.0, "degrees")?,
        ("device", "pulse") => range(family, op, 1, args, 0.0, 3600.0, "seconds")?,
        ("device", "read_gpio") => range(family, op, 0, args, 0.0, 255.0, "pin")?,
        ("sensor", "read") => {
            if args.is_empty() {
                return Err(Denial::new(
                    "E_KAZE_RANGE",
                    "H-4/range",
                    "sensor.read needs a pin or sensor name as argument 1, but got none (rejected, not clamped)".to_string(),
                ));
            }
            if matches!(args.first(), Some(Value::Num(_))) {
                range(family, op, 0, args, 0.0, 255.0, "pin")?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(name: &str) -> Vec<String> {
        vec![name.to_string()]
    }

    #[test]
    fn undeclared_family_is_refused() {
        let d = authorise(&one("log"), &[], None, "drone", "arm").unwrap_err();
        assert_eq!(d.code, "E_CAP_UNDECLARED");
        assert!(d.reason.contains("needs:"), "{}", d.reason);
    }

    #[test]
    fn declared_effector_without_backend_is_refused() {
        let d = authorise(&one("drone"), &[], None, "drone", "arm").unwrap_err();
        assert_eq!(d.code, "E_CAP_MISSING");
        assert!(d.reason.contains("--cap drone=sim"), "{}", d.reason);
    }

    #[test]
    fn declared_effector_with_backend_passes() {
        assert!(authorise(&one("drone"), &one("drone"), None, "drone", "arm").is_ok());
    }

    #[test]
    fn in_process_family_needs_no_backend_grant() {
        assert!(authorise(&one("log"), &[], None, "log", "info").is_ok());
    }

    #[test]
    fn policy_denial_wins_even_when_everything_else_passes() {
        let policy = json!({ "tools": { "deny": ["kaze.drone.*"] } });
        let d = authorise(&one("drone"), &one("drone"), Some(&policy), "drone", "arm").unwrap_err();
        assert_eq!(d.code, "E_CAP_DENIED");
        assert!(d.reason.contains("denied"), "{}", d.reason);
    }

    #[test]
    fn an_unreadable_policy_decision_denies() {
        let decision = json!({ "no_decision_field": true });
        let d = evaluate_verdict(&decision, "drone", "arm").unwrap_err();
        assert_eq!(d.code, "E_CAP_DENIED");
        assert!(d.reason.to_lowercase().contains("unreadable"), "{}", d.reason);
    }

    #[test]
    fn an_ask_verdict_does_not_permit_a_hardware_call() {
        let p = json!({ "approval": { "mode": "always" } });
        let d = authorise(&one("drone"), &one("drone"), Some(&p), "drone", "arm").unwrap_err();
        assert_eq!(d.code, "E_CAP_APPROVAL_REQUIRED");
    }

    #[test]
    fn a_missing_argument_is_refused_not_ignored() {
        let a = Arena::new(1024);
        let d = validate_arguments("drone", "takeoff", &[], &a).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
    }

    #[test]
    fn goto_three_arg_checks_altitude_not_longitude() {
        let a = Arena::new(1024);
        // Valid western-hemisphere waypoints must be accepted
        assert!(validate_arguments("drone", "goto", &[Value::Num(52.1), Value::Num(-4.1), Value::Num(15.0)], &a).is_ok());
        assert!(validate_arguments("drone", "goto", &[Value::Num(40.7), Value::Num(-74.0), Value::Num(30.0)], &a).is_ok());
    }

    #[test]
    fn goto_three_arg_rejects_over_altitude() {
        let a = Arena::new(1024);
        let d = validate_arguments("drone", "goto", &[Value::Num(52.1), Value::Num(4.1), Value::Num(900.0)], &a).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
        assert_eq!(d.rule, "H-4/range");
    }

    #[test]
    fn goto_two_arg_still_checks_altitude() {
        let a = Arena::new(1024);
        let d = validate_arguments("drone", "goto", &[Value::Num(52.1), Value::Num(900.0)], &a).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
    }

    #[test]
    fn out_of_range_arguments_are_rejected_not_clamped() {
        let arena = Arena::new(1024);
        let d = validate_arguments("drone", "takeoff", &[Value::Num(900.0)], &arena).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
        assert!(d.reason.contains("not clamped"), "{}", d.reason);
        assert!(validate_arguments("drone", "takeoff", &[Value::Num(15.0)], &arena).is_ok());
    }
}
