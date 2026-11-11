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
//   H-2  the family has an explicit host grant                     E_CAP_MISSING
//        and the granted backend is one this runtime has          E_CAP_INVALID
//   H-3  the host policy does not deny this exact operation        E_CAP_DENIED
//   H-4  every numeric argument is inside its physical range       E_KAZE_RANGE
//
// "Reject never clamp": a takeoff to 900 m does not quietly become 500 m. The
// call fails, the message says what the limits are, and the script decides.
//
// A declaration is not a grant. Every capability call requires an explicit
// family grant and a valid, nonempty policy before the host is reached.

use serde_json::Value as Json;
use serde_json::json;

use crate::sfs::arena::Arena;
use crate::sfs::value::Value;

/// Every registered family requires an explicit host grant.
pub const EFFECTOR_FAMILIES: &[&str] = crate::sfs::caps::FAMILIES;

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

/// The backend names a `--cap <family>=<backend>` grant may carry.
///
/// `sim` is an explicit drone/rover test simulator, `real` asks for the
/// family's live hardware, `mavlink` for its MAVLink link. Anything else is a
/// typo — and a typo used to be accepted at grant time, with the simulator
/// answering every call behind a backend nobody implemented (RB-08).
pub const BACKENDS: &[&str] = &["sim", "real", "mavlink"];

/// The live backends a family actually has. A family missing from this table
/// has none, so a grant naming one is refused at grant time instead of being
/// quietly simulated.
pub fn live_backends(family: &str) -> &'static [&'static str] {
    match family {
        "drone" => &["real", "mavlink"],
        _ => &[],
    }
}

/// The transport spellings a `drone` grant may use instead of a backend name:
/// `udp:HOST:PORT`, `udpout:HOST:PORT`, `uart:PATH[@BAUD]`. These are the exact
/// strings the link layer opens, so the shape is checked at grant time rather
/// than after the script has started.
fn is_link_device(device: &str) -> bool {
    if super::mavlink_link::parse_endpoint(device).is_some() {
        return true;
    }
    let Some(rest) = device.trim().strip_prefix("uart:") else {
        return false;
    };
    let (path, baud) = match rest.rsplit_once('@') {
        Some((p, b)) => (p, Some(b)),
        None => (rest, None),
    };
    !path.trim().is_empty() && baud.map_or(true, |b| b.trim().parse::<u32>().is_ok())
}

/// H-2 at GRANT time (RB-08): one `--cap <family>=<backend>` from the host,
/// judged before the script runs rather than when the first capability fires.
///
/// An unrecognized backend is a usage mistake the host must not paper over: a
/// grant that says `sensor=bogus` used to be accepted, the run proceeded, and
/// the simulator answered for a backend nobody had implemented. It fails here
/// as E_CAP_INVALID instead.
pub fn validate_grant(family: &str, backend: &str) -> Result<(), Denial> {
    let name = backend.trim();
    if name.is_empty() {
        return Ok(());
    }
    let fam = crate::sfs::caps::family_of(family);
    if name == "sim" {
        return if matches!(fam, "drone" | "rover") {
            Ok(())
        } else {
            Err(Denial::new("E_CAP_INVALID", "H-2/backend", format!(
                "simulation is available only for explicitly selected drone or rover tests, not '{}'", fam
            )))
        };
    }
    if BACKENDS.contains(&name) {
        return if live_backends(fam).contains(&name) {
            Ok(())
        } else {
            Err(Denial::new(
                "E_CAP_INVALID",
                "H-2/backend",
                format!(
                    "'{}' has no supported live backend for --cap {}={}",
                    fam, family, name
                ),
            ))
        };
    }
    if fam == "drone" && is_link_device(name) {
        return Ok(());
    }
    Err(Denial::new(
        "E_CAP_INVALID",
        "H-2/backend",
        format!(
            "'{}' is not a capability backend: a grant names sim, real or mavlink (a drone may also name udp:HOST:PORT, udpout:HOST:PORT or uart:PATH@BAUD)",
            name
        ),
    ))
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
                "the host did not grant '{}'. Grant it explicitly, e.g. --cap {}",
                family, family
            ),
        ));
    }
    let p = policy.ok_or_else(|| Denial::new(
        "E_CAP_DENIED", "H-3/policy", format!("no policy is configured for {}.{}", family, op)
    ))?;
    crate::rt::policy::validate_policy(p).map_err(|e| Denial::new(
        "E_CAP_DENIED", "H-3/policy", format!("invalid policy: {}", e)
    ))?;
    // The R7 evaluator defaults to allow when no tools rules are supplied.
    // An unrelated nonempty section must not implicitly authorize Kaze calls.
    let has_tool_rules = p.get("tools").and_then(Json::as_object)
        .is_some_and(|tools| tools.contains_key("allow") || tools.contains_key("deny"));
    if !has_tool_rules {
        return Err(Denial::new(
            "E_CAP_DENIED", "H-3/policy", format!("policy needs tools.allow or tools.deny to authorize {}.{}", family, op)
        ));
    }
    // R7 normally treats an empty allowlist as absent. For Kaze an explicitly
    // empty allowlist grants no operations; do not turn it into default allow.
    if p.pointer("/tools/allow").and_then(Json::as_array).is_some_and(Vec::is_empty) {
        return Err(Denial::new(
            "E_CAP_DENIED", "H-3/policy", format!("tools.allow permits no capabilities, including {}.{}", family, op)
        ));
    }
    let req = json!({ "op": "tool.run", "tool": format!("kaze.{}.{}", family, op) });
    let decision = crate::rt::policy::evaluate(&req, p);
    evaluate_verdict(&decision, family, op)?;
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
        ("device", "set_duty") => range(family, op, 1, args, 0.0, 100.0, "percent")?,
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
        assert!(d.reason.contains("--cap drone"), "{}", d.reason);
    }

    #[test]
    fn declared_effector_with_backend_passes() {
        let p = json!({"tools": {"allow": ["kaze.drone.*"]}});
        assert!(authorise(&one("drone"), &one("drone"), Some(&p), "drone", "arm").is_ok());
    }

    #[test]
    fn in_process_family_requires_a_grant_and_policy() {
        let p = json!({"tools": {"allow": ["kaze.log.*"]}});
        assert_eq!(authorise(&one("log"), &[], Some(&p), "log", "info").unwrap_err().code, "E_CAP_MISSING");
        assert_eq!(authorise(&one("log"), &one("log"), None, "log", "info").unwrap_err().code, "E_CAP_DENIED");
        assert!(authorise(&one("log"), &one("log"), Some(&p), "log", "info").is_ok());
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
        let p = json!({ "tools": {"allow": ["kaze.drone.*"]}, "approval": { "mode": "always" } });
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
    fn a_duty_cycle_outside_zero_to_hundred_percent_is_refused() {
        let arena = Arena::new(1024);
        let d = validate_arguments("device", "set_duty", &[Value::Unit, Value::Num(101.0)], &arena)
            .unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
        assert_eq!(d.rule, "H-4/range");
        assert!(d.reason.contains("percent"), "{}", d.reason);
        assert!(validate_arguments("device", "set_duty", &[Value::Unit, Value::Num(40.0)], &arena)
            .is_ok());
    }

    #[test]
    fn out_of_range_arguments_are_rejected_not_clamped() {
        let arena = Arena::new(1024);
        let d = validate_arguments("drone", "takeoff", &[Value::Num(900.0)], &arena).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
        assert!(d.reason.contains("not clamped"), "{}", d.reason);
        assert!(validate_arguments("drone", "takeoff", &[Value::Num(15.0)], &arena).is_ok());
    }

    /// RB-09: the refusal happens AT the ceiling, not somewhere near it. A
    /// takeoff one metre past the limit fails and the message reports the
    /// altitude the script asked for — a clamped 500 m flight would be a
    /// machine acting on a number nobody wrote.
    #[test]
    fn a_takeoff_past_the_ceiling_is_rejected_at_the_boundary() {
        let arena = Arena::new(1024);
        assert!(validate_arguments("drone", "takeoff", &[Value::Num(500.0)], &arena).is_ok());
        let d = validate_arguments("drone", "takeoff", &[Value::Num(501.0)], &arena).unwrap_err();
        assert_eq!(d.code, "E_KAZE_RANGE");
        assert_eq!(d.rule, "H-4/range");
        assert!(d.reason.contains("got 501 meters"), "{}", d.reason);
        assert!(d.reason.contains("not clamped"), "{}", d.reason);
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let d = validate_arguments("drone", "takeoff", &[Value::Num(bad)], &arena).unwrap_err();
            assert_eq!(d.code, "E_KAZE_RANGE", "takeoff {} was not refused", bad);
        }
    }

    /// RB-08: a `--cap family=backend` grant is judged at grant time. A typo is
    /// E_CAP_INVALID immediately, and a valid live backend is refused for a
    /// family that has none rather than being answered by the simulator.
    #[test]
    fn backend_grants_are_validated_at_grant_time() {
        for backend in ["sim", "real", "mavlink"] {
            assert!(validate_grant("drone", backend).is_ok(), "--cap drone={}", backend);
        }
        assert!(validate_grant("drone", "udp:127.0.0.1:14550").is_ok());
        assert!(validate_grant("drone", "uart:/dev/ttyAMA0@921600").is_ok());
        // The grant the audit found accepted.
        let d = validate_grant("sensor", "bogus").unwrap_err();
        assert_eq!(d.code, "E_CAP_INVALID");
        assert_eq!(d.rule, "H-2/backend");
        assert!(d.reason.contains("bogus"), "{}", d.reason);
        // A device string is a drone transport, so it is judged for shape.
        assert_eq!(validate_grant("drone", "udp:no-port").unwrap_err().code, "E_CAP_INVALID");
        assert_eq!(validate_grant("drone", "bogus").unwrap_err().code, "E_CAP_INVALID");
        // No live backend exists for these families, so a live grant is refused.
        for (family, backend) in [
            ("sensor", "real"),
            ("sensor", "mavlink"),
            ("rover", "mavlink"),
            ("alert", "real"),
            ("device", "real"),
        ] {
            let d = validate_grant(family, backend).unwrap_err();
            assert_eq!(d.code, "E_CAP_INVALID", "--cap {}={}", family, backend);
            assert!(d.reason.contains("no supported live backend"), "{}", d.reason);
        }
        // Instance names (`lidar`, `fan`) resolve to their family first.
        assert_eq!(validate_grant("lidar", "sim").unwrap_err().code, "E_CAP_INVALID");
        assert_eq!(validate_grant("lidar", "bogus").unwrap_err().code, "E_CAP_INVALID");
        assert_eq!(validate_grant("fan", "sim").unwrap_err().code, "E_CAP_INVALID");
        // A bare grant does not select a backend.
        assert!(validate_grant("drone", "").is_ok());
    }
}
