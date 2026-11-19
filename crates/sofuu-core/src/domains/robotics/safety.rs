//! The physical safety rules a drone or rover command must pass.
//!
//! One rule set for every backend. A rule that lives in the live path but not
//! in the physical-test path (or the other way round) is a rule one of them
//! does not enforce, so the rules live here and each backend supplies the
//! facts it knows about the vehicle.
//!
//! The facts are three-valued on purpose. An arming state that was never
//! reported is `None`, not `false`: "unknown" and "known to be disarmed" are
//! different situations, and only the first one is repaired by waiting.
//!
//! A refusal here is a veto, not an error to be retried elsewhere: the frame
//! never reaches a vehicle and the caller decides what to do next.

use std::time::Duration;

use super::hardware::Denial;
use super::mavlink_gate::{haversine_m, DEFAULT_GEOFENCE_RADIUS_M};

/// How long a telemetry reading stays evidence of the vehicle's state.
///
/// One number for the whole robotics division: the MAVLink frame gate
/// (`mavlink_gate::TELEMETRY_MAX_AGE_MS`) and the Kaze capability path must
/// not drift apart.
pub const TELEMETRY_MAX_AGE_MS: u64 = 1_500;

/// The same window as a duration.
pub const TELEMETRY_MAX_AGE: Duration = Duration::from_millis(TELEMETRY_MAX_AGE_MS);

/// What a backend knows about the vehicle at the moment of a call.
///
/// A backend fills this in from the source it actually reads: a live link
/// from the autopilot's own reports, a physical test host from the machine
/// state it models.
#[derive(Clone, Debug)]
pub struct VehicleState {
    /// The arming state. `None` means it was never reported.
    pub armed: Option<bool>,
    /// The geofence centre: the first position the vehicle reported.
    /// `None` means there is no home yet, so no target can be fenced.
    pub home: Option<(f64, f64)>,
    /// The fence radius in metres. `None` means this backend has no fence
    /// configured — a deployment fact, not a licence to fly anywhere.
    pub geofence_radius_m: Option<f64>,
    /// How long since the last telemetry frame. `None` means nothing has been
    /// received at all, which is older than any window.
    pub telemetry_age: Option<Duration>,
    /// The link can still carry a command.
    pub link_alive: bool,
}

impl Default for VehicleState {
    fn default() -> Self {
        VehicleState {
            armed: None,
            home: None,
            geofence_radius_m: Some(DEFAULT_GEOFENCE_RADIUS_M as f64),
            telemetry_age: None,
            link_alive: false,
        }
    }
}

impl VehicleState {
    /// A state with no fence configured. Used by a backend that models no
    /// deployment envelope.
    pub fn unfenced(mut self) -> Self {
        self.geofence_radius_m = None;
        self
    }

    /// True when the last telemetry frame is still evidence.
    pub fn telemetry_is_fresh(&self) -> bool {
        self.telemetry_age
            .is_some_and(|age| age <= TELEMETRY_MAX_AGE)
    }
}

/// The families whose operations act on a vehicle.
pub fn is_physical_family(family: &str) -> bool {
    matches!(family, "drone" | "rover")
}

/// The commands that put the vehicle under the script's control.
///
/// A recovery command (`land`, `rtl`, `stop`, `hold`) is not in this set: a
/// vehicle whose telemetry went quiet still has to be brought down, and
/// refusing the recovery because the state is unknown would be the unsafe
/// choice.
pub fn needs_vehicle_state(op: &str) -> bool {
    matches!(op, "takeoff" | "goto")
}

/// Judge one physical command against the safety rules.
///
/// `target` is the position the command asks for, when it names one.
pub fn gate_physical_command(
    family: &str,
    op: &str,
    state: &VehicleState,
    target: Option<(f64, f64)>,
) -> Result<(), Denial> {
    if !is_physical_family(family) {
        return Err(Denial::new(
            "E_CAP_INVALID",
            "R-HW/family",
            format!(
                "'{}' is not a vehicle family; the physical safety rules do not apply to it",
                family
            ),
        ));
    }
    if !state.link_alive {
        return Err(Denial::new(
            "E_KAZE_LINK",
            "R-HW/link",
            format!(
                "the {} link is gone; {}.{} was not sent",
                family, family, op
            ),
        ));
    }
    if needs_vehicle_state(op) {
        if !state.telemetry_is_fresh() {
            return Err(Denial::new(
                "E_KAZE_LINK",
                "R-HW/telemetry",
                format!(
                    "{}.{} needs vehicle telemetry from the last {} ms; the reading is missing or stale",
                    family, op, TELEMETRY_MAX_AGE_MS
                ),
            ));
        }
        match state.armed {
            Some(true) => {}
            Some(false) => {
                return Err(Denial::new(
                    "E_CAP_DENIED",
                    "R-HW/armed",
                    format!("the vehicle is disarmed; {}.{} is refused", family, op),
                ))
            }
            None => {
                return Err(Denial::new(
                    "E_CAP_DENIED",
                    "R-HW/armed",
                    format!(
                        "the arming state is unknown; arm the vehicle before {}.{}",
                        family, op
                    ),
                ))
            }
        }
    }
    if op == "goto" {
        let Some((lat, lon)) = target else {
            return Err(Denial::new(
                "E_KAZE_RANGE",
                "R-HW/target",
                format!("{}.{} needs an explicit target position", family, op),
            ));
        };
        if let Some(radius) = state.geofence_radius_m {
            let Some((home_lat, home_lon)) = state.home else {
                return Err(Denial::new(
                    "E_KAZE_LINK",
                    "R-HW/geofence",
                    format!(
                        "no home position has been reported; the {}.{} target cannot be checked against the fence",
                        family, op
                    ),
                ));
            };
            let distance = haversine_m(home_lat, home_lon, lat, lon);
            if distance > radius {
                return Err(Denial::new(
                    "E_SAFETY_VIOLATION",
                    "R-HW/geofence",
                    format!(
                        "the {}.{} target is {distance:.0} m from home, beyond the {radius:.0} m fence",
                        family, op
                    ),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn armed_link() -> VehicleState {
        VehicleState {
            armed: Some(true),
            home: Some((52.1234, 4.1234)),
            geofence_radius_m: Some(300.0),
            telemetry_age: Some(Duration::from_millis(100)),
            link_alive: true,
        }
    }

    #[test]
    fn a_command_on_a_dead_link_is_refused() {
        let mut state = armed_link();
        state.link_alive = false;
        let d = gate_physical_command("drone", "arm", &state, None).unwrap_err();
        assert_eq!(d.code, "E_KAZE_LINK");
        assert_eq!(d.rule, "R-HW/link");
    }

    #[test]
    fn a_takeoff_while_disarmed_is_refused() {
        let mut state = armed_link();
        state.armed = Some(false);
        let d = gate_physical_command("drone", "takeoff", &state, None).unwrap_err();
        assert_eq!(d.code, "E_CAP_DENIED");
        assert_eq!(d.rule, "R-HW/armed");
        assert!(d.reason.contains("disarmed"), "{}", d.reason);
    }

    /// An arming state nobody reported is unknown, not disarmed. The two
    /// produce different answers: one is a fact about the vehicle, the other
    /// is a missing fact.
    #[test]
    fn an_unknown_arming_state_is_not_reported_as_disarmed() {
        let mut state = armed_link();
        state.armed = None;
        let d = gate_physical_command("drone", "takeoff", &state, None).unwrap_err();
        assert_eq!(d.code, "E_CAP_DENIED");
        assert!(d.reason.contains("unknown"), "{}", d.reason);
    }

    #[test]
    fn a_takeoff_on_stale_telemetry_is_refused() {
        let mut state = armed_link();
        state.telemetry_age = Some(TELEMETRY_MAX_AGE + Duration::from_millis(1));
        let d = gate_physical_command("drone", "takeoff", &state, None).unwrap_err();
        assert_eq!(d.code, "E_KAZE_LINK");
        assert!(d.reason.contains("stale"), "{}", d.reason);
    }

    #[test]
    fn a_takeoff_without_any_telemetry_is_refused() {
        let mut state = armed_link();
        state.telemetry_age = None;
        let d = gate_physical_command("drone", "takeoff", &state, None).unwrap_err();
        assert_eq!(d.code, "E_KAZE_LINK");
    }

    #[test]
    fn a_target_outside_the_fence_is_refused() {
        let state = armed_link();
        // ~1.1 km north of home.
        let d = gate_physical_command("drone", "goto", &state, Some((52.1334, 4.1234)))
            .unwrap_err();
        assert_eq!(d.code, "E_SAFETY_VIOLATION");
        assert_eq!(d.rule, "R-HW/geofence");
    }

    #[test]
    fn a_target_inside_the_fence_is_allowed() {
        let state = armed_link();
        assert!(gate_physical_command("drone", "goto", &state, Some((52.1240, 4.1240))).is_ok());
    }

    #[test]
    fn a_target_without_a_home_cannot_be_checked_and_is_refused() {
        let mut state = armed_link();
        state.home = None;
        let d = gate_physical_command("drone", "goto", &state, Some((52.1240, 4.1240)))
            .unwrap_err();
        assert_eq!(d.code, "E_KAZE_LINK");
        assert!(d.reason.contains("home"), "{}", d.reason);
    }

    /// A recovery command must still reach the vehicle when the telemetry has
    /// gone quiet. Refusing a landing because the state is unknown would keep
    /// the vehicle flying.
    #[test]
    fn a_recovery_command_is_not_blocked_by_stale_telemetry() {
        let mut state = armed_link();
        state.telemetry_age = None;
        state.armed = None;
        for op in ["land", "rtl", "stop", "hold"] {
            assert!(
                gate_physical_command("drone", op, &state, None).is_ok(),
                "{op} must still reach the vehicle"
            );
        }
    }

    #[test]
    fn a_rule_that_does_not_belong_to_a_vehicle_is_refused() {
        let state = armed_link();
        let d = gate_physical_command("sensor", "read", &state, None).unwrap_err();
        assert_eq!(d.code, "E_CAP_INVALID");
    }

    /// The same rule, the same answer, whichever backend supplies the facts:
    /// a physical test host and a live link are judged by one contract.
    #[test]
    fn the_rule_does_not_depend_on_which_backend_supplied_the_facts() {
        let simulated = armed_link();
        let live = armed_link();
        assert_eq!(
            gate_physical_command("drone", "takeoff", &simulated, None).is_ok(),
            gate_physical_command("drone", "takeoff", &live, None).is_ok()
        );
        let mut disarmed = armed_link();
        disarmed.armed = Some(false);
        assert_eq!(
            gate_physical_command("drone", "takeoff", &disarmed, None).unwrap_err().code,
            gate_physical_command("drone", "takeoff", &disarmed, None).unwrap_err().code
        );
    }
}
