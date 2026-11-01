# Ground rover — policy, missions, pre-flight (M-R2)

The rover shares the drone plan's codec, R7 hardware gate and mission runner.
What is ground-specific is **data**: bounds, the send allowlist, and the
mission files. That is the doctrine the plan settled on — one MAVLink
architecture, one gate, two vehicle profiles.

This directory is the ground half that can exist without a second code path:

```
rover.policy.json                  ground hardware policy (SITL-first)
mission.patrol.json                the v1 demo: arm → patrol loop → hold → RTL
mission.geofence-violation.json    refusal case: target outside 50 m
mission.speed-violation.json       refusal case: 8 m/s > maxSpeedMs 3
```

The runner today is `sofuu drone mission` — the mission path is
vehicle-agnostic. `sofuu rover` (and `sofuu.rover.mission`) is the ground
namespace; it is not shipped yet and lands with the rover JS surface (see
*Not here yet* below).

## Pre-flight only (no SITL, no hardware)

```bash
scripts/rover/preflight_cases.sh --binary ./sofuu --kaze
```

Expected:

```
PASS patrol mission passes pre-flight
PASS geofence violation refused before anything was sent
PASS speed violation refused before anything was sent
PASS Kaze rover obstacle loop runs on the sim backend   (with --kaze)
```

`--dry-run` stops after pre-flight, before a link is opened, so this is safe
on a machine with no vehicle and no simulator. It exercises the real gate:
the same `HardwarePolicy` and `gate_send` the flight path uses.

## Bounds

| Field | Value | Why |
|---|---:|---|
| `maxSpeedMs` | 3 | Ground-speed envelope; the plan's rover bound is single digits |
| `maxAltRelM` | 1 | A rover has no altitude. The bound is a sentinel: `goto` passes `alt: 0` |
| `maxYawRateDs` | 90 | Body yaw-rate ceiling for a skid-steer chassis |
| `geofenceRadiusM` | 50 | Demo-scale fence; the FC's own fence is the real one |

Tune these in `rover.policy.json` per deployment. The gate is
**reject-never-clamp**: a value above a bound is refused with
`E_SAFETY_VIOLATION`, not scaled down.

## SITL loop (Mac ↔ Pi 5 over Wi-Fi)

The FC side is ArduPilot Rover SITL. It speaks the same MAVLink v2 messages
as the drone SITL, so the runner is unchanged.

**On the Mac** (SITL is the flight controller):

```bash
sim_vehicle.py -v Rover --console --map
# SITL listens for commands on udp:14550 and sends telemetry back to the
# address that spoke first.
```

**On the Pi 5** (Sofuu decides; the FC stabilises and fails safe):

```bash
# 1. Add the Mac's address to the policy's devices.allow, e.g.
#      {"path": "udpout:192.168.1.10:14550", "class": "mavlink-udp"}
#    (the shipped file only carries loopback + 0.0.0.0 bind entries)

# 2. Prove the mission before it moves:
./sofuu drone mission deploy/rover/mission.patrol.json \
    --policy deploy/rover/rover.policy.json \
    --device udpout:192.168.1.10:14550 --dry-run

# 3. Fly it:
./sofuu drone mission deploy/rover/mission.patrol.json \
    --policy deploy/rover/rover.policy.json \
    --device udpout:192.168.1.10:14550
```

`udpout:` is the right scheme when the Pi must speak first (the SITL box
does not know the Pi's address). The link emits a 1 Hz HEARTBEAT in every
mode; the FC's GCS failsafe uses its absence as the dead-man switch.

If you prefer the Mac to push telemetry at the Pi, run SITL with
`--out=udp:<pi-ip>:14550`, bind the Pi to `udp:0.0.0.0:14550`, and keep that
entry in `devices.allow`.

## What the refusal cases prove

- `mission.geofence-violation.json` targets a point ~100 m east of home while
  the policy says 50 m. Pre-flight answers `E_SAFETY_VIOLATION` /
  `R-HW/bounds/geofenceRadiusM`; nothing is opened, nothing is sent.
- `mission.speed-violation.json` commands 8 m/s against `maxSpeedMs: 3`.
  Pre-flight answers `E_SAFETY_VIOLATION` / `R-HW/bounds/maxSpeedMs`.

Both are the plan's §5.2 "geofence-rejection case", expanded to speed.

## Not here yet (tracked in the plan, §8.2)

- **`sofuu.rover` namespace** — `sofuu.drone.profile/validate/mission` works
  for any `vehicle` string, so a rover mission is valid today; the dedicated
  namespace is a thin alias and lands with the rover JS surface.
- **Completion-aware `turn right by 90 degrees`** — needs a yaw step plus a
  heading wait in the mission runner (±2°). The Kaze sim backend already
  expresses it (`examples/kaze/rover.kaze`); the MAVLink runner does not yet.
- **Sensor-stutter fail-closed** (`E_SENSOR_TIMEOUT` > 500 ms) — needs a real
  lidar backend; the sim backend has the sensor, the vehicle backend does not.
- **`PACKS.rover`** — the agent toolpack; `PACKS.drone` registers through
  `sofuu.drone.registerTools`, which is the pattern to copy.
- **PreArm `STATUSTEXT` extraction** — already implemented in the shared link
  (`rt/mavlink_link.rs` correlates recent STATUSTEXTs with a failed arm), so a
  rover inherits it unchanged.
- **Filmed SITL demo** — owner-run, after the above.

## Safety posture

Failsafes live in firmware, not in prompts: ArduPilot Rover's GCS failsafe
stops/holds when Sofuu's heartbeat stops, its geofence and E-stop are FC-side,
and nothing in this directory can widen the gate. The agent can only emit
frames the policy's `hardware` section allows; a skill is a pre-vetted shape,
not a bypass.
