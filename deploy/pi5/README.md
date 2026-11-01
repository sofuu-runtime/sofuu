# Pi 5 ground/ambient deployment (M-R0 + M-R3)

This directory holds the **ground/ambient program** of
`docs/PLAN-ROBOTICS-PI5-EDGE.md`: the Pi 5 deployment baseline (M-R0) and the
compiled ambient-device demo (M-R3).

It deliberately does **not** contain `rt/mavlink.rs`, `rt/hardware.rs`, the
`hardware` policy section, or the rover skills/mission layer (M-R1/M-R2). Those
are the shared safety core specified once in
`docs/PLAN-DRONES-MAVLINK.md` and owned by the drone/MAVLink lane — the ground
plan consumes them and adds only device classes (i2c/gpio later; usb-serial is
the same code path as uart). Keeping this directory free of them is what lets
both lanes land without touching each other's files.

```
deploy/pi5/
├── README.md                  this page
├── install.sh                 install the systemd service (idempotent)
├── sofuu-agent.service.in     unit template (hardening + offline guarantee)
├── agent.env.example          /etc/sofuu/agent.env reference
└── ambient/                   the M-R3 device program + policy
scripts/pi5/
├── build_on_pi.sh             native QTSQ build shape (CONTRIBUTING workaround)
├── bench_baseline.py          M-R0 receipt: cold start, RSS, inference, thermals
└── verify_deploy.sh           M-R3 acceptance: seed, offline, WAL, restart
```

---

## 1. Build shapes (M-R0)

The plan asks to prove **both** shapes on the Pi and record which ships.

### Shape A — static musl (portable CLI, **no brain**)

Cross-built on the Mac; the Linux build currently exceeds the 5.5 MiB cap
(tracked release work, not a per-platform exemption):

```bash
# on the Mac, in the repo
make linux-arm64                       # → dist/sofuu-linux-arm64 (statically linked)
scp dist/sofuu-linux-arm64 priyanshu@10.53.147.78:~/Downloads/sofuu-linux-arm64.new
ssh priyanshu@10.53.147.78 'mv ~/Downloads/sofuu-linux-arm64{.new,} && ~/Downloads/sofuu-linux-arm64 version'
```

`scripts/cross/build_linux_arm64.sh` skips QTSQ, so this shape has **no brain
and no `sofuu brain` commands**. It is the right shape for `sofuu eval`,
`sofuu infer`, and the CLI baseline — not for the ambient device.

### Shape B — native QTSQ build (the ambient ship shape)

On the Pi, from a QTSQ checkout (the engine refuses to build without it):

```bash
# once: toolchain
sudo apt-get install -y build-essential pkg-config libssl-dev libcurl4-openssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh      # if absent
git clone <qtsq-checkout> ~/projects/qtsq

# repo checkout on the Pi, then:
bash scripts/pi5/build_on_pi.sh --qtsq-dir ~/projects/qtsq
```

The script compiles the QTSQ C translation units directly and archives them
(the upstream Makefile pulls `qtsq_metal.mm` on every target — the documented
Linux workaround in CONTRIBUTING.md), then runs `make` and `make size-check`.

Record which shape ships in the M-R0 receipt:

| Shape | Brain | Use | Size |
|---|---|---|---|
| static musl (cross) | no | CLI, inference baseline | over cap on Linux (tracked) |
| native QTSQ | yes | ambient device, brain ops | same cap, on-device number goes in the receipt |

Both are needed: the device program cannot open a brain on shape A, and shape B
cannot be cross-built for another architecture.

---

## 2. M-R0 receipt (benchmark + thermals)

Run on the Pi, **one benchmark at a time** (concurrent runs read 10.7 t/s where
idle reads 26.9), with the governor at `performance`:

```bash
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

python3 scripts/pi5/bench_baseline.py \
  --model ~/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf \
  --model ~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --soak-seconds 300 \
  --out ~/pi5-baseline.json
```

Each `--model` gets one prefill/decode pass; the thermal soak runs on the
first model (the plan's "brain" model) and samples temperature between passes,
so the receipt shows the decay curve rather than a single peak.

The receipt records, in one JSON file plus a readable table:

- **cold start** — `sofuu eval "1+1"`, p50/p95 over 30 runs (spawn included);
- **idle RSS** — KiB after boot, sampled from `/proc/<pid>/status`;
- **binary** — revision stamp, exact bytes, cap verdict;
- **decode/prefill** — parsed from the runtime's own `── prefill … · decode … ──`
  receipt line (the `--trace` JSONL stays on disk in `traces_dir` for per-token
  `forward_ns` inspection);
- **thermals** — temperature before/after each inference pass, maximum reached,
  `vcgencmd get_throttled` decoded, and the wall-clock time at which the soft
  limit first appeared.

Expected thermal shape on this board (2.8 GHz + `over_voltage_delta=25000`):
sustained 4-core inference hits the soft limit (~80–84 °C,
`throttled=0xe0008`) after roughly 70 s and then oscillates 2.55–2.8 GHz. That
window is the receipt — do not quote a single peak number as sustained.

Do not download models to fill a gap: use what is on the device
(`~/models/qwen3-0.6b/Qwen3-0.6B-Q4_K_M.gguf`,
`~/Downloads/Llama-3.2-1B-Instruct-Q8_0.gguf`). If a model is absent, the
script records the row as skipped and says so.

---

## 3. Deploy the ambient device (M-R3)

### 3.1 Install the service

```bash
# from the repo checkout on the Pi — the QTSQ build from §1 shape B.
# (The static musl binary has no codec and cannot open or embed a brain.)
sudo deploy/pi5/install.sh --binary ~/sofuu-build/sofuu-linux-arm64-qtsq
```

That creates the `sofuu` service user, installs the runtime, the program, the
policy (`/var/lib/sofuu/.sofuu/policy.json` — the unit's working directory in
both shapes), the environment file (`/etc/sofuu/agent.env`, 0600) and the unit,
then starts it.

Check it:

```bash
systemctl status sofuu-agent --no-pager
journalctl -u sofuu-agent -n 20 --no-pager   # one JSON line per event
```

The service is structurally offline: the unit sets
`RestrictAddressFamilies=AF_UNIX`, so the kernel refuses an AF_INET socket even
if a later program asks for one. The app additionally loads the R7 policy and
logs its `net.fetch` decision at startup.

### 3.2 Seed the brain and compile the one-file artifact

**As the service user** (`-H` sets `HOME=/var/lib/sofuu`). The seed is
encrypted with the machine key in `$HOME/.sofuu/brain.key`; building as one
user and running as another yields an artifact whose seed cannot open.

```bash
sudo -u sofuu -H /usr/local/bin/sofuu run \
     /usr/local/lib/sofuu/ambient/seed.js /var/lib/sofuu/seed.brain.qtsq

sudo -u sofuu -H /usr/local/bin/sofuu compile \
     /usr/local/lib/sofuu/ambient/agent.js \
     --embed-brain /var/lib/sofuu/seed.brain.qtsq \
     -o /var/lib/sofuu/ambient

sudo deploy/pi5/install.sh --binary ~/sofuu-build/sofuu-linux-arm64-qtsq \
     --artifact /var/lib/sofuu/ambient
```

First launch materialises the seed at `SOFUU_EMBEDDED_BRAIN`
(`/var/lib/sofuu/ambient.brain.qtsq`). **An existing brain file is never
overwritten** — replacing `/var/lib/sofuu/ambient` with a newer artifact keeps
every record the device has written. That is the update story; there is no
delta/update-signing pipeline yet (plan §4.6).

The development-shape store (`device.brain.qtsq`) is a different file and stays
behind; the compiled shape starts from the seed. Keeping them separate is
deliberate — it is what makes the "seed not overwritten" check meaningful.

### 3.3 Verify

```bash
sudo scripts/pi5/verify_deploy.sh \
  --brain /var/lib/sofuu/ambient.brain.qtsq \
  --artifact /var/lib/sofuu/ambient \
  --write-test
```

Checks, each printed PASS/FAIL:

1. unit enabled + active, expected revision, size vs cap;
2. `sofuu brain verify` clean;
3. `--stats` leaves the brain byte-identical (seed non-overwrite);
4. `--recall device` returns the seeded profile (offline recall);
5. no TCP/UDP sockets held by the service;
6. `kill -9` → systemd restarts → brain still verifies (WAL recovery);
7. `--write-test`: a record is appended, the count grows, and it survives the
   restart.

### 3.4 24-hour soak (plan §5.3)

Leave the service running, then:

```bash
watch -n 60 'ss -H -tunp | grep sofuu || true; \
             journalctl -u sofuu-agent --since -1h | tail -3; \
             du -sh /var/lib/sofuu'
```

What to look for: RSS flat (no growth per reading), the brain's segments
bounded by the record cap, no `E_*` lines, and `systemctl show -p NRestarts`
still 0. If a reading fails, the app logs `E_WRITE` and keeps serving.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `E_NO_SEED` (exit 3) | artifact's seed could not mount | build the artifact as the service user, or set `SOFUU_BRAIN_KEY` to the build key |
| `E_POLICY` (exit 4) | malformed `.sofuu/policy.json` | `sofuu policy check <file>` reports the line; the device refuses to run allow-all |
| `E_PERMISSION_DENIED … thermal_zone0` | `fs.read` allowlist does not name the sensor | add `/sys/class/thermal` to `fs.read` (the shipped policy does) |
| brain in `/tmp` | executable's directory not writable | set `SOFUU_EMBEDDED_BRAIN` to a persistent path and restart |
| `E_LOCKED` on a manual run | the service holds the brain (single writer) | `sudo systemctl stop sofuu-agent`, run the command, start it again |
| service restarts repeatedly | watchdog | `WatchdogSec` must stay commented out: the runtime does not call `sd_notify` yet |
| `sofuu brain` unknown command | static musl shape (no QTSQ) | use the native build (shape B) |
| `--embed-brain` → "this build has no QTSQ codec" | the installed `sofuu` is shape A | reinstall with the shape B binary (`build_on_pi.sh`) |
| size over cap | Linux builds are over today | tracked release work; `verify_deploy.sh` warns unless `--strict-size` |

---

## 5. What is deliberately absent

- **Rover / MAVLink** (M-R1/M-R2): the codec, the R7 hardware gate, `uv_udp`
  and the SITL loop are the drone lane's files; the rover adds policy profiles
  and skills on top of them once they land.
- **The `hardware` policy section**: it is landing in the shared core, and the
  ambient policy does **not** carry it yet on purpose — a section this binary's
  validator does not know fails closed (`E_POLICY`), so a policy written for a
  future build would brick a device on today's. Once the section is committed,
  a read-only device declares it with empty allowlists (plan §4.2/§4.4).
- **i2c / GPIO device classes** (M-R4): proposal-only in the plan; they need
  the shared gate's schema first.
- **Camera / IMU / vision** (M-R5): no V4L2 ingestion exists; `embedImage` is a
  3,408-parameter colour/layout projector, not perception.
- **Offline voice**: ASR/TTS are provider-backed over HTTP; nothing in-tree.
- **sd_notify / WatchdogSec**: v2. Until then `Restart=on-failure` is the
  supervisor, and on a vehicle the FC's GCS failsafe is the real dead-man.
