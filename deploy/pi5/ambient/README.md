# Ambient device program (M-R3)

One small program, deployed two ways, that exercises the whole M-R3 claim on
real hardware: a compiled one-file Sofuu artifact with an embedded brain, a
first-launch seed, offline recall, and WAL recovery after a hard kill.

```
agent.js                 the device program (serve / --once / --recall / --stats)
seed.js                  writes the seed brain the artifact carries
policy.json              the R7 profile; install.sh places it at <workdir>/.sofuu/policy.json
```

## What it does

- Opens its brain: `sofuu.memory.embedded` when compiled, a file when run
  directly. A missing seed is a hard error (`E_NO_SEED`, exit 3) — never a
  silent fresh brain.
- On an empty brain, writes one `device profile:` record and flushes it.
- Reads `/sys/class/thermal/thermal_zone0/temp` through `sofuu.fs` (so through
  the R7 policy), appends one `reading soc_temp_c=…` record, flushes, and in
  serve mode repeats on `SOFUU_READING_INTERVAL_MS`.
- Loads the policy explicitly (`sofuu.policy.load()`), refuses a malformed one,
  and logs the effective `net.fetch` decision at startup.
- Never touches the network, never spawns a process, never calls a tool.

The sensor is the SoC thermal zone because it exists on the Pi and nowhere
else is invented: on a machine without it the record says `unavailable`. This
is a deployment/recovery fixture, not a perception demo — camera, IMU and
actuation are M-R4/M-R5 (proposal-only, and the hardware classes wait for the
shared R7 gate owned by the drone lane).

## Run it directly (development)

```bash
cd deploy/pi5/ambient
mkdir -p .sofuu && cp policy.json .sofuu/policy.json   # dev shape only
sofuu run agent.js --stats
sofuu run agent.js --once
sofuu run agent.js --recall seed 3
```

The `.sofuu/policy.json` copy is what `sofuu.policy.load()` discovers from the
working directory (`.sofuu/` is gitignored, so the repo keeps the plainly
named `policy.json`). Without it the app runs but logs `E_POLICY_ABSENT`: the
runtime default is allow-all, which is not a device posture.

The brain lands at `SOFUU_DEVICE_BRAIN` or `./device.brain.qtsq`
(`*.qtsq` is gitignored here). This shape needs a QTSQ-linked build; the static
musl cross build is QTSQ-free and will refuse `memory.open`.

## Build the one-file artifact (production)

On the Pi, as the **service user**, so the seed's machine key is the one the
service reads:

```bash
sudo -u sofuu -H /usr/local/bin/sofuu run \
     /usr/local/lib/sofuu/ambient/seed.js /var/lib/sofuu/seed.brain.qtsq

sudo -u sofuu -H /usr/local/bin/sofuu compile \
     /usr/local/lib/sofuu/ambient/agent.js \
     --embed-brain /var/lib/sofuu/seed.brain.qtsq \
     -o /var/lib/sofuu/ambient
```

On first launch the artifact materialises the brain at
`SOFUU_EMBEDDED_BRAIN` (or `<artifact>.brain.qtsq` beside it). If that file
already exists it is **reused, never overwritten** — that is the update story:
replace the artifact, keep the memories.

A brain key is machine-local (`$HOME/.sofuu/brain.key`). Building the artifact
as one user and running it as another produces a seed the service cannot open.
Either build as the service user (above) or give both the same
`SOFUU_BRAIN_KEY` (install.sh `--brain-key`).

## What to check

`scripts/pi5/verify_deploy.sh` automates all of it:

1. the service is active and enabled, with the expected revision;
2. `sofuu brain verify` is clean before and after a `kill -9`;
3. `--stats` does not rewrite the brain (seed non-overwrite);
4. `--recall device` returns the seeded profile offline;
5. the process holds no TCP/UDP sockets while serving;
6. an opt-in write test proves records grow and survive a restart.
