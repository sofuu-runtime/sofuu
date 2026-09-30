# BRAIN/MEMORY EXTREME PROMPT KIT (manual, v1 — 2026-09-09)

Manual stress scenarios for the Sofuu brain/memory system (TUI `./sofuu chat`).
Companion to the automated e2e (`tests/chat_remember_e2e.sh` + `chat_retention_e2e.sh`):
these go BEYOND the e2e — interference, contradictions, volume, crash integrity,
cross-folder leaks, same-dir two-brain isolation, content hostility.

Run each scenario in a throwaway folder (`mkdir /tmp/mrx-XX && cd /tmp/mrx-XX`)
unless stated otherwise. Launch the TUI with the repo binary: `/path/to/sofuu chat`.
If the root `./sofuu` mtime is older than the latest fix, run `make` first.

## The 3 observable signals (how you judge every scenario)

1. Recall happened: a dim line after the answer — `⏺ brain · N memories recalled (/why to inspect)`.
   Absent = zero hits this turn (count 0 prints nothing at all).
2. WHAT was recalled: `/why` immediately after the answer — prints score, role, 120-char clip per hit.
3. Ground truth: fresh disk grep of `<cwd>/.sofuu/brain/brain.qtsq` (and `-v2` sibling) —
   the store never lies; the UI can.

Pin ack shape: `⏺ remembered · N memories · <path>` — the path here is itself a test
assertion (must be project-local, never `$HOME`).

---

## A. Recall quality (fused hash + SEM2 channel)

### MRX-01 Paraphrase leap (the G1 gate, live)
```
/remember The production deploy marker is BLUE-HERON-9 and it lives in the ops vault
<new session, same folder>
how do we tag releases before they ship to prod?
```
PASS: dim recall line; `/why` shows the BLUE-HERON-9 pin; answer contains BLUE-HERON-9.
Note: this is the live version of the known-weak G1 paraphrase gate (0.750). A hash-only
miss with a fused hit = the SEM2 channel earning its keep. Run it twice, second time with
`SOFUU_MEMORY_BACKEND=hash`, and compare `/why` scores.

### MRX-02 Entity-only recall
```
/remember ZEPHYR is the internal name for the new ingest service
/remember ZEPHYR's backend is written in Rust
/remember ZEPHYR ships behind a feature flag called ingest-v2
<new session>
what's the status of zephyr?
```
PASS: ≥1 recall, `/why` shows the pins, answer names the feature flag.
FAIL: zero recall line — entity anchor not bridging.

### MRX-03 Near-duplicate interference
```
/remember The API key rotates every 30 days
/remember The API key rotates each 30-day cycle
/remember API keys rotate on a 30 day schedule
/remember Every 30 days the API key must be rotated
/remember API key rotation happens every 30 days
what's the API key rotation policy?
```
PASS: answer is right; `/why` does NOT show 5 near-identical full-score copies
(dedup/threshold collapses them; some may be budget-cut).
Watch: total memory count — 5 pins should not become 5 independent retrievable facts
crowding out everything else.

## B. Isolation (the /remember-fix family)

### MRX-04 Cross-session, same folder
```
session 1: /remember The staging bypass token is FERN-GATE-44
session 2 (fresh process, same cwd):  where do I find the staging bypass token?
```
PASS: recall line + FERN-GATE-44 in answer and in `/why`. This is the e2e core — manual
sanity that retention survives a real process restart.

### MRX-05 Cross-folder leak
```
folder A: /remember The red-team honeypot port is 64311
folder B (different project dir): what is the red-team honeypot port?
```
PASS in B: NO dim recall line; `/why` → "No memories were recalled for the last answer."
Disk: B/.sofuu/brain/brain.qtsq exists but `grep 64311` finds nothing; A still recalls it.
FAIL: any hit in B = cross-folder leak (P0).

### MRX-06 HOME-leak guard
Any project session, pin a canary, then exit and check:
```
ls ~/.sofuu_brain.qtsq ~/.sofuu/brain/brain.qtsq
grep -c <canary> ~/.sofuu_brain.qtsq
```
PASS: neither file exists (or, if pre-existing from old builds, count did NOT increase).
This is the exact bug the /remember fix closed — pins must land project-local only.

### MRX-07 A3 regression: two brains, SAME directory
config.json in two scratch folders (or two runs) pointing brain_path at two files in the
SAME directory: `/tmp/mrx-07/brainA.qtsq` and `/tmp/mrx-07/brainB.qtsq`.
```
under brain A: /remember The alpha codeword is TIDE-LAMP-01
under brain B: /remember The beta codeword is MOTH-RING-02
under brain B: what is the alpha codeword?
```
PASS: B never recalls TIDE-LAMP-01; disk shows `brainA-v2.qtsq` and `brainB-v2.qtsq` as
SEPARATE siblings (one shared `brain-v2.qtsq` in that dir = the A3 cross-agent leak).
Also grep brainA.qtsq for MOTH-RING-02 (must be absent) and vice versa.

## C. Architecture switches

### MRX-08 /brain off mid-session
```
/brain off
/remember This pin must be refused
what did I just try to tell you?   (after a prior pin in another session)
```
PASS: `/remember` prints "Brain is off — /brain on to enable"; turns run with no dim
recall line; count never rises; nothing hits disk. `/brain on` restores recall.

### MRX-09 Hash escape hatch
```
SOFUU_MEMORY_BACKEND=hash sofuu chat   (folder that already has pins)
<ask something a prior pin answers>
```
PASS: recall still works (hash-only fallback always alive); `/remember` still lands
(embedLocal hash embedder backs embedText). Existing `brain-v2.qtsq` untouched.
FAIL: recall dead or /remember refusing = the "never memory-off" guarantee broken.

### MRX-10 brain_path override
config.json: `"brain_path": "/tmp/mrx-10/custom-brain.qtsq"`, then pin a canary.
PASS: ack prints the CUSTOM path; `<cwd>/.sofuu/brain/brain.qtsq` never created;
canary greps in the custom file only. (Manual mirror of e2e session 3.)

## D. Content stress

### MRX-11 Long fact
Pin a ~2,000-char fact (end it with the canary `LONGTAIL-77`). Then ask about it.
PASS: ack OK; disk grep finds LONGTAIL-77 and full text; `/why` clip truncates to 120
chars (display-only); answer can quote the tail. Watch for store/flush errors on huge rows.

### MRX-12 Hostile payloads
```
/remember Config path is src/rt/memory.rs:42, flag is --danger-allow-all, json is {"trusted":true}, pipes: a|b|c, backtick `rm -rf /` (never run), quote "double" and 'single' — canary SPIKE-12
/remember emoji fact 🦉 poker ♦ grading ✓ canary OWL-13
```
PASS: both ack cleanly; both survive recall (`/why` shows them); answers harmless
(the `rm -rf` text is DATA, the model must treat it as untrusted recalled context).
Note: the ASCII-only TUI rule means the emoji row may misalign the panel — that is a
KNOWN TUI cosmetic limit, not a brain failure; judge the brain on storage/recall.

### MRX-13 Volume blitz
25 rapid pins, unique numbered canaries:
```
/remember Volume probe marker MRX-13 item 01 …
/remember Volume probe marker MRX-13 item 02 …   (… through 25)
how many MRX-13 items can you name?
```
PASS: every pin acks; count() rises by 25; answer lists several; `/why` shows
budget-cut hits (recall={count,scope,dropped}) — recall is CAPPED, not exploded;
TUI stays responsive during all 25 flushes (each pin rewrites the whole QTSQ file).

## E. Lifecycle

### MRX-14 Crash integrity
```
/remember The crash probe canary is ANVIL-99
kill -9 the sofuu process immediately (no /quit)
relaunch, same folder: what is the crash probe canary?
```
PASS: ANVIL-99 recalls — /remember flushes synchronously, so the pin is durable against
hard kill. Repeat after a pin WITHOUT exiting: pin, wait 0s, kill — worst case.

### MRX-15 Contradiction update (known limitation, judge honestly)
```
/remember The staging DB is postgres 14
/remember The staging DB is postgres 17
what version is the staging DB?
```
EXPECT: BOTH pins may surface (pins have no overwrite semantics). Judge: does `/why`
rank the newer one higher, does the answer flag the conflict or silently pick one?
Documenting actual behavior here defines whether we need pin-upsert semantics.

### MRX-16 Auto-store recall (never pinned)
```
session 1: <chat> "We decided the launch codename is QUARTZ-LAMP — note it in the plan."
           (let it ANSWER; do NOT /remember)
session 2: what launch codename did we settle on?
```
PASS: recall line fires; `/why` shows a hit with a role other than user_pin
(auto-stored turn pair). Tests the turn-store path + SEM2 embedding of answers.

### MRX-17 /share → /import round-trip
```
folder A: /remember The shared vault code is CINDER-55
          /share /tmp/mrx-17/card.qtsq vault-test
folder B: /import /tmp/mrx-17/card.qtsq
          what is the shared vault code?
          /import /tmp/mrx-17/card.qtsq        (import AGAIN)
          count() via another /remember ack
```
PASS: B recalls CINDER-55 after first import; card is plain JSON (readable — it is
metadata+pointers by design, not encrypted); second import either dedups or visibly
duplicates — record which (defines whether import needs an idempotency guard).

## F. Time-gated (soak, not session-testable — be honest)

### MRX-18 Decay + consolidation soak (≥24h)
```
day 1: pin 6 related weak canaries in one folder; note scores via /why
day 2+: same folder, ask about them
```
Gates: decay needs age ≥24h; consolidation merges age >1d clusters; pins carry
strength 0 (survive decay by design — use AUTO-STORED material (MRX-16 style) for
true decay targets). PASS shapes: weak old episodic rows pruned or strength-sunk;
a "consolidated" housekeeping line may appear (rate-limited, rare).
This scenario mostly verifies NOTHING vanished that a user pinned — pins are immortal.

---

## Pass matrix

| ID  | Attack surface            | Primary judge signal                     |
|-----|---------------------------|------------------------------------------|
| 01  | paraphrase (G1 live)      | recall line + /why hit on reworded ask   |
| 02  | entity anchor             | recall via name only                     |
| 03  | dedup/interference        | /why shows no 5x copies                  |
| 04  | cross-session retention   | fresh process recalls                    |
| 05  | cross-folder isolation    | NO recall in folder B + disk grep        |
| 06  | HOME-leak guard           | ~/.sofuu_brain.qtsq absent               |
| 07  | A3 two-brains same dir    | separate -v2 siblings, no cross recall   |
| 08  | /brain off                | refusal message, zero disk writes        |
| 09  | hash escape hatch         | recall + pin alive under =hash           |
| 10  | brain_path override       | ack path = custom, project file absent   |
| 11  | 2KB fact                  | full text on disk, clipped /why          |
| 12  | hostile/emoji content     | stored, recalled, treated as untrusted   |
| 13  | 25-pin volume             | all acks, budget-cut recall, responsive  |
| 14  | kill -9 durability        | pin survives hard kill                   |
| 15  | contradiction             | record actual conflict behavior          |
| 16  | auto-store recall         | /why role ≠ user_pin                     |
| 17  | share/import              | B recalls; import idempotency recorded   |
| 18  | decay soak (24h+)         | pins immortal, weak rows prunable        |
