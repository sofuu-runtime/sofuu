// tests/chat_secrets_guard_driver.js — js-5 (AUDIT-2026-09-07) RED/GREEN
// driver, run by tests/chat_secrets_guard_e2e.sh (which provides HOME +
// config + cwd = project AND creates the canary fixture files). Drives
// the JS chat engine (sofuu.chat.submit — the desktop driver).
//
// js-5: read_file refuses credential stores (assertNotSensitive), but
// grep/glob — both AUTO-approved — walk and print anything: an explicit
// path=".sofuu" walks INSIDE the config dir (SKIP_DIRS only filters child
// dirs during a walk), ".ssh/" is not filtered at all, and "*.pem" files
// are read and printed wherever they live. A routine "find where X is
// used" scan therefore exfiltrates API keys / key material into the
// transcript that goes to the provider.
//
// Scenario (one server, branched on REQUEST CONTENT markers; the engine
// retains transcripts between turns, so leg selection scans user markers
// BACKWARDS to find the most recent submit):
//   leg 1  grep {pattern ".*", path ".sofuu"}   → config canary (key)
//   leg 2  glob {pattern "**/*", path "."}      → .ssh + .pem PATHS
//   leg 3  grep {pattern PEM-CANARY-BLOCK}      → .pem CONTENT
//   leg 4  read_file notes.txt                  → control (must keep working)
// The mock scans EVERY tool-role message server-side (Mimosa-XSS-safe)
// for the fake canaries and replies with constant sentences only.

const PORT = parseInt(process.argv[process.argv.length - 1] || "18993", 10);
const PROTECTED = "SECRETS-CANARY-PROTECTED: nothing sensitive reached me";
const LEAKED = "SECRETS-CANARY-LEAKED: credentials reached the model";
const KEY_CANARY = "sk-FAKE-KEYCANARY-9911";
const CFG_LINE = "KEYCANARY-VALUE=" + KEY_CANARY;
const PEM_CANARY = "FAKE-PEM-CANARY-BLOCK";
/* Leg-3 grep pattern: matches the pem fixture lines (plain substring, no
 * regex metachars) but is NOT the canary itself — grep's "no matches"
 * result echoes the pattern, and the pattern is model-supplied input
 * already in the transcript, so echoing it is not a file-content leak. */
const PEM_PATTERN = "PEM-CANARY-BLOCK";
const SSH_MARK = ".ssh/id_rsa";
const PEM_MARK = ".pem";
const NOTES_OK = "SECRETS-NOTES-PLAIN-OK";

let ok = 0, bad = 0;
function check(label, cond) {
  if (cond) { console.log("PASS " + label); ok++; }
  else { console.error("FAIL " + label); bad++; }
}

function sse(res, frames) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  for (const f of frames) res.write("data: " + f + "\n\n");
  res.end();
}

function contentReply(res, text) {
  sse(res, [
    JSON.stringify({ choices: [{ delta: { content: text } }] }),
    JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 8 } }),
    "[DONE]",
  ]);
}

function toolReply(res, id, name, args) {
  const tcs = [{
    index: 0, id: id, type: "function",
    function: { name: name, arguments: JSON.stringify(args) },
  }];
  sse(res, [
    JSON.stringify({ choices: [{ delta: { tool_calls: tcs } }] }),
    JSON.stringify({ choices: [{ delta: {} }], usage: { prompt_tokens: 40, completion_tokens: 8 } }),
    "[DONE]",
  ]);
}

const LEGS = [
  { mark: "SECRETS-CANARY-LEG1", name: "grep",      args: { pattern: ".*", path: ".sofuu" } },
  { mark: "SECRETS-CANARY-LEG2", name: "glob",      args: { pattern: "**/*", path: "." } },
  { mark: "SECRETS-CANARY-LEG3", name: "grep",      args: { pattern: PEM_PATTERN, path: "." } },
  { mark: "SECRETS-CANARY-LEG4", name: "read_file", args: { path: "notes.txt" } },
];
const served = [0, 0, 0, 0];

let sawKeyLeak = false, sawPemLeak = false, sawSshPath = false, sawPemPath = false;
let sawNotes = false;

const server = sofuu.createServer(function (req, res) {
  let parsed = {};
  try { parsed = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = parsed.messages || [];

  /* Server-side assertion (never echo request bytes): inspect every tool
   * result the engine wants to send upstream for the fake canaries. */
  for (const m of msgs) {
    if (m.role !== "tool") continue;
    const t = String(m.content || "");
    if (t.indexOf(KEY_CANARY) >= 0 || t.indexOf(CFG_LINE) >= 0) sawKeyLeak = true;
    if (t.indexOf(PEM_CANARY) >= 0) sawPemLeak = true;
    if (t.indexOf(SSH_MARK) >= 0) sawSshPath = true;
    if (t.indexOf(PEM_MARK) >= 0) sawPemPath = true;
    if (t.indexOf(NOTES_OK) >= 0) sawNotes = true;
  }

  /* Most recent leg: scan user messages from the END for a leg marker.
   * Footers/recalls carry no marker, and transcript retention means older
   * markers are still present but never the most recent one. */
  let leg = -1;
  for (let i = msgs.length - 1; i >= 0 && leg < 0; i--) {
    if (msgs[i].role !== "user") continue;
    const t = String(msgs[i].content || "");
    for (let L = 0; L < LEGS.length; L++) {
      if (t.indexOf(LEGS[L].mark) >= 0) { leg = L; break; }
    }
  }
  if (leg < 0) { contentReply(res, "SECRETS-CANARY-IDLE"); return; }

  if (served[leg] === 0) {
    served[leg]++;
    console.log("MOCK-SECRETS: serving leg " + (leg + 1) + " " + LEGS[leg].name);
    toolReply(res, "call_sg_" + (leg + 1), LEGS[leg].name, LEGS[leg].args);
  } else {
    console.log("MOCK-SECRETS: settle leg " + (leg + 1) +
                " keyLeak=" + sawKeyLeak + " pemLeak=" + sawPemLeak +
                " sshPath=" + sawSshPath + " pemPath=" + sawPemPath);
    contentReply(res, (sawKeyLeak || sawPemLeak || sawSshPath || sawPemPath) ? LEAKED : PROTECTED);
  }
});

async function main() {
  server.listen(PORT, "127.0.0.1");
  console.log("MOCK-SECRETS-READY " + PORT);

  const proj = process.env.SOFUU_PROJECT || process.cwd();

  const init = sofuu.chat.init({
    host: 'st', project: proj,
    /* full: tools auto-pass (no approvals needed for this driver). */
    permissionProfile: 'full',
  });
  if (!init.ok) { console.error("FAIL harness: chat.init failed " + JSON.stringify(init)); process.exit(1); }

  const r1 = await sofuu.chat.submit("SECRETS-CANARY-LEG1 grep the config dir", {});
  check("leg 1 grep .sofuu done", !!(r1 && r1.ok === true));
  const r2 = await sofuu.chat.submit("SECRETS-CANARY-LEG2 glob the whole tree", {});
  check("leg 2 glob tree done", !!(r2 && r2.ok === true));
  const r3 = await sofuu.chat.submit("SECRETS-CANARY-LEG3 grep for the pem marker", {});
  check("leg 3 grep pem done", !!(r3 && r3.ok === true));
  const r4 = await sofuu.chat.submit("SECRETS-CANARY-LEG4 read the notes file", {});
  check("leg 4 read notes done", !!(r4 && r4.ok === true));

  check("mock served all four legs",
        served[0] === 1 && served[1] === 1 && served[2] === 1 && served[3] === 1);
  check("notes content reached the mock (read_file works)", sawNotes === true);
  check("fake-key canary never reached the LLM", sawKeyLeak === false);
  check("pem canary never reached the LLM", sawPemLeak === false);
  check("ssh key path never reached the LLM", sawSshPath === false);
  check("pem path never reached the LLM", sawPemPath === false);

  console.log(bad === 0 ? "\nSECRETS-GUARD DRIVER: ALL PASSED"
                        : "\nSECRETS-GUARD DRIVER: " + bad + " FAILED");
  process.exit(bad === 0 ? 0 : 1);
}

main().catch(e => { console.error("FAIL harness: " + (e && e.stack || e)); process.exit(1); });
