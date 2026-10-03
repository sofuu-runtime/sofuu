/* tools_diff_test.js — TUI edit/write diff regression driver.
 *
 * edit_file/write_file must stash a bounded unified diff for the TUI
 * (globalThis.__sofuu_last_diff) while leaving their return strings
 * BYTE-IDENTICAL (the model reads them; other tests parse them).
 *
 * Invoked via tools_diff_e2e.sh, which cds into a scratch project dir
 * first — this driver uses cwd-relative bare filenames only, and removes
 * every fixture it creates.
 */
const results = { passed: 0, failed: 0 };
function assert(label, cond, detail) {
  if (cond) { results.passed++; console.log("PASS " + label); }
  else { results.failed++; console.log("FAIL " + label + (detail ? "  [" + detail + "]" : "")); }
}
function lastDiff() { return globalThis.__sofuu_last_diff || null; }

/* Minimal OpenAI-wire SSE mock: first request per agent issues one
 * edit_file call, every later request answers in text (so neither agent
 * can spin). The two agents are told apart by their system marker. */
let mockCalls = {};
function diffMockHandler(req, res) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  let body = {};
  try {
    const b = req.body;
    body = typeof b === "string" ? JSON.parse(b || "{}") : (b || {});
  } catch (e) { body = {}; }
  const blob = JSON.stringify(body);
  const who = blob.indexOf("AGENT=diffagent-err") >= 0 ? "err" : "ok";
  mockCalls[who] = (mockCalls[who] || 0) + 1;
  const chunk = (delta, finish) => ({
    id: "d", object: "chat.completion.chunk", model: "mock",
    choices: [{ index: 0, delta: delta, finish_reason: finish }],
  });
  if (mockCalls[who] === 1) {
    const args = who === "err"
      ? { path: "diff_agent_target.txt", old_string: "no-such-line", new_string: "q" }
      : { path: "diff_agent_target.txt", old_string: "alpha", new_string: "beta" };
    res.write("data: " + JSON.stringify(chunk(
      { tool_calls: [{ index: 0, id: "c1", type: "function",
        function: { name: "edit_file", arguments: JSON.stringify(args) } }] }, null)) + "\n\n");
    res.write("data: " + JSON.stringify(chunk({}, "tool_calls")) + "\n\n");
  } else {
    res.write("data: " + JSON.stringify(chunk({ content: "AGENT-DONE" }, null)) + "\n\n");
    res.write("data: " + JSON.stringify(chunk({}, "stop")) + "\n\n");
  }
  res.write("data: [DONE]\n\n");
  res.end();
}

async function agentDiffChecks(files) {
  let server = null, port = 0;
  const base = 29300 + (Date.now() % 500);
  for (let i = 0; i < 8 && !server; i++) {
    try {
      server = sofuu.createServer(diffMockHandler);
      server.listen(base + i * 37, "127.0.0.1");
      port = base + i * 37;
    } catch (e) { server = null; }
  }
  if (!server) { assert("agent: mock server bound", false); return; }
  const BASE = "http://127.0.0.1:" + port + "/v1/chat/completions";
  const prov = { provider: "openai", model: "mock", api_key: "x", base_url: BASE };

  await sofuu.fs.writeFile("diff_agent_target.txt", "one alpha two\n");
  files.push("diff_agent_target.txt");
  mockCalls = {};
  sofuu.agent.define({
    name: "diffagent", system: "AGENT=diffagent-ok grind the edit",
    tools: ["code"], memory: "off", rlm: "off",
    budget: { maxSteps: 6, maxContinuations: 0 },
    provider: prov.provider, model: prov.model, api_key: prov.api_key, base_url: prov.base_url,
  });
  const r = await sofuu.agent.run("diffagent", "do the edit", {});
  const evs = (r.trace || []).filter((e) => e.kind === "tool_result" &&
    e.payload && e.payload.name === "edit_file");
  assert("agent: one edit_file result event", evs.length === 1, "got " + evs.length);
  const p = evs.length ? evs[0].payload : {};
  assert("agent: event carries the unified diff",
         typeof p.diff === "string" && p.diff.indexOf("-one alpha two") >= 0 &&
         p.diff.indexOf("+one beta two") >= 0 && p.diff.indexOf("@@") >= 0,
         JSON.stringify((p.diff || "").slice(0, 80)));
  assert("agent: event result still reports the replacement",
         String(p.result || "").indexOf("1 replacement") >= 0);
  assert("agent: stash consumed, not left behind", lastDiff() === null);

  /* Error path: the event carries the error and NO diff (the red ✗ line
   * renders from p.error; a stale diff must never ride along). */
  mockCalls = {};
  sofuu.agent.define({
    name: "diffagent-err", system: "AGENT=diffagent-err grind the failing edit",
    tools: ["code"], memory: "off", rlm: "off",
    budget: { maxSteps: 6, maxContinuations: 0 },
    provider: prov.provider, model: prov.model, api_key: prov.api_key, base_url: prov.base_url,
  });
  const r2 = await sofuu.agent.run("diffagent-err", "do the failing edit", {});
  const evs2 = (r2.trace || []).filter((e) => e.kind === "tool_result" &&
    e.payload && e.payload.name === "edit_file");
  assert("agent error: one edit_file result event", evs2.length >= 1, "got " + evs2.length);
  const p2 = evs2.length ? evs2[0].payload : {};
  assert("agent error: event carries the error text", !!p2.error,
         JSON.stringify(p2.error || "").slice(0, 60));
  assert("agent error: no diff rides a failure", !p2.diff);
}

async function main() {
  const ef = sofuu.tools.TOOLS.edit_file.execute;
  const wf = sofuu.tools.TOOLS.write_file.execute;
  const files = ["diff_e1.txt", "diff_e2.txt", "diff_e3.txt", "diff_big.txt",
                 "diff_w1.txt", "diff_w2.txt", "diff_wbig.txt", "diff_fail.txt"];
  try {
    /* (1) edit_file basic: single-line change on line 5 of 10. */
    const body1 = ["l1", "l2", "l3", "l4", "OLD", "l6", "l7", "l8", "l9", "l10"].join("\n") + "\n";
    await sofuu.fs.writeFile(files[0], body1);
    globalThis.__sofuu_last_diff = null;
    const r1 = await ef({ path: files[0], old_string: "OLD", new_string: "NEW" });
    assert("edit: return string unchanged in shape", r1 === "edited " + files[0] + ": 1 replacement (+0 chars)",
           JSON.stringify(r1));
    const d1 = lastDiff();
    assert("edit: diff stashed under the tool name", !!(d1 && d1.name === "edit_file"),
           JSON.stringify(d1 && d1.name));
    assert("edit: removed line present", d1 && d1.diff.indexOf("-OLD") >= 0);
    assert("edit: added line present", d1 && d1.diff.indexOf("+NEW") >= 0);
    assert("edit: hunk header names the line", d1 && d1.diff.indexOf("(line 5)") >= 0,
           (d1 && d1.diff.split("\n").slice(0, 3).join(" | ")) || "no diff");
    assert("edit: context lines carried", d1 && d1.diff.indexOf(" l4") >= 0 && d1.diff.indexOf(" l6") >= 0);
    assert("edit: file actually changed", (await sofuu.fs.readFile(files[0], "utf8")).indexOf("\nNEW\n") >= 0);

    /* (2) edit_file multi-line old/new. */
    await sofuu.fs.writeFile(files[1], "a\nb1\nb2\nc\n");
    const r2 = await ef({ path: files[1], old_string: "b1\nb2", new_string: "B1\nB2\nB3" });
    assert("edit multi-line: reports 1 replacement", String(r2).indexOf("1 replacement") >= 0);
    const d2 = lastDiff();
    assert("edit multi-line: both removed lines", d2 && d2.diff.indexOf("-b1") >= 0 && d2.diff.indexOf("-b2") >= 0);
    assert("edit multi-line: all added lines",
           d2 && d2.diff.indexOf("+B1") >= 0 && d2.diff.indexOf("+B2") >= 0 && d2.diff.indexOf("+B3") >= 0);

    /* (3) replace_all: first hunk shown, total noted, return keeps count. */
    await sofuu.fs.writeFile(files[2], "x\nx\nx\n");
    const r3 = await ef({ path: files[2], old_string: "x", new_string: "y", replace_all: true });
    assert("edit replace_all: reports 3 replacements", String(r3).indexOf("3 replacements") >= 0);
    const d3 = lastDiff();
    assert("edit replace_all: notes the total", d3 && d3.diff.indexOf("3 total replacements; showing the first") >= 0,
           (d3 && d3.diff.split("\n").slice(-2).join(" | ")) || "no diff");

    /* (4) huge change: diff is bounded, file is not. */
    const bigOld = [];
    for (let i = 0; i < 200; i++) bigOld.push("old-line-" + i);
    await sofuu.fs.writeFile(files[3], bigOld.join("\n") + "\n");
    const bigNew = [];
    for (let i = 0; i < 200; i++) bigNew.push("new-line-" + i);
    await ef({ path: files[3], old_string: bigOld.join("\n"), new_string: bigNew.join("\n") });
    const d4 = lastDiff();
    assert("edit huge: diff truncated with a marker", d4 && d4.diff.indexOf("diff truncated") >= 0);
    assert("edit huge: diff bounded (~40 body lines)",
           d4 && d4.diff.split("\n").length <= 50, String(d4 && d4.diff.split("\n").length));
    assert("edit huge: file fully written anyway",
           (await sofuu.fs.readFile(files[3], "utf8")).indexOf("new-line-199") >= 0);

    /* (5) write_file create: preview bounded, return unchanged. */
    const wlines = [];
    for (let i = 0; i < 40; i++) wlines.push("wline-" + i);
    const r5 = await wf({ path: files[4], content: wlines.join("\n") + "\n" });
    assert("write create: return string unchanged in shape",
           r5 === "created " + files[4] + " (" + (wlines.join("\n") + "\n").length + " chars)",
           JSON.stringify(r5));
    const d5 = lastDiff();
    assert("write create: preview stashed", !!(d5 && d5.name === "write_file"));
    assert("write create: preview lines prefixed +", d5 && d5.diff.indexOf("+wline-0") >= 0);
    assert("write create: preview bounded with overflow note",
           d5 && d5.diff.indexOf("more lines") >= 0, (d5 && d5.diff.split("\n").slice(-1)[0]) || "");
    assert("write create: preview does not dump the whole file",
           d5 && d5.diff.indexOf("wline-39") < 0);

    /* (6) write_file overwrite: same shape. */
    await sofuu.fs.writeFile(files[5], "stale\n");
    const r6 = await wf({ path: files[5], content: "fresh-a\nfresh-b\n" });
    assert("write overwrite: return says overwrote", String(r6).indexOf("overwrote " + files[5]) === 0,
           JSON.stringify(r6));
    const d6 = lastDiff();
    assert("write overwrite: preview of new content", d6 && d6.diff.indexOf("+fresh-a") >= 0);

    /* (7) failed edit: throws AND leaves no fresh diff behind. */
    await sofuu.fs.writeFile(files[7], "nothing matches here\n");
    globalThis.__sofuu_last_diff = null;
    let threw = false;
    try { await ef({ path: files[7], old_string: "zzz-nope", new_string: "q" }); }
    catch (e) { threw = true; }
    assert("edit miss: throws", threw);
    assert("edit miss: no diff stashed on failure", lastDiff() === null);

    /* (8) agent loop: the tool_result EVENT carries the diff (the event
     * result is clipped to 200 chars and could never hold it), the model
     * result is untouched, and failures carry an error with no diff. */
    await agentDiffChecks(files);
  } finally {
    for (const f of files) { try { await sofuu.fs.rm(f); } catch (e) {} }
  }

  console.log("");
  console.log("=== RESULTS ===");
  console.log("Passed: " + results.passed + " | Failed: " + results.failed);
  if (results.failed > 0) { console.log("TOOLS-DIFF TEST: FAILED"); process.exit(1); }
  console.log("TOOLS-DIFF TEST: ALL PASSED");
}

main().catch((e) => {
  console.error("Test failed:", e && e.message ? e.message : e);
  process.exit(1);
});
