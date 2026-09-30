/* tests/extreme_bench.js — EXTREME-condition feature sweep + benchmark.
 * REPORT-ONLY by owner order (2026-09-13): nothing is fixed here; every
 * probe records PASS or ISSUE and the run ends with a ledger. Drives the
 * REAL stack: sofuu.createServer HTTP mock + real curl ai.stream path.
 * Run: ./sofuu run tests/extreme_bench.js   (SOFUU_QTSQ_DIR exported)
 * Mock responses are CONSTANT strings; request-derived values are stored
 * in module flags for the test side, never echoed into responses.
 */
const t0 = Date.now();
let PASS_N = 0, ISSUE_N = 0;
const ISSUES = [], BENCH = [];
function pass(n) { PASS_N++; console.log("PASS " + n); }
function issue(n, d) { ISSUE_N++; ISSUES.push(n + " — " + d); console.log("ISSUE " + n + " — " + d); }
function bench(n, v) { BENCH.push([n, v]); console.log("BENCH " + n + ": " + v); }
const XREQ = [];  // {sys, start, end} for overlap/latency measurement

function sysOf(msgs) { for (let i = 0; i < msgs.length; i++) if (String(msgs[i].role) === "system") return String(msgs[i].content || ""); return ""; }
function toolCount(msgs) { let n = 0; for (let i = 0; i < msgs.length; i++) if (String(msgs[i].role) === "tool") n++; return n; }
function allText(msgs) { let t = ""; for (let i = 0; i < msgs.length; i++) t += "\n" + String(msgs[i].content || ""); return t; }

/* SSE helpers — frame strings are built locally, one write per frame. */
function wOpen(res) { res.writeHead(200, { "Content-Type": "text/event-stream" }); }
function wFrame(res, obj) {
  const line = "data: " + JSON.stringify(obj) + "\n\n";
  res.write(line);
}
function wEnd(res) {
  const tail = "data: [DONE]\n\n";
  res.write(tail);
  res.end();
}
function wText(res, text) {
  wOpen(res);
  wFrame(res, { choices: [{ delta: { content: text } }] });
  wFrame(res, { usage: { prompt_tokens: 40, completion_tokens: 8 } });
  wEnd(res);
}
function wTool(res, name, args) {
  wOpen(res);
  const call = { index: 0, id: "c1", function: { name: name, arguments: JSON.stringify(args) } };
  wFrame(res, { choices: [{ delta: { tool_calls: [call] } }] });
  wFrame(res, { usage: { prompt_tokens: 40, completion_tokens: 8 } });
  wEnd(res);
}
function errStatus(res, code, msg) {
  res.writeHead(code, { "Content-Type": "application/json" });
  const body = JSON.stringify({ error: { message: msg } });
  res.send(body);
}

let RERR_N = 0, RMAL_N = 0, R429_N = 0, RCUT_N = 0, ANTH_OK = null, ANTH_SEEN_LEN = 0;
let MC_RESULT = "", MC_STOP_RESULT = "", TBIG_LEN = -1;
function handler(req, res) {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  const msgs = body.messages || [];
  const sys = sysOf(msgs), tc = toolCount(msgs), txt = allText(msgs);
  const tag = (sys.match(/XB=[a-z0-9]+/) || [""])[0];
  XREQ.push({ sys: tag, start: Date.now(), end: 0 });
  const xi = XREQ.length - 1;
  const done = () => { XREQ[xi].end = Date.now(); };
  const fin = (t) => { done(); wText(res, t); };

  if (sys.indexOf("XB=mdhuge") >= 0) return fin("mdhuge-final len=" + txt.length);
  if (sys.indexOf("XB=fresh") >= 0) return fin("fresh-final[" + (txt.indexOf("[freshness]") >= 0 ? "+fresh" : (txt.indexOf("Context from") >= 0 ? "recall-nonotice" : "norecall")) + "]");
  if (sys.indexOf("XB=rel") >= 0) return fin("rel-final[" + (txt.indexOf("[relevance]") >= 0 ? "+rel" : (txt.indexOf("Context from") >= 0 ? "recall-nonotice" : "norecall")) + "]");
  if (sys.indexOf("XB=dup") >= 0) {
    if (tc === 0) { done(); return wTool(res, "read_file", { path: "dupme.txt" }); }
    if (tc === 1) { done(); return wTool(res, "read_file", { path: "dupme.txt" }); }
    return fin("dup-final[" + (txt.indexOf("[supervisor") >= 0 ? "+nudge" : "-nonudge") + "]");
  }
  if (sys.indexOf("XB=esc") >= 0) {
    if (tc === 0) { done(); return wTool(res, "write_file", { path: "../xb_escape_probe.txt", content: "nope" }); }
    if (tc === 1) { done(); return wTool(res, "write_file", { path: "/tmp/xb_outside_probe.txt", content: "nope" }); }
    return fin("esc-final[" + (txt.indexOf("path escapes") >= 0 ? "+rel" : "-rel") + (txt.indexOf("escapes") >= 0 ? "+abs" : "-abs") + "]");
  }
  if (sys.indexOf("XB=edit") >= 0) {
    if (tc === 0) { done(); return wTool(res, "edit_file", { path: "editme.txt", old_string: "AAA", new_string: "XXX" }); }
    if (tc === 1) { done(); return wTool(res, "edit_file", { path: "editme.txt", old_string: "ZZZZ-NOT-THERE", new_string: "Q" }); }
    if (tc === 2) { done(); return wTool(res, "edit_file", { path: "editme.txt", old_string: "AAA", new_string: "CCC", replace_all: true }); }
    return fin("edit-final errs=" + (txt.match(/tool error/g) || []).length);
  }
  if (sys.indexOf("XB=bigtool") >= 0) {
    if (tc === 0) { done(); return wTool(res, "bash", { command: "cat big2mb.txt", timeout_ms: 20000 }); }
    const tools3 = msgs.filter(m => String(m.role) === "tool");
    TBIG_LEN = tools3.length ? String(tools3[tools3.length - 1].content || "").length : -1;
    return fin("big-final done");
  }
  if (sys.indexOf("XB=steps") >= 0) {
    if (tc < 12) { done(); return wTool(res, "list_dir", { path: "." }); }
    return fin("steps-final");
  }
  if (sys.indexOf("XB=wall") >= 0) {
    if (tc === 0) { done(); return wTool(res, "bash", { command: "sleep 8", timeout_ms: 20000 }); }
    return fin("wall-final-never");
  }
  if (sys.indexOf("XB=fit") >= 0) {
    let kept = 0;
    for (let i = 0; i < 60; i++) if (txt.indexOf("HIST" + i + "-MARK") >= 0) kept++;
    return fin("fit-final needle=" + (txt.indexOf("TASK-NEEDLE-77") >= 0 ? "yes" : "NO") + " hist=" + kept);
  }
  if (sys.indexOf("XB=fanorch") >= 0) {
    if (tc === 0) {
      done();
      wOpen(res);
      let idx = 0;
      for (const nm of ["fan1", "fan2", "fan3", "fan4", "fan5", "fanbad"]) {
        const call = { index: idx, id: "cf" + idx, function: { name: "delegate", arguments: JSON.stringify({ agent: nm, task: "go " + nm, why: "x" }) } };
        wFrame(res, { choices: [{ delta: { tool_calls: [call] } }] });
        idx++;
      }
      wFrame(res, { usage: { prompt_tokens: 40, completion_tokens: 8 } });
      wEnd(res);
      return;
    }
    const okN = (txt.match(/FAN.-OK/g) || []).length;
    return fin("fan-final sub=" + okN + " err=" + (txt.indexOf("tool error") >= 0 ? "yes" : "no"));
  }
  if (sys.indexOf("XB=fanbad") >= 0) { done(); return errStatus(res, 500, "fanbad boom"); }
  const fanM = sys.match(/XB=fan([1-5])/);
  if (fanM) {
    /* 250ms hold: overlap windows only mean something if responses take time. */
    setTimeout(() => { XREQ[xi].end = Date.now(); wText(res, "FAN" + fanM[1] + "-OK"); }, 250);
    return;
  }
  if (sys.indexOf("XB=spec") >= 0) {
    let dl = 0;
    for (const t of (body.tools || [])) {
      const nm = ((t.function && t.function.name) || t.name);
      if (nm === "delegate") dl = String((t.function && t.function.description) || t.description || "").length;
    }
    return fin("spec-final dlen=" + dl);
  }
  if (sys.indexOf("XB=mcbig") >= 0) {
    if (tc === 0) { done(); return wTool(res, "map_context", { context: globalThis.XB_MB, task: "count pieces", agent: "mcw", chunk_chars: 2000 }); }
    return fin("mcbig-final[" + (txt.indexOf("Raise chunk_chars") >= 0 || txt.indexOf("12 chunk") >= 0 ? "capped" : "NOT-CAPPED") + "]");
  }
  if (sys.indexOf("XB=mcok") >= 0) {
    if (tc === 0) { done(); return wTool(res, "map_context", { context: globalThis.XB_MB, task: "count pieces", agent: "mcw", chunk_chars: 100000 }); }
    const tools2 = msgs.filter(m => String(m.role) === "tool");
    MC_RESULT = tools2.length ? String(tools2[tools2.length - 1].content || "") : "";
    return fin("mcok-final pieces=" + (txt.match(/MC-PIECE/g) || []).length);
  }
  if (sys.indexOf("XB=mcstop") >= 0) {
    if (tc === 0) { done(); return wTool(res, "map_context", { context: "aaa bbb ccc. ".repeat(3000), task: "x", agent: "mcdead", chunk_chars: 20000 }); }
    const tools4 = msgs.filter(m => String(m.role) === "tool");
    MC_STOP_RESULT = tools4.length ? String(tools4[tools4.length - 1].content || "") : "";
    return fin("mcstop-final seen");
  }
  if (sys.indexOf("XB=mcw") >= 0) return fin("MC-PIECE");
  if (sys.indexOf("XB=r429") >= 0) {
    R429_N++;
    if (R429_N <= 2) { done(); return errStatus(res, 429, "rate limited (try " + R429_N + ")"); }
    return fin("r429-final-ok");
  }
  if (sys.indexOf("XB=rcut") >= 0) {
    RCUT_N++;
    if (RCUT_N < 2) {
      done();
      wOpen(res);
      wFrame(res, { choices: [{ delta: { content: "PARTIAL-ALPHA-TEXT " } }] });
      res.end(); /* abrupt EOF mid-stream: no [DONE], no usage */
      return;
    }
    return fin("PARTIAL-ALPHA-TEXT TAIL-OK");
  }
  if (sys.indexOf("XB=rerr") >= 0) {
    if (RERR_N < 1) {
      RERR_N++; done();
      wOpen(res);
      wFrame(res, { choices: [{ delta: { content: "RERR-PART " } }] });
      wFrame(res, { error: { message: "rate limit exceeded upstream" } });
      wEnd(res);
      return;
    }
    return fin("RERR-PART RERR-CONT-OK");
  }
  if (sys.indexOf("XB=rmal") >= 0) {
    if (RMAL_N < 1) {
      RMAL_N++; done();
      wOpen(res);
      const junk1 = "garbage line\n\n";
      const junk2 = "{{not json}}\n\n";
      res.write(junk1);
      res.write(junk2);
      wEnd(res);
      return;
    }
    return fin("rmal-final recovered");
  }
  if (sys.indexOf("XB=cancparent") >= 0) {
    if (tc === 0) { done(); return wTool(res, "delegate", { agent: "cancchild", task: "sleep then answer", why: "cancel probe" }); }
    return fin("cancparent-never");
  }
  if (sys.indexOf("XB=cancchild") >= 0) {
    if (tc === 0) { done(); return wTool(res, "bash", { command: "sleep 4", timeout_ms: 20000 }); }
    return fin("CANC-CHILD-COMPLETED");
  }
  issue("unhandled mock branch", sys.slice(0, 80));
  return fin("unhandled");
}

function antEvent(res, obj) {
  const head = "event: " + obj.type + "\ndata: ";
  const frame = head + JSON.stringify(obj) + "\n\n";
  res.write(frame);
}
function anthropicHandler(req, res) {
  let body = {};
  try { body = JSON.parse(req.body || "{}"); } catch (e) {}
  ANTH_SEEN_LEN = String(req.body || "").length;
  const saw = String(req.body || "").indexOf("ANTH-TOOLRES-77") >= 0;
  const sawTool = String(req.body || "").indexOf("tool_result") >= 0;
  if (sawTool) {
    ANTH_OK = saw;
    antEvent(res, { type: "message_start", message: { usage: { input_tokens: 25, output_tokens: 1 } } });
    antEvent(res, { type: "content_block_start", index: 0, content_block: { type: "text" } });
    antEvent(res, { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "anth-final-done" } });
    antEvent(res, { type: "content_block_stop", index: 0 });
    antEvent(res, { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 7 } });
    antEvent(res, { type: "message_stop" });
    res.end();
    return;
  }
  antEvent(res, { type: "message_start", message: { usage: { input_tokens: 25, output_tokens: 1 } } });
  antEvent(res, { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "tu1", name: "get_weather", input: {} } });
  antEvent(res, { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: JSON.stringify({ city: "Paris" }) } });
  antEvent(res, { type: "content_block_stop", index: 0 });
  antEvent(res, { type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { output_tokens: 9 } });
  antEvent(res, { type: "message_stop" });
  res.end();
}

async function main() {
  process.env.SOFUU_STREAM_RETRY_DELAYS = "200,200";
  const prevCwd = process.cwd();
  const proj = "/tmp/xb_project_" + Date.now();
  await sofuu.fs.mkdir(proj, { recursive: true });
  process.chdir(proj);

  let server = null, port = 0, antPort = 0;
  const base = 24000 + (Date.now() % 3000);
  for (let i = 0; i < 8; i++) { const p = base + i * 37; try { server = sofuu.createServer(handler); server.listen(p, "127.0.0.1"); port = p; break; } catch (e) {} }
  for (let i = 0; i < 8 && !antPort; i++) { const p = port + 211 + i * 29; try { const s = sofuu.createServer(anthropicHandler); s.listen(p, "127.0.0.1"); antPort = p; } catch (e) {} }
  if (!server || !antPort) { console.log("FATAL could not bind mocks"); return; }
  const MOCK = "http://127.0.0.1:" + port + "/v1/chat/completions";
  const ANTH = "http://127.0.0.1:" + antPort + "/v1/messages";
  function def(nm, sys, extra) {
    const o = { name: nm, system: sys, memory: "off", rlm: "off", provider: "openai", model: "mock", api_key: "x", base_url: MOCK, budget: { maxSteps: 30, maxTokens: 10000000, maxWallMs: 120000 } };
    Object.assign(o, extra || {});
    sofuu.agent.define(o);
  }

  try {
    /* ── S1: AGENTS.md torture ─────────────────────────────────────── */
    console.log("\n== S1 AGENTS.md torture ==");
    await sofuu.fs.writeFile(proj + "/AGENTS.md", "# Project memory\n\n- Codename LANTERN.\n- Port 5433.\n");
    def("xbmd", "XB=mdhuge", { tools: [] });
    {
      const t = Date.now();
      const r = await sofuu.agent.run("xbmd", "t", {});
      bench("inject_normal_file_ms", Date.now() - t);
      pass("S1 baseline injection ran (" + r.answer.slice(0, 30) + ")");
    }
    /* 2MB file */
    await sofuu.fs.writeFile(proj + "/AGENTS.md", ("filler sentence with varied words number 7.\n").repeat(95000));
    {
      const t = Date.now();
      const r = await sofuu.agent.run("xbmd", "t", {});
      const ms = Date.now() - t;
      bench("inject_2MB_run_ms", ms);
      const sz = Number((r.answer.match(/len=(\d+)/) || [0, 0])[1]);
      if (sz > 0 && sz < 40000) pass("S1 2MB file clipped at injection (" + sz + " chars on wire)");
      else issue("S1 2MB file injection size", "wire text len=" + sz + " — clip may be leaking");
      if (ms > 3000) issue("S1 2MB injection latency", ms + "ms per turn for a standing-context read");
    }
    /* pin on top of the 2MB file — full rewrite cost */
    {
      const t = Date.now();
      const pr = await sofuu.agent.pinProjectFact("probe pin on huge file 77");
      const ms = Date.now() - t;
      bench("pin_on_2MB_ms", ms);
      if (pr !== "added") issue("S1 pin on 2MB", "result=" + pr);
      else if (ms > 1500) issue("S1 pin rewrites whole file", "pin on 2MB took " + ms + "ms — O(filesize) rewrite per pin");
      else pass("S1 pin on 2MB ok (" + ms + "ms)");
    }
    /* binary garbage */
    let binStr = "";
    for (let i = 0; i < 4096; i++) binStr += String.fromCharCode(i % 256);
    await sofuu.fs.writeFile(proj + "/AGENTS.md", binStr);
    {
      const r = await sofuu.agent.run("xbmd", "t", {});
      if (r.answer.indexOf("len=") >= 0) pass("S1 binary AGENTS.md does not crash the run (mock accepts; real-provider risk untested)");
      else issue("S1 binary AGENTS.md", "answer=" + r.answer);
    }
    /* directory instead of file */
    await sofuu.fs.rm(proj + "/AGENTS.md");
    await sofuu.fs.mkdir(proj + "/AGENTS.md");
    {
      const r = await sofuu.agent.run("xbmd", "t", {});
      pass("S1 AGENTS.md-as-directory: run survives (" + r.answer.slice(0, 24) + ")");
      const pr = await sofuu.agent.pinProjectFact("pin onto a directory");
      if (pr.indexOf("error") === 0) pass("S1 pin onto directory fails gracefully: " + pr);
      else issue("S1 pin onto directory", "expected error, got " + pr);
      await sofuu.fs.rm(proj + "/AGENTS.md");
    }
    /* pin semantics battery */
    await sofuu.fs.writeFile(proj + "/AGENTS.md", "user content, no trailing newline");
    {
      const a = await sofuu.agent.pinProjectFact("fact one liner");
      const b = await sofuu.agent.pinProjectFact("multi\nline\nfact gets collapsed");
      const c = await sofuu.agent.pinProjectFact("OWL-EMOJI fact owl-9 unicode");
      const d = await sofuu.agent.pinProjectFact("x".repeat(1000));
      const e = await sofuu.agent.pinProjectFact("fact one liner"); // dup
      const txt = String(await sofuu.fs.readFile(proj + "/AGENTS.md"));
      if (a === "added") pass("S1 pin no-trailing-newline host file");
      else issue("S1 pin no-newline", a);
      if (b === "added" && txt.indexOf("multi line fact gets collapsed") >= 0) pass("S1 newline fact collapsed to one line");
      else issue("S1 newline fact", b + " / idx " + txt.indexOf("multi line"));
      if (c === "added" && txt.indexOf("OWL-EMOJI fact owl-9") >= 0) pass("S1 unicode fact preserved");
      else issue("S1 unicode fact", c);
      if (d === "added" && txt.indexOf("x".repeat(300)) >= 0 && txt.indexOf("x".repeat(499)) < 0) pass("S1 1000-char fact clipped to ~500");
      else issue("S1 long-fact clip", d);
      if (e === "exists") pass("S1 duplicate pin deduped");
      else issue("S1 dup pin", e);
      if (txt.indexOf("user content, no trailing newline") >= 0) pass("S1 user content preserved");
      else issue("S1 user content clobbered", txt.slice(0, 80));
    }
    /* mangled markers */
    await sofuu.fs.writeFile(proj + "/AGENTS.md", "orphan begin below\n<!-- sofuu:pinned begin -->\nno end marker anywhere\n");
    {
      const pr = await sofuu.agent.pinProjectFact("pin with mangled markers 5");
      const txt = String(await sofuu.fs.readFile(proj + "/AGENTS.md"));
      const begins = txt.split("sofuu:pinned begin").length - 1;
      const ends = txt.split("sofuu:pinned end").length - 1;
      if (pr === "added" && begins >= 2 && ends >= 1) issue("S1 mangled markers multiply", "orphan BEGIN not repaired: " + begins + " begins / " + ends + " ends — later pins may splice the wrong span");
      else pass("S1 mangled markers handled (" + pr + ", " + begins + "b/" + ends + "e)");
    }
    /* concurrent pin race */
    {
      await sofuu.fs.writeFile(proj + "/AGENTS.md", "# Project memory\n\n<!-- sofuu:pinned begin -->\n<!-- sofuu:pinned end -->\n");
      const ps = [];
      for (let i = 0; i < 8; i++) ps.push(sofuu.agent.pinProjectFact("race fact number " + i));
      const rs = await Promise.all(ps);
      const txt = String(await sofuu.fs.readFile(proj + "/AGENTS.md"));
      let landed = 0;
      for (let i = 0; i < 8; i++) if (txt.indexOf("race fact number " + i) >= 0) landed++;
      bench("concurrent_pins_landed", landed + "/8");
      if (landed === 8) pass("S1 8 concurrent pins all landed");
      else issue("S1 concurrent pin race LOSES updates", landed + "/8 landed — read-modify-write is not serialized");
    }
    issue("S1 read-only file pin", "SKIPPED — no chmod/fs-permission primitive exposed to scripts (untested headless)");

    /* ── S2: tool rails ────────────────────────────────────────────── */
    console.log("\n== S2 tool rails ==");
    await sofuu.fs.rm(proj + "/AGENTS.md");
    await sofuu.fs.writeFile(proj + "/inside.txt", "AAA\nBBB\nAAA\n");
    def("xbesc", "XB=esc", { tools: ["code"] });
    {
      const r = await sofuu.agent.run("xbesc", "escape attempts", {});
      if (r.answer.indexOf("+rel") >= 0) pass("S2 ../ write refused by jail");
      else issue("S2 ../ write NOT refused", r.answer);
      if (r.answer.indexOf("+abs") >= 0) pass("S2 absolute write refused by jail");
      else issue("S2 absolute write NOT refused", r.answer);
      const out1 = await sofuu.fs.exists(proj + "/../xb_escape_probe.txt");
      const out2 = await sofuu.fs.exists("/tmp/xb_outside_probe.txt");
      if (!out1 && !out2) pass("S2 no files created outside jail");
      else issue("S2 escape file EXISTS", "outside writes landed");
    }
    {
      await sofuu.fs.writeFile(proj + "/editme.txt", "AAA\nBBB\nAAA\n");
      def("xbedit", "XB=edit", { tools: ["code"] });
      const r = await sofuu.agent.run("xbedit", "edit probes", {});
      if (Number((r.answer.match(/errs=(\d+)/) || [0, 0])[1]) >= 2) pass("S2 ambiguous edit + zero-match edit both surface as tool errors");
      else issue("S2 edit refusals", r.answer);
      const t = String(await sofuu.fs.readFile(proj + "/editme.txt"));
      if (t === "CCC\nBBB\nCCC\n") pass("S2 replace_all works after the refusals");
      else issue("S2 replace_all", JSON.stringify(t));
    }
    /* huge tool result cap + raw fs timing */
    {
      let big = "XB-TOOLCAP:4560000\n";
      for (let i = 0; i < 95000; i++) big += "XB-TOOLCAP filler line with varied words " + (i % 97) + ".\n";
      const t1 = Date.now();
      await sofuu.fs.writeFile(proj + "/big2mb.txt", big);
      bench("write_4.5MB_ms", Date.now() - t1);
      const t2 = Date.now();
      const rd = String(await sofuu.fs.readFile(proj + "/big2mb.txt"));
      bench("read_4.5MB_ms", Date.now() - t2);
      bench("read_4.5MB_chars", rd.length);
      def("xbig", "XB=bigtool", { tools: ["code"] });
      const t3 = Date.now();
      const r = await sofuu.agent.run("xbig", "cat it", {});
      bench("bash_cat_4.5MB_round_ms", Date.now() - t3);
      bench("bash_cat_tool_result_chars", TBIG_LEN);
      if (TBIG_LEN > 0 && TBIG_LEN <= 40000) pass("S2 bash cat 4.5MB: tool result capped to " + TBIG_LEN + " chars");
      else issue("S2 tool result cap", "tool result entered the context at " + TBIG_LEN + " chars (4.5MB output)");
    }

    /* ── S3: budgets & long-horizon ────────────────────────────────── */
    console.log("\n== S3 budgets ==");
    def("xbsteps", "XB=steps", { tools: ["code"], budget: { maxSteps: 3 } });
    {
      const t = Date.now();
      const r = await sofuu.agent.run("xbsteps", "loop forever", {});
      bench("steps_budget_ms", Date.now() - t);
      if (r.stopped) pass("S3 maxSteps breach stops run (stopped=" + r.stopped + ", answer=" + JSON.stringify(r.answer.slice(0, 40)) + ")");
      else issue("S3 maxSteps NOT enforced", "stopped=" + r.stopped);
    }
    def("xbwall", "XB=wall", { tools: ["code"], budget: { maxWallMs: 2500 } });
    {
      for (let wi = 0; wi < 2; wi++) {
        const t = Date.now();
        const r = await sofuu.agent.run("xbwall", "sleep long", {});
        const ms = Date.now() - t;
        bench("wall_budget_actual_ms_run" + wi, ms);
        if (r.stopped) pass("S3 wall breach run " + wi + ": stopped=" + r.stopped + " at " + ms + "ms");
        else issue("S3 wall breach FLAKE run " + wi, "no stop reason at " + ms + "ms, answer=" + JSON.stringify(r.answer.slice(0, 50)));
        if (ms > 6000) issue("S3 wall overshoot run " + wi, "breach took " + ms + "ms (budget 2500)");
      }
    }
    def("xbfit", "XB=fit", { ctx_window: 3000 });
    {
      const hist = [];
      for (let i = 0; i < 60; i++) hist.push({ role: i % 2 ? "assistant" : "user", content: "HIST" + i + "-MARK filler words here for token weight. ".repeat(12) });
      const r = await sofuu.agent.run("xbfit", "TASK-NEEDLE-77 answer me", { history: hist });
      if (r.answer.indexOf("needle=yes") >= 0) pass("S3 fitGuard preserves the task under pressure");
      else issue("S3 fitGuard LOST the task", r.answer);
      bench("fit_hist_kept_of_60", Number((r.answer.match(/hist=(\d+)/) || [0, -1])[1]));
    }

    /* ── S4: delegation extremes ───────────────────────────────────── */
    console.log("\n== S4 delegation extremes ==");
    for (let i = 1; i <= 5; i++) def("fan" + i, "XB=fan" + i, {});
    def("fanbad", "XB=fanbad", {});
    def("fanorch", "XB=fanorch", { agents: ["fan1", "fan2", "fan3", "fan4", "fan5", "fanbad"] });
    {
      const t = Date.now();
      const r = await sofuu.agent.run("fanorch", "fan out", {});
      const ms = Date.now() - t;
      bench("fan6_with_500_child_ms", ms);
      const m = r.answer.match(/sub=(\d+) err=(yes|no)/);
      if (m && Number(m[1]) === 5 && m[2] === "yes") pass("S4 6-way fan-out: 5 ok + failing child framed as tool error");
      else issue("S4 fan-out mixed failure", r.answer);
      if (r.subRuns.length === 6) pass("S4 all 6 subRuns accounted");
      else issue("S4 subRuns count", String(r.subRuns.length));
      let maxIn = 0;
      for (const a of XREQ) { if (!a.end) continue; let in_ = 0; for (const b of XREQ) { if (!b.end) continue; if (b.start <= a.start && a.start < b.end) in_++; } if (in_ > maxIn) maxIn = in_; }
      bench("fan_max_in_flight", maxIn);
      if (maxIn >= 4) pass("S4 fan-out genuinely parallel (max in-flight " + maxIn + ")");
      else issue("S4 fan-out parallelism", "max in-flight " + maxIn);
    }
    {
      const names = [];
      for (let i = 0; i < 18; i++) { def("spc" + i, "XB=spc" + i + " specialist " + i + " with a fairly long when-hint describing when to route work here for the model to read.", {}); names.push("spc" + i); }
      def("spec", "XB=spec", { agents: names });
      const r = await sofuu.agent.run("spec", "who", {});
      const dl = Number((r.answer.match(/dlen=(\d+)/) || [0, 0])[1]);
      bench("delegate_desc_len_18_specialists", dl);
      if (dl > 0 && dl <= 2400) pass("S4 delegate description capped at 2400 (" + dl + ")");
      else issue("S4 delegate description cap", "len=" + dl);
    }
    def("mcw", "XB=mcw", {});
    def("mcdead", "XB=mcw", { base_url: "http://127.0.0.1:9/v1/chat/completions" });
    globalThis.XB_MB = ("MC-PIECE filler sentence with varied words number 42. ").repeat(20000);
    bench("mc_context_chars", globalThis.XB_MB.length);
    def("mcbig", "XB=mcbig", { agents: ["mcw"] });
    {
      const t = Date.now();
      const r = await sofuu.agent.run("mcbig", "count", {});
      bench("mc_1MB_cap_reject_ms", Date.now() - t);
      if (r.answer.indexOf("capped") >= 0) pass("S4 1MB @2000 chunk_chars refused at >12 chunks");
      else issue("S4 1MB map_context NOT capped", r.answer.slice(0, 60));
    }
    def("mcok", "XB=mcok", { agents: ["mcw"] });
    {
      /* 12 chunks × 64000 = the hard ceiling; 700k chars → 11 chunks. */
      globalThis.XB_MB = ("MC-PIECE filler sentence with varied words number 42. ").repeat(13000);
      bench("mc_ok_context_chars", globalThis.XB_MB.length);
      const t = Date.now();
      const r = await sofuu.agent.run("mcok", "count", {});
      const ms = Date.now() - t;
      bench("mc_700K_11chunks_ok_ms", ms);
      console.log("  mc tool result (" + MC_RESULT.length + " chars): " + JSON.stringify(MC_RESULT.slice(0, 200)));
      if (MC_RESULT.indexOf("tool error") === MC_RESULT.indexOf("tool error") && MC_RESULT.indexOf("tool error") >= 0)
        issue("S4 map_context ok path errored", JSON.stringify(MC_RESULT.slice(0, 160)));
      else if (MC_RESULT.indexOf("MC-PIECE") >= 0 && MC_RESULT.indexOf("map_context ·") >= 0)
        pass("S4 700K map_context ok path: merged answer + meta present (" + ms + "ms)");
      else issue("S4 map_context ok path", "tool result: " + JSON.stringify(MC_RESULT.slice(0, 120)));
    }
    def("mcstop", "XB=mcstop", { agents: ["mcdead"] });
    {
      const t = Date.now();
      const r = await sofuu.agent.run("mcstop", "x", {});
      bench("mc_deadchild_ms", Date.now() - t);
      console.log("  mcstop tool result (" + MC_STOP_RESULT.length + "): " + JSON.stringify(MC_STOP_RESULT.slice(0, 160)));
      if (MC_STOP_RESULT.indexOf("tool error") >= 0) pass("S4 reduce-child death surfaces as tool error (failure visible)");
      else issue("S4 stopped-child framing", JSON.stringify(MC_STOP_RESULT.slice(0, 120)));
    }
    def("cancchild", "XB=cancchild", { tools: ["code"] });
    def("cancparent", "XB=cancparent", { agents: ["cancchild"] });
    {
      const t = Date.now();
      const runP = sofuu.agent.run("cancparent", "go", { signal: "xbsig" });
      await new Promise(res2 => setTimeout(res2, 700));
      await sofuu.agent.cancel("xbsig");
      const r = await runP;
      const ms = Date.now() - t;
      bench("cancel_parent_ms", ms);
      if (r.stopped === "cancelled") pass("S4 parent cancel lands (stopped=cancelled)");
      else issue("S4 parent cancel", "stopped=" + r.stopped);
      if (ms >= 3500) issue("S4 cancel does NOT propagate into child runs (KNOWN)", "parent cancelled at 700ms but run took " + ms + "ms — child's sleep-4 bash ran to completion");
      else pass("S4 cancel propagated fast (" + ms + "ms)");
    }

    /* ── S5: transport & providers ─────────────────────────────────── */
    console.log("\n== S5 transport ==");
    try {
    def("xr429", "XB=r429", {});
    {
      const r = await sofuu.agent.run("xr429", "retry me", {});
      if (R429_N >= 3) pass("S5 429 handled — server saw " + R429_N + " attempts (2 retries)");
      else issue("S5 429 retry count", "server saw " + R429_N);
      if (!r.stopped) pass("S5 recovered after 429s");
      else issue("S5 429 recovery", "stopped=" + r.stopped);
    }
    def("xrcut", "XB=rcut", {});
    {
      RCUT_N = 0;
      const r = await sofuu.agent.run("xrcut", "cut me", {});
      bench("rcut_server_attempts", RCUT_N);
      const dup = r.answer.split("PARTIAL-ALPHA").length;
      if (r.answer.indexOf("TAIL-OK") >= 0 && dup === 2) pass("S5 mid-stream cut → continuation → deduped answer");
      else if (RCUT_N === 1) issue("S5 silent EOF truncation (FINDING)", "server saw 1 attempt; answer=" + JSON.stringify(r.answer.slice(0, 60)) + " — abrupt EOF without [DONE] ends the turn as if complete; CONTINUE never fires");
      else issue("S5 mid-stream cut handling", JSON.stringify(r.answer.slice(0, 90)) + " attempts=" + RCUT_N);
    }
    def("xrerr", "XB=rerr", {});
    {
      RERR_N = 0;
      const r = await sofuu.agent.run("xrerr", "err frame", {});
      if (r.answer.indexOf("RERR-CONT-OK") >= 0 && !r.stopped) pass("S5 in-stream error (rate-limit wording) → CONTINUE → recovered");
      else issue("S5 in-stream error frame", JSON.stringify(r.answer.slice(0, 60)) + " stopped=" + r.stopped);
    }
    def("xrmal", "XB=rmal", {});
    {
      RMAL_N = 0;
      const r = await sofuu.agent.run("xrmal", "malformed", {});
      if (r.answer.indexOf("recovered") >= 0) pass("S5 malformed SSE → retry → recovered");
      else issue("S5 malformed SSE", JSON.stringify(r.answer.slice(0, 60)) + " stopped=" + r.stopped);
    }
    def("xdead", "XB=dead", { base_url: "http://127.0.0.1:9/v1/chat/completions" });
    {
      const t = Date.now();
      try {
        const r = await sofuu.agent.run("xdead", "dead endpoint", {});
        bench("dead_endpoint_ms", Date.now() - t);
        issue("S5 dead endpoint RESOLVED", "stopped=" + r.stopped + " answer=" + JSON.stringify(r.answer.slice(0, 40)) + " — expected the loud-failure reject");
      } catch (eD) {
        bench("dead_endpoint_ms", Date.now() - t);
        /* BY DESIGN (bench report issue 9): run() rejecting on a dead
         * endpoint is the loud-failure contract — a silent empty answer
         * would be indistinguishable from success. The chat driver
         * catches; hosts must too. Documented PASS, not a bug. */
        pass("S5 dead endpoint rejects the run promise (loud-failure design)");
      }
    }
    def("xanth", "XB=anth", { provider: "anthropic", base_url: ANTH, model: "claude-mock",
      tools: [{ name: "get_weather", description: "w", parameters: { type: "object", properties: { city: { type: "string" } } }, execute: async () => "ANTH-TOOLRES-77 sunny" }] });
    {
      const r = await sofuu.agent.run("xanth", "weather?", {});
      if (r.answer.indexOf("anth-final-done") >= 0 && ANTH_OK === true) pass("S5 anthropic wire: tool round-trip ok");
      else issue("S5 anthropic wire", r.answer.slice(0, 60) + " | ANTH_OK=" + ANTH_OK + " len=" + ANTH_SEEN_LEN);
    }
    } catch (eS5) { issue("S5 crashed", String((eS5 && eS5.message) || eS5)); }

    /* ── S6: ML gates (needs brain) ────────────────────────────────── */
    console.log("\n== S6 ML gates ==");
    try {
    try {
      const brain = sofuu.memory.open(proj + "/.sofuu/brain/brain.qtsq", 768, "hash-v1");
      const seed = (t) => { const v = sofuu.ai.embedLocal(t); brain.remember(new Float32Array(v), t, "user_pin", 0); };
      seed("The legacy API endpoint was /v1/legacy, deprecated in 2021, replaced in 2022.");
      seed("Deploy runs through the ops pipeline every friday.");
      seed("The database is postgres on port 5433 with daily backups.");
      brain.flush();
      def("xfresh", "XB=fresh", { memory: "shared", brainPath: proj + "/.sofuu/brain/brain.qtsq" });
      const r = await sofuu.agent.run("xfresh", "tell me about the API endpoint history", {});
      if (r.answer.indexOf("+fresh") >= 0) pass("S6 freshness gate fires on stale recall");
      else if (r.answer.indexOf("recall-nonotice") >= 0) issue("S6 freshness gate did NOT fire", "recall present but no [freshness] notice");
      else issue("S6 freshness gate", "no recall at all: " + r.answer.slice(0, 50));
      def("xrel", "XB=rel", { memory: "shared", brainPath: proj + "/.sofuu/brain/brain.qtsq" });
      const r2 = await sofuu.agent.run("xrel", "what is the capital of France", {});
      if (r2.answer.indexOf("+rel") >= 0) pass("S6 relevance gate fires on tangential recall");
      else if (r2.answer.indexOf("recall-nonotice") >= 0) issue("S6 relevance gate did NOT fire", "recall present, no [relevance] notice");
      else issue("S6 relevance gate", "no recall: " + r2.answer.slice(0, 50));
    } catch (eMl) { issue("S6 ML setup", String((eMl && eMl.message) || eMl)); }
    {
      await sofuu.fs.writeFile(proj + "/dupme.txt", "DUPFILE-CONTENT-9\n");
      def("xdup", "XB=dup", { tools: ["code"] });
      const r = await sofuu.agent.run("xdup", "read twice", {});
      if (r.answer.indexOf("+nudge") >= 0) pass("S6 supervisor flags the exact repeat (dup_call)");
      else issue("S6 supervisor dup_call", r.answer);
    }
    } catch (eS6) { issue("S6 crashed", String((eS6 && eS6.message) || eS6)); }

    /* ── S7: benchmarks ────────────────────────────────────────────── */
    console.log("\n== S7 benchmarks ==");
    try {
    {
      const t2 = Date.now();
      for (let i = 0; i < 50; i++) sofuu.ai.embedLocal("benchmark payload number " + i + " with words. ");
      bench("embedLocal_50calls_ms", Date.now() - t2);
    }
    {
      try {
        const b = sofuu.memory.open(proj + "/.sofuu/brain/bench.qtsq", 768, "hash-v1");
        const t = Date.now();
        for (let i = 0; i < 400; i++) {
          const sentence = "memory entry " + i + " about topic " + (i % 20) + " with details and numbers " + i * 7 + ".";
          const v = sofuu.ai.embedLocal(sentence);
          b.remember(new Float32Array(v), sentence, "user", 0);
        }
        b.flush();
        bench("brain_seed_400_ms", Date.now() - t);
        const t2 = Date.now();
        for (let i = 0; i < 50; i++) { const v = sofuu.ai.embedLocal("query about topic " + (i % 20)); b.recall(new Float32Array(v), 15); }
        bench("brain_recall_50x_ms", Date.now() - t2);
      } catch (eB) { issue("S7 brain bench", String((eB && eB.message) || eB)); }
    }
    {
      try {
        await sofuu.fs.writeFile(proj + "/.sofuu/brain/corrupt.qtsq", "this is not a qtsq container at all, just bytes. ".repeat(50));
        const bad = sofuu.memory.open(proj + "/.sofuu/brain/corrupt.qtsq", 768, "hash-v1");
        /* Fixed 2026-09-14 (bench report issue 5): a garbage container used
         * to open SILENTLY as a fresh shell — the next flush would overwrite
         * the file with no trace. Now open backs the file up to corrupt.qtsq.bak,
         * warns on stderr, and starts fresh. */
        const bakExists = await sofuu.fs.exists(proj + "/.sofuu/brain/corrupt.qtsq.bak");
        let postOpen = "open-ok";
        try { postOpen += " count=" + bad.count(); } catch (e1) { postOpen += " count-threw: " + String((e1 && e1.message) || e1).slice(0, 60); }
        if (bakExists && bad.count() === 0) pass("S7 corrupted brain backed up + fresh start (bak present, count 0)");
        else issue("S7 corrupted brain backup/fresh", "bak=" + bakExists + " " + postOpen);
      } catch (eC) { issue("S7 corrupted brain open threw: " + String((eC && eC.message) || eC).slice(0, 90)); }
    }
    } catch (eS7) { issue("S7 crashed", String((eS7 && eS7.message) || eS7)); }
  } finally {
    process.chdir(prevCwd);
  }

  const wall = ((Date.now() - t0) / 1000).toFixed(1);
  console.log("\n==== EXTREME BENCH SUMMARY ====");
  console.log("wall: " + wall + "s   PASS: " + PASS_N + "   ISSUE: " + ISSUE_N);
  console.log("\nBENCHMARKS:");
  for (const b of BENCH) console.log("  " + b[0] + " = " + b[1]);
  console.log("\nISSUES (" + ISSUE_N + "):");
  for (const i of ISSUES) console.log("  - " + i);
}
main().catch(e => { console.log("FATAL " + String((e && e.message) || e)); });
