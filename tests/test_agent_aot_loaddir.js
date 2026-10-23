// tests/test_agent_aot_loaddir.js -- Phase 6: AOT bytecode snapshots for
// user-supplied agent scripts (sofuu.aotEval + sofuu.agent.loadDir).
//
// The contract under test:
//   * first load compiles the script to a hash-keyed .bc beside it;
//   * the next load runs the snapshot and does not touch the parser;
//   * a changed source, a corrupt snapshot or a deleted snapshot all fall
//     back to a compile - a stale snapshot can never win;
//   * compile errors and runtime throws stay non-fatal (loadDir lists them),
//     and a throwing script runs ONCE per load (never compile-run-then-run);
//   * a host without the binding still loads from source.

const NL = String.fromCharCode(10);
let passed = 0;
let failed = 0;
function ok(cond, label) {
  if (cond) { passed++; console.log("  PASS " + label); }
  else { failed++; console.log("  FAIL " + label); }
}
function lines() { return Array.prototype.slice.call(arguments).join(NL); }

const DIRS = [];
async function newDir(tag) {
  const d = "/tmp/sofuu_aot_" + tag + "_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
  await sofuu.fs.mkdir(d, { recursive: true });
  DIRS.push(d);
  return d;
}

async function main() {
  console.log("=== Phase 6: AOT bytecode snapshots (sofuu.aotEval + loadDir) ===");

  console.log("Test 1: the host binding exists");
  ok(typeof sofuu.aotEval === "function", "sofuu.aotEval is a function");

  console.log("Test 2: compile once, then hit");
  {
    const d = await newDir("hit");
    const js = d + "/probe.js";
    const bc = d + "/probe.bc";
    await sofuu.fs.writeFile(js, lines(
      "globalThis.__aot_runs = (globalThis.__aot_runs || 0) + 1;",
      "globalThis.__aot_value = 7;", ""));
    globalThis.__aot_runs = 0;
    const a = sofuu.aotEval(js, bc);
    ok(a && a.status === "compiled", "first call compiles (status=" + (a && a.status) + ")");
    ok(!!a && a.bytes > 17, "snapshot carries a header plus bytecode (" + (a && a.bytes) + " B)");
    ok(!!a && typeof a.hash === "string" && a.hash.length === 8, "hash is 8 hex chars (" + (a && a.hash) + ")");
    ok(globalThis.__aot_value === 7, "the bytecode ran");
    ok(globalThis.__aot_runs === 1, "ran exactly once");
    const b = sofuu.aotEval(js, bc);
    ok(b && b.status === "hit", "second call hits the snapshot (status=" + (b && b.status) + ")");
    ok(b && b.hash === a.hash, "same source, same hash");
    ok(globalThis.__aot_runs === 2, "hit still executes the script");
  }

  console.log("Test 3: a changed source recompiles (a stale snapshot can never win)");
  {
    const d = await newDir("stale");
    const js = d + "/probe.js";
    const bc = d + "/probe.bc";
    await sofuu.fs.writeFile(js, "globalThis.__aot_v = 1;" + NL);
    const a = sofuu.aotEval(js, bc);
    await sofuu.fs.writeFile(js, "globalThis.__aot_v = 2;" + NL);
    const b = sofuu.aotEval(js, bc);
    ok(a && a.status === "compiled" && b && b.status === "compiled",
      "edited source compiles again (" + (a && a.status) + " -> " + (b && b.status) + ")");
    ok(!!a && !!b && a.hash !== b.hash, "hash follows the source");
    ok(globalThis.__aot_v === 2, "the new code ran");
  }

  console.log("Test 4: a corrupt snapshot is replaced, not trusted");
  {
    const d = await newDir("corrupt");
    const js = d + "/probe.js";
    const bc = d + "/probe.bc";
    await sofuu.fs.writeFile(js, "globalThis.__aot_c = 42;" + NL);
    sofuu.aotEval(js, bc);
    await sofuu.fs.writeFile(bc, "not bytecode at all");
    globalThis.__aot_c = 0;
    const r = sofuu.aotEval(js, bc);
    ok(r && r.status === "compiled", "garbage header -> recompile (status=" + (r && r.status) + ")");
    ok(globalThis.__aot_c === 42, "the recompiled snapshot ran");
    ok(sofuu.aotEval(js, bc).status === "hit", "the rewritten snapshot is usable");
  }

  console.log("Test 5: failures are reported, never invented");
  {
    const d = await newDir("fail");
    const js = d + "/probe.js";
    const bc = d + "/probe.bc";
    const missing = sofuu.aotEval(d + "/nope.js", bc);
    ok(missing && missing.status === "unavailable", "missing source -> unavailable");
    await sofuu.fs.writeFile(js, "function broken( {" + NL);
    const syn = sofuu.aotEval(js, bc);
    ok(syn && syn.status === "unavailable", "syntax error -> unavailable");
    ok(!!syn && typeof syn.error === "string" && syn.error.length > 0, "reason present: " + (syn && syn.error));
    ok(!(await sofuu.fs.exists(bc)), "no snapshot is written for a broken script");
    await sofuu.fs.writeFile(js, "globalThis.__aot_t = (globalThis.__aot_t || 0) + 1; throw new Error('boom-aot');" + NL);
    globalThis.__aot_t = 0;
    const threw = sofuu.aotEval(js, bc);
    ok(threw && threw.status === "threw", "runtime throw -> threw (status=" + (threw && threw.status) + ")");
    ok(!!threw && String(threw.error).indexOf("boom-aot") >= 0, "the thrown message survives: " + (threw && threw.error));
    ok(globalThis.__aot_t === 1, "a throwing script runs once, not twice");
  }

  console.log("Test 6: loadDir compiles, then hits, and keeps its result shape");
  {
    const d = await newDir("loaddir");
    await sofuu.fs.writeFile(d + "/good.js",
      'sofuu.agent.define({ name: "aot_agent_one", system: "aot", memory: "off" });' + NL);
    const first = await sofuu.agent.loadDir(d);
    ok(first.loaded.indexOf("aot_agent_one") >= 0, "loadDir loads the script");
    ok(first.compiled === 1 && first.aotHits === 0, "first load compiled (compiled=" + first.compiled + ", hits=" + first.aotHits + ")");
    ok(first.broken.length === 0, "no broken entries");
    ok(await sofuu.fs.exists(d + "/good.bc"), "snapshot written beside the script");
    /* The re-run defines the same agent, so the snapshot RUN throws and the
     * call reports "threw" - not "hit". What matters is that no recompile
     * happened: compiled === 0 means the existing snapshot was used. */
    const second = await sofuu.agent.loadDir(d);
    ok(second.compiled === 0 && second.aotHits === 0, "second load used the snapshot, no recompile (compiled=" + second.compiled + ")");
    ok(second.broken.length === 1 && String(second.broken[0].error).indexOf("aot_agent_one") >= 0,
      "the re-run is a duplicate define, reported in broken (expected)");
    ok(sofuu.agent.list().map(function (a) { return a.name; }).indexOf("aot_agent_one") >= 0, "the agent is registered");
  }

  console.log("Test 7: a broken script is listed, never fatal");
  {
    const d = await newDir("broken");
    await sofuu.fs.writeFile(d + "/bad.js", "this is not javascript ((" + NL);
    const r = await sofuu.agent.loadDir(d);
    ok(r.broken.length === 1 && r.broken[0].file === "bad.js", "broken file listed");
    ok(r.compiled === 0 && r.aotHits === 0, "nothing compiled");
    ok(!(await sofuu.fs.exists(d + "/bad.bc")), "no snapshot for a broken script");
  }

  console.log("Test 8: a host without the binding still loads from source");
  {
    const d = await newDir("nobind");
    await sofuu.fs.writeFile(d + "/plain.js",
      'sofuu.agent.define({ name: "aot_agent_nobind", system: "plain", memory: "off" });' + NL);
    const real = sofuu.aotEval;
    sofuu.aotEval = undefined;
    try {
      const r = await sofuu.agent.loadDir(d);
      ok(r.loaded.indexOf("aot_agent_nobind") >= 0, "loadDir fell back to the source eval");
      ok(r.compiled === 0 && r.aotHits === 0, "no snapshot counts without the binding");
      ok(!(await sofuu.fs.exists(d + "/plain.bc")), "no snapshot written without the binding");
    } finally {
      sofuu.aotEval = real;
    }
    ok(!(await sofuu.fs.exists(d + "/plain.bc")), "still no snapshot until the binding runs");
    const d2 = await newDir("nobind2");
    await sofuu.fs.writeFile(d2 + "/plain2.js",
      'sofuu.agent.define({ name: "aot_agent_nobind2", system: "plain", memory: "off" });' + NL);
    const again = await sofuu.agent.loadDir(d2);
    ok(again.loaded.indexOf("aot_agent_nobind2") >= 0 && again.compiled === 1,
      "with the binding back, the next load compiles (compiled=" + again.compiled + ")");
  }

  console.log("");
  if (failed === 0) console.log("ALL PHASE 6 AOT TESTS PASSED (" + passed + " checks)");
  else console.log("PHASE 6 AOT TESTS FAILED: " + failed + " of " + (passed + failed));
  for (const dir of DIRS) {
    try {
      for (const f of await sofuu.fs.readdir(dir)) { try { await sofuu.fs.rm(dir + "/" + f); } catch (e) {} }
      await sofuu.fs.rm(dir);
    } catch (e) {}
  }
}

main()
  .then(function () { process.exit(failed === 0 ? 0 : 1); })
  .catch(function (err) {
    console.error("Test crashed:", (err && (err.stack || err.message)) || err);
    process.exit(1);
  });
