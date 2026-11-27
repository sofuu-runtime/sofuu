// builtin-plugins/collab/test/collab_test.js — the collab contract, proven
// against the real runtime primitives (sofuu.fs, sofuu.policy, the plugin
// service dispatcher). No model is involved: every case drives the same
// entry points the agent loop drives.
//
// Run from a throwaway workspace that contains plugins/collab.ts:
//   builtin-plugins/collab/test/run.sh            (sets one up for you)
//
// Terminology: A and B are opaque agent labels, not personas. The test does
// not know or care what the workspace contains.

const P = sofuu.plugins;
const TOOLS = ["collab_claim", "collab_release", "collab_renew", "collab_post", "collab_sync"];

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log("PASS " + name + (detail ? "  [" + detail + "]" : "")); }
  else { fail++; console.log("FAIL " + name + (detail ? "  [" + detail + "]" : "")); }
}
function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }async function throwsCode(fn, code) {
  try { await fn(); return false; }
  catch (e) { return String((e && e.code) || "") === code; }
}
async function tool(name, args) {
  const t = P.TOOLS[name];
  if (!t || typeof t.execute !== "function") throw new Error("missing tool " + name);
  return await t.execute(args);
}
async function sync(args) { return await tool("collab_sync", args || {}); }
function leasePaths(res, agent) {
  const out = [];
  const leases = (res && res.activeLeases) || [];
  for (const l of leases) if (!agent || l.ownerAgentId === agent) out.push(l.path);
  out.sort();
  return out;
}

async function writePolicy(doc) {
  await sofuu.fs.mkdir(".sofuu", { recursive: true });
  await sofuu.fs.writeFile(".sofuu/policy.json", JSON.stringify(doc), "utf8");
  const raw = sofuu.policy.load();
  const res = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!res || res.ok !== true) throw new Error("policy load failed: " + JSON.stringify(res));
}

async function main() {
  /* ── registration ─────────────────────────────────────────────────── */
  /* `sofuu run` autoloads ./plugins before the script starts, so a normal
   * test run may already have the tools. Discovery's "registered nothing"
   * verdict in that case is expected (the registry diff has nothing to add);
   * only the fresh-load case can assert on its loaded list. */
  const preloaded = !!(P.TOOLS.collab_claim);
  const disc = await P.discover("./plugins");
  if (!preloaded) {
    check("discovery loads collab.ts", disc.loaded.length === 1 && disc.loaded[0].file === "collab.ts", JSON.stringify(disc.broken));
  } else {
    check("plugin was already auto-loaded", true);
  }
  check("all five coordination tools registered", TOOLS.every(function (n) { return !!P.TOOLS[n]; }));
  check("collab service registered", !!(P.SERVICES.collab && P.SERVICES.collab.spec));

  const started = await P.startServices();
  check("service starts", started.started.indexOf("collab") >= 0 || P.SERVICES.collab.started === true, JSON.stringify(started.errors));

  /* ── claim / conflict / overlap ───────────────────────────────────── */
  let r = await tool("collab_claim", { paths: ["src/a.ts"], purpose: "build the thing", ttlMs: 120000, agentId: "agent-a" });
  check("claim granted", r.ok === true && r.granted === true, JSON.stringify(r));
  check("claim returns the canonical path", r.leases && r.leases[0] && r.leases[0].path === "src/a.ts");

  r = await tool("collab_claim", { paths: ["src/a.ts"], purpose: "same owner again", agentId: "agent-a" });
  let s = await sync();
  check("same owner re-claim is idempotent", r.ok === true && leasePaths(s, "agent-a").length === 1, JSON.stringify(leasePaths(s, "agent-a")));

  r = await tool("collab_claim", { paths: ["src/a.ts"], purpose: "peer claim", agentId: "agent-b" });
  check("foreign file claim is refused", r.ok === false && r.error && r.error.code === "E_LEASE_CONFLICT", JSON.stringify(r));
  check("conflict names the holder", r.conflicts && r.conflicts[0] && r.conflicts[0].heldBy === "agent-a");

  await sofuu.fs.mkdir("src", { recursive: true });
  r = await tool("collab_claim", { paths: ["src/"], purpose: "directory claim", agentId: "agent-b" });
  check("foreign directory claim overlaps a file beneath it", r.ok === false && r.error.code === "E_LEASE_CONFLICT", JSON.stringify(r));

  r = await tool("collab_claim", { paths: ["src"], purpose: "dir without slash", agentId: "agent-b" });
  check("existing directory is detected without a trailing slash", r.ok === false && r.error.code === "E_LEASE_CONFLICT", JSON.stringify(r));

  r = await tool("collab_claim", { paths: ["src2/a.ts"], purpose: "sibling prefix", agentId: "agent-b" });
  check("src/ does not overlap src2/ (directory boundary)", r.ok === true, JSON.stringify(r));

  r = await tool("collab_claim", { paths: ["./src/a.ts"], purpose: "lexical alias", agentId: "agent-b" });
  check("./src/a.ts canonicalizes to the same lease", r.ok === false && r.error.code === "E_LEASE_CONFLICT", JSON.stringify(r));

  r = await tool("collab_claim", { paths: ["../escape.ts"], purpose: "escape", agentId: "agent-b" });
  check("workspace escape is denied", r.ok === false && r.error.code === "E_DENIED", JSON.stringify(r));

  /* ── all-or-nothing multi-path claim ─────────────────────────────── */
  r = await tool("collab_claim", { paths: ["src/b.ts", "src/a.ts"], purpose: "partial", agentId: "agent-b" });
  s = await sync();
  check("multi-path claim is all-or-nothing", r.ok === false && leasePaths(s, "agent-b").indexOf("src/b.ts") < 0, JSON.stringify(leasePaths(s, "agent-b")));

  /* ── release / renew / expiry ─────────────────────────────────────── */
  r = await tool("collab_release", { paths: ["src/a.ts"], agentId: "agent-a" });
  s = await sync();
  check("release removes only the named path", r.ok === true && r.released >= 1 && leasePaths(s, "agent-a").indexOf("src/a.ts") < 0);
  check("released path can be re-claimed", (await tool("collab_claim", { paths: ["src/a.ts"], purpose: "reuse", agentId: "agent-b" })).ok === true);
  await tool("collab_release", { agentId: "agent-b" });

  r = await tool("collab_claim", { paths: ["src/c.ts"], purpose: "renew me", ttlMs: 1000, agentId: "agent-a" });
  r = await tool("collab_renew", { paths: ["src/c.ts"], ttlMs: 60000, agentId: "agent-a" });
  await sleep(1200);
  s = await sync();
  check("renew extends past the original TTL", r.ok === true && leasePaths(s, "agent-a").indexOf("src/c.ts") >= 0);
  await tool("collab_release", { agentId: "agent-a" });

  await tool("collab_claim", { paths: ["src/d.ts"], purpose: "short", ttlMs: 1000, agentId: "agent-a" });
  await sleep(1200);
  s = await sync();
  check("expired lease is swept", leasePaths(s).indexOf("src/d.ts") < 0, JSON.stringify(leasePaths(s)));

  /* ── board ordering / cursor ──────────────────────────────────────── */
  const d1 = await tool("collab_post", { kind: "decision", payload: { key: "api-shape", statement: "requests are opaque JSON" }, agentId: "agent-a" });
  const d2 = await tool("collab_post", { kind: "file_change", payload: { path: "src/e.ts", summary: "created e" }, agentId: "agent-b" });
  check("board assigns monotonic seq", d1.ok && d2.ok && d2.seq === d1.seq + 1, d1.seq + " -> " + d2.seq);
  s = await sync({ sinceSeq: d1.seq, limit: 10 });
  check("sinceSeq returns only newer records", s.records.length === 1 && s.records[0].seq === d2.seq, JSON.stringify(s.records.map(function (x) { return x.seq; })));
  check("decision is folded into current state", (s.decisions || []).some(function (x) { return x.payload.key === "api-shape"; }));

  /* ── compaction keeps state and the sequence monotonic ───────────── */
  await sofuu.fs.mkdir(".sofuu/collab", { recursive: true });
  await sofuu.fs.writeFile(".sofuu/collab/config.json",
    JSON.stringify({ boardMaxBytes: 65536, boardRetainRecords: 16 }), "utf8");
  const preCompact = await tool("collab_post", {
    kind: "decision", payload: { key: "compact-key", statement: "first" }, agentId: "agent-a"
  });
  let bulkOk = true;
  for (let i = 0; i < 120; i++) {
    const rr = await tool("collab_post", {
      kind: "file_change",
      payload: { path: "bulk/f" + i + ".txt", summary: "x".repeat(800) },
      agentId: "agent-a"
    });
    if (!rr.ok) { bulkOk = false; check("bulk post " + i + " succeeded", false, JSON.stringify(rr)); break; }
  }
  const postCompact = await tool("collab_post", {
    kind: "decision", payload: { key: "compact-key", statement: "second" }, agentId: "agent-b"
  });
  check("seq keeps advancing across compaction",
    bulkOk && postCompact.ok && postCompact.seq === preCompact.seq + 121,
    preCompact.seq + " -> " + postCompact.seq);
  const boardSize = sofuu.fs.statSync(".sofuu/collab/board.ndjson").size;
  check("board is compacted below boardMaxBytes", boardSize <= 65536, String(boardSize));
  const cs = await sync({ sinceSeq: 1, limit: 200 });
  check("sinceSeq before the compacted head reports a gap", cs.gap === true);
  const compactDecisions = (cs.decisions || []).filter(function (x) { return x.payload.key === "compact-key"; });
  check("compaction keeps only the latest decision per key",
    compactDecisions.length === 1 && compactDecisions[0].payload.statement === "second",
    JSON.stringify(compactDecisions.map(function (x) { return x.payload.statement; })));
  await sofuu.fs.writeFile(".sofuu/collab/config.json", "{}", "utf8");

  /* ── turn-start injection ─────────────────────────────────────────── */
  const evt1 = await P.emitTurnStart({ turnId: "run-z1", prompt: "do work", agent: "agent-z", depth: 0 });
  check("turn start injects unseen board deltas", evt1.prompt.indexOf("<shared_context") >= 0 && evt1.prompt.indexOf("api-shape") >= 0);
  const evt2 = await P.emitTurnStart({ turnId: "run-z2", prompt: "do work", agent: "agent-z", depth: 0 });
  check("cursor prevents replay on the next turn", evt2.prompt.indexOf("<shared_context") < 0);

  /* ── write-time enforcement ───────────────────────────────────────── */
  await tool("collab_claim", { paths: ["guard/x.ts"], purpose: "guarded", agentId: "agent-a" });
  let blocked = false, code = "";
  try {
    await P.emitToolCall({ tool: "write_file", args: { path: "guard/x.ts", content: "x" }, agent: "agent-b", turnId: "run-b" });
  } catch (e) { blocked = true; code = String((e && e.code) || ""); }
  check("onToolCall blocks a foreign-leased write", blocked && code === "E_LEASE_CONFLICT", code);

  let okB = true;
  try {
    await P.emitToolCall({ tool: "write_file", args: { path: "guard/y.ts", content: "y" }, agent: "agent-b", turnId: "run-b" });
  } catch (e) { okB = false; }
  s = await sync();
  check("onToolCall auto-claims a free path", okB && leasePaths(s, "agent-b").indexOf("guard/y.ts") >= 0, JSON.stringify(leasePaths(s, "agent-b")));

  /* ── automatic file_change broadcast ─────────────────────────────── */
  const before = (await sync()).nextSeq || 0;
  await P.emitToolResult({
    tool: "write_file",
    args: { path: "guard/y.ts", content: "y" },
    result: { text: "created guard/y.ts (1 chars)" },
    ok: true, agent: "agent-b", turnId: "run-b"
  });
  s = await sync({ sinceSeq: before, limit: 10 });
  check("successful write publishes file_change", s.records.some(function (x) {
    return x.kind === "file_change" && x.payload.path === "guard/y.ts";
  }), JSON.stringify(s.records.map(function (x) { return x.kind; })));

  /* ── failed tool rolls back its auto-claim ────────────────────────── */
  await P.emitToolCall({ tool: "write_file", args: { path: "fail/a.ts", content: "x" }, agent: "agent-b", turnId: "run-fail" });
  s = await sync();
  check("auto-claim exists before the tool result", leasePaths(s, "agent-b").indexOf("fail/a.ts") >= 0, JSON.stringify(leasePaths(s, "agent-b")));
  await P.emitToolResult({
    tool: "write_file",
    args: { path: "fail/a.ts", content: "x" },
    result: { text: "E_FILE_NOT_FOUND: no such directory" },
    ok: false, agent: "agent-b", turnId: "run-fail"
  });
  s = await sync();
  check("failed tool releases its auto-claim", leasePaths(s, "agent-b").indexOf("fail/a.ts") < 0, JSON.stringify(leasePaths(s, "agent-b")));

  /* ── apply_patch gate covers every path in the patch ─────────────── */
  await tool("collab_claim", { paths: ["patch/owned.ts"], purpose: "patch guard", agentId: "agent-a" });
  blocked = false; code = "";
  try {
    await P.emitToolCall({
      tool: "apply_patch",
      args: { patch: "*** Begin Patch\n*** Update File: patch/owned.ts\n@@\n-a\n+b\n*** End Patch" },
      agent: "agent-b", turnId: "run-patch"
    });
  } catch (e) { blocked = true; code = String((e && e.code) || ""); }
  check("apply_patch is blocked by a foreign lease", blocked && code === "E_LEASE_CONFLICT", code);

  let patchOk = true;
  try {
    await P.emitToolCall({
      tool: "apply_patch",
      args: { patch: "*** Begin Patch\n*** Add File: patch/free-a.ts\n+x\n*** Add File: patch/free-b.ts\n+y\n*** End Patch" },
      agent: "agent-b", turnId: "run-patch"
    });
  } catch (e) { patchOk = false; }
  s = await sync();
  check("apply_patch auto-claims every free path",
    patchOk &&
    leasePaths(s, "agent-b").indexOf("patch/free-a.ts") >= 0 &&
    leasePaths(s, "agent-b").indexOf("patch/free-b.ts") >= 0,
    JSON.stringify(leasePaths(s, "agent-b")));
  await P.emitTurnEnd({ turnId: "run-patch", agent: "agent-b" });

  /* ── turn-end cleanup ─────────────────────────────────────────────── */
  await P.emitTurnEnd({ turnId: "run-b", agent: "agent-b" });
  s = await sync();
  check("turn end releases the run's leases", leasePaths(s, "agent-b").length === 0, JSON.stringify(leasePaths(s, "agent-b")));

  /* ── shell visibility ─────────────────────────────────────────────── */
  await tool("collab_claim", { paths: ["guard/z.ts"], purpose: "shell guard", agentId: "agent-a" });
  blocked = false; code = "";
  try {
    await P.emitToolCall({ tool: "bash", args: { command: "echo hi > guard/z.ts" }, agent: "agent-b", turnId: "run-s" });
  } catch (e) { blocked = true; code = String((e && e.code) || ""); }
  check("bash naming a foreign-leased path is blocked", blocked && code === "E_LEASE_CONFLICT", code);

  await tool("collab_claim", { paths: ["guard/exp.ts"], purpose: "expired shell guard", ttlMs: 1000, agentId: "agent-a" });
  await sleep(1200);
  let staleOk = true;
  try {
    await P.emitToolCall({ tool: "bash", args: { command: "echo hi > guard/exp.ts" }, agent: "agent-b", turnId: "run-s2" });
  } catch (e) { staleOk = false; }
  check("bash ignores an expired lease", staleOk);

  /* ── strict mode ──────────────────────────────────────────────────── */
  await writePolicy({ collab: { enforcement: "strict_lease" } });
  blocked = false; code = "";
  try {
    await P.emitToolCall({ tool: "write_file", args: { path: "strict/new.ts", content: "n" }, agent: "agent-b", turnId: "run-strict" });
  } catch (e) { blocked = true; code = String((e && e.code) || ""); }
  check("strict mode refuses an unleased write", blocked && code === "E_LEASE_REQUIRED", code);

  const strictArgs = { paths: ["strict/new.ts"], purpose: "strict claim", agentId: "agent-b" };
  await P.emitToolCall({ tool: "collab_claim", args: strictArgs, agent: "agent-b", turnId: "run-strict" });
  await tool("collab_claim", strictArgs);
  let strictOk = true;  try {
    await P.emitToolCall({ tool: "write_file", args: { path: "strict/new.ts", content: "n" }, agent: "agent-b", turnId: "run-strict" });
  } catch (e) { strictOk = false; }
  check("strict mode allows an owned write", strictOk);
  await P.emitTurnEnd({ turnId: "run-strict", agent: "agent-b" });
  await writePolicy({ collab: { enforcement: "auto_claim" } });

  /* ── runtime ownership is injected, not model-supplied ───────────── */
  const claimArgs = { paths: ["owner/a.ts"], purpose: "owner injection", agentId: "forged-name" };
  await P.emitToolCall({ tool: "collab_claim", args: claimArgs, agent: "agent-own", turnId: "run-own" });
  check("hook overwrites agentId with the runtime identity", claimArgs.agentId === "agent-own");
  check("hook injects the owner record", claimArgs.__owner && claimArgs.__owner.agent === "agent-own" && claimArgs.__owner.instanceId);
  r = await tool("collab_claim", claimArgs);
  check("tool uses the injected owner", r.ok === true && r.ownerAgentId === "agent-own", JSON.stringify(r));

  /* A caller that names a peer in a release must not touch the peer's lease. */
  const releaseArgs = { agentId: "agent-a" };
  await P.emitToolCall({ tool: "collab_release", args: releaseArgs, agent: "agent-own", turnId: "run-own" });
  await tool("collab_release", releaseArgs);
  s = await sync();
  check("release cannot reach a peer's lease through agentId", leasePaths(s, "agent-a").indexOf("guard/x.ts") >= 0, JSON.stringify(leasePaths(s, "agent-a")));

  /* ── cleanup ──────────────────────────────────────────────────────── */
  await tool("collab_release", { agentId: "agent-a" });
  await tool("collab_release", { agentId: "agent-own" });
  s = await sync();
  check("final state has no active leases", leasePaths(s).length === 0, JSON.stringify(leasePaths(s)));

  /* ── process shutdown releases this instance's leases ─────────────── */
  await tool("collab_claim", { paths: ["stop/a.ts"], purpose: "stop cleanup", agentId: "agent-stop" });
  await P.SERVICES.collab.spec.onStop();
  s = await sync();
  check("onStop releases this process's leases", leasePaths(s).length === 0, JSON.stringify(leasePaths(s)));

  /* ── a build without the atomic primitive fails closed ────────────── */
  const savedRename = sofuu.fs.renameSync;
  delete sofuu.fs.renameSync;
  if (typeof sofuu.fs.renameSync === "function") {
    check("capability probe skipped (fs property is sealed)", true);
  } else {
    let capCode = "";
    try {
      await P.emitToolCall({ tool: "write_file", args: { path: "cap/a.ts", content: "x" }, agent: "agent-b", turnId: "run-cap" });
    } catch (e) { capCode = String((e && e.code) || ""); }
    check("missing renameSync fails the write gate closed", capCode === "E_CAPABILITY", capCode);
    const capSync = await sync({});
    check("tools report the missing capability", capSync.ok === false && capSync.error.code === "E_CAPABILITY", JSON.stringify(capSync.error || {}));
    sofuu.fs.renameSync = savedRename;
  }

  console.log("\ncollab_test: " + (fail === 0 ? "OK" : "FAILED (" + fail + ")") + " — " + pass + " passed");
  process.exit(fail === 0 ? 0 : 1);
}

main().catch(function (e) {
  console.error("collab_test: " + String((e && e.stack) || e));
  process.exit(1);
});
