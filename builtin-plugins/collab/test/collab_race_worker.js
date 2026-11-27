// builtin-plugins/collab/test/collab_race_worker.js — one contender in the
// cross-process lease race. Started by run.sh with COLLAB_RACE_ID set; every
// worker claims the same path and reports its verdict to results/<id>.json.
//
// This is the case a process-local mutex cannot pass: each worker is a
// separate OS process with its own QuickJS heap and its own INSTANCE_ID.

const id = String((typeof process !== "undefined" && process.env && process.env.COLLAB_RACE_ID) || "0");

function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

(async function () {
  if (!sofuu.plugins.TOOLS || !sofuu.plugins.TOOLS.collab_claim) {
    await sofuu.plugins.discover("./plugins");
  }
  const tool = sofuu.plugins.TOOLS.collab_claim;
  if (!tool) throw new Error("collab_claim was not registered");

  const res = await tool.execute({
    paths: ["race/file.ts"],
    purpose: "race contender " + id,
    ttlMs: 60000,
    agentId: "agent-" + id
  });

  await sofuu.fs.mkdir("results", { recursive: true });
  await sofuu.fs.writeFile("results/" + id + ".json", JSON.stringify({
    id: id,
    ok: !!res.ok,
    granted: !!res.granted,
    code: (res.error && res.error.code) || "",
    conflicts: (res.conflicts || []).length
  }), "utf8");

  /* Keep the winner's lease alive while the slowest contender is still in
   * its claim window; otherwise an early process exit plus a host that
   * releases on shutdown could make the race pass for the wrong reason. */
  await sleep(600);
  process.exit(0);
})().catch(function (e) {
  console.error("race worker " + id + ": " + String((e && e.stack) || e));
  process.exit(1);
});
