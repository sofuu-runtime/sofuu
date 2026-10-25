package com.sofuu.engine

import org.json.JSONArray
import org.json.JSONObject

/**
 * Direct access to the runtime's cognitive and vector surfaces.
 *
 * Everything here runs inside the engine through [SofuuEngine]'s query
 * channel, on the engine thread, as one eval'd expression — so it obeys the
 * same one-runtime-one-thread rule as every other engine call.
 *
 * The brain handle is never opened here. `sofuu.agent.brainFor` hands out
 * the ONE cached CMA handle for the active workspace, and a QTSQ flush
 * rewrites the whole file, so a second long-lived handle would silently
 * clobber the first on every write. Everything routes through `brainFor`.
 */
class RuntimeSurface(
    private val engine: SofuuEngine,
    /** Reads the app's Brain toggle at call time. A diagnostic pane must
     *  not open a brain the user turned off, and must say WHICH reason it
     *  is refusing rather than blaming a missing artifact. */
    private val memoryEnabled: () -> Boolean = { true },
) {

    // ── Memory (CMA brain) ───────────────────────────────────────

    /**
     * One JS preamble shared by every direct brain operation.
     *
     * `brainFor`'s cache key is (brain path + backend id), NOT the agent
     * name — so a def carrying the chat project's brain path resolves to
     * the SAME live handle the chat turn uses. Opening a second handle
     * would clobber the first on the next QTSQ flush.
     *
     * `vecFor` embeds in the brain's OWN space. The previous version tried
     * the semantic projector first regardless, which is a 64-dim vector
     * against a 768-dim hash index.
     */
    private fun brainScript(body: String): String {
        if (!memoryEnabled()) {
            return "(function(){ return { ok:false, error:'memory is off — enable the Brain " +
                "in Settings → Context & reasoning' }; })()"
        }
        return """
        (function(){ try {
          var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {};
          var proj = (st && st.project) ? String(st.project) : '';
          if (!proj) return { ok:false, error:'no project — open a workspace first' };
          var def = { name:'chat', memory:'shared',
                      brainPath: proj + '/.sofuu/brain/brain.qtsq' };
          var b = sofuu.agent.brainFor(def);
          if (!b || !b.cma) return { ok:false, error:'brain unavailable (memory off or artifact missing)' };
          function kindOf(){ return (b.backend && b.backend.kind) ? b.backend.kind : 'hash'; }
          function dimOf(){ return (b.backend && b.backend.dim) ? b.backend.dim : 0; }
          function vecFor(text){
            var k = kindOf();
            if (k === 'semantic' && sofuu.ai && typeof sofuu.ai.embedLocalSemantic === 'function') {
              return sofuu.ai.embedLocalSemantic(String(text));
            }
            if (k === 'remote') return null;
            if (sofuu.ai && typeof sofuu.ai.embedLocal === 'function') {
              return sofuu.ai.embedLocal(String(text));
            }
            return null;
          }
          function dimError(v){
            if (!v) return 'remote embedding backend has no synchronous embedder in this pane';
            if (dimOf() > 0 && v.length !== dimOf()) {
              return 'embedding space mismatch (got ' + v.length + ' dims, brain uses ' + dimOf() + ')';
            }
            return null;
          }
          $body
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
    """.trimIndent()
    }

    /**
     * Count, backend id and recall health for the active brain, or null
     * when memory is off or the artifact is unavailable.
     */
    fun brainInfo(): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            return {
              ok: true,
              count: b.cma.count(),
              backend: b.backend ? b.backend.id : null,
              dim: dimOf(),
              semantic: !!b.sem
            };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /**
     * Nearest memories for [query].
     *
     * Embeds with the brain's own backend rather than passing a raw
     * vector, so the query lands in the same space the memories were
     * stored in — mixing spaces is the one way to get confidently wrong
     * recall results.
     */
    fun brainRecall(query: String, k: Int = 8): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            var v = vecFor(${SofuuEngine.quote(query)});
            var bad = dimError(v);
            if (bad) return { ok:false, error: bad };
            var hits = b.cma.recall(v, ${k.coerceIn(1, 64)}) || [];
            return { ok:true, hits: Array.isArray(hits) ? hits : Array.prototype.slice.call(hits) };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /** Drop one memory by id. The id comes from a recall hit. */
    fun brainForget(id: Int): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            b.cma.forget($id);
            return { ok:true, count: b.cma.count() };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /** Merge and de-duplicate stored memories. Irreversible. */
    fun brainConsolidate(): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            var before = b.cma.count();
            var n = b.cma.consolidate();
            return { ok:true, merged:n, before:before, after:b.cma.count() };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /** Run one Ebbinghaus decay tick. The runtime REQUIRES dt_seconds
     *  (`memory.rs` throws on a missing argument); one day is the tick the
     *  desktop's maintenance pass applies. */
    fun brainDecay(dtSeconds: Int = 86_400): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            var before = b.cma.count();
            b.cma.decayTick(${dtSeconds.coerceIn(1, 31_536_000)});
            return { ok:true, before:before, after:b.cma.count(), dtSeconds:${dtSeconds.coerceIn(1, 31_536_000)} };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /**
     * Pin a fact to the project brain (the `/remember` command).
     *
     * Goes through the same cached handle the chat turn uses and embeds in
     * the brain's own space. `flush()` lands it on disk — the CMA's
     * finalizer never flushes on its own.
     */
    fun brainRemember(fact: String): JSONObject? = engine.evalJsonOnEngine(
        brainScript(
            """
            var text = ${SofuuEngine.quote(fact)};
            var v = vecFor(text);
            var bad = dimError(v);
            if (bad) return { ok:false, error: bad };
            var id = b.cma.remember(v, text, 'user', 0);
            b.cma.flush();
            return { ok: id >= 0, id: id, count: b.cma.count() };
            """.trimIndent()
        )
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    // ── ML gates ─────────────────────────────────────────────────

    /** Gate states, working-set counters, online-learner status. */
    fun mlState(): JSONObject? = engine.evalJsonOnEngine(
        // Both entry points return JSON STRINGS (ml/mod.rs to_json); handing
        // them through unparsed left optJSONObject("info") null, so the pane
        // rendered 0 runs / no gate rows against a healthy engine.
        "(function(){ function o(x){ if (typeof x === 'string') { try { return JSON.parse(x); } catch(e){ return null; } } return x; } " +
            "try { return { ok:true, info: o(sofuu.ml.info()), workset: o(sofuu.ml.workset()) }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /**
     * Teach a gate from an outcome.
     *
     * The runtime's contract is narrow and exact (`ml/mod.rs:344`): the
     * payload must carry `model:"supervisor"`, and `kind` must be one of
     * `outcome`, `wrong` or `wasted`. Anything else is refused silently,
     * so the fields are set here rather than left to a caller to guess.
     *
     * `wrong` and `wasted` are gold labels — they name the most recent
     * call and count double against a proxy's weight, which makes them
     * the right choice for a turn you know was bad. `outcome` needs a
     * confidence the engine will accept; it abstains on ambiguous results
     * rather than send a coin-flip label into its own weights.
     *
     * Returns the engine's own `{ok:false}` when it declines, which is the
     * honest answer and not an error to swallow.
     */
    fun mlFeedback(kind: String, run: String = "", step: Int = 0, wasted: Boolean = false, conf: Double = 1.0): JSONObject? {
        val payload = JSONObject()
            .put("model", "supervisor")
            .put("kind", kind)
        if (run.isNotEmpty()) payload.put("run", run)
        if (step > 0) payload.put("step", step)
        if (wasted) payload.put("wasted", true)
        payload.put("conf", conf)
        return engine.evalJsonOnEngine(
            "(function(){ try { return sofuu.ml.feedback(${SofuuEngine.quote(payload.toString())}); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ).get(30, java.util.concurrent.TimeUnit.SECONDS)
    }

    /**
     * Ask a gate for its verdict on a proposed action, without running it.
     *
     * `supervisor.check` is the pre-tool-call checkpoint and `loop` the
     * loop-boundary one; both return `{ok, reason, nudge, score, source}`
     * where `source` is `rule` or `model`. Exposed because it is the only
     * way to see WHY a call was nudged or blocked — the agent only ever
     * sees the nudge, not the verdict behind it.
     */
    fun mlCheck(tool: String, args: JSONObject, run: String = "", step: Int = 0): JSONObject? {
        val payload = JSONObject().put("tool", tool).put("args", args)
        if (run.isNotEmpty()) payload.put("run", run)
        if (step > 0) payload.put("step", step)
        return engine.evalJsonOnEngine(
            "(function(){ try { return sofuu.ml.supervisor.check(${SofuuEngine.quote(payload.toString())}); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ).get(30, java.util.concurrent.TimeUnit.SECONDS)
    }

    // ── Vectors ──────────────────────────────────────────────────

    /**
     * Cosine similarity between two strings, embedding each with the
     * space named (blank = the brain's default, hash-768).
     *
     * Computed host-side from the two vectors rather than through
     * `sofuu.ai.similarity`, so the same space is used for both sides —
     * mixing spaces gives a number that looks fine and means nothing.
     */
    fun similarity(a: String, b: String, space: String? = null): JSONObject? {
        val va = engine.embedLocal(a, space).get(5, java.util.concurrent.TimeUnit.SECONDS) ?: return null
        val vb = engine.embedLocal(b, space).get(5, java.util.concurrent.TimeUnit.SECONDS) ?: return null
        if (va.size != vb.size || va.isEmpty()) return null
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in va.indices) {
            dot += va[i].toDouble() * vb[i]
            na += va[i].toDouble() * va[i]
            nb += vb[i].toDouble() * vb[i]
        }
        if (na == 0.0 || nb == 0.0) return null
        return JSONObject().put("ok", true).put("similarity", dot / (Math.sqrt(na) * Math.sqrt(nb)))
            .put("dim", va.size).put("space", space ?: "(default)")
    }

    /**
     * Which embedding spaces this build can actually use, and why one
     * might be unavailable. `embedImage` in particular needs its
     * projector artifact, and that is missing on most builds rather than
     * failing at call time.
     */
    fun spaces(): JSONObject? = engine.evalJsonOnEngine(
        // embedInfo / embedInfoV2 also return JSON STRINGS (they are parsed
        // by agent.js itself), and v2 is absent on some builds — parse what
        // is there and leave v2 null rather than failing the whole call.
        "(function(){ function o(x){ if (typeof x === 'string') { try { return JSON.parse(x); } catch(e){ return null; } } return x; } " +
            "function f(n){ return (sofuu.ai && typeof sofuu.ai[n] === 'function') ? o(sofuu.ai[n]()) : null; } " +
            "try { return { ok:true, info: f('embedInfo'), v2: f('embedInfoV2') }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /**
     * Functional QTSQ persistence probe.
     *
     * `typeof __qtsq_session_save === 'function'` is useless here: the
     * session primitives are registered unconditionally and their bodies
     * degrade to `-1` when the binary was built without the codec
     * (`#[cfg(not(has_qtsq))]`). The only honest test is to write and read
     * a tiny file and look at the return code.
     */
    fun qtsqState(): JSONObject? = engine.evalJsonOnEngine(
        """
        (function(){ try {
          var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {};
          var p = (st && st.project) ? String(st.project) : '';
          if (!p) return { ok:true, canPersist:false, write:-2, reason:'no project open' };
          var f = p + '/.sofuu/qtsq-probe.qtsq';
          // The codec does not mkdir; touch a file through the primitive that
          // does, so a fresh project tree does not read as "no QTSQ".
          try { if (typeof __session_write_file === 'function') __session_write_file(p + '/.sofuu/.keep', ''); } catch (e0) {}
          var w = __qtsq_session_save(f, '{"probe":1}', 'sofuu-probe');
          var r = __qtsq_session_load(f, 'sofuu-probe');
          var hit = String(r) === '{"probe":1}';
          try { __qtsq_session_save(f, '{}', 'sofuu-probe'); } catch (e2) {}
          return { ok:true, canPersist: (w === 0 && hit), write: w, read: hit,
                   reason: (w === 0 && hit) ? '' : ('save returned ' + w) };
        } catch(e){ return { ok:false, canPersist:false, write:-3, reason:String(e&&e.message||e) }; } })()
        """.trimIndent()
    ).get(15, java.util.concurrent.TimeUnit.SECONDS)

    // ── Agents ─────────────────────────────────────────────────────

    /** Registered agents with their tools, budgets and sub-agent fan-out. */
    fun agents(): JSONArray? = engine.evalJsonOnEngine(
        "(function(){ try { return { ok:true, agents: sofuu.agent.list() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)?.optJSONArray("agents")

    /**
     * Register an agent from the phone.
     *
     * The shape is the runtime's, not the UI's: `tools` takes `'web'`,
     * `'code'`, or `{builtin:"name"}` for anything already in
     * `sofuu.tools.TOOLS` — which, after host-tool registration, includes
     * the phone's own tools. `budget` is a guardrail rather than a wall:
     * breaching `maxSteps` salvages an answer and reports `budget_steps`
     * rather than throwing, so a generous default is the safe default.
     *
     * Refuses a duplicate name the same way the runtime does, so the error
     * a user sees is the one the engine actually raised.
     */
    fun defineAgent(
        name: String,
        system: String,
        tools: List<String>,
        budget: JSONObject = JSONObject(),
        provider: String = "",
        model: String = "",
        approve: String = "sensitive",
    ): JSONObject? {
        val toolsJson = JSONArray().apply {
            tools.forEach { t ->
                when {
                    t == "web" || t == "code" -> put(t)
                    else -> put(JSONObject().put("builtin", t))
                }
            }
        }
        val def = JSONObject()
            .put("name", name)
            .put("system", system)
            .put("tools", toolsJson)
            // The runtime skips the interactive gate entirely at the default
            // 'none' — an agent defined without this could never produce the
            // approval card this pane renders and polls for.
            .put("approve", approve)
        if (budget.length() > 0) def.put("budget", budget)
        if (provider.isNotEmpty()) def.put("provider", provider)
        if (model.isNotEmpty()) def.put("model", model)
        return engine.evalJsonOnEngine(
            """
            (function(){ try {
              var h = sofuu.agent.define(JSON.parse(${SofuuEngine.quote(def.toString())}));
              return { ok:true, name: h.name };
            } catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()
            """.trimIndent()
        ).get(30, java.util.concurrent.TimeUnit.SECONDS)
    }

    /**
     * Run an agent headlessly and wait for its answer.
     *
     * Goes through the same `agent.run` the chat turn uses, so the tools,
     * the permission profile, the budget and the brain recall all behave
     * identically — the only difference is that the result is collected
     * instead of streamed. Long: a phone caller should expect tens of
     * seconds, which is why this is off the engine thread.
     */
    fun runAgent(
        name: String,
        task: String,
        timeoutMs: Long,
        signal: String? = null,
    ): java.util.concurrent.CompletableFuture<JSONObject?> {
        // approveInteractive parks an approval-gated tool instead of denying
        // it outright; the pane drains agentPendingApprovals() while the run
        // is in flight. `signal` is the handle agent.cancel() takes.
        val opts = JSONObject().put("approveInteractive", true)
        if (!signal.isNullOrEmpty()) opts.put("signal", signal)
        return engine.queryFuture(
            "(function(){ var r = sofuu.agent.run(" +
                SofuuEngine.quote(name) + ", " + SofuuEngine.quote(task) + ", " +
                opts.toString() + "); " +
                "return r.then(function(v){ return { ok:true, answer:v.answer, steps:v.steps, " +
                "stopped:v.stopped, usage:v.usage, runId:v.runId }; }, " +
                "function(e){ return { ok:false, error:String(e&&e.message||e) }; }); })()",
            timeoutMs,
        )
    }

    /** Parked approval requests from headless agent runs ([] when none). */
    fun agentPendingApprovals(): JSONArray? =
        engine.evalJsonOnEngine(
            "(function(){ try { return { ok:true, list: sofuu.agent.pendingApprovals() }; } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ).get(10, java.util.concurrent.TimeUnit.SECONDS)?.optJSONArray("list")

    /** Answer a parked headless-run approval. */
    fun resolveAgentApproval(id: String, allow: Boolean, always: Boolean): JSONObject? =
        engine.evalJsonOnEngine(
            "(function(){ try { return sofuu.agent.resolveApproval(" +
                SofuuEngine.quote(id) + ", $allow, $always); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ).get(10, java.util.concurrent.TimeUnit.SECONDS)

    /** Stop a headless run by the signal id passed to [runAgent]. */
    fun cancelAgent(signal: String): Boolean =
        engine.evalJsonOnEngine(
            "(function(){ try { return { ok: !!sofuu.agent.cancel(" +
                SofuuEngine.quote(signal) + ") }; } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ).get(10, java.util.concurrent.TimeUnit.SECONDS)?.optBoolean("ok") ?: false

    /** Caps for a model id, without switching to it. */
    fun modelCaps(model: String): JSONObject? = engine.evalJsonOnEngine(
        "(function(){ try { return { ok:true, caps: sofuu.ai.modelCaps(${SofuuEngine.quote(model)}) }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)

    /**
     * Embed a picked image with the multimodal tower.
     *
     * `sofuu.ai.embedImage` takes a Uint8Array, not a path — passing a path
     * always threw "expected a Uint8Array", which the pane reported as
     * "the projector is not in this build". Read the bytes here and use
     * the native ABI, which reaches the same tower.
     */
    fun embedImageFile(path: String): java.util.concurrent.CompletableFuture<JSONObject?> {
        val bytes = try {
            java.io.File(path).readBytes()
        } catch (e: Throwable) {
            return java.util.concurrent.CompletableFuture.completedFuture(
                JSONObject().put("ok", false)
                    .put("error", "cannot read ${path.substringAfterLast('/')}")
            )
        }
        return engine.embedImage(bytes).handle { v, e ->
            when {
                e != null -> JSONObject().put("ok", false)
                    .put("error", e.message ?: "image embed failed")
                v == null -> JSONObject().put("ok", false)
                    .put("error", "image projector unavailable in this build")
                else -> JSONObject().put("ok", true).put("dim", v.size)
                    .put("vec", JSONArray().apply { v.forEach { put(it.toDouble()) } })
            }
        }
    }

    /** Cosine between an image vector and a text vector, both in img1's space. */
    fun imageTextSimilarity(imagePath: String, text: String): JSONObject? {
        val f = embedImageFile(imagePath).get(30, java.util.concurrent.TimeUnit.SECONDS)
        if (f?.optBoolean("ok") != true) return f
        val img = f.optJSONArray("vec") ?: return null
        // img1 is image-only; its bytes live in the same geometry as the
        // sem2-64 TEXT space (ai.rs) — "img1" is not a valid embed space and
        // made this call return null.
        val tv = engine.embedLocal(text, "sem2-64").get(5, java.util.concurrent.TimeUnit.SECONDS) ?: return null
        if (img.length() != tv.size) {
            return JSONObject().put("ok", false)
                .put("error", "image is ${img.length()}-dim, text is ${tv.size}-dim — different spaces")
        }
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in 0 until tv.size) {
            val a = img.optDouble(i)
            val b = tv[i].toDouble()
            dot += a * b; na += a * a; nb += b * b
        }
        if (na == 0.0 || nb == 0.0) return null
        return JSONObject().put("ok", true).put("similarity", dot / (Math.sqrt(na) * Math.sqrt(nb)))
            .put("dim", tv.size)
    }

    /**
     * Force a QuickJS collection.
     *
     * `__sofuu_gc` is installed by the runtime specifically for hosts that
     * fan out — a headless agent run, a batch of embeddings — and then
     * want the garbage back before a long-lived process keeps growing.
     * A phone is the worst case for that: one runtime for the whole
     * process lifetime, so nothing else would ever run the collector at
     * the right moment.
     */
    fun collectGarbage(): Boolean = engine.evalJsonOnEngine(
        "(function(){ try { globalThis.__sofuu_gc(); return { ok:true }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    ).get(30, java.util.concurrent.TimeUnit.SECONDS)?.optBoolean("ok") ?: false
}

