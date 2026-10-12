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
class RuntimeSurface(private val engine: SofuuEngine) {

    // ── Memory (CMA brain) ───────────────────────────────────────

    /**
     * Count, backend id and recall health for the active brain, or null
     * when memory is off or the artifact is unavailable.
     */
    fun brainInfo(): JSONObject? = engine.evalJson(
        """
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable (memory off or artifact missing)' };
          return {
            ok: true,
            count: b.cma.count(),
            backend: b.backend ? b.backend.id : null,
            dim: b.backend ? b.backend.dim : null,
            semantic: !!b.sem
          };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """.trimIndent()
    )

    /**
     * Nearest memories for [query].
     *
     * Embeds with the brain's own backend rather than passing a raw
     * vector, so the query lands in the same space the memories were
     * stored in — mixing spaces is the one way to get confidently wrong
     * recall results.
     */
    fun brainRecall(query: String, k: Int = 8): JSONObject? = engine.evalJson(
        """
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var v = null;
          try { v = sofuu.ai.embedLocalSemantic ? sofuu.ai.embedLocalSemantic(${SofuuEngine.quote(query)}) : null; } catch (e1) {}
          if (!v) { try { v = sofuu.ai.embedLocal(${SofuuEngine.quote(query)}); } catch (e2) {} }
          if (!v) return { ok:false, error:'no embedder available' };
          var hits = b.cma.recall(v, ${k.coerceIn(1, 64)}) || [];
          return { ok:true, hits: Array.isArray(hits) ? hits : Array.prototype.slice.call(hits) };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """.trimIndent()
    )

    /** Drop one memory by id. The id comes from a recall hit. */
    fun brainForget(id: Int): JSONObject? = engine.evalJson(
        """
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          b.cma.forget($id);
          return { ok:true, count: b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """.trimIndent()
    )

    /** Merge and de-duplicate stored memories. Irreversible. */
    fun brainConsolidate(): JSONObject? = engine.evalJson(
        """
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var before = b.cma.count();
          var n = b.cma.consolidate();
          return { ok:true, merged:n, before:before, after:b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """.trimIndent()
    )

    /** Run one Ebbinghaus decay/consolidation tick. */
    fun brainDecay(): JSONObject? = engine.evalJson(
        """
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var before = b.cma.count();
          b.cma.decayTick();
          return { ok:true, before:before, after:b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """.trimIndent()
    )

    /** Identity of the active embedding spaces: hash + semantic. */
    fun embeddingInfo(): JSONObject? = engine.evalJson(
        "(function(){ try { return { ok:true, info: sofuu.memory.embeddingInfo() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    )

    // ── ML gates ─────────────────────────────────────────────────

    /** Gate states, working-set counters, online-learner status. */
    fun mlState(): JSONObject? = engine.evalJson(
        "(function(){ try { return { ok:true, info: sofuu.ml.info(), workset: sofuu.ml.workset() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    )

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
        return engine.evalJson(
            "(function(){ try { return sofuu.ml.feedback(${SofuuEngine.quote(payload.toString())}); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
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
        return engine.evalJson(
            "(function(){ try { return sofuu.ml.supervisor.check(${SofuuEngine.quote(payload.toString())}); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    /** The loop-boundary verdict: spinning, over-budget, wasted effort. */
    fun mlLoop(run: String, step: Int, budget: Int): JSONObject? {
        val payload = JSONObject().put("run", run).put("step", step).put("budget", budget)
        return engine.evalJson(
            "(function(){ try { return sofuu.ml.supervisor.loop(${SofuuEngine.quote(payload.toString())}); } " +
                "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
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
    fun spaces(): JSONObject? = engine.evalJson(
        "(function(){ try { return { ok:true, info: sofuu.ai.embedInfo(), v2: sofuu.ai.embedInfoV2() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    )

    /** Encode a message with the model's own chat template. */
    fun applyChatTemplate(messages: JSONArray): JSONObject? = engine.evalJson(
        "(function(){ try { return { ok:true, text: sofuu.ai.applyChatTemplate(${SofuuEngine.quote(messages.toString())}) }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    )

    // ── Agents ───────────────────────────────────────────────────

    /** Registered agents with their tools, budgets and sub-agent fan-out. */
    fun agents(): JSONArray? = engine.evalJson(
        "(function(){ try { return { ok:true, agents: sofuu.agent.list() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
    )?.optJSONArray("agents")
}