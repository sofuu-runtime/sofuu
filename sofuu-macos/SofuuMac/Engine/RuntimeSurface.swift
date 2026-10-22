import Foundation

/// Direct access to the runtime's cognitive and vector surfaces.
///
/// Everything here runs inside the engine through `SofuuEngine`'s query
/// channel, on the engine thread, as one evaluated expression — so it obeys
/// the same one-runtime-one-thread rule as every other engine call.
///
/// The brain handle is never opened here. `sofuu.agent.brainFor` hands out
/// the ONE cached CMA handle for the active workspace, and a QTSQ flush
/// rewrites the whole file, so a second long-lived handle would silently
/// clobber the first on every write. Everything routes through `brainFor`.
struct RuntimeSurface {

    let engine: SofuuEngine

    // ── Memory (CMA brain) ─────────────────────────────────────────────

    /// Count, backend id and recall health for the active brain, or nil
    /// when memory is off or the artifact is unavailable.
    func brainInfo() async -> JSONObject? {
        await engine.evalJson("""
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
        """)
    }

    /// Nearest memories for `query`.
    ///
    /// Embeds with the brain's own backend rather than passing a raw vector,
    /// so the query lands in the same space the memories were stored in —
    /// mixing spaces is the one way to get confidently wrong recall results.
    func brainRecall(_ query: String, k: Int = 8) async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var v = null;
          try { v = sofuu.ai.embedLocalSemantic ? sofuu.ai.embedLocalSemantic(\(jsonQuote(query))) : null; } catch (e1) {}
          if (!v) { try { v = sofuu.ai.embedLocal(\(jsonQuote(query))); } catch (e2) {} }
          if (!v) return { ok:false, error:'no embedder available' };
          var hits = b.cma.recall(v, \(max(1, min(k, 64)))) || [];
          return { ok:true, hits: Array.isArray(hits) ? hits : Array.prototype.slice.call(hits) };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    /// Drop one memory by id. The id comes from a recall hit.
    func brainForget(_ id: Int) async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          b.cma.forget(\(id));
          return { ok:true, count: b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    /// Merge and de-duplicate stored memories. Irreversible.
    func brainConsolidate() async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var before = b.cma.count();
          var n = b.cma.consolidate();
          return { ok:true, merged:n, before:before, after:b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    /// Run one Ebbinghaus decay/consolidation tick.
    func brainDecay() async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try {
          var b = sofuu.agent.brainFor({ memory: 'auto' });
          if (!b || !b.cma) return { ok:false, error:'brain unavailable' };
          var before = b.cma.count();
          b.cma.decayTick();
          return { ok:true, before:before, after:b.cma.count() };
        } catch(e) { return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    // ── ML gates ───────────────────────────────────────────────────────

    /// Gate states, working-set counters, online-learner status.
    func mlState() async -> JSONObject? {
        await engine.evalJson(
            "(function(){ try { return { ok:true, info: sofuu.ml.info(), workset: sofuu.ml.workset() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    /// Teach a gate from an outcome.
    ///
    /// The runtime's contract is narrow and exact (`ml/mod.rs:344`): the
    /// payload must carry `model:"supervisor"`, and `kind` must be one of
    /// `outcome`, `wrong` or `wasted`. Anything else is refused silently,
    /// so the fields are set here rather than left to a caller to guess.
    ///
    /// `wrong` and `wasted` are gold labels — they name the most recent
    /// call and count double against a proxy's weight, which makes them the
    /// right choice for a turn you know was bad.
    ///
    /// Returns the engine's own `{ok:false}` when it declines, which is the
    /// honest answer and not an error to swallow.
    func mlFeedback(kind: String, run: String = "", step: Int = 0, wasted: Bool = false, conf: Double = 1.0) async -> JSONObject? {
        var payload: JSONObject = ["model": "supervisor", "kind": kind]
        if !run.isEmpty { payload["run"] = run }
        if step > 0 { payload["step"] = step }
        if wasted { payload["wasted"] = true }
        payload["conf"] = conf
        return await engine.evalJson(
            "(function(){ try { return sofuu.ml.feedback(\(jsonQuote(jsonEncode(payload)))); } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    /// Ask a gate for its verdict on a proposed action, without running it.
    ///
    /// `supervisor.check` is the pre-tool-call checkpoint; it returns
    /// `{ok, reason, nudge, score, source}` where `source` is `rule` or
    /// `model`. Exposed because it is the only way to see WHY a call was
    /// nudged or blocked — the agent only ever sees the nudge, never the
    /// verdict behind it.
    func mlCheck(tool: String, args: JSONObject) async -> JSONObject? {
        let payload: JSONObject = ["tool": tool, "args": args]
        return await engine.evalJson(
            "(function(){ try { return sofuu.ml.supervisor.check(\(jsonQuote(jsonEncode(payload)))); } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    // ── Vectors ────────────────────────────────────────────────────────

    /// Which embedding spaces this build can actually use.
    ///
    /// `embedImage` in particular needs its projector artifact, and that is
    /// missing on most builds rather than failing at call time.
    func spaces() async -> JSONObject? {
        await engine.evalJson(
            "(function(){ try { return { ok:true, info: sofuu.ai.embedInfo(), v2: sofuu.ai.embedInfoV2() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    /// Cosine similarity between two strings, embedding each with the
    /// space named (nil = the brain's default, hash-768).
    ///
    /// Computed host-side from the two vectors rather than through
    /// `sofuu.ai.similarity`, so the same space is used for both sides —
    /// mixing spaces gives a number that looks fine and means nothing.
    func similarity(_ a: String, _ b: String, space: String? = nil) -> JSONObject? {
        guard let va = SofuuBridge.embedLocal(a, space: space),
              let vb = SofuuBridge.embedLocal(b, space: space),
              va.count == vb.count, !va.isEmpty else { return nil }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in va.indices {
            dot += Double(va[i]) * Double(vb[i])
            na += Double(va[i]) * Double(va[i])
            nb += Double(vb[i]) * Double(vb[i])
        }
        guard na > 0, nb > 0 else { return nil }
        return [
            "ok": true,
            "similarity": dot / (na.squareRoot() * nb.squareRoot()),
            "dim": va.count,
            "space": space ?? "(default)",
        ]
    }

    /// Embed a picked image with the multimodal tower.
    ///
    /// `embedImage` needs its projector artifact, which most builds do not
    /// carry — so this reports unavailability rather than returning a zero
    /// vector that would silently never match anything.
    func embedImage(at url: URL) async -> JSONObject? {
        guard let data = try? Data(contentsOf: url) else {
            return ["ok": false, "error": "could not read that image"]
        }
        guard let vec = SofuuBridge.embedImage([UInt8](data)) else {
            return ["ok": false, "error": "the image projector is not in this build"]
        }
        // Doubles, not Floats: JSON has no float type, and a downstream
        // `as? [Double]` on a [Float] is NOT element-wise — it is nil, so
        // the vector silently looked unusable.
        return ["ok": true, "dim": vec.count, "vec": vec.map(Double.init)]
    }

    /// Cosine between an image vector and a text vector, both in img1's space.
    func imageTextSimilarity(imageURL: URL, text: String) async -> JSONObject? {
        // An unavailable projector is reported as-is rather than swallowed
        // into nil, so the caller can say why instead of "no answer".
        guard let embedded = await embedImage(at: imageURL) else {
            return ["ok": false, "error": "could not embed that image"]
        }
        guard let ok = embedded["ok"] as? Bool, ok, let img = embedded["vec"] as? [Double] else {
            return embedded
        }
        guard let tv = SofuuBridge.embedLocal(text, space: "img1") else { return nil }
        guard img.count == tv.count else {
            return ["ok": false, "error": "image is \(img.count)-dim, text is \(tv.count)-dim — different spaces"]
        }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in tv.indices {
            let a = img[i], b = Double(tv[i])
            dot += a * b; na += a * a; nb += b * b
        }
        guard na > 0, nb > 0 else { return nil }
        return ["ok": true, "similarity": dot / (na.squareRoot() * nb.squareRoot()), "dim": tv.count]
    }

    /// Force a QuickJS collection.
    ///
    /// `__sofuu_gc` is installed by the runtime specifically for hosts that
    /// fan out and then want the garbage back before a long-lived process
    /// keeps growing. A phone is the worst case for that: one runtime for
    /// the whole process lifetime, so nothing else would ever run the
    /// collector at the right moment.
    @discardableResult
    func collectGarbage() async -> Bool {
        let r = await engine.evalJson(
            "(function(){ try { globalThis.__sofuu_gc(); return { ok:true }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
        return (r?["ok"] as? Bool) ?? false
    }

    // ── Agents ─────────────────────────────────────────────────────────

    /// Registered agents with their tools, budgets and sub-agent fan-out.
    func agents() async -> [JSONObject] {
        guard let r = await engine.evalJson(
            "(function(){ try { return { ok:true, agents: sofuu.agent.list() }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        ), let list = r["agents"] as? [JSONObject] else { return [] }
        return list
    }

    /// Register an agent from the phone.
    ///
    /// The shape is the runtime's, not the UI's: `tools` takes `'web'`,
    /// `'code'`, or `{builtin:"name"}` for anything already in
    /// `sofuu.tools.TOOLS` — which, after host-tool registration, includes
    /// the phone's own tools. `budget` is a guardrail rather than a wall:
    /// breaching `maxSteps` salvages an answer and reports `budget_steps`
    /// rather than throwing, so a generous default is the safe default.
    func defineAgent(
        name: String,
        system: String,
        tools: [String],
        budget: JSONObject = [:]
    ) async -> JSONObject? {
        // 'web' and 'code' expand to groups; anything else must name a
        // registered builtin, which after host-tool registration includes
        // the phone's own tools.
        let toolDefs: [Any] = tools.map { t -> Any in
            if t == "web" || t == "code" { return t }
            return ["builtin": t]
        }
        var def: JSONObject = [
            "name": name,
            "system": system,
            "tools": toolDefs,
        ]
        if !budget.isEmpty { def["budget"] = budget }
        return await engine.evalJson("""
        (function(){ try {
          var h = sofuu.agent.define(JSON.parse(\(jsonQuote(jsonEncode(def)))));
          return { ok:true, name: h.name };
        } catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    /// Run an agent headlessly and wait for its answer.
    ///
    /// Goes through the same `agent.run` the chat turn uses, so the tools,
    /// the permission profile, the budget and the brain recall all behave
    /// identically — the only difference is that the result is collected
    /// instead of streamed.
    func runAgent(name: String, task: String) async -> JSONObject? {
        await engine.queryFuture("""
        (function(){ var r = sofuu.agent.run(\(jsonQuote(name)), \(jsonQuote(task)));
          return r.then(function(v){ return { ok:true, answer:v.answer, steps:v.steps,
            stopped:v.stopped, usage:v.usage, runId:v.runId }; },
            function(e){ return { ok:false, error:String(e&&e.message||e) }; }); })()
        """, timeout: 300)
    }

    /// Caps for a model id, without switching to it.
    func modelCaps(_ model: String) async -> JSONObject? {
        await engine.evalJson(
            "(function(){ try { return { ok:true, caps: sofuu.ai.modelCaps(\(jsonQuote(model))) }; } " +
            "catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()"
        )
    }

    /// Read the real caps out of a GGUF header, so a model opened on the
    /// phone is configured the same way the CLI's `/model x.gguf` does.
    func localModelCaps(_ path: String) async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try { var c = sofuu.ai.modelCaps(\(jsonQuote(path)));
          return { ok:true, caps:c||{} }; }
        catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()
        """)
    }

    /// Confirm the engine can actually open a model before the app points
    /// its config at it — `modelCaps` only parses the header, so this also
    /// runs one short generation, which is what surfaces a model whose
    /// tensors are the wrong shape or unsupported.
    func probeLocalModel(_ path: String, prompt: String = "hi", maxTokens: Int = 8) async -> JSONObject? {
        await engine.evalJson("""
        (function(){ try { var t = sofuu.ai.inferLocal(\(jsonQuote(path)), \(jsonQuote(prompt)),
            {maxTokens:\(max(1, min(maxTokens, 64))), temperature:0});
          return { ok:true, text: String(t == null ? '' : t) }; }
        catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()
        """, timeout: 180)
    }
}