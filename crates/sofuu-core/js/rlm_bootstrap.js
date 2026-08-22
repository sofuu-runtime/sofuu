/* rlm_bootstrap.js — sandbox API shim (PLAN-RLM R0).
 *
 * Runs ONCE inside a fresh hermetic QuickJS context where the only
 * pre-existing globals are the __hx_* natives. It wraps them into the
 * friendly API the model is told about (see rlm/episode.rs SYSTEM_PROMPT).
 * No logic beyond argument plumbing + the llm() answer cache lives here —
 * the host owns the context, the chunk map and every budget.
 *
 * Determinism note: a snippet that calls llm() SUSPENDS and is later
 * re-executed from the top once answers arrive. Wall-clock time and true
 * randomness would change between attempts and make re-runs diverge, so
 * Date/Math.random are replaced with deterministic stubs, and the host
 * calls __rlmResetDeterminism() before EVERY snippet attempt — attempt N+1
 * replays the same clock tick / rng sequence as attempt N. Every public
 * name is then frozen (writable:false, configurable:false) so model code
 * cannot redefine the whitelist mid-episode. */
(function () {
    "use strict";
    var g = globalThis;
    if (typeof g.__LLM_CACHE !== "object" || g.__LLM_CACHE === null) g.__LLM_CACHE = {};
    if (typeof g.__TOOL_CACHE !== "object" || g.__TOOL_CACHE === null) g.__TOOL_CACHE = {};

    function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n) + "…" : s; }

    function toRegExp(re) {
        if (re instanceof RegExp) {
            var flags = "g" + (re.ignoreCase ? "i" : "") + (re.multiline ? "m" : "") + (re.unicode ? "u" : "");
            return new RegExp(re.source, flags);
        }
        return new RegExp(String(re), "g");
    }

    /* [{chunk, index, line, text}...] across all chunks. */
    function grep(re, maxHits) {
        if (maxHits === undefined || maxHits === null) maxHits = 50;
        var rx = toRegExp(re);
        var hits = [];
        var n = __hx_count();
        for (var i = 0; i < n && hits.length < maxHits; i++) {
            var text = __hx_chunk(i);
            rx.lastIndex = 0;
            var m;
            while ((m = rx.exec(text)) !== null) {
                var line = 1;
                for (var k = 0; k < m.index; k++) if (text.charCodeAt(k) === 10) line++;
                var lineStart = text.lastIndexOf("\n", m.index) + 1;
                var lineEnd = text.indexOf("\n", m.index);
                if (lineEnd < 0) lineEnd = text.length;
                hits.push({
                    chunk: i, index: m.index, line: line,
                    text: trunc(m[0], 160),
                    line_text: trunc(text.slice(lineStart, lineEnd), 160)
                });
                if (m[0].length === 0) rx.lastIndex++;
                if (hits.length >= maxHits) break;
            }
        }
        __hx_emit("grep", String(rx) + " → " + hits.length + " hit(s)");
        return hits;
    }

    /* Recursive batch call. Answers come back through __LLM_CACHE after a
     * suspension; a cache miss throws the \0RLM_SUSPEND sentinel via
     * __hx_llmBatch and NEVER returns. */
    function llmBatch(prompts) {
        if (!Array.isArray(prompts)) prompts = [String(prompts)];
        prompts = prompts.map(String);
        var key = JSON.stringify(prompts);
        if (Object.prototype.hasOwnProperty.call(g.__LLM_CACHE, key)) return g.__LLM_CACHE[key];
        __hx_llmBatch(key);
        throw new Error("rlm: suspend sentinel did not fire");
    }

    /* PLAN-AGENTS A4: host tools (the agent's own whitelist) with the same
     * suspension model as llm(). toolBatch([{name, args}...]) suspends on a
     * cache miss via __hx_tool; the host executes each call through the
     * agent's normal tool path and re-runs the snippet with the string
     * results in __TOOL_CACHE. */
    function toolBatch(calls) {
        if (!Array.isArray(calls)) calls = [{ name: String(calls), args: {} }];
        calls = calls.map(function (c) {
            return {
                name: String(c && c.name),
                args: (c && c.args && typeof c.args === "object") ? c.args : {}
            };
        });
        var key = JSON.stringify(calls);
        if (Object.prototype.hasOwnProperty.call(g.__TOOL_CACHE, key)) return g.__TOOL_CACHE[key];
        __hx_tool(key);
        throw new Error("rlm: tool suspend sentinel did not fire");
    }

    g.len = function () { return __hx_len(); };
    g.count = function () { return __hx_count(); };
    g.chunk = function (i, a, b) {
        var t = String(__hx_chunk(i));
        if (a === undefined && b === undefined) return t;
        return t.slice(a || 0, b);
    };
    g.peek = function (a, b) { return __hx_peek(a, b); };
    g.grep = grep;
    g.lines = function (i, from, to) {
        var ls = String(__hx_chunk(i)).split("\n");
        return ls.slice(from || 0, to === undefined ? ls.length : to).join("\n");
    };
    g.llmBatch = llmBatch;
    g.llm = function (p) { return llmBatch([p])[0]; };
    g.toolBatch = toolBatch;
    g.tool = function (name, args) { return toolBatch([{ name: name, args: args || {} }])[0]; };
    g.final = function (answer) {
        __hx_final(String(answer));
        throw new Error("rlm: final sentinel did not fire");
    };
    g.emit = function (tag, text) { __hx_emit(String(tag), trunc(text, 4000)); };

    /* ── Deterministic time (G1) ───────────────────────────────────
     * A suspended snippet is re-executed from the top after its llm()
     * answers arrive; wall clock / real randomness would make the re-run
     * diverge from the original attempt. Date and Math.random are therefore
     * stubs driven by counters the host reset before every attempt. */
    var CLOCK0 = 1700000000000;
    var RNG0 = 0x9e3779b9;
    var clock = CLOCK0;
    var rng = RNG0;
    function FakeDate() { this._t = clock++; }
    FakeDate.now = function () { return clock++; };
    FakeDate.prototype.getTime = function () { return this._t; };
    FakeDate.prototype.valueOf = function () { return this._t; };
    FakeDate.prototype.toISOString = function () { return "2023-11-14T22:13:20.000Z"; };
    g.Date = FakeDate;
    Math.random = function () {
        /* xorshift32, fixed seed — uniform enough for snippet logic. */
        var x = rng;
        x ^= x << 13; x >>>= 0;
        x ^= x >>> 17;
        x ^= x << 5; x >>>= 0;
        rng = x === 0 ? RNG0 : x;
        return rng / 4294967296;
    };
    /* Host hook: reset the clock/rng counters before every snippet
     * attempt, so a suspension re-run replays them identically. */
    g.__rlmResetDeterminism = function () { clock = CLOCK0; rng = RNG0; };

    /* ── Frozen whitelist (G1) ───────────────────────────────────────
     * Model code must not redefine the API mid-episode (e.g. shadowing
     * final() to swallow the sentinel). Assignments to these names
     * silently no-op in sloppy snippets / throw a catchable TypeError in
     * strict ones. __LLM_CACHE / __TOOL_CACHE are deliberately NOT frozen:
     * the host reassigns them before every eval to deliver fresh llm()
     * answers and tool() results. */
    [
        "len", "count", "chunk", "peek", "grep", "lines", "llm", "llmBatch",
        "tool", "toolBatch", "final", "emit", "__rlmResetDeterminism",
        "__hx_len", "__hx_count", "__hx_chunk", "__hx_peek", "__hx_llmBatch",
        "__hx_tool", "__hx_final", "__hx_emit"
    ].forEach(function (name) {
        Object.defineProperty(g, name, {
            value: g[name], writable: false, enumerable: false, configurable: false
        });
    });
    Object.freeze(Math);
})();
