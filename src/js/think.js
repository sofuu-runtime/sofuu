/**
 * think.js — Sofuu Cognitive Thinking Layer
 *
 * Point 2: Reasoning Pattern Recall
 *   Stores successful reasoning chains in CMA. On future similar
 *   problems, recalls HOW the agent previously reasoned and injects
 *   that as a guide — giving non-thinking models learned reasoning.
 *
 * Point 3: Multi-Step Agent Thinking Loop (ReAct style)
 *   Think → Reflect → Answer, with each step stored in CMA so the
 *   agent's reasoning history is persistent and searchable.
 */

const OLLAMA_BASE = "http://localhost:11434";

// ── Low-level helpers ────────────────────────────────────────────────────────

export async function embed(text, model = "nomic-embed-text") {
    const res  = await sofuu.fetch(`${OLLAMA_BASE}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: text })
    });
    return new Float32Array((await res.json()).embeddings[0]);
}

export async function llm(messages, model = "qwen3.5:4b", think = false) {
    const res  = await sofuu.fetch(`${OLLAMA_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, messages, stream: false, think })
    });
    const data = await res.json();
    const raw  = data.message?.content ?? "";

    // Extract <think> block if present
    const thinkMatch = raw.match(/<think>([\s\S]*?)<\/think>/);
    const thought    = thinkMatch ? thinkMatch[1].trim() : null;
    const answer     = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    return { answer, thought };
}

// ── Point 2: Reasoning Pattern Recall ───────────────────────────────────────

/**
 * Store a successful reasoning chain in CMA.
 * Call this after a good answer to remember HOW you reasoned.
 */
export async function learnReasoning(mem, problem, reasoning, answer) {
    const patternText = `REASONING_PATTERN for "${problem}": ${reasoning}`;
    const vec = await embed(patternText);
    mem.remember(vec, patternText, "reasoning_pattern", 0);

    const answerText = `SOLVED: "${problem}" \u2192 ${answer}`;
    const aVec = await embed(answerText);
    mem.remember(aVec, answerText, "reasoning_pattern", 0);
}

/**
 * Recall similar past reasoning patterns for a new problem.
 * Returns an array of pattern strings to inject into the prompt.
 */
export async function recallReasoning(mem, problem, topK = 3) {
    const vec     = await embed(problem);
    const results = mem.recall(vec, topK * 2);

    return results
        .filter(r => r.role === "reasoning_pattern" && r.distance < 1.0)  // tighter: 1.0 not 1.1
        .slice(0, topK)
        .map(r => r.text);
}

// ── Point 3: Multi-Step Thinking Loop ───────────────────────────────────────

/**
 * Full thinking pipeline:
 *   Step 1 — Recall: pull relevant memories + past reasoning patterns
 *   Step 2 — Think:  generate explicit reasoning (with or without native think tokens)
 *   Step 3 — Answer: generate final answer grounded in reasoning + memory
 *   Step 4 — Store:  write user msg, thoughts, and answer back to CMA
 */
export async function thinkAndAnswer(mem, userMsg, {
    model     = "qwen3.5:4b",
    useNativeThink = true,   // use qwen's built-in <think> tokens
    verbose   = true
} = {}) {

    // ── Step 1: Recall relevant memories AND reasoning patterns
    const queryVec  = await embed(userMsg);
    const memories  = mem.recall(queryVec, 10);
    
    // Facts: user/assistant/thought entries, closest first. Facts take priority.
    const facts     = memories
        .filter(m => m.role !== "reasoning_pattern" && m.distance < 1.2)
        .slice(0, 6);
    const factsCtx  = facts
        .map(m => `- [${m.role}] ${m.text}`)
        .join("\n");

    // Only recall patterns if no strong factual hit exists (dist < 0.6)
    const hasClearFact = facts.some(m => m.distance < 0.6);
    const patterns     = hasClearFact ? [] : await recallReasoning(mem, userMsg, 2);
    const patternsCtx  = patterns.map(p => `- ${p}`).join("\n");

    if (verbose && factsCtx)    console.log(`  [think] recalling ${factsCtx.split("\n").length} facts`);
    if (verbose && patternsCtx) console.log(`  [think] recalling ${patterns.length} reasoning patterns`);

    // ── Step 2: Think step
    let reasoning = "";
    if (useNativeThink) {
        // Let qwen3.5 think natively via <think> tokens
        const thinkSys = [
            "You are a thoughtful assistant with persistent memory.",
            "IMPORTANT: Verified facts always override reasoning patterns.",
            factsCtx    ? `\nVerified facts from memory (TRUST THESE):\n${factsCtx}` : "",
            patternsCtx ? `\nPast reasoning patterns (use only if no fact contradicts):\n${patternsCtx}` : "",
            "\nThink carefully then answer. Always respond in English."
        ].filter(Boolean).join("\n");

        const { answer, thought } = await llm(
            [{ role: "system", content: thinkSys }, { role: "user", content: userMsg }],
            model,
            true   // enable native thinking
        );
        reasoning = thought ?? "";

        if (verbose && reasoning) {
            console.log("\n  ┌── Thinking ──────────────────────────────");
            reasoning.split("\n").slice(0, 6).forEach(l => console.log(`  │ ${l}`));
            if (reasoning.split("\n").length > 6) console.log("  │ ...");
            console.log("  └──────────────────────────────────────────");
        }

        // Store the thought in CMA
        if (reasoning) {
            const tVec = await embed(reasoning);
            mem.remember(tVec, `Thought about "${userMsg.substring(0,50)}": ${reasoning.substring(0,300)}`, "thought", 0);
        }

        // Answer already produced in same pass — return it
        await learnReasoning(mem, userMsg, reasoning, answer);
        mem.remember(await embed(userMsg), `User said: ${userMsg}`,     "user",      0);
        mem.remember(await embed(answer),  `Assistant said: ${answer}`, "assistant", 0);
        mem.flush();

        return { answer, reasoning };

    } else {
        // Fallback: explicit two-pass thinking for models without native think

        // Pass 1 — Reasoning
        const reasonSys = [
            "You are a reasoning engine. Given the question, write concise step-by-step reasoning.",
            "IMPORTANT: Verified facts take absolute priority over patterns.",
            factsCtx    ? `\nVerified facts (TRUST THESE FIRST):\n${factsCtx}` : "",
            patternsCtx ? `\nPast patterns (use only if no fact contradicts):\n${patternsCtx}` : "",
        ].filter(Boolean).join("\n");

        const { answer: rawReasoning } = await llm(
            [{ role: "system", content: reasonSys }, { role: "user", content: `Reason about: ${userMsg}` }],
            model, false
        );
        reasoning = rawReasoning;

        if (verbose) {
            console.log("\n  ┌── Reasoning ──────────────────────────────");
            reasoning.split("\n").slice(0, 5).forEach(l => console.log(`  │ ${l}`));
            console.log("  └──────────────────────────────────────────");
        }

        // Store the reasoning in CMA (full text)
        const tVec = await embed(reasoning);
        mem.remember(tVec, `Reasoning about "${userMsg}": ${reasoning}`, "thought", 0);

        // Pass 2 — Answer using reasoning + facts
        const answerSys = [
            "You are a helpful assistant. Always respond in English. Be concise.",
            `Your reasoning: ${reasoning}`,
            factsCtx ? `\nRelevant facts:\n${factsCtx}` : ""
        ].filter(Boolean).join("\n");

        const { answer } = await llm(
            [{ role: "system", content: answerSys }, { role: "user", content: userMsg }],
            model, false
        );

        await learnReasoning(mem, userMsg, reasoning, answer);
        mem.remember(await embed(userMsg), `User said: ${userMsg}`,     "user",      0);
        mem.remember(await embed(answer),  `Assistant said: ${answer}`, "assistant", 0);
        mem.flush();

        return { answer, reasoning };
    }
}
