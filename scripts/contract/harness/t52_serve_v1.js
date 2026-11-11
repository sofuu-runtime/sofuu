// t52_serve_v1.js — R5.4: `sofuu serve` is a first-class OpenAI-compatible
// surface — GET /v1/models lists the loaded model; POST /v1/chat/completions
// works with stream:true (valid SSE data: chunks + [DONE]) and without
// (OpenAI JSON shape). No Sofuu SDK, a standard client contract only.
//
// Requires SOFUU_CONTRACT_MODEL to name an existing real-weight GGUF and
// SOFUU_HOME to name a temporary home. No weights are downloaded. The small
// synthetic contract fixture is a parser fixture, not evidence of inference.
// Run: SOFUU_ALLOW_LOOPBACK=1 SOFUU_CONTRACT_MODEL=/path/to/model.gguf \
//      SOFUU_HOME=/path/to/temporary/home ./sofuu run scripts/contract/harness/t52_serve_v1.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
let srv = null;
setTimeout(() => {
    try { if (srv && srv.kill) srv.kill(); } catch (e) { /* best-effort cleanup */ }
    console.error("  ❌ WATCHDOG: harness overran 150 s"); process.exit(1);
}, 150000);

const P = (s) => JSON.parse(s);

// One HTTP round-trip helper against the serve process (curl keeps this a
// standards check — the assertion is about the wire, not our client).
function curl(args) {
    return new Promise((resolve) => {
        let out = "";
        sofuu.spawn({
            command: "/usr/bin/curl",
            args: ["-s", "-S", "-f", "-m", "45", "-H", "Authorization: Bearer t52-contract-token"].concat(args),
            onStdout: (c) => { out += String(c); },
            onExit: (code) => resolve({ code: code, out: out }),
        });
    });
}

async function main() {
    const modelPath = process.env.SOFUU_CONTRACT_MODEL || "";
    if (!modelPath || !/\.gguf$/i.test(modelPath) || !sofuu.fs.existsSync(modelPath) ||
        /(^|\/)sofuu-fixture\.gguf$/i.test(modelPath)) {
        console.error("BLOCKED: set SOFUU_CONTRACT_MODEL to an existing compatible real-weight GGUF, not the synthetic fixture");
        process.exit(2);
    }
    if (!process.env.SOFUU_HOME) {
        console.error("BLOCKED: set SOFUU_HOME to an isolated temporary directory");
        process.exit(2);
    }
    if (process.env.SOFUU_SERVE_TOKEN) {
        console.error("BLOCKED: unset SOFUU_SERVE_TOKEN so the harness can use its own test token");
        process.exit(2);
    }
    const binary = process.env.SOFUU_BIN || "./sofuu";
    const port = 27800 + (Date.now() % 800);
    const modelId = "t52-contract-real-model";
    srv = sofuu.spawn({
        command: binary,
        args: ["serve", "--model", modelPath, "--model-id", modelId, "--port", String(port), "--host", "127.0.0.1", "--token", "t52-contract-token"],
        onStdout: () => {},
        onStderr: () => {},
        onExit: () => {},
    });
    const base = "http://127.0.0.1:" + port;
    // Wait for the listener.
    let up = false;
    for (let i = 0; i < 80 && !up; i++) {
        await new Promise((r) => setTimeout(r, 250));
        try {
            const probe = await curl([base + "/v1/models"]);
            up = probe.out.indexOf('"object"') >= 0 || probe.out.indexOf('"data"') >= 0;
        } catch (e) { up = false; }
    }
    assert("serve came up with the configured real model", up);

    try {
        if (up) {
            // ── T5.2: GET /v1/models ─────────────────────────────────────
            const r1 = await curl([base + "/v1/models"]);
            assert("GET /v1/models returns 200", r1.code === 0 && r1.out.length > 0);
            let m1 = null;
            try { m1 = P(r1.out); } catch (e) {}
            assert("models payload is the OpenAI shape ({object:'list', data:[...]})",
                m1 && m1.object === "list" && Array.isArray(m1.data) && m1.data.length >= 1
                && m1.data.some((model) => model.id === modelId && model.object === "model"));

            // ── T5.3: POST /v1/chat/completions stream:true ─────────────
            const r2 = await curl(["-X", "POST", base + "/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", JSON.stringify({ model: modelId, stream: true,
                    messages: [{ role: "user", content: "hi" }], max_tokens: 8 })]);
            const sse = r2.out.split("\n").filter((l) => l.indexOf("data:") === 0);
            assert("stream: yields SSE data: chunks", sse.length >= 2);
            let sawDone = false, validChunks = 0, textChunks = 0;
            for (const line of sse) {
                const body = line.slice(5).trim();
                if (body === "[DONE]") { sawDone = true; continue; }
                try {
                    const fr = P(body);
                    if (fr.choices && fr.choices.length && "delta" in fr.choices[0]) {
                        validChunks++;
                        if (typeof fr.choices[0].delta.content === "string" && fr.choices[0].delta.content.length > 0) textChunks++;
                    }
                } catch (e) {}
            }
            assert("stream: request succeeded and yielded OpenAI delta frames", r2.code === 0 && validChunks >= 1);
            assert("stream: real model returned nonempty text", textChunks >= 1);
            assert("stream: terminated by [DONE]", sawDone);

            // ── T5.4: POST /v1/chat/completions (JSON) ───────────────────
            const r3 = await curl(["-X", "POST", base + "/v1/chat/completions",
                "-H", "Content-Type: application/json",
                "-d", JSON.stringify({ model: modelId,
                    messages: [{ role: "user", content: "hi" }], max_tokens: 8 })]);
            let j3 = null;
            try { j3 = P(r3.out); } catch (e) {}
            assert("json: OpenAI completion shape",
                r3.code === 0 && j3 && Array.isArray(j3.choices) && j3.choices.length >= 1
                && j3.choices[0].message && typeof j3.choices[0].message.role === "string"
                && typeof j3.choices[0].message.content === "string"
                && j3.choices[0].message.content.length > 0
                && j3.object === "chat.completion");
        }
    } finally {
        try { srv.kill && srv.kill(); } catch (e) {}
        // Wait out the port; the suite runs other serve cases later.
        await new Promise((r) => setTimeout(r, 300));
    }

    console.log(`\n=== t52 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => { console.error("FAIL t52:", (e && e.message) || e); process.exit(1); });
