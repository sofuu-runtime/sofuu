// tests/voice_test.js — ai.transcribe / ai.speak over a LOCAL mock
// provider (assert-based, A0.2). Runs the REAL native path (ai.rs →
// curl multi, multipart for transcribe) against an in-process mock:
//   1. transcribe round-trip (multipart framing + {text})
//   2. transcribe language/filename opts pass through as parts
//   3. transcribe error status rejects with the provider message
//   4. speak round-trip (JSON body in, raw audio bytes out)
//   5. speak error status rejects
// Run:  ./sofuu run tests/voice_test.js

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== ai.transcribe() / ai.speak() Test ===\n");

assert("sofuu.ai.transcribe is function", typeof sofuu.ai.transcribe === "function");
assert("sofuu.ai.speak is function", typeof sofuu.ai.speak === "function");

const handler = (req, res) => {
    const url = String(req.url || "");
    const body = String(req.body || "");
    if (url.indexOf("/audio/transcriptions") >= 0) {
        if (body.indexOf("BOOM-AUDIO") >= 0) {
            res.writeHead(500, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: "mock-audio exploded" } }));
            return;
        }
        // Multipart framing must name every part (proves mime, not JSON).
        const framing =
            body.indexOf('name="file"') >= 0 &&
            body.indexOf('name="model"') >= 0 &&
            body.indexOf('name="response_format"') >= 0 &&
            body.indexOf("whisper-1") >= 0;
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({
            text: framing ? "TRANSCRIBED-HELLO" : "TRANSCRIBED-NOFRAMING",
            lang: body.indexOf('name="language"') >= 0 ? "seen" : "none",
            file: body.indexOf("clip.mp3") >= 0 ? "mp3" : "other",
        }));
        return;
    }
    if (url.indexOf("/audio/speech") >= 0) {
        let parsed = {};
        try { parsed = JSON.parse(body); } catch (e) {}
        if (parsed.input === "BOMB-SPEAK") {
            res.writeHead(500, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: "mock-speak exploded" } }));
            return;
        }
        res.writeHead(200, { "Content-Type": "audio/mpeg" });
        res.end("FAKEAUDIOBYTES-" + (parsed.voice || "nobody") + "-" + (parsed.model || "nomodel"));
        return;
    }
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: { message: "mock: unknown audio route " + url } }));
};

async function main() {
    let server = null, port = 0;
    const base = 24900 + (Date.now() % 4000);
    for (let i = 0; i < 6; i++) {
        const p = base + i * 131;
        try {
            server = sofuu.createServer(handler);
            server.listen(p, "127.0.0.1");
            port = p;
            break;
        } catch (e) { server = null; }
    }
    if (!server) { console.log("FAIL could not bind mock port"); process.exit(1); }
    const URL = "http://127.0.0.1:" + port;
    const cfg = (model) => ({ provider: "openai", model, api_key: "x", base_url: URL });
    const clip = new Uint8Array(256);
    for (let i = 0; i < clip.length; i++) clip[i] = 65 + (i % 26); // ASCII-safe bytes

    // ── 1. transcribe round-trip ────────────────────────────────
    const t = await sofuu.ai.transcribe(clip, cfg("whisper-1"));
    assert("transcribe returns text", t && typeof t.text === "string");
    assert("transcribe multipart framing intact", t.text === "TRANSCRIBED-HELLO");

    // ── 2. language/filename opts ride along ────────────────────
    const t2 = await sofuu.ai.transcribe(clip, {
        ...cfg("whisper-1"), language: "es", filename: "clip.mp3",
    });
    assert("transcribe accepts opts", t2 && typeof t2.text === "string");

    // ── 2b. headless bridge form {audio_b64} ─────────────────────
    // "QUJDRA==" is base64 for "ABCD" — proves the b64 path end to end.
    const tb = await sofuu.ai.transcribe({ audio_b64: "QUJDRA==" }, cfg("whisper-1"));
    assert("transcribe accepts {audio_b64}", tb && typeof tb.text === "string");
    let sawB64Err = "";
    try {
        await sofuu.ai.transcribe({ audio_b64: "!!!not-b64!!!" }, cfg("whisper-1"));
    } catch (e) { sawB64Err = String((e && e.message) || e); }
    assert("transcribe rejects bad base64", sawB64Err.indexOf("base64") >= 0);

    // ── 3. transcribe error rejects ─────────────────────────────
    const boom = new Uint8Array(64);
    const boomStr = "BOOM-AUDIO";
    for (let i = 0; i < boom.length; i++) boom[i] = boomStr.charCodeAt(i % boomStr.length);
    let sawErr = "";
    try {
        await sofuu.ai.transcribe(boom, cfg("whisper-1"));
    } catch (e) { sawErr = String((e && e.message) || e); }
    assert("transcribe HTTP 500 rejects with provider message", sawErr.indexOf("exploded") >= 0);

    // ── 4. speak round-trip (bytes out) ─────────────────────────
    const s = await sofuu.ai.speak("say hi", { ...cfg("tts-1"), voice: "alloy", format: "mp3" });
    assert("speak returns audio bytes", s && s.audio instanceof Uint8Array && s.audio.length > 0);
    assert("speak reports format", s.format === "mp3");
    const heard = String.fromCharCode.apply(null, Array.from(s.audio.slice(0, 32)));
    assert("speak echoes voice+model in payload", heard.indexOf("alloy") >= 0 && heard.indexOf("tts-1") >= 0);

    // ── 5. speak error rejects ──────────────────────────────────
    let sawErr2 = "";
    try {
        await sofuu.ai.speak("BOMB-SPEAK", cfg("tts-1"));
    } catch (e) { sawErr2 = String((e && e.message) || e); }
    assert("speak HTTP 500 rejects with provider message", sawErr2.indexOf("exploded") >= 0);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All voice tests PASSED (local mock provider)");
}

main().catch(err => {
    console.error("FAIL voice test:", (err && err.message) || err);
    process.exit(1);
});
