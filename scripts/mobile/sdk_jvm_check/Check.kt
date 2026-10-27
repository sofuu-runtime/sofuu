// Desktop JVM check for the Sofuu Android SDK wrapper (SofuuBridge.kt).
//
// Run by scripts/mobile/sdk_jvm_check.sh: that script compiles the SAME
// jni_bridge.c the NDK builds for Android, but for the host OS, links it
// against dist/libsofuu, compiles the real Kotlin wrapper, and then runs
// this file. No device and no network are involved.
//
// What it pins down:
//   - the UTF-8 boundary (JNI "UTF-8" is CESU-8 for non-BMP text: an emoji
//     in a prompt used to arrive at serde_json as invalid UTF-8, so the
//     engine saw no arguments at all)
//   - streaming deltas and the single terminal `done` event
//   - mid-flight cancel via the listener returning false
//   - streamLocal's positional-argument contract
//   - host tools: sync results, and deferred ones resolved from another
//     thread (uv_async_send → the engine loop resumes the turn)
//   - close(): the wrapper goes inert and releases its registrations

import com.sofuu.runtime.SofuuBridge
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import kotlin.system.exitProcess

private var failures = 0
private fun check(cond: Boolean, what: String) {
    if (cond) println("  ok  : $what") else { failures++; println("  FAIL: $what") }
}

private fun jsonQuote(s: String) = org.json.JSONObject.quote(s)

// Generic RS256 fixture shared with the Rust crypto tests. The token is
// signed for https://app.example.com and expires in 2100.
private const val TEST_JWKS = """
{"keys":[{
  "kty":"RSA",
  "n":"ntahXsbkEUCoSvUhRJtDJSf_XE4uHz4KDeGeSRblbbrhIJdHfxl8vVWMGRI9VHXDZbjBkBAKsOjRM_cRNO2n7K9TksqSAnmc4rDfzMIc4wAIYiWYW5txI23Zk26FigaFS-Px-KkBKSmH_2tEB9D-Ens8avhnciaT1b3Z1Q425vgG8MNCOHaYhmIuMtNZRYGSQe9AYfirdhGn8FqyhcGgeDea8iZXwQ1mDAypJgN5X4k5yDs2engA_dzPbjWvnZOPJiftIgT4ZpgvahY7zCZscofz4BKgPRPmIpXOdIKTHqQSpyDvDmg9j8HBcgo_x7iPWQB0iHmB60sb7Lg_QdYnaw",
  "e":"AQAB","alg":"RS256","use":"sig","kid":"test-key-1"
}]}
"""

private const val TEST_JWT =
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InRlc3Qta2V5LTEifQ.eyJpc3MiOiJodHRwczovL2FwcC5leGFtcGxlLmNvbSIsInN1YiI6InVzZXJfdGVzdF8xMjMiLCJzaWQiOiJzZXNzX3Rlc3RfNDU2IiwiYXpwIjoiaHR0cHM6Ly9hcHAuZXhhbXBsZS5jb20iLCJleHAiOjQxMDI0NDQ4MDAsIm5iZiI6MTcwMDAwMDAwMCwiaWF0IjoxNzAwMDAwMDAwfQ.GhyZRPrfTY4lDCrM4iaJ_WXdu1tonF8UF32GwQNNswKM2fpSf4HK7vqkCm2TzzlmUlRVsCjNpKftBKJ8R5Byf0IS1Yp65_kXKZY47sYqiv5W2tcYUB1IH1b4m2sx6PSUmNhpqzDUoN8wx9CCsJ_r-QZpvphenUr9OYC4Cx2NfuyuitZLSuDzFZV54knUf9Obm_tmaP8QqfJfmJLFsmeUUQGctf7VXTDJv7CPvLVAYWxyMBzRyt48ZuxH0HLJOqHEa2fFBHrXbbPrBGdRZHrcMLpDbcUa2jZ1y3g98tbOFF8ZnjBZecs3yMiybxfga0FYRg5Cs4PMsdEGS3UHYGDpug"

private const val JWT_OPTIONS =
    """{"issuer":"https://app.example.com","authorizedParties":["https://app.example.com"],"nowMs":1800000000000}"""

/** Mock provider that calls [tool] in round 1 and reports CONFIRMED only
 *  when it sees [token] in the messages it is handed in round 2. */
private fun mockAgent(token: String, agent: String, tool: String): String = """
(function(){
  var TOKEN = ${jsonQuote(token)};
  var round = 0;
  sofuu.ai.stream = function(opts) {
    round++;
    var toolRound = (round === 1);
    var chunks;
    if (toolRound) { chunks = []; }
    else {
      var seen = JSON.stringify((opts && opts.messages) || []);
      chunks = [{ text: seen.indexOf(TOKEN) >= 0 ? "CONFIRMED-TOOL-RESULT" : "MISSING-TOOL-RESULT" }];
    }
    var i = 0;
    var it = {
      usage: { promptTokens: 11, completionTokens: 3, finishReason: "stop" },
      toolCalls: toolRound
        ? [{ index: 0, id: "call_1", function: { name: ${jsonQuote(tool)}, arguments: "{}" } }]
        : [],
      abort: function() {},
      next: function() {
        if (i < chunks.length) return Promise.resolve({ value: chunks[i++], done: false });
        return Promise.resolve({ value: undefined, done: true });
      }
    };
    it[Symbol.asyncIterator] = function() { return it; };
    return it;
  };
  sofuu.agent.define({
    name: ${jsonQuote(agent)},
    system: "You are a test agent.",
    provider: "mock",
    model: "mock-tool-test",
    tools: [{ builtin: ${jsonQuote(tool)} }],
    budget: { maxSteps: 4 }
  });
})()
"""

private fun collectAgent(rt: SofuuBridge, target: String): List<String> {
    val events = mutableListOf<String>()
    val ok = rt.callStream("agent.run", "{\"target\":\"$target\",\"task\":\"take a photo\"}") { evt ->
        events.add(evt)
        true
    }
    check(ok, "agent.run dispatches ($target)")
    return events
}

fun main() {
    println("== sofuu Kotlin SDK — desktop JVM check ==")
    val rt = SofuuBridge()
    check(rt.abiVersion == 2, "abiVersion == 2 (got ${rt.abiVersion})")

    // 1) Non-BMP (emoji) round-trip through eval — the CESU-8 boundary
    //    both ways. Without the UTF-8 conversion this fails: the JS source
    //    arrives as invalid UTF-8, or the result comes back as mojibake.
    val emoji = "\uD83E\uDD8A\u65E5\u672C" // 🦊日本
    val evalOut = rt.eval("JSON.stringify({t:\"$emoji\"})")
    check(evalOut != null && evalOut.contains(emoji), "eval round-trips emoji + CJK (got: $evalOut)")

    // 2) Crypto/JWT primitives exposed by the same bridge. The fixture is
    //    generic (app.example.com) and the signature is pre-generated.
    val caps = rt.checkCaps()
    check(caps.contains("\"has_crypto\":true"), "checkCaps reports crypto (got: $caps)")
    check(caps.contains("\"has_storage\":true"), "checkCaps reports storage")
    check(
        rt.sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256 FIPS vector",
    )
    check(rt.base64urlEncode("hello") == "aGVsbG8", "base64url encodes without padding")
    check(rt.base64urlDecode("aGVsbG8") == "hello", "base64url decodes without padding")
    check(rt.base64urlDecode("aGk==") == null, "base64url rejects over-padding")

    val jwtOut = rt.verifyJWT(TEST_JWT, TEST_JWKS, JWT_OPTIONS)
    check(jwtOut != null && jwtOut.contains("\"ok\":true"), "valid RS256 JWT verifies (got: $jwtOut)")
    check(jwtOut != null && jwtOut.contains("user_test_123"), "verified JWT exposes sub")
    val tampered = TEST_JWT.substringBeforeLast('.') + "." + rt.base64urlEncode("tampered")
    val bad = rt.verifyJWT(tampered, TEST_JWKS, JWT_OPTIONS)
    check(bad != null && bad.contains("\"ok\":false"), "tampered JWT is rejected (got: $bad)")

    // 3) Malformed argsJson is rejected before it reaches the ABI.
    var threw = false
    try {
        rt.callStream("ai.stream", "{not json") { true }
    } catch (e: IllegalArgumentException) {
        threw = true
    }
    check(threw, "callStream rejects malformed argsJson")

    // 4) Streaming: deltas, terminal done, emoji in ARGS and in EVENTS.
    rt.eval("""
      sofuu.ai.stream = function(opts){
        return (async function*(){
          yield {text: "echo:" + ((opts && opts.prompt) || "")};
          yield {text: "\u4E16\u754C"};
          yield {text: "\uD83E\uDD8A"};
        })();
      };
    """.trimIndent())
    val events = mutableListOf<String>()
    val ok = rt.callStream("ai.stream", "{\"prompt\":\"$emoji\"}") { evt ->
        events.add(evt)
        true
    }
    check(ok, "callStream dispatches")
    check(events.count { it.contains("\"kind\":\"delta\"") } == 3, "three delta events")
    check(events.count { it.contains("\"kind\":\"done\"") } == 1, "exactly one terminal done event")
    check(events.last().contains("\"aborted\":false"), "done is not aborted")
    check(events.any { it.contains("echo:$emoji") }, "emoji survived args → JS → event")
    check(events.any { it.contains("\uD83E\uDD8A") }, "emoji survived a streamed delta")

    // 4) Mid-flight cancel: returning false from the listener stops the pump.
    rt.eval("""
      sofuu.ai.stream = function(opts){
        return (async function*(){ for (var i=0;i<1000;i++){ yield {text:"t"+i}; } })();
      };
    """.trimIndent())
    var deltas = 0
    val ev2 = mutableListOf<String>()
    val ok2 = rt.callStream("ai.stream", null) { evt ->
        ev2.add(evt)
        if (evt.contains("\"kind\":\"delta\"")) { deltas++; deltas < 3 } else true
    }
    check(ok2 && deltas == 3, "cancel took effect after 3 deltas (got $deltas)")
    check(ev2.last().contains("\"aborted\":true"), "done reports aborted:true (got: ${ev2.last()})")

    // 5) streamLocal: positional args + the options builder, mocked.
    rt.eval("""
      sofuu.ai.streamLocal = function(path, prompt, opts){
        return (async function*(){
          yield {text: "path=" + path};
          yield {text: "prompt=" + prompt};
          yield {text: "max=" + (opts && opts.maxTokens) + " temp=" + (opts && opts.temperature)};
        })();
      };
    """.trimIndent())
    val ev5 = mutableListOf<String>()
    val ok5 = rt.streamLocal("/models/x.gguf", "h\u00E9llo $emoji", maxTokens = 7, temperature = 0.3f) { e ->
        ev5.add(e)
        true
    }
    val j5 = ev5.joinToString("\n")
    check(ok5, "streamLocal dispatches")
    check(j5.contains("path=/models/x.gguf"), "streamLocal passes modelPath positionally")
    check(j5.contains("prompt=h\u00E9llo $emoji"), "streamLocal passes prompt positionally (emoji intact)")
    check(j5.contains("max=7") && j5.contains("temp=0.3"), "streamLocal builds options (max=7, temp=0.3)")

    // 6) Host tool (sync): its result reaches the model's transcript.
    var syncCalls = 0
    val regSync = rt.registerHostTool(
        "take_photo", "Takes a photo with the device camera",
        "{\"type\":\"object\",\"properties\":{}}",
    ) { _, _ ->
        syncCalls++
        "{\"path\":\"/tmp/TOOLRES-photo-9f3.jpg\"}"
    }
    check(regSync, "registerHostTool(take_photo)")
    rt.eval(mockAgent("TOOLRES-photo-9f3", "photo-agent", "take_photo"))
    val joined3 = collectAgent(rt, "photo-agent").joinToString("\n")
    check(syncCalls == 1, "the host tool ran exactly once (got $syncCalls)")
    check(
        joined3.contains("CONFIRMED-TOOL-RESULT") && !joined3.contains("MISSING-TOOL-RESULT"),
        "sync tool result reached the model transcript",
    )

    // 7) Host tool (deferred): invoke returns null, another thread settles it.
    val resolved = CompletableFuture<Boolean>()
    val regSlow = rt.registerHostTool(
        "slow_photo", "Takes a photo (slowly)",
        "{\"type\":\"object\",\"properties\":{}}",
    ) { _, callId ->
        Thread {
            Thread.sleep(120)
            resolved.complete(rt.resolveHostTool(callId, "\"TOOLRES-slow-7c1\"", null))
        }.start()
        null // pending — settled from the other thread
    }
    check(regSlow, "registerHostTool(slow_photo)")
    rt.eval(mockAgent("TOOLRES-slow-7c1", "slow-agent", "slow_photo"))
    val joined4 = collectAgent(rt, "slow-agent").joinToString("\n")
    check(resolved.get(10, TimeUnit.SECONDS), "resolveHostTool accepted the id from another thread")
    check(joined4.contains("CONFIRMED-TOOL-RESULT"), "deferred tool result reached the transcript")

    // 8) Unknown call id is refused, not crashed on.
    check(!rt.resolveHostTool(9999, "\"x\"", null), "unknown call id refused")

    // 9) close(): the wrapper goes inert and releases its registrations.
    rt.close()
    check(rt.isClosed, "isClosed after close()")
    check(rt.eval("1+1") == null, "eval inert after close()")
    check(rt.resolveHostTool(1, "\"x\"", null) == false, "resolve inert after close()")
    check(!rt.registerHostTool("x", "x", "{}") { _, _ -> null }, "register inert after close()")
    rt.close() // idempotent
    check(true, "double close() is a no-op")

    if (failures == 0) {
        println("ALL CHECKS PASSED")
    } else {
        println("$failures CHECK(S) FAILED")
        exitProcess(1)
    }
}
