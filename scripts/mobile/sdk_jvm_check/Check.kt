// scripts/mobile/sdk_jvm_check/Check.kt — desktop JVM check for the Sofuu
// Android SDK wrapper (SofuuBridge.kt).
//
// Run by scripts/mobile/sdk_jvm_check.sh: that script compiles the SAME
// jni_bridge.c the NDK builds for Android, but for the host OS, links it
// against the runtime built from this source tree, compiles the real Kotlin
// wrapper, and then runs this file.
//
// REAL EXECUTION (RULES.md §1.9)
//   * Every model call below runs the REAL in-process engine against an
//     existing local GGUF (SmolLM2-360M-Instruct by default). No provider is
//     stubbed, no answer is canned: the deltas asserted here are the model's
//     own, and the token counts come from the engine.
//   * The host-tool callbacks read a REAL temporary file off disk, and the
//     answers travel back through the runtime's own tool registry.
//   * The camera is NOT exercised: a camera needs a physical Android device,
//     so camera coverage is reported as UNVERIFIED here instead of being
//     faked with a canned path.
//
// SCOPE: host JVM only. This is NOT Android device coverage.
//
// What it pins down:
//   - the UTF-8 boundary (JNI "UTF-8" is CESU-8 for non-BMP text) in both
//     directions: Kotlin → runtime → Kotlin, and file → host tool → runtime
//   - real streaming deltas from the local engine and the single terminal
//     `done` event
//   - mid-flight cancel: the listener returns false and the stream stops
//   - ai.streamLocal's positional-argument contract, with maxTokens proven
//     to have reached the engine by the token count it allows
//   - host tools: a sync answer and one settled from another thread
//   - close(): the wrapper goes inert and releases its registrations

import com.sofuu.runtime.SofuuBridge
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import kotlin.math.abs
import kotlin.system.exitProcess

private var checks = 0
private var failures = 0
private var blocked = 0

private fun check(cond: Boolean, what: String) {
    checks++
    if (cond) {
        println("  ok  : $what")
    } else {
        failures++
        println("  FAIL: $what")
    }
}

/** A case that could not run here (no local model, no device). Not a pass. */
private fun block(what: String, why: String) {
    blocked++
    println("  BLOCKED: $what — $why")
}

private fun eventOf(evt: String): JSONObject? = try {
    JSONObject(evt)
} catch (e: org.json.JSONException) {
    null
}

private fun kindOf(evt: String): String? =
    eventOf(evt)?.optString("kind")?.takeIf { it.isNotEmpty() }

private fun textOf(evt: String): String =
    eventOf(evt)?.optString("text") ?: ""

private fun doneEventOf(events: List<String>): JSONObject? =
    events.lastOrNull { kindOf(it) == "done" }?.let { eventOf(it) }

/** The real local GGUF the model cases run against, or null when none is available. */
private fun localModel(): String? {
    val raw = System.getProperty("sofuu.jvm.check.model")
        ?: System.getenv("SOFUU_JVM_CHECK_MODEL")
        ?: return null
    if (raw.isBlank()) return null
    val f = File(raw)
    return if (f.isFile) f.absolutePath else null
}

// Generic RS256 fixture shared with the Rust crypto tests
// (crates/sofuu-core/src/rt/crypto.rs — tests::TEST_JWK / tests::TEST_JWT).
// The token is signed for https://app.example.com and expires in 2100; its
// payload carries the `azp` claim the verifier checks against
// authorizedParties.
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

fun main() {
    println("== sofuu Kotlin SDK — desktop JVM check ==")
    val model = localModel()
    val rt = SofuuBridge()
    check(rt.abiVersion == 2, "abiVersion == 2 (got ${rt.abiVersion})")

    // 1) Non-BMP (emoji) round trip through eval — the CESU-8 boundary
    //    both ways. Without the UTF-8 conversion this fails: the JS source
    //    arrives as invalid UTF-8, or the result comes back as mojibake.
    val emoji = "\uD83E\uDD8A\u65E5\u672C" // 🦊日本
    val evalOut = rt.eval("JSON.stringify({t:\"$emoji\"})")
    check(evalOut != null && evalOut.contains(emoji), "eval round-trips emoji + CJK (got: $evalOut)")

    // 2) The renderer the local engine builds its prompt with: the emoji has
    //    to survive Kotlin → JSON → JS → prompt → Kotlin, or the engine
    //    would be prompted with mojibake. The C ABI spreads a JSON array as
    //    POSITIONAL arguments, so the message array is nested one level.
    val templateArgs = JSONArray()
        .put(
            JSONArray().put(
                JSONObject().put("role", "user").put("content", emoji)
            )
        )
        .toString()
    val templateEnvelope = rt.call("ai.applyChatTemplate", templateArgs)
    check(
        templateEnvelope != null && templateEnvelope.contains(emoji),
        "the rendered prompt keeps the emoji intact (got: $templateEnvelope)",
    )

    // 3) Crypto/JWT primitives exposed by the same bridge. The fixture is
    //    generic (app.example.com) and the signature is pre-generated: a
    //    static vector for a deterministic pure computation, not a stand-in
    //    for a live service.
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

    // 3b) The AAR's similarity() helper: the REAL cosine similarity over
    //     real vectors, unwrapped from the ABI's JSON envelope. A helper
    //     that answered a constant 0 would look like a measurement, so the
    //     check pins the actual value the engine computed.
    val simExposed = rt.eval("typeof sofuu.ai.similarity === 'function' ? 'yes' : 'no'")
    if (simExposed?.contains("yes") == true) {
        val v1 = floatArrayOf(1f, 0f, 0f, 0f)
        val v2 = floatArrayOf(0f, 1f, 0f, 0f)
        val simSame = rt.similarity(v1, v1)
        val simOrtho = rt.similarity(v1, v2)
        check(
            simSame != null && abs(simSame - 1f) < 1e-5,
            "similarity of a vector with itself is 1 (got: $simSame)",
        )
        check(
            simOrtho != null && abs(simOrtho) < 1e-5,
            "similarity of orthogonal vectors is 0 (got: $simOrtho)",
        )
    } else {
        block(
            "the AAR similarity() helper",
            "this runtime build does not expose sofuu.ai.similarity",
        )
    }

    val jwtOut = rt.verifyJWT(TEST_JWT, TEST_JWKS, JWT_OPTIONS)
    check(jwtOut != null && jwtOut.contains("\"ok\":true"), "valid RS256 JWT verifies (got: $jwtOut)")
    check(jwtOut != null && jwtOut.contains("user_test_123"), "verified JWT exposes sub")
    val tampered = TEST_JWT.substringBeforeLast('.') + "." + rt.base64urlEncode("tampered")
    val bad = rt.verifyJWT(tampered, TEST_JWKS, JWT_OPTIONS)
    check(bad != null && bad.contains("\"ok\":false"), "tampered JWT is rejected (got: $bad)")

    // 4) Malformed argsJson is rejected before it reaches the ABI.
    var threw = false
    try {
        rt.callStream("ai.stream", "{not json") { true }
    } catch (e: IllegalArgumentException) {
        threw = true
    }
    check(threw, "callStream rejects malformed argsJson")

    if (model == null) {
        block(
            "real inference cases (ai.stream, ai.streamLocal, agent.run, mid-flight cancel)",
            "no local GGUF available — set SOFUU_JVM_CHECK_MODEL or pass -Dsofuu.jvm.check.model=<file.gguf>",
        )
    } else {
        println("  ..   model: $model")
        // ai.streamLocal is the RAW-prompt API: the caller owns the prompt
        // format, so the chat-formatted prompt the engine would have built
        // itself is handed over as-is.
        val rawPrompt = "user\nName three sea birds, one per line.\nassistant\n"

        // 5) REAL streaming: the local engine decodes a prompt that carries
        //    the emoji, and the deltas below are the model's own tokens.
        val streamEvents = mutableListOf<String>()
        val streamArgs = JSONObject()
            .put("model", model)
            .put("prompt", "Reply with one short sentence about $emoji.")
            .put("maxTokens", 24)
            .toString()
        val dispatched = rt.callStream("ai.stream", streamArgs) { evt ->
            streamEvents.add(evt)
            true
        }
        check(dispatched, "ai.stream dispatched to the local engine")
        val deltas = streamEvents.filter { kindOf(it) == "delta" }
        val done = doneEventOf(streamEvents)
        check(deltas.isNotEmpty(), "the local engine streamed real deltas (${deltas.size})")
        check(streamEvents.count { kindOf(it) == "done" } == 1, "exactly one terminal done event")
        check(done?.optJSONObject("error") == null, "the terminal done carries no error")
        check(done?.optBoolean("aborted") == false, "the terminal done is not aborted")
        val completionTokens = done?.optJSONObject("usage")?.optInt("completionTokens") ?: 0
        check(completionTokens > 0, "the engine counted real completion tokens ($completionTokens)")
        check(deltas.joinToString("") { textOf(it) }.isNotBlank(), "the model produced non-empty text")
        check(
            streamEvents.all { eventOf(it) != null && kindOf(it) != null },
            "every streamed event crossed the JNI boundary as well-formed JSON",
        )

        // 6) REAL mid-flight cancel: the listener stops the pump and the
        //    runtime reports the abort.
        var cancelDeltas = 0
        val cancelEvents = mutableListOf<String>()
        val cancelDispatched = rt.streamLocal(
            model, rawPrompt, maxTokens = 32, temperature = 0.2f,
        ) { evt ->
            cancelEvents.add(evt)
            if (kindOf(evt) == "delta") { cancelDeltas++; false } else true
        }
        check(cancelDispatched, "the cancelled stream dispatched")
        check(
            cancelDeltas == 1,
            "the listener stopped the pump after the first delta (got $cancelDeltas; events: $cancelEvents)",
        )
        val cancelDone = doneEventOf(cancelEvents)
        check(cancelDone?.optBoolean("aborted") == true, "the terminal done reports aborted:true")
        check(
            cancelDone?.optInt("deltas") == cancelDeltas,
            "the terminal counts the deltas that were delivered",
        )

        // 7) The agent loop on the real engine: the answer comes from the
        //    local model, not from a scripted provider.
        val defined = rt.eval(
            """
            sofuu.agent.define({
              name: 'jvm-check-agent',
              system: 'You answer with one short sentence.',
              provider: 'local',
              model: ${JSONObject.quote(model)},
              budget: { maxSteps: 1 }
            })
            """.trimIndent()
        )
        check(defined != null && defined.contains("\"ok\":true"), "agent.define accepted the local agent (got: $defined)")
        val agentEvents = mutableListOf<String>()
        val agentDispatched = rt.callStream(
            "agent.run",
            """{"target":"jvm-check-agent","task":"Say hello in one short sentence."}""",
        ) { evt ->
            agentEvents.add(evt)
            true
        }
        check(agentDispatched, "agent.run dispatched to the local engine")
        val agentDone = doneEventOf(agentEvents)
        check(
            agentDone != null && agentDone.optJSONObject("error") == null,
            "the agent turn ended without an error (got: ${agentDone?.optJSONObject("error")})",
        )
        check(
            agentEvents.filter { kindOf(it) != "done" }.joinToString("").isNotBlank(),
            "the agent produced a real answer",
        )

        // 8) streamLocal's positional contract: (modelPath, prompt, options).
        //    maxTokens is proven to have reached the engine by the number of
        //    tokens it allows.
        val localEvents = mutableListOf<String>()
        val localDispatched = rt.streamLocal(
            model, rawPrompt, maxTokens = 8, temperature = 0.2f,
        ) { evt ->
            localEvents.add(evt)
            true
        }
        check(localDispatched, "streamLocal dispatched")
        val localDeltas = localEvents.count { kindOf(it) == "delta" }
        check(localDeltas in 1..8, "streamLocal honoured maxTokens=8 (deltas=$localDeltas; events: $localEvents)")
        check(localEvents.count { kindOf(it) == "done" } == 1, "streamLocal ends with exactly one done event")
        check(
            localEvents.filter { kindOf(it) == "delta" }.joinToString("") { textOf(it) }.isNotBlank(),
            "streamLocal produced non-empty model text",
        )
    }

    // 9) Host tool (sync): the callback reads a REAL temporary file, and
    //    the answer comes back through the runtime's own tool registry —
    //    the same entry the agent's tool loop calls.
    val sample = Files.createTempFile("sofuu-jvm-check-", ".txt")
    val marker = "host-read-${System.nanoTime()}"
    val payload = "$marker $emoji"
    Files.write(sample, payload.toByteArray(Charsets.UTF_8))
    var syncCalls = 0
    val regSync = rt.registerHostTool(
        "read_sample", "Reads the check's sample file",
        "{\"type\":\"object\",\"properties\":{}}",
    ) { _, _ ->
        syncCalls++
        JSONObject()
            .put("content", String(Files.readAllBytes(sample), Charsets.UTF_8))
            .put("path", sample.toString())
            .toString()
    }
    check(regSync, "registerHostTool(read_sample)")
    val syncResult = rt.callResult("tools.TOOLS.read_sample.execute", "{}")
    check(
        syncResult != null && syncResult.contains(marker),
        "the host tool's real file read reached the runtime (got: $syncResult)",
    )
    check(
        syncResult != null && syncResult.contains(emoji),
        "the file's non-ASCII content survived the host-tool round trip",
    )
    check(syncCalls == 1, "the host tool ran exactly once (got $syncCalls)")

    // The agent's tool set picks the host tool up from the runtime's
    // registry — the same entry the tool loop executes above.
    val definedWithTool = rt.eval(
        """
        sofuu.agent.define({
          name: 'jvm-check-host-agent',
          provider: 'local',
          ${if (model != null) "model: ${JSONObject.quote(model)}," else ""}
          tools: [{ builtin: 'read_sample' }]
        })
        """.trimIndent()
    )
    check(
        definedWithTool != null && definedWithTool.contains("\"ok\":true"),
        "agent.define accepts the registered host tool (got: $definedWithTool)",
    )
    check(
        definedWithTool != null && definedWithTool.contains("read_sample"),
        "the agent's tool set carries the host tool",
    )

    // 10) Host tool (deferred): the callback takes the call, a worker thread
    //    settles it later, and the runtime's promise resolves with the
    //    file's real content.
    var asyncCalls = 0
    val regAsync = rt.registerHostTool(
        "read_sample_async", "Reads the check's sample file on a worker thread",
        "{\"type\":\"object\",\"properties\":{}}",
    ) { _, callId ->
        asyncCalls++
        Thread {
            Thread.sleep(150)
            val content = String(Files.readAllBytes(sample), Charsets.UTF_8)
            rt.resolveHostTool(callId, JSONObject().put("content", content).toString(), null)
        }.start()
        null
    }
    check(regAsync, "registerHostTool(read_sample_async)")
    // A pending host-tool call is a promise, not a libuv handle: the event
    // loop would drain before the worker answers. A real agent turn keeps
    // itself alive with its tool-timeout timer; for a direct tool call the
    // check arms the same keep-alive and clears it when the answer lands.
    val armed = rt.eval(
        """
        (function(){
          var orig = sofuu.tools.TOOLS.read_sample_async.execute;
          sofuu.tools.TOOLS.read_sample_async.execute = function(args) {
            var keep = setTimeout(function () {}, 10000);
            return orig.call(null, args).then(
              function (v) { clearTimeout(keep); return v; },
              function (e) { clearTimeout(keep); throw e; });
          };
          return 'armed';
        })()
        """.trimIndent()
    )
    check(armed != null && armed.contains("armed"), "the deferred tool's keep-alive is armed")
    val asyncResult = rt.callResult("tools.TOOLS.read_sample_async.execute", "{}")
    check(
        asyncResult != null && asyncResult.contains(marker),
        "the deferred host tool result arrived from the worker thread (got: $asyncResult)",
    )
    check(asyncCalls == 1, "the deferred host tool ran exactly once (got $asyncCalls)")

    // 11) Unknown call id is refused, not crashed on.
    check(!rt.resolveHostTool(9999, "\"x\"", null), "unknown call id refused")

    // 12) close(): the wrapper goes inert and releases its registrations.
    rt.close()
    check(rt.isClosed, "isClosed after close()")
    check(rt.eval("1+1") == null, "eval inert after close()")
    check(!rt.resolveHostTool(1, "\"x\"", null), "resolve inert after close()")
    check(!rt.registerHostTool("x", "x", "{}") { _, _ -> null }, "register inert after close()")
    rt.close() // idempotent
    check(true, "double close() is a no-op")

    Files.deleteIfExists(sample)

    // 13) Camera: NOT covered here. A camera needs a physical Android
    //     device and the host JVM has none, so camera coverage is recorded
    //     as UNVERIFIED — it is not simulated and not asserted.
    println("  UNVERIFIED: camera capture — needs a physical Android device (no camera on the host JVM)")

    println()
    if (failures == 0 && blocked == 0) {
        println("ALL CHECKS PASSED ($checks checks) — host-JVM coverage only, not Android device coverage")
    } else {
        if (failures > 0) println("$failures CHECK(S) FAILED")
        if (blocked > 0) println("$blocked CASE(S) BLOCKED")
        exitProcess(if (failures > 0) 1 else 2)
    }
}
