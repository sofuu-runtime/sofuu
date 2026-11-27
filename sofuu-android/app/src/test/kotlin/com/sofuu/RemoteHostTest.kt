package com.sofuu

import com.sofuu.engine.EngineEvent
import com.sofuu.engine.remote.HostHttpClient
import com.sofuu.engine.remote.HostProfile
import com.sofuu.engine.remote.RemoteEventMapper
import com.sofuu.store.SessionMessageStore
import com.sofuu.store.Usage
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress
import java.net.ServerSocket
import java.nio.charset.StandardCharsets

class HostProfileTest {

    @Test
    fun bareAddressGetsHttpAndNoTrailingSlash() {
        val p = HostProfile.create("Mac", "192.168.1.10:7707/", "dev_x")
        assertEquals("http://192.168.1.10:7707", p.baseUrl())
        assertEquals("192.168.1.10:7707", p.displayAddress())
    }

    @Test
    fun httpsAddressIsPreserved() {
        val p = HostProfile.create("Mac", "https://mac.tailnet.ts.net", "dev_x")
        assertEquals("https://mac.tailnet.ts.net", p.baseUrl())
    }

    @Test
    fun jsonRoundTripKeepsTheToken() {
        val p = HostProfile.create("Mac", "10.0.0.2:7707", "dev_abc")
        val back = HostProfile.fromJson(p.toJson())
        assertEquals(p.id, back.id)
        assertEquals("dev_abc", back.token)
        assertEquals("10.0.0.2:7707", back.address)
    }
}

class RemoteEventMapperTest {

    @Test
    fun startUsesThePayloadSessionId() {
        val ev = RemoteEventMapper.map(
            JSONObject("""{"kind":"start","sid":"proj","payload":{"sessionId":"s-1"}}""")
        )
        assertEquals(EngineEvent.TurnStarted("s-1"), ev)
    }

    @Test
    fun doneCarriesErrorAndUsage() {
        val ev = RemoteEventMapper.map(
            JSONObject(
                """{"kind":"done","payload":{"error":{"message":"boom"},
                   "usage":{"promptTokens":3,"completionTokens":4,"costUsd":0.5,"servedBy":"m"}}}"""
            )
        ) as EngineEvent.TurnFinished
        assertEquals("boom", ev.error)
        assertEquals(3L, ev.usage?.promptTokens)
        assertEquals(4L, ev.usage?.completionTokens)
        assertEquals(0.5, ev.usage?.costUsd ?: 0.0, 0.0)
    }

    @Test
    fun approvalMapsItsArguments() {
        val ev = RemoteEventMapper.map(
            JSONObject("""{"kind":"approval_request","payload":{"id":"ap-1","tool":"write_file","args":{"path":"a"},"risky":true}}""")
        ) as EngineEvent.ApprovalRequested
        assertEquals("ap-1", ev.id)
        assertEquals("write_file", ev.tool)
        assertEquals("a", ev.args.optString("path"))
    }

    @Test
    fun unknownKindStaysAStepForTheTimeline() {
        val raw = JSONObject("""{"kind":"answer_delta","payload":{"text":"hi"},"seq":7}""")
        val ev = RemoteEventMapper.map(raw) as EngineEvent.Step
        assertEquals("answer_delta", ev.kind)
        assertEquals("hi", ev.raw.optJSONObject("payload")?.optString("text"))
    }

    @Test
    fun missingKindIsIgnored() {
        assertNull(RemoteEventMapper.map(JSONObject("""{"payload":{}}""")))
    }
}

/**
 * A one-shot HTTP server on a plain [ServerSocket]. `com.sun.net.httpserver`
 * lives in the `jdk.httpserver` module and is NOT on the Android unit-test
 * compile classpath, so the tests own their transport the same way the
 * client does: sockets + bytes.
 */
private class FakeHost(
    private val response: (requestLine: String, headers: String) -> Pair<String, String>,
) : AutoCloseable {
    private val socket = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
    val baseUrl: String = "http://127.0.0.1:${socket.localPort}"
    private val thread = Thread {
        try {
            socket.accept().use { conn ->
                conn.soTimeout = 5_000
                val reader = conn.getInputStream().bufferedReader(StandardCharsets.UTF_8)
                val requestLine = reader.readLine().orEmpty()
                val headers = StringBuilder()
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.isEmpty()) break
                    headers.append(line).append('\n')
                }
                val (contentType, body) = response(requestLine, headers.toString())
                val out = conn.getOutputStream()
                val head = "HTTP/1.1 200 OK\r\n" +
                    "Content-Type: $contentType\r\n" +
                    "Connection: close\r\n" +
                    "Content-Length: ${body.toByteArray(StandardCharsets.UTF_8).size}\r\n\r\n"
                out.write(head.toByteArray(StandardCharsets.UTF_8))
                out.write(body.toByteArray(StandardCharsets.UTF_8))
                out.flush()
            }
        } catch (_: Exception) {
        }
    }.apply {
        isDaemon = true
        start()
    }

    override fun close() {
        try { socket.close() } catch (_: Exception) {}
        thread.join(2_000)
    }
}

class HostSessionUsageTest {

    @Test
    fun compactTurnUsageRebuildsTheMeter() {
        val turns = org.json.JSONArray(
            """[{"prompt":"hi","answer":"yo",
                 "usage":{"model":"m","provider":"p","pt":11,"ct":13,"cost":0.02},
                 "servedBy":"m"}]"""
        )
        val msgs = SessionMessageStore.turnsToMessages(turns)
        val assistant = msgs.first { it.role == com.sofuu.store.MsgRole.ASSISTANT }
        assertEquals(11L, assistant.usage?.promptTokens)
        assertEquals(13L, assistant.usage?.completionTokens)
        assertEquals(0.02, assistant.usage?.costUsd ?: 0.0, 0.0001)
        assertEquals("m", assistant.usage?.servedBy)
    }

    @Test
    fun bothSpellingsParse() {
        assertEquals(5L, Usage.fromJson(JSONObject("""{"promptTokens":5}""")).promptTokens)
        assertEquals(5L, Usage.fromJson(JSONObject("""{"pt":5}""")).promptTokens)
    }
}

class HostHttpClientTest {

    @Test
    fun requestSendsTheBearerAndParsesJson() {
        var seenPath = ""
        var seenAuth = ""
        FakeHost { requestLine, headers ->
            seenPath = requestLine.split(' ').getOrElse(1) { "" }
            seenAuth = headers.lineSequence()
                .firstOrNull { it.startsWith("Authorization:", ignoreCase = true) }
                .orEmpty()
            "application/json" to """{"ok":true,"name":"mac"}"""
        }.use { host ->
            val client = HostHttpClient(host.baseUrl, "dev_token")
            val out = client.request("GET", "/v1/host")
            assertEquals("/v1/host", seenPath)
            assertTrue(seenAuth.contains("Bearer dev_token"))
            assertTrue(out.optBoolean("ok"))
            assertEquals("mac", out.optString("name"))
        }
    }

    @Test
    fun sseFramesAreDeliveredInOrderAndHeartbeatsIgnored() {
        val body = buildString {
            append("data: {\"kind\":\"sync\",\"seq\":0}\n\n")
            append(": ping\n\n")
            append("data: {\"kind\":\"start\",\"seq\":1}\n\n")
        }
        FakeHost { _, _ -> "text/event-stream" to body }.use { host ->
            val got = ArrayList<String>()
            HostHttpClient(host.baseUrl, "t").stream("/events") { got.add(it.optString("kind")) }
            assertEquals(listOf("sync", "start"), got)
        }
    }
}
