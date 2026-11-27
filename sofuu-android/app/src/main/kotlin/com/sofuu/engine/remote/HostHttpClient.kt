package com.sofuu.engine.remote

import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/** A structured failure from the host (or its transport). */
class HostApiException(
    val status: Int,
    val code: String?,
    message: String,
) : Exception(message)

/**
 * The host's HTTP surface over `HttpURLConnection` — no new Gradle
 * dependency, and the same transport the local provider calls already use.
 * Every call is blocking; [RemoteChatBackend] runs them on Dispatchers.IO.
 */
class HostHttpClient(private val baseUrl: String, private val token: String) {

    @Volatile private var liveStream: HttpURLConnection? = null

    /** Abort a blocking SSE read immediately (session switch / disconnect). */
    fun close() {
        try { liveStream?.disconnect() } catch (_: Exception) {}
        liveStream = null
    }

    fun request(method: String, path: String, body: JSONObject? = null): JSONObject {
        val conn = open(method, path)
        try {
            if (body != null) {
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json")
                conn.outputStream.use { it.write(body.toString().toByteArray(StandardCharsets.UTF_8)) }
            }
            val status = conn.responseCode
            val stream = if (status >= 400) conn.errorStream else conn.inputStream
            val text = stream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (status >= 400) throw apiError(status, text)
            if (text.isBlank()) return JSONObject()
            return try {
                JSONObject(text)
            } catch (e: Exception) {
                throw HostApiException(status, "bad_json", "host returned malformed JSON: ${e.message}")
            }
        } finally {
            conn.disconnect()
        }
    }

    /**
     * Read an SSE stream until it ends or throws. Each complete `data:`
     * frame is decoded and handed to [onEvent]; `:` heartbeat comments are
     * ignored. A read timeout means the host went silent — the caller
     * reconnects with its cursor.
     */
    fun stream(path: String, onOpen: () -> Unit = {}, onEvent: (JSONObject) -> Unit) {
        val conn = open("GET", path)
        liveStream = conn
        conn.setRequestProperty("Accept", "text/event-stream")
        conn.readTimeout = 45_000
        try {
            val status = conn.responseCode
            if (status >= 400) {
                val text = conn.errorStream?.bufferedReader(StandardCharsets.UTF_8)?.use { it.readText() }.orEmpty()
                throw apiError(status, text)
            }
            onOpen()
            val reader = BufferedReader(InputStreamReader(conn.inputStream, StandardCharsets.UTF_8))
            val data = StringBuilder()
            while (true) {
                val line = reader.readLine() ?: break
                when {
                    line.isEmpty() -> {
                        if (data.isNotEmpty()) {
                            val payload = data.toString()
                            data.setLength(0)
                            val obj = try { JSONObject(payload) } catch (_: Exception) { null }
                            if (obj != null) onEvent(obj)
                        }
                    }
                    line.startsWith("data:") -> {
                        if (data.isNotEmpty()) data.append('\n')
                        data.append(line.removePrefix("data:").trimStart())
                    }
                    // ": ping ..." and any other SSE field is ignored.
                }
            }
            // A stream that ends mid-frame still delivers what it had.
            if (data.isNotEmpty()) {
                runCatching { JSONObject(data.toString()) }.getOrNull()?.let(onEvent)
            }
        } finally {
            if (liveStream === conn) liveStream = null
            conn.disconnect()
        }
    }

    private fun open(method: String, path: String): HttpURLConnection {
        val url = URL(baseUrl + path)
        return (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 5_000
            readTimeout = 30_000
            instanceFollowRedirects = false
            setRequestProperty("Accept", "application/json")
            if (token.isNotBlank()) setRequestProperty("Authorization", "Bearer $token")
        }
    }

    private fun apiError(status: Int, text: String): HostApiException {
        var code: String? = null
        var message = "host returned HTTP $status"
        try {
            val obj = JSONObject(text)
            val err = obj.optJSONObject("error")
            if (err != null) {
                message = err.optString("message").ifBlank { message }
                code = err.optString("code").ifBlank { null }
            }
        } catch (_: Exception) {
            if (text.isNotBlank()) message = "host returned HTTP $status: ${text.take(200)}"
        }
        return HostApiException(status, code, message)
    }
}
