package com.sofuu.engine

import android.content.Context
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.os.Build
import android.util.Base64
import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Voice capture + playback for the provider-backed speech surface.
 *
 * The engine funnels both directions to `ai.transcribe` / `ai.speak`
 * (crates/sofuu-core/src/rt/ai.rs). Nothing is done on-device here: this
 * only records, base64-encodes, and plays back what the provider returns.
 *
 * Format: AAC in an MPEG-4 container (.m4a). The transcribe funnel derives
 * the multipart MIME from the filename extension and accepts .mp3/.m4a/
 * .mp4/.ogg/.webm/.wav (ai.rs:4806-4820), so the recorder is told to
 * report `recording.m4a` — get that wrong and the provider rejects the
 * upload for the wrong content-type.
 *
 * No-wrap base64 is mandatory: sofuu-capi validates the alphabet before
 * splicing the string into JS, and line breaks are not in the allowlist.
 */
class Speech(private val context: Context) {

    private var recorder: MediaRecorder? = null
    private var target: File? = null

    val isRecording: Boolean get() = recorder != null

    // ── Capture ────────────────────────────────────────────────────

    fun startRecording(): Boolean {
        if (recorder != null) return false
        val f = File(context.cacheDir, "sofuu-rec-${System.currentTimeMillis()}.m4a")
        val mr = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) MediaRecorder(context) else MediaRecorder()
        return try {
            mr.setAudioSource(MediaRecorder.AudioSource.MIC)
            mr.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            mr.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            mr.setAudioEncodingBitRate(64_000)
            mr.setAudioSamplingRate(16_000)
            mr.setOutputFile(f.absolutePath)
            mr.prepare()
            mr.start()
            recorder = mr
            target = f
            true
        } catch (e: Throwable) {
            Log.w("sofuu-voice", "recorder start failed: ${e.message}")
            runCatching { mr.release() }
            f.delete()
            recorder = null
            target = null
            false
        }
    }

    /** Stop and return base64 audio, or null if nothing usable was captured. */
    fun stopRecording(): String? {
        val mr = recorder ?: return null
        val f = target
        recorder = null
        target = null
        try {
            mr.stop()
        } catch (e: Throwable) {
            // stop() throws when the clip is ~empty (tapped to start and
            // straight back off). Treat that as "no audio", not a crash.
            Log.i("sofuu-voice", "recording too short: ${e.message}")
            runCatching { mr.release() }
            f?.delete()
            return null
        }
        runCatching { mr.release() }
        if (f == null || !f.exists() || f.length() == 0L) {
            f?.delete()
            return null
        }
        if (f.length() > MAX_AUDIO_BYTES) {
            Log.w("sofuu-voice", "clip ${f.length()} over the ${MAX_AUDIO_BYTES}-byte cap")
            f.delete()
            return null
        }
        val b64 = try {
            Base64.encodeToString(f.readBytes(), Base64.NO_WRAP)
        } catch (e: Throwable) {
            Log.w("sofuu-voice", "read failed: ${e.message}")
            null
        }
        f.delete()
        return b64
    }

    fun cancelRecording() {
        val mr = recorder ?: return
        recorder = null
        target = null
        runCatching { mr.stop() }
        runCatching { mr.release() }
    }

    // ── Playback ───────────────────────────────────────────────────

    private var player: MediaPlayer? = null

    /** Play raw audio bytes. `format` is the provider's label ("mp3"); the
     *  bytes go to a cache file because MediaPlayer wants a source. */
    fun play(bytes: ByteArray, format: String) {
        if (bytes.isEmpty()) return
        stopPlayback()
        val ext = format.ifBlank { "mp3" }.take(6).filter { it.isLetterOrDigit() }.ifEmpty { "mp3" }
        val f = File(context.cacheDir, "sofuu-say-${System.currentTimeMillis()}.$ext")
        return try {
            f.writeBytes(bytes)
            player = MediaPlayer().apply {
                setDataSource(f.absolutePath)
                setOnCompletionListener { stopPlayback() }
                prepare()
                start()
            }
        } catch (e: Throwable) {
            Log.w("sofuu-voice", "playback failed: ${e.message}")
            f.delete()
            stopPlayback()
        }
    }

    fun stopPlayback() {
        player?.let { runCatching { it.stop() } }
        player?.let { runCatching { it.release() } }
        player = null
    }

    val isPlaying: Boolean get() = player?.isPlaying == true

    companion object {
        /** Matches AUDIO_MAX_INPUT in the engine (32 MB). */
        const val MAX_AUDIO_BYTES = 32L * 1024 * 1024

        /** Transcribe opts. `filename` is not cosmetic — it picks the MIME. */
        val TRANSCRIBE_OPTS = JSONObject()
            .put("filename", "recording.m4a")
            .toString()

        val SPEAK_OPTS = JSONObject()
            .put("format", "mp3")
            .toString()

        /**
         * Rebuild the speak payload. Over the JSON funnel a Uint8Array
         * travels as an indexed object (`{"0":12,"1":255,…}`), so the
         * bytes have to be reassembled before playback.
         */
        fun audioBytesFrom(result: JSONObject?): ByteArray? {
            val audio = result?.opt("audio") ?: return null
            if (audio is ByteArray) return audio
            val obj = audio as? JSONObject ?: return null
            val n = obj.length()
            if (n == 0) return null
            val out = ByteArray(n)
            for (i in 0 until n) out[i] = (obj.optInt(i.toString()) and 0xFF).toByte()
            return out
        }
    }
}
