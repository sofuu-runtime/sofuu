// examples/headless/KotlinSample/SofuuVoice.kt
//
// M2: on-device voice for Sofuu Android apps — OS speech APIs, zero
// servers, zero per-minute metering. Orchestrate with the agent loop:
// mic → STT → SofuuBridge.call("agent.run") → TTS.
//
// AndroidManifest.xml:
//   <uses-permission android:name="android.permission.RECORD_AUDIO" />
//   <uses-permission android:name="android.permission.INTERNET" /> (cloud fallback only)
// Request RECORD_AUDIO at runtime (ActivityResultContracts.RequestPermission)
// before starting recognition. NOTE: not compiled in CI here (no Android
// SDK) — verify in Android Studio before shipping.

package com.sofuu.runtime

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import java.util.Locale

/// On-device speech I/O. Recognition prefers on-device models
/// (EXTRA_PREFER_OFFLINE); synthesis is always local. Provider STT/TTS
/// via libsofuu remains the fallback (see SofuuBridge voice funnel).
class SofuuVoice(private val context: Context) {

    // ── Speech-to-text (one shot) ──────────────────────────────────

    /// Recognize a single utterance. `onPartial` fires for interim
    /// hypotheses, `onFinal` exactly once. Returns a stopper.
    /// Throws when the device has no recognizer (e.g. no GMS / no engine).
    fun listenOnce(
        locale: Locale = Locale.getDefault(),
        preferOffline: Boolean = true,
        onPartial: (String) -> Unit = {},
        onFinal: (String) -> Unit,
    ): () -> Unit {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            throw RuntimeException("no speech recognizer on device")
        }
        val recognizer = SpeechRecognizer.createSpeechRecognizer(context)
        var finished = false
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onPartialResults(partial: Bundle) {
                if (!finished) {
                    onPartial(best(partial))
                }
            }
            override fun onResults(results: Bundle) {
                if (!finished) {
                    finished = true
                    onFinal(best(results))
                    recognizer.destroy()
                }
            }
            override fun onError(error: Int) {
                if (!finished) {
                    finished = true
                    onFinal("")
                    recognizer.destroy()
                }
            }
            override fun onReadyForSpeech(p: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onEvent(type: Int, p: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, locale.toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, preferOffline)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        recognizer.startListening(intent)
        return {
            if (!finished) {
                finished = true
                recognizer.cancel()
                recognizer.destroy()
            }
        }
    }

    // ── Text-to-speech (always local) ──────────────────────────────

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val ttsQueue = ArrayDeque<() -> Unit>()

    private fun ensureTts(locale: Locale, then: () -> Unit) {
        if (ttsReady) {
            then()
            return
        }
        ttsQueue.add(then)
        if (tts != null) return
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = locale
                ttsReady = true
                while (ttsQueue.isNotEmpty()) {
                    ttsQueue.removeFirst().invoke()
                }
            } else {
                ttsQueue.clear()
                tts?.shutdown()
                tts = null
            }
        }
    }

    /// Speak text (queues behind in-flight speech). Engine init is lazy.
    fun speak(text: String, locale: Locale = Locale.getDefault(), flush: Boolean = false) {
        ensureTts(locale) {
            tts?.speak(
                text,
                if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD,
                null,
                "sofuu-${System.currentTimeMillis()}",
            )
        }
    }

    /// Stop all queued speech.
    fun stopSpeaking() {
        tts?.stop()
    }

    /// True while the engine is producing audio.
    val isSpeaking: Boolean
        get() = tts?.isSpeaking == true

    fun shutdown() {
        tts?.shutdown()
        tts = null
        ttsReady = false
        ttsQueue.clear()
    }

    private fun best(bundle: Bundle): String {
        val list = bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        return list?.firstOrNull() ?: ""
    }
}
