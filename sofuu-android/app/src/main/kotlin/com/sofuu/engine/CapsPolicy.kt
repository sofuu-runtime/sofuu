package com.sofuu.engine

/**
 * Pure decisions about a model's capability payload.
 *
 * Extracted from `AppStore.pickLocalModel` so the rule that decides whether a
 * local GGUF gets reasoning effort can be unit-tested without an engine or an
 * Android device. `modelCaps` reports `thinking` as a KIND STRING
 * (`"none"`, `"effort"`, `"budget"`, `"unknown"`); an older shape used a
 * boolean, which is why both are accepted.
 */
object CapsPolicy {
    fun thinkingEnabled(kind: String?): Boolean {
        val k = kind?.trim().orEmpty()
        return k.isNotEmpty() && k != "none" && k != "unknown"
    }
}
