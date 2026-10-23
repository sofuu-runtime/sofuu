package com.sofuu

import com.sofuu.engine.CapsPolicy
import com.sofuu.engine.SofuuEngine
import com.sofuu.ui.chat.diffLines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM tests for logic that needs no engine and no device.
 *
 * These cover the two places that produced real defects in this app:
 * the provider-listing URL (A1 — a wrong URL is a silent "no models") and
 * the caps payload interpretation (A4 — a wrong field name silently wrote 0
 * and forced effort off).
 */
class ProviderUrlTest {

    @Test
    fun plainV1BaseAppendsModels() {
        assertEquals(
            "https://api.openai.com/v1/models",
            SofuuEngine.modelsUrl("https://api.openai.com/v1", ""),
        )
    }

    @Test
    fun fullCompletionsPathIsStrippedToTheCore() {
        assertEquals(
            "https://api.openai.com/v1/models",
            SofuuEngine.modelsUrl("https://api.openai.com/v1/chat/completions", ""),
        )
        assertEquals(
            "https://api.example.com/models",
            SofuuEngine.modelsUrl("https://api.example.com/completions", ""),
        )
    }

    @Test
    fun trailingSlashesDoNotDoubleUp() {
        assertEquals(
            "https://api.example.com/v1/models",
            SofuuEngine.modelsUrl("https://api.example.com/v1///", ""),
        )
    }

    @Test
    fun anthropicHasNoModelsEndpoint() {
        assertNull(SofuuEngine.modelsUrl("https://api.anthropic.com/v1/messages", ""))
        assertNull(SofuuEngine.modelsUrl("https://anywhere.example.com", "anthropic"))
    }

    @Test
    fun localProviderUsesOllamaTags() {
        assertEquals("http://127.0.0.1:11434/api/tags", SofuuEngine.modelsUrl("", "local"))
        assertEquals(
            "http://192.168.1.5:11434/api/tags",
            SofuuEngine.modelsUrl("http://192.168.1.5:11434/", "local"),
        )
        // Port 11434 alone is enough to recognise Ollama.
        assertTrue(SofuuEngine.isLocal("http://127.0.0.1:11434", ""))
    }

    @Test
    fun blankEndpointHasNoUrl() {
        assertNull(SofuuEngine.modelsUrl("   ", ""))
    }
}

class CapsPolicyTest {

    @Test
    fun kindStrings() {
        assertFalse(CapsPolicy.thinkingEnabled(null))
        assertFalse(CapsPolicy.thinkingEnabled(""))
        assertFalse(CapsPolicy.thinkingEnabled("none"))
        assertFalse(CapsPolicy.thinkingEnabled("unknown"))
        assertTrue(CapsPolicy.thinkingEnabled("effort"))
        assertTrue(CapsPolicy.thinkingEnabled("budget"))
    }
}

class DiffLinesTest {

    @Test
    fun identicalTextIsAllContext() {
        val rows = diffLines("a\nb", "a\nb")
        assertEquals(listOf("ctx", "ctx"), rows.map { it.kind })
        assertEquals(listOf("a", "b"), rows.map { it.text })
    }

    @Test
    fun changedLineIsDeleteThenInsert() {
        val rows = diffLines("a\nb\nc", "a\nB\nc")
        assertEquals("ctx", rows[0].kind)
        assertTrue(rows.any { it.kind == "del" && it.text == "b" })
        assertTrue(rows.any { it.kind == "add" && it.text == "B" })
        assertEquals("ctx", rows.last().kind)
    }

    @Test
    fun oversizedInputCollapses() {
        val big = (1..500).joinToString("\n")
        val rows = diffLines(big, big)
        assertEquals(2, rows.size)
        assertEquals("del", rows[0].kind)
        assertEquals("add", rows[1].kind)
    }
}
