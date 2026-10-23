package com.sofuu

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sofuu.engine.EngineEvent
import com.sofuu.engine.SofuuEngine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.After
import org.junit.Assert.assertNull
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Device smoke test for the engine seam.
 *
 * Needs the real `libsofuu.so` on the test device (arm64). The turn test is
 * skipped, not failed, when the device has no model configured: it is there
 * to prove the boot → stream → finish path, not to test a provider.
 */
@RunWith(AndroidJUnit4::class)
class EngineSmokeTest {

    private lateinit var engine: SofuuEngine

    @Before
    fun boot() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val home = File(ctx.filesDir, "smoke-home").apply { mkdirs() }
        val project = File(ctx.filesDir, "smoke-workspace").apply { mkdirs() }
        engine = SofuuEngine(home.absolutePath, project.absolutePath, ctx)
        assertTrue("engine.start() refused", engine.start())
    }

    @After
    fun shutdown() {
        engine.shutdown()
    }

    @Test
    fun bootReachesReadyAndRegistersHostTools() = runBlocking {
        val first = withTimeoutOrNull(30_000) {
            engine.events.first { it is EngineEvent.EngineReady || it is EngineEvent.BootError }
        }
        assertNotNull("engine neither became ready nor reported a boot error", first)
        assertTrue("boot failed: ${(first as? EngineEvent.BootError)?.message}", first is EngineEvent.EngineReady)
        assertTrue("no host tool registered at boot", engine.hostTools.isNotEmpty())

        val state = engine.chatState().get(10, TimeUnit.SECONDS)
        assertNotNull("chat.state() returned nothing after EngineReady", state)
        assertTrue("chat.state() did not report ok", state?.optBoolean("ok") != false)
    }

    @Test
    fun turnStreamsAndFinishes() = runBlocking {
        withTimeoutOrNull(30_000) { engine.events.first { it is EngineEvent.EngineReady } }
        val state = engine.chatState().get(10, TimeUnit.SECONDS)
        Assume.assumeTrue(
            "no model configured on this device — turn smoke skipped",
            !state?.optString("model").isNullOrEmpty(),
        )

        engine.submitTurn("say hi", false)
        val finished = withTimeoutOrNull(120_000) { engine.events.first { it is EngineEvent.TurnFinished } }
        assertNotNull("turn never finished", finished)
        assertNull("turn failed: ${(finished as? EngineEvent.TurnFinished)?.error}", (finished as EngineEvent.TurnFinished).error)
    }
}
