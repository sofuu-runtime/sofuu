package com.sofuu.render

import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device checks for the embedded UI JNI seam (Phase C4).
 *
 * These run without a GL surface: they prove session creation/free, insets,
 * the empty event queue, and host-owned session adoption (the rotation path)
 * survive the JNI boundary. The renderer itself is exercised by attaching a
 * [SofuuGpuSurfaceView] to an Activity and watching [SofuuGpuSurfaceView.frameCount]
 * stop changing once the scene settles.
 */
@RunWith(AndroidJUnit4::class)
class SofuuUiJniTest {

    @Test
    fun sessionLifecycleAndHostAdoption() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext

        instrumentation.runOnMainSync {
            val view = SofuuGpuSurfaceView(context)
            assertNotEquals("session_new returned 0", 0L, view.uiSession)
            assertEquals("a fresh session is view-owned", true, view.ownsSession)

            // Insets cross the JNI boundary; a session with no pending events
            // returns null rather than a bogus JSON string.
            view.setSafeInsets(24.0f, 8.0f)
            assertNull(view.pollSemanticEvent())

            // Rotation path: a ViewModel-owned session is adopted without the
            // view taking ownership; releasing the view must not free it.
            val external = SofuuGpuSurfaceView.createSession(
                """{"title":"JNI test","theme":"dark"}""",
            )
            assertNotEquals("external session_new returned 0", 0L, external)
            view.adoptSession(external, takeOwnership = false)
            assertEquals(external, view.uiSession)
            assertEquals(false, view.ownsSession)
            view.releaseSession()
            assertEquals(0L, view.uiSession)

            // The host still owns it and can free it exactly once.
            SofuuGpuSurfaceView.freeSession(external)
        }
    }

    @Test
    fun repeatedCreateFreeIsStable() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        instrumentation.runOnMainSync {
            repeat(64) {
                val view = SofuuGpuSurfaceView(context)
                assertTrue(view.uiSession != 0L)
                view.releaseSession()
            }
        }
    }

    /**
     * The real Phase C4 gate: attach a canvas to an Activity, wait for the
     * GLES renderer to produce a frame, prove `RENDERMODE_WHEN_DIRTY` stops
     * the loop once the scene settles (0% idle), then recreate the Activity
     * and prove the ViewModel-owned session survives GL context loss.
     */
    @Test
    fun canvasRendersIdlesAndSurvivesRecreation() {
        ActivityScenario.launch(SofuuCanvasTestActivity::class.java).use { scenario ->
            try {
                var frames = 0L
                val firstDeadline = SystemClock.uptimeMillis() + 10_000
                while (SystemClock.uptimeMillis() < firstDeadline) {
                    scenario.onActivity { frames = it.canvas.frameCount }
                    if (frames > 0L) break
                    Thread.sleep(50)
                }
                assertTrue("renderer never produced a frame", frames > 0L)

                // Wait until the frame counter has been stable for 800 ms:
                // an empty settled scene must stop requesting frames.
                var stable = frames
                var stableAt = SystemClock.uptimeMillis()
                while (SystemClock.uptimeMillis() - stableAt < 800L) {
                    Thread.sleep(100L)
                    var current = stable
                    scenario.onActivity { current = it.canvas.frameCount }
                    if (current != stable) {
                        stable = current
                        stableAt = SystemClock.uptimeMillis()
                    }
                }
                Thread.sleep(500L)
                var idle = stable
                scenario.onActivity { idle = it.canvas.frameCount }
                assertEquals(
                    "canvas kept rendering after the scene settled (no 0% idle)",
                    stable,
                    idle,
                )

                scenario.recreate()
                var recreatedFrames = 0L
                var recreatedSession = 0L
                val recreateDeadline = SystemClock.uptimeMillis() + 10_000
                while (SystemClock.uptimeMillis() < recreateDeadline) {
                    scenario.onActivity {
                        recreatedFrames = it.canvas.frameCount
                        recreatedSession = it.canvas.uiSession
                    }
                    if (recreatedFrames > 0L) break
                    Thread.sleep(50)
                }
                assertTrue("renderer did not resume after recreation", recreatedFrames > 0L)
                assertEquals(
                    "the ViewModel-owned session was replaced across recreation",
                    SofuuCanvasTestActivity.sharedSession,
                    recreatedSession,
                )
            } finally {
                val shared = SofuuCanvasTestActivity.sharedSession
                scenario.close()
                SofuuGpuSurfaceView.freeSession(shared)
                SofuuCanvasTestActivity.sharedSession = 0L
            }
        }
    }
}
