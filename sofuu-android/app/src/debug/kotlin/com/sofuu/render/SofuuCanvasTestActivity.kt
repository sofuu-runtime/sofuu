package com.sofuu.render

import android.app.Activity
import android.os.Bundle

/**
 * Minimal host for the Phase C4 renderer gate. The session is deliberately
 * owned by the companion object (the ViewModel stand-in) and adopted by each
 * recreated view, so `ActivityScenario.recreate()` exercises the rotation
 * path without re-running a JS mount.
 */
class SofuuCanvasTestActivity : Activity() {

    lateinit var canvas: SofuuGpuSurfaceView
        private set

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (sharedSession == 0L) {
            sharedSession = SofuuGpuSurfaceView.createSession()
        }
        canvas = SofuuGpuSurfaceView(this)
        canvas.adoptSession(sharedSession, takeOwnership = false)
        setContentView(canvas)
    }

    companion object {
        @Volatile
        var sharedSession: Long = 0L
    }
}
