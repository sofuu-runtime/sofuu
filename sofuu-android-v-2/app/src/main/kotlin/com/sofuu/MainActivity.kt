package com.sofuu

import android.content.res.Configuration
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.sofuu.engine.SofuuBridge
import com.sofuu.render.SofuuGpuSurfaceView
import com.sofuu.render.SofuuUiNative
import java.io.File
import kotlin.math.abs

/**
 * sofuu-android-v-2 host activity — zero Jetpack Compose.
 *
 * The whole visible app is a `sofuu.ui` scene: this activity owns one
 * `SofuuGpuSurfaceView` (the GLES 3.0 DrawList renderer), one `sofuu` runtime,
 * and evaluates the JS bundle in assets/v2_ui/. The scene is fully
 * relative-sized, so rotation/resize does not need a remount; only the
 * bottom safe-area inset is baked into the tree (the top inset is reserved by
 * `sofuu_ui_session_set_safe_insets` in the renderer). Moving the session
 * into a ViewModel is the next phase, not needed for a single screen.
 *
 * Threading: the runtime is created and evaluated on the UI thread. The C ABI
 * allows exactly one owning thread per runtime, and every call here (attach,
 * eval, detach) happens on that same thread. The 20 Hz timer only ticks the
 * JS loop; it does not touch the renderer.
 */
class MainActivity : ComponentActivity() {

    private var bridge: SofuuBridge? = null
    private lateinit var canvas: SofuuGpuSurfaceView
    private var scripts: String = ""
    private var mountedOnce = false
    private var insetsKnown = false
    private var topInsetDp = 0f
    private var bottomInsetDp = 0f
    private var workspaceName: String = "workspace"

    private val pumpHandler = Handler(Looper.getMainLooper())
    private val pump = object : Runnable {
        override fun run() {
            // Draws are pull-based on the GL thread; this tick exists so the
            // JS event pump (sofuu.ui, 16 ms interval) runs and can push
            // patches. An idle tick is one JSON envelope and nothing else.
            bridge?.evalNb("0")
            pumpHandler.postDelayed(this, PUMP_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        WindowCompat.setDecorFitsSystemWindows(window, false)

        // Android has no $HOME. The runtime derives config / brain / session
        // paths from it, so pin it before the runtime exists.
        SofuuBridge.setHome(filesDir.absolutePath)
        SofuuBridge.chdir(filesDir.absolutePath)
        SofuuBridge.setLogEnabled(true)

        // v1 boots with projectDir = filesDir/workspace and shows its basename
        // under the home heading ("Workspace: workspace"). Create it so v2's
        // first frame matches, and so engine tools have a cwd.
        workspaceName = File(filesDir, "workspace").apply { mkdirs() }.name

        val rt = try {
            SofuuBridge.create(null)
        } catch (t: Throwable) {
            Log.e(TAG, "sofuu runtime failed to start", t)
            showFatal("Sofuu engine failed to start: ${t.message}")
            return
        }
        bridge = rt

        canvas = SofuuGpuSurfaceView(this)
        // The basic shell has no focusable text input yet; opening the IME on
        // every tap would be a lie about where typing goes.
        canvas.showKeyboardOnTap = false

        // A wrapper owns the insets listener: the view installs its own (it
        // forwards safe insets to the native session), so overriding it here
        // would silently disable the renderer's top clearance.
        val root = FrameLayout(this)
        root.addView(
            canvas,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
        setContentView(root)

        val rc = SofuuUiNative.nativeAttachRuntime(rt.handle(), canvas.uiSession)
        if (rc != 0) {
            Log.e(TAG, "sofuu_ui_attach_runtime failed rc=$rc")
        }

        scripts = loadUiScripts()

        // v2 keeps every inset in the scene: the root background must be
        // full-bleed cream from y=0, and the engine's own Theme::bg quad sits
        // underneath (a native top_inset would leave a bluish status-bar
        // strip). Replacing the canvas listener pins the session's top_inset
        // at 0 on purpose; v1's Compose shell behaved the same way.
        ViewCompat.setOnApplyWindowInsetsListener(canvas) { _, insets ->
            canvas.requestRender()
            insets
        }

        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
            )
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            val d = density()
            val top = bars.top / d
            val bottom = maxOf(bars.bottom, ime.bottom) / d
            val first = !insetsKnown
            insetsKnown = true
            if (!mountedOnce || first ||
                abs(top - topInsetDp) > 0.5f || abs(bottom - bottomInsetDp) > 0.5f
            ) {
                topInsetDp = top
                bottomInsetDp = bottom
                mount(initial = !mountedOnce)
            }
            insets
        }

        // The surface has no size until the first layout pass, and insets may
        // not have been dispatched yet. Mount from whichever happens last.
        root.post {
            if (!mountedOnce && insetsKnown && canvas.width > 0) mount(initial = true)
        }

        applyBarAppearance()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                // No routes/sheets exist in the basic shell yet. Once they do,
                // this asks the scene first (`sofuu.v2.back()`).
                if (askSceneBack()) return
                if (isEnabled) {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }

    override fun onResume() {
        super.onResume()
        if (!::canvas.isInitialized) return
        canvas.onResume()
        pumpHandler.removeCallbacks(pump)
        pumpHandler.post(pump)
    }

    override fun onPause() {
        pumpHandler.removeCallbacks(pump)
        if (::canvas.isInitialized) canvas.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        pumpHandler.removeCallbacksAndMessages(null)
        bridge?.let { rt ->
            // Detach before the view releases the session it owns.
            val rc = SofuuUiNative.nativeDetachRuntime(rt.handle())
            if (rc != 0) Log.w(TAG, "sofuu_ui_detach_runtime rc=$rc")
            rt.close()
        }
        bridge = null
        super.onDestroy()
    }

    // ── UI mount ─────────────────────────────────────────────────────

    private fun mount(initial: Boolean) {
        val rt = bridge ?: return
        val d = density()
        if (canvas.width <= 0 || canvas.height <= 0 || d <= 0f) return

        val w = canvas.width / d
        val h = canvas.height / d
        val dark = isDark()
        val env = "globalThis.__v2={w:$w,h:$h,top:$topInsetDp,bottom:$bottomInsetDp,dark:$dark," +
            "workspace:\"$workspaceName\"};"

        if (initial) {
            // One eval for the whole bundle; `sofuu.v2.mount()` is called by
            // app.js. Any exception renders into the scene itself instead of
            // leaving a blank surface.
            val code = "try{\n$env\n$scripts\n}catch(e){sofuu.ui.app('Sofuu').add(" +
                "sofuu.ui.text('boot_error',String(e && e.stack || e),{wrap:true,color:0xFF1C1CB9})).mount();}"
            val out = rt.eval(code)
            if (out == null) {
                Log.e(TAG, "UI mount eval failed")
                showSceneError(rt)
            }
            mountedOnce = true
        } else {
            // Remount reads __v2 from the global, so no bundle re-execution
            // (and no duplicate 16 ms event pumps).
            rt.evalNb("$env\ntry{sofuu.v2.mount();}catch(e){}")
        }
    }

    private fun showSceneError(rt: SofuuBridge) {
        rt.evalNb(
            "sofuu.ui.app('Sofuu').add(sofuu.ui.text('boot_error'," +
                "'UI bundle failed to load. See logcat tag ${TAG}.'," +
                "{wrap:true,color:0xFF1C1CB9})).mount();",
        )
    }

    private fun askSceneBack(): Boolean {
        val rt = bridge ?: return false
        val out = rt.evalNb(
            "!!(sofuu.v2 && sofuu.v2.back && sofuu.v2.back())",
        ) ?: return false
        // The envelope is {"ok":true,"result":true|false}; a failed tick
        // reports ok:false and must not swallow the system back gesture.
        return out.contains("\"result\":true")
    }

    // ── Assets / theme ───────────────────────────────────────────────

    private fun loadUiScripts(): String {
        val files = listOf(
            "v2_ui/theme.js",
            "v2_ui/icons.js",
            "v2_ui/bridge.js",
            "v2_ui/screens/home.js",
            "v2_ui/app.js",
        )
        return files.joinToString("\n") { path ->
            try {
                assets.open(path).bufferedReader().use { it.readText() }
            } catch (t: Throwable) {
                throw IllegalStateException("cannot read asset $path", t)
            }
        }
    }

    private fun applyBarAppearance() {
        val dark = isDark()
        WindowCompat.getInsetsController(window, window.decorView).apply {
            isAppearanceLightStatusBars = !dark
            isAppearanceLightNavigationBars = !dark
        }
    }

    private fun isDark(): Boolean =
        (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
            Configuration.UI_MODE_NIGHT_YES

    private fun density(): Float = resources.displayMetrics.density

    private fun showFatal(message: String) {
        val view = FrameLayout(this)
        view.setBackgroundColor(0xFFFFFBF0.toInt())
        val text = TextView(this).apply {
            text = message
            setTextColor(0xFF1C1CB9.toInt())
            textSize = 14f
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
        }
        view.addView(
            text,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER,
            ),
        )
        setContentView(view)
    }

    private companion object {
        const val TAG = "sofuu-v2"
        const val PUMP_MS = 50L
    }
}
