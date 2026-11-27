package com.sofuu.render

import android.content.Context
import android.graphics.PixelFormat
import android.opengl.GLSurfaceView
import android.text.InputType
import android.util.AttributeSet
import android.util.Log
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.BaseInputConnection
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.io.IOException
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs

/**
 * Embedded Sofuu canvas for Android — the Kotlin twin of
 * `GpuUiMetalView.swift` (PLAN-EMBEDDED-NATIVE-HOST-C-ABI Phase C4).
 *
 * Mode A: the host owns the EGL/GLES 3.0 context (this `GLSurfaceView`), the
 * native renderer in `ui_gles.c` pulls a zero-copy `SofuuUiDrawList` from the
 * session on every dirty frame, and `RENDERMODE_WHEN_DIRTY` keeps the GPU at
 * 0% once springs settle. The session's wake callback calls `requestRender()`
 * from whichever thread pushed a patch or dispatched input.
 *
 * The session lives independently of the runtime, but `attachToRuntime` MUST
 * be marshalled onto the runtime's owning thread (SofuuEngine.onEngine).
 * `sofuu_rt_free` detaches automatically; a view detached after that is a
 * safe tombstone (-2), never a use-after-free.
 *
 * Rotation: create the session once in a ViewModel and hand it to each new
 * view via [adoptSession] before the view attaches. The default constructor
 * owns its session and frees it on detach, which is the simple single-screen
 * case.
 */
class SofuuGpuSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    configJson: String? = null,
) : GLSurfaceView(context, attrs) {

    /** The live `SofuuUiSession *` (0 when creation failed). */
    @Volatile
    var uiSession: Long = SofuuUiNative.nativeSessionNew(configJson ?: DEFAULT_CONFIG)
        private set

    /** True while this view is responsible for freeing [uiSession]. */
    var ownsSession: Boolean = true
        private set

    /** Show the IME on the next touch; set false for a button-only canvas. */
    var showKeyboardOnTap: Boolean = true

    /** Optional sink for native-orchestrated semantic events. Do not set this
     *  when the JS `sofuu.ui` / Auto-Composer loop is polling the same queue. */
    var onSemanticEvent: ((String) -> Unit)? = null

    /** Frames rendered since the view was created (diagnostics / idle tests). */
    @Volatile
    var frameCount: Long = 0L
        private set

    private var wakeBound = false
    private var attachedRuntime = 0L
    private var shaderSource: String? = null
    private var lastFrameNanos = 0L
    private var lastTouchY = 0f

    @Volatile
    private var rendererStarted = false

    private val density: Float
        get() = resources.displayMetrics.density

    private val renderer = object : Renderer {
        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            rendererStarted = true
            val source = loadShaderSource()
            val rc = if (uiSession != 0L && source != null) {
                SofuuUiNative.nativeGlesCreate(uiSession, source)
            } else {
                -1
            }
            if (rc != 0) {
                Log.e(TAG, "GLES renderer init failed (rc=$rc)")
            }
            lastFrameNanos = 0L
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            val scale = density
            if (width > 0 && height > 0 && scale > 0f) {
                SofuuUiNative.nativeGlesResize(width / scale, height / scale, scale)
            }
            requestRender()
        }

        override fun onDrawFrame(gl: GL10?) {
            if (uiSession == 0L || width == 0 || height == 0) return
            frameCount += 1
            val now = System.nanoTime()
            val dt = if (lastFrameNanos == 0L) {
                1f / 60f
            } else {
                ((now - lastFrameNanos) / 1_000_000_000.0).toFloat().coerceIn(0f, 0.25f)
            }
            lastFrameNanos = now

            val scale = density
            if (scale <= 0f) return
            val needsNext = SofuuUiNative.nativeGlesDraw(
                uiSession,
                dt,
                width / scale,
                height / scale,
                scale,
            )
            if (needsNext > 0) {
                // Springs / kinetic scroll are live: keep the GL thread warm.
                requestRender()
            }
        }
    }

    init {
        setEGLContextClientVersion(3)
        setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        preserveEGLContextOnPause = true
        holder.setFormat(PixelFormat.TRANSLUCENT)
        isFocusable = true
        isFocusableInTouchMode = true
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY

        if (uiSession == 0L) {
            Log.e(TAG, "sofuu_ui_session_new failed")
        } else {
            wakeBound = SofuuUiNative.nativeSetWakeView(uiSession, this) == 0
        }

        ViewCompat.setOnApplyWindowInsetsListener(this) { _, insets ->
            applyInsets(insets)
            insets
        }
    }

    // ── Session lifecycle ────────────────────────────────────────────

    /**
     * Reuse a session owned by the host (typically held in a ViewModel so a
     * rotation does not re-run the JS mount). Must be called before the GL
     * renderer starts; after that the view's frame loop is already bound.
     */
    fun adoptSession(session: Long, takeOwnership: Boolean = false) {
        if (rendererStarted) {
            throw IllegalStateException("adoptSession must run before the view attaches")
        }
        if (session == uiSession) {
            ownsSession = takeOwnership
            return
        }
        if (wakeBound) {
            SofuuUiNative.nativeSetWakeView(uiSession, null)
            wakeBound = false
        }
        if (ownsSession && uiSession != 0L) {
            SofuuUiNative.nativeSessionFree(uiSession)
        }
        uiSession = session
        ownsSession = takeOwnership
        if (session != 0L) {
            wakeBound = SofuuUiNative.nativeSetWakeView(session, this) == 0
        }
    }

    /** Free the session. Only valid after the GL thread has stopped. */
    fun releaseSession() {
        if (uiSession == 0L) return
        if (wakeBound) {
            SofuuUiNative.nativeSetWakeView(uiSession, null)
            wakeBound = false
        }
        if (ownsSession) {
            SofuuUiNative.nativeSessionFree(uiSession)
            ownsSession = false
        }
        uiSession = 0L
    }

    /**
     * Bind this canvas to the engine runtime. MUST run on the runtime's owning
     * thread; use `SofuuEngine.attachUISession(...)` which marshals it there.
     */
    fun attachToRuntime(runtime: Long): Int {
        if (runtime == 0L || uiSession == 0L) return -2
        attachedRuntime = runtime
        return SofuuUiNative.nativeAttachRuntime(runtime, uiSession)
    }

    /** Unbind without freeing either side. Runtime thread only. */
    fun detachFromRuntime(): Int {
        if (attachedRuntime == 0L) return 0
        val rc = SofuuUiNative.nativeDetachRuntime(attachedRuntime)
        attachedRuntime = 0L
        return rc
    }

    // ── UI lifecycle ─────────────────────────────────────────────────

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        requestApplyInsets()
        requestRender()
    }

    override fun onDetachedFromWindow() {
        if (wakeBound) {
            SofuuUiNative.nativeSetWakeView(uiSession, null)
            wakeBound = false
        }
        // super stops the GL thread (requestExitAndWait); only then is it safe
        // to drop the session the render thread was polling.
        super.onDetachedFromWindow()
        SofuuUiNative.nativeGlesDestroy()
        rendererStarted = false
        if (ownsSession && uiSession != 0L) {
            SofuuUiNative.nativeSessionFree(uiSession)
            ownsSession = false
        }
        uiSession = 0L
    }

    override fun onCheckIsTextEditor(): Boolean = uiSession != 0L

    // ── Insets (status bar / notch / IME) ────────────────────────────

    private fun applyInsets(insets: WindowInsetsCompat) {
        if (uiSession == 0L) return
        val d = density
        if (d <= 0f) return
        val bars = insets.getInsets(
            WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
        )
        val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
        val top = maxOf(bars.top, ime.top) / d
        val bottom = maxOf(bars.bottom, ime.bottom) / d
        SofuuUiNative.nativeSetSafeInsets(uiSession, top, bottom)
        requestRender()
    }

    /** Manual inset override for hosts that own the window (e.g. split-screen). */
    fun setSafeInsets(topLogicalPx: Float, bottomLogicalPx: Float) {
        if (uiSession == 0L) return
        SofuuUiNative.nativeSetSafeInsets(uiSession, topLogicalPx, bottomLogicalPx)
        requestRender()
    }

    // ── Touch / scroll ───────────────────────────────────────────────

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (uiSession == 0L) return false
        val d = density
        if (d <= 0f) return false
        val x = event.x / d
        val y = event.y / d
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchY = y
                dispatchPointer(INPUT_POINTER_DOWN, x, y)
                if (showKeyboardOnTap) showKeyboard()
            }

            MotionEvent.ACTION_MOVE -> {
                dispatchPointer(INPUT_POINTER_MOVE, x, y)
                val dy = y - lastTouchY
                if (abs(dy) > 0.5f) {
                    dispatchScroll(x, y, 0f, dy)
                    lastTouchY = y
                }
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                dispatchPointer(INPUT_POINTER_UP, x, y)
            }

            else -> return super.onTouchEvent(event)
        }
        return true
    }

    private fun dispatchPointer(kind: Int, x: Float, y: Float) {
        SofuuUiNative.nativeDispatchPointer(
            uiSession, kind, x, y, 0f, 0f, density, 0, 0,
        )
    }

    private fun dispatchScroll(x: Float, y: Float, dx: Float, dy: Float) {
        SofuuUiNative.nativeDispatchPointer(
            uiSession, INPUT_SCROLL, x, y, dx, dy, density, 0, 0,
        )
    }

    // ── Hardware keyboard / IME ──────────────────────────────────────

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (uiSession == 0L) return super.onKeyDown(keyCode, event)
        val key = when (keyCode) {
            KeyEvent.KEYCODE_DEL -> "Backspace"
            KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_NUMPAD_ENTER -> "Enter"
            KeyEvent.KEYCODE_ESCAPE -> "Escape"
            KeyEvent.KEYCODE_DPAD_LEFT -> "ArrowLeft"
            KeyEvent.KEYCODE_DPAD_RIGHT -> "ArrowRight"
            KeyEvent.KEYCODE_DPAD_UP -> "ArrowUp"
            KeyEvent.KEYCODE_DPAD_DOWN -> "ArrowDown"
            else -> null
        }
        if (key != null) {
            SofuuUiNative.nativeDispatchKey(uiSession, key, modifiersOf(event))
            return true
        }
        val unicode = event.unicodeChar
        if (unicode != 0 && !event.isCtrlPressed && !event.isAltPressed && !event.isMetaPressed) {
            SofuuUiNative.nativeDispatchText(uiSession, String(Character.toChars(unicode)))
            return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onCreateInputConnection(outAttrs: EditorInfo): InputConnection {
        outAttrs.inputType = InputType.TYPE_CLASS_TEXT or
            InputType.TYPE_TEXT_FLAG_MULTI_LINE or
            InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        outAttrs.imeOptions = EditorInfo.IME_ACTION_SEND
        return object : BaseInputConnection(this@SofuuGpuSurfaceView, false) {
            override fun commitText(text: CharSequence?, newCursorPosition: Int): Boolean {
                if (text != null && text.isNotEmpty()) {
                    SofuuUiNative.nativeDispatchText(uiSession, text.toString())
                }
                return true
            }

            override fun deleteSurroundingText(beforeLength: Int, afterLength: Int): Boolean {
                repeat(beforeLength) {
                    SofuuUiNative.nativeDispatchKey(uiSession, "Backspace", 0)
                }
                return true
            }

            override fun performEditorAction(actionCode: Int): Boolean {
                SofuuUiNative.nativeDispatchKey(uiSession, "Enter", 0)
                return true
            }
        }
    }

    private fun showKeyboard() {
        requestFocus()
        val imm = context.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.showSoftInput(this, InputMethodManager.SHOW_IMPLICIT)
    }

    private fun modifiersOf(event: KeyEvent): Int {
        var mods = 0
        if (event.isShiftPressed) mods = mods or MOD_SHIFT
        if (event.isCtrlPressed) mods = mods or MOD_CTRL
        if (event.isAltPressed) mods = mods or MOD_ALT
        if (event.isMetaPressed) mods = mods or MOD_META
        return mods
    }

    // ── Shader asset ─────────────────────────────────────────────────

    private fun loadShaderSource(): String? {
        shaderSource?.let { return it }
        return try {
            val text = context.assets.open(SHADER_ASSET).use { input ->
                input.readBytes().toString(Charsets.UTF_8)
            }
            shaderSource = text
            text
        } catch (e: IOException) {
            Log.e(TAG, "missing $SHADER_ASSET — Gradle should sync it from crates/sofuu-gpu", e)
            null
        }
    }

    /** Drain one semantic event (native-orchestrated hosts only). */
    fun pollSemanticEvent(): String? {
        if (uiSession == 0L) return null
        return SofuuUiNative.nativePollEventJson(uiSession)?.also { json ->
            onSemanticEvent?.invoke(json)
        }
    }

    companion object {
        private const val TAG = "SofuuGpuSurfaceView"
        private const val SHADER_ASSET = "ui_shaders_es3.glsl"
        const val DEFAULT_CONFIG = """{"title":"Sofuu","theme":"light"}"""

        // Mirrors include/sofuu_embed.h (SOFUU_INPUT_* and SOFUU_MOD_*).
        const val INPUT_POINTER_MOVE = 1
        const val INPUT_POINTER_DOWN = 2
        const val INPUT_POINTER_UP = 3
        const val INPUT_SCROLL = 4
        const val INPUT_KEY_DOWN = 5
        const val INPUT_TEXT_INPUT = 6
        const val INPUT_RESIZE = 7
        const val INPUT_CLOSE_REQUESTED = 8
        const val MOD_SHIFT = 1
        const val MOD_CTRL = 2
        const val MOD_ALT = 4
        const val MOD_META = 8

        /** Create a session the host owns (e.g. across rotation in a ViewModel). */
        fun createSession(configJson: String? = null): Long =
            SofuuUiNative.nativeSessionNew(configJson ?: DEFAULT_CONFIG)

        /** Free a host-owned session; no-op for 0. */
        fun freeSession(session: Long) {
            if (session != 0L) SofuuUiNative.nativeSessionFree(session)
        }
    }
}

/**
 * The raw `sofuu_ui_*` JNI surface. One object keeps the exported symbol
 * names stable (`Java_com_sofuu_render_SofuuUiNative_*`) and lets both the
 * view and the engine bridge use the same thin funnels.
 */
internal object SofuuUiNative {
    init {
        System.loadLibrary("sofuu_jni")
    }

    external fun nativeSessionNew(configJson: String?): Long
    external fun nativeSessionFree(session: Long)
    external fun nativeSetWakeView(session: Long, view: Any?): Int
    external fun nativeAttachRuntime(runtime: Long, session: Long): Int
    external fun nativeDetachRuntime(runtime: Long): Int
    external fun nativeSetSafeInsets(session: Long, topLogicalPx: Float, bottomLogicalPx: Float): Int
    external fun nativeGetAtlas(session: Long, outMeta: IntArray): ByteArray?
    external fun nativePushPatchJson(session: Long, patchJson: String?): Int
    external fun nativeIngestAgentEventJson(session: Long, eventJson: String?): Int
    external fun nativeDispatchPointer(
        session: Long,
        kind: Int,
        x: Float,
        y: Float,
        dx: Float,
        dy: Float,
        scale: Float,
        button: Int,
        modifiers: Int,
    ): Int
    external fun nativeDispatchText(session: Long, text: String?): Int
    external fun nativeDispatchKey(session: Long, key: String?, modifiers: Int): Int
    external fun nativePollEventJson(session: Long): String?
    external fun nativeGlesCreate(session: Long, shaderSource: String?): Int
    external fun nativeGlesResize(widthLogical: Float, heightLogical: Float, scale: Float)
    external fun nativeGlesDraw(
        session: Long,
        dtSec: Float,
        widthLogical: Float,
        heightLogical: Float,
        scale: Float,
    ): Int
    external fun nativeGlesDestroy()
}
