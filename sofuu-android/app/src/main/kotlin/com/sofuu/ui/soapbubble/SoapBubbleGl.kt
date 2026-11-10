package com.sofuu.ui.soapbubble

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Shader
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES30
import android.opengl.GLUtils
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Process
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

internal data class SoapBubbleRegion(
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
)

internal class SoapBubbleFrame(
    val image: ImageBitmap,
    val regions: Map<Long, SoapBubbleRegion>,
    val generation: Long,
)

/**
 * One shared OpenGL ES 3.0 context draws every soap bubble on the page into a
 * single atlas, then hands the atlas to Compose. This is the Android shape of
 * the component's shared WebGL context + per-surface copy: one program, one
 * film table, one noise volume, one read-back per frame.
 */
internal object SoapBubbleGl {
    private const val TAG = "SoapBubbleGl"
    private const val ATLAS_START = 1024
    private const val ATLAS_MAX = 2048
    private const val FRAME_NS = 16_666_667L
    private const val EGL_OPENGL_ES3_BIT_KHR = 0x40

    val frame = mutableStateOf<SoapBubbleFrame?>(null)
    val ready = mutableStateOf(false)
    val failed = mutableStateOf(false)

    private val controllers = CopyOnWriteArrayList<SoapBubbleController>()
    private val thread = HandlerThread("mejiri-soap-bubble").apply { start() }
    private val glHandler = Handler(thread.looper)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val running = java.util.concurrent.atomic.AtomicBoolean(false)

    @Volatile private var appActive = true

    private var display: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var context: EGLContext = EGL14.EGL_NO_CONTEXT
    private var surface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var program = 0
    private var uniform = ProgramUniforms(0)
    private var filmTexture = 0
    private var noiseTexture = 0
    private var emptyTexture = 0
    private var lightBackdrop = 0
    private var darkBackdrop = 0
    private var fbo = 0
    private var colorTexture = 0
    private var atlasWidth = ATLAS_START
    private var atlasHeight = ATLAS_START
    private var bitmapA: Bitmap? = null
    private var bitmapB: Bitmap? = null
    private var imageA: ImageBitmap? = null
    private var imageB: ImageBitmap? = null
    private var bitmapIndex = 0
    private var generation = 0L
    private var lastNanos = 0L
    private var initialized = false
    private var initFailed = false

    // Reused every frame: the old path allocated a direct buffer and an IntArray
    // at 30fps, which made the collector do more work than the shader.
    private var readBuffer: ByteBuffer? = null
    private var readInts: IntBuffer? = null
    private var pixels: IntArray = IntArray(0)

    private var fpsFrames = 0
    private var fpsStartNanos = 0L
    private var workNanos = 0L
    private var readNanos = 0L
    private var copyNanos = 0L
    private var displayPrioritySet = false
    private val visibleControllers = ArrayList<SoapBubbleController>(32)
    private val loop = object : Runnable {
        override fun run() {
            if (!displayPrioritySet) {
                Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY)
                displayPrioritySet = true
            }
            if (!appActive) {
                running.set(false)
                return
            }
            visibleControllers.clear()
            for (c in controllers) if (c.visible) visibleControllers.add(c)
            if (visibleControllers.isEmpty()) {
                running.set(false)
                return
            }
            if (!ensureGl()) {
                running.set(false)
                return
            }
            val now = System.nanoTime()
            val dt = if (lastNanos == 0L) 1f / 60f
            else min(1f / 20f, (now - lastNanos) / 1_000_000_000f)
            lastNanos = now

            visibleControllers.forEach { it.tick(dt) }
            val workStart = System.nanoTime()
            render(visibleControllers)
            workNanos += System.nanoTime() - workStart

            fpsFrames++
            if (fpsStartNanos == 0L) fpsStartNanos = now
            val elapsed = now - fpsStartNanos
            if (elapsed >= 5_000_000_000L) {
                val frames = max(1, fpsFrames)
                android.util.Log.d(
                    TAG,
                    "fps " + String.format("%.1f", fpsFrames * 1e9 / elapsed) +
                        " · " + visibleControllers.size + " bubbles" +
                        " · work " + String.format("%.1f", workNanos / 1e6 / frames) + "ms" +
                        " · read " + String.format("%.1f", readNanos / 1e6 / frames) + "ms" +
                        " · copy " + String.format("%.1f", copyNanos / 1e6 / frames) + "ms",
                )
                fpsFrames = 0
                fpsStartNanos = now
                workNanos = 0
                readNanos = 0
                copyNanos = 0
            }

            // Keep the cadence at 60Hz without letting render time accumulate
            // into drift: subtract the work just done from the next deadline.
            val spent = System.nanoTime() - now
            val delayMs = max(0L, (FRAME_NS - spent) / 1_000_000L)
            glHandler.postDelayed(this, delayMs)
        }
    }

    fun register(controller: SoapBubbleController) {
        if (!controllers.contains(controller)) controllers.add(controller)
        start()
    }

    fun unregister(controller: SoapBubbleController) {
        controllers.remove(controller)
    }

    fun invalidate() = start()

    fun setAppActive(active: Boolean) {
        appActive = active
        lastNanos = 0L
        if (active) start() else {
            running.set(false)
            glHandler.removeCallbacks(loop)
        }
    }

    private fun start() {
        if (running.compareAndSet(false, true)) {
            glHandler.post(loop)
        }
    }

    /* ------------------------------------------------------------ */
    /* Context                                                       */
    /* ------------------------------------------------------------ */

    private fun ensureGl(): Boolean {
        if (initialized) return !initFailed
        if (initFailed) return false

        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (display == EGL14.EGL_NO_DISPLAY) return fail("no EGL display")
        val version = IntArray(2)
        if (!EGL14.eglInitialize(display, version, 0, version, 1)) return fail("eglInitialize")

        val configs = arrayOfNulls<EGLConfig>(1)
        val configCount = IntArray(1)
        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE,
        )
        if (!EGL14.eglChooseConfig(display, configAttribs, 0, configs, 0, 1, configCount, 0) ||
            configCount[0] == 0
        ) return fail("no ES3 EGL config")

        context = EGL14.eglCreateContext(
            display,
            configs[0],
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE),
            0,
        )
        if (context == EGL14.EGL_NO_CONTEXT) return fail("no ES3 context")

        surface = EGL14.eglCreatePbufferSurface(
            display,
            configs[0],
            intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE),
            0,
        )
        if (surface == EGL14.EGL_NO_SURFACE) return fail("no pbuffer")
        if (!EGL14.eglMakeCurrent(display, surface, surface, context)) return fail("eglMakeCurrent")

        val versionString = GLES30.glGetString(GLES30.GL_VERSION) ?: ""
        if (!versionString.contains("OpenGL ES 3")) return fail("GLES3 unavailable: $versionString")

        val vs = compile(GLES30.GL_VERTEX_SHADER, SOAP_BUBBLE_VERTEX)
        val fs = compile(GLES30.GL_FRAGMENT_SHADER, SOAP_BUBBLE_FRAGMENT)
        if (vs == 0 || fs == 0) return fail("shader compile failed")
        program = GLES30.glCreateProgram()
        GLES30.glAttachShader(program, vs)
        GLES30.glAttachShader(program, fs)
        GLES30.glLinkProgram(program)
        GLES30.glDeleteShader(vs)
        GLES30.glDeleteShader(fs)
        val linked = IntArray(1)
        GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0)
        if (linked[0] == 0) return fail("link failed: " + GLES30.glGetProgramInfoLog(program))

        uniform = ProgramUniforms(program)
        GLES30.glUseProgram(program)
        val quad = IntArray(1)
        GLES30.glGenVertexArrays(1, quad, 0)
        GLES30.glBindVertexArray(quad[0])
        val buffer = FloatBuffer.allocate(6)
        buffer.put(floatArrayOf(-1f, -1f, 3f, -1f, -1f, 3f)).position(0)
        val vbo = IntArray(1)
        GLES30.glGenBuffers(1, vbo, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo[0])
        GLES30.glBufferData(
            GLES30.GL_ARRAY_BUFFER,
            6 * 4,
            buffer,
            GLES30.GL_STATIC_DRAW,
        )
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 0, 0)

        filmTexture = uploadFilmTable()
        noiseTexture = uploadNoise()
        emptyTexture = uploadEmpty()
        lightBackdrop = uploadBackdrop(0xFFFFFF, 0xFAF7F2)
        darkBackdrop = uploadBackdrop(0x211D19, 0x262219)
        if (!createAtlas(ATLAS_START)) return fail("atlas failed")

        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)

        // Joins and see-through are not used by the app's surfaces; set them
        // once instead of re-sending eight zero uniforms per bubble per frame.
        repeat(4) { i ->
            uniform.f4("uJoin$i", 0f, 0f, 0f, 0f)
            uniform.f4("uJoinTint$i", 0f, 0f, 0f, 0f)
        }
        uniform.f4("uJoinCorners", 0f, 0f, 0f, 0f)
        uniform.f4("uJoinAmounts", 0f, 0f, 0f, 0f)
        uniform.f1("uSeeThrough", 0f)
        uniform.i1("uHasCovered", 0)
        uniform.i1("uHasBackdrop", 1)
        uniform.f2("uBackdropOrigin", 0.5f, 0.5f)
        uniform.f2("uBackdropSize", 64f, 64f)

        initialized = true
        android.util.Log.i(TAG, "soap bubble GL ready: $versionString")
        mainHandler.post {
            ready.value = true
            failed.value = false
        }
        return true
    }

    private fun fail(reason: String): Boolean {
        initFailed = true
        android.util.Log.e(TAG, "soap bubble GL unavailable: $reason")
        mainHandler.post {
            failed.value = true
            ready.value = false
        }
        return false
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES30.glCreateShader(type)
        GLES30.glShaderSource(shader, source)
        GLES30.glCompileShader(shader)
        val ok = IntArray(1)
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            android.util.Log.e(TAG, "shader compile failed: " + GLES30.glGetShaderInfoLog(shader))
            GLES30.glDeleteShader(shader)
            return 0
        }
        return shader
    }

    /* ------------------------------------------------------------ */
    /* Textures                                                      */
    /* ------------------------------------------------------------ */

    private fun uploadFilmTable(): Int {
        val table = SoapBubbleTables.filmTable()
        val buffer = ByteBuffer.allocateDirect(table.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(table).position(0)
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ids[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D,
            0,
            GLES30.GL_RGBA16F,
            SoapBubbleTables.FILM_COLUMNS,
            SoapBubbleTables.FILM_ROWS,
            0,
            GLES30.GL_RGBA,
            GLES30.GL_FLOAT,
            buffer,
        )
        return ids[0]
    }

    private fun uploadNoise(): Int {
        val volume = SoapBubbleTables.noiseVolume()
        val buffer = ByteBuffer.allocateDirect(volume.size).order(ByteOrder.nativeOrder())
        buffer.put(volume).position(0)
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, ids[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        for (wrap in intArrayOf(
            GLES30.GL_TEXTURE_WRAP_S,
            GLES30.GL_TEXTURE_WRAP_T,
            GLES30.GL_TEXTURE_WRAP_R,
        )) GLES30.glTexParameteri(GLES30.GL_TEXTURE_3D, wrap, GLES30.GL_REPEAT)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        GLES30.glTexImage3D(
            GLES30.GL_TEXTURE_3D,
            0,
            GLES30.GL_R8,
            SoapBubbleTables.NOISE_SIZE,
            SoapBubbleTables.NOISE_SIZE,
            SoapBubbleTables.NOISE_SIZE,
            0,
            GLES30.GL_RED,
            GLES30.GL_UNSIGNED_BYTE,
            buffer,
        )
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        return ids[0]
    }

    private fun uploadEmpty(): Int {
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ids[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, 1, 1, 0,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE,
            ByteBuffer.allocateDirect(4),
        )
        return ids[0]
    }

    /** A 64 x 64 card wash used as the mirrored surroundings behind a control.
     *  Light pages get the reference's near-white wash; dark pages get a dark
     *  wash so the film reads as dark glass and light text stays legible. */
    private fun uploadBackdrop(topColor: Int, bottomColor: Int): Int {
        val top = topColor
        val bottom = bottomColor
        val bitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply {
            shader = LinearGradient(
                0f, 0f, 0f, 64f, top, bottom, Shader.TileMode.CLAMP
            )
        }
        canvas.drawRect(0f, 0f, 64f, 64f, paint)
        val ids = IntArray(1)
        GLES30.glGenTextures(1, ids, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, ids[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        bitmap.recycle()
        return ids[0]
    }

    private fun createAtlas(size: Int): Boolean {
        if (fbo != 0) {
            GLES30.glDeleteFramebuffers(1, intArrayOf(fbo), 0)
            GLES30.glDeleteTextures(1, intArrayOf(colorTexture), 0)
        }
        atlasWidth = size
        atlasHeight = size
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        colorTexture = tex[0]
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, colorTexture)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA,
            atlasWidth, atlasHeight, 0, GLES30.GL_RGBA,
            GLES30.GL_UNSIGNED_BYTE, null,
        )
        val ids = IntArray(1)
        GLES30.glGenFramebuffers(1, ids, 0)
        fbo = ids[0]
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER,
            GLES30.GL_COLOR_ATTACHMENT0,
            GLES30.GL_TEXTURE_2D,
            colorTexture,
            0,
        )
        val status = GLES30.glCheckFramebufferStatus(GLES30.GL_FRAMEBUFFER)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        if (status != GLES30.GL_FRAMEBUFFER_COMPLETE) return false
        bitmapA = Bitmap.createBitmap(atlasWidth, atlasHeight, Bitmap.Config.ARGB_8888)
        bitmapB = Bitmap.createBitmap(atlasWidth, atlasHeight, Bitmap.Config.ARGB_8888)
        // Wrap once. Creating a new ImageBitmap wrapper every frame makes
        // Compose/Skia treat the atlas as a new image and upload it afresh.
        imageA = bitmapA!!.asImageBitmap()
        imageB = bitmapB!!.asImageBitmap()
        val buffer = ByteBuffer.allocateDirect(atlasWidth * atlasHeight * 4)
            .order(ByteOrder.nativeOrder())
        readBuffer = buffer
        readInts = buffer.asIntBuffer()
        pixels = IntArray(atlasWidth * atlasHeight)
        return true
    }

    /* ------------------------------------------------------------ */
    /* Drawing                                                       */
    /* ------------------------------------------------------------ */

    private class DrawItem {
        lateinit var controller: SoapBubbleController
        var dpr = 1f
        var canvasW = 0
        var canvasH = 0
        var centerX = 0f
        var centerY = 0f
        var radius = 0f
        var hostW = 0f
        var hostH = 0f
        var em = 1f
        var x = 0
        var y = 0
        val motion = MotionSnapshot()
    }

    private val itemPool = ArrayList<DrawItem>(32)
    private val items = ArrayList<DrawItem>(32)

    private fun render(itemsIn: List<SoapBubbleController>) {
        items.clear()
        for (c in itemsIn) {
            val maxDpr = c.settings.maxDpr
            val dpr = min(1f, maxDpr / max(1f, c.density))
            val bleed = max(12f * c.density, c.fontPx)
            val hostW = if (c.hostWidthPx > 0f) c.hostWidthPx else max(1f, c.widthPx)
            val hostH = if (c.hostHeightPx > 0f) c.hostHeightPx else max(1f, c.heightPx)
            val canvasWpx = if (c.canvasWidthPx > 0f) c.canvasWidthPx else hostW + bleed * 2f
            val canvasHpx = if (c.canvasHeightPx > 0f) c.canvasHeightPx else hostH + bleed * 2f
            val centerX = if (c.centerXPx >= 0f) c.centerXPx else canvasWpx / 2f
            val centerY = if (c.centerYPx >= 0f) c.centerYPx else canvasHpx / 2f
            val canvasW = ceil(canvasWpx * dpr).toInt().coerceAtLeast(1)
            val canvasH = ceil(canvasHpx * dpr).toInt().coerceAtLeast(1)
            if (canvasW > atlasWidth || canvasH > atlasHeight) continue
            val index = items.size
            val item = if (index < itemPool.size) itemPool[index] else DrawItem().also { itemPool.add(it) }
            item.controller = c
            item.dpr = dpr
            item.canvasW = canvasW
            item.canvasH = canvasH
            item.centerX = centerX * dpr
            item.centerY = centerY * dpr
            item.radius = min(hostW, hostH) / 2f * dpr
            item.hostW = hostW
            item.hostH = hostH
            item.em = c.fontPx / 16f
            item.x = 0
            item.y = 0
            items.add(item)
        }
        items.sortByDescending { it.canvasH }

        var usedW = 0
        var usedH = 0
        var overflow = false
        var placed = 0
        run {
            var x = 0
            var y = 0
            var shelf = 0
            for (item in items) {
                if (x + item.canvasW > atlasWidth) {
                    x = 0
                    y += shelf
                    shelf = 0
                }
                if (y + item.canvasH > atlasHeight) {
                    overflow = true
                    break
                }
                item.x = x
                item.y = y
                x += item.canvasW
                shelf = max(shelf, item.canvasH)
                usedW = max(usedW, x)
                usedH = max(usedH, y + item.canvasH)
                placed++
            }
        }
        if (placed == 0) return
        val drawn = items.subList(0, placed)

        if (overflow && atlasWidth < ATLAS_MAX) {
            if (createAtlas(min(ATLAS_MAX, atlasWidth * 2))) {
                render(itemsIn)
            }
            return
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        // Only the packed rectangle is read back, so only that band needs
        // clearing; clearing all 1024² every frame was pure waste.
        GLES30.glEnable(GLES30.GL_SCISSOR_TEST)
        GLES30.glScissor(0, 0, usedW, usedH)
        GLES30.glClearColor(0f, 0f, 0f, 0f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        GLES30.glDisable(GLES30.GL_SCISSOR_TEST)
        GLES30.glUseProgram(program)

        // Samplers are fixed once; backdrops alternate between the two washes.
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, filmTexture)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_3D, noiseTexture)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE3)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, emptyTexture)
        uniform.i1("uFilmTable", 1)
        uniform.i1("uNoise", 2)
        uniform.i1("uCovered", 3)
        uniform.i1("uBackdrop", 0)

        for (item in drawn) {
            item.controller.snapshotInto(item.motion)
            draw(item)
        }

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        readBack(drawn, usedW, usedH)
    }

    private fun draw(item: DrawItem) {
        val c = item.controller
        val s = c.settings
        val paint = c.paint
        val motion = item.motion
        val radius = item.radius
        val em = item.em
        val cssRadius = min(item.hostW, item.hostH) / 2f
        val centerX = item.x + item.centerX
        val centerY = item.y + item.centerY

        GLES30.glViewport(item.x, item.y, item.canvasW, item.canvasH)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, if (s.dark) darkBackdrop else lightBackdrop)

        uniform.f2("uResolution", atlasWidth.toFloat(), atlasHeight.toFloat())
        uniform.f2("uCenter", centerX, centerY)
        uniform.f1("uRadius", radius)
        uniform.f2("uExtent", item.hostW / 2f / cssRadius, item.hostH / 2f / cssRadius)
        uniform.f1("uCorner", min(1f, s.radius * em / cssRadius))
        uniform.f1("uBevel", s.bevel * em / cssRadius)
        uniform.f4("uModeBasis", motion.modeBasis)
        uniform.f2("uSquash", motion.squash[0], motion.squash[1])
        uniform.f1("uPressScale", motion.pressScale)
        uniform.f1("uTime", motion.time)
        uniform.f1("uFlow", motion.flow)
        uniform.f1("uThickness", s.thickness)
        uniform.f1("uIridescence", s.iridescence)
        uniform.f1("uOilOpacity", s.oilOpacity)
        uniform.f1("uDrainage", s.drainage)
        uniform.f1("uSwirl", s.swirl)
        uniform.f1("uFilmScale", s.filmScale)
        uniform.f1("uReflectivity", s.reflectivity)
        uniform.f1("uMirage", s.mirage)
        uniform.f1("uCoveredBend", s.seeThroughBend)
        uniform.f1("uFloorLight", s.floorLight)
        uniform.f1("uReach", s.reach)
        uniform.f1("uRefraction", s.refraction)
        uniform.f1("uGlass", s.glass)
        uniform.f1("uDispersion", s.dispersion)
        uniform.f1("uSeed", (s.seed.toInt() % 97) * 0.731f)
        uniform.i1("uEnvironment", environmentIndex(s.environment))
        envRotation(motion.envRotation, motion.tilt, motion.time, s.tilt)
        uniform.mat3("uEnvRotation", motion.envRotation)
        uniform.f1("uOilCoverage", s.oilCoverage)
        uniform.f1("uOilEdge", s.oilEdge * em * item.dpr)
        uniform.f1("uOilFade", s.oilEdgeFade)
        uniform.f1("uClearCenter", s.clearCenter)
        val long = max(item.hostW, item.hostH) / (16f * em)
        val large = min(1f, max(0f, (long - 8f) / 6f))
        uniform.f1("uOilLarge", large * large * (3f - 2f * large))

        val tint = paint.tint
        uniform.f4("uTint", tint[0], tint[1], tint[2], tint[3])
        uniform.f4("uSoapColor", paint.soap)
        uniform.f4("uSoapShadeColor", paint.shade)
        uniform.f4("uSoapHighlightColor", paint.highlight)
        uniform.i1("uCustomSoapPalette", if (paint.customPalette) 1 else 0)
        uniform.f4("uSoapPaletteRed", paint.paletteRed)
        uniform.f4("uSoapPaletteGold", paint.paletteGold)
        uniform.f4("uSoapPaletteGreen", paint.paletteGreen)
        uniform.f4("uSoapPaletteCyan", paint.paletteCyan)
        uniform.f4("uSoapPaletteBlue", paint.paletteBlue)
        uniform.f4("uSoapPaletteViolet", paint.paletteViolet)
        uniform.i1("uLightTheme", if (s.dark) 0 else 1)
        uniform.i1("uWindowLight", if (s.windowLight) 1 else 0)
        uniform.i1("uKeyLight", if (s.keyLight) 1 else 0)
        uniform.i1("uWindowTwin", if (s.windowTwin) 1 else 0)
        uniform.f1("uWindowSize", s.windowSize)
        uniform.f1("uWindowWidth", s.windowWidth)
        uniform.f1("uWindowGradient", s.windowGradient)
        uniform.f1("uTopFade", s.topFade)
        uniform.f1("uTopTail", s.topTail)
        uniform.f1("uTopLength", s.topLength)
        uniform.f1("uWindowHead", s.windowHead)
        uniform.f1("uWindowTail", s.windowTail)
        uniform.i1("uGlintShape", glintIndex(s.glintShape))
        uniform.f1("uGlintSize", s.glintSize)
        uniform.f1("uGlintIntensity", s.glintIntensity)
        uniform.f2("uGlintOffset", s.glintDistance, s.glintHeight)
        uniform.i1("uWindowStyle", if (s.windowStyle == "bubble") 1 else 0)
        uniform.f1("uWindowLength", s.windowLength)
        uniform.f1("uWindowMargin", s.windowMargin)
        uniform.f1("uWindowBend", s.windowBend)
        uniform.f1("uWindowIntensity", s.windowIntensity)
        uniform.i1("uTwinShape", twinIndex(s.windowTwinShape))
        uniform.f1("uSpark", motion.spark)
        uniform.i1("uBottomShade", if (s.bottomShade) 1 else 0)
        val shadeColor = paint.shadeColor
        uniform.f4("uShadeColor", shadeColor[0], shadeColor[1], shadeColor[2], shadeColor[3])

        // One radius in page UV. The app's content column is 430dp; the wash
        // stands in for that page, so the rim reflection samples a broad area
        // instead of the bubble's own few pixels.
        val pageWidth = 430f * min(c.density, s.maxDpr)
        uniform.f2("uBackdropScale", radius / pageWidth, radius / pageWidth)

        val layer = c.layer
        if (layer != null) {
            uniform.i1("uHasLayer", 1)
            // Layer geometry in radii around the host centre.
            uniform.f4(
                "uLayerRect",
                (layer.x + layer.width / 2f - item.centerX) / cssRadius,
                -(layer.y + layer.height / 2f - item.centerY) / cssRadius,
                layer.width / 2f / cssRadius,
                layer.height / 2f / cssRadius,
            )
            uniform.f1("uLayerCorner", layer.radius / cssRadius)
            uniform.f1(
                "uLayerFill",
                (layer.x + layer.width * layer.fill.coerceIn(0f, 1f) - item.centerX) / cssRadius,
            )
            uniform.f4("uLayerTrack", cssColor(layer.track))
            uniform.f4("uLayerFrom", cssColor(layer.from))
            uniform.f4("uLayerTo", cssColor(layer.to ?: layer.from))
        } else {
            uniform.i1("uHasLayer", 0)
            uniform.f4("uLayerRect", 0f, 0f, 0f, 0f)
            uniform.f1("uLayerCorner", 0f)
            uniform.f1("uLayerFill", 0f)
            uniform.f4("uLayerTrack", 0f, 0f, 0f, 0f)
            uniform.f4("uLayerFrom", 0f, 0f, 0f, 0f)
            uniform.f4("uLayerTo", 0f, 0f, 0f, 0f)
        }

        GLES30.glDrawArrays(GLES30.GL_TRIANGLES, 0, 3)
    }

    private fun readBack(items: List<DrawItem>, usedW: Int, usedH: Int) {
        if (usedW <= 0 || usedH <= 0) return
        val buffer = readBuffer ?: return
        val ints = readInts ?: return
        val target = if (bitmapIndex == 0) bitmapA else bitmapB
        val image = if (bitmapIndex == 0) imageA else imageB
        bitmapIndex = 1 - bitmapIndex
        if (target == null || image == null) return

        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        buffer.clear()
        val readStart = System.nanoTime()
        GLES30.glReadPixels(
            0, 0, usedW, usedH,
            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer,
        )
        readNanos += System.nanoTime() - readStart
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        ints.clear()
        val pixels = pixels
        // GL rows start at the bottom; Bitmap rows start at the top. The shader
        // writes straight alpha, so premultiply while copying for Skia. One
        // getInt per pixel instead of four get() calls is most of the win here.
        for (row in 0 until usedH) {
            var dst = (usedH - 1 - row) * usedW
            repeat(usedW) {
                val value = ints.get()
                val r = value and 0xff
                val g = (value ushr 8) and 0xff
                val b = (value ushr 16) and 0xff
                val a = (value ushr 24) and 0xff
                val pr = (r * a + 127) / 255
                val pg = (g * a + 127) / 255
                val pb = (b * a + 127) / 255
                pixels[dst++] = (a shl 24) or (pr shl 16) or (pg shl 8) or pb
            }
        }
        val copyStart = System.nanoTime()
        target.setPixels(pixels, 0, usedW, 0, 0, usedW, usedH)
        copyNanos += System.nanoTime() - copyStart
        val regions = HashMap<Long, SoapBubbleRegion>(items.size)
        for (item in items) {
            regions[item.controller.id] = SoapBubbleRegion(
                x = item.x,
                y = usedH - (item.y + item.canvasH),
                width = item.canvasW,
                height = item.canvasH,
            )
        }
        generation++
        val frame = SoapBubbleFrame(image, regions, generation)
        mainHandler.post { SoapBubbleGl.frame.value = frame }
    }

    /* ------------------------------------------------------------ */
    /* Helpers                                                       */
    /* ------------------------------------------------------------ */

    private fun environmentIndex(value: String): Int = when (value) {
        "window" -> 1
        "sky" -> 2
        "soft" -> 3
        else -> 0
    }

    private fun glintIndex(value: String): Int = when (value) {
        "dot" -> 1
        "dash" -> 2
        "pair" -> 3
        "none" -> 4
        else -> 0
    }

    private fun twinIndex(value: String): Int = when (value) {
        "arc" -> 0
        "crescent" -> 1
        "streak" -> 2
        "dot" -> 3
        else -> 4
    }

    private fun envRotation(
        out: FloatArray,
        tilt: FloatArray,
        time: Float,
        amount: Float,
    ) {
        val yaw = (tilt[0] * 0.35f + kotlin.math.sin(time * 0.13f) * 0.05f) * amount
        val pitch = (tilt[1] * 0.25f + kotlin.math.cos(time * 0.11f) * 0.03f) * amount
        val cy = kotlin.math.cos(yaw)
        val sy = kotlin.math.sin(yaw)
        val cp = kotlin.math.cos(pitch)
        val sp = kotlin.math.sin(pitch)
        out[0] = cy; out[1] = sy * sp; out[2] = -sy * cp
        out[3] = 0f; out[4] = cp; out[5] = sp
        out[6] = sy; out[7] = -cy * sp; out[8] = cy * cp
    }

    private class ProgramUniforms(private val program: Int) {
        private val cache = HashMap<String, Int>()

        private fun loc(name: String): Int =
            cache.getOrPut(name) { if (program == 0) -1 else GLES30.glGetUniformLocation(program, name) }

        fun f1(name: String, x: Float) {
            val l = loc(name); if (l >= 0) GLES30.glUniform1f(l, x)
        }

        fun f2(name: String, x: Float, y: Float) {
            val l = loc(name); if (l >= 0) GLES30.glUniform2f(l, x, y)
        }

        fun f4(name: String, x: Float, y: Float, z: Float, w: Float) {
            val l = loc(name); if (l >= 0) GLES30.glUniform4f(l, x, y, z, w)
        }

        fun f4(name: String, values: FloatArray) {
            if (values.size >= 4) f4(name, values[0], values[1], values[2], values[3])
        }

        fun i1(name: String, value: Int) {
            val l = loc(name); if (l >= 0) GLES30.glUniform1i(l, value)
        }

        fun mat3(name: String, values: FloatArray) {
            val l = loc(name)
            if (l >= 0) GLES30.glUniformMatrix3fv(l, 1, false, values, 0)
        }
    }
}
