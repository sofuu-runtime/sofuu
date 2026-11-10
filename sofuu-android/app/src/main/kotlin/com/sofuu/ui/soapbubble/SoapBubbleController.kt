package com.sofuu.ui.soapbubble

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** One flat track drawn under the film, in CSS pixels from the canvas top left. */
data class SoapBubbleLayer(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val radius: Float,
    val fill: Float,
    val track: String,
    val from: String,
    val to: String? = null,
)

internal class Spring(var value: Float = 0f, var velocity: Float = 0f)

/** The per-bubble motion state from `soap-bubble-renderer.ts`. */
internal class SoapBubbleMotionState {
    val modes = Array(4) { Spring() }
    val squash = Spring()
    val pressure = Spring()
    var pressed = false
    var still = false
    var time = 0f
    var flow = 0f
    var spark = ((System.nanoTime() ushr 8) and 0xffff).toFloat() / 65536f
    var pointerInside = false
    var pointerX = 0f
    var pointerY = 0f
    var pointerAtNanos = 0L
    val tiltTarget = floatArrayOf(0f, 0f)
    val tiltValue = floatArrayOf(0f, 0f)

    fun pressScale(wobble: Float): Float =
        1f + 0.36f * (if (still) 0f else pressure.value * min(1.6f, wobble / 0.4f))

    fun step(dt: Float, wobble: Float, speed: Float, paused: Boolean, isButton: Boolean) {
        if (still) {
            pressure.value = 0f
            pressure.velocity = 0f
            pressed = false
            return
        }
        val steps = max(1, kotlin.math.ceil(dt * 240f).toInt())
        val h = dt / steps
        for (i in 0 until steps) {
            val stiffness = if (pressed) 1250f else 420f
            val damping = if (pressed) 40f else 16f
            pressure.velocity += ((if (pressed) 1f else 0f) - pressure.value) * stiffness * h -
                pressure.velocity * damping * h
            pressure.value += pressure.velocity * h
        }
        if (kotlin.math.abs((if (pressed) 1f else 0f) - pressure.value) +
            kotlin.math.abs(pressure.velocity) < 0.001f
        ) {
            pressure.value = if (pressed) 1f else 0f
            pressure.velocity = 0f
        }
        val omega2 = (2f * Math.PI * 2.1).toFloat()
        val omega3 = omega2 * sqrt(30f / 12f)
        val damping = 0.09f
        for (index in 0..3) {
            val mode = modes[index]
            val omega = if (index < 2) omega2 else omega3
            mode.velocity += (-omega * omega * mode.value - 2f * damping * omega * mode.velocity) * dt
            mode.value += mode.velocity * dt
        }
        val squashOmega = (2f * Math.PI * 3.2).toFloat()
        val target = if (pressed && !isButton) 0.075f * wobble else 0f
        squash.velocity += (-squashOmega * squashOmega * (squash.value - target) -
            2f * 0.22f * squashOmega * squash.velocity) * dt
        squash.value += squash.velocity * dt
        for (axis in 0..1) {
            tiltValue[axis] += (tiltTarget[axis] - tiltValue[axis]) * min(1f, dt * 4f)
        }
        if (!paused) {
            time += dt * speed
            flow += dt * speed
        }
    }

    fun kick(strength: Float, angle: Float, wobble: Float) {
        if (still) return
        val amount = strength * wobble
        modes[0].velocity += cos(angle * 2f) * amount
        modes[1].velocity += sin(angle * 2f) * amount
        modes[2].velocity += cos(angle * 3f) * amount * 0.5f
        modes[3].velocity += sin(angle * 3f) * amount * 0.5f
    }

    fun pointer(x: Float, y: Float, inside: Boolean, wobble: Float) {
        val now = System.nanoTime()
        if (inside) {
            val dt = max(16_000_000L, now - pointerAtNanos) / 1e9f
            val vx = if (pointerInside) (x - pointerX) / dt else 0f
            val vy = if (pointerInside) (y - pointerY) / dt else 0f
            if (!pointerInside) {
                kick(0.35f, atan2(y, x), wobble)
            } else {
                val speed = min(6f, hypot(vx, vy))
                if (speed > 0.2f) kick(speed * 0.02f, atan2(vy, vx), wobble)
            }
            tiltTarget[0] = x
            tiltTarget[1] = y
        } else {
            if (pointerInside) kick(0.25f, atan2(pointerY, pointerX), wobble)
            tiltTarget[0] = 0f
            tiltTarget[1] = 0f
        }
        pointerX = x
        pointerY = y
        pointerInside = inside
        pointerAtNanos = now
    }

    fun rebound(wobble: Float) {
        if (still) return
        kick(1.4f, (Math.PI / 5).toFloat(), wobble)
        squash.velocity -= 0.7f * wobble
    }
}

/**
 * Parsed colours for one controller. Rebuilt only when its settings change,
 * instead of re-parsing the same CSS strings for every bubble every frame.
 */
internal class SoapBubblePaint(settings: SoapBubbleSettings) {
    val tint = if (settings.tint.trim().lowercase() == "auto") {
        floatArrayOf(1f, 1f, 1f, max(0f, settings.opacity))
    } else {
        val colour = cssColor(settings.tint)
        if (settings.opacity < 0f) colour
        else floatArrayOf(colour[0], colour[1], colour[2], settings.opacity)
    }
    val soap = cssColor(settings.soapColor)
    val shade = cssColor(settings.soapShadeColor)
    val highlight = cssColor(settings.soapHighlightColor)
    val paletteRed = cssColor(settings.soapPaletteRed)
    val paletteGold = cssColor(settings.soapPaletteGold)
    val paletteGreen = cssColor(settings.soapPaletteGreen)
    val paletteCyan = cssColor(settings.soapPaletteCyan)
    val paletteBlue = cssColor(settings.soapPaletteBlue)
    val paletteViolet = cssColor(settings.soapPaletteViolet)
    val customPalette = settings.soapPalette == "custom"
    val shadeColor = if (settings.bottomShadeColor.trim().lowercase() == "auto") {
        floatArrayOf(0f, 0f, 0f, -1f)
    } else cssColor(settings.bottomShadeColor)
}

/** CSS hex/rgb/named parsing, shared with the layer and backdrop uniforms. */
internal fun cssColor(value: String): FloatArray {
    val v = value.trim().lowercase()
    if (v.isEmpty() || v == "auto" || v == "transparent") return floatArrayOf(0f, 0f, 0f, 0f)
    if (v.startsWith("#")) {
        val hex = v.substring(1)
        return when (hex.length) {
            3 -> floatArrayOf(
                hex[0].digit() / 15f, hex[1].digit() / 15f, hex[2].digit() / 15f, 1f
            )
            6 -> floatArrayOf(
                hex.substring(0, 2).toInt(16) / 255f,
                hex.substring(2, 4).toInt(16) / 255f,
                hex.substring(4, 6).toInt(16) / 255f,
                1f,
            )
            8 -> floatArrayOf(
                hex.substring(0, 2).toInt(16) / 255f,
                hex.substring(2, 4).toInt(16) / 255f,
                hex.substring(4, 6).toInt(16) / 255f,
                hex.substring(6, 8).toInt(16) / 255f,
            )
            else -> floatArrayOf(1f, 0f, 1f, 1f)
        }
    }
    return try {
        val c = android.graphics.Color.parseColor(v)
        floatArrayOf(
            android.graphics.Color.red(c) / 255f,
            android.graphics.Color.green(c) / 255f,
            android.graphics.Color.blue(c) / 255f,
            android.graphics.Color.alpha(c) / 255f,
        )
    } catch (e: Exception) {
        floatArrayOf(1f, 0f, 1f, 1f)
    }
}

private fun Char.digit(): Int = when (this) {
    in '0'..'9' -> this - '0'
    in 'a'..'f' -> this - 'a' + 10
    else -> 0
}

/** Reused by the GL thread; no per-frame array allocation. */
internal class MotionSnapshot {
    var pressScale = 1f
    var time = 0f
    var flow = 0f
    var spark = 0f
    val modeBasis = FloatArray(4)
    val squash = floatArrayOf(1f, 1f)
    val tilt = FloatArray(2)
    val envRotation = FloatArray(9)
}

/**
 * A registered bubble. Compose owns its size and interaction; the GL thread
 * owns its motion and draws it into the shared atlas.
 */
internal class SoapBubbleController(val id: Long) {
    @Volatile var settings: SoapBubbleSettings = SoapBubbleSettings.lightPreset
        set(value) {
            val normalized = value.normalized()
            field = normalized
            paint = SoapBubblePaint(normalized)
        }

    @Volatile var paint: SoapBubblePaint = SoapBubblePaint(SoapBubbleSettings.lightPreset)
        private set

    @Volatile var widthPx: Float = 1f
    @Volatile var heightPx: Float = 1f
    /** Host bubble size, when it is smaller than the drawn canvas (a thumb). */
    @Volatile var hostWidthPx: Float = 0f
    @Volatile var hostHeightPx: Float = 0f
    /** Whole drawing canvas, including the bleed around the host. */
    @Volatile var canvasWidthPx: Float = 0f
    @Volatile var canvasHeightPx: Float = 0f
    /** Host centre inside the canvas; negative means the canvas centre. */
    @Volatile var centerXPx: Float = -1f
    @Volatile var centerYPx: Float = -1f
    @Volatile var density: Float = 2f
    @Volatile var fontPx: Float = 16f
    @Volatile var bleedPx: Float = 12f
    @Volatile var visible: Boolean = true
    @Volatile var reducedMotion: Boolean = false
    @Volatile var interactive: Boolean = true
    @Volatile var isButton: Boolean = true
    @Volatile var layer: SoapBubbleLayer? = null
    @Volatile var pressScaleHint: Float? = null

    val motion = SoapBubbleMotionState()

    @Synchronized
    fun press(on: Boolean) {
        if (motion.pressed == on) return
        motion.pressed = on
        if (!on) motion.kick(0.6f, (Math.PI / 2).toFloat(), settings.wobble)
    }

    @Synchronized
    fun pointer(x: Float, y: Float, inside: Boolean) {
        motion.pointer(x, y, inside, settings.wobble)
    }

    @Synchronized
    fun rebound() {
        motion.rebound(settings.wobble)
    }

    @Synchronized
    fun tick(dt: Float) {
        motion.still = reducedMotion
        motion.step(dt, settings.wobble, settings.speed, settings.paused, isButton)
    }

    @Synchronized
    fun snapshotInto(out: MotionSnapshot) {
        out.pressScale = pressScaleHint ?: motion.pressScale(settings.wobble)
        out.time = motion.time
        out.flow = motion.flow
        out.spark = motion.spark
        val amplitude2 = hypot(motion.modes[0].value, motion.modes[1].value)
        val amplitude3 = hypot(motion.modes[2].value, motion.modes[3].value)
        val limit2 = if (amplitude2 > 0f) min(0.12f, amplitude2) / amplitude2 else 0f
        val limit3 = if (amplitude3 > 0f) min(0.08f, amplitude3) / amplitude3 else 0f
        out.modeBasis[0] = motion.modes[0].value * limit2
        out.modeBasis[1] = motion.modes[1].value * limit2
        out.modeBasis[2] = motion.modes[2].value * limit3
        out.modeBasis[3] = motion.modes[3].value * limit3
        out.squash[0] = 1f + motion.squash.value * 0.7f
        out.squash[1] = 1f - motion.squash.value
        out.tilt[0] = motion.tiltValue[0]
        out.tilt[1] = motion.tiltValue[1]
    }
}

internal object SoapBubbleIds {
    private var next = 1L
    @Synchronized fun newId(): Long = next++
}
