package com.sofuu.ui.soapbubble

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The two CPU-built lookup tables Bubble UI uploads alongside its shader.
 *
 * Ported line for line from `soap-bubble-gl.ts`: the interference reflectance
 * table (thickness x incidence) and the tiling 3D gradient-noise volume used
 * by gravity drainage and Marangoni swirls. They are generated once, on a
 * background thread, and shared by every bubble.
 */
internal object SoapBubbleTables {
    const val FILM_MAX = 3000f
    const val FILM_COLUMNS = 768
    const val FILM_ROWS = 48
    const val SPECTRAL_SAMPLES = 16
    const val NOISE_PERIOD = 8
    const val NOISE_SIZE = 64

    private const val WATER = 1.333

    private fun lobe(lambda: Double, center: Double, below: Double, above: Double): Double {
        val t = (lambda - center) * (if (lambda < center) below else above)
        return exp(-0.5 * t * t)
    }

    private fun cie(lambda: Double): DoubleArray = doubleArrayOf(
        0.362 * lobe(lambda, 442.0, 0.0624, 0.0374) +
            1.056 * lobe(lambda, 599.8, 0.0264, 0.0323) -
            0.065 * lobe(lambda, 501.1, 0.049, 0.0382),
        0.821 * lobe(lambda, 568.8, 0.0213, 0.0247) +
            0.286 * lobe(lambda, 530.9, 0.0613, 0.0322),
        1.217 * lobe(lambda, 437.0, 0.0845, 0.0278) +
            0.681 * lobe(lambda, 459.0, 0.0385, 0.0725),
    )

    private class Spectrum(
        val lambda: DoubleArray,
        val bandwidth: DoubleArray,
        val weights: DoubleArray,
    )

    private fun spectralWeights(): Spectrum {
        val count = SPECTRAL_SAMPLES
        val start = 385.0
        val end = 715.0
        val step = (end - start) / count
        val lambda = DoubleArray(count)
        val bandwidth = DoubleArray(count)
        val weights = DoubleArray(count * 3)
        val sum = doubleArrayOf(0.0, 0.0, 0.0)
        for (i in 0 until count) {
            val l = start + (i + 0.5) * step
            val (x, y, z) = cie(l)
            val rgb = doubleArrayOf(
                3.2406 * x - 1.5372 * y - 0.4986 * z,
                -0.9689 * x + 1.8758 * y + 0.0415 * z,
                0.0557 * x - 0.204 * y + 1.057 * z,
            )
            lambda[i] = l
            bandwidth[i] = step / l
            for (c in 0..2) {
                weights[i * 3 + c] = rgb[c]
                sum[c] += rgb[c]
            }
        }
        for (i in 0 until count) for (c in 0..2) weights[i * 3 + c] /= sum[c]
        return Spectrum(lambda, bandwidth, weights)
    }

    private fun airy(r2: Double, c: Double): Double =
        (2 * r2 * (1 - c)) / (1 + r2 * r2 - 2 * r2 * c)

    /** RGBA float table, 768 x 48, row-major from the bottom as GL expects. */
    fun filmTable(): FloatArray {
        val spectrum = spectralWeights()
        val lambda = spectrum.lambda
        val bandwidth = spectrum.bandwidth
        val weights = spectrum.weights
        val columns = FILM_COLUMNS
        val rows = FILM_ROWS
        val count = lambda.size
        val table = FloatArray(columns * rows * 4)
        val cosPhase = DoubleArray(count)
        val sinPhase = DoubleArray(count)
        val stepCos = DoubleArray(count)
        val stepSin = DoubleArray(count)
        val spreadCos = DoubleArray(count)
        val spreadSin = DoubleArray(count)
        val spreadStepCos = DoubleArray(count)
        val spreadStepSin = DoubleArray(count)
        val spreadStep = DoubleArray(count)

        for (row in 0 until rows) {
            val cosI = max(1e-4, row.toDouble() / (rows - 1))
            val sin2 = 1.0 - cosI * cosI
            val cosT = sqrt(max(0.0, 1.0 - sin2 / (WATER * WATER)))
            val rs = (cosI - WATER * cosT) / (cosI + WATER * cosT)
            val rp = (WATER * cosI - cosT) / (WATER * cosI + cosT)
            val rs2 = rs * rs
            val rp2 = rp * rp
            val pathStep = (4 * PI * WATER * FILM_MAX * cosT) / (columns - 1)
            for (i in 0 until count) {
                val phase = pathStep / lambda[i]
                val spread = 0.5 * phase * bandwidth[i]
                cosPhase[i] = 1.0
                sinPhase[i] = 0.0
                stepCos[i] = cos(phase)
                stepSin[i] = sin(phase)
                spreadCos[i] = 1.0
                spreadSin[i] = 0.0
                spreadStepCos[i] = cos(spread)
                spreadStepSin[i] = sin(spread)
                spreadStep[i] = spread
            }
            var index = row * columns * 4
            for (column in 0 until columns) {
                var r = 0.0
                var g = 0.0
                var b = 0.0
                for (i in 0 until count) {
                    val spread = column * spreadStep[i]
                    val sinc = if (spread < 1e-3) 1.0 else spreadSin[i] / spread
                    val c = cosPhase[i] * sinc
                    val reflectance = 0.5 * (airy(rs2, c) + airy(rp2, c))
                    r += reflectance * weights[i * 3]
                    g += reflectance * weights[i * 3 + 1]
                    b += reflectance * weights[i * 3 + 2]
                    val phaseCos = cosPhase[i]
                    cosPhase[i] = phaseCos * stepCos[i] - sinPhase[i] * stepSin[i]
                    sinPhase[i] = sinPhase[i] * stepCos[i] + phaseCos * stepSin[i]
                    val bandCos = spreadCos[i]
                    spreadCos[i] = bandCos * spreadStepCos[i] - spreadSin[i] * spreadStepSin[i]
                    spreadSin[i] = spreadSin[i] * spreadStepCos[i] + bandCos * spreadStepSin[i]
                }
                table[index++] = max(0.0, r).toFloat()
                table[index++] = max(0.0, g).toFloat()
                table[index++] = max(0.0, b).toFloat()
                table[index++] = 1f
            }
        }
        return table
    }

    /** Tiling 64^3 gradient noise, one byte per texel. */
    fun noiseVolume(): ByteArray {
        val period = NOISE_PERIOD
        val size = NOISE_SIZE
        var state = 0x2545f491L
        fun random(): Double {
            // JavaScript's bitwise operators coerce to uint32 after every step.
            state = (state xor (state shl 13)) and 0xffffffffL
            state = (state xor (state ushr 17)) and 0xffffffffL
            state = (state xor (state shl 5)) and 0xffffffffL
            return state.toDouble() / 4294967296.0
        }

        val gradients = DoubleArray(period * period * period * 3)
        var gi = 0
        while (gi < gradients.size) {
            val z = random() * 2 - 1
            val angle = random() * PI * 2
            val r = sqrt(1 - z * z)
            gradients[gi] = r * cos(angle)
            gradients[gi + 1] = r * sin(angle)
            gradients[gi + 2] = z
            gi += 3
        }

        fun fade(t: Double): Double = t * t * t * (t * (t * 6 - 15) + 10)

        val step = period.toDouble() / size
        val cells = IntArray(size * 2)
        val offsets = DoubleArray(size * 2)
        val weights = DoubleArray(size * 2)
        for (n in 0 until size) {
            val p = n * step
            val cell = kotlin.math.floor(p).toInt()
            val f = p - cell
            val u = fade(f)
            for (o in 0..1) {
                cells[n * 2 + o] = ((cell + o) % period + period) % period
                offsets[n * 2 + o] = f - o
                weights[n * 2 + o] = if (o == 1) u else 1 - u
            }
        }

        val volume = ByteArray(size * size * size)
        var index = 0
        for (k in 0 until size) for (j in 0 until size) for (i in 0 until size) {
            var value = 0.0
            for (corner in 0 until 8) {
                val x = i * 2 + (corner and 1)
                val y = j * 2 + ((corner ushr 1) and 1)
                val z = k * 2 + ((corner ushr 2) and 1)
                val g = 3 * (cells[x] + period * (cells[y] + period * cells[z]))
                val dot = gradients[g] * offsets[x] +
                    gradients[g + 1] * offsets[y] +
                    gradients[g + 2] * offsets[z]
                value += dot * (weights[x] * weights[y] * weights[z])
            }
            val encoded = min(1.0, max(0.0, 0.5 + 0.5 * value * 1.6))
            volume[index++] = (encoded * 255).roundToInt().toByte()
        }
        return volume
    }
}
