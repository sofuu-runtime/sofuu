package com.sofuu.store

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GGUF models stored on the device.
 *
 * The infer engine opens a model by absolute path through
 * `MmapFile::open` (`crates/sofuu-core/src/infer/mmap.rs:21`), so a picked
 * `content://` URI is useless to it. Every import is copied into
 * `filesDir/models/` and referred to by that path.
 *
 * The app already pins `HOME` at `filesDir` (`SofuuBridge.setHome`), so
 * `~/model.gguf` would also resolve — but an absolute path is what the
 * engine logs and what `clean_model_path` round-trips cleanly, so we store
 * and pass the absolute form.
 *
 * Nothing here talks to the engine; that is [AppStore]'s job.
 */
class LocalModels(private val context: Context) {

    /** Where imports land. App-private, so no permission is involved. */
    val dir: File get() = File(context.filesDir, "models").apply { mkdirs() }

    fun list(): List<File> =
        dir.listFiles { f -> f.isFile && (f.name.endsWith(".gguf", ignoreCase = true) || f.name.endsWith(".safetensors", ignoreCase = true)) }
            ?.sortedBy { it.lastModified() }
            ?.reversed()
            ?: emptyList()

    /**
     * Copy a picked model into [dir]. Returns the destination on success,
     * null on any failure (unreadable URI, disk full, a duplicate name).
     *
     * The display name is untrusted input, so it is reduced to the last
     * path segment and rejected if it is empty, dot-prefixed, or an
     * extension other than .gguf — a picker can hand back anything.
     */
    fun import(uri: Uri): File? {
        val name = displayName(uri) ?: return null
        val safe = File(name).name
        if (safe.isEmpty() || safe == "." || safe == ".." || safe.startsWith(".")) return null
        // GGUF is the native format; single-file SafeTensors also routes
        // (rt::ai::looks_like_local_model accepts a .safetensors path). A
        // Multi-file HF directory cannot come through a single-file picker.
        if (!safe.endsWith(".gguf", ignoreCase = true) && !safe.endsWith(".safetensors", ignoreCase = true)) return null

        val dest = File(dir, safe)
        return try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                dest.outputStream().use { out -> input.copyTo(out, 64 * 1024) }
            } ?: return null
            dest
        } catch (e: Exception) {
            dest.delete()
            null
        }
    }

    fun delete(file: File): Boolean = file.parentFile == dir && file.delete()

    private fun displayName(uri: Uri): String? = try {
        context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c ->
                val i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (i >= 0 && c.moveToFirst()) c.getString(i) else null
            }
    } catch (e: Exception) {
        null
    } ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotEmpty() }

    /** Human label for the model picker: the GGUF header's general.name plus
     *  the quantization class (general.file_type) — "Llama 3.2 1B Instruct
     *  · Q8_0". Falls back to the file name when the header can't be read. */
    fun label(file: File): String = friendlyGgufLabel(file)
}

// ── GGUF display metadata ─────────────────────────────────────────────
//
// A model's file name is not its name: "…Distilled.i1-Q4_K_M" is a
// distribution-tag soup. The GGUF header carries `general.name` and
// `general.file_type` (the llama.cpp quant class), and both live in the
// first few key/value pairs — a small bounded read answers without
// touching tensor data. Any parse failure degrades to the file name.

/** general.file_type → quant class (llama.cpp LLAMA_FTYPE_*). */
private val GGUF_FTYPE_LABELS = mapOf(
    0L to "F32", 1L to "F16", 2L to "Q4_0", 3L to "Q4_1", 4L to "Q4_1",
    7L to "Q8_0", 8L to "Q5_0", 9L to "Q5_1",
    10L to "Q2_K", 11L to "Q3_K_S", 12L to "Q3_K_M", 13L to "Q3_K_L",
    14L to "Q4_K_S", 15L to "Q4_K_M", 16L to "Q5_K_S", 17L to "Q5_K_M",
    18L to "Q6_K", 19L to "IQ2_XXS", 20L to "IQ2_XS", 21L to "Q2_K_S",
    22L to "IQ3_XS", 23L to "IQ3_XXS", 24L to "IQ1_S", 25L to "IQ4_NL",
    26L to "IQ3_S", 27L to "IQ3_M", 28L to "IQ2_S", 29L to "IQ2_M",
    30L to "IQ4_XS", 31L to "IQ1_M", 32L to "BF16",
    36L to "TQ1_0", 37L to "TQ2_0", 38L to "MXFP4",
)

/** Trailing quant token in a distribution file name:
 *  "…-Instruct-Q8_0" → ("…-Instruct", "Q8_0"). Used when the header is
 *  unreadable, and to build UI-thread-safe labels without disk access. */
private val FILE_QUANT_RE = Regex(
    """[._-](IQ[1-4](?:_[A-Z0-9]+)?|Q[1-8](?:_[A-Z0-9]+)*|TQ[12]_0|BF16|F16|F32|FP16|FP32|MXFP4)\z""",
    RegexOption.IGNORE_CASE,
)

fun splitFileNameQuant(fileName: String): Pair<String, String?> {
    val stem = fileName.let { if (it.length > 5 && it.lowercase().endsWith(".gguf")) it.dropLast(5) else it }
    val m = FILE_QUANT_RE.find(stem) ?: return stem to null
    return stem.substring(0, m.range.first) to m.groupValues[1].uppercase()
}

/** Display label from the name/path only — no disk access, safe on the UI
 *  thread. Path prefixes pass through the same split; a quant-less name
 *  ("gpt-5") is returned unchanged. */
fun fallbackModelLabel(pathOrName: String): String {
    val name = pathOrName.substringAfterLast('/')
    val (base, quant) = splitFileNameQuant(name)
    return if (quant != null) "$base · $quant" else base
}

private class GgufHeaderException : Exception()

/** (general.name, general.file_type) straight off the GGUF header, or null. */
private fun readGgufMeta(file: File): Pair<String?, Long?>? = try {
    RandomAccessFile(file, "r").use { raf ->
        // general.* keys lead the header; 4 MB covers every converter we
        // see models from while never reaching tensor bytes.
        val size = minOf(raf.length(), 4L * 1024 * 1024).toInt()
        if (size < 32) null
        else {
            val bb = ByteBuffer.wrap(ByteArray(size).also { raf.readFully(it) }).order(ByteOrder.LITTLE_ENDIAN)
            if (bb.get() != 'G'.code.toByte() || bb.get() != 'G'.code.toByte() ||
                bb.get() != 'U'.code.toByte() || bb.get() != 'F'.code.toByte()
            ) throw GgufHeaderException()
            bb.int // version
            bb.long // tensors
            val kvCount = bb.long
            var name: String? = null
            var ftype: Long? = null
            var i = 0L
            while (i < kvCount && (name == null || ftype == null) && bb.hasRemaining()) {
                i++
                val key = ggufString(bb) ?: break
                // llama.cpp writers emit every general.* key before the
                // tokenizer arrays; the first non-general key means nothing
                // ahead can change the label — stop instead of walking
                // 150k token strings to the end of the window.
                if (!key.startsWith("general.")) break
                when (val vt = (bb.int.toLong() and 0xFFFFFFFFL).toInt()) {
                    8 -> {
                        val s = ggufString(bb) ?: throw GgufHeaderException()
                        if (key == "general.name") name = s
                    }
                    4 -> {
                        val v = bb.int.toLong() and 0xFFFFFFFFL
                        if (key == "general.file_type") ftype = v
                    }
                    else -> skipGgufValue(bb, vt)
                }
            }
            name to ftype
        }
    }
} catch (e: Exception) {
    null
}

private fun ggufString(bb: ByteBuffer): String? {
    val len = bb.long
    if (len < 0 || len > 16_000_000) throw GgufHeaderException()
    val bytes = ByteArray(len.toInt())
    bb.get(bytes)
    return String(bytes, Charsets.UTF_8)
}

private fun skipGgufValue(bb: ByteBuffer, type: Int, depth: Int = 0) {
    if (depth > 8) throw GgufHeaderException()
    when (type) {
        0, 1, 7 -> bb.position(bb.position() + 1)
        2, 3 -> bb.position(bb.position() + 2)
        4, 5, 6 -> bb.position(bb.position() + 4)
        10, 11, 12 -> bb.position(bb.position() + 8)
        8 -> ggufString(bb)
        9 -> {
            val elemType = (bb.int.toLong() and 0xFFFFFFFFL).toInt()
            val count = bb.long
            if (count < 0 || count > 10_000_000) throw GgufHeaderException()
            repeat(count.toInt()) { skipGgufValue(bb, elemType, depth + 1) }
        }
        else -> throw GgufHeaderException()
    }
}

/** The model-picker label: header name plus quant class, de-duplicated
 *  against the name; file-name fallback for unreadable headers. Reads a
 *  bounded slice of the file — keep callers off the UI thread. */
fun friendlyGgufLabel(file: File): String = try {
    val stem = file.name.let { if (it.length > 5 && it.lowercase().endsWith(".gguf")) it.dropLast(5) else it }
    val meta = readGgufMeta(file)
    val fromName = splitFileNameQuant(stem)
    val name = meta?.first?.takeIf { it.isNotBlank() } ?: fromName.first
    var quant = meta?.second?.let { GGUF_FTYPE_LABELS[it] } ?: fromName.second
    if (quant != null && name.contains(quant, ignoreCase = true)) quant = null
    if (quant != null) "$name · $quant" else name
} catch (e: Exception) {
    file.name
}