package com.sofuu.store

import android.content.Context
import android.net.Uri
import java.io.File

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
        dir.listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
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
        if (!safe.endsWith(".gguf", ignoreCase = true)) return null

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

    /** Human label for the model picker: the file name minus the suffix. */
    fun label(file: File): String = file.name.removeSuffix(".gguf").removeSuffix(".GGUF")
}