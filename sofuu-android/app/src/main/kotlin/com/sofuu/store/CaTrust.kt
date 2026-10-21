package com.sofuu.store

import android.content.Context
import java.io.File

/**
 * The trust store handed to the engine's static libcurl.
 *
 * The Android curl is built with `CURL_CA_BUNDLE=none` /
 * `CURL_CA_PATH=none`, and the OpenSSL it links has an OPENSSLDIR that
 * points into the host build tree. curl 8.6 only reads CA locations as
 * compile-time defaults, so on the phone there is no trust anchor at all
 * and every `https://` request fails with "SSL peer certificate or SSH
 * remote key was not OK" (CURLE_PEER_FAILED_VERIFICATION).
 *
 * Android stores its anchors as individual PEM files with MD5-hashed
 * names; OpenSSL 3 looks hash directories up by SHA-1, so pointing curl
 * at that directory is unreliable. Concatenating the files into one PEM
 * bundle sidesteps the hash question entirely — curl then loads it via
 * CURLOPT_CAINFO from `SOFUU_CA_BUNDLE` (see sofuu-ffi/src/curl.rs).
 */
internal object CaTrust {

    /** System store first; the APEX copy is the same set on Android 14+
     *  (the /system path is a symlink to it), and the per-user directory
     *  holds CAs the owner installed. Duplicates are dropped by filename. */
    private val SOURCES = listOf(
        "/system/etc/security/cacerts",
        "/apex/com.android.conscrypt/cacerts",
        "/data/misc/user/0/cacerts-added",
    )

    /**
     * Path for `SOFUU_CA_BUNDLE`, or null when no anchor was readable.
     * Cheap after the first run: the stamp records the source listing, so
     * a launch with unchanged certs only lists two directories.
     */
    fun ensureBundle(context: Context): String? {
        val dir = File(context.filesDir, "ca")
        val out = File(dir, "cacert.pem")
        val stamp = File(dir, "cacert.stamp")
        val sources = SOURCES.map { File(it) }.filter { it.isDirectory }
        if (sources.isEmpty()) return null

        val key = sources.joinToString("|") { d ->
            // runCatching: a directory we cannot list is not an error, just
            // an empty contribution.
            val names = runCatching { d.list()?.sorted()?.joinToString(",") }.getOrNull().orEmpty()
            "${d.path}:${d.lastModified()}:$names"
        }
        val cached = out.isFile && out.length() > 0 &&
            runCatching { stamp.readText() }.getOrNull() == key
        if (cached) return out.absolutePath

        return runCatching {
            dir.mkdirs()
            val byName = LinkedHashMap<String, File>()
            for (d in sources) {
                for (f in d.listFiles().orEmpty()) {
                    if (f.isFile) byName.putIfAbsent(f.name, f)
                }
            }
            out.outputStream().buffered().use { os ->
                for (f in byName.values) {
                    val bytes = runCatching { f.readBytes() }.getOrNull() ?: continue
                    if (bytes.isEmpty()) continue
                    if (!String(bytes, Charsets.ISO_8859_1).contains("BEGIN CERTIFICATE")) continue
                    os.write(bytes)
                    if (bytes.last() != '\n'.code.toByte()) os.write('\n'.code)
                }
            }
            if (out.length() == 0L) {
                out.delete()
                null
            } else {
                stamp.writeText(key)
                out.absolutePath
            }
        }.getOrNull()
    }
}
