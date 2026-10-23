package com.sofuu

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.sofuu.store.AppStore
import com.sofuu.ui.Notifier
import com.sofuu.ui.SofuuApp
import com.sofuu.ui.theme.LocalSofuuColors
import com.sofuu.ui.theme.SofuuTheme
import java.io.File

/**
 * The single activity. Edge-to-edge; the Compose tree insets itself from
 * the system bars. It hosts the Sofuu shell (drawer + chat + composer +
 * settings) over the in-process Sofuu engine (started by the ViewModel).
 */
class MainActivity : ComponentActivity() {

    private val store: AppStore by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Persist Java crashes to app-private storage: a crash that happens
        // with the cable unplugged still leaves a full stack on-device
        // (readable via `adb shell run-as com.sofuu cat files/app-crash.log`).
        val prevHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                java.io.File(filesDir, "app-crash.log").appendText(
                    "\n===== " + t.name + " @ " + java.lang.System.currentTimeMillis() + " =====\n" +
                        android.util.Log.getStackTraceString(e) + "\n"
                )
            } catch (_: Throwable) {
            }
            prevHandler?.uncaughtException(t, e)
        }

        Notifier.ensureChannel(this)
        store.notifier = { title, body -> Notifier.post(this, title, body) }

        // Notification permission (approval/done alerts) — API 33+.
        if (Build.VERSION.SDK_INT >= 33 && !hasNotificationPermission()) {
            requestNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        // Hardware back: close the top-most overlay first, then leave.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (store.closeTopOverlay()) return
                if (isEnabled) {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        setContent {
            val dark = store.resolvedDark(isSystemInDarkTheme())
            SofuuTheme(dark = dark) {
                Surface(color = LocalSofuuColors.current.bgApp, modifier = Modifier.fillMaxSize()) {
                    SofuuApp(store, onPickFiles = ::launchFilePicker, onRequestMic = ::ensureMic)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        store.foreground = true
    }

    override fun onPause() {
        super.onPause()
        store.foreground = false
        store.saveCurrentSessionMessages()
        // Never leave the mic hot in the background.
        store.cancelDictation()
    }

    private fun hasNotificationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

    private val requestNotifications = registerForActivityResult(RequestPermission()) { }

    /**
     * RECORD_AUDIO for dictation. Asked lazily on the first mic tap rather
     * than at launch — an up-front mic prompt on a chat app is the kind of
     * thing people deny reflexively and then never grant.
     */
    private val requestMic = registerForActivityResult(RequestPermission()) { granted ->
        if (granted) store.toggleDictation() else store.systemRow("Microphone permission denied — dictation is off.")
    }

    private fun hasMicPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun ensureMic(): Boolean {
        if (hasMicPermission()) return true
        requestMic.launch(Manifest.permission.RECORD_AUDIO)
        return false
    }

    private fun uriToDataUrl(uri: Uri, maxDim: Int): String? = try {
        val bmp = android.graphics.ImageDecoder.decodeBitmap(
            android.graphics.ImageDecoder.createSource(contentResolver, uri),
        ) { decoder, info, _ ->
            val scale = maxDim.toFloat() / maxOf(info.size.width, info.size.height).coerceAtLeast(1)
            if (scale < 1f) {
                decoder.setTargetSize((info.size.width * scale).toInt(), (info.size.height * scale).toInt())
            }
        }
        val stream = java.io.ByteArrayOutputStream()
        bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 82, stream)
        "data:image/jpeg;base64," + android.util.Base64.encodeToString(stream.toByteArray(), android.util.Base64.NO_WRAP)
    } catch (e: Exception) {
        null
    }

    // ── File picker (SAF) → @mentions ────────────────────────────
    // The engine reads POSIX paths inside its workspace, so a picked
    // content:// URI is copied into the workspace and mentioned by name.

    private var pendingFiles: (List<String>, List<String>) -> Unit = { _, _ -> }

    private val filePickerLauncher =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris: List<Uri> ->
            if (uris.isEmpty()) return@registerForActivityResult
            val workspace = File(store.engine.projectDir).apply { mkdirs() }
            val images = ArrayList<String>()
            val mentions = ArrayList<String>()
            for (uri in uris) {
                val raw = displayName(uri) ?: continue
                // A display name is untrusted input: keep the last path
                // segment only, never a traversal or a dotfile.
                val name = File(raw).name.takeIf { it.isNotEmpty() && it != "." && it != ".." && !it.startsWith('.') } ?: continue
                val mime = contentResolver.getType(uri).orEmpty()
                if (mime.startsWith("image/")) {
                    uriToDataUrl(uri, 1400)?.let { images.add(it) }
                } else {
                    runCatching {
                        val dest = File(workspace, name)
                        contentResolver.openInputStream(uri)?.use { input ->
                            dest.outputStream().use { out -> input.copyTo(out) }
                        }
                        mentions.add(name)
                    }
                }
            }
            pendingFiles(images, mentions)
        }

    private fun launchFilePicker() {
        pendingFiles = { images, mentions -> store.attachPicked(images, mentions) }
        // "image/*" + text-ish types the engine tools can read.
        filePickerLauncher.launch(arrayOf("image/*", "text/*", "application/json", "application/javascript"))
    }

    private fun displayName(uri: Uri): String? = runCatching {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (i >= 0 && c.moveToFirst()) c.getString(i) else null
        }
    }.getOrNull() ?: uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotEmpty() }
}
