package com.sofuu.engine

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import org.json.JSONArray
import org.json.JSONObject

/**
 * Native capabilities the agent can call, implemented by the host.
 *
 * The runtime's QuickJS sandbox cannot reach the Android SDK, so anything
 * device-specific has to be answered from here via
 * [SofuuBridge.registerHostTool]. That is the whole point of the host-tool
 * ABI: it is what makes a phone able to do things the CLI cannot, rather
 * only fewer things.
 *
 * Every tool here is registered as a normal agent tool, so it shows up in
 * `/tools`, honours the permission profile (`plan` mode refuses writes,
 * `prompt` mode asks first), and lands in the normal tool timeline.
 *
 * Scope is deliberately permission-light: the tools that need a runtime
 * permission a user might refuse are not registered rather than registered
 * and silently failing.
 */
class DeviceTools(
    private val context: Context,
    /** Mirrors SOFUU_NO_SHELL: the engine gates its own bash on that env
     *  var, so a host shell tool that ignored it would be a hole in the
     *  same policy. Read per invocation, so the Settings toggle applies
     *  without a restart. */
    private val isShellAllowed: () -> Boolean = { true },
) {

    /** The set to hand to [SofuuBridge.registerHostTool]. */
    fun all(): List<Registration> = listOf(
        Registration(
            name = "device_info",
            description = "Report this phone's model, OS version, battery level, free storage " +
                "and free memory. Use when you need to know what you are running on.",
            parameters = """{"type":"object","properties":{},"required":[]}""",
        ) { deviceInfo() },
        Registration(
            name = "open_url",
            description = "Open a web page or app link in the phone's browser. " +
                "Only use it when the user asks you to show them something.",
            parameters = """{"type":"object","properties":{"url":{"type":"string","description":"Absolute http(s) URL to open"}},"required":["url"]}""",
        ) { args -> openUrl(args) },
        Registration(
            name = "share_text",
            description = "Open the Android share sheet so the user can send text " +
                "(a link, a note, a summary) to another app.",
            parameters = """{"type":"object","properties":{"text":{"type":"string","description":"Text to share"},"subject":{"type":"string","description":"Optional title for the share target"}},"required":["text"]}""",
        ) { args -> shareText(args) },
        Registration(
            name = "vibrate",
            description = "Buzz the phone briefly to get the user's attention. " +
                "Use sparingly — a vibration for every finished turn trains people to ignore it.",
            parameters = """{"type":"object","properties":{"ms":{"type":"integer","description":"Duration in milliseconds (default 200, max 1000)"}},"required":[]}""",
        ) { args -> vibrate(args) },
        Registration(
            name = "run_command",
            description = "Execute a shell command on Android (/system/bin/sh -c) and return output and exit code. Allows running file inspection, process tools, and system commands.",
            parameters = """{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute on Android"},"timeout_ms":{"type":"integer","description":"Timeout in milliseconds (default 30000)"}},"required":["command"]}""",
        ) { args -> runPhoneShell(args) },
    )

    /** One tool: its schema plus the handler that answers a call. */
    class Registration(
        val name: String,
        val description: String,
        val parameters: String,
        val handler: (JSONObject) -> String,
    )

    /** Attach every tool to a runtime. Engine thread only. */
    fun registerAll(bridge: SofuuBridge): List<String> = all().mapNotNull { r ->
        val ok = bridge.registerHostTool(r.name, r.description, r.parameters,
            object : SofuuBridge.HostTool {
                override fun invoke(argsJson: String, callId: Long): String? = try {
                    r.handler(JSONObject(argsJson.ifBlank { "{}" }))
                } catch (e: Exception) {
                    // A throwing tool would surface to the model as a hang
                    // (the JNI trampoline returns null on exception), so
                    // report it as a failed call instead.
                    JSONObject().put("error", e.message ?: e.toString()).toString()
                }
            })
        if (ok) r.name else null
    }

    // ── Handlers ─────────────────────────────────────────────────

    private fun deviceInfo(): String {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
        val data = JSONObject()
            .put("model", "${Build.MANUFACTURER} ${Build.MODEL}".trim())
            .put("android", Build.VERSION.RELEASE)
            .put("sdk_int", Build.VERSION.SDK_INT)
            .put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: "?")
        bm?.let {
            // Integer.MIN_VALUE when the property is unsupported — reported
            // straight to the model as a level, so drop it instead.
            val level = it.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            if (level != Int.MIN_VALUE && level in 0..100) data.put("battery_percent", level)
        }
        runCatching {
            val stat = Environment.getDataDirectory()
            data.put("storage_free_gb", ((stat.usableSpace) / 1_073_741_824.0).toDouble())
            data.put("storage_total_gb", ((stat.totalSpace) / 1_073_741_824.0).toDouble())
        }
        runCatching {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? android.app.ActivityManager
            val info = android.app.ActivityManager.MemoryInfo()
            am?.getMemoryInfo(info)
            data.put("memory_free_gb", (info.availMem / 1_073_741_824.0).toDouble())
            data.put("memory_total_gb", (info.totalMem / 1_073_741_824.0).toDouble())
        }
        return data.toString()
    }

    private fun openUrl(args: JSONObject): String {
        val url = args.optString("url")
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            return err("open_url needs an http(s) URL")
        }
        return try {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            })
            ok("opened $url")
        } catch (e: ActivityNotFoundException) {
            err("no app on this device can open that URL")
        }
    }

    private fun shareText(args: JSONObject): String {
        val text = args.optString("text")
        if (text.isEmpty()) return err("share_text needs text")
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, text)
            args.optString("subject").takeIf { it.isNotEmpty() }?.let { putExtra(Intent.EXTRA_SUBJECT, it) }
        }
        return try {
            context.startActivity(Intent.createChooser(intent, "Share").apply {
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            })
            ok("share sheet opened")
        } catch (e: ActivityNotFoundException) {
            err("this device has no share targets")
        }
    }

    private fun vibrate(args: JSONObject): String {
        val ms = args.optInt("ms", 200).coerceIn(1, 1000)
        // API 31 split Vibrator out behind a manager; both paths are one
        // class of call otherwise.
        @Suppress("DEPRECATION")
        val vib = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
        } else {
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        } ?: return err("this device has no vibrator")
        if (!vib.hasVibrator()) return err("this device has no vibrator")
        // VibrationEffect is API 26; minSdk is 24.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vib.vibrate(VibrationEffect.createOneShot(ms.toLong(), VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vib.vibrate(ms.toLong())
        }
        return ok("vibrated ${ms}ms")
    }

    private fun runPhoneShell(args: JSONObject): String {
        val command = args.optString("command").trim()
        if (command.isEmpty()) return err("command is required")
        if (!isShellAllowed()) return err("run_command: shell is disabled (SOFUU_NO_SHELL is set)")
        val timeoutMs = args.optLong("timeout_ms", 30_000L).coerceIn(1000L, 120_000L)
        return try {
            val process = ProcessBuilder("/system/bin/sh", "-c", command)
                .redirectErrorStream(true)
                .start()

            val output = StringBuilder()
            // Read on a worker thread: readLine() blocks until a line
            // arrives, so a child that prints a partial line and goes
            // silent used to hold the ENGINE thread forever. The timeout
            // is enforced on waitFor instead.
            val reader = Thread({
                try {
                    java.io.BufferedReader(java.io.InputStreamReader(process.inputStream)).use { r ->
                        var line: String?
                        while (r.readLine().also { line = it } != null) {
                            synchronized(output) {
                                if (output.length < 32_000) output.append(line).append('\n')
                            }
                        }
                    }
                } catch (_: Throwable) {
                    // Destroyed process: the stream closes under us.
                }
            }, "sofuu-run-command")
            reader.isDaemon = true
            reader.start()

            // No Process.waitFor(long, TimeUnit) or destroyForcibly here:
            // both are API 26 and minSdk is 24. A waiter thread plus a
            // deadline loop is the same contract on every API level.
            val exit = java.util.concurrent.atomic.AtomicInteger(Int.MIN_VALUE)
            val waiter = Thread({
                try { exit.set(process.waitFor()) } catch (_: Throwable) {}
            }, "sofuu-run-command-wait")
            waiter.isDaemon = true
            waiter.start()

            val deadline = System.currentTimeMillis() + timeoutMs
            while (waiter.isAlive && System.currentTimeMillis() < deadline) {
                Thread.sleep(50)
            }
            if (waiter.isAlive) {
                process.destroy()
                waiter.join(500)
                reader.join(500)
                synchronized(output) {
                    if (output.length < 32_000) output.append("\n[Command timed out after ${timeoutMs}ms]")
                }
                return JSONObject()
                    .put("ok", false)
                    .put("exit_code", -1)
                    .put("output", synchronized(output) { output.toString().trim() })
                    .toString()
            }
            // The child is gone; the reader drains the remaining bytes.
            reader.join(1_000)
            val code = exit.get()
            val exitCode = if (code == Int.MIN_VALUE) -1 else code
            JSONObject()
                .put("ok", exitCode == 0)
                .put("exit_code", exitCode)
                .put("output", synchronized(output) { output.toString().trim() })
                .toString()
        } catch (e: Exception) {
            err("failed to execute command: ${e.message}")
        }
    }

    private fun ok(note: String): String = JSONObject().put("ok", true).put("note", note).toString()

    private fun err(message: String): String =
        JSONObject().put("error", message).toString()
}