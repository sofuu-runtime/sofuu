package com.sofuu.ui

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat

/** System notifications for approvals and finished turns (the desktop's
 *  notification plugin, on Android terms). */
object Notifier {
    private const val CHANNEL_ID = "sofuu.turns"

    fun ensureChannel(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val mgr = context.getSystemService(NotificationManager::class.java) ?: return
        if (mgr.getNotificationChannel(CHANNEL_ID) != null) return
        val ch = NotificationChannel(
            CHANNEL_ID,
            "Turns & approvals",
            NotificationManager.IMPORTANCE_DEFAULT,
        ).apply { description = "Approval requests and finished turns" }
        mgr.createNotificationChannel(ch)
    }

    fun post(context: Context, title: String, body: String) {
        ensureChannel(context)
        val n = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_chat)
            .setContentTitle(title)
            .setContentText(body)
            .setAutoCancel(true)
            .build()
        runCatching { NotificationManagerCompat.from(context).notify((System.currentTimeMillis() % Int.MAX_VALUE).toInt(), n) }
    }
}
