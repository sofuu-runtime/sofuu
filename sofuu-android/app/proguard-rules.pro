# sofuu-android ProGuard/R8 rules.
#
# Referenced by buildTypes.release in app/build.gradle.kts. The file is
# required even though isMinifyEnabled is currently false — AGP resolves
# every declared proguard file at configure time, so a missing one fails
# the release build before a single line is compiled.

# ── JNI ────────────────────────────────────────────────────────────
# The JNI bridge resolves these by exact name at dlopen time
# (Java_com_sofuu_runtime_SofuuBridge_* / _engine_SofuuBridge_*), so R8
# must not rename or strip them.
-keepclasseswithmembernames class com.sofuu.runtime.SofuuBridge {
    native <methods>;
}
-keepclasseswithmembernames class com.sofuu.engine.SofuuBridge {
    native <methods>;
}

# A StreamListener implementation is handed to C as a global JNI ref and
# called back through its onStreamEvent methodID.
-keepclassmembers class * implements com.sofuu.engine.SofuuBridge$StreamListener {
    void onStreamEvent(java.lang.String);
}

# ── Compose ────────────────────────────────────────────────────────
# Composables are constructed reflectively by the compiler plugin.
-keepclassmembers class androidx.compose.** {
    public static ** Companion;
}
-dontwarn androidx.compose.**

# ── org.json / android.util.Base64 ────────────────────────────────
# Framework classes, not packaged ones; R8 only needs to know not to warn.
-dontwarn org.json.**
-dontwarn android.util.Base64

# ── Coroutines ─────────────────────────────────────────────────────
-dontwarn kotlinx.coroutines.**
-keepclassmembers class kotlin.Metadata { *; }