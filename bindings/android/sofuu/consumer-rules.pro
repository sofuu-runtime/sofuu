# Keep the JNI entry points and the Kotlin surface they bind to. The native
# library resolves these names at runtime; R8 shrinking them would produce
# UnsatisfiedLinkError in a release APK.
-keep class com.sofuu.runtime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
