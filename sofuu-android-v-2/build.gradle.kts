// Root build — plugin versions live in gradle/libs.versions.toml.
//
// This project is the Compose-free edition (PLAN-SOFUU-ANDROID-V2-GPU-REWRITE):
// every pixel is drawn by sofuu-ui through ui_gles.c. There is deliberately
// no `kotlin.compose` plugin and no Compose/Material dependency here.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}
