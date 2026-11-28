import java.util.Properties
import org.gradle.api.tasks.Copy

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Compile the canonical ES 3.0 UI shader from the UI repository, never a
// stale in-app copy. `deps/sofuu-ui` is the post-split submodule path;
// `crates/sofuu-ui` is the pre-split in-tree path. Generating the asset at
// build time keeps the APK in lock-step with the Rust GLES renderer.
val uiShaderSource = sequenceOf(
    rootProject.file("../deps/sofuu-ui/shaders/glsl/ui_shaders_es3.glsl"),
    rootProject.file("../crates/sofuu-ui/shaders/glsl/ui_shaders_es3.glsl"),
).firstOrNull { it.exists() }
    ?: error("ui_shaders_es3.glsl not found (expected deps/sofuu-ui or crates/sofuu-ui)")
val uiShaderAssets = layout.buildDirectory.dir("generated/uiAssets")
val syncUiShader = tasks.register<Copy>("syncUiShader") {
    from(uiShaderSource)
    into(uiShaderAssets)
}
tasks.named("preBuild").configure { dependsOn(syncUiShader) }

// Prebuilt engine payload (libsofuu.so per ABI), staged by
// scripts/mobile/build_engine_android.sh. v2 reuses v1's staged payload
// read-only so the 270 MB of debug .so files are not duplicated in git; point
// -PsofuuPrebuiltJniDir=<dir> at a v2-specific staging directory to detach.
val prebuiltJniDir: String = (findProperty("sofuuPrebuiltJniDir") as String?)
    ?: rootProject.file("../sofuu-android/app/src/main/jniLibs").absolutePath

// Release signing. Optional on purpose: keystore.properties is not checked
// in, so a fresh clone still configures and still runs `assembleDebug`.
// Without it, assembleRelease is signed with the debug key and warns loudly.
val keystorePropsFile = rootProject.file("keystore.properties")
val keystoreProps = Properties().apply {
    if (keystorePropsFile.exists()) {
        keystorePropsFile.inputStream().use { load(it) }
    }
}
val hasSigning = keystoreProps.getProperty("storeFile") != null

android {
    namespace = "com.sofuu"
    compileSdk = 35
    ndkVersion = "27.0.12077973"

    defaultConfig {
        // Distinct applicationId: v2 installs side by side with v1, while the
        // Java package stays com.sofuu so the existing JNI symbol names hold.
        applicationId = "com.sofuu.v2"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cFlags += "-O2"
                arguments += "-DSOFUU_PREBUILT_JNI_DIR=$prebuiltJniDir"
            }
        }
        ndk {
            abiFilters += "arm64-v8a"
            abiFilters += "x86_64"
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = true
        }
    }

    signingConfigs {
        if (hasSigning) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = if (hasSigning) {
                signingConfigs.getByName("release")
            } else {
                logger.warn("keystore.properties not found — signing the release APK with the DEBUG key (test builds only)")
                signingConfigs.getByName("debug")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets.getByName("main") {
        assets.srcDir(uiShaderAssets)
        jniLibs.srcDir(prebuiltJniDir)
    }

    packaging {
        jniLibs.useLegacyPackaging = false
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.security.crypto)
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation("androidx.test:core:1.6.1")
}

// androidx.core 1.15.0 asks for collection-jvm 1.4.2, an artifact that is not
// in the offline cache on this machine (1.4.4/1.5.0 are). Forcing the cached
// patch release keeps `--offline` builds reproducible without a network fetch;
// collection 1.4.x is binary-compatible across patches.
configurations.configureEach {
    resolutionStrategy.force("androidx.collection:collection-jvm:1.4.4")
}
