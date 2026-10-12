# sofuu-android JNI bridge — links the prebuilt libsofuu per ABI.
cmake_minimum_required(VERSION 3.22.1)
project(sofuu_jni)

add_library(sofuu SHARED IMPORTED)
set_target_properties(sofuu PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsofuu.so
    # Without IMPORTED_SONAME the linker bakes the build-time absolute
    # path into DT_NEEDED and the APK fails to load at runtime.
    IMPORTED_SONAME libsofuu.so
)

add_library(sofuu_embed INTERFACE)
target_include_directories(sofuu_embed INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_library(sofuu_jni SHARED jni_bridge.c)
target_link_libraries(sofuu_jni sofuu sofuu_embed log)

# targetSdk 35 requires every .so in the APK to load on Android 15+ devices
# with a 16 KiB page size (most current Pixels). The NDK toolchain only
# aligns the FIRST segment by default, so this lib has to ask explicitly —
# the prebuilt libsofuu.so gets the same flag on the Rust link side (see
# scripts/mobile/build_engine_android.sh).
target_link_options(sofuu_jni PRIVATE "-Wl,-z,max-page-size=16384")
