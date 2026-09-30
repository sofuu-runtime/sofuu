# bindings/android/sofuu/src/main/cpp/CMakeLists.txt
# Native build for the com.sofuu:sofuu-android AAR.
#
# The heavy lifting (the whole AI runtime) is already compiled into the
# prebuilt libsofuu .so that `make dist-android` produces and that CI stages
# into jniLibs/<abi>/. This only builds the thin JNI shim that exposes the C
# ABI to Kotlin.
#
# If the prebuilt .so is missing, the build still succeeds (so a fresh clone
# can run `gradle tasks` without an NDK pack) but any call throws
# UnsatisfiedLinkError at runtime with a message pointing here — a missing
# pack is a packaging mistake, not a silent no-op.

cmake_minimum_required(VERSION 3.22.1)
project(sofuu_jni C)

set(CMAKE_C_STANDARD 11)

set(SOFUU_PREBUILT "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsofuu.so")

if(EXISTS "${SOFUU_PREBUILT}")
    add_library(sofuu SHARED IMPORTED)
    set_target_properties(sofuu PROPERTIES IMPORTED_LOCATION "${SOFUU_PREBUILT}")
    set(SOFUU_LINK_LIB sofuu)
    message(STATUS "Sofuu: linking prebuilt ${SOFUU_PREBUILT}")
else()
    set(SOFUU_LINK_LIB "")
    message(WARNING
        "Sofuu: no prebuilt libsofuu.so for ${ANDROID_ABI}.\n"
        "Run `make dist-android` (or let CI stage it) before packaging the AAR. "
        "The AAR will build, but calls into the runtime will fail to link at runtime.")
endif()

add_library(sofuu_jni SHARED jni_bridge.c)
target_include_directories(sofuu_jni PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sofuu_jni ${SOFUU_LINK_LIB} log)
