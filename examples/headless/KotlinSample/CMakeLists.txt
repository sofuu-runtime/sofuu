# examples/headless/KotlinSample/CMakeLists.txt
#
# CMake build for the Sofuu JNI bridge (Android). Links against the
# prebuilt libsofuu .so for the target ABI.
#
# Usage (in app/build.gradle):
#   externalNativeBuild { cmake { path "src/main/cpp/CMakeLists.txt" } }
#
# Copy libsofuu-android-<abi>.so into app/src/main/jniLibs/<abi>/

cmake_minimum_required(VERSION 3.22.1)
project(sofuu_jni)

# Import the prebuilt libsofuu shared library.
add_library(sofuu SHARED IMPORTED)
set_target_properties(sofuu PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsofuu.so
)

# Include the header from the dist pack.
add_library(sofuu_embed INTERFACE)
target_include_directories(sofuu_embed INTERFACE ${CMAKE_SOURCE_DIR}/include)

# Build the JNI bridge.
add_library(sofuu_jni SHARED jni_bridge.c)
target_link_libraries(sofuu_jni sofuu sofuu_embed log)
