/* examples/headless/KotlinSample/jni_bridge.c
 *
 * JNI bridge for SofuuBridge.kt — wraps the libsofuu C ABI for Android.
 *
 * Compile as part of the NDK shared library (libsofuu_jni.so) alongside
 * linking against libsofuu-android-<abi>.so. See SofuuBridge.kt for usage.
 */

#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include "sofuu_embed.h"

/* Cache the JVM + class refs for callback threading. */
static JavaVM *g_jvm = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

/* SofuuBridge.nativeAbiVersion() → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeAbiVersion(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    return (jint)sofuu_embed_abi_version();
}

/* SofuuBridge.nativeRtNew(config) → long (pointer as jlong) */
JNIEXPORT jlong JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeRtNew(JNIEnv *env, jclass cls, jstring config) {
    (void)cls;
    const char *config_c = NULL;
    if (config != NULL) {
        config_c = (*env)->GetStringUTFChars(env, config, NULL);
    }
    SofuuRuntime *rt = sofuu_rt_new(config_c);
    if (config != NULL && config_c != NULL) {
        (*env)->ReleaseStringUTFChars(env, config, config_c);
    }
    return (jlong)(intptr_t)rt;
}

/* SofuuBridge.nativeRtFree(rt) */
JNIEXPORT void JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeRtFree(JNIEnv *env, jclass cls, jlong rt_ptr) {
    (void)env; (void)cls;
    if (rt_ptr != 0) {
        sofuu_rt_free((SofuuRuntime *)(intptr_t)rt_ptr);
    }
}

/* SofuuBridge.nativeRtEval(rt, source) → String? */
JNIEXPORT jstring JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeRtEval(JNIEnv *env, jclass cls, jlong rt_ptr, jstring source) {
    (void)cls;
    if (rt_ptr == 0 || source == NULL) return NULL;

    const char *source_c = (*env)->GetStringUTFChars(env, source, NULL);
    if (source_c == NULL) return NULL;

    char *out = NULL;
    int rc = sofuu_rt_eval((SofuuRuntime *)(intptr_t)rt_ptr, source_c, &out);
    (*env)->ReleaseStringUTFChars(env, source, source_c);

    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }

    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}

/* SofuuBridge.nativeRtCall(rt, method, args) → String? */
JNIEXPORT jstring JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeRtCall(JNIEnv *env, jclass cls, jlong rt_ptr, jstring method, jstring args) {
    (void)cls;
    if (rt_ptr == 0 || method == NULL) return NULL;

    const char *method_c = (*env)->GetStringUTFChars(env, method, NULL);
    const char *args_c = (args != NULL) ? (*env)->GetStringUTFChars(env, args, NULL) : NULL;

    char *out = NULL;
    int rc = sofuu_rt_call((SofuuRuntime *)(intptr_t)rt_ptr, method_c, args_c, &out);

    if (method_c != NULL) (*env)->ReleaseStringUTFChars(env, method, method_c);
    if (args != NULL && args_c != NULL) (*env)->ReleaseStringUTFChars(env, args, args_c);

    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }

    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}
