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

/* H-E1: SofuuBridge.nativeEmbedLocal(rt, text, space) → float[]?
 * Direct vector path (no JSON): malloc'd by libsofuu, copied into a
 * jfloatArray, freed with sofuu_free(). space NULL → default. */
JNIEXPORT jfloatArray JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeEmbedLocal(JNIEnv *env, jclass cls, jlong rt_ptr, jstring text, jstring space) {
    (void)cls;
    if (rt_ptr == 0 || text == NULL) return NULL;

    const char *text_c = (*env)->GetStringUTFChars(env, text, NULL);
    if (text_c == NULL) return NULL;
    const char *space_c = (space != NULL) ? (*env)->GetStringUTFChars(env, space, NULL) : NULL;

    float *out = NULL;
    size_t dim = 0;
    int rc = sofuu_embed_local((SofuuRuntime *)(intptr_t)rt_ptr, text_c, space_c, &out, &dim);

    (*env)->ReleaseStringUTFChars(env, text, text_c);
    if (space != NULL && space_c != NULL) (*env)->ReleaseStringUTFChars(env, space, space_c);

    if (rc != SOFUU_OK || out == NULL || dim == 0 || dim > (size_t)0x7fffffff) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jfloatArray arr = (*env)->NewFloatArray(env, (jsize)dim);
    if (arr == NULL) {
        sofuu_free(out);
        return NULL;
    }
    (*env)->SetFloatArrayRegion(env, arr, 0, (jsize)dim, out);
    sofuu_free(out);
    return arr;
}

/* H-E1: SofuuBridge.nativeEmbedBatch(rt, texts, space) → float[][]?
 * Row-major flat buffer from libsofuu, split into one jfloatArray per
 * input. Row local refs are deleted as we go so large batches cannot
 * overflow the JNI local-ref table. */
JNIEXPORT jobjectArray JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeEmbedBatch(JNIEnv *env, jclass cls, jlong rt_ptr, jobjectArray texts, jstring space) {
    (void)cls;
    jclass farray = (*env)->FindClass(env, "[F");
    if (farray == NULL) return NULL;
    if (rt_ptr == 0 || texts == NULL) return NULL;

    jsize n = (*env)->GetArrayLength(env, texts);
    if (n <= 0) return (*env)->NewObjectArray(env, 0, farray, NULL);

    const char **cstrs = (const char **)malloc((size_t)n * sizeof(*cstrs));
    jstring *refs = (jstring *)malloc((size_t)n * sizeof(*refs));
    if (cstrs == NULL || refs == NULL) {
        free(cstrs);
        free(refs);
        return NULL;
    }
    for (jsize i = 0; i < n; i++) refs[i] = NULL;
    const char *space_c = (space != NULL) ? (*env)->GetStringUTFChars(env, space, NULL) : NULL;

    jsize i;
    for (i = 0; i < n; i++) {
        jstring s = (jstring)(*env)->GetObjectArrayElement(env, texts, i);
        if (s == NULL) break;
        refs[i] = s;
        cstrs[i] = (*env)->GetStringUTFChars(env, s, NULL);
        if (cstrs[i] == NULL) break;
    }
    float *out = NULL;
    size_t out_n = 0, dim = 0;
    int rc = (i == n)
        ? sofuu_embed_batch((SofuuRuntime *)(intptr_t)rt_ptr, cstrs, (size_t)n,
                            space_c, &out, &out_n, &dim)
        : SOFUU_ERR_INVALID_ARG;
    for (jsize k = 0; k < i; k++) {
        if (cstrs[k] != NULL) (*env)->ReleaseStringUTFChars(env, refs[k], cstrs[k]);
    }
    if (space != NULL && space_c != NULL) (*env)->ReleaseStringUTFChars(env, space, space_c);
    free(cstrs);
    free(refs);

    if (rc != SOFUU_OK || out == NULL || out_n != (size_t)n || dim == 0) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jobjectArray res = (*env)->NewObjectArray(env, (jsize)out_n, farray, NULL);
    if (res == NULL) {
        sofuu_free(out);
        return NULL;
    }
    for (size_t r = 0; r < out_n; r++) {
        jfloatArray row = (*env)->NewFloatArray(env, (jsize)dim);
        if (row == NULL) {
            sofuu_free(out);
            return NULL;
        }
        (*env)->SetFloatArrayRegion(env, row, 0, (jsize)dim, out + r * dim);
        (*env)->SetObjectArrayElement(env, res, (jsize)r, row);
        (*env)->DeleteLocalRef(env, row);
    }
    sofuu_free(out);
    return res;
}

/* M1: SofuuBridge.nativeEmbedImage(rt, bytes) → float[]?
 * Image bytes (PNG/JPEG) into img1-64. Input capped at 32MB (the Rust
 * side enforces its own pixel caps after decode). */
JNIEXPORT jfloatArray JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeEmbedImage(JNIEnv *env, jclass cls, jlong rt_ptr, jbyteArray bytes) {
    (void)cls;
    if (rt_ptr == 0 || bytes == NULL) return NULL;
    jsize len = (*env)->GetArrayLength(env, bytes);
    if (len <= 0 || len > 32 * 1024 * 1024) return NULL;
    jbyte *buf = (jbyte *)malloc((size_t)len);
    if (buf == NULL) return NULL;
    (*env)->GetByteArrayRegion(env, bytes, 0, len, buf);
    float *out = NULL;
    size_t dim = 0;
    int rc = sofuu_embed_image((SofuuRuntime *)(intptr_t)rt_ptr, (const uint8_t *)buf, (size_t)len, &out, &dim);
    free(buf);
    if (rc != SOFUU_OK || out == NULL || dim == 0 || dim > (size_t)0x7fffffff) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jfloatArray arr = (*env)->NewFloatArray(env, (jsize)dim);
    if (arr == NULL) {
        sofuu_free(out);
        return NULL;
    }
    (*env)->SetFloatArrayRegion(env, arr, 0, (jsize)dim, out);
    sofuu_free(out);
    return arr;
}

/* H-E1: SofuuBridge.nativeEmbedInfo(rt) → String? (space manifest JSON) */
JNIEXPORT jstring JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeEmbedInfo(JNIEnv *env, jclass cls, jlong rt_ptr) {
    (void)cls;
    if (rt_ptr == 0) return NULL;
    char *out = NULL;
    int rc = sofuu_embed_info((SofuuRuntime *)(intptr_t)rt_ptr, &out);
    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}

/* M2C: SofuuBridge.nativeVoiceTranscribe(rt, audioB64, optsJson) → String?
 * Thin funnel routing to ai.transcribe (base64 bridge — hosts encode). */
JNIEXPORT jstring JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeVoiceTranscribe(JNIEnv *env, jclass cls, jlong rt_ptr, jstring audio_b64, jstring opts) {
    (void)cls;
    if (rt_ptr == 0 || audio_b64 == NULL) return NULL;
    const char *b64_c = (*env)->GetStringUTFChars(env, audio_b64, NULL);
    const char *opts_c = (opts != NULL) ? (*env)->GetStringUTFChars(env, opts, NULL) : NULL;
    if (b64_c == NULL) {
        if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);
        return NULL;
    }
    char *out = NULL;
    int rc = sofuu_voice_transcribe((SofuuRuntime *)(intptr_t)rt_ptr, b64_c, opts_c, &out);
    (*env)->ReleaseStringUTFChars(env, audio_b64, b64_c);
    if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);
    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}

/* M2C: SofuuBridge.nativeVoiceSpeak(rt, text, optsJson) → String? */
JNIEXPORT jstring JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeVoiceSpeak(JNIEnv *env, jclass cls, jlong rt_ptr, jstring text, jstring opts) {
    (void)cls;
    if (rt_ptr == 0 || text == NULL) return NULL;
    const char *text_c = (*env)->GetStringUTFChars(env, text, NULL);
    const char *opts_c = (opts != NULL) ? (*env)->GetStringUTFChars(env, opts, NULL) : NULL;
    if (text_c == NULL) {
        if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);
        return NULL;
    }
    char *out = NULL;
    int rc = sofuu_voice_speak((SofuuRuntime *)(intptr_t)rt_ptr, text_c, opts_c, &out);
    (*env)->ReleaseStringUTFChars(env, text, text_c);
    if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);
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
