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
#include <android/log.h>
#include "sofuu_embed.h"

/* Cache the JVM + class refs for callback threading. */
static JavaVM *g_jvm = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

/* ── UTF-8 at the JNI boundary ────────────────────────────────────
 *
 * JNI's "UTF-8" is MODIFIED UTF-8: GetStringUTFChars hands back CESU-8 for
 * anything outside the BMP (an emoji arrives as a 6-byte surrogate *pair*),
 * and NewStringUTF expects that same encoding. The runtime speaks standard
 * UTF-8 JSON on both sides, so the text paths below convert explicitly:
 *
 *   sofuu_mutf8_to_utf8(GetStringUTFChars result)  — Kotlin → C
 *   sofuu_utf8_to_jstring(C string)                — C → Kotlin
 *
 * Skipping this is not loud. An emoji in a prompt reaches serde_json as
 * invalid UTF-8, so the call fails as a bare null; an emoji in a result
 * comes back as mojibake. (ART only learned to accept 4-byte sequences in
 * NewStringUTF around API 29 — minSdk here is 24.)
 *
 * Scope: the text in/out paths (eval, call, streaming, host tools). The
 * vector/voice entry points below still use GetStringUTFChars directly;
 * their inputs are short labels whose non-BMP behavior is unchanged.
 */

/* standard UTF-8 → a Java String. NULL only on allocation failure; bytes
 * that are not valid UTF-8 become U+FFFD rather than an error. */
static jstring sofuu_utf8_to_jstring(JNIEnv *env, const char *in) {
    if (in == NULL) return NULL;
    size_t n = strlen(in);
    /* Every UTF-8 sequence decodes to at most as many UTF-16 units as it
     * has bytes (1B→1, 2B→1, 3B→1, 4B→2), so this always fits. */
    jchar *units = (jchar *)malloc((n + 1) * sizeof(jchar));
    if (units == NULL) return NULL;
    size_t u = 0, i = 0;
    while (i < n) {
        unsigned char b = (unsigned char)in[i];
        unsigned int cp;
        if (b < 0x80) {
            cp = b;
            i += 1;
        } else if ((b & 0xE0) == 0xC0 && i + 1 < n) {
            cp = ((b & 0x1Fu) << 6) | ((unsigned char)in[i + 1] & 0x3Fu);
            i += 2;
        } else if ((b & 0xF0) == 0xE0 && i + 2 < n) {
            cp = ((b & 0x0Fu) << 12) |
                 (((unsigned char)in[i + 1] & 0x3Fu) << 6) |
                 ((unsigned char)in[i + 2] & 0x3Fu);
            i += 3;
        } else if ((b & 0xF8) == 0xF0 && i + 3 < n) {
            cp = ((b & 0x07u) << 18) |
                 (((unsigned char)in[i + 1] & 0x3Fu) << 12) |
                 (((unsigned char)in[i + 2] & 0x3Fu) << 6) |
                 ((unsigned char)in[i + 3] & 0x3Fu);
            i += 4;
        } else {
            cp = 0xFFFD; /* malformed byte — never fail the whole string over it */
            i += 1;
        }
        if (cp >= 0x10000) {
            cp -= 0x10000;
            units[u++] = (jchar)(0xD800 + (cp >> 10));
            units[u++] = (jchar)(0xDC00 + (cp & 0x3FF));
        } else {
            units[u++] = (jchar)cp;
        }
    }
    jstring s = (*env)->NewString(env, units, (jsize)u);
    free(units);
    return s;
}

/* modified UTF-8 (a GetStringUTFChars result) → malloc'd standard UTF-8,
 * or NULL on allocation failure. Surrogate pairs *shrink* (6 bytes → 4),
 * but a malformed byte expands to U+FFFD (3 bytes), so the buffer is sized
 * for the worst case. The caller frees. */
static char *sofuu_mutf8_to_utf8(const char *in) {
    if (in == NULL) return NULL;
    size_t n = strlen(in);
    char *out = (char *)malloc(n * 3 + 1);
    if (out == NULL) return NULL;
    size_t o = 0, i = 0;
    while (i < n) {
        unsigned char b = (unsigned char)in[i];
        unsigned int cp;
        if (b < 0x80) {
            cp = b;
            i += 1;
        } else if ((b & 0xE0) == 0xC0 && i + 1 < n) {
            cp = ((b & 0x1Fu) << 6) | ((unsigned char)in[i + 1] & 0x3Fu);
            i += 2;
        } else if ((b & 0xF0) == 0xE0 && i + 2 < n) {
            cp = ((b & 0x0Fu) << 12) |
                 (((unsigned char)in[i + 1] & 0x3Fu) << 6) |
                 ((unsigned char)in[i + 2] & 0x3Fu);
            i += 3;
            /* CESU-8: a high surrogate is always followed by its low half. */
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 2 < n &&
                ((unsigned char)in[i] & 0xF0) == 0xE0) {
                unsigned int lo = (((unsigned char)in[i] & 0x0Fu) << 12) |
                                  (((unsigned char)in[i + 1] & 0x3Fu) << 6) |
                                  ((unsigned char)in[i + 2] & 0x3Fu);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 3;
                }
            }
        } else {
            cp = 0xFFFD;
            i += 1;
        }
        if (cp < 0x80) {
            out[o++] = (char)cp;
        } else if (cp < 0x800) {
            out[o++] = (char)(0xC0 | (cp >> 6));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out[o++] = (char)(0xE0 | (cp >> 12));
            out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        } else {
            out[o++] = (char)(0xF0 | (cp >> 18));
            out[o++] = (char)(0x80 | ((cp >> 12) & 0x3F));
            out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[o++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    out[o] = '\0';
    return out;
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

    const char *source_mu = (*env)->GetStringUTFChars(env, source, NULL);
    if (source_mu == NULL) return NULL;
    char *source_c = sofuu_mutf8_to_utf8(source_mu);
    (*env)->ReleaseStringUTFChars(env, source, source_mu);
    if (source_c == NULL) return NULL;

    char *out = NULL;
    int rc = sofuu_rt_eval((SofuuRuntime *)(intptr_t)rt_ptr, source_c, &out);
    free(source_c);

    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }

    jstring result = sofuu_utf8_to_jstring(env, out);
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

    const char *method_mu = (*env)->GetStringUTFChars(env, method, NULL);
    if (method_mu == NULL) return NULL;
    char *method_c = sofuu_mutf8_to_utf8(method_mu);
    (*env)->ReleaseStringUTFChars(env, method, method_mu);
    if (method_c == NULL) return NULL;

    char *args_c = NULL;
    if (args != NULL) {
        const char *args_mu = (*env)->GetStringUTFChars(env, args, NULL);
        if (args_mu != NULL) {
            args_c = sofuu_mutf8_to_utf8(args_mu);
            (*env)->ReleaseStringUTFChars(env, args, args_mu);
        }
    }

    char *out = NULL;
    int rc = sofuu_rt_call((SofuuRuntime *)(intptr_t)rt_ptr, method_c, args_c, &out);
    free(method_c);
    free(args_c);

    if (rc != SOFUU_OK || out == NULL) {
        if (out) sofuu_free(out);
        return NULL;
    }

    jstring result = sofuu_utf8_to_jstring(env, out);
    sofuu_free(out);
    return result;
}

/* ── Streaming (sofuu_rt_call_stream) ────────────────────────────
 *
 * A streaming call BLOCKS the calling thread until the stream ends: the
 * runtime is thread-affine (one runtime per thread), and the events are
 * delivered by the same event loop that runs the call. Events arrive on
 * the engine thread — a Java thread — as JSON strings, and every stream
 * ends with exactly one terminal `{"kind":"done",…}` event.
 *
 * The live cancel id is readable *during* the call:
 * sofuu_rt_call_stream writes *out_cancel_id before the JS starts, and the
 * slot below keeps a pointer to it — same thread, so reading it from a
 * callback is sequenced after the write. That is the only in-flight cancel
 * this ABI has: the id is not knowable before the call and cancel from a
 * foreign thread is not allowed. A listener that returns false (or throws)
 * is therefore the host's real "stop" switch.
 */

typedef struct SofuuStreamSlot {
    jobject listener;    /* global ref: StreamListener */
    jmethodID on_event;  /* onStreamEvent(String) -> boolean */
    SofuuRuntime *rt;
    uint64_t *cancel_id; /* the caller's slot, filled before the first event */
} SofuuStreamSlot;

static void stream_event_trampoline(const char *event_json, void *opaque) {
    SofuuStreamSlot *slot = (SofuuStreamSlot *)opaque;
    JNIEnv *env = NULL;
    if (slot == NULL || g_jvm == NULL) return;
    /* Fires on the engine thread — already attached to the JVM, so GetEnv
     * (not AttachCurrentThread) is the right call. */
    if ((*g_jvm)->GetEnv(g_jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK || env == NULL) return;

    jstring s = sofuu_utf8_to_jstring(env, event_json ? event_json : "");
    if (s == NULL) return;
    jboolean keep = (*env)->CallBooleanMethod(env, slot->listener, slot->on_event, s);
    (*env)->DeleteLocalRef(env, s);

    if ((*env)->ExceptionCheck(env)) {
        /* A throwing listener leaves a pending exception on the engine
         * thread, and every later JNI call on it would run with that
         * exception still armed. Clear it — and treat the throw as a stop,
         * because continuing means streaming into a host that has just
         * proven it cannot receive. */
        __android_log_print(ANDROID_LOG_ERROR, "sofuu-js",
                            "stream listener threw; stream cancelled and exception cleared");
        (*env)->ExceptionClear(env);
        keep = JNI_FALSE;
    }
    if (!keep && slot->cancel_id != NULL && *slot->cancel_id != 0) {
        /* Same thread as the pump — which is what sofuu_rt_cancel needs.
         * After the terminal `done` event this is a harmless no-op. */
        (void)sofuu_rt_cancel(slot->rt, *slot->cancel_id);
    }
}

/* SofuuBridge.nativeCallStream(rt, method, argsJson, listener) → boolean */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeCallStream(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                    jstring method, jstring args, jobject listener) {
    (void)cls;
    if (rt_ptr == 0 || method == NULL || listener == NULL) return JNI_FALSE;

    jclass iface = (*env)->GetObjectClass(env, listener);
    if (iface == NULL) return JNI_FALSE;
    jmethodID on_event = (*env)->GetMethodID(env, iface, "onStreamEvent", "(Ljava/lang/String;)Z");
    (*env)->DeleteLocalRef(env, iface);
    if (on_event == NULL) return JNI_FALSE; /* pending NoSuchMethodError reaches Kotlin */

    const char *method_mu = (*env)->GetStringUTFChars(env, method, NULL);
    if (method_mu == NULL) return JNI_FALSE;
    char *method_c = sofuu_mutf8_to_utf8(method_mu);
    (*env)->ReleaseStringUTFChars(env, method, method_mu);
    if (method_c == NULL) return JNI_FALSE;

    char *args_c = NULL;
    if (args != NULL) {
        const char *args_mu = (*env)->GetStringUTFChars(env, args, NULL);
        if (args_mu != NULL) {
            args_c = sofuu_mutf8_to_utf8(args_mu);
            (*env)->ReleaseStringUTFChars(env, args, args_mu);
        }
    }

    uint64_t cancel_id = 0;
    SofuuStreamSlot slot;
    slot.listener = (*env)->NewGlobalRef(env, listener);
    slot.on_event = on_event;
    slot.rt = (SofuuRuntime *)(intptr_t)rt_ptr;
    slot.cancel_id = &cancel_id;
    if (slot.listener == NULL) {
        free(method_c);
        free(args_c);
        return JNI_FALSE;
    }

    int rc = sofuu_rt_call_stream(slot.rt, method_c, args_c,
                                  stream_event_trampoline, &slot, &cancel_id);

    (*env)->DeleteGlobalRef(env, slot.listener);
    free(method_c);
    free(args_c);
    return rc == SOFUU_OK ? JNI_TRUE : JNI_FALSE;
}

/* ── Host tools (PLAN-EMBED-SDK E2) ──────────────────────────────
 *
 * sofuu_rt_register_host_tool hands the agent a native capability with no
 * JS in the loop — camera, location, the share sheet, the app's own data.
 * It is the one part of the ABI that exists specifically for a mobile
 * embedder: things a QuickJS sandbox cannot reach.
 *
 * Three things shape this shim:
 *
 * 1. The C callback signature carries no tool name:
 *      const char *(*fn)(const char *args_json, uint64_t call_id, void *opaque)
 *    so the Kotlin receiver, its method id and the runtime it answers for
 *    all ride in `opaque` — one heap slot per registration.
 *
 * 2. The returned pointer only has to stay valid until the callback
 *    returns (the library copies it — capi js_host_tool_call), but a Java
 *    string has no address to hand back. So the result is copied into a
 *    per-thread buffer that stays valid until that thread's next host-tool
 *    call. Returning NULL means "pending" — the host owns the call and
 *    settles it with nativeHostToolResolve, from any thread.
 *
 * 3. The ABI keeps its own C pointer for the runtime's lifetime and gives
 *    no "runtime destroyed" callback, so a registration can never be freed
 *    from here. Every slot is chained onto g_host_tools instead, and
 *    nativeFreeHostTools releases them (slot, name, and the JNI global ref
 *    that would otherwise pin the host's Activity) once the runtime is
 *    gone. The list is touched only from the engine thread: registrations
 *    arrive there, and SofuuBridge.close() frees there.
 */

typedef struct SofuuHostToolSlot {
    SofuuRuntime *rt;    /* the runtime holding this registration */
    char *name;
    jobject listener;    /* global ref: HostTool */
    jmethodID method;    /* invoke(String, long) -> String? */
    struct SofuuHostToolSlot *next;
} SofuuHostToolSlot;

static SofuuHostToolSlot *g_host_tools = NULL;

/* One per host-tool call thread; see note 2 above. */
static __thread char *g_tool_result = NULL;
static __thread size_t g_tool_result_cap = 0;

static const char *host_tool_trampoline(const char *args_json,
                                        uint64_t call_id,
                                        void *opaque) {
    SofuuHostToolSlot *slot = (SofuuHostToolSlot *)opaque;
    JNIEnv *env = NULL;
    if (slot == NULL || g_jvm == NULL) return NULL;
    /* The call arrives on the engine thread — a Java thread already
     * attached, so GetEnv is right here (same reasoning as the stream
     * trampoline). */
    if ((*g_jvm)->GetEnv(g_jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK || env == NULL) return NULL;

    jstring jargs = sofuu_utf8_to_jstring(env, args_json ? args_json : "{}");
    if (jargs == NULL) return NULL;

    jobject jres = (*env)->CallObjectMethod(env, slot->listener, slot->method,
                                            jargs, (jlong)call_id);
    (*env)->DeleteLocalRef(env, jargs);
    if ((*env)->ExceptionCheck(env)) {
        /* A throwing tool must not poison the engine thread. Unlike the
         * app-level bridge (which reports the throw and leaves the call
         * pending until the agent's 30 s tool timeout), the turn here is
         * kept moving: the throw becomes the tool's result so the model
         * can react immediately, and logcat keeps the stack. */
        (*env)->ExceptionDescribe(env);
        (*env)->ExceptionClear(env);
        return "{\"error\":\"host tool threw an exception (see logcat)\"}";
    }
    if (jres == NULL) return NULL; /* deferred to the host */

    const char *utf_mu = (*env)->GetStringUTFChars(env, (jstring)jres, NULL);
    if (utf_mu == NULL) {
        (*env)->DeleteLocalRef(env, jres);
        return NULL;
    }
    char *utf8 = sofuu_mutf8_to_utf8(utf_mu);
    (*env)->ReleaseStringUTFChars(env, (jstring)jres, utf_mu);
    (*env)->DeleteLocalRef(env, jres);
    if (utf8 == NULL) return NULL;

    size_t n = strlen(utf8);
    if (n + 1 > g_tool_result_cap) {
        char *grown = (char *)realloc(g_tool_result, n + 1);
        if (grown == NULL) {
            free(utf8);
            return NULL;
        }
        g_tool_result = grown;
        g_tool_result_cap = n + 1;
    }
    memcpy(g_tool_result, utf8, n + 1);
    free(utf8);
    return g_tool_result;
}

/* SofuuBridge.nativeRegisterHostTool(rt, name, description, paramsJson, tool) */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeRegisterHostTool(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                          jstring name, jstring description,
                                                          jstring params_json, jobject listener) {
    (void)cls;
    if (rt_ptr == 0 || name == NULL || listener == NULL) return JNI_FALSE;

    jclass iface = (*env)->GetObjectClass(env, listener);
    if (iface == NULL) return JNI_FALSE;
    jmethodID mid = (*env)->GetMethodID(env, iface, "invoke",
                                        "(Ljava/lang/String;J)Ljava/lang/String;");
    (*env)->DeleteLocalRef(env, iface);
    if (mid == NULL) return JNI_FALSE;

    const char *name_mu = (*env)->GetStringUTFChars(env, name, NULL);
    if (name_mu == NULL) return JNI_FALSE;
    char *name_c = sofuu_mutf8_to_utf8(name_mu);
    (*env)->ReleaseStringUTFChars(env, name, name_mu);
    if (name_c == NULL) return JNI_FALSE;

    SofuuHostToolSlot *slot = (SofuuHostToolSlot *)calloc(1, sizeof(SofuuHostToolSlot));
    if (slot == NULL) {
        free(name_c);
        return JNI_FALSE;
    }
    slot->name = name_c;
    slot->listener = (*env)->NewGlobalRef(env, listener);
    slot->method = mid;
    slot->rt = (SofuuRuntime *)(intptr_t)rt_ptr;

    /* description / params are optional (NULL means "the default" to the ABI). */
    char *desc_c = NULL, *params_c = NULL;
    if (description != NULL) {
        const char *mu = (*env)->GetStringUTFChars(env, description, NULL);
        if (mu != NULL) {
            desc_c = sofuu_mutf8_to_utf8(mu);
            (*env)->ReleaseStringUTFChars(env, description, mu);
        }
    }
    if (params_json != NULL) {
        const char *mu = (*env)->GetStringUTFChars(env, params_json, NULL);
        if (mu != NULL) {
            params_c = sofuu_mutf8_to_utf8(mu);
            (*env)->ReleaseStringUTFChars(env, params_json, mu);
        }
    }

    int rc = sofuu_rt_register_host_tool(slot->rt, slot->name, desc_c, params_c,
                                         host_tool_trampoline, (void *)slot);
    free(desc_c);
    free(params_c);

    if (rc != SOFUU_OK) {
        /* The runtime kept nothing, so this slot is ours to free. */
        if (slot->listener) (*env)->DeleteGlobalRef(env, slot->listener);
        free(slot->name);
        free(slot);
        return JNI_FALSE;
    }
    /* Chained only after the ABI accepted it: it now holds `slot` as the
     * opaque pointer for this registration. */
    slot->next = g_host_tools;
    g_host_tools = slot;
    return JNI_TRUE;
}

/* SofuuBridge.nativeHostToolResolve(rt, callId, resultJson | null, error | null) */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeHostToolResolve(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                         jlong call_id, jstring result_json,
                                                         jstring error) {
    (void)cls;
    if (rt_ptr == 0 || call_id == 0) return JNI_FALSE;

    char *res_c = NULL, *err_c = NULL;
    if (result_json != NULL) {
        const char *mu = (*env)->GetStringUTFChars(env, result_json, NULL);
        if (mu != NULL) {
            res_c = sofuu_mutf8_to_utf8(mu);
            (*env)->ReleaseStringUTFChars(env, result_json, mu);
        }
    }
    if (error != NULL) {
        const char *mu = (*env)->GetStringUTFChars(env, error, NULL);
        if (mu != NULL) {
            err_c = sofuu_mutf8_to_utf8(mu);
            (*env)->ReleaseStringUTFChars(env, error, mu);
        }
    }

    int rc = sofuu_rt_host_tool_resolve((SofuuRuntime *)(intptr_t)rt_ptr,
                                        (uint64_t)call_id, res_c, err_c);
    free(res_c);
    free(err_c);
    return rc == SOFUU_OK ? JNI_TRUE : JNI_FALSE;
}

/* SofuuBridge.nativeFreeHostTools(rt) → int
 *
 * Release every registration made against `rt`: the slot, the name, and
 * the JNI global ref pinning the Kotlin receiver. Must run AFTER
 * sofuu_rt_free(rt) — while the runtime lives the ABI still holds these
 * pointers. Returns how many registrations were released. */
JNIEXPORT jint JNICALL
Java_com_sofuu_runtime_SofuuBridge_nativeFreeHostTools(JNIEnv *env, jclass cls, jlong rt_ptr) {
    (void)cls;
    SofuuHostToolSlot **link = &g_host_tools;
    jint freed = 0;
    while (*link != NULL) {
        SofuuHostToolSlot *slot = *link;
        if (slot->rt == (SofuuRuntime *)(intptr_t)rt_ptr) {
            *link = slot->next;
            if (slot->listener) (*env)->DeleteGlobalRef(env, slot->listener);
            free(slot->name);
            free(slot);
            freed++;
        } else {
            link = &slot->next;
        }
    }
    return freed;
}
