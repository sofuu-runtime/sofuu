/* sofuu-android JNI bridge (Kotlin ↔ libsofuu C ABI v2).
 *
 * Wraps the documented embedding ABI (include/sofuu_embed.h) for the
 * Android app. Everything goes through the stable funnel/eval surface —
 * no engine-internal symbols, no struct-layout assumptions.
 *
 * Threading contract (docs/EMBEDDING.md): one SofuuRuntime is owned by
 * ONE thread. The app creates it on SofuuEngine's worker thread and makes
 * every call from that thread (the pump); the UI thread never touches
 * the runtime.
 *
 * HOME: Android has no $HOME, and the runtime derives config / brain /
 * session paths from it ($HOME/.sofuu, <project>/.sofuu). The app points
 * HOME at its private files dir (context.filesDir) through setHome()
 * BEFORE any runtime is created — setenv is process-wide, and nothing
 * else in the process reads HOME.
 */

#include <jni.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <android/log.h>
#include "sofuu_embed.h"

static JavaVM *g_jvm = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

/* SofuuBridge.nativeSetHome(path) — pin HOME for the process. */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeSetHome(JNIEnv *env, jclass cls, jstring path) {
    (void)cls;
    if (path == NULL) return;
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    if (p == NULL) return;
    setenv("HOME", p, 1);
    (*env)->ReleaseStringUTFChars(env, path, p);
}


/* SofuuBridge.nativeChdir(path) — set the process cwd (the engine's
 * tool-jail default + the cwd MCP children inherit). chdir is
 * process-wide on Linux; the app is single-purpose, so this is safe and
 * mirrors the desktop backend (which chdirs its engine thread before
 * the runtime boots). */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeChdir(JNIEnv *env, jclass cls, jstring path) {
    (void)cls;
    if (path == NULL) return;
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    if (p == NULL) return;
    if (chdir(p) != 0) {
        __android_log_print(ANDROID_LOG_WARN, "sofuu", "chdir %s failed", p);
    }
    (*env)->ReleaseStringUTFChars(env, path, p);
}

/* SofuuBridge.nativeAbiVersion() → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeAbiVersion(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    return (jint)sofuu_embed_abi_version();
}

/* SofuuBridge.nativeRtNew(config) → long (runtime pointer) */
JNIEXPORT jlong JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtNew(JNIEnv *env, jclass cls, jstring config) {
    (void)cls;
    const char *config_c = NULL;
    if (config != NULL) {
        config_c = (*env)->GetStringUTFChars(env, config, NULL);
        if (config_c == NULL) return 0;
    }
    SofuuRuntime *rt = sofuu_rt_new(config_c);
    if (config != NULL && config_c != NULL) {
        (*env)->ReleaseStringUTFChars(env, config, config_c);
    }
    return (jlong)(intptr_t)rt;
}

/* SofuuBridge.nativeRtFree(rt) */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtFree(JNIEnv *env, jclass cls, jlong rt_ptr) {
    (void)env; (void)cls;
    if (rt_ptr != 0) {
        sofuu_rt_free((SofuuRuntime *)(intptr_t)rt_ptr);
    }
}

/* SofuuBridge.nativeRtEval(rt, source) → String?
 * Returns the JSON-stringified completion value; NULL on failure. */
JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtEval(JNIEnv *env, jclass cls, jlong rt_ptr, jstring source) {
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

/* SofuuBridge.nativeRtEvalNb(rt, source) → String?
 * Non-blocking eval: identical JS semantics to nativeRtEval, but the
 * loop only ticks once (UV_RUN_NOWAIT) instead of draining until idle.
 * Returns the JSON-stringified completion value; NULL on failure. */
JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtEvalNb(JNIEnv *env, jclass cls, jlong rt_ptr, jstring source) {
    (void)cls;
    if (rt_ptr == 0 || source == NULL) return NULL;

    const char *source_c = (*env)->GetStringUTFChars(env, source, NULL);
    if (source_c == NULL) return NULL;

    char *out = NULL;
    int rc = sofuu_rt_eval_nb((SofuuRuntime *)(intptr_t)rt_ptr, source_c, &out);
    (*env)->ReleaseStringUTFChars(env, source, source_c);

    /* Never HIDE a JS exception behind null here: the envelope carries the
     * error text and the Kotlin side logs it. A caller that filters on
     * "ev"/"rp" treats the envelope as an empty drain. */
    if (rc != SOFUU_OK || out == NULL) {
        if (out) {
            jstring errEnv = (*env)->NewStringUTF(env, out);
            sofuu_free(out);
            return errEnv;
        }
        return (*env)->NewStringUTF(env,
            "{\"ok\":false,\"error\":{\"code\":\"internal\",\"message\":\"runtime returned no result\"}}");
    }
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}

/* SofuuBridge.nativeRtCall(rt, method, args) → String? (JSON envelope) */
JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtCall(JNIEnv *env, jclass cls, jlong rt_ptr,
                                               jstring method, jstring args) {
    (void)cls;
    if (rt_ptr == 0 || method == NULL) return NULL;

    const char *method_c = (*env)->GetStringUTFChars(env, method, NULL);
    const char *args_c = NULL;
    if (args != NULL) {
        args_c = (*env)->GetStringUTFChars(env, args, NULL);
    }
    if (method_c == NULL) return NULL;

    char *out = NULL;
    int rc = sofuu_rt_call((SofuuRuntime *)(intptr_t)rt_ptr, method_c, args_c, &out);

    (*env)->ReleaseStringUTFChars(env, method, method_c);
    if (args != NULL && args_c != NULL) {
        (*env)->ReleaseStringUTFChars(env, args, args_c);
    }

    if (out == NULL) return NULL;
    jstring result = (rc == SOFUU_OK) ? (*env)->NewStringUTF(env, out) : NULL;
    sofuu_free(out);
    return result;
}

/* ── Streaming (sofuu_rt_call_stream) ─────────────────────────────
 * The C callback fires on the engine thread with one JSON event string
 * (valid for the call's duration). We forward it to the Kotlin listener
 * through the JNI global ref it registered, then return the cancel id
 * the ABI handed out. */

static jobject g_stream_listener = NULL;
static jmethodID g_stream_on_event = NULL;

static void on_stream_event(const char *event_json, void *opaque) {
    (void)opaque;
    if (g_jvm == NULL || g_stream_listener == NULL || g_stream_on_event == NULL) return;
    JNIEnv *env = NULL;
    /* The callback runs on the engine thread — a Java thread, already
     * attached to the JVM, so GetEnv (not AttachCurrentThread) is right. */
    if ((*g_jvm)->GetEnv(g_jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK || env == NULL) return;
    jstring s = (*env)->NewStringUTF(env, event_json ? event_json : "");
    if (s == NULL) return;
    (*env)->CallVoidMethod(env, g_stream_listener, g_stream_on_event, s);
    (*env)->DeleteLocalRef(env, s);
    /* A listener that throws leaves a pending exception on this thread,
     * and every later JNI call on it then runs with that exception still
     * armed — undefined behaviour. Clear it: the event is lost, which is
     * far better than corrupting the engine thread. */
    if ((*env)->ExceptionCheck(env)) {
        /* JNI has no "describe the pending exception" call (that is JVMTI),
         * and the engine log already carries whatever the listener wrote.
         * Naming the event is the useful part. */
        __android_log_print(ANDROID_LOG_ERROR, "sofuu-js",
                            "stream listener threw; event dropped and exception cleared");
        (*env)->ExceptionClear(env);
    }
}

/* SofuuBridge.nativeSetStreamListener(listener | null) */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeSetStreamListener(JNIEnv *env, jclass cls, jobject listener) {
    (void)cls;
    if (g_stream_listener != NULL) {
        (*env)->DeleteGlobalRef(env, g_stream_listener);
        g_stream_listener = NULL;
    }
    g_stream_on_event = NULL;
    if (listener == NULL) return;
    jclass cls_l = (*env)->GetObjectClass(env, listener);
    if (cls_l == NULL) return;
    g_stream_on_event = (*env)->GetMethodID(env, cls_l, "onStreamEvent", "(Ljava/lang/String;)V");
    (*env)->DeleteLocalRef(env, cls_l);
    if (g_stream_on_event == NULL) return;
    g_stream_listener = (*env)->NewGlobalRef(env, listener);
}

/* SofuuBridge.nativeRtCallStream(rt, method, args) → long (cancel id; 0 on failure) */
JNIEXPORT jlong JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtCallStream(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                     jstring method, jstring args) {
    (void)cls;
    if (rt_ptr == 0 || method == NULL) return 0;
    if (g_stream_listener == NULL || g_stream_on_event == NULL) return 0;

    const char *method_c = (*env)->GetStringUTFChars(env, method, NULL);
    const char *args_c = NULL;
    if (args != NULL) {
        args_c = (*env)->GetStringUTFChars(env, args, NULL);
    }
    if (method_c == NULL) return 0;

    uint64_t cancel_id = 0;
    int rc = sofuu_rt_call_stream((SofuuRuntime *)(intptr_t)rt_ptr,
                                 method_c, args_c,
                                 on_stream_event, NULL, &cancel_id);

    (*env)->ReleaseStringUTFChars(env, method, method_c);
    if (args != NULL && args_c != NULL) {
        (*env)->ReleaseStringUTFChars(env, args, args_c);
    }
    if (rc != SOFUU_OK) return 0;
    return (jlong)cancel_id;
}

/* SofuuBridge.nativeRtCancel(rt, cancelId) → boolean */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRtCancel(JNIEnv *env, jclass cls, jlong rt_ptr, jlong cancel_id) {
    (void)env; (void)cls;
    if (rt_ptr == 0) return JNI_FALSE;
    return sofuu_rt_cancel((SofuuRuntime *)(intptr_t)rt_ptr, (uint64_t)cancel_id) == SOFUU_OK
        ? JNI_TRUE : JNI_FALSE;
}

/* ── Host log callback (sofuu_embed_set_log_cb) ───────────────────
 * Without this the engine's console.log/info/warn/error had nowhere to
 * go on Android (the desktop routes them to stderr via
 * embed_config::set_log_callback), so every JS-side warning vanished.
 * The callback fires on the owning engine thread with C strings valid
 * only for the call, so we print them and are done — no JNI round-trip
 * and no lifetime questions. */
/* Console lines ALSO append to an app-named file when
 * nativeSetEngineLogFile was called: on several OEM phones (vivo at
 * least) logcat never receives app output, and the console is the only
 * window into the engine's internals. Append-mode per line — the path
 * is set before the runtime boots, writers are single (engine thread),
 * and fopen/fclose per line is cheap at chat rates. */
static char g_engine_log_path[512] = {0};

static void on_engine_log(const char *level, const char *message, void *opaque) {
    (void)opaque;
    int prio = ANDROID_LOG_INFO;
    if (level != NULL) {
        if (strcmp(level, "error") == 0)       prio = ANDROID_LOG_ERROR;
        else if (strcmp(level, "warn") == 0)   prio = ANDROID_LOG_WARN;
        else if (strcmp(level, "debug") == 0)  prio = ANDROID_LOG_DEBUG;
        else if (strcmp(level, "trace") == 0)  prio = ANDROID_LOG_VERBOSE;
    }
    const char *msg = message ? message : "";
    __android_log_print(prio, "sofuu-js", "%s", msg);
    if (g_engine_log_path[0] != 0) {
        FILE *f = fopen(g_engine_log_path, "a");
        if (f) {
            fprintf(f, "%s: %s\n", level ? level : "info", msg);
            fclose(f);
        }
    }
}

/* SofuuBridge.nativeSetLogEnabled(boolean) */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeSetLogEnabled(JNIEnv *env, jclass cls, jboolean enabled) {
    (void)env; (void)cls;
    sofuu_embed_set_log_cb(enabled == JNI_TRUE ? on_engine_log : NULL, NULL);
}

/* SofuuBridge.nativeSetEngineLogFile(path): route console lines to a
 * file as well as logcat. Call BEFORE the runtime boots. */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeSetEngineLogFile(JNIEnv *env, jclass cls, jstring path) {
    (void)cls;
    if (path == NULL) { g_engine_log_path[0] = 0; return; }
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    if (p) {
        snprintf(g_engine_log_path, sizeof g_engine_log_path, "%s", p);
        (*env)->ReleaseStringUTFChars(env, path, p);
    }
}

/* ── Vector embeddings (H-E1 / M1) ─────────────────────────────────
 * The ABI hands out raw float vectors so a host never JSON-encodes them.
 * Every call here is a single ABI call: we ask for the vector, read the
 * dim it reports, then copy into a jfloatArray — the array length IS the
 * dim, so Kotlin never needs a second round trip to learn it.
 *
 * These four take `rt` as NULL on purpose: the embedders are stateless
 * pure functions (sofuu_embed.h), so they need no runtime and no thread
 * affinity. The default space is hash-768 — the SAME space the brain
 * stores in, so embed -> remember needs no space argument.
 */

JNIEXPORT jfloatArray JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeEmbedLocal(JNIEnv *env, jclass cls,
                                                    jstring text, jstring space) {
    (void)cls;
    if (text == NULL) return NULL;
    const char *text_c = (*env)->GetStringUTFChars(env, text, NULL);
    if (text_c == NULL) return NULL;
    const char *space_c = NULL;
    if (space != NULL) space_c = (*env)->GetStringUTFChars(env, space, NULL);

    float *out = NULL;
    size_t dim = 0;
    int rc = sofuu_embed_local(NULL, text_c, space_c, &out, &dim);

    (*env)->ReleaseStringUTFChars(env, text, text_c);
    if (space != NULL && space_c != NULL) (*env)->ReleaseStringUTFChars(env, space, space_c);

    if (rc != SOFUU_OK || out == NULL || dim == 0) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jfloatArray arr = (*env)->NewFloatArray(env, (jsize)dim);
    if (arr == NULL) { sofuu_free(out); return NULL; }
    (*env)->SetFloatArrayRegion(env, arr, 0, (jsize)dim, (const jfloat *)out);
    sofuu_free(out);
    return arr;
}

JNIEXPORT jobjectArray JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeEmbedBatch(JNIEnv *env, jclass cls,
                                                    jobjectArray texts, jstring space) {
    (void)cls;
    if (texts == NULL) return NULL;
    jsize n = (*env)->GetArrayLength(env, texts);
    if (n <= 0) return (*env)->NewObjectArray(env, 0, NULL, NULL);

    const char **utf = (const char **)calloc((size_t)n, sizeof(char *));
    jobject *held = (jobject *)calloc((size_t)n, sizeof(jobject));
    if (utf == NULL || held == NULL) { free((void *)utf); free(held); return NULL; }
    for (jsize i = 0; i < n; i++) {
        jstring s = (jstring)(*env)->GetObjectArrayElement(env, texts, i);
        held[i] = s;
        utf[i] = s ? (*env)->GetStringUTFChars(env, s, NULL) : NULL;
    }
    const char *space_c = space ? (*env)->GetStringUTFChars(env, space, NULL) : NULL;

    float *out = NULL;
    size_t got = 0, dim = 0;
    int rc = sofuu_embed_batch(NULL, utf, (size_t)n, space_c, &out, &got, &dim);

    for (jsize i = 0; i < n; i++) {
        if (utf[i] != NULL && held[i] != NULL) (*env)->ReleaseStringUTFChars(env, (jstring)held[i], utf[i]);
        if (held[i] != NULL) (*env)->DeleteLocalRef(env, held[i]);
    }
    free((void *)utf);
    free(held);
    if (space != NULL && space_c != NULL) (*env)->ReleaseStringUTFChars(env, space, space_c);

    if (rc != SOFUU_OK || out == NULL || dim == 0) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jclass farr = (*env)->FindClass(env, "[F");
    jobjectArray arr = (*env)->NewObjectArray(env, (jsize)got, farr, NULL);
    if (arr != NULL) {
        for (size_t i = 0; i < got; i++) {
            jfloatArray one = (*env)->NewFloatArray(env, (jsize)dim);
            if (one == NULL) continue;
            (*env)->SetFloatArrayRegion(env, one, 0, (jsize)dim, (const jfloat *)(out + i * dim));
            (*env)->SetObjectArrayElement(env, arr, (jsize)i, one);
            (*env)->DeleteLocalRef(env, one);
        }
    }
    sofuu_free(out);
    return arr;
}

JNIEXPORT jfloatArray JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeEmbedImage(JNIEnv *env, jclass cls, jbyteArray bytes) {
    (void)cls;
    if (bytes == NULL) return NULL;
    jsize len = (*env)->GetArrayLength(env, bytes);
    if (len <= 0) return NULL;
    jbyte *raw = (*env)->GetByteArrayElements(env, bytes, NULL);
    if (raw == NULL) return NULL;

    float *out = NULL;
    size_t dim = 0;
    /* Undecodable input reports SOFUU_ERR_MODEL_UNAVAIL — surfaced to
     * Kotlin as null rather than a fake zero vector. */
    int rc = sofuu_embed_image(NULL, (const uint8_t *)raw, (size_t)len, &out, &dim);
    (*env)->ReleaseByteArrayElements(env, bytes, raw, JNI_ABORT);

    if (rc != SOFUU_OK || out == NULL || dim == 0) {
        if (out) sofuu_free(out);
        return NULL;
    }
    jfloatArray arr = (*env)->NewFloatArray(env, (jsize)dim);
    if (arr != NULL) (*env)->SetFloatArrayRegion(env, arr, 0, (jsize)dim, (const jfloat *)out);
    sofuu_free(out);
    return arr;
}

/* SofuuBridge.nativeEmbedInfo() → String? (space manifest JSON).
 * Static: the ABI allows a NULL rt. */
JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeEmbedInfo(JNIEnv *env, jclass cls) {
    (void)cls;
    char *out = NULL;
    int rc = sofuu_embed_info(NULL, &out);
    if (out == NULL) return NULL;
    jstring result = (rc == SOFUU_OK) ? (*env)->NewStringUTF(env, out) : NULL;
    sofuu_free(out);
    return result;
}

/* ── Provider voice (M2C) ──────────────────────────────────────────
 * Thin funnels to ai.transcribe / ai.speak — the same implementation the
 * JS surface uses, so there is no duplicated HTTP here. The host
 * base64-encodes the audio; the library validates the alphabet before it
 * reaches JS, so no string escape is possible. Both return the standard
 * envelope. These DO need the runtime (they go through the funnel), so
 * they must run on the engine thread.
 */

JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeVoiceTranscribe(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                         jstring audio_b64, jstring opts) {
    (void)cls;
    if (rt_ptr == 0 || audio_b64 == NULL) return NULL;
    const char *b64 = (*env)->GetStringUTFChars(env, audio_b64, NULL);
    if (b64 == NULL) return NULL;
    const char *opts_c = NULL;
    if (opts != NULL) opts_c = (*env)->GetStringUTFChars(env, opts, NULL);

    char *out = NULL;
    int rc = sofuu_voice_transcribe((SofuuRuntime *)(intptr_t)rt_ptr, b64, opts_c, &out);

    (*env)->ReleaseStringUTFChars(env, audio_b64, b64);
    if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);

    if (out == NULL) return NULL;
    /* The envelope is returned even on failure — the caller reads ok/error. */
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    (void)rc;
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeVoiceSpeak(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                    jstring text, jstring opts) {
    (void)cls;
    if (rt_ptr == 0 || text == NULL) return NULL;
    const char *t = (*env)->GetStringUTFChars(env, text, NULL);
    if (t == NULL) return NULL;
    const char *opts_c = NULL;
    if (opts != NULL) opts_c = (*env)->GetStringUTFChars(env, opts, NULL);

    char *out = NULL;
    int rc = sofuu_voice_speak((SofuuRuntime *)(intptr_t)rt_ptr, t, opts_c, &out);

    (*env)->ReleaseStringUTFChars(env, text, t);
    if (opts != NULL && opts_c != NULL) (*env)->ReleaseStringUTFChars(env, opts, opts_c);

    if (out == NULL) return NULL;
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    (void)rc;
    return result;
}

/* ── Host tools (PLAN-EMBED-SDK E2) ─────────────────────────────────
 *
 * sofuu_rt_register_host_tool lets the HOST hand the agent a native
 * capability with no JS in the loop. It is the one part of the ABI that
 * exists specifically for a mobile embedder: camera, location, the share
 * sheet, biometrics — things JS in a QuickJS sandbox cannot reach.
 *
 * Two things shape this shim:
 *
 * 1. The C callback signature carries no tool name:
 *      const char *(*fn)(const char *args_json, uint64_t call_id, void *opaque)
 *    so the name, the Kotlin receiver and its method id all ride in
 *    `opaque`, one malloc'd slot per registration.
 *
 * 2. The returned pointer is consumed synchronously by the caller
 *    (capi/src/lib.rs: js_host_tool_call copies it before returning), but
 *    a Java string has no address we can hand back. So the result is
 *    copied into a per-thread buffer that stays valid until that thread's
 *    next host-tool call. Returning NULL means "pending" — the host owns
 *    the call and settles it with nativeHostToolResolve, from any thread.
 */

typedef struct SofuuHostToolSlot {
    char *name;
    jobject listener;   /* global ref to the Kotlin HostTool */
    jmethodID method;   /* invoke(String, long) -> String? */
    struct SofuuHostToolSlot *next;
} SofuuHostToolSlot;

/* Every slot ever allocated for this process. The ABI keeps the C side of
 * a registration forever and gives no "runtime destroyed" callback, so the
 * JNI global ref would otherwise pin its Kotlin receiver — and through it
 * the Activity's Context — for the life of the app. One list, torn down by
 * nativeFreeHostTools() once the runtime that owns them is gone. */
static SofuuHostToolSlot *g_host_tools = NULL;

/* One slot per host-tool call thread; see note 2 above. */
static __thread char *g_host_result = NULL;
static __thread size_t g_host_result_cap = 0;

static const char *sofuu_host_tool_trampoline(const char *args_json,
                                              uint64_t call_id,
                                              void *opaque) {
    SofuuHostToolSlot *slot = (SofuuHostToolSlot *)opaque;
    JNIEnv *env = NULL;

    if (slot == NULL || g_jvm == NULL) return NULL;
    /* The call arrives on the engine thread — a Java thread that is
     * already attached, so GetEnv is the right call here (same reasoning
     * as on_stream_event). */
    if ((*g_jvm)->GetEnv(g_jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK || env == NULL) {
        return NULL;
    }

    jstring jargs = (*env)->NewStringUTF(env, args_json ? args_json : "{}");
    if (jargs == NULL) return NULL;

    jobject jres = (*env)->CallObjectMethod(env, slot->listener, slot->method,
                                            jargs, (jlong)call_id);
    (*env)->DeleteLocalRef(env, jargs);
    if ((*env)->ExceptionCheck(env)) {
        /* A throwing tool must not poison the engine thread. Treat it as
         * "pending" is wrong too — the call would hang forever — so the
         * exception is swallowed and the agent sees an error via the
         * resolve path's own timeout. */
        (*env)->ExceptionDescribe(env);
        (*env)->ExceptionClear(env);
        return NULL;
    }
    if (jres == NULL) return NULL; /* deferred to the host */

    const char *utf = (*env)->GetStringUTFChars(env, (jstring)jres, NULL);
    if (utf == NULL) {
        (*env)->DeleteLocalRef(env, jres);
        return NULL;
    }
    size_t n = strlen(utf);
    if (n + 1 > g_host_result_cap) {
        char *grown = (char *)realloc(g_host_result, n + 1);
        if (grown == NULL) {
            (*env)->ReleaseStringUTFChars(env, (jstring)jres, utf);
            (*env)->DeleteLocalRef(env, jres);
            return NULL;
        }
        g_host_result = grown;
        g_host_result_cap = n + 1;
    }
    memcpy(g_host_result, utf, n + 1);
    (*env)->ReleaseStringUTFChars(env, (jstring)jres, utf);
    (*env)->DeleteLocalRef(env, jres);
    return g_host_result;
}

/* SofuuBridge.nativeRegisterHostTool(rt, name, description, paramsJson, listener) */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeRegisterHostTool(JNIEnv *env, jclass cls, jlong rt_ptr,
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

    const char *name_c = (*env)->GetStringUTFChars(env, name, NULL);
    if (name_c == NULL) return JNI_FALSE;

    SofuuHostToolSlot *slot = (SofuuHostToolSlot *)calloc(1, sizeof(SofuuHostToolSlot));
    if (slot == NULL) {
        (*env)->ReleaseStringUTFChars(env, name, name_c);
        return JNI_FALSE;
    }
    slot->name = strdup(name_c);
    slot->listener = (*env)->NewGlobalRef(env, listener);
    slot->method = mid;
    slot->next = g_host_tools;
    g_host_tools = slot;

    const char *desc_c = NULL, *params_c = NULL;
    if (description != NULL) desc_c = (*env)->GetStringUTFChars(env, description, NULL);
    if (params_json != NULL) params_c = (*env)->GetStringUTFChars(env, params_json, NULL);

    int rc = sofuu_rt_register_host_tool((SofuuRuntime *)(intptr_t)rt_ptr,
                                         slot->name, desc_c, params_c,
                                         sofuu_host_tool_trampoline, (void *)slot);

    if (params_c) (*env)->ReleaseStringUTFChars(env, params_json, params_c);
    if (desc_c) (*env)->ReleaseStringUTFChars(env, description, desc_c);
    (*env)->ReleaseStringUTFChars(env, name, name_c);

    if (rc != SOFUU_OK) {
        /* The runtime kept nothing, so this slot is ours to free. */
        if (slot->listener) (*env)->DeleteGlobalRef(env, slot->listener);
        free(slot->name);
        free(slot);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

/* SofuuBridge.nativeHostToolResolve(rt, callId, resultJson | null, error | null) */
JNIEXPORT jboolean JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeHostToolResolve(JNIEnv *env, jclass cls, jlong rt_ptr,
                                                        jlong call_id, jstring result_json,
                                                        jstring error) {
    (void)cls;
    if (rt_ptr == 0 || call_id == 0) return JNI_FALSE;

    const char *res_c = NULL, *err_c = NULL;
    if (result_json != NULL) res_c = (*env)->GetStringUTFChars(env, result_json, NULL);
    if (error != NULL) err_c = (*env)->GetStringUTFChars(env, error, NULL);

    int rc = sofuu_rt_host_tool_resolve((SofuuRuntime *)(intptr_t)rt_ptr,
                                        (uint64_t)call_id, res_c, err_c);

    if (res_c) (*env)->ReleaseStringUTFChars(env, result_json, res_c);
    if (err_c) (*env)->ReleaseStringUTFChars(env, error, err_c);
    return rc == SOFUU_OK ? JNI_TRUE : JNI_FALSE;
}

/* ── Process environment ──────────────────────────────────────────
 *
 * The engine reads its knobs with plain std::env::var (SOFUU_NO_TOOLS,
 * SOFUU_MEMORY_BACKEND, SOFUU_WEB_ENGINE, …). Kotlin cannot reach libc's
 * setenv, and there is deliberately no ABI call for them — they are
 * process configuration, not runtime state — so the host sets them here
 * before sofuu_rt_new.
 */

/* SofuuBridge.nativeSetEnv(name, value) — a null value unsets the key. */
JNIEXPORT void JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeSetEnv(JNIEnv *env, jclass cls, jstring name, jstring value) {
    (void)cls;
    if (name == NULL) return;
    const char *n = (*env)->GetStringUTFChars(env, name, NULL);
    if (n == NULL) return;
    if (value == NULL) {
        unsetenv(n);
    } else {
        const char *v = (*env)->GetStringUTFChars(env, value, NULL);
        if (v != NULL) {
            setenv(n, v, 1);
            (*env)->ReleaseStringUTFChars(env, value, v);
        }
    }
    (*env)->ReleaseStringUTFChars(env, name, n);
}

/* SofuuBridge.nativeFreeHostTools() — release every host-tool registration.
 *
 * Call this AFTER the runtime that used them has been freed. The JNI global
 * refs have to go through a JNIEnv; by this point the engine thread is
 * gone, so attach for the duration rather than assume. Returns how many
 * were released. */
JNIEXPORT jint JNICALL
Java_com_sofuu_engine_SofuuBridge_nativeFreeHostTools(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    int n = 0;
    JNIEnv *e = env;
    int attached = 0;
    if (g_jvm != NULL &&
        (*g_jvm)->GetEnv(g_jvm, (void **)&e, JNI_VERSION_1_6) != JNI_OK) {
        if ((*g_jvm)->AttachCurrentThread(g_jvm, &e, NULL) != JNI_OK) { e = NULL; }
        else { attached = 1; }
    }
    SofuuHostToolSlot *s = g_host_tools;
    g_host_tools = NULL;
    while (s != NULL) {
        SofuuHostToolSlot *next = s->next;
        if (e != NULL && s->listener != NULL) (*e)->DeleteGlobalRef(e, s->listener);
        free(s->name);
        free(s);
        s = next;
        n++;
    }
    if (g_host_result != NULL) { free(g_host_result); g_host_result = NULL; g_host_result_cap = 0; }
    if (attached) (*g_jvm)->DetachCurrentThread(g_jvm);
    return n;
}

/* ── Embedded GPU UI (PLAN-EMBEDDED-NATIVE-HOST-C-ABI) ──────────────
 *
 * Thin JNI funnels to the additive `sofuu_ui_*` C surface. The renderer
 * itself lives in `ui_gles.c` (host-owned GL context, zero-copy DrawList),
 * so `libsofuu` keeps zero GPU/EGL linkage. Session creation and input are
 * thread-safe; attach/detach must run on the runtime's owning thread.
 */

typedef struct SofuuUiWakeBinding {
    jlong session;
    jobject view;          /* global ref to the GLSurfaceView */
    jmethodID wake_method; /* requestRender()V or postInvalidate()V */
    struct SofuuUiWakeBinding *next;
} SofuuUiWakeBinding;

static SofuuUiWakeBinding *g_ui_wakes = NULL;
static pthread_mutex_t g_ui_wake_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Runs on whatever thread pushed a patch / dispatched input. The session's
 * wake slot has already released its locks, so re-entering JNI here is safe. */
static void ui_wake_trampoline(void *opaque) {
    jlong session = (jlong)(intptr_t)opaque;
    JNIEnv *env = NULL;
    int attached = 0;
    if (g_jvm == NULL) return;

    /* Hold the lock across the call: detach deletes the global ref, and a
     * borrowed pointer surviving that DeleteGlobalRef would be a UAF. The
     * wake method only flags the GL thread, so it never calls back in. */
    pthread_mutex_lock(&g_ui_wake_mutex);
    SofuuUiWakeBinding *b = g_ui_wakes;
    while (b != NULL && b->session != session) b = b->next;
    if (b == NULL || b->view == NULL || b->wake_method == NULL) {
        pthread_mutex_unlock(&g_ui_wake_mutex);
        return;
    }
    if ((*g_jvm)->GetEnv(g_jvm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        if ((*g_jvm)->AttachCurrentThread(g_jvm, &env, NULL) != JNI_OK) {
            pthread_mutex_unlock(&g_ui_wake_mutex);
            return;
        }
        attached = 1;
    }
    (*env)->CallVoidMethod(env, b->view, b->wake_method);
    if ((*env)->ExceptionCheck(env)) {
        (*env)->ExceptionClear(env);
    }
    if (attached) (*g_jvm)->DetachCurrentThread(g_jvm);
    pthread_mutex_unlock(&g_ui_wake_mutex);
}

/* Drop the wake binding for one session. Returns the previous view ref so
 * the caller can release it outside the lock (JNI calls may block). */
static jobject ui_wake_unbind(JNIEnv *env, jlong session) {
    jobject old = NULL;
    pthread_mutex_lock(&g_ui_wake_mutex);
    SofuuUiWakeBinding **link = &g_ui_wakes;
    while (*link != NULL && (*link)->session != session) link = &(*link)->next;
    if (*link != NULL) {
        SofuuUiWakeBinding *dead = *link;
        *link = dead->next;
        old = dead->view;
        free(dead);
    }
    pthread_mutex_unlock(&g_ui_wake_mutex);
    if (old != NULL && env != NULL) (*env)->DeleteGlobalRef(env, old);
    return old;
}

/* SofuuUiNative.nativeSessionNew(configJson) → session pointer (0 on error) */
JNIEXPORT jlong JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeSessionNew(JNIEnv *env, jobject thiz,
                                                     jstring config_json) {
    (void)thiz;
    const char *config_c = NULL;
    if (config_json != NULL) {
        config_c = (*env)->GetStringUTFChars(env, config_json, NULL);
        if (config_c == NULL) return 0;
    }
    char *err = NULL;
    SofuuUiSession *ui = sofuu_ui_session_new(config_c, &err);
    if (config_json != NULL && config_c != NULL) {
        (*env)->ReleaseStringUTFChars(env, config_json, config_c);
    }
    if (err != NULL) {
        __android_log_print(ANDROID_LOG_ERROR, "sofuu-ui", "session_new: %s", err);
        sofuu_free(err);
    }
    return (jlong)(intptr_t)ui;
}

/* SofuuUiNative.nativeSessionFree(session) */
JNIEXPORT void JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeSessionFree(JNIEnv *env, jobject thiz,
                                                      jlong session) {
    (void)thiz;
    if (session == 0) return;
    ui_wake_unbind(env, session);
    sofuu_ui_session_set_wake_cb((SofuuUiSession *)(intptr_t)session, NULL, NULL);
    sofuu_ui_session_free((SofuuUiSession *)(intptr_t)session);
}

/* SofuuUiNative.nativeSetWakeView(session, view | null) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeSetWakeView(JNIEnv *env, jobject thiz,
                                                      jlong session, jobject view) {
    (void)thiz;
    if (session == 0) return SOFUU_ERR_INVALID_ARG;
    if (view == NULL) {
        ui_wake_unbind(env, session);
        return sofuu_ui_session_set_wake_cb((SofuuUiSession *)(intptr_t)session,
                                            NULL, NULL);
    }

    jclass cls = (*env)->GetObjectClass(env, view);
    if (cls == NULL) return SOFUU_ERR_INVALID_ARG;
    jmethodID mid = (*env)->GetMethodID(env, cls, "requestRender", "()V");
    if (mid == NULL) {
        /* Fall back for a plain View / a test double. postInvalidate is the
         * one repaint request that is legal from any thread. */
        (*env)->ExceptionClear(env);
        mid = (*env)->GetMethodID(env, cls, "postInvalidate", "()V");
    }
    (*env)->DeleteLocalRef(env, cls);
    if (mid == NULL) {
        (*env)->ExceptionClear(env);
        return SOFUU_ERR_INVALID_ARG;
    }
    jobject global = (*env)->NewGlobalRef(env, view);
    if (global == NULL) return SOFUU_ERR_PANIC;

    pthread_mutex_lock(&g_ui_wake_mutex);
    SofuuUiWakeBinding *b = g_ui_wakes;
    while (b != NULL && b->session != session) b = b->next;
    jobject old = NULL;
    if (b == NULL) {
        b = (SofuuUiWakeBinding *)calloc(1, sizeof(SofuuUiWakeBinding));
        if (b == NULL) {
            pthread_mutex_unlock(&g_ui_wake_mutex);
            (*env)->DeleteGlobalRef(env, global);
            return SOFUU_ERR_PANIC;
        }
        b->session = session;
        b->next = g_ui_wakes;
        g_ui_wakes = b;
    } else {
        old = b->view;
    }
    b->view = global;
    b->wake_method = mid;
    pthread_mutex_unlock(&g_ui_wake_mutex);
    if (old != NULL) (*env)->DeleteGlobalRef(env, old);

    int rc = sofuu_ui_session_set_wake_cb((SofuuUiSession *)(intptr_t)session,
                                          ui_wake_trampoline,
                                          (void *)(intptr_t)session);
    if (rc != SOFUU_OK) ui_wake_unbind(env, session);
    return rc;
}

/* SofuuUiNative.nativeAttachRuntime(rt, session) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeAttachRuntime(JNIEnv *env, jobject thiz,
                                                        jlong rt_ptr, jlong session) {
    (void)env; (void)thiz;
    if (rt_ptr == 0 || session == 0) return SOFUU_ERR_INVALID_ARG;
    return sofuu_ui_attach_runtime((SofuuRuntime *)(intptr_t)rt_ptr,
                                   (SofuuUiSession *)(intptr_t)session);
}

/* SofuuUiNative.nativeDetachRuntime(rt) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeDetachRuntime(JNIEnv *env, jobject thiz,
                                                        jlong rt_ptr) {
    (void)env; (void)thiz;
    if (rt_ptr == 0) return SOFUU_ERR_INVALID_ARG;
    return sofuu_ui_detach_runtime((SofuuRuntime *)(intptr_t)rt_ptr);
}

/* SofuuUiNative.nativeSetSafeInsets(session, topPx, bottomPx) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeSetSafeInsets(JNIEnv *env, jobject thiz,
                                                        jlong session, jfloat top,
                                                        jfloat bottom) {
    (void)env; (void)thiz;
    if (session == 0) return SOFUU_ERR_INVALID_ARG;
    return sofuu_ui_session_set_safe_insets((SofuuUiSession *)(intptr_t)session,
                                            top, bottom);
}

/* SofuuUiNative.nativeGetAtlas(session, outMeta) → ByteArray?
 * outMeta = [width, height, Float.floatToRawIntBits(px_range)]. */
JNIEXPORT jbyteArray JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeGetAtlas(JNIEnv *env, jobject thiz,
                                                   jlong session, jintArray out_meta) {
    (void)thiz;
    if (session == 0) return NULL;
    const uint8_t *pixels = NULL;
    uint32_t width = 0, height = 0;
    float px_range = 0.0f;
    int rc = sofuu_ui_session_get_atlas((SofuuUiSession *)(intptr_t)session,
                                        &pixels, &width, &height, &px_range);
    if (rc != SOFUU_OK || pixels == NULL || width == 0 || height == 0) return NULL;
    if (out_meta != NULL) {
        union { float f; int32_t i; } bits;
        bits.f = px_range;
        jint meta[3] = {(jint)width, (jint)height, (jint)bits.i};
        (*env)->SetIntArrayRegion(env, out_meta, 0, 3, meta);
    }
    jsize len = (jsize)((size_t)width * (size_t)height);
    jbyteArray arr = (*env)->NewByteArray(env, len);
    if (arr == NULL) return NULL;
    (*env)->SetByteArrayRegion(env, arr, 0, len, (const jbyte *)pixels);
    return arr;
}

/* SofuuUiNative.nativePushPatchJson(session, patch) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativePushPatchJson(JNIEnv *env, jobject thiz,
                                                        jlong session, jstring patch) {
    (void)thiz;
    if (session == 0 || patch == NULL) return SOFUU_ERR_INVALID_ARG;
    const char *json = (*env)->GetStringUTFChars(env, patch, NULL);
    if (json == NULL) return SOFUU_ERR_PANIC;
    int rc = sofuu_ui_push_patch_json((SofuuUiSession *)(intptr_t)session, json, 0);
    (*env)->ReleaseStringUTFChars(env, patch, json);
    return rc;
}

/* SofuuUiNative.nativeIngestAgentEventJson(session, event) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeIngestAgentEventJson(JNIEnv *env, jobject thiz,
                                                               jlong session, jstring event) {
    (void)thiz;
    if (session == 0 || event == NULL) return SOFUU_ERR_INVALID_ARG;
    const char *json = (*env)->GetStringUTFChars(env, event, NULL);
    if (json == NULL) return SOFUU_ERR_PANIC;
    int rc = sofuu_ui_ingest_agent_event_json((SofuuUiSession *)(intptr_t)session, json);
    (*env)->ReleaseStringUTFChars(env, event, json);
    return rc;
}

/* SofuuUiNative.nativeDispatchPointer(...) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeDispatchPointer(JNIEnv *env, jobject thiz,
                                                          jlong session, jint kind,
                                                          jfloat x, jfloat y,
                                                          jfloat dx, jfloat dy,
                                                          jfloat scale, jint button,
                                                          jint modifiers) {
    (void)env; (void)thiz;
    if (session == 0) return SOFUU_ERR_INVALID_ARG;
    SofuuUiInputEvent e;
    memset(&e, 0, sizeof e);
    e.kind = (uint32_t)kind;
    e.x = x;
    e.y = y;
    e.dx = dx;
    e.dy = dy;
    e.scale_factor = scale;
    e.button = (uint32_t)button;
    e.modifiers = (uint32_t)modifiers;
    return sofuu_ui_dispatch_input((SofuuUiSession *)(intptr_t)session, &e);
}

/* SofuuUiNative.nativeDispatchText(session, text) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeDispatchText(JNIEnv *env, jobject thiz,
                                                       jlong session, jstring text) {
    (void)thiz;
    if (session == 0 || text == NULL) return SOFUU_ERR_INVALID_ARG;
    const char *utf = (*env)->GetStringUTFChars(env, text, NULL);
    if (utf == NULL) return SOFUU_ERR_PANIC;
    SofuuUiInputEvent e;
    memset(&e, 0, sizeof e);
    e.kind = SOFUU_INPUT_TEXT_INPUT;
    e.text_utf8 = utf;
    int rc = sofuu_ui_dispatch_input((SofuuUiSession *)(intptr_t)session, &e);
    (*env)->ReleaseStringUTFChars(env, text, utf);
    return rc;
}

/* SofuuUiNative.nativeDispatchKey(session, key, modifiers) → int */
JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeDispatchKey(JNIEnv *env, jobject thiz,
                                                      jlong session, jstring key,
                                                      jint modifiers) {
    (void)thiz;
    if (session == 0 || key == NULL) return SOFUU_ERR_INVALID_ARG;
    const char *utf = (*env)->GetStringUTFChars(env, key, NULL);
    if (utf == NULL) return SOFUU_ERR_PANIC;
    SofuuUiInputEvent e;
    memset(&e, 0, sizeof e);
    e.kind = SOFUU_INPUT_KEY_DOWN;
    e.text_utf8 = utf;
    e.modifiers = (uint32_t)modifiers;
    int rc = sofuu_ui_dispatch_input((SofuuUiSession *)(intptr_t)session, &e);
    (*env)->ReleaseStringUTFChars(env, key, utf);
    return rc;
}

/* SofuuUiNative.nativePollEventJson(session) → String?
 * Competing with the JS side for the same queue is intentional: call this
 * only from a native-orchestrated host, never alongside sofuu.ui's JS
 * Auto-Composer loop (it polls __sofuu_ui_poll_event itself). */
JNIEXPORT jstring JNICALL
Java_com_sofuu_render_SofuuUiNative_nativePollEventJson(JNIEnv *env, jobject thiz,
                                                        jlong session) {
    (void)thiz;
    if (session == 0) return NULL;
    char *out = NULL;
    int rc = sofuu_ui_poll_event_json((SofuuUiSession *)(intptr_t)session, &out);
    if (rc != 1 || out == NULL) {
        if (out != NULL) sofuu_free(out);
        return NULL;
    }
    jstring result = (*env)->NewStringUTF(env, out);
    sofuu_free(out);
    return result;
}
