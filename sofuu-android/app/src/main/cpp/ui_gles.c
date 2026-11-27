/*
 * ui_gles.c — embedded Sofuu canvas on Android (Mode A: GpuUi draw-list pull).
 *
 * This is the Android twin of `GpuUiMetalView.swift`: the HOST owns the GPU
 * context (a GLSurfaceView/EGL context), compiles the shared ES 3.0 UI
 * shaders once, polls `sofuu_ui_poll_draw_list` on the GL thread every dirty
 * frame, uploads the borrowed instance slices and issues four instanced
 * passes in `cmds[]` order. Keeping the renderer here means `libsofuu` stays
 * completely free of GPU/EGL linkage (Phase C5 headless guarantee).
 *
 * The shader source is the canonical
 *   crates/sofuu-gpu/src/shaders/glsl/ui_shaders_es3.glsl
 * compiled four times with SOFUU_PIPELINE_{QUAD,RIBBON,STROKE,GLYPH} and
 * SOFUU_VERTEX / SOFUU_FRAGMENT, exactly like crates/sofuu-gpu/backend/gles.rs.
 * Gradle syncs it into the APK assets; Kotlin passes the text in once.
 *
 * Threading: every function here must run on the GLSurfaceView render thread
 * (the thread with the current EGL context) except `nativeGlesDestroy`, which
 * only releases host memory. `sofuu_ui_poll_draw_list` is called there too.
 */

#include <jni.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "sofuu_embed.h"

#define TAG "sofuu-ui-gl"
#define MAX_CLIPS 256

#define QUAD_STRIDE ((GLsizei)sizeof(SofuuUiQuadInstance))
#define RIBBON_STRIDE ((GLsizei)sizeof(SofuuUiRibbonInstance))
#define STROKE_STRIDE ((GLsizei)sizeof(SofuuUiStrokeInstance))
#define GLYPH_STRIDE ((GLsizei)sizeof(SofuuUiGlyphInstance))

typedef struct {
    GLuint program[4];
    GLuint vao[4];
    GLuint inst_vbo[4];
    GLsizeiptr inst_cap[4];
    GLuint scratch_vbo[4];
    GLsizeiptr scratch_cap[4];
    unsigned char *scratch[4];
    size_t scratch_bytes[4];
    GLuint corner_vbo;
    GLuint clip_ubo;
    GLuint atlas_tex;
    GLint loc_viewport[4];
    GLint loc_atlas[4];
    GLint loc_scale[4];
    GLint loc_time[4];
    int ready;
    int logged_failure;
} UiGles;

static UiGles *g_gles = NULL;

static const char *const kPipelineNames[4] = {"QUAD", "RIBBON", "STROKE", "GLYPH"};

static GLsizei stride_of(int kind) {
    switch (kind) {
        case 0: return QUAD_STRIDE;
        case 1: return RIBBON_STRIDE;
        case 2: return STROKE_STRIDE;
        default: return GLYPH_STRIDE;
    }
}

static size_t clip_offset_of(int kind) {
    switch (kind) {
        case 0: return offsetof(SofuuUiQuadInstance, clip_id);
        case 1: return offsetof(SofuuUiRibbonInstance, clip_id);
        case 2: return offsetof(SofuuUiStrokeInstance, clip_id);
        default: return offsetof(SofuuUiGlyphInstance, clip_id);
    }
}

/* ── Shader compilation ─────────────────────────────────────────── */

static char *stage_source(const char *src, const char *pipeline, int vertex) {
    static const char *marker = "// __SOFUU_PIPELINE_DEFINES__";
    const char *at = strstr(src, marker);
    if (at == NULL) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "shader is missing the %s placeholder", marker);
        return NULL;
    }
    char defs[128];
    int n = snprintf(defs, sizeof defs, "#define SOFUU_PIPELINE_%s\n#define SOFUU_%s\n",
                     pipeline, vertex ? "VERTEX" : "FRAGMENT");
    if (n <= 0 || (size_t)n >= sizeof defs) return NULL;

    size_t before = (size_t)(at - src);
    size_t after = strlen(at + strlen(marker));
    size_t total = before + (size_t)n + after + 1;
    char *out = (char *)malloc(total);
    if (out == NULL) return NULL;
    memcpy(out, src, before);
    memcpy(out + before, defs, (size_t)n);
    memcpy(out + before + (size_t)n, at + strlen(marker), after);
    out[total - 1] = '\0';
    return out;
}

static GLuint compile_stage(GLenum type, const char *source,
                            const char *pipeline, const char *stage) {
    GLuint shader = glCreateShader(type);
    if (shader == 0) return 0;
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei len = 0;
        log[0] = '\0';
        glGetShaderInfoLog(shader, (GLsizei)sizeof log, &len, log);
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%s %s compile failed: %.1900s",
                            pipeline, stage, log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint build_program(const char *source, const char *pipeline, int kind, UiGles *st) {
    char *vs_src = stage_source(source, pipeline, 1);
    char *fs_src = stage_source(source, pipeline, 0);
    if (vs_src == NULL || fs_src == NULL) {
        free(vs_src);
        free(fs_src);
        return 0;
    }

    GLuint vs = compile_stage(GL_VERTEX_SHADER, vs_src, pipeline, "vertex");
    GLuint fs = compile_stage(GL_FRAGMENT_SHADER, fs_src, pipeline, "fragment");
    free(vs_src);
    free(fs_src);
    if (vs == 0 || fs == 0) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program == 0) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        GLsizei len = 0;
        log[0] = '\0';
        glGetProgramInfoLog(program, (GLsizei)sizeof log, &len, log);
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%s link failed: %.1900s", pipeline, log);
        glDeleteProgram(program);
        return 0;
    }

    GLuint block = glGetUniformBlockIndex(program, "ClipsBlock");
    if (block != GL_INVALID_INDEX) {
        glUniformBlockBinding(program, block, 0);
    }
    st->loc_viewport[kind] = glGetUniformLocation(program, "u_viewport_size");
    st->loc_atlas[kind] = glGetUniformLocation(program, "u_atlas_size");
    st->loc_scale[kind] = glGetUniformLocation(program, "u_scale_factor");
    st->loc_time[kind] = glGetUniformLocation(program, "u_time_sec");

    glUseProgram(program);
    GLint atlas = glGetUniformLocation(program, "u_atlas");
    if (atlas >= 0) glUniform1i(atlas, 0);
    return program;
}

/* ── Attribute layout (mirrors sofuu-gpu/backend/gles.rs) ───────── */

static void configure_attributes(int kind, size_t offset) {
    GLsizei stride = stride_of(kind);
    const void *base = (const void *)offset;
#define P(o) ((const void *)(offset + (o)))
    switch (kind) {
        case 0:
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, P(0));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, P(16));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(3, 4, GL_UNSIGNED_INT, stride, P(32));
            glEnableVertexAttribArray(4);
            glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, P(48));
            glEnableVertexAttribArray(5);
            glVertexAttribIPointer(5, 1, GL_UNSIGNED_INT, stride, P(64));
            glVertexAttribDivisor(1, 1);
            glVertexAttribDivisor(2, 1);
            glVertexAttribDivisor(3, 1);
            glVertexAttribDivisor(4, 1);
            glVertexAttribDivisor(5, 1);
            break;
        case 1:
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, P(0));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, P(16));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(3, 4, GL_UNSIGNED_INT, stride, P(24));
            glVertexAttribDivisor(1, 1);
            glVertexAttribDivisor(2, 1);
            glVertexAttribDivisor(3, 1);
            break;
        case 2:
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, P(0));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, P(16));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(3, 4, GL_UNSIGNED_INT, stride, P(32));
            glVertexAttribDivisor(1, 1);
            glVertexAttribDivisor(2, 1);
            glVertexAttribDivisor(3, 1);
            break;
        default:
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, P(0));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, P(16));
            glEnableVertexAttribArray(3);
            glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, stride, P(32));
            glEnableVertexAttribArray(4);
            glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, P(36));
            glEnableVertexAttribArray(5);
            glVertexAttribIPointer(5, 1, GL_UNSIGNED_INT, stride, P(40));
            glVertexAttribDivisor(1, 1);
            glVertexAttribDivisor(2, 1);
            glVertexAttribDivisor(3, 1);
            glVertexAttribDivisor(4, 1);
            glVertexAttribDivisor(5, 1);
            break;
    }
#undef P
    (void)base;
}

/* ── Resource lifetime ──────────────────────────────────────────── */

/* Free the host struct. `delete_gl` is 1 only when the current thread owns
 * the live EGL context (a rebuild); after surface loss the object names are
 * already dead, so deleting them is both pointless and unsafe. */
static void ui_gles_free(UiGles *st, int delete_gl) {
    if (st == NULL) return;
    if (delete_gl) {
        for (int i = 0; i < 4; i++) {
            if (st->program[i]) glDeleteProgram(st->program[i]);
            if (st->vao[i]) glDeleteVertexArrays(1, &st->vao[i]);
            if (st->inst_vbo[i]) glDeleteBuffers(1, &st->inst_vbo[i]);
            if (st->scratch_vbo[i]) glDeleteBuffers(1, &st->scratch_vbo[i]);
        }
        if (st->corner_vbo) glDeleteBuffers(1, &st->corner_vbo);
        if (st->clip_ubo) glDeleteBuffers(1, &st->clip_ubo);
        if (st->atlas_tex) glDeleteTextures(1, &st->atlas_tex);
    }
    for (int i = 0; i < 4; i++) free(st->scratch[i]);
    free(st);
}

static int upload_streaming(UiGles *st, int kind, const void *data, GLsizeiptr bytes) {
    if (bytes <= 0 || data == NULL) return 0;
    glBindBuffer(GL_ARRAY_BUFFER, st->inst_vbo[kind]);
    if (bytes > st->inst_cap[kind]) {
        glBufferData(GL_ARRAY_BUFFER, bytes, data, GL_STREAM_DRAW);
        st->inst_cap[kind] = bytes;
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
    }
    return 1;
}

static int upload_scratch(UiGles *st, int kind, const void *data, GLsizeiptr bytes) {
    if (bytes <= 0 || data == NULL) return 0;
    glBindBuffer(GL_ARRAY_BUFFER, st->scratch_vbo[kind]);
    if (bytes > st->scratch_cap[kind]) {
        glBufferData(GL_ARRAY_BUFFER, bytes, data, GL_STREAM_DRAW);
        st->scratch_cap[kind] = bytes;
    } else {
        glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
    }
    return 1;
}

/* Copy only instances whose clip id falls in [base, base+count) into the
 * per-kind scratch buffer. Used only when a frame exceeds the 256-clip UBO;
 * returns the filtered byte count. */
static size_t filter_instances(UiGles *st, int kind, const void *array,
                               uint32_t start, uint32_t count,
                               uint32_t base, uint32_t window) {
    GLsizei stride = stride_of(kind);
    size_t need = (size_t)count * (size_t)stride;
    if (need > st->scratch_bytes[kind]) {
        unsigned char *grown = (unsigned char *)realloc(st->scratch[kind], need);
        if (grown == NULL) return 0;
        st->scratch[kind] = grown;
        st->scratch_bytes[kind] = need;
    }
    const unsigned char *src = (const unsigned char *)array + (size_t)start * (size_t)stride;
    unsigned char *dst = st->scratch[kind];
    size_t used = 0;
    size_t clip_off = clip_offset_of(kind);
    for (uint32_t i = 0; i < count; i++) {
        const unsigned char *inst = src + (size_t)i * (size_t)stride;
        uint32_t clip;
        memcpy(&clip, inst + clip_off, sizeof clip);
        if (clip >= base && clip < base + window) {
            memcpy(dst + used, inst, (size_t)stride);
            /* The window is uploaded at slot 0; rebase the copied id so the
             * shader's `u_clips[clip_id]` stays in bounds. */
            uint32_t local = clip - base;
            memcpy(dst + used + clip_off, &local, sizeof local);
            used += (size_t)stride;
        }
    }
    return used;
}

/* ── JNI: renderer lifecycle ────────────────────────────────────── */

JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeGlesCreate(JNIEnv *env, jobject thiz,
                                                     jlong session, jstring shader_source) {
    (void)thiz;
    if (session == 0 || shader_source == NULL) return -1;

    const char *src = (*env)->GetStringUTFChars(env, shader_source, NULL);
    if (src == NULL) return -1;

    /* A rebuild on the same context (theme swap / manual reload). */
    ui_gles_free(g_gles, 1);
    g_gles = NULL;

    UiGles *st = (UiGles *)calloc(1, sizeof(UiGles));
    if (st == NULL) {
        (*env)->ReleaseStringUTFChars(env, shader_source, src);
        return -1;
    }
    for (int i = 0; i < 4; i++) {
        st->loc_viewport[i] = -1;
        st->loc_atlas[i] = -1;
        st->loc_scale[i] = -1;
        st->loc_time[i] = -1;
    }

    int failed = 0;
    for (int i = 0; i < 4 && !failed; i++) {
        st->program[i] = build_program(src, kPipelineNames[i], i, st);
        if (st->program[i] == 0) failed = 1;
    }
    (*env)->ReleaseStringUTFChars(env, shader_source, src);
    if (failed) {
        ui_gles_free(st, 1);
        return -1;
    }

    /* Full-screen unit quad; the vertex shader expands it per instance. */
    static const GLfloat corners[8] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    glGenBuffers(1, &st->corner_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, st->corner_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof corners, corners, GL_STATIC_DRAW);

    for (int kind = 0; kind < 4; kind++) {
        glGenBuffers(1, &st->inst_vbo[kind]);
        glGenBuffers(1, &st->scratch_vbo[kind]);
        glGenVertexArrays(1, &st->vao[kind]);
        glBindVertexArray(st->vao[kind]);
        glBindBuffer(GL_ARRAY_BUFFER, st->corner_vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * (GLsizei)sizeof(GLfloat), NULL);
        glBindBuffer(GL_ARRAY_BUFFER, st->inst_vbo[kind]);
        configure_attributes(kind, 0);
    }
    glBindVertexArray(0);

    /* Clip rectangles: 256 vec4 std140 block (4 KiB, inside the GLES 3.0
     * guaranteed 16 KiB minimum). Oversized frames use ordered windows. */
    glGenBuffers(1, &st->clip_ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, st->clip_ubo);
    glBufferData(GL_UNIFORM_BUFFER, MAX_CLIPS * 16, NULL, GL_STREAM_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, st->clip_ubo);

    const uint8_t *pixels = NULL;
    uint32_t width = 0, height = 0;
    float px_range = 0.0f;
    if (sofuu_ui_session_get_atlas((SofuuUiSession *)(intptr_t)session,
                                   &pixels, &width, &height, &px_range) == 0 &&
        pixels != NULL && width > 0 && height > 0) {
        glGenTextures(1, &st->atlas_tex);
        glBindTexture(GL_TEXTURE_2D, st->atlas_tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, (GLsizei)width, (GLsizei)height, 0,
                     GL_RED, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        __android_log_print(ANDROID_LOG_WARN, TAG, "glyph atlas unavailable");
    }

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    /* Premultiplied alpha, identical to Metal and gles.rs. */
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                        GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "GL init error 0x%04x", err);
        ui_gles_free(st, 1);
        return -1;
    }

    st->ready = 1;
    g_gles = st;
    return 0;
}

JNIEXPORT void JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeGlesResize(JNIEnv *env, jobject thiz,
                                                     jfloat width, jfloat height,
                                                     jfloat scale) {
    (void)env; (void)thiz;
    if (width <= 0 || height <= 0 || scale <= 0) return;
    glViewport(0, 0,
               (GLsizei)lroundf(width * scale),
               (GLsizei)lroundf(height * scale));
}

JNIEXPORT jint JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeGlesDraw(JNIEnv *env, jobject thiz,
                                                   jlong session, jfloat dt,
                                                   jfloat width, jfloat height,
                                                   jfloat scale) {
    (void)env; (void)thiz;
    UiGles *st = g_gles;
    if (st == NULL || !st->ready || session == 0) return -1;
    if (width <= 0 || height <= 0 || scale <= 0) return 0;

    SofuuUiDrawList dl;
    memset(&dl, 0, sizeof dl);
    int rc = sofuu_ui_poll_draw_list((SofuuUiSession *)(intptr_t)session,
                                     dt, width, height, scale, &dl);
    if (rc != 0) {
        if (!st->logged_failure) {
            st->logged_failure = 1;
            __android_log_print(ANDROID_LOG_ERROR, TAG, "poll_draw_list rc=%d", rc);
        }
        return -1;
    }

    float s = dl.uniforms.scale_factor;
    if (!(s > 0.0f)) s = scale;
    GLsizei vw = (GLsizei)lroundf(dl.uniforms.viewport_size[0] * s);
    GLsizei vh = (GLsizei)lroundf(dl.uniforms.viewport_size[1] * s);
    if (vw < 1) vw = 1;
    if (vh < 1) vh = 1;
    glViewport(0, 0, vw, vh);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (dl.cmd_count == 0 || dl.cmds == NULL) {
        return dl.needs_next_frame ? 1 : 0;
    }

    const void *arrays[4] = {dl.quads, dl.ribbons, dl.strokes, dl.glyphs};
    uint32_t lengths[4] = {dl.quad_count, dl.ribbon_count, dl.stroke_count, dl.glyph_count};
    const GLsizeiptr bytes[4] = {
        (GLsizeiptr)((size_t)dl.quad_count * sizeof(SofuuUiQuadInstance)),
        (GLsizeiptr)((size_t)dl.ribbon_count * sizeof(SofuuUiRibbonInstance)),
        (GLsizeiptr)((size_t)dl.stroke_count * sizeof(SofuuUiStrokeInstance)),
        (GLsizeiptr)((size_t)dl.glyph_count * sizeof(SofuuUiGlyphInstance)),
    };
    for (int kind = 0; kind < 4; kind++) {
        if (arrays[kind] != NULL && lengths[kind] > 0) {
            upload_streaming(st, kind, arrays[kind], bytes[kind]);
        }
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, st->atlas_tex);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA,
                        GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    /* clip_id 0 is always the full viewport; a nil clip array still needs
     * one valid rectangle for the shader's clip test. */
    SofuuUiClipRect fallback = {0.0f, 0.0f, dl.uniforms.viewport_size[0],
                                dl.uniforms.viewport_size[1]};
    const SofuuUiClipRect *clips = dl.clips != NULL ? dl.clips : &fallback;
    uint32_t clip_total = (dl.clips != NULL && dl.clip_count != 0) ? dl.clip_count : 1;
    uint32_t windows = (clip_total + MAX_CLIPS - 1) / MAX_CLIPS;
    if (windows == 0) windows = 1;

    for (uint32_t win = 0; win < windows; win++) {
        uint32_t base = win * MAX_CLIPS;
        uint32_t count = clip_total - base;
        if (count > MAX_CLIPS) count = MAX_CLIPS;
        if (count == 0) continue;

        glBindBuffer(GL_UNIFORM_BUFFER, st->clip_ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0,
                        (GLsizeiptr)(count * sizeof(SofuuUiClipRect)), clips + base);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, st->clip_ubo);

        for (uint32_t ci = 0; ci < dl.cmd_count; ci++) {
            const SofuuUiDrawCmd *cmd = &dl.cmds[ci];
            int kind = (int)cmd->kind;
            if (kind < 0 || kind > 3 || cmd->count == 0) continue;
            if (arrays[kind] == NULL) continue;
            uint32_t start = cmd->start;
            uint32_t count_i = cmd->count;
            uint32_t len = lengths[kind];
            if (start >= len || count_i > len - start) continue;

            uint32_t draw_count;
            glBindVertexArray(st->vao[kind]);
            if (windows == 1) {
                /* Zero-copy: point the attribute offsets into the frame's
                 * uploaded array; no per-command copy. */
                glBindBuffer(GL_ARRAY_BUFFER, st->inst_vbo[kind]);
                configure_attributes(kind, (size_t)start * (size_t)stride_of(kind));
                draw_count = count_i;
            } else {
                size_t used = filter_instances(st, kind, arrays[kind], start,
                                               count_i, base, count);
                if (used == 0) continue;
                upload_scratch(st, kind, st->scratch[kind], (GLsizeiptr)used);
                configure_attributes(kind, 0);
                draw_count = (uint32_t)(used / (size_t)stride_of(kind));
            }

            GLuint program = st->program[kind];
            glUseProgram(program);
            if (st->loc_viewport[kind] >= 0) {
                glUniform2f(st->loc_viewport[kind], dl.uniforms.viewport_size[0],
                            dl.uniforms.viewport_size[1]);
            }
            if (st->loc_atlas[kind] >= 0) {
                glUniform2f(st->loc_atlas[kind], dl.uniforms.atlas_size[0],
                            dl.uniforms.atlas_size[1]);
            }
            if (st->loc_scale[kind] >= 0) {
                glUniform1f(st->loc_scale[kind], dl.uniforms.scale_factor);
            }
            if (st->loc_time[kind] >= 0) {
                glUniform1f(st->loc_time[kind], dl.uniforms.time_sec);
            }

            glBindVertexArray(st->vao[kind]);
            glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)draw_count);
        }
    }

    glBindVertexArray(0);
    glUseProgram(0);
    return dl.needs_next_frame ? 1 : 0;
}

/* Release host memory only. Safe to call from the UI thread after the GL
 * thread stopped; object names are reclaimed with the EGL context. */
JNIEXPORT void JNICALL
Java_com_sofuu_render_SofuuUiNative_nativeGlesDestroy(JNIEnv *env, jobject thiz) {
    (void)env; (void)thiz;
    ui_gles_free(g_gles, 0);
    g_gles = NULL;
}
