/*
 * fs.c — Async File I/O backed by libuv
 */
#include "fs.h"
#include "promises.h"
#include "loop.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

/* ------------------------------------------------------------------ */
/* read_file: open → fstat → read → close → resolve                    */
/* ------------------------------------------------------------------ */

typedef struct {
    sofuu_promise_t *promise;   /* NULL after resolve/reject */
    int              fd;
    char            *buf;
    size_t           buf_size;
    char            *path;      /* for open op */
} read_req_t;

static void rf_close_cb(uv_fs_t *req) {
    uv_fs_req_cleanup(req);
    read_req_t *r = req->data;
    free(r->path);
    free(r->buf);
    free(r);
    free(req);
}

static void rf_read_cb(uv_fs_t *req) {
    read_req_t *r = req->data;
    ssize_t     n = req->result;   /* save before cleanup */
    uv_fs_req_cleanup(req);
    free(req);

    JSContext *ctx = r->promise->ctx;

    if (n < 0) {
        sofuu_promise_reject_str(r->promise, uv_strerror((int)n));
    } else {
        r->buf[n] = '\0';
        JSValue str = JS_NewString(ctx, r->buf);
        sofuu_promise_resolve(r->promise, str);
        JS_FreeValue(ctx, str);
        sofuu_flush_jobs(ctx);
    }
    r->promise = NULL;

    /* Close the fd */
    uv_fs_t *cr = calloc(1, sizeof(uv_fs_t));
    cr->data = r;
    uv_fs_close(sofuu_loop_get(), cr, r->fd, rf_close_cb);
}

static void rf_open_cb(uv_fs_t *req) {
    read_req_t *r      = req->data;
    ssize_t     result = req->result;   /* save before cleanup */
    uv_fs_req_cleanup(req);
    free(req);

    if (result < 0) {
        sofuu_promise_reject_str(r->promise, uv_strerror((int)result));
        free(r->path); free(r->buf); free(r);
        return;
    }
    r->fd = (int)result;

    /* Synchronous fstat to get file size */
    uv_fs_t st;
    uv_fs_fstat(sofuu_loop_get(), &st, r->fd, NULL);
    size_t sz = (size_t)st.statbuf.st_size;
    uv_fs_req_cleanup(&st);

    r->buf      = malloc(sz + 1);
    r->buf_size = sz;

    uv_fs_t *rr = calloc(1, sizeof(uv_fs_t));
    rr->data = r;
    uv_buf_t buf = uv_buf_init(r->buf, (unsigned int)sz);
    uv_fs_read(sofuu_loop_get(), rr, r->fd, &buf, 1, 0, rf_read_cb);
}

static JSValue js_readFile(JSContext *ctx, JSValueConst this_val,
                            int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_ThrowTypeError(ctx, "readFile: path required");

    const char *path = JS_ToCString(ctx, argv[0]);
    if (!path) return JS_EXCEPTION;

    sofuu_promise_t *p;
    JSValue promise = sofuu_promise_new(ctx, &p);

    read_req_t *r = calloc(1, sizeof(read_req_t));
    r->promise = p;
    r->path    = strdup(path);
    JS_FreeCString(ctx, path);

    uv_fs_t *req = calloc(1, sizeof(uv_fs_t));
    req->data = r;
    uv_fs_open(sofuu_loop_get(), req, r->path, O_RDONLY, 0, rf_open_cb);

    return promise;
}

/* ------------------------------------------------------------------ */
/* write_file: open → write → close → resolve                          */
/* ------------------------------------------------------------------ */

typedef struct {
    sofuu_promise_t *promise;
    int              fd;
    char            *buf;
    size_t           buf_len;
    char            *path;
} write_req_t;

static void wf_close_cb(uv_fs_t *req) {
    uv_fs_req_cleanup(req);
    write_req_t *r = req->data;
    free(r->path);
    free(r->buf);
    free(r);
    free(req);
}

static void wf_write_cb(uv_fs_t *req) {
    write_req_t *r = req->data;
    JSContext   *ctx = r->promise->ctx;
    ssize_t      result = req->result;   /* save before cleanup */
    uv_fs_req_cleanup(req);
    free(req);

    if (result < 0)
        sofuu_promise_reject_str(r->promise, uv_strerror((int)result));
    else
        sofuu_promise_resolve(r->promise, JS_UNDEFINED);
    sofuu_flush_jobs(ctx);
    r->promise = NULL;

    uv_fs_t *cr = calloc(1, sizeof(uv_fs_t));
    cr->data = r;
    uv_fs_close(sofuu_loop_get(), cr, r->fd, wf_close_cb);
}

static void wf_open_cb(uv_fs_t *req) {
    write_req_t *r      = req->data;
    ssize_t     result = req->result;
    uv_fs_req_cleanup(req);
    free(req);

    if (result < 0) {
        sofuu_promise_reject_str(r->promise, uv_strerror((int)result));
        free(r->path); free(r->buf); free(r);
        return;
    }
    r->fd = (int)result;

    uv_fs_t   *wr  = calloc(1, sizeof(uv_fs_t));
    wr->data        = r;
    uv_buf_t buf    = uv_buf_init(r->buf, (unsigned int)r->buf_len);
    uv_fs_write(sofuu_loop_get(), wr, r->fd, &buf, 1, -1, wf_write_cb);
}

static JSValue js_write_impl(JSContext *ctx,
                              JSValueConst path_val, JSValueConst data_val,
                              int flags) {
    const char *path = JS_ToCString(ctx, path_val);
    if (!path) return JS_EXCEPTION;

    const char *data = JS_ToCString(ctx, data_val);
    if (!data) { JS_FreeCString(ctx, path); return JS_EXCEPTION; }

    sofuu_promise_t *p;
    JSValue promise = sofuu_promise_new(ctx, &p);

    write_req_t *r = calloc(1, sizeof(write_req_t));
    r->promise = p;
    r->path    = strdup(path);
    r->buf_len = strlen(data);
    r->buf     = malloc(r->buf_len + 1);
    memcpy(r->buf, data, r->buf_len);

    JS_FreeCString(ctx, path);
    JS_FreeCString(ctx, data);

    uv_fs_t *req = calloc(1, sizeof(uv_fs_t));
    req->data = r;
    uv_fs_open(sofuu_loop_get(), req, r->path, flags, 0644, wf_open_cb);
    return promise;
}

static JSValue js_writeFile(JSContext *ctx, JSValueConst this_val,
                             int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2) return JS_ThrowTypeError(ctx, "writeFile: 2 args required");
    return js_write_impl(ctx, argv[0], argv[1],
                         O_WRONLY | O_CREAT | O_TRUNC);
}

static JSValue js_appendFile(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2) return JS_ThrowTypeError(ctx, "appendFile: 2 args required");
    return js_write_impl(ctx, argv[0], argv[1],
                         O_WRONLY | O_CREAT | O_APPEND);
}

/* ------------------------------------------------------------------ */
/* exists — sync stat wrapped in a resolved promise                    */
/* ------------------------------------------------------------------ */
static JSValue js_exists(JSContext *ctx, JSValueConst this_val,
                          int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_FALSE;
    const char *path = JS_ToCString(ctx, argv[0]);
    if (!path) return JS_EXCEPTION;

    uv_fs_t req;
    int r = uv_fs_stat(sofuu_loop_get(), &req, path, NULL);
    uv_fs_req_cleanup(&req);
    JS_FreeCString(ctx, path);

    sofuu_promise_t *p;
    JSValue promise = sofuu_promise_new(ctx, &p);
    sofuu_promise_resolve(p, r == 0 ? JS_TRUE : JS_FALSE);
    return promise;
}

/* ------------------------------------------------------------------ */
/* readdir — sync scandir wrapped in a resolved promise                */
/* ------------------------------------------------------------------ */
static JSValue js_readdir(JSContext *ctx, JSValueConst this_val,
                           int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_ThrowTypeError(ctx, "readdir: path required");
    const char *path = JS_ToCString(ctx, argv[0]);
    if (!path) return JS_EXCEPTION;

    uv_fs_t req;
    int r = uv_fs_scandir(sofuu_loop_get(), &req, path, 0, NULL);
    JS_FreeCString(ctx, path);

    sofuu_promise_t *p;
    JSValue promise = sofuu_promise_new(ctx, &p);

    if (r < 0) {
        uv_fs_req_cleanup(&req);
        sofuu_promise_reject_str(p, uv_strerror(r));
        return promise;
    }

    JSValue arr = JS_NewArray(ctx);
    uv_dirent_t ent;
    uint32_t idx = 0;
    while (uv_fs_scandir_next(&req, &ent) != UV_EOF)
        JS_SetPropertyUint32(ctx, arr, idx++, JS_NewString(ctx, ent.name));
    uv_fs_req_cleanup(&req);

    sofuu_promise_resolve(p, arr);
    JS_FreeValue(ctx, arr);  /* resolve dups it */
    return promise;
}

/* ------------------------------------------------------------------ */
/* Registration                                                         */
/* ------------------------------------------------------------------ */
void mod_fs_register(JSContext *ctx) {
    JSValue global = JS_GetGlobalObject(ctx);

    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global, "sofuu");
    if (JS_IsUndefined(sofuu_obj))
        sofuu_obj = JS_NewObject(ctx);

    JSValue fs = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, fs, "readFile",
        JS_NewCFunction(ctx, js_readFile,   "readFile",   1));
    JS_SetPropertyStr(ctx, fs, "writeFile",
        JS_NewCFunction(ctx, js_writeFile,  "writeFile",  2));
    JS_SetPropertyStr(ctx, fs, "appendFile",
        JS_NewCFunction(ctx, js_appendFile, "appendFile", 2));
    JS_SetPropertyStr(ctx, fs, "exists",
        JS_NewCFunction(ctx, js_exists,     "exists",     1));
    JS_SetPropertyStr(ctx, fs, "readdir",
        JS_NewCFunction(ctx, js_readdir,    "readdir",    1));

    JS_SetPropertyStr(ctx, sofuu_obj, "fs", fs);
    JS_SetPropertyStr(ctx, global, "sofuu", sofuu_obj);
    JS_FreeValue(ctx, global);
}
