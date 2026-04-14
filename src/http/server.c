/*
 * src/http/server.c — Sofuu HTTP/1.1 server
 *
 * APIs:
 *   Sofuu.createServer(handler) → server object with .listen(port, cb)
 *   sofuu.serve(port, handler)  → legacy
 *
 * req: { method, url, body }
 * res: { writeHead(status, headers), end(body), send(body) }
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <uv.h>
#include "quickjs.h"
#include "quickjs-libc.h"
#include "http_parser.h"
#include "server.h"
#include "loop.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* ─── Class IDs ─────────────────────────────────────────────────── */
static JSClassID res_class_id  = 0;
static JSClassID srv_class_id  = 0;

/* ─── Structs ────────────────────────────────────────────────────── */
typedef struct sofuu_server_s {
    uv_tcp_t  tcp;
    JSContext *ctx;
    JSValue    handler;
} sofuu_server_t;

typedef struct sofuu_client_s {
    uv_tcp_t        tcp;
    sofuu_server_t *srv;
    http_parser     parser;
    /* request accumulation */
    char   *url;       size_t url_len;
    char   *body;      size_t body_len;
    /* response state */
    int     status;
    char    ctype[256];
    int     headers_set;
    /* JS objects */
    JSValue req_val;
    JSValue res_val;
} sofuu_client_t;

/* ─── write request: heap-allocates the full response buffer ─────── */
static void client_free(uv_handle_t *h);  /* forward declaration */

typedef struct {
    uv_write_t  w;      /* MUST be first */
    char       *data;
    sofuu_client_t *client;
} write_req_t;

static void on_write_done(uv_write_t *req, int status) {
    (void)status;
    write_req_t *wr = (write_req_t *)req;
    free(wr->data);
    uv_close((uv_handle_t *)&wr->client->tcp, client_free);
    free(wr);
}

static void client_free(uv_handle_t *h) { free(h); }

static void send_response(sofuu_client_t *c,
                           int status, const char *ct,
                           const char *body, size_t blen) {
    const char *st = "OK";
    if (status==201) st="Created"; else if(status==400) st="Bad Request";
    else if(status==404) st="Not Found"; else if(status==500) st="Internal Server Error";

    char hdr[1024];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n\r\n",
        status, st, ct ? ct : "text/plain", blen);

    write_req_t *wr = (write_req_t *)calloc(1, sizeof(*wr));
    wr->data   = (char *)malloc((size_t)hl + blen);
    wr->client = c;
    memcpy(wr->data, hdr, (size_t)hl);
    if (body && blen) memcpy(wr->data + hl, body, blen);

    uv_buf_t buf = uv_buf_init(wr->data, (unsigned)(hl + blen));
    uv_write(&wr->w, (uv_stream_t *)&c->tcp, &buf, 1, on_write_done);
}

/* ─── res JS methods ─────────────────────────────────────────────── */
static JSValue js_res_writeHead(JSContext *ctx, JSValueConst this_val,
                                 int argc, JSValueConst *argv) {
    sofuu_client_t *c = (sofuu_client_t *)JS_GetOpaque(this_val, res_class_id);
    if (!c) return JS_UNDEFINED;
    if (argc >= 1) JS_ToInt32(ctx, &c->status, argv[0]);
    if (argc >= 2 && JS_IsObject(argv[1])) {
        const char *keys[] = {"Content-Type","content-type",NULL};
        for (int i = 0; keys[i]; i++) {
            JSValue v = JS_GetPropertyStr(ctx, argv[1], keys[i]);
            if (!JS_IsUndefined(v)) {
                const char *s = JS_ToCString(ctx, v);
                if (s) { strncpy(c->ctype, s, 255); JS_FreeCString(ctx, s); }
                JS_FreeValue(ctx, v);
                break;
            }
            JS_FreeValue(ctx, v);
        }
    }
    c->headers_set = 1;
    return JS_UNDEFINED;
}

static JSValue js_res_end(JSContext *ctx, JSValueConst this_val,
                           int argc, JSValueConst *argv) {
    sofuu_client_t *c = (sofuu_client_t *)JS_GetOpaque(this_val, res_class_id);
    if (!c) return JS_UNDEFINED;
    const char *body = ""; size_t blen = 0;
    const char *tmp = NULL;
    if (argc >= 1 && !JS_IsUndefined(argv[0]) && !JS_IsNull(argv[0])) {
        tmp = JS_ToCStringLen(ctx, &blen, argv[0]);
        if (tmp) body = tmp;
    }
    int st = (c->status > 0) ? c->status : 200;
    const char *ct = c->ctype[0] ? c->ctype : "text/plain";
    send_response(c, st, ct, body, blen);
    if (tmp) JS_FreeCString(ctx, tmp);
    return JS_UNDEFINED;
}

static JSValue js_res_send(JSContext *ctx, JSValueConst this_val,
                            int argc, JSValueConst *argv) {
    sofuu_client_t *c = (sofuu_client_t *)JS_GetOpaque(this_val, res_class_id);
    if (!c) return JS_UNDEFINED;
    if (c->status == 0) c->status = 200;
    return js_res_end(ctx, this_val, argc, argv);
}

/* ─── http_parser callbacks ──────────────────────────────────────── */
static int cb_url(http_parser *p, const char *at, size_t len) {
    sofuu_client_t *c = (sofuu_client_t *)p->data;
    c->url = (char *)realloc(c->url, c->url_len + len + 1);
    memcpy(c->url + c->url_len, at, len);
    c->url_len += len;
    c->url[c->url_len] = '\0';
    return 0;
}
static int cb_body(http_parser *p, const char *at, size_t len) {
    sofuu_client_t *c = (sofuu_client_t *)p->data;
    c->body = (char *)realloc(c->body, c->body_len + len + 1);
    memcpy(c->body + c->body_len, at, len);
    c->body_len += len;
    c->body[c->body_len] = '\0';
    return 0;
}
static int cb_message_complete(http_parser *p) {
    sofuu_client_t *c   = (sofuu_client_t *)p->data;
    JSContext      *ctx = c->srv->ctx;

    c->req_val = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, c->req_val, "method",
        JS_NewString(ctx, http_method_str(p->method)));
    JS_SetPropertyStr(ctx, c->req_val, "url",
        JS_NewString(ctx, c->url ? c->url : "/"));
    if (c->body)
        JS_SetPropertyStr(ctx, c->req_val, "body",
            JS_NewStringLen(ctx, c->body, c->body_len));

    c->res_val = JS_NewObjectClass(ctx, res_class_id);
    JS_SetOpaque(c->res_val, c);
    JS_SetPropertyStr(ctx, c->res_val, "writeHead",
        JS_NewCFunction(ctx, js_res_writeHead, "writeHead", 2));
    JS_SetPropertyStr(ctx, c->res_val, "end",
        JS_NewCFunction(ctx, js_res_end, "end", 1));
    JS_SetPropertyStr(ctx, c->res_val, "send",
        JS_NewCFunction(ctx, js_res_send, "send", 1));

    JSValue args[2] = { c->req_val, c->res_val };
    JSValue ret = JS_Call(ctx, c->srv->handler, JS_UNDEFINED, 2, args);
    if (JS_IsException(ret)) js_std_dump_error(ctx);
    JS_FreeValue(ctx, ret);
    JS_FreeValue(ctx, c->req_val); c->req_val = JS_UNDEFINED;
    JS_FreeValue(ctx, c->res_val); c->res_val = JS_UNDEFINED;
    return 0;
}

static http_parser_settings g_settings = {
    .on_url              = cb_url,
    .on_body             = cb_body,
    .on_message_complete = cb_message_complete,
};

/* ─── libuv callbacks ────────────────────────────────────────────── */
static void on_alloc(uv_handle_t *h, size_t sz, uv_buf_t *b) {
    (void)h; b->base = (char *)malloc(sz); b->len = sz;
}
static void on_read(uv_stream_t *s, ssize_t n, const uv_buf_t *b) {
    sofuu_client_t *c = (sofuu_client_t *)s;
    if (n > 0)
        http_parser_execute(&c->parser, &g_settings, b->base, (size_t)n);
    else if (n < 0)
        uv_close((uv_handle_t *)s, client_free);
    if (b->base) free(b->base);
}
static void on_connection(uv_stream_t *s, int status) {
    if (status < 0) return;
    sofuu_server_t *srv = (sofuu_server_t *)s;
    sofuu_client_t *c   = (sofuu_client_t *)calloc(1, sizeof(*c));
    c->srv     = srv;
    c->req_val = JS_UNDEFINED;
    c->res_val = JS_UNDEFINED;
    uv_tcp_init(sofuu_loop_get(), &c->tcp);
    http_parser_init(&c->parser, HTTP_REQUEST);
    c->parser.data = c;
    if (uv_accept(s, (uv_stream_t *)&c->tcp) == 0)
        uv_read_start((uv_stream_t *)&c->tcp, on_alloc, on_read);
    else
        uv_close((uv_handle_t *)&c->tcp, client_free);
}

/* ─── Start listening ────────────────────────────────────────────── */
static int do_listen(sofuu_server_t *srv, int port) {
    uv_tcp_init(sofuu_loop_get(), &srv->tcp);
    struct sockaddr_in a;
    uv_ip4_addr("0.0.0.0", port, &a);
    uv_tcp_bind(&srv->tcp, (struct sockaddr *)&a, 0);
    return uv_listen((uv_stream_t *)&srv->tcp, 128, on_connection);
}

/* ─── server.listen(port, cb) ────────────────────────────────────── */
static JSValue js_server_listen(JSContext *ctx, JSValueConst this_val,
                                 int argc, JSValueConst *argv) {
    sofuu_server_t *srv = (sofuu_server_t *)JS_GetOpaque(this_val, srv_class_id);
    if (!srv) return JS_ThrowTypeError(ctx, "server.listen: bad object");
    int32_t port = 3000;
    if (argc >= 1) JS_ToInt32(ctx, &port, argv[0]);
    int r = do_listen(srv, (int)port);
    if (r < 0) return JS_ThrowTypeError(ctx, "%s", uv_strerror(r));
    if (argc >= 2 && JS_IsFunction(ctx, argv[1])) {
        JSValue ret = JS_Call(ctx, argv[1], JS_UNDEFINED, 0, NULL);
        JS_FreeValue(ctx, ret);
    }
    return this_val;
}

/* ─── createServer(handler) ──────────────────────────────────────── */
static JSValue js_createServer(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_IsFunction(ctx, argv[0]))
        return JS_ThrowTypeError(ctx, "createServer: function required");
    sofuu_server_t *srv = (sofuu_server_t *)calloc(1, sizeof(*srv));
    srv->ctx     = ctx;
    srv->handler = JS_DupValue(ctx, argv[0]);
    JSValue obj  = JS_NewObjectClass(ctx, srv_class_id);
    JS_SetOpaque(obj, srv);
    JS_SetPropertyStr(ctx, obj, "listen",
        JS_NewCFunction(ctx, js_server_listen, "listen", 2));
    return obj;
}

/* ─── Legacy sofuu.serve(port, handler) ──────────────────────────── */
static JSValue js_serve(JSContext *ctx, JSValueConst this_val,
                         int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2 || !JS_IsFunction(ctx, argv[1]))
        return JS_ThrowTypeError(ctx, "serve(port, fn)");
    int32_t port = 0; JS_ToInt32(ctx, &port, argv[0]);
    sofuu_server_t *srv = (sofuu_server_t *)calloc(1, sizeof(*srv));
    srv->ctx     = ctx;
    srv->handler = JS_DupValue(ctx, argv[1]);
    int r = do_listen(srv, (int)port);
    if (r < 0) { free(srv); return JS_ThrowTypeError(ctx, "%s", uv_strerror(r)); }
    return JS_UNDEFINED;
}

/* ─── Class defs ─────────────────────────────────────────────────── */
static JSClassDef res_class_def = { "SofuuResponse", .finalizer = NULL };
static JSClassDef srv_class_def = { "SofuuServer",   .finalizer = NULL };

/* ─── Registration ───────────────────────────────────────────────── */
void mod_http_server_register(JSContext *ctx) {
    JS_NewClassID(&res_class_id);
    JS_NewClass(JS_GetRuntime(ctx), res_class_id, &res_class_def);
    JS_NewClassID(&srv_class_id);
    JS_NewClass(JS_GetRuntime(ctx), srv_class_id, &srv_class_def);


    JSValue global = JS_GetGlobalObject(ctx);

    /* Global createServer (also copied to Sofuu by engine) */
    JS_SetPropertyStr(ctx, global, "createServer",
        JS_NewCFunction(ctx, js_createServer, "createServer", 1));

    /* sofuu.serve (legacy) */
    JSValue sofuu = JS_GetPropertyStr(ctx, global, "sofuu");
    if (JS_IsUndefined(sofuu) || JS_IsNull(sofuu)) {
        sofuu = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global, "sofuu", JS_DupValue(ctx, sofuu));
    }
    JS_SetPropertyStr(ctx, sofuu, "serve",
        JS_NewCFunction(ctx, js_serve, "serve", 2));
    JS_SetPropertyStr(ctx, sofuu, "createServer",
        JS_NewCFunction(ctx, js_createServer, "createServer", 1));
    
    JS_FreeValue(ctx, sofuu);
    JS_FreeValue(ctx, global);
}
