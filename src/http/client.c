/*
 * client.c — Sofuu fetch() API
 *
 * Implements the Fetch API as a proper Response object:
 *   fetch(url, {method, headers, body})  → Promise<Response>
 *
 *   Response.status          → number
 *   Response.ok              → boolean (200-299)
 *   Response.statusText      → string
 *   Response.url             → string
 *   Response.headers         → { get(name) → string }
 *   Response.text()          → Promise<string>
 *   Response.json()          → Promise<any>
 *   Response.arrayBuffer()   → Promise<ArrayBuffer>
 */

#include "client.h"
#include "loop.h"
#include "promises.h"

#include <curl/curl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Global curl-multi / libuv state                                      */
/* ------------------------------------------------------------------ */

static CURLM     *curl_handle   = NULL;
static uv_timer_t timeout_timer;
static int        global_init_done = 0;

typedef struct {
    uv_poll_t    poll_handle;
    curl_socket_t sockfd;
} curl_context_t;

/* ------------------------------------------------------------------ */
/* fetch_req_t — per-request state                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    JSContext        *ctx;
    sofuu_promise_t  *promise;

    /* Response body */
    char             *body;
    size_t            body_len;

    /* Raw response headers (concatenated) */
    char             *headers_raw;
    size_t            headers_raw_len;

    /* libcurl handle + slist */
    CURL             *easy;
    struct curl_slist *req_headers;
} fetch_req_t;

/* ------------------------------------------------------------------ */
/* Response JS class                                                    */
/* ------------------------------------------------------------------ */

static JSClassID  response_class_id;

typedef struct {
    char    *body;          /* response body text */
    size_t   body_len;
    char    *headers_raw;   /* raw header string "K: V\r\n..." */
    char    *url;
    long     status;
    char    *status_text;
} response_data_t;

static void response_finalizer(JSRuntime *rt, JSValue val) {
    (void)rt;
    response_data_t *r = JS_GetOpaque(val, response_class_id);
    if (r) {
        free(r->body);
        free(r->headers_raw);
        free(r->url);
        free(r->status_text);
        free(r);
    }
}

static JSClassDef response_class = {
    "Response",
    .finalizer = response_finalizer,
};

/* Response.text() → Promise<string> */
static JSValue response_text(JSContext *ctx, JSValueConst this_val,
                             int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    response_data_t *r = JS_GetOpaque(this_val, response_class_id);
    if (!r) return JS_ThrowTypeError(ctx, "Invalid Response");

    JSValue str = JS_NewStringLen(ctx, r->body ? r->body : "", r->body_len);

    /* Wrap in a pre-resolved promise (spec says .text() returns Promise) */
    JSValue resolvers[2];
    JSValue promise = JS_NewPromiseCapability(ctx, resolvers);
    JSValue ret = JS_Call(ctx, resolvers[0], JS_UNDEFINED, 1, &str);
    JS_FreeValue(ctx, ret);
    JS_FreeValue(ctx, str);
    JS_FreeValue(ctx, resolvers[0]);
    JS_FreeValue(ctx, resolvers[1]);
    return promise;
}

/* Response.json() → Promise<any> */
static JSValue response_json(JSContext *ctx, JSValueConst this_val,
                             int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    response_data_t *r = JS_GetOpaque(this_val, response_class_id);
    if (!r) return JS_ThrowTypeError(ctx, "Invalid Response");

    const char *src = r->body ? r->body : "null";
    JSValue parsed = JS_ParseJSON(ctx, src, strlen(src), "<fetch response>");

    JSValue resolvers[2];
    JSValue promise = JS_NewPromiseCapability(ctx, resolvers);
    if (JS_IsException(parsed)) {
        /* Reject with the parse error */
        JSValue exc = JS_GetException(ctx);
        JSValue ret = JS_Call(ctx, resolvers[1], JS_UNDEFINED, 1, &exc);
        JS_FreeValue(ctx, ret);
        JS_FreeValue(ctx, exc);
    } else {
        JSValue ret = JS_Call(ctx, resolvers[0], JS_UNDEFINED, 1, &parsed);
        JS_FreeValue(ctx, ret);
        JS_FreeValue(ctx, parsed);
    }
    JS_FreeValue(ctx, resolvers[0]);
    JS_FreeValue(ctx, resolvers[1]);
    return promise;
}

/* Response.arrayBuffer() → Promise<ArrayBuffer> */
static JSValue response_array_buffer(JSContext *ctx, JSValueConst this_val,
                                     int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    response_data_t *r = JS_GetOpaque(this_val, response_class_id);
    if (!r) return JS_ThrowTypeError(ctx, "Invalid Response");

    size_t len = r->body_len;
    void *buf = js_malloc(ctx, len ? len : 1);
    if (len) memcpy(buf, r->body, len);
    JSValue ab = JS_NewArrayBuffer(ctx, buf, len, NULL, NULL, 0);

    JSValue resolvers[2];
    JSValue promise = JS_NewPromiseCapability(ctx, resolvers);
    JSValue ret = JS_Call(ctx, resolvers[0], JS_UNDEFINED, 1, &ab);
    JS_FreeValue(ctx, ret);
    JS_FreeValue(ctx, ab);
    JS_FreeValue(ctx, resolvers[0]);
    JS_FreeValue(ctx, resolvers[1]);
    return promise;
}

/*
 * headers.get(name) — minimal but correct case-insensitive lookup
 * Works on "Key: Value\r\nKey2: Value2\r\n..." format
 */
static JSValue headers_get(JSContext *ctx, JSValueConst this_val,
                           int argc, JSValueConst *argv) {
    if (argc < 1) return JS_NULL;

    /* headers_raw is stored as property __raw on the headers object */
    JSValue raw_val = JS_GetPropertyStr(ctx, this_val, "__raw");
    const char *raw = JS_ToCString(ctx, raw_val);
    JS_FreeValue(ctx, raw_val);
    if (!raw) return JS_NULL;

    const char *name = JS_ToCString(ctx, argv[0]);
    if (!name) { JS_FreeCString(ctx, raw); return JS_NULL; }

    /* Case-insensitive scan through raw headers */
    JSValue result = JS_NULL;
    const char *p = raw;
    size_t name_len = strlen(name);

    while (*p) {
        /* Find end of this header line */
        const char *eol = strstr(p, "\r\n");
        if (!eol) eol = p + strlen(p);

        /* Find the colon */
        const char *colon = memchr(p, ':', eol - p);
        if (colon) {
            size_t key_len = colon - p;
            if (key_len == name_len &&
                strncasecmp(p, name, name_len) == 0) {
                /* Found it — extract value */
                const char *vstart = colon + 1;
                while (*vstart == ' ') vstart++;
                size_t vlen = eol - vstart;
                /* Trim trailing \r if present */
                while (vlen > 0 && (vstart[vlen-1] == '\r' ||
                                     vstart[vlen-1] == ' ')) vlen--;
                result = JS_NewStringLen(ctx, vstart, vlen);
                break;
            }
        }

        p = (*eol == '\r') ? eol + 2 : eol + 1;
        if (!*p) break;
    }

    JS_FreeCString(ctx, raw);
    JS_FreeCString(ctx, name);
    return result;
}

static const JSCFunctionListEntry headers_proto[] __attribute__((unused)) = {
    JS_CFUNC_DEF("get", 1, headers_get),
};

/* Build a Response JS object from a completed fetch_req_t */
static JSValue build_response(JSContext *ctx, fetch_req_t *req,
                               long status, const char *final_url) {
    response_data_t *data = calloc(1, sizeof(*data));
    data->body        = req->body;        req->body = NULL;
    data->body_len    = req->body_len;
    data->headers_raw = req->headers_raw; req->headers_raw = NULL;
    data->url         = strdup(final_url ? final_url : "");
    data->status      = status;

    /* status text */
    if      (status == 200) data->status_text = strdup("OK");
    else if (status == 201) data->status_text = strdup("Created");
    else if (status == 204) data->status_text = strdup("No Content");
    else if (status == 400) data->status_text = strdup("Bad Request");
    else if (status == 401) data->status_text = strdup("Unauthorized");
    else if (status == 403) data->status_text = strdup("Forbidden");
    else if (status == 404) data->status_text = strdup("Not Found");
    else if (status == 500) data->status_text = strdup("Internal Server Error");
    else                    data->status_text = strdup("");

    /* Create the JS object */
    JSValue obj = JS_NewObjectClass(ctx, response_class_id);
    JS_SetOpaque(obj, data);

    /* Properties */
    JS_SetPropertyStr(ctx, obj, "status",
                      JS_NewInt32(ctx, (int32_t)status));
    JS_SetPropertyStr(ctx, obj, "ok",
                      JS_NewBool(ctx, status >= 200 && status < 300));
    JS_SetPropertyStr(ctx, obj, "statusText",
                      JS_NewString(ctx, data->status_text));
    JS_SetPropertyStr(ctx, obj, "url",
                      JS_NewString(ctx, data->url));

    /* Methods */
    JS_SetPropertyStr(ctx, obj, "text",
                      JS_NewCFunction(ctx, response_text, "text", 0));
    JS_SetPropertyStr(ctx, obj, "json",
                      JS_NewCFunction(ctx, response_json, "json", 0));
    JS_SetPropertyStr(ctx, obj, "arrayBuffer",
                      JS_NewCFunction(ctx, response_array_buffer, "arrayBuffer", 0));

    /* headers object with .get() method */
    JSValue headers_obj = JS_NewObject(ctx);
    /* Store raw headers as hidden property */
    JS_SetPropertyStr(ctx, headers_obj, "__raw",
                      JS_NewString(ctx, data->headers_raw ? data->headers_raw : ""));
    JS_SetPropertyStr(ctx, headers_obj, "get",
                      JS_NewCFunction(ctx, headers_get, "get", 1));
    JS_SetPropertyStr(ctx, obj, "headers", headers_obj);

    return obj;
}

/* ------------------------------------------------------------------ */
/* curl-multi ↔ libuv bridge                                            */
/* ------------------------------------------------------------------ */

static curl_context_t *create_curl_context(curl_socket_t sockfd) {
    curl_context_t *context = malloc(sizeof(*context));
    context->sockfd = sockfd;
    uv_poll_init_socket(sofuu_loop_get(), &context->poll_handle, sockfd);
    context->poll_handle.data = context;
    return context;
}

static void curl_close_cb(uv_handle_t *handle) {
    curl_context_t *context = (curl_context_t*)handle->data;
    free(context);
}

static void destroy_curl_context(curl_context_t *context) {
    uv_close((uv_handle_t*)&context->poll_handle, curl_close_cb);
}

/* ------------------------------------------------------------------ */
/* Response body + header write callbacks                               */
/* ------------------------------------------------------------------ */

static size_t write_body_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t n = size * nmemb;
    fetch_req_t *req = (fetch_req_t*)userdata;
    req->body = realloc(req->body, req->body_len + n + 1);
    memcpy(req->body + req->body_len, ptr, n);
    req->body_len += n;
    req->body[req->body_len] = '\0';
    return n;
}

static size_t write_header_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t n = size * nmemb;
    fetch_req_t *req = (fetch_req_t*)userdata;
    req->headers_raw = realloc(req->headers_raw, req->headers_raw_len + n + 1);
    memcpy(req->headers_raw + req->headers_raw_len, ptr, n);
    req->headers_raw_len += n;
    req->headers_raw[req->headers_raw_len] = '\0';
    return n;
}

/* ------------------------------------------------------------------ */
/* Multi-handle completion                                               */
/* ------------------------------------------------------------------ */

static void check_multi_info_global(void) {
    int pending;
    CURLMsg *message;

    while ((message = curl_multi_info_read(curl_handle, &pending))) {
        if (message->msg != CURLMSG_DONE) continue;

        CURL     *easy        = message->easy_handle;
        CURLcode  return_code = message->data.result;

        fetch_req_t *req = NULL;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &req);

        if (req) {
            JSContext *ctx = req->ctx;

            if (return_code == CURLE_OK) {
                long status = 0;
                char *final_url = NULL;
                curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
                curl_easy_getinfo(easy, CURLINFO_EFFECTIVE_URL, &final_url);

                JSValue response = build_response(ctx, req, status, final_url);
                sofuu_promise_resolve(req->promise, response);
                JS_FreeValue(ctx, response);
            } else {
                JSValue err = JS_NewError(ctx);
                JS_SetPropertyStr(ctx, err, "message",
                    JS_NewString(ctx, curl_easy_strerror(return_code)));
                sofuu_promise_reject(req->promise, err);
            }

            if (req->body)        free(req->body);
            if (req->headers_raw) free(req->headers_raw);
            if (req->req_headers) curl_slist_free_all(req->req_headers);
            free(req);

            sofuu_flush_jobs(ctx);
        }

        curl_multi_remove_handle(curl_handle, easy);
        curl_easy_cleanup(easy);
    }
}

static void curl_perform_global(uv_poll_t *req, int status, int events) {
    (void)status;
    int running_handles, flags = 0;
    if (events & UV_READABLE) flags |= CURL_CSELECT_IN;
    if (events & UV_WRITABLE) flags |= CURL_CSELECT_OUT;
    curl_context_t *context = (curl_context_t*)req->data;
    curl_multi_socket_action(curl_handle, context->sockfd, flags, &running_handles);
    check_multi_info_global();
}

static void on_timeout(uv_timer_t *t) {
    (void)t;
    int running_handles;
    curl_multi_socket_action(curl_handle, CURL_SOCKET_TIMEOUT, 0, &running_handles);
    check_multi_info_global();
}

static int handle_socket(CURL *easy, curl_socket_t s, int action,
                         void *userp, void *socketp) {
    (void)easy; (void)userp;
    curl_context_t *curl_context;
    if (action == CURL_POLL_IN || action == CURL_POLL_OUT || action == CURL_POLL_INOUT) {
        if (socketp) {
            curl_context = (curl_context_t*)socketp;
        } else {
            curl_context = create_curl_context(s);
            curl_multi_assign(curl_handle, s, (void*)curl_context);
        }
        int events = 0;
        if (action != CURL_POLL_IN)  events |= UV_WRITABLE;
        if (action != CURL_POLL_OUT) events |= UV_READABLE;
        uv_poll_start(&curl_context->poll_handle, events, curl_perform_global);
    } else {
        if (socketp) {
            uv_poll_stop(&((curl_context_t*)socketp)->poll_handle);
            destroy_curl_context((curl_context_t*)socketp);
            curl_multi_assign(curl_handle, s, NULL);
        }
    }
    return 0;
}

static int start_timeout(CURLM *multi, long timeout_ms, void *userp) {
    (void)multi; (void)userp;
    if (timeout_ms < 0) {
        uv_timer_stop(&timeout_timer);
    } else {
        if (timeout_ms == 0) timeout_ms = 1;
        uv_timer_start(&timeout_timer, on_timeout, timeout_ms, 0);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* JS: fetch(url, options?) → Promise<Response>                         */
/* ------------------------------------------------------------------ */

static JSValue js_sofuu_fetch(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_ThrowTypeError(ctx, "fetch requires a URL");

    const char *url = JS_ToCString(ctx, argv[0]);
    if (!url) return JS_EXCEPTION;

    fetch_req_t *req = calloc(1, sizeof(fetch_req_t));
    req->ctx  = ctx;
    req->easy = curl_easy_init();

    curl_easy_setopt(req->easy, CURLOPT_URL,            url);
    curl_easy_setopt(req->easy, CURLOPT_WRITEFUNCTION,  write_body_cb);
    curl_easy_setopt(req->easy, CURLOPT_WRITEDATA,      req);
    curl_easy_setopt(req->easy, CURLOPT_HEADERFUNCTION, write_header_cb);
    curl_easy_setopt(req->easy, CURLOPT_HEADERDATA,     req);
    curl_easy_setopt(req->easy, CURLOPT_PRIVATE,        req);
    curl_easy_setopt(req->easy, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(req->easy, CURLOPT_TIMEOUT,        0L); /* Infinite wait for local LLM */

    /* Parse options: { method, headers, body } */
    if (argc > 1 && JS_IsObject(argv[1])) {
        /* method */
        JSValue method_val = JS_GetPropertyStr(ctx, argv[1], "method");
        if (!JS_IsUndefined(method_val)) {
            const char *method = JS_ToCString(ctx, method_val);
            if (method) {
                curl_easy_setopt(req->easy, CURLOPT_CUSTOMREQUEST, method);
                JS_FreeCString(ctx, method);
            }
        }
        JS_FreeValue(ctx, method_val);

        /* headers */
        JSValue headers_val = JS_GetPropertyStr(ctx, argv[1], "headers");
        if (JS_IsObject(headers_val)) {
            JSPropertyEnum *ptab;
            uint32_t plen;
            if (JS_GetOwnPropertyNames(ctx, &ptab, &plen, headers_val,
                                       JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0) {
                for (uint32_t i = 0; i < plen; i++) {
                    const char *key = JS_AtomToCString(ctx, ptab[i].atom);
                    JSValue val     = JS_GetProperty(ctx, headers_val, ptab[i].atom);
                    const char *vs  = JS_ToCString(ctx, val);
                    if (key && vs) {
                        char hbuf[1024];
                        snprintf(hbuf, sizeof(hbuf), "%s: %s", key, vs);
                        req->req_headers = curl_slist_append(req->req_headers, hbuf);
                    }
                    if (key) JS_FreeCString(ctx, key);
                    if (vs)  JS_FreeCString(ctx, vs);
                    JS_FreeValue(ctx, val);
                    JS_FreeAtom(ctx, ptab[i].atom);
                }
                js_free(ctx, ptab);
            }
        }
        JS_FreeValue(ctx, headers_val);
        if (req->req_headers)
            curl_easy_setopt(req->easy, CURLOPT_HTTPHEADER, req->req_headers);

        /* body */
        JSValue body_val = JS_GetPropertyStr(ctx, argv[1], "body");
        if (!JS_IsUndefined(body_val) && !JS_IsNull(body_val)) {
            const char *body = JS_ToCString(ctx, body_val);
            if (body) {
                curl_easy_setopt(req->easy, CURLOPT_COPYPOSTFIELDS, body);
                JS_FreeCString(ctx, body);
            }
        }
        JS_FreeValue(ctx, body_val);
    }

    JS_FreeCString(ctx, url);

    JSValue promise = sofuu_promise_new(ctx, &req->promise);

    curl_multi_add_handle(curl_handle, req->easy);

    int running_handles;
    curl_multi_socket_action(curl_handle, CURL_SOCKET_TIMEOUT, 0, &running_handles);

    return promise;
}

/* ------------------------------------------------------------------ */
/* Module registration                                                   */
/* ------------------------------------------------------------------ */

void mod_http_client_register(JSContext *ctx) {
    /* Register Response class */
    JS_NewClassID(&response_class_id);
    JS_NewClass(JS_GetRuntime(ctx), response_class_id, &response_class);

    if (!global_init_done) {
        curl_global_init(CURL_GLOBAL_ALL);
        uv_loop_t *loop = sofuu_loop_get();
        uv_timer_init(loop, &timeout_timer);

        curl_handle = curl_multi_init();
        curl_multi_setopt(curl_handle, CURLMOPT_SOCKETFUNCTION, handle_socket);
        curl_multi_setopt(curl_handle, CURLMOPT_TIMERFUNCTION,  start_timeout);

        global_init_done = 1;
    }

    JSValue global = JS_GetGlobalObject(ctx);
    JSValue sofuu  = JS_GetPropertyStr(ctx, global, "sofuu");

    if (JS_IsUndefined(sofuu)) {
        sofuu = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global, "sofuu", JS_DupValue(ctx, sofuu));
    }

    JS_SetPropertyStr(ctx, sofuu, "fetch",
                      JS_NewCFunction(ctx, js_sofuu_fetch, "fetch", 1));

    JS_FreeValue(ctx, sofuu);
    JS_FreeValue(ctx, global);

    /* Also install as global `fetch` (mirrors browser/Deno/Bun behavior) */
    JS_SetPropertyStr(ctx, JS_GetGlobalObject(ctx), "fetch",
                      JS_NewCFunction(ctx, js_sofuu_fetch, "fetch", 1));
}
