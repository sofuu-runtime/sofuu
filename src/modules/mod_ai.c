/*
 * mod_ai.c — Sofuu AI Module
 *
 * Provides:
 *   sofuu.ai.complete(prompt, options)  → Promise<{text}>
 *   sofuu.ai.stream(prompt, options)    → AsyncIterator<{text}>
 *
 * Supported providers: openai | anthropic | gemini | ollama
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <uv.h>
#include <curl/curl.h>
#include "quickjs.h"
#include "quickjs-libc.h"
#include "mod_ai.h"
#include "loop.h"
#include "promises.h"
#include "simd.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* ------------------------------------------------------------------ */
/* Providers                                                            */
/* ------------------------------------------------------------------ */

typedef enum {
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_OLLAMA
} sofuu_provider_t;

static sofuu_provider_t parse_provider(const char *name) {
    if (!name)                           return PROVIDER_OPENAI;
    if (strcmp(name, "anthropic") == 0)  return PROVIDER_ANTHROPIC;
    if (strcmp(name, "gemini")    == 0)  return PROVIDER_GEMINI;
    if (strcmp(name, "ollama")    == 0)  return PROVIDER_OLLAMA;
    return PROVIDER_OPENAI;
}

static const char *provider_default_model(sofuu_provider_t p) {
    switch (p) {
        case PROVIDER_ANTHROPIC: return "claude-3-5-sonnet-20241022";
        case PROVIDER_GEMINI:    return "gemini-1.5-pro";
        case PROVIDER_OLLAMA:    return "llama3";
        default:                 return "gpt-4o";
    }
}

static const char *provider_env_key(sofuu_provider_t p) {
    switch (p) {
        case PROVIDER_ANTHROPIC: return "ANTHROPIC_API_KEY";
        case PROVIDER_GEMINI:    return "GEMINI_API_KEY";
        case PROVIDER_OLLAMA:    return NULL;
        default:                 return "OPENAI_API_KEY";
    }
}

static const char *provider_api_url(sofuu_provider_t p, const char *model) {
    static char gbuf[256];
    switch (p) {
        case PROVIDER_ANTHROPIC: return "https://api.anthropic.com/v1/messages";
        case PROVIDER_OLLAMA:    return "http://localhost:11434/api/chat";
        case PROVIDER_GEMINI:
            snprintf(gbuf, sizeof(gbuf),
                "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent",
                model ? model : "gemini-1.5-pro");
            return gbuf;
        default:                 return "https://api.openai.com/v1/chat/completions";
    }
}

/* ------------------------------------------------------------------ */
/* Request body builders (returns heap string — caller must free)       */
/* ------------------------------------------------------------------ */

/* Simple JSON string escaping — replaces " and \ and newlines */
static char *json_escape(const char *s) {
    size_t len = strlen(s);
    char *out = malloc(len * 6 + 1);
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if      (c == '"')  { out[j++]='\\'; out[j++]='"';  }
        else if (c == '\\') { out[j++]='\\'; out[j++]='\\'; }
        else if (c == '\n') { out[j++]='\\'; out[j++]='n';  }
        else if (c == '\r') { out[j++]='\\'; out[j++]='r';  }
        else if (c == '\t') { out[j++]='\\'; out[j++]='t';  }
        else                { out[j++] = c;                  }
    }
    out[j] = '\0';
    return out;
}

static char *build_openai_body(const char *model, const char *prompt,
                               const char *sys, int stream) {
    char *ep = json_escape(prompt);
    char *es = json_escape(sys ? sys : "You are a helpful assistant.");
    size_t cap = strlen(ep) + strlen(es) + 256;
    char *body = malloc(cap);
    snprintf(body, cap,
        "{\"model\":\"%s\",\"stream\":%s,"
        "\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},"
                      "{\"role\":\"user\",\"content\":\"%s\"}]}",
        model, stream ? "true" : "false", es, ep);
    free(ep); free(es);
    return body;
}

static char *build_anthropic_body(const char *model, const char *prompt,
                                  const char *sys, int stream) {
    char *ep = json_escape(prompt);
    char *es = json_escape(sys ? sys : "You are a helpful assistant.");
    size_t cap = strlen(ep) + strlen(es) + 256;
    char *body = malloc(cap);
    snprintf(body, cap,
        "{\"model\":\"%s\",\"max_tokens\":4096,\"stream\":%s,"
        "\"system\":\"%s\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
        model, stream ? "true" : "false", es, ep);
    free(ep); free(es);
    return body;
}

static char *build_ollama_body(const char *model, const char *prompt,
                               const char *sys, int stream) {
    char *ep = json_escape(prompt);
    char *es = json_escape(sys ? sys : "You are a helpful assistant.");
    size_t cap = strlen(ep) + strlen(es) + 256;
    char *body = malloc(cap);
    snprintf(body, cap,
        "{\"model\":\"%s\",\"stream\":%s,"
        "\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},"
                      "{\"role\":\"user\",\"content\":\"%s\"}]}",
        model, stream ? "true" : "false", es, ep);
    free(ep); free(es);
    return body;
}

static char *build_request_body(sofuu_provider_t p, const char *model,
                                const char *prompt, const char *sys, int stream) {
    switch (p) {
        case PROVIDER_ANTHROPIC: return build_anthropic_body(model, prompt, sys, stream);
        case PROVIDER_OLLAMA:    return build_ollama_body(model, prompt, sys, stream);
        default:                 return build_openai_body(model, prompt, sys, stream);
    }
}

static struct curl_slist *build_headers(sofuu_provider_t p, const char *api_key) {
    struct curl_slist *h = NULL;
    h = curl_slist_append(h, "Content-Type: application/json");
    if (api_key) {
        char buf[512];
        if (p == PROVIDER_ANTHROPIC) {
            snprintf(buf, sizeof(buf), "x-api-key: %s", api_key);
            h = curl_slist_append(h, buf);
            h = curl_slist_append(h, "anthropic-version: 2023-06-01");
        } else {
            snprintf(buf, sizeof(buf), "Authorization: Bearer %s", api_key);
            h = curl_slist_append(h, buf);
        }
    }
    return h;
}

/* ------------------------------------------------------------------ */
/* Shared curl-multi / libuv bridge                                     */
/* ------------------------------------------------------------------ */

static CURLM      *g_multi       = NULL;
static uv_timer_t  g_timer;
static int         g_init_done   = 0;

/* Tag to distinguish complete vs stream requests stored in CURLINFO_PRIVATE */
#define REQ_TAG_COMPLETE 1
#define REQ_TAG_STREAM   2

typedef struct { int tag; } req_base_t;

/* forward declarations */
static void ai_check_multi(void);

typedef struct {
    uv_poll_t    poll;
    curl_socket_t sockfd;
} ai_poll_ctx_t;

static void ai_poll_close_cb(uv_handle_t *h) {
    free(h->data);
}

static void ai_poll_cb(uv_poll_t *req, int status, int events) {
    int running, flags = 0;
    if (events & UV_READABLE) flags |= CURL_CSELECT_IN;
    if (events & UV_WRITABLE) flags |= CURL_CSELECT_OUT;
    ai_poll_ctx_t *c = (ai_poll_ctx_t*)req->data;
    curl_multi_socket_action(g_multi, c->sockfd, flags, &running);
    ai_check_multi();
}

static void ai_timer_cb(uv_timer_t *t) {
    (void)t;
    int running;
    curl_multi_socket_action(g_multi, CURL_SOCKET_TIMEOUT, 0, &running);
    ai_check_multi();
}

static int ai_socket_fn(CURL *e, curl_socket_t s, int action,
                        void *userp, void *socketp) {
    (void)e; (void)userp;
    ai_poll_ctx_t *c;
    if (action == CURL_POLL_IN || action == CURL_POLL_OUT || action == CURL_POLL_INOUT) {
        if (socketp) {
            c = (ai_poll_ctx_t*)socketp;
        } else {
            c = malloc(sizeof(*c));
            c->sockfd = s;
            c->poll.data = c;
            uv_poll_init_socket(sofuu_loop_get(), &c->poll, s);
            curl_multi_assign(g_multi, s, c);
        }
        int ev = 0;
        if (action != CURL_POLL_IN)  ev |= UV_WRITABLE;
        if (action != CURL_POLL_OUT) ev |= UV_READABLE;
        uv_poll_start(&c->poll, ev, ai_poll_cb);
    } else {
        if (socketp) {
            c = (ai_poll_ctx_t*)socketp;
            uv_poll_stop(&c->poll);
            c->poll.data = c;
            uv_close((uv_handle_t*)&c->poll, ai_poll_close_cb);
            curl_multi_assign(g_multi, s, NULL);
        }
    }
    return 0;
}

static int ai_timer_fn(CURLM *m, long ms, void *userp) {
    (void)m; (void)userp;
    if (ms < 0) {
        uv_timer_stop(&g_timer);
    } else {
        uv_timer_start(&g_timer, ai_timer_cb, ms == 0 ? 1 : ms, 0);
    }
    return 0;
}

static void ai_ensure_init(void) {
    if (g_init_done) return;
    curl_global_init(CURL_GLOBAL_ALL);
    uv_timer_init(sofuu_loop_get(), &g_timer);
    g_multi = curl_multi_init();
    curl_multi_setopt(g_multi, CURLMOPT_SOCKETFUNCTION, ai_socket_fn);
    curl_multi_setopt(g_multi, CURLMOPT_TIMERFUNCTION,  ai_timer_fn);
    g_init_done = 1;
}

/* ------------------------------------------------------------------ */
/* ai.complete() request                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    req_base_t        base;       /* must be first — tag = REQ_TAG_COMPLETE */
    JSContext        *ctx;
    sofuu_promise_t  *promise;
    sofuu_provider_t  provider;
    char             *response_body;
    size_t            response_len;
    struct curl_slist *headers;
    char             *post_body;
    CURL             *easy;
} ai_complete_req_t;

static size_t complete_write_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    size_t n = size * nmemb;
    ai_complete_req_t *req = (ai_complete_req_t*)ud;
    req->response_body = realloc(req->response_body, req->response_len + n + 1);
    memcpy(req->response_body + req->response_len, ptr, n);
    req->response_len += n;
    req->response_body[req->response_len] = '\0';
    return n;
}

/* Minimal "find first string field" extractor — no full JSON parser needed */
static char *extract_text(sofuu_provider_t p, const char *body) {
    const char *needle;
    if (p == PROVIDER_ANTHROPIC)
        needle = strstr(body, "\"text\":");
    else
        needle = strstr(body, "\"content\":");

    if (!needle) return strdup("");
    needle = strchr(needle, ':');
    if (!needle) return strdup("");
    needle++;
    while (*needle == ' ' || *needle == '\n') needle++;
    if (*needle != '"') return strdup("");
    needle++;

    size_t cap = 4096;
    char *out = malloc(cap);
    size_t len = 0;
    while (*needle && *needle != '"') {
        if (*needle == '\\') {
            needle++;
            if      (*needle == 'n')  { if (len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='\n'; needle++; continue; }
            else if (*needle == '"')  { if (len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='"';  needle++; continue; }
            else if (*needle == '\\') { if (len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='\\'; needle++; continue; }
            continue;
        }
        if (len+1>=cap){cap*=2;out=realloc(out,cap);}
        out[len++] = *needle++;
    }
    out[len] = '\0';
    return out;
}

/* ------------------------------------------------------------------ */
/* ai.stream() request                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    req_base_t        base;       /* must be first — tag = REQ_TAG_STREAM */
    JSContext        *ctx;
    sofuu_provider_t  provider;
    JSValue           push_fn;
    JSValue           done_fn;
    JSValue           error_fn;
    char             *sse_buf;
    size_t            sse_len;
    struct curl_slist *headers;
    char             *post_body;
    CURL             *easy;
} ai_stream_req_t;

static char *extract_stream_delta(sofuu_provider_t p, const char *data) {
    if (strcmp(data, "[DONE]") == 0) return NULL;
    const char *needle;
    if (p == PROVIDER_ANTHROPIC)
        needle = strstr(data, "\"text\":");
    else
        needle = strstr(data, "\"content\":");
    if (!needle) return NULL;
    needle = strchr(needle, ':');
    if (!needle) return NULL;
    needle++;
    while (*needle == ' ') needle++;
    if (*needle != '"') return NULL;
    needle++;

    size_t cap = 512;
    char *out = malloc(cap);
    size_t len = 0;
    while (*needle && *needle != '"') {
        if (*needle == '\\') {
            needle++;
            if      (*needle == 'n')  { if(len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='\n'; needle++; continue; }
            else if (*needle == '"')  { if(len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='"';  needle++; continue; }
            else if (*needle == '\\') { if(len+1>=cap){cap*=2;out=realloc(out,cap);} out[len++]='\\'; needle++; continue; }
            continue;
        }
        if(len+1>=cap){cap*=2;out=realloc(out,cap);}
        out[len++] = *needle++;
    }
    out[len] = '\0';
    if (len == 0) { free(out); return NULL; }
    return out;
}

static size_t stream_write_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    size_t n = size * nmemb;
    ai_stream_req_t *req = (ai_stream_req_t*)ud;
    JSContext *ctx = req->ctx;

    req->sse_buf = realloc(req->sse_buf, req->sse_len + n + 1);
    memcpy(req->sse_buf + req->sse_len, ptr, n);
    req->sse_len += n;
    req->sse_buf[req->sse_len] = '\0';

    char *head = req->sse_buf;
    char *end  = req->sse_buf + req->sse_len;

    while (head < end) {
        char *nl = (char*)memchr(head, '\n', end - head);
        if (!nl) break;
        *nl = '\0';

        char *line = head;
        size_t llen = nl - head;
        if (llen > 0 && line[llen-1] == '\r') line[--llen] = '\0';

        if (strncmp(line, "data: ", 6) == 0) {
            const char *data = line + 6;

            if (strcmp(data, "[DONE]") == 0) {
                /* Stream finished — signal done */
                if (JS_IsFunction(ctx, req->done_fn)) {
                    JSValue ret = JS_Call(ctx, req->done_fn, JS_UNDEFINED, 0, NULL);
                    if (JS_IsException(ret)) js_std_dump_error(ctx);
                    JS_FreeValue(ctx, ret);
                }
                sofuu_flush_jobs(ctx);
                head = nl + 1;
                continue;
            }

            char *delta = extract_stream_delta(req->provider, data);
            if (delta) {
                if (JS_IsFunction(ctx, req->push_fn)) {
                    JSValue jd  = JS_NewString(ctx, delta);
                    JSValue ret = JS_Call(ctx, req->push_fn, JS_UNDEFINED, 1, &jd);
                    if (JS_IsException(ret)) js_std_dump_error(ctx);
                    JS_FreeValue(ctx, ret);
                    JS_FreeValue(ctx, jd);
                }
                sofuu_flush_jobs(ctx);
                free(delta);
            }
        }

        head = nl + 1;
    }

    /* Slide remaining bytes to front */
    size_t consumed = head - req->sse_buf;
    if (consumed > 0) {
        size_t rem = req->sse_len - consumed;
        memmove(req->sse_buf, head, rem);
        req->sse_len = rem;
        req->sse_buf[req->sse_len] = '\0';
    }

    return n;
}

/* ------------------------------------------------------------------ */
/* Shared multi-handle completion checker                               */
/* ------------------------------------------------------------------ */

static void ai_check_multi(void) {
    int pending;
    CURLMsg *msg;

    while ((msg = curl_multi_info_read(g_multi, &pending))) {
        if (msg->msg != CURLMSG_DONE) continue;

        CURL *easy     = msg->easy_handle;
        CURLcode code  = msg->data.result;

        req_base_t *base = NULL;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &base);

        if (!base) {
            curl_multi_remove_handle(g_multi, easy);
            curl_easy_cleanup(easy);
            continue;
        }

        if (base->tag == REQ_TAG_COMPLETE) {
            ai_complete_req_t *req = (ai_complete_req_t*)base;
            JSContext *ctx = req->ctx;

            if (code == CURLE_OK) {
                char *text = extract_text(req->provider,
                    req->response_body ? req->response_body : "");
                JSValue result = JS_NewObject(ctx);
                JS_SetPropertyStr(ctx, result, "text", JS_NewString(ctx, text));
                free(text);
                sofuu_promise_resolve(req->promise, result);
                JS_FreeValue(ctx, result);
            } else {
                JSValue err = JS_NewString(ctx, curl_easy_strerror(code));
                sofuu_promise_reject(req->promise, err);
                JS_FreeValue(ctx, err);
            }

            if (req->response_body) free(req->response_body);
            if (req->post_body)     free(req->post_body);
            if (req->headers)       curl_slist_free_all(req->headers);
            sofuu_flush_jobs(ctx);
            free(req);

        } else if (base->tag == REQ_TAG_STREAM) {
            ai_stream_req_t *req = (ai_stream_req_t*)base;
            JSContext *ctx = req->ctx;

            if (code != CURLE_OK) {
                if (JS_IsFunction(ctx, req->error_fn)) {
                    JSValue e = JS_NewString(ctx, curl_easy_strerror(code));
                    JSValue r = JS_Call(ctx, req->error_fn, JS_UNDEFINED, 1, &e);
                    JS_FreeValue(ctx, e); JS_FreeValue(ctx, r);
                }
            } else {
                /* Ensure done is called even if server omitted [DONE] */
                if (JS_IsFunction(ctx, req->done_fn)) {
                    JSValue r = JS_Call(ctx, req->done_fn, JS_UNDEFINED, 0, NULL);
                    if (JS_IsException(r)) js_std_dump_error(ctx);
                    JS_FreeValue(ctx, r);
                }
            }

            sofuu_flush_jobs(ctx);
            JS_FreeValue(ctx, req->push_fn);
            JS_FreeValue(ctx, req->done_fn);
            JS_FreeValue(ctx, req->error_fn);
            if (req->sse_buf)    free(req->sse_buf);
            if (req->post_body)  free(req->post_body);
            if (req->headers)    curl_slist_free_all(req->headers);
            free(req);
        }

        curl_multi_remove_handle(g_multi, easy);
        curl_easy_cleanup(easy);
    }
}

/* ------------------------------------------------------------------ */
/* JS: sofuu.ai.complete(prompt, opts?)  → Promise<{text}>             */
/* ------------------------------------------------------------------ */

static JSValue js_ai_complete(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    if (argc < 1) return JS_ThrowTypeError(ctx, "ai.complete: prompt required");

    const char *prompt = JS_ToCString(ctx, argv[0]);
    if (!prompt) return JS_EXCEPTION;

    const char *provider_str = NULL, *model_str = NULL, *system_str = NULL;

    if (argc > 1 && JS_IsObject(argv[1])) {
        JSValue v;
        v = JS_GetPropertyStr(ctx, argv[1], "provider");
        if (!JS_IsUndefined(v)) provider_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);

        v = JS_GetPropertyStr(ctx, argv[1], "model");
        if (!JS_IsUndefined(v)) model_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);

        v = JS_GetPropertyStr(ctx, argv[1], "system");
        if (!JS_IsUndefined(v)) system_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);
    }

    sofuu_provider_t prov  = parse_provider(provider_str);
    const char *model      = model_str  ? model_str  : provider_default_model(prov);
    const char *env_key    = provider_env_key(prov);
    const char *api_key    = env_key ? getenv(env_key) : NULL;
    const char *url        = provider_api_url(prov, model);

    ai_complete_req_t *req = calloc(1, sizeof(*req));
    req->base.tag  = REQ_TAG_COMPLETE;
    req->ctx       = ctx;
    req->provider  = prov;
    req->post_body = build_request_body(prov, model, prompt, system_str, 0);
    req->headers   = build_headers(prov, api_key);

    if (provider_str) JS_FreeCString(ctx, provider_str);
    if (model_str)    JS_FreeCString(ctx, model_str);
    if (system_str)   JS_FreeCString(ctx, system_str);
    JS_FreeCString(ctx, prompt);

    req->easy = curl_easy_init();
    curl_easy_setopt(req->easy, CURLOPT_URL,           url);
    curl_easy_setopt(req->easy, CURLOPT_POSTFIELDS,    req->post_body);
    curl_easy_setopt(req->easy, CURLOPT_POSTFIELDSIZE, (long)strlen(req->post_body));
    curl_easy_setopt(req->easy, CURLOPT_HTTPHEADER,    req->headers);
    curl_easy_setopt(req->easy, CURLOPT_WRITEFUNCTION, complete_write_cb);
    curl_easy_setopt(req->easy, CURLOPT_WRITEDATA,     req);
    curl_easy_setopt(req->easy, CURLOPT_PRIVATE,       req);
    curl_easy_setopt(req->easy, CURLOPT_TIMEOUT,       60L);

    JSValue promise = sofuu_promise_new(ctx, &req->promise);

    ai_ensure_init();
    curl_multi_add_handle(g_multi, req->easy);

    int running;
    curl_multi_socket_action(g_multi, CURL_SOCKET_TIMEOUT, 0, &running);

    return promise;
}

/* ------------------------------------------------------------------ */
/* JS: sofuu.ai.stream(prompt, opts?)  → AsyncIterator<{text}>         */
/*                                                                      */
/* Returns an object with [Symbol.asyncIterator] and .next().          */
/* A tiny JS factory manages the queue; C calls push/done/error.       */
/* ------------------------------------------------------------------ */

static const char *STREAM_FACTORY =
    "(function() {"
    "  var queue = [];"
    "  var _resolve = null, _done = false, _error = null;"
    "  var iterator = {"
    "    [Symbol.asyncIterator]: function() { return this; },"
    "    next: function() {"
    "      if (queue.length > 0) {"
    "        return Promise.resolve({ value: { text: queue.shift() }, done: false });"
    "      }"
    "      if (_done)  return Promise.resolve({ value: undefined, done: true });"
    "      if (_error) return Promise.reject(_error);"
    "      return new Promise(function(res, rej) {"
    "        _resolve = { res: res, rej: rej };"
    "      });"
    "    }"
    "  };"
    "  function push(text) {"
    "    if (_resolve) { var r = _resolve; _resolve = null; r.res({ value: { text: text }, done: false }); }"
    "    else { queue.push(text); }"
    "  }"
    "  function done() {"
    "    _done = true;"
    "    if (_resolve) { var r = _resolve; _resolve = null; r.res({ value: undefined, done: true }); }"
    "  }"
    "  function error(msg) {"
    "    _error = new Error(msg);"
    "    if (_resolve) { var r = _resolve; _resolve = null; r.rej(_error); }"
    "  }"
    "  return { push: push, done: done, error: error, iterator: iterator };"
    "})()";

static JSValue js_ai_stream(JSContext *ctx, JSValueConst this_val,
                            int argc, JSValueConst *argv) {
    if (argc < 1) return JS_ThrowTypeError(ctx, "ai.stream: prompt required");

    const char *prompt = JS_ToCString(ctx, argv[0]);
    if (!prompt) return JS_EXCEPTION;

    const char *provider_str = NULL, *model_str = NULL, *system_str = NULL;

    if (argc > 1 && JS_IsObject(argv[1])) {
        JSValue v;
        v = JS_GetPropertyStr(ctx, argv[1], "provider");
        if (!JS_IsUndefined(v)) provider_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);

        v = JS_GetPropertyStr(ctx, argv[1], "model");
        if (!JS_IsUndefined(v)) model_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);

        v = JS_GetPropertyStr(ctx, argv[1], "system");
        if (!JS_IsUndefined(v)) system_str = JS_ToCString(ctx, v);
        JS_FreeValue(ctx, v);
    }

    sofuu_provider_t prov = parse_provider(provider_str);
    const char *model     = model_str  ? model_str  : provider_default_model(prov);
    const char *env_key   = provider_env_key(prov);
    const char *api_key   = env_key ? getenv(env_key) : NULL;
    const char *url       = provider_api_url(prov, model);

    /* Create the JS-side async queue */
    JSValue factory = JS_Eval(ctx, STREAM_FACTORY, strlen(STREAM_FACTORY),
                              "<ai.stream>", JS_EVAL_TYPE_GLOBAL);
    if (JS_IsException(factory)) {
        JS_FreeCString(ctx, prompt);
        return JS_EXCEPTION;
    }

    JSValue push_fn  = JS_GetPropertyStr(ctx, factory, "push");
    JSValue done_fn  = JS_GetPropertyStr(ctx, factory, "done");
    JSValue error_fn = JS_GetPropertyStr(ctx, factory, "error");
    JSValue iterator = JS_GetPropertyStr(ctx, factory, "iterator");
    JS_FreeValue(ctx, factory);

    ai_stream_req_t *req = calloc(1, sizeof(*req));
    req->base.tag  = REQ_TAG_STREAM;
    req->ctx       = ctx;
    req->provider  = prov;
    req->push_fn   = JS_DupValue(ctx, push_fn);
    req->done_fn   = JS_DupValue(ctx, done_fn);
    req->error_fn  = JS_DupValue(ctx, error_fn);
    req->post_body = build_request_body(prov, model, prompt, system_str, 1);
    req->headers   = build_headers(prov, api_key);

    JS_FreeValue(ctx, push_fn);
    JS_FreeValue(ctx, done_fn);
    JS_FreeValue(ctx, error_fn);

    if (provider_str) JS_FreeCString(ctx, provider_str);
    if (model_str)    JS_FreeCString(ctx, model_str);
    if (system_str)   JS_FreeCString(ctx, system_str);
    JS_FreeCString(ctx, prompt);

    req->easy = curl_easy_init();
    curl_easy_setopt(req->easy, CURLOPT_URL,           url);
    curl_easy_setopt(req->easy, CURLOPT_POSTFIELDS,    req->post_body);
    curl_easy_setopt(req->easy, CURLOPT_POSTFIELDSIZE, (long)strlen(req->post_body));
    curl_easy_setopt(req->easy, CURLOPT_HTTPHEADER,    req->headers);
    curl_easy_setopt(req->easy, CURLOPT_WRITEFUNCTION, stream_write_cb);
    curl_easy_setopt(req->easy, CURLOPT_WRITEDATA,     req);
    curl_easy_setopt(req->easy, CURLOPT_PRIVATE,       req);
    curl_easy_setopt(req->easy, CURLOPT_TIMEOUT,       120L);

    ai_ensure_init();
    curl_multi_add_handle(g_multi, req->easy);

    int running;
    curl_multi_socket_action(g_multi, CURL_SOCKET_TIMEOUT, 0, &running);

    return iterator; /* caller does: for await (const {text} of ai.stream(...)) */
}

/* ------------------------------------------------------------------ */
/* SIMD Vector JS Bindings                                              */
/* ------------------------------------------------------------------ */

static float *extract_f32(JSContext *ctx, JSValueConst val, size_t *out_len) {
    size_t byte_offset, byte_length;
    JSValue buf = JS_GetTypedArrayBuffer(ctx, val, &byte_offset, &byte_length, NULL);
    if (!JS_IsException(buf)) {
        size_t n = byte_length / sizeof(float);
        uint8_t *ptr;
        size_t buf_size;
        ptr = JS_GetArrayBuffer(ctx, &buf_size, buf);
        JS_FreeValue(ctx, buf);
        if (!ptr) return NULL;
        float *out = (float *)malloc(n * sizeof(float));
        if (!out) return NULL;
        memcpy(out, ptr + byte_offset, n * sizeof(float));
        *out_len = n;
        return out;
    }
    JS_FreeValue(ctx, buf);
    JSValue len_v = JS_GetPropertyStr(ctx, val, "length");
    uint32_t len32;
    JS_ToUint32(ctx, &len32, len_v);
    JS_FreeValue(ctx, len_v);
    float *out = (float *)malloc(len32 * sizeof(float));
    if (!out) return NULL;
    for (uint32_t i = 0; i < len32; i++) {
        JSValue el = JS_GetPropertyUint32(ctx, val, i);
        double d; JS_ToFloat64(ctx, &d, el);
        JS_FreeValue(ctx, el);
        out[i] = (float)d;
    }
    *out_len = len32;
    return out;
}

static JSValue js_ai_similarity(JSContext *ctx, JSValueConst this_val,
                                int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2) return JS_ThrowTypeError(ctx, "similarity(a,b) requires 2 vectors");
    size_t na, nb;
    float *a = extract_f32(ctx, argv[0], &na);
    float *b = extract_f32(ctx, argv[1], &nb);
    if (!a || !b || na != nb) { free(a); free(b); return JS_ThrowTypeError(ctx, "vectors must be same length"); }
    float r = sofuu_cosine_f32(a, b, na);
    free(a); free(b);
    return JS_NewFloat64(ctx, r);
}

static JSValue js_ai_dot(JSContext *ctx, JSValueConst this_val,
                         int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2) return JS_ThrowTypeError(ctx, "dot(a,b) requires 2 vectors");
    size_t na, nb;
    float *a = extract_f32(ctx, argv[0], &na);
    float *b = extract_f32(ctx, argv[1], &nb);
    if (!a || !b || na != nb) { free(a); free(b); return JS_ThrowTypeError(ctx, "same length required"); }
    float r = sofuu_dot_f32(a, b, na);
    free(a); free(b);
    return JS_NewFloat64(ctx, r);
}

static JSValue js_ai_l2(JSContext *ctx, JSValueConst this_val,
                        int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 2) return JS_ThrowTypeError(ctx, "l2(a,b) requires 2 vectors");
    size_t na, nb;
    float *a = extract_f32(ctx, argv[0], &na);
    float *b = extract_f32(ctx, argv[1], &nb);
    if (!a || !b || na != nb) { free(a); free(b); return JS_ThrowTypeError(ctx, "same length required"); }
    float r = sofuu_l2_f32(a, b, na);
    free(a); free(b);
    return JS_NewFloat64(ctx, r);
}

/* ------------------------------------------------------------------ */
/* Module registration                                                   */
/* ------------------------------------------------------------------ */

static const JSCFunctionListEntry ai_funcs[] = {
    JS_CFUNC_DEF("complete",   2, js_ai_complete),
    JS_CFUNC_DEF("stream",     2, js_ai_stream),
    JS_CFUNC_DEF("similarity", 2, js_ai_similarity),
    JS_CFUNC_DEF("dot",        2, js_ai_dot),
    JS_CFUNC_DEF("l2",         2, js_ai_l2),
};

void mod_ai_register(JSContext *ctx) {
    ai_ensure_init();

    JSValue global = JS_GetGlobalObject(ctx);
    JSValue sofuu  = JS_GetPropertyStr(ctx, global, "sofuu");

    if (JS_IsUndefined(sofuu)) {
        sofuu = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global, "sofuu", JS_DupValue(ctx, sofuu));
    }

    JSValue ai_obj = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, ai_obj, ai_funcs, countof(ai_funcs));
    JS_SetPropertyStr(ctx, sofuu, "ai", ai_obj);

    JS_FreeValue(ctx, sofuu);
    JS_FreeValue(ctx, global);
}
