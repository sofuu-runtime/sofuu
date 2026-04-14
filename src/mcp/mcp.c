/*
 * mcp.c — Sofuu MCP (Model Context Protocol) Module
 *
 * Provides:
 *   sofuu.mcp.connect(command)            → Promise<MCPClient>
 *   client.call(method, params)           → Promise<result>
 *   client.listTools()                    → Promise<Tool[]>
 *   client.disconnect()                   → void
 *
 *   sofuu.mcp.serve(opts)                 → MCPServer
 *   server.tool(name, schema, handler)    → void
 *   server.start()                        → void  (stdio transport)
 *
 * Architecture:
 *   MCP client: spawns child process (e.g. "npx @mcp/server-filesystem /tmp")
 *               communicates over stdin/stdout using JSON-RPC 2.0
 *               uses same libuv subprocess infrastructure as subprocess.c
 *
 *   MCP server: reads JSON-RPC from process.stdin, routes to tools,
 *               writes responses to process.stdout
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <uv.h>
#include "quickjs.h"
#include "quickjs-libc.h"
#include "mcp.h"
#include "loop.h"
#include "promises.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* ------------------------------------------------------------------ */
/* JSON-RPC 2.0 helpers                                                 */
/* ------------------------------------------------------------------ */

static int g_next_id = 1;

/* Build a JSON-RPC request string. Caller must free(). */
static char *jsonrpc_request(int id, const char *method, const char *params_json) {
    size_t cap = strlen(method) + (params_json ? strlen(params_json) : 2) + 128;
    char *buf = malloc(cap);
    snprintf(buf, cap,
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"%s\",\"params\":%s}\n",
        id, method, params_json ? params_json : "{}");
    return buf;
}

/* Build a JSON-RPC notification (no id). Caller must free(). */
static char *jsonrpc_notify(const char *method, const char *params_json) {
    size_t cap = strlen(method) + (params_json ? strlen(params_json) : 2) + 128;
    char *buf = malloc(cap);
    snprintf(buf, cap,
        "{\"jsonrpc\":\"2.0\",\"method\":\"%s\",\"params\":%s}\n",
        method, params_json ? params_json : "{}");
    return buf;
}

/* Build a JSON-RPC success response. Caller must free(). */
static char *jsonrpc_response(int id, const char *result_json) {
    size_t cap = (result_json ? strlen(result_json) : 4) + 64;
    char *buf = malloc(cap);
    snprintf(buf, cap,
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"result\":%s}\n",
        id, result_json ? result_json : "null");
    return buf;
}

/* Build a JSON-RPC error response. Caller must free(). */
static char *jsonrpc_error(int id, int code, const char *message) {
    size_t cap = strlen(message) + 128;
    char *buf = malloc(cap);
    snprintf(buf, cap,
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"error\":{\"code\":%d,\"message\":\"%s\"}}\n",
        id, code, message);
    return buf;
}

/* Extract JSON field value as a newly malloc'd string (very basic) */
static char *json_get_field(const char *json, const char *key) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ') p++;

    if (*p == '"') {
        /* String value */
        p++;
        const char *start = p;
        size_t len = 0;
        char *out = malloc(4096);
        while (*p && *p != '"') {
            if (*p == '\\') { p++; }
            out[len++] = *p++;
        }
        out[len] = '\0';
        (void)start;
        return out;
    } else if (*p == '{' || *p == '[') {
        /* Object or array — find matching close */
        char open = *p, close = (*p == '{') ? '}' : ']';
        int depth = 0;
        const char *start = p;
        while (*p) {
            if (*p == open)  depth++;
            if (*p == close) { depth--; if (depth == 0) { p++; break; } }
            p++;
        }
        size_t len = p - start;
        char *out = malloc(len + 1);
        memcpy(out, start, len);
        out[len] = '\0';
        return out;
    } else {
        /* Number or boolean — read until delimiter */
        const char *start = p;
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != '\n') p++;
        size_t len = p - start;
        char *out = malloc(len + 1);
        memcpy(out, start, len);
        out[len] = '\0';
        return out;
    }
}

/* ------------------------------------------------------------------ */
/* Pending call tracking                                                 */
/* ------------------------------------------------------------------ */

#define MAX_PENDING 64

typedef struct {
    int              id;
    sofuu_promise_t *promise;
    JSContext       *ctx;
} pending_call_t;

/* ------------------------------------------------------------------ */
/* MCP Client (connects to an MCP server subprocess)                    */
/* ------------------------------------------------------------------ */

#define MCP_READ_BUF  65536

static JSClassID mcp_client_class_id;

typedef struct {
    uv_process_t     process;
    uv_pipe_t        stdin_pipe;
    uv_pipe_t        stdout_pipe;
    uv_pipe_t        stderr_pipe;

    JSContext       *ctx;
    JSValue          self;          /* the JS MCPClient object */

    char             read_buf[MCP_READ_BUF];
    size_t           read_len;

    pending_call_t   pending[MAX_PENDING];
    int              pending_count;

    int              initialized;   /* whether initialize handshake done */
    sofuu_promise_t *connect_promise;
} mcp_client_t;

static void mcp_client_finalizer(JSRuntime *rt, JSValue val) {
    mcp_client_t *client = JS_GetOpaque(val, mcp_client_class_id);
    if (client) {
        free(client);
    }
}

static void on_mcp_write_done(uv_write_t *req, int status) {
    (void)status;
    free(req->data);
    free(req);
}

static void mcp_client_send(mcp_client_t *client, const char *msg) {
    size_t len = strlen(msg);
    char *copy = strdup(msg);
    uv_buf_t buf = uv_buf_init(copy, len);
    uv_write_t *req = malloc(sizeof(uv_write_t));
    req->data = copy;
    uv_write(req, (uv_stream_t*)&client->stdin_pipe, &buf, 1, on_mcp_write_done);
}

/* Handle a complete JSON-RPC line from the server */
static void mcp_process_line(mcp_client_t *client, const char *line) {
    JSContext *ctx = client->ctx;

    char *id_str = json_get_field(line, "id");
    int   msg_id = id_str ? atoi(id_str) : -1;
    free(id_str);

    /* Check if this is a response to a pending call */
    for (int i = 0; i < client->pending_count; i++) {
        if (client->pending[i].id != msg_id) continue;

        sofuu_promise_t *promise = client->pending[i].promise;

        char *result = json_get_field(line, "result");
        char *error  = json_get_field(line, "error");

        if (result) {
            /* Parse result into a JS value */
            JSValue jresult = JS_ParseJSON(ctx, result, strlen(result), "<mcp>");
            if (JS_IsException(jresult)) jresult = JS_NewString(ctx, result);
            sofuu_promise_resolve(promise, jresult);
            JS_FreeValue(ctx, jresult);
            free(result);
        } else if (error) {
            char *msg = json_get_field(error, "message");
            JSValue jerr = JS_NewString(ctx, msg ? msg : "MCP error");
            sofuu_promise_reject(promise, jerr);
            JS_FreeValue(ctx, jerr);
            free(msg);
            free(error);
        } else {
            JSValue jnull = JS_NULL;
            sofuu_promise_resolve(promise, jnull);
        }

        sofuu_flush_jobs(ctx);

        /* Remove from pending */
        client->pending[i] = client->pending[--client->pending_count];

        /* If this was the initialize response, mark initialized */
        if (!client->initialized && client->connect_promise) {
            client->initialized = 1;
            /* Send initialized notification */
            char *notif = jsonrpc_notify("notifications/initialized", NULL);
            mcp_client_send(client, notif);
            free(notif);

            /* Resolve the connect promise with the client object itself */
            JSValue self = JS_DupValue(ctx, client->self);
            sofuu_promise_resolve(client->connect_promise, self);
            JS_FreeValue(ctx, self);
            sofuu_flush_jobs(ctx);
            client->connect_promise = NULL;
        }
        return;
    }
}

/* Called by libuv when data arrives from child stdout */
static void mcp_on_read(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf) {
    mcp_client_t *client = (mcp_client_t*)stream->data;

    if (nread > 0) {
        /* Append to line buffer */
        size_t space = MCP_READ_BUF - client->read_len - 1;
        size_t copy  = (size_t)nread < space ? (size_t)nread : space;
        memcpy(client->read_buf + client->read_len, buf->base, copy);
        client->read_len += copy;
        client->read_buf[client->read_len] = '\0';

        /* Process complete newline-delimited JSON lines */
        char *head = client->read_buf;
        char *nl;
        while ((nl = memchr(head, '\n', client->read_len - (head - client->read_buf)))) {
            *nl = '\0';
            if (nl > head && *(nl-1) == '\r') *(nl-1) = '\0';
            if (*head) mcp_process_line(client, head);
            head = nl + 1;
        }

        /* Slide remaining partial line to front */
        size_t remaining = client->read_len - (head - client->read_buf);
        memmove(client->read_buf, head, remaining);
        client->read_len = remaining;
        client->read_buf[remaining] = '\0';

    } else if (nread < 0 && nread != UV_EOF) {
        /* Connection broken */
        if (client->connect_promise) {
            JSContext *ctx = client->ctx;
            JSValue err = JS_NewString(ctx, "MCP server disconnected");
            sofuu_promise_reject(client->connect_promise, err);
            JS_FreeValue(ctx, err);
            sofuu_flush_jobs(ctx);
            client->connect_promise = NULL;
        }
    }

    if (buf->base) free(buf->base);
}

static void mcp_on_alloc(uv_handle_t *handle, size_t suggested, uv_buf_t *buf) {
    (void)handle;
    buf->base = malloc(suggested);
    buf->len  = suggested;
}

static void mcp_on_exit(uv_process_t *proc, int64_t exit_status, int term_signal) {
    (void)proc; (void)exit_status; (void)term_signal;
}

/* JS method: client.call(method, params?) → Promise<result> */
static JSValue js_mcp_call(JSContext *ctx, JSValueConst this_val,
                           int argc, JSValueConst *argv) {
    mcp_client_t *client = JS_GetOpaque(this_val, mcp_client_class_id);
    if (!client) return JS_ThrowTypeError(ctx, "Invalid MCPClient");
    if (argc < 1) return JS_ThrowTypeError(ctx, "call: method required");

    const char *method = JS_ToCString(ctx, argv[0]);
    if (!method) return JS_EXCEPTION;

    /* Serialize params to JSON */
    char *params_json = NULL;
    if (argc > 1 && !JS_IsUndefined(argv[1])) {
        JSValue json_str = JS_JSONStringify(ctx, argv[1], JS_UNDEFINED, JS_UNDEFINED);
        if (!JS_IsException(json_str)) {
            const char *s = JS_ToCString(ctx, json_str);
            if (s) { params_json = strdup(s); JS_FreeCString(ctx, s); }
            JS_FreeValue(ctx, json_str);
        }
    }

    int id = g_next_id++;
    char *msg = jsonrpc_request(id, method, params_json);
    free(params_json);

    /* Register pending promise */
    if (client->pending_count >= MAX_PENDING) {
        JS_FreeCString(ctx, method);
        free(msg);
        return JS_ThrowRangeError(ctx, "Too many pending MCP calls");
    }

    sofuu_promise_t *promise;
    JSValue ret = sofuu_promise_new(ctx, &promise);

    client->pending[client->pending_count].id      = id;
    client->pending[client->pending_count].promise  = promise;
    client->pending[client->pending_count].ctx      = ctx;
    client->pending_count++;

    mcp_client_send(client, msg);
    free(msg);
    JS_FreeCString(ctx, method);

    return ret;
}

/* JS method: client.listTools() → Promise<Tool[]> */
static JSValue js_mcp_list_tools(JSContext *ctx, JSValueConst this_val,
                                  int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    JSValue method = JS_NewString(ctx, "tools/list");
    JSValue result = js_mcp_call(ctx, this_val, 1, &method);
    JS_FreeValue(ctx, method);
    return result;
}

/* JS method: client.listResources() → Promise */
static JSValue js_mcp_list_resources(JSContext *ctx, JSValueConst this_val,
                                      int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    JSValue method = JS_NewString(ctx, "resources/list");
    JSValue result = js_mcp_call(ctx, this_val, 1, &method);
    JS_FreeValue(ctx, method);
    return result;
}

/* JS method: client.disconnect() */
static JSValue js_mcp_disconnect(JSContext *ctx, JSValueConst this_val,
                                  int argc, JSValueConst *argv) {
    (void)ctx; (void)argc; (void)argv;
    mcp_client_t *client = JS_GetOpaque(this_val, mcp_client_class_id);
    if (client) {
        uv_process_kill(&client->process, SIGTERM);
    }
    return JS_UNDEFINED;
}

static const JSCFunctionListEntry mcp_client_proto[] = {
    JS_CFUNC_DEF("call",           2, js_mcp_call),
    JS_CFUNC_DEF("listTools",      0, js_mcp_list_tools),
    JS_CFUNC_DEF("listResources",  0, js_mcp_list_resources),
    JS_CFUNC_DEF("disconnect",     0, js_mcp_disconnect),
};

static JSClassDef mcp_client_class = {
    "MCPClient",
    .finalizer = mcp_client_finalizer,
};

/* JS: sofuu.mcp.connect("npx @mcp/server-filesystem /tmp") → Promise<MCPClient> */
static JSValue js_mcp_connect(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1) return JS_ThrowTypeError(ctx, "mcp.connect: command required");

    const char *cmd_str = JS_ToCString(ctx, argv[0]);
    if (!cmd_str) return JS_EXCEPTION;

    /* Tokenize command into argv */
    char *cmd_copy = strdup(cmd_str);
    JS_FreeCString(ctx, cmd_str);

    char  *args_arr[64];
    int    args_count = 0;
    char  *tok = strtok(cmd_copy, " ");
    while (tok && args_count < 63) {
        args_arr[args_count++] = tok;
        tok = strtok(NULL, " ");
    }
    args_arr[args_count] = NULL;

    if (args_count == 0) {
        free(cmd_copy);
        return JS_ThrowTypeError(ctx, "mcp.connect: empty command");
    }

    /* Allocate client */
    mcp_client_t *client = calloc(1, sizeof(mcp_client_t));
    client->ctx = ctx;

    /* Create JS object for the client */
    client->self = JS_NewObjectClass(ctx, mcp_client_class_id);
    JS_SetOpaque(client->self, client);

    /* Setup pipes */
    uv_loop_t *loop = sofuu_loop_get();
    uv_pipe_init(loop, &client->stdin_pipe,  0);
    uv_pipe_init(loop, &client->stdout_pipe, 0);
    uv_pipe_init(loop, &client->stderr_pipe, 0);

    uv_stdio_container_t stdio[3];
    stdio[0].flags        = UV_CREATE_PIPE | UV_READABLE_PIPE;
    stdio[0].data.stream  = (uv_stream_t*)&client->stdin_pipe;
    stdio[1].flags        = UV_CREATE_PIPE | UV_WRITABLE_PIPE;
    stdio[1].data.stream  = (uv_stream_t*)&client->stdout_pipe;
    stdio[2].flags        = UV_IGNORE;

    uv_process_options_t opts = {0};
    opts.file       = args_arr[0];
    opts.args       = args_arr;
    opts.stdio      = stdio;
    opts.stdio_count = 3;
    opts.exit_cb    = mcp_on_exit;

    int r = uv_spawn(loop, &client->process, &opts);
    free(cmd_copy);

    if (r < 0) {
        JS_FreeValue(ctx, client->self);
        free(client);
        return JS_ThrowTypeError(ctx, "%s", uv_strerror(r));
    }

    /* Start reading stdout */
    client->stdout_pipe.data = client;
    uv_read_start((uv_stream_t*)&client->stdout_pipe, mcp_on_alloc, mcp_on_read);

    /* Create the connect promise */
    JSValue promise = sofuu_promise_new(ctx, &client->connect_promise);

    /* Send MCP initialize request */
    int init_id = g_next_id++;
    char *init_msg = jsonrpc_request(init_id,
        "initialize",
        "{"
          "\"protocolVersion\":\"2024-11-05\","
          "\"capabilities\":{},"
          "\"clientInfo\":{\"name\":\"sofuu\",\"version\":\"0.1.0\"}"
        "}");

    /* Register initialize as a pending call */
    client->pending[0].id      = init_id;
    client->pending[0].promise = NULL;  /* handled specially via connect_promise */
    client->pending[0].ctx     = ctx;
    client->pending_count      = 1;

    mcp_client_send(client, init_msg);
    free(init_msg);

    return promise;
}

/* ------------------------------------------------------------------ */
/* MCP Server (stdio transport — reads from stdin, writes to stdout)    */
/* ------------------------------------------------------------------ */

static JSClassID mcp_server_class_id;

#define MAX_TOOLS 64

typedef struct {
    char    name[128];
    char    description[512];
    char   *schema_json;     /* JSON string for input schema */
    JSValue handler;         /* JS async function */
} mcp_tool_t;

typedef struct {
    JSContext *ctx;
    JSValue    self;
    mcp_tool_t tools[MAX_TOOLS];
    int        tool_count;

    uv_pipe_t  stdin_pipe;
    uv_pipe_t  stdout_pipe;

    char       read_buf[MCP_READ_BUF];
    size_t     read_len;
} mcp_server_t;

static void mcp_server_finalizer(JSRuntime *rt, JSValue val) {
    (void)rt;
    mcp_server_t *srv = JS_GetOpaque(val, mcp_server_class_id);
    if (srv) {
        for (int i = 0; i < srv->tool_count; i++) {
            /* handlers were dup'd with JS_DupValue — free them using the runtime */
            JS_FreeValueRT(rt, srv->tools[i].handler);
            if (srv->tools[i].schema_json) free(srv->tools[i].schema_json);
        }
        free(srv);
    }
}

static void mcp_server_write_str(mcp_server_t *srv, const char *msg) {
    size_t len = strlen(msg);
    char  *copy = strdup(msg);
    uv_buf_t buf = uv_buf_init(copy, len);
    uv_write_t *req = malloc(sizeof(uv_write_t));
    req->data = copy;
    uv_write(req, (uv_stream_t*)&srv->stdout_pipe, &buf, 1, on_mcp_write_done);
}

/* Route an incoming RPC request to a registered tool */
static void mcp_server_handle_request(mcp_server_t *srv, const char *line) {
    JSContext *ctx = srv->ctx;

    char *method = json_get_field(line, "method");
    char *id_str = json_get_field(line, "id");
    int   req_id = id_str ? atoi(id_str) : 0;
    free(id_str);

    if (!method) { free(method); return; }

    /* --- initialize --- */
    if (strcmp(method, "initialize") == 0) {
        char *resp = jsonrpc_response(req_id,
            "{\"protocolVersion\":\"2024-11-05\","
             "\"capabilities\":{\"tools\":{}},"
             "\"serverInfo\":{\"name\":\"sofuu-mcp\",\"version\":\"0.1.0\"}}");
        mcp_server_write_str(srv, resp);
        free(resp);

    /* --- tools/list --- */
    } else if (strcmp(method, "tools/list") == 0) {
        /* Build tools array */
        char tools_json[16384];
        size_t p = 0;
        tools_json[p++] = '[';
        for (int i = 0; i < srv->tool_count; i++) {
            if (i > 0) tools_json[p++] = ',';
            p += snprintf(tools_json + p, sizeof(tools_json) - p,
                "{\"name\":\"%s\",\"description\":\"%s\","
                 "\"inputSchema\":%s}",
                srv->tools[i].name,
                srv->tools[i].description,
                srv->tools[i].schema_json ? srv->tools[i].schema_json : "{}");
        }
        tools_json[p++] = ']';
        tools_json[p]   = '\0';

        char result[16512];
        snprintf(result, sizeof(result), "{\"tools\":%s}", tools_json);
        char *resp = jsonrpc_response(req_id, result);
        mcp_server_write_str(srv, resp);
        free(resp);

    /* --- tools/call --- */
    } else if (strcmp(method, "tools/call") == 0) {
        char *params   = json_get_field(line, "params");
        char *toolname = params ? json_get_field(params, "name") : NULL;
        char *args_str = params ? json_get_field(params, "arguments") : NULL;

        /* Find the tool */
        mcp_tool_t *tool = NULL;
        for (int i = 0; i < srv->tool_count; i++) {
            if (toolname && strcmp(srv->tools[i].name, toolname) == 0) {
                tool = &srv->tools[i];
                break;
            }
        }

        if (!tool) {
            char *err = jsonrpc_error(req_id, -32601, "Tool not found");
            mcp_server_write_str(srv, err);
            free(err);
        } else {
            /* Parse args into JS object */
            JSValue js_args = JS_UNDEFINED;
            if (args_str) {
                js_args = JS_ParseJSON(ctx, args_str, strlen(args_str), "<mcp-args>");
                if (JS_IsException(js_args)) js_args = JS_NewObject(ctx);
            } else {
                js_args = JS_NewObject(ctx);
            }

            /* Call the JS handler */
            JSValue ret = JS_Call(ctx, tool->handler, JS_UNDEFINED, 1, &js_args);
            JS_FreeValue(ctx, js_args);

            /* Handler may return a Promise — resolve it */
            if (JS_IsException(ret)) {
                js_std_dump_error(ctx);
                char *err = jsonrpc_error(req_id, -32603, "Tool handler threw");
                mcp_server_write_str(srv, err);
                free(err);
            } else {
                /* For now, resolve synchronously (no async handlers yet) */
                const char *result_str = NULL;
                JSValue json_val = JS_JSONStringify(ctx, ret, JS_UNDEFINED, JS_UNDEFINED);
                if (!JS_IsException(json_val)) {
                    result_str = JS_ToCString(ctx, json_val);
                }
                char result_buf[4096];
                snprintf(result_buf, sizeof(result_buf),
                    "{\"content\":[{\"type\":\"text\",\"text\":%s}]}",
                    result_str ? result_str : "\"\"");
                char *resp = jsonrpc_response(req_id, result_buf);
                mcp_server_write_str(srv, resp);
                free(resp);
                if (result_str) JS_FreeCString(ctx, result_str);
                if (!JS_IsException(json_val)) JS_FreeValue(ctx, json_val);
                JS_FreeValue(ctx, ret);
            }
        }

        free(toolname);
        free(args_str);
        free(params);

    /* --- unknown method --- */
    } else {
        char *err = jsonrpc_error(req_id, -32601, "Method not found");
        mcp_server_write_str(srv, err);
        free(err);
    }

    free(method);
    sofuu_flush_jobs(ctx);
}

static void mcp_server_on_read(uv_stream_t *stream, ssize_t nread, const uv_buf_t *buf) {
    mcp_server_t *srv = (mcp_server_t*)stream->data;

    if (nread > 0) {
        size_t space = MCP_READ_BUF - srv->read_len - 1;
        size_t copy  = (size_t)nread < space ? (size_t)nread : space;
        memcpy(srv->read_buf + srv->read_len, buf->base, copy);
        srv->read_len += copy;
        srv->read_buf[srv->read_len] = '\0';

        char *head = srv->read_buf;
        char *nl;
        while ((nl = memchr(head, '\n', srv->read_len - (head - srv->read_buf)))) {
            *nl = '\0';
            if (nl > head && *(nl-1) == '\r') *(nl-1) = '\0';
            if (*head) mcp_server_handle_request(srv, head);
            head = nl + 1;
        }

        size_t remaining = srv->read_len - (head - srv->read_buf);
        memmove(srv->read_buf, head, remaining);
        srv->read_len = remaining;
        srv->read_buf[remaining] = '\0';
    }

    if (buf->base) free(buf->base);
}

static void mcp_server_on_alloc(uv_handle_t *h, size_t sz, uv_buf_t *buf) {
    (void)h;
    buf->base = malloc(sz);
    buf->len  = sz;
}

/* JS: server.tool(name, { description, schema }, handler) */
static JSValue js_server_tool(JSContext *ctx, JSValueConst this_val,
                              int argc, JSValueConst *argv) {
    mcp_server_t *srv = JS_GetOpaque(this_val, mcp_server_class_id);
    if (!srv) return JS_ThrowTypeError(ctx, "Invalid MCPServer");
    if (argc < 3) return JS_ThrowTypeError(ctx, "tool: name, opts, handler required");
    if (srv->tool_count >= MAX_TOOLS) return JS_ThrowRangeError(ctx, "Too many tools");

    const char *name = JS_ToCString(ctx, argv[0]);
    if (!name) return JS_EXCEPTION;

    mcp_tool_t *t = &srv->tools[srv->tool_count];
    strncpy(t->name, name, sizeof(t->name) - 1);
    JS_FreeCString(ctx, name);

    /* opts: { description, schema } */
    if (JS_IsObject(argv[1])) {
        JSValue desc = JS_GetPropertyStr(ctx, argv[1], "description");
        if (!JS_IsUndefined(desc)) {
            const char *d = JS_ToCString(ctx, desc);
            if (d) { strncpy(t->description, d, sizeof(t->description)-1); JS_FreeCString(ctx, d); }
        }
        JS_FreeValue(ctx, desc);

        JSValue schema = JS_GetPropertyStr(ctx, argv[1], "schema");
        if (!JS_IsUndefined(schema)) {
            JSValue json = JS_JSONStringify(ctx, schema, JS_UNDEFINED, JS_UNDEFINED);
            if (!JS_IsException(json)) {
                const char *s = JS_ToCString(ctx, json);
                if (s) { t->schema_json = strdup(s); JS_FreeCString(ctx, s); }
                JS_FreeValue(ctx, json);
            }
        }
        JS_FreeValue(ctx, schema);
    }

    if (!JS_IsFunction(ctx, argv[2])) {
        return JS_ThrowTypeError(ctx, "tool: handler must be a function");
    }
    t->handler = JS_DupValue(ctx, argv[2]);
    srv->tool_count++;

    return JS_UNDEFINED;
}

/* JS: server.start() — begins listening on stdio */
static JSValue js_server_start(JSContext *ctx, JSValueConst this_val,
                               int argc, JSValueConst *argv) {
    (void)argc; (void)argv;
    mcp_server_t *srv = JS_GetOpaque(this_val, mcp_server_class_id);
    if (!srv) return JS_ThrowTypeError(ctx, "Invalid MCPServer");

    uv_loop_t *loop = sofuu_loop_get();
    uv_pipe_init(loop, &srv->stdin_pipe,  0);
    uv_pipe_init(loop, &srv->stdout_pipe, 0);

    uv_pipe_open(&srv->stdin_pipe,  0); /* fd 0 = stdin  */
    uv_pipe_open(&srv->stdout_pipe, 1); /* fd 1 = stdout */

    srv->stdin_pipe.data = srv;
    uv_read_start((uv_stream_t*)&srv->stdin_pipe, mcp_server_on_alloc, mcp_server_on_read);

    fprintf(stderr, "[sofuu mcp] server started (%d tools registered)\n", srv->tool_count);
    return JS_UNDEFINED;
}

static const JSCFunctionListEntry mcp_server_proto[] = {
    JS_CFUNC_DEF("tool",  3, js_server_tool),
    JS_CFUNC_DEF("start", 0, js_server_start),
};

static JSClassDef mcp_server_class = {
    "MCPServer",
    .finalizer = mcp_server_finalizer,
};

/* JS: sofuu.mcp.serve(opts?) → MCPServer */
static JSValue js_mcp_serve(JSContext *ctx, JSValueConst this_val,
                            int argc, JSValueConst *argv) {
    (void)this_val; (void)argc; (void)argv;

    mcp_server_t *srv = calloc(1, sizeof(mcp_server_t));
    srv->ctx = ctx;

    JSValue obj = JS_NewObjectClass(ctx, mcp_server_class_id);
    JS_SetOpaque(obj, srv);
    srv->self = obj;

    return obj;
}

/* ------------------------------------------------------------------ */
/* Module registration                                                   */
/* ------------------------------------------------------------------ */

static const JSCFunctionListEntry mcp_funcs[] = {
    JS_CFUNC_DEF("connect", 1, js_mcp_connect),
    JS_CFUNC_DEF("serve",   1, js_mcp_serve),
};

void mod_mcp_register(JSContext *ctx) {
    /* Register MCPClient class */
    JS_NewClassID(&mcp_client_class_id);
    JS_NewClass(JS_GetRuntime(ctx), mcp_client_class_id, &mcp_client_class);
    JSValue client_proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, client_proto, mcp_client_proto, countof(mcp_client_proto));
    JS_SetClassProto(ctx, mcp_client_class_id, client_proto);

    /* Register MCPServer class */
    JS_NewClassID(&mcp_server_class_id);
    JS_NewClass(JS_GetRuntime(ctx), mcp_server_class_id, &mcp_server_class);
    JSValue server_proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, server_proto, mcp_server_proto, countof(mcp_server_proto));
    JS_SetClassProto(ctx, mcp_server_class_id, server_proto);

    /* Attach mcp object to sofuu global */
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue sofuu  = JS_GetPropertyStr(ctx, global, "sofuu");

    if (JS_IsUndefined(sofuu)) {
        sofuu = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global, "sofuu", JS_DupValue(ctx, sofuu));
    }

    JSValue mcp_obj = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, mcp_obj, mcp_funcs, countof(mcp_funcs));
    JS_SetPropertyStr(ctx, sofuu, "mcp", mcp_obj);

    JS_FreeValue(ctx, sofuu);
    JS_FreeValue(ctx, global);
}
