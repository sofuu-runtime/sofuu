#include "subprocess.h"
#include "loop.h"
#include "promises.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

typedef struct {
    JSContext *ctx;
    uv_process_t process;
    uv_process_options_t options;
    uv_pipe_t stdin_pipe;
    uv_pipe_t stdout_pipe;
    uv_pipe_t stderr_pipe;
    uv_stdio_container_t stdio[3];
    
    JSValue onStdout;
    JSValue onStderr;
    JSValue onExit;
    
    char *command;
    char **args;
    size_t args_count;
} subprocess_req_t;

static void alloc_buffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf) {
    (void)handle;
    buf->base = malloc(suggested_size);
    buf->len = suggested_size;
}

static void on_stdout_read(uv_stream_t *pipe, ssize_t nread, const uv_buf_t *buf) {
    subprocess_req_t *req = pipe->data;
    if (nread > 0) {
        if (JS_IsFunction(req->ctx, req->onStdout)) {
            JSValue str = JS_NewStringLen(req->ctx, buf->base, nread);
            JSValue ret = JS_Call(req->ctx, req->onStdout, JS_UNDEFINED, 1, &str);
            JS_FreeValue(req->ctx, str);
            JS_FreeValue(req->ctx, ret);
            sofuu_flush_jobs(req->ctx);
        }
    } else if (nread < 0) {
        uv_read_stop(pipe);
    }
    if (buf->base) free(buf->base);
}

static void on_stderr_read(uv_stream_t *pipe, ssize_t nread, const uv_buf_t *buf) {
    subprocess_req_t *req = pipe->data;
    if (nread > 0) {
        if (JS_IsFunction(req->ctx, req->onStderr)) {
            JSValue str = JS_NewStringLen(req->ctx, buf->base, nread);
            JSValue ret = JS_Call(req->ctx, req->onStderr, JS_UNDEFINED, 1, &str);
            JS_FreeValue(req->ctx, str);
            JS_FreeValue(req->ctx, ret);
            sofuu_flush_jobs(req->ctx);
        }
    } else if (nread < 0) {
        uv_read_stop(pipe);
    }
    if (buf->base) free(buf->base);
}

static void on_process_exit(uv_process_t *process, int64_t exit_status, int term_signal) {
    (void)term_signal;
    subprocess_req_t *req = process->data;
    
    uv_read_stop((uv_stream_t*)&req->stdout_pipe);
    uv_read_stop((uv_stream_t*)&req->stderr_pipe);
    
    uv_close((uv_handle_t*)&req->stdin_pipe, NULL);
    uv_close((uv_handle_t*)&req->stdout_pipe, NULL);
    uv_close((uv_handle_t*)&req->stderr_pipe, NULL);
    uv_close((uv_handle_t*)process, NULL);

    if (JS_IsFunction(req->ctx, req->onExit)) {
        JSValue val = JS_NewInt32(req->ctx, (int)exit_status);
        JSValue ret = JS_Call(req->ctx, req->onExit, JS_UNDEFINED, 1, &val);
        JS_FreeValue(req->ctx, val);
        JS_FreeValue(req->ctx, ret);
        sofuu_flush_jobs(req->ctx);
    }

    JS_FreeValue(req->ctx, req->onStdout);
    JS_FreeValue(req->ctx, req->onStderr);
    JS_FreeValue(req->ctx, req->onExit);
    req->onStdout = JS_UNDEFINED;
    req->onStderr = JS_UNDEFINED;
    req->onExit = JS_UNDEFINED;
}

static JSClassID js_subprocess_class_id;

static void js_subprocess_finalizer(JSRuntime *rt, JSValue val) {
    subprocess_req_t *req = JS_GetOpaque(val, js_subprocess_class_id);
    if (req) {
        JS_FreeValue(req->ctx, req->onStdout);
        JS_FreeValue(req->ctx, req->onStderr);
        JS_FreeValue(req->ctx, req->onExit);
        
        for (size_t i = 0; i < req->args_count; i++) {
            free(req->args[i]);
        }
        free(req->args);
        free(req->command);
        if (req->options.cwd) free((void*)req->options.cwd);
        free(req);
    }
}

static JSValue js_subprocess_kill(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    (void)argc;
    (void)argv;
    subprocess_req_t *req = JS_GetOpaque(this_val, js_subprocess_class_id);
    if (!req) return JS_EXCEPTION;
    uv_process_kill(&req->process, 15); /* SIGTERM */
    return JS_UNDEFINED;
}

typedef struct {
    uv_write_t req;
    uv_buf_t buf;
} write_req_t;

static void on_write_done(uv_write_t *wreq, int status) {
    (void)status;
    write_req_t *wr = (write_req_t*)wreq;
    free(wr->buf.base);
    free(wr);
}

static JSValue js_subprocess_write(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 1) return JS_ThrowTypeError(ctx, "write expects 1 arg");
    subprocess_req_t *req = JS_GetOpaque(this_val, js_subprocess_class_id);
    if (!req) return JS_EXCEPTION;

    size_t len;
    const char *str = JS_ToCStringLen(ctx, &len, argv[0]);
    if (!str) return JS_EXCEPTION;

    write_req_t *wr = malloc(sizeof(write_req_t));
    wr->buf.base = malloc(len);
    wr->buf.len = len;
    memcpy(wr->buf.base, str, len);
    JS_FreeCString(ctx, str);

    uv_write(&wr->req, (uv_stream_t*)&req->stdin_pipe, &wr->buf, 1, on_write_done);
    return JS_UNDEFINED;
}

static JSClassDef js_subprocess_class = {
    "Subprocess",
    .finalizer = js_subprocess_finalizer,
};

static const JSCFunctionListEntry js_subprocess_proto_funcs[] = {
    JS_CFUNC_DEF("kill", 0, js_subprocess_kill),
    JS_CFUNC_DEF("write", 1, js_subprocess_write),
};

static JSValue js_sofuu_spawn(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    (void)this_val;
    if (argc < 1 || !JS_IsObject(argv[0])) {
        return JS_ThrowTypeError(ctx, "spawn requires options object");
    }
    
    JSValue opts = argv[0];
    JSValue cmd_val = JS_GetPropertyStr(ctx, opts, "command");
    if (JS_IsUndefined(cmd_val)) {
        return JS_ThrowTypeError(ctx, "options.command is required");
    }
    
    subprocess_req_t *req = calloc(1, sizeof(subprocess_req_t));
    req->ctx = ctx;
    
    const char *cmd_str = JS_ToCString(ctx, cmd_val);
    req->command = strdup(cmd_str);
    JS_FreeCString(ctx, cmd_str);
    JS_FreeValue(ctx, cmd_val);
    
    JSValue args_val = JS_GetPropertyStr(ctx, opts, "args");
    if (JS_IsArray(ctx, args_val)) {
        JSValue len_val = JS_GetPropertyStr(ctx, args_val, "length");
        uint32_t len;
        JS_ToUint32(ctx, &len, len_val);
        JS_FreeValue(ctx, len_val);
        
        req->args_count = len + 2; /* command + args + NULL */
        req->args = malloc(sizeof(char*) * req->args_count);
        req->args[0] = strdup(req->command);
        
        for (uint32_t i = 0; i < len; i++) {
            JSValue el = JS_GetPropertyUint32(ctx, args_val, i);
            const char* el_str = JS_ToCString(ctx, el);
            req->args[i + 1] = strdup(el_str);
            JS_FreeCString(ctx, el_str);
            JS_FreeValue(ctx, el);
        }
        req->args[len + 1] = NULL;
    } else {
        req->args_count = 2;
        req->args = malloc(sizeof(char*) * 2);
        req->args[0] = strdup(req->command);
        req->args[1] = NULL;
    }
    JS_FreeValue(ctx, args_val);
    
    req->onStdout = JS_GetPropertyStr(ctx, opts, "onStdout");
    req->onStderr = JS_GetPropertyStr(ctx, opts, "onStderr");
    req->onExit = JS_GetPropertyStr(ctx, opts, "onExit");
    
    uv_loop_t *loop = sofuu_loop_get();
    
    uv_pipe_init(loop, &req->stdin_pipe, 0);
    uv_pipe_init(loop, &req->stdout_pipe, 0);
    uv_pipe_init(loop, &req->stderr_pipe, 0);
    req->stdin_pipe.data = req;
    req->stdout_pipe.data = req;
    req->stderr_pipe.data = req;
    
    req->options.file = req->command;
    req->options.args = req->args;
    req->options.exit_cb = on_process_exit;
    
    req->stdio[0].flags = UV_CREATE_PIPE | UV_READABLE_PIPE;
    req->stdio[0].data.stream = (uv_stream_t*)&req->stdin_pipe;
    
    req->stdio[1].flags = UV_CREATE_PIPE | UV_WRITABLE_PIPE;
    req->stdio[1].data.stream = (uv_stream_t*)&req->stdout_pipe;
    
    req->stdio[2].flags = UV_CREATE_PIPE | UV_WRITABLE_PIPE;
    req->stdio[2].data.stream = (uv_stream_t*)&req->stderr_pipe;
    
    req->options.stdio_count = 3;
    req->options.stdio = req->stdio;
    
    JSValue cwd_val = JS_GetPropertyStr(ctx, opts, "cwd");
    if (!JS_IsUndefined(cwd_val)) {
        const char *cwd_str = JS_ToCString(ctx, cwd_val);
        req->options.cwd = strdup(cwd_str);
        JS_FreeCString(ctx, cwd_str);
    }
    JS_FreeValue(ctx, cwd_val);
    
    int r = uv_spawn(loop, &req->process, &req->options);
    if (r) {
        /* Cleanup on failure */
        free(req->command);
        for (size_t i = 0; i < req->args_count - 1; i++) free(req->args[i]);
        free(req->args);
        if (req->options.cwd) free((void*)req->options.cwd);
        free(req);
        return JS_ThrowTypeError(ctx, "spawn failed: %s", uv_strerror(r));
    }
    
    req->process.data = req;
    req->stdin_pipe.data = req;
    uv_read_start((uv_stream_t*)&req->stdout_pipe, alloc_buffer, on_stdout_read);
    uv_read_start((uv_stream_t*)&req->stderr_pipe, alloc_buffer, on_stderr_read);
    
    JSValue obj = JS_NewObjectClass(ctx, js_subprocess_class_id);
    JS_SetOpaque(obj, req);
    
    /* We don't free options.cwd here directly since uv_spawn might not make a deep copy synchronous in all platforms depending on libuv version (Wait, libuv doesn't copy it. I will keep it until exit or finalizer...)Actually libuv DOES NOT copy arguments/env/cwd. They must be valid until exit_cb! But I'm waiting for exit_cb. */
    
    return obj;
}

void mod_subprocess_register(JSContext *ctx) {
    JS_NewClassID(&js_subprocess_class_id);
    JS_NewClass(JS_GetRuntime(ctx), js_subprocess_class_id, &js_subprocess_class);
    
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, js_subprocess_proto_funcs, countof(js_subprocess_proto_funcs));
    JS_SetClassProto(ctx, js_subprocess_class_id, proto);
    
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue sofuu_obj = JS_GetPropertyStr(ctx, global, "sofuu");
    if (JS_IsUndefined(sofuu_obj)) sofuu_obj = JS_NewObject(ctx);
    
    JS_SetPropertyStr(ctx, sofuu_obj, "spawn", JS_NewCFunction(ctx, js_sofuu_spawn, "spawn", 1));
    JS_SetPropertyStr(ctx, global, "sofuu", sofuu_obj);
    JS_FreeValue(ctx, global);
}
