#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "quickjs.h"
#include "quickjs-libc.h"
#include "sse.h"

#ifndef countof
#define countof(x) (sizeof(x) / sizeof((x)[0]))
#endif

static JSClassID sofuu_sse_class_id;

typedef struct {
    JSContext *ctx;
    char *buffer;
    size_t size;
    JSValue this_val;
} sofuu_sse_t;

static void sofuu_sse_finalizer(JSRuntime *rt, JSValue val) {
    sofuu_sse_t *sse = JS_GetOpaque(val, sofuu_sse_class_id);
    if (sse) {
        if (sse->buffer) free(sse->buffer);
        free(sse);
    }
}

static JSValue js_sse_constructor(JSContext *ctx, JSValueConst new_target, int argc, JSValueConst *argv) {
    JSValue obj = JS_NewObjectClass(ctx, sofuu_sse_class_id);
    if (JS_IsException(obj)) return obj;
    
    sofuu_sse_t *sse = calloc(1, sizeof(sofuu_sse_t));
    sse->ctx = ctx;
    sse->this_val = obj;
    
    JS_SetOpaque(obj, sse);
    return obj;
}

// Emits an event back to JS
static void emit_event(sofuu_sse_t *sse, const char *event_name, const char *data, size_t data_len) {
    JSValue on_message = JS_GetPropertyStr(sse->ctx, sse->this_val, "onMessage");
    if (JS_IsFunction(sse->ctx, on_message)) {
        JSValue args[2];
        args[0] = JS_NewString(sse->ctx, event_name);
        args[1] = JS_NewStringLen(sse->ctx, data, data_len);
        
        JSValue ret = JS_Call(sse->ctx, on_message, sse->this_val, 2, args);
        if (JS_IsException(ret)) {
            js_std_dump_error(sse->ctx); // Assuming caller includes quickjs-libc or we just ignore
        }
        JS_FreeValue(sse->ctx, ret);
        JS_FreeValue(sse->ctx, args[0]);
        JS_FreeValue(sse->ctx, args[1]);
    }
    JS_FreeValue(sse->ctx, on_message);
}

// Feeds raw chunk into the state machine
static JSValue js_sse_feed(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    sofuu_sse_t *sse = JS_GetOpaque(this_val, sofuu_sse_class_id);
    if (!sse) return JS_ThrowTypeError(ctx, "Invalid SSEParser");
    
    if (argc < 1) return JS_UNDEFINED;
    
    size_t chunk_len;
    const char *chunk = JS_ToCStringLen(ctx, &chunk_len, argv[0]);
    if (!chunk) return JS_EXCEPTION;
    
    // Append to internal buffer
    sse->buffer = realloc(sse->buffer, sse->size + chunk_len + 1);
    memcpy(sse->buffer + sse->size, chunk, chunk_len);
    sse->size += chunk_len;
    sse->buffer[sse->size] = '\0';
    
    JS_FreeCString(ctx, chunk);
    
    // Simple basic scan for \n\n
    // Note: The correct SSE spec says blocks separated by \n\n or \r\n\r\n. 
    // And within a block, there are lines like `data: { JSON }`
    char *head = sse->buffer;
    char *end = sse->buffer + sse->size;
    
    while (head < end) {
        char *block_end = strstr(head, "\n\n");
        if (!block_end) break;
        
        // We have a full block! Let's parse it
        char *line = head;
        // Basic parser: look for "data: "
        char *data_start = strstr(line, "data: ");
        if (data_start && data_start < block_end) {
            data_start += 6; // move past "data: "
            char *data_end = strchr(data_start, '\n');
            if (!data_end || data_end > block_end) data_end = block_end;
            
            // Emit
            emit_event(sse, "message", data_start, data_end - data_start);
        }
        
        head = block_end + 2; 
    }
    
    // Remove consumed data
    size_t consumed = head - sse->buffer;
    if (consumed > 0) {
        size_t remaining = sse->size - consumed;
        memmove(sse->buffer, head, remaining);
        sse->size = remaining;
        sse->buffer[sse->size] = '\0';
    }
    
    return JS_UNDEFINED;
}

static const JSCFunctionListEntry sse_proto_funcs[] = {
    JS_CFUNC_DEF("feed", 1, js_sse_feed),
};

static JSClassDef sofuu_sse_class = {
    "SSEParser",
    .finalizer = sofuu_sse_finalizer,
};

void mod_http_sse_register(JSContext *ctx) {
    JS_NewClassID(&sofuu_sse_class_id);
    JS_NewClass(JS_GetRuntime(ctx), sofuu_sse_class_id, &sofuu_sse_class);
    
    JSValue proto = JS_NewObject(ctx);
    JS_SetPropertyFunctionList(ctx, proto, sse_proto_funcs, countof(sse_proto_funcs));
    JS_SetClassProto(ctx, sofuu_sse_class_id, proto);
    
    JSValue global = JS_GetGlobalObject(ctx);
    JSValue sofuu = JS_GetPropertyStr(ctx, global, "sofuu");
    
    if (JS_IsUndefined(sofuu)) {
        sofuu = JS_NewObject(ctx);
        JS_SetPropertyStr(ctx, global, "sofuu", JS_DupValue(ctx, sofuu));
    }
    
    JSValue constructor = JS_NewCFunction2(ctx, js_sse_constructor, "SSEParser", 0, JS_CFUNC_constructor, 0);
    
    JS_SetPropertyStr(ctx, sofuu, "SSEParser", constructor);
    
    JS_FreeValue(ctx, sofuu);
    JS_FreeValue(ctx, global);
}
