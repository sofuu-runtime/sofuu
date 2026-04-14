/*
 * src/npm/cjs.c — CommonJS compatibility shim for QuickJS
 *
 * Provides native synchronous require() functionality for CommonJS modules,
 * and a wrapper to convert CJS code into ESM so it can be 'import'ed.
 */

#include "cjs.h"
#include "resolver.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int is_cjs(const char *source) {
    /* Simple heuristic: looks for CJS keywords but NO ES module exports/imports */
    if (strstr(source, "module.exports") || strstr(source, "exports.") || strstr(source, "require(")) {
        if (!strstr(source, "export default ") && 
            !strstr(source, "export {") && 
            !strstr(source, "import {") && 
            !strstr(source, "import \"") && 
            !strstr(source, "import '")) {
            return 1;
        }
    }
    return 0;
}

char *cjs_to_esm(const char *source, size_t len, size_t *out_len) {
    const char *header = "var module = { exports: {} }; var exports = module.exports;\n";
    const char *footer = "\nexport default module.exports;\n";
    size_t hl = strlen(header);
    size_t fl = strlen(footer);
    char *wrap = (char *)malloc(hl + len + fl + 1);
    if (!wrap) return NULL;
    memcpy(wrap, header, hl);
    memcpy(wrap + hl, source, len);
    memcpy(wrap + hl + len, footer, fl);
    wrap[hl + len + fl] = '\0';
    if (out_len) *out_len = hl + len + fl;
    return wrap;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    sz = (long)fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

JSValue js_require(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv) {
    if (argc < 1) return JS_UNDEFINED;
    const char *spec = JS_ToCString(ctx, argv[0]);
    
    char cwd[2048];
    if (!getcwd(cwd, sizeof(cwd))) {
        JS_FreeCString(ctx, spec);
        return JS_ThrowReferenceError(ctx, "Cannot determine cwd");
    }
    
    char *resolved = npm_resolve(cwd, spec);
    
    if (!resolved) {
        JS_ThrowReferenceError(ctx, "Cannot find module '%s'", spec);
        JS_FreeCString(ctx, spec);
        return JS_EXCEPTION;
    }
    JS_FreeCString(ctx, spec);
    
    char *source = read_file(resolved);
    if (!source) {
        free(resolved);
        return JS_ThrowReferenceError(ctx, "Failed to read module");
    }
    
    const char *wrap_head = "(function(exports, require, module, __filename, __dirname) { ";
    const char *wrap_tail = "\n})";
    size_t wl = strlen(wrap_head) + strlen(source) + strlen(wrap_tail) + 1;
    char *wrapped = (char *)malloc(wl);
    snprintf(wrapped, wl, "%s%s%s", wrap_head, source, wrap_tail);
    free(source);
    
    JSValue func = JS_Eval(ctx, wrapped, strlen(wrapped), resolved, JS_EVAL_TYPE_GLOBAL);
    free(wrapped);
    
    if (JS_IsException(func)) {
        free(resolved);
        return func;
    }
    
    JSValue module_obj = JS_NewObject(ctx);
    JSValue exports_obj = JS_NewObject(ctx);
    JS_SetPropertyStr(ctx, module_obj, "exports", JS_DupValue(ctx, exports_obj));
    
    JSValue require_func = JS_NewCFunction(ctx, js_require, "require", 1);
    JSValue filename_val = JS_NewString(ctx, resolved);
    JSValue dirname_val  = JS_NewString(ctx, resolved); // simplistic
    free(resolved);
    
    JSValue call_args[] = { exports_obj, require_func, module_obj, filename_val, dirname_val };
    JSValue ret = JS_Call(ctx, func, JS_UNDEFINED, 5, call_args);
    
    JS_FreeValue(ctx, func);
    JS_FreeValue(ctx, exports_obj);
    JS_FreeValue(ctx, require_func);
    JS_FreeValue(ctx, filename_val);
    JS_FreeValue(ctx, dirname_val);
    
    if (JS_IsException(ret)) {
        JS_FreeValue(ctx, module_obj);
        return ret;
    }
    JS_FreeValue(ctx, ret);
    
    JSValue final_exports = JS_GetPropertyStr(ctx, module_obj, "exports");
    JS_FreeValue(ctx, module_obj);
    return final_exports;
}

void mod_cjs_register(JSContext *ctx) {
    JSValue global = JS_GetGlobalObject(ctx);
    JS_SetPropertyStr(ctx, global, "require", JS_NewCFunction(ctx, js_require, "require", 1));
    JS_FreeValue(ctx, global);
}
