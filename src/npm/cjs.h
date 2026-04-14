#pragma once
#include <stddef.h>
#include "quickjs.h"

/* Check if source looks like CommonJS */
int is_cjs(const char *source);

/* Wrap CJS source into an ESM that exports default module.exports */
char *cjs_to_esm(const char *source, size_t len, size_t *out_len);

/* Global require() function for QuickJS */
JSValue js_require(JSContext *ctx, JSValueConst this_val, int argc, JSValueConst *argv);

/* Register global require in context */
void mod_cjs_register(JSContext *ctx);
