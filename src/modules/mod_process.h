/*
 * mod_process.h — process.argv, process.env, process.exit()
 */
#ifndef SOFUU_MOD_PROCESS_H
#define SOFUU_MOD_PROCESS_H

#include "../../deps/quickjs/quickjs.h"

/* argc/argv from main() so process.argv is populated */
void mod_process_set_args(int argc, char **argv);

void mod_process_register(JSContext *ctx);

/* Must be called BEFORE JS_FreeContext to release stored JSValue references */
void mod_process_cleanup(JSContext *ctx);

#endif
