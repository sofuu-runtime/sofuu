/*
 * mod_console.h — console.log / error / warn / info bindings
 */
#ifndef SOFUU_MOD_CONSOLE_H
#define SOFUU_MOD_CONSOLE_H

#include "../../deps/quickjs/quickjs.h"

void mod_console_register(JSContext *ctx);

#endif
