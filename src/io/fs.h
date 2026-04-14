/*
 * fs.h — Async File I/O module for Sofuu
 *
 * Provides sofuu.fs namespace:
 *   sofuu.fs.readFile(path)            → Promise<string>
 *   sofuu.fs.writeFile(path, data)     → Promise<void>
 *   sofuu.fs.appendFile(path, data)    → Promise<void>
 *   sofuu.fs.readFileBytes(path)       → Promise<Uint8Array>  [Phase 3]
 *   sofuu.fs.exists(path)              → Promise<boolean>
 *   sofuu.fs.mkdir(path, {recursive})  → Promise<void>
 *   sofuu.fs.rm(path)                  → Promise<void>
 *   sofuu.fs.readdir(path)             → Promise<string[]>
 */
#ifndef SOFUU_FS_H
#define SOFUU_FS_H

#include "quickjs.h"

void mod_fs_register(JSContext *ctx);

#endif
