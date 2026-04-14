#pragma once
#include <stddef.h>

/*
 * ts_strip — Strip TypeScript type annotations from source.
 *
 * Returns a newly malloc'd buffer of the same byte length as the input,
 * with all type-only syntax replaced by spaces (newlines preserved so
 * line numbers in error messages stay accurate).
 *
 * Caller must free() the returned pointer.
 * Returns NULL on allocation failure.
 *
 * Handles erasable TypeScript syntax:
 *   : TypeAnnotation         const x: number = 5
 *   as Type                  x as string
 *   satisfies Type           obj satisfies Foo
 *   interface Foo {}         full declaration stripped
 *   type Foo = ...           full alias stripped
 *   import type {...}        whole import stripped
 *   export type {...}        "type" keyword stripped
 *   declare ...              full declaration stripped
 *   <TypeParams>             generic params on functions/classes
 *   public/private/protected/readonly/abstract/override
 *   !                        non-null assertions
 *   ?:                       optional parameter marker
 */
char *ts_strip(const char *src, size_t len, size_t *out_len);
