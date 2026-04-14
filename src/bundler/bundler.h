#pragma once
#include <stddef.h>

/*
 * sofuu_bundle — resolve the full import graph starting from entry_path,
 * transform all ESM imports/exports into a CJS-style bundle registry,
 * and write the result to output_path as a single self-contained JS file.
 *
 * entry_path  : path to the root .js or .ts file
 * output_path : destination file (created or overwritten)
 * minify      : if 1, collapse whitespace in the preamble/footer
 *
 * Returns 0 on success, non-zero on failure.
 */
int sofuu_bundle(const char *entry_path, const char *output_path, int minify);
