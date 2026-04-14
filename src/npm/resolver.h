#pragma once
#include <stddef.h>

/*
 * npm_resolve — Find the absolute path of a bare module specifier.
 *
 * Implements the Node.js require() resolution algorithm:
 *  1. Walk up from `start_dir` looking for node_modules/<name>
 *  2. Inside node_modules/<name>: check package.json "main", then index.js
 *  3. Returns malloc'd absolute path or NULL if not found.
 *
 * caller must free() the returned string.
 */
char *npm_resolve(const char *start_dir, const char *module_name);

/*
 * npm_install — Download and extract a package from the npm registry.
 *
 * pkg_spec  : "lodash" or "lodash@4.17.21"
 * dest_dir  : directory containing node_modules/ (usually cwd)
 *
 * Returns 0 on success, non-zero on failure.
 */
int npm_install(const char *pkg_spec, const char *dest_dir);

/*
 * npm_install_local_package_json — Reads package.json in dest_dir and installs
 * all "dependencies" and "devDependencies".
 *
 * Returns 0 on success, non-zero on failure.
 */
int npm_install_local_package_json(const char *dest_dir);
