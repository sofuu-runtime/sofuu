/*
 * src/npm/resolver.c — npm module resolution + package installer
 *
 * Resolution follows the Node.js algorithm:
 *   Given import 'lodash' from file '/proj/src/app.js':
 *   1. Try /proj/src/node_modules/lodash
 *   2. Try /proj/node_modules/lodash
 *   3. Try /node_modules/lodash  (stops at fs root)
 *   At each candidate directory:
 *     a. Read package.json → "main" field
 *     b. Try candidate/<main>
 *     c. Try candidate/index.js
 *
 * Install uses libcurl to fetch from registry.npmjs.org,
 * then extracts the .tgz with the system `tar` command.
 */
#include "resolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <curl/curl.h>
#include "quickjs.h"

/* ── helpers ──────────────────────────────────────────────── */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

/* Minimal JSON string extraction: find "key": "value", return malloc'd value */
static char *json_get_str(const char *json, const char *key) {
    if (!json || !key) return NULL;
    /* Build search pattern: "key": " */
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);

    /* skip whitespace and colon */
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return NULL;
    p++; /* skip opening quote */

    const char *start = p;
    while (*p && *p != '"') {
        if (*p == '\\') p++;
        p++;
    }
    size_t len = (size_t)(p - start);
    char *result = (char *)malloc(len + 1);
    if (!result) return NULL;
    memcpy(result, start, len);
    result[len] = '\0';
    return result;
}

/* Read whole file into malloc'd buffer. Returns NULL on failure. */
static char *read_file_str(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* ── Resolution ───────────────────────────────────────────── */

/*
 * Try to resolve a module inside a specific node_modules/<name> directory.
 * Returns malloc'd path or NULL.
 */
static char *resolve_in_pkg_dir(const char *pkg_dir) {
    char path[PATH_MAX];

    /* 1. Try package.json "main" field */
    snprintf(path, sizeof(path), "%s/package.json", pkg_dir);
    char *pkg_json = read_file_str(path);
    if (pkg_json) {
        /* Try "exports" → "." → "default" (simplified) */
        char *main_val = json_get_str(pkg_json, "main");
        free(pkg_json);

        if (main_val) {
            char main_path[PATH_MAX];
            /* Remove leading ./ if present */
            const char *mrel = main_val;
            if (mrel[0] == '.' && mrel[1] == '/') mrel += 2;

            snprintf(main_path, sizeof(main_path), "%s/%s", pkg_dir, mrel);
            free(main_val);

            char resolved[PATH_MAX];
            if (realpath(main_path, resolved) && file_exists(resolved))
                return strdup(resolved);

            /* Try with .js appended */
            char with_js[PATH_MAX];
            snprintf(with_js, sizeof(with_js), "%s.js", main_path);
            if (realpath(with_js, resolved) && file_exists(resolved))
                return strdup(resolved);
        }
    }

    /* 2. Try index.js */
    snprintf(path, sizeof(path), "%s/index.js", pkg_dir);
    char resolved[PATH_MAX];
    if (realpath(path, resolved) && file_exists(resolved))
        return strdup(resolved);

    /* 3. Try index.mjs */
    snprintf(path, sizeof(path), "%s/index.mjs", pkg_dir);
    if (realpath(path, resolved) && file_exists(resolved))
        return strdup(resolved);

    return NULL;
}

char *npm_resolve(const char *start_dir, const char *module_name) {
    if (!start_dir || !module_name) return NULL;

    /* Handle scoped packages: @scope/pkg stays as a single folder path */
    char dir[PATH_MAX];
    strncpy(dir, start_dir, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';

    /* Walk up the directory tree */
    while (1) {
        char nm_dir[PATH_MAX];
        snprintf(nm_dir, sizeof(nm_dir), "%s/node_modules", dir);

        if (dir_exists(nm_dir)) {
            char pkg_dir[PATH_MAX];
            snprintf(pkg_dir, sizeof(pkg_dir), "%s/%s", nm_dir, module_name);

            if (dir_exists(pkg_dir)) {
                char *result = resolve_in_pkg_dir(pkg_dir);
                if (result) return result;
            }

            /* Also try as a direct file: node_modules/pkg.js */
            char direct[PATH_MAX];
            snprintf(direct, sizeof(direct), "%s/%s.js", nm_dir, module_name);
            char resolved[PATH_MAX];
            if (realpath(direct, resolved) && file_exists(resolved))
                return strdup(resolved);
        }

        /* Go up one directory */
        char parent[PATH_MAX];
        snprintf(parent, sizeof(parent), "%s/..", dir);
        char resolved_parent[PATH_MAX];
        if (!realpath(parent, resolved_parent)) break;
        if (strcmp(resolved_parent, dir) == 0) break; /* reached root */
        strncpy(dir, resolved_parent, sizeof(dir) - 1);
    }

    return NULL;
}

/* ── Package Install ──────────────────────────────────────── */

typedef struct {
    char   *buf;
    size_t  len;
    size_t  cap;
} CurlBuf;

static size_t curl_write_cb(void *data, size_t size, size_t nmemb, void *userp) {
    CurlBuf *b = (CurlBuf *)userp;
    size_t total = size * nmemb;
    if (b->len + total + 1 > b->cap) {
        size_t new_cap = (b->cap + total) * 2 + 4096;
        char *new_buf = (char *)realloc(b->buf, new_cap);
        if (!new_buf) return 0;
        b->buf = new_buf;
        b->cap = new_cap;
    }
    memcpy(b->buf + b->len, data, total);
    b->len += total;
    b->buf[b->len] = '\0';
    return total;
}

typedef struct {
    char **installed;
    size_t count;
    size_t cap;
} InstallCtx;

static int is_installed(InstallCtx *ctx, const char *name) {
    if (!ctx) return 0;
    for (size_t i = 0; i < ctx->count; i++) {
        if (strcmp(ctx->installed[i], name) == 0) return 1;
    }
    return 0;
}

static void mark_installed(InstallCtx *ctx, const char *name) {
    if (!ctx) return;
    if (ctx->count >= ctx->cap) {
        ctx->cap = ctx->cap == 0 ? 16 : ctx->cap * 2;
        ctx->installed = (char **)realloc(ctx->installed, ctx->cap * sizeof(char *));
    }
    ctx->installed[ctx->count++] = strdup(name);
}

static int npm_install_recursive(const char *pkg_spec, const char *dest_dir, InstallCtx *inst_ctx) {
    if (!pkg_spec || !dest_dir) return 1;

    /* Split name@version */
    char name[256] = {0};
    char version[64] = {0};

    /* Handle scoped packages: @scope/name@version */
    const char *at = NULL;
    if (pkg_spec[0] == '@') {
        /* scoped: find @ after the / */
        const char *slash = strchr(pkg_spec + 1, '/');
        if (slash) at = strchr(slash + 1, '@');
    } else {
        at = strchr(pkg_spec, '@');
    }

    if (at && at != pkg_spec) {
        size_t nlen = (size_t)(at - pkg_spec);
        if (nlen >= sizeof(name)) nlen = sizeof(name) - 1;
        memcpy(name, pkg_spec, nlen);
        name[nlen] = '\0';
        strncpy(version, at + 1, sizeof(version) - 1);
    } else {
        strncpy(name, pkg_spec, sizeof(name) - 1);
        strncpy(version, "latest", sizeof(version) - 1);
    }

    /* Strip semver constraint symbols so the registry URL works */
    char clean_version[64];
    char *v_ptr = version;
    while (*v_ptr && strchr("^~=<>v ", *v_ptr)) v_ptr++;
    if (!*v_ptr || *v_ptr == '*') v_ptr = "latest";
    strncpy(clean_version, v_ptr, sizeof(clean_version) - 1);
    clean_version[sizeof(clean_version) - 1] = '\0';

    if (is_installed(inst_ctx, name)) {
        return 0; /* Already installed or being installed in this session */
    }
    mark_installed(inst_ctx, name);

    printf("  \033[36m→\033[0m Resolving %s@%s from registry.npmjs.org...\n", name, clean_version);

    /* ── Step 1: fetch package metadata ── */
    char meta_url[512];
    snprintf(meta_url, sizeof(meta_url),
             "https://registry.npmjs.org/%s/%s", name, clean_version);

    CURL *curl = curl_easy_init();
    if (!curl) { fprintf(stderr, "curl init failed\n"); return 1; }

    CurlBuf meta_buf = {0};
    meta_buf.buf = (char *)malloc(4096);
    meta_buf.cap = 4096;
    if (!meta_buf.buf) { curl_easy_cleanup(curl); return 1; }

    curl_easy_setopt(curl, CURLOPT_URL, meta_url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &meta_buf);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "sofuu/0.1");

    CURLcode rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK || !meta_buf.buf) {
        fprintf(stderr, "  \033[31m✗\033[0m Failed to fetch metadata: %s\n",
                curl_easy_strerror(rc));
        free(meta_buf.buf);
        return 1;
    }

    /* ── Step 2: extract tarball URL from metadata ── */
    char *tarball_url = json_get_str(meta_buf.buf, "tarball");
    char *resolved_version = json_get_str(meta_buf.buf, "version");
    free(meta_buf.buf);

    if (!tarball_url) {
        fprintf(stderr, "  \033[31m✗\033[0m Could not parse registry response\n");
        free(resolved_version);
        return 1;
    }

    if (resolved_version)
        printf("  \033[36m→\033[0m Installing %s@%s\n", name, resolved_version);

    /* ── Step 3: download .tgz ── */
    char tmp_tgz[PATH_MAX];
    snprintf(tmp_tgz, sizeof(tmp_tgz), "/tmp/sofuu_%s.tgz", name);
    /* Replace / with _ for scoped packages */
    for (char *p = tmp_tgz + 5; *p; p++) if (*p == '/') *p = '_';

    curl = curl_easy_init();
    if (!curl) { free(tarball_url); free(resolved_version); return 1; }

    FILE *tgz_f = fopen(tmp_tgz, "wb");
    if (!tgz_f) {
        fprintf(stderr, "  \033[31m✗\033[0m Cannot write to %s\n", tmp_tgz);
        curl_easy_cleanup(curl);
        free(tarball_url); free(resolved_version);
        return 1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, tarball_url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL); /* use default */
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, tgz_f);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "sofuu/0.1");
    rc = curl_easy_perform(curl);
    fclose(tgz_f);
    curl_easy_cleanup(curl);
    free(tarball_url);

    if (rc != CURLE_OK) {
        fprintf(stderr, "  \033[31m✗\033[0m Download failed: %s\n", curl_easy_strerror(rc));
        free(resolved_version);
        return 1;
    }

    /* ── Step 4: extract into node_modules/<name> ── */
    char nm_dir[PATH_MAX];
    snprintf(nm_dir, sizeof(nm_dir), "%s/node_modules", dest_dir);
    mkdir(nm_dir, 0755);

    char pkg_dir[PATH_MAX];
    snprintf(pkg_dir, sizeof(pkg_dir), "%s/%s", nm_dir, name);
    /* Create parent directories for scoped packages */
    if (name[0] == '@') {
        char scope_dir[PATH_MAX];
        const char *slash = strchr(name + 1, '/');
        if (slash) {
            size_t scope_len = (size_t)(slash - name);
            snprintf(scope_dir, sizeof(scope_dir), "%s/%.*s",
                     nm_dir, (int)scope_len, name);
            mkdir(scope_dir, 0755);
        }
    }
    mkdir(pkg_dir, 0755);

    /*
     * npm tarballs have an extra "package/" prefix inside the .tgz.
     * We strip it with --strip-components=1.
     */
    char cmd[PATH_MAX * 2 + 128];
    snprintf(cmd, sizeof(cmd),
             "tar -xzf '%s' --strip-components=1 -C '%s'",
             tmp_tgz, pkg_dir);

    int tar_rc = system(cmd);
    unlink(tmp_tgz);

    if (tar_rc != 0) {
        fprintf(stderr, "  \033[31m✗\033[0m Extraction failed\n");
        free(resolved_version);
        return 1;
    }

    printf("  \033[32m✓\033[0m Installed %s@%s → node_modules/%s\n",
           name, resolved_version ? resolved_version : "?", name);
    free(resolved_version);

    /* ── Step 5: Read package.json and resolve transitive dependencies ── */
    char pjson_path[PATH_MAX];
    snprintf(pjson_path, sizeof(pjson_path), "%s/package.json", pkg_dir);
    char *pkg_json = read_file_str(pjson_path);
    if (pkg_json) {
        JSRuntime *rt = JS_NewRuntime();
        if (rt) {
            JSContext *ctx = JS_NewContext(rt);
            if (ctx) {
                JSValue parsed = JS_ParseJSON(ctx, pkg_json, strlen(pkg_json), "package.json");
                if (!JS_IsException(parsed) && !JS_IsUndefined(parsed)) {
                    JSValue deps = JS_GetPropertyStr(ctx, parsed, "dependencies");
                    if (JS_IsObject(deps)) {
                        JSPropertyEnum *ptab;
                        uint32_t plen;
                        if (JS_GetOwnPropertyNames(ctx, &ptab, &plen, deps, JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0) {
                            for (uint32_t i = 0; i < plen; i++) {
                                const char *dep_name = JS_AtomToCString(ctx, ptab[i].atom);
                                JSValue ver_val = JS_GetProperty(ctx, deps, ptab[i].atom);
                                const char *dep_ver = JS_ToCString(ctx, ver_val);

                                if (dep_name && dep_ver) {
                                    char child_spec[512];
                                    snprintf(child_spec, sizeof(child_spec), "%s@%s", dep_name, dep_ver);
                                    npm_install_recursive(child_spec, dest_dir, inst_ctx);
                                }

                                if (dep_ver) JS_FreeCString(ctx, dep_ver);
                                JS_FreeValue(ctx, ver_val);
                                if (dep_name) JS_FreeCString(ctx, dep_name);
                                JS_FreeAtom(ctx, ptab[i].atom);
                            }
                            js_free(ctx, ptab);
                        }
                    }
                    JS_FreeValue(ctx, deps);
                }
                JS_FreeValue(ctx, parsed);
                JS_FreeContext(ctx);
            }
            JS_FreeRuntime(rt);
        }
        free(pkg_json);
    }

    return 0;
}

int npm_install(const char *pkg_spec, const char *dest_dir) {
    InstallCtx ctx = {0};
    int rc = npm_install_recursive(pkg_spec, dest_dir, &ctx);
    for (size_t i = 0; i < ctx.count; i++) {
        free(ctx.installed[i]);
    }
    free(ctx.installed);
    return rc;
}

int npm_install_local_package_json(const char *dest_dir) {
    char pjson_path[PATH_MAX];
    snprintf(pjson_path, sizeof(pjson_path), "%s/package.json", dest_dir);
    char *pkg_json = read_file_str(pjson_path);
    if (!pkg_json) {
        fprintf(stderr, "\033[31mError:\033[0m No package.json found in %s\n", dest_dir);
        return 1;
    }

    int rc = 0;
    JSRuntime *rt = JS_NewRuntime();
    if (rt) {
        JSContext *ctx = JS_NewContext(rt);
        if (ctx) {
            JSValue parsed = JS_ParseJSON(ctx, pkg_json, strlen(pkg_json), "package.json");
            if (!JS_IsException(parsed) && !JS_IsUndefined(parsed)) {
                /* Install dependencies */
                JSValue deps = JS_GetPropertyStr(ctx, parsed, "dependencies");
                if (JS_IsObject(deps)) {
                    printf("\n\033[1mInstalling dependencies...\033[0m\n");
                    JSPropertyEnum *ptab;
                    uint32_t plen;
                    if (JS_GetOwnPropertyNames(ctx, &ptab, &plen, deps, JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0) {
                        for (uint32_t i = 0; i < plen; i++) {
                            const char *dep_name = JS_AtomToCString(ctx, ptab[i].atom);
                            JSValue ver_val = JS_GetProperty(ctx, deps, ptab[i].atom);
                            const char *dep_ver = JS_ToCString(ctx, ver_val);

                            if (dep_name && dep_ver) {
                                char child_spec[512];
                                snprintf(child_spec, sizeof(child_spec), "%s@%s", dep_name, dep_ver);
                                if (npm_install(child_spec, dest_dir) != 0) rc = 1;
                            }

                            if (dep_ver) JS_FreeCString(ctx, dep_ver);
                            JS_FreeValue(ctx, ver_val);
                            if (dep_name) JS_FreeCString(ctx, dep_name);
                            JS_FreeAtom(ctx, ptab[i].atom);
                        }
                        js_free(ctx, ptab);
                    }
                }
                JS_FreeValue(ctx, deps);

                /* Install devDependencies */
                JSValue devDeps = JS_GetPropertyStr(ctx, parsed, "devDependencies");
                if (JS_IsObject(devDeps)) {
                    printf("\n\033[1mInstalling devDependencies...\033[0m\n");
                    JSPropertyEnum *ptab;
                    uint32_t plen;
                    if (JS_GetOwnPropertyNames(ctx, &ptab, &plen, devDeps, JS_GPN_STRING_MASK | JS_GPN_ENUM_ONLY) == 0) {
                        for (uint32_t i = 0; i < plen; i++) {
                            const char *dep_name = JS_AtomToCString(ctx, ptab[i].atom);
                            JSValue ver_val = JS_GetProperty(ctx, devDeps, ptab[i].atom);
                            const char *dep_ver = JS_ToCString(ctx, ver_val);

                            if (dep_name && dep_ver) {
                                char child_spec[512];
                                snprintf(child_spec, sizeof(child_spec), "%s@%s", dep_name, dep_ver);
                                if (npm_install(child_spec, dest_dir) != 0) rc = 1;
                            }

                            if (dep_ver) JS_FreeCString(ctx, dep_ver);
                            JS_FreeValue(ctx, ver_val);
                            if (dep_name) JS_FreeCString(ctx, dep_name);
                            JS_FreeAtom(ctx, ptab[i].atom);
                        }
                        js_free(ctx, ptab);
                    }
                }
                JS_FreeValue(ctx, devDeps);
            } else {
                fprintf(stderr, "\033[31mError:\033[0m Invalid package.json\n");
                rc = 1;
            }
            JS_FreeValue(ctx, parsed);
            JS_FreeContext(ctx);
        }
        JS_FreeRuntime(rt);
    }
    free(pkg_json);
    return rc;
}
