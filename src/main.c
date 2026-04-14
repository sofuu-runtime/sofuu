/*
 * main.c — sofuu CLI entry point
 *
 * Commands:
 *   sofuu run <file.js>     — Execute a JS/TS file
 *   sofuu bundle <entry.js>  — Output single-file bundle
 *   sofuu eval "<code>"     — Execute a JS string
 *   sofuu install           — Install from package.json
 *   sofuu add <pkg>         — Install npm package
 *   sofuu version           — Print version info
 *   sofuu help              — Print help
 *   sofuu                   — REPL
 */
#include "sofuu.h"
#include "modules/mod_process.h"
#include "resolver.h"
#include "repl.h"
#include "bundler.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define SOFUU_VERSION "0.1.0-alpha"

static void print_banner(void) {
    printf("\033[1;36m");
    printf("  ┌─────────────────────────────────┐\n");
    printf("  │   ⚡ Sofuu (素風) Runtime        │\n");
    printf("  │   v%-31s│\n", SOFUU_VERSION);
    printf("  │   Simple/Pure Wind — AI-Native  │\n");
    printf("  └─────────────────────────────────┘\n");
    printf("\033[0m");
}

static void print_help(void) {
    print_banner();
    printf("\n\033[1mUsage:\033[0m\n");
    printf("  sofuu                    Start interactive REPL\n");
    printf("  sofuu run <file.js>      Run a JavaScript file\n");
    printf("  sofuu run <file.ts>      Run a TypeScript file\n");
    printf("  sofuu eval \"<code>\"      Evaluate JS inline\n");
    printf("  sofuu bundle <entry.js>  Bundle to single file\n");
    printf("  sofuu bundle <entry.js> -o <out.js>\n");
    printf("  sofuu install            Install from package.json\n");
    printf("  sofuu add <pkg>          Install npm package\n");
    printf("  sofuu version            Print version and exit\n");
    printf("  sofuu licenses           Print open-source licenses\n");
    printf("  sofuu help               Print this help\n");
    printf("\n\033[1mExamples:\033[0m\n");
    printf("  sofuu                    (starts REPL)\n");
    printf("  sofuu run app.js\n");
    printf("  sofuu run server.ts\n");
    printf("  sofuu bundle src/main.js -o dist/app.bundle.js\n");
    printf("  sofuu install\n");
    printf("  sofuu add lodash\n");
    printf("  sofuu eval \"console.log('Sofuu!')\"\n");
    printf("\n\033[90mBuilt with QuickJS + C | MIT License\033[0m\n\n");
}

static void print_version(void) {
    printf("sofuu %s\n", SOFUU_VERSION);
    printf("quickjs 2024-01-13\n");
    printf("platform: ");

#if defined(__APPLE__)
    printf("macOS");
#elif defined(__linux__)
    printf("Linux");
#elif defined(_WIN32)
    printf("Windows");
#else
    printf("unknown");
#endif

#if defined(__aarch64__) || defined(__arm64__)
    printf(" arm64");
#elif defined(__x86_64__)
    printf(" x86_64");
#endif
    printf("\n");
}

static void print_licenses(void) {
    printf("\n\033[1m⚡ Sofuu (素風) Open Source Notices\033[0m\n\n");
    printf("Sofuu is MIT Licensed.\n");
    printf("Copyright (c) 2024 Priyanshu Boruah\n\n");
    printf("\033[1mThird-party components bundled in this binary:\033[0m\n\n");
    printf("  QuickJS (JS Engine)\n");
    printf("    Copyright (c) 2017-2021 Fabrice Bellard, Charlie Gordon\n");
    printf("    License: MIT  |  https://bellard.org/quickjs/\n\n");
    printf("  libuv (Async I/O)\n");
    printf("    Copyright (c) 2015-present libuv project contributors\n");
    printf("    Copyright (c) Joyent, Inc. and other Node contributors\n");
    printf("    License: MIT  |  https://github.com/libuv/libuv\n\n");
    printf("  http-parser (HTTP/1.1 Parser)\n");
    printf("    Copyright (c) Joyent, Inc. and other Node contributors\n");
    printf("    License: MIT  |  https://github.com/nodejs/http-parser\n\n");
    printf("  libcurl (HTTP Client)\n");
    printf("    Copyright (c) 1996-2024 Daniel Stenberg and curl contributors\n");
    printf("    License: curl (MIT-style)  |  https://curl.se/docs/copyright.html\n\n");
    printf("\033[90mFull notices: see NOTICE file or https://sofuu.dev/licenses\033[0m\n\n");
}

int main(int argc, char **argv) {
    /* Pass argv to process module before init */
    mod_process_set_args(argc, argv);

    if (argc < 2) {
        return sofuu_repl();
    }

    const char *cmd = argv[1];

    /* sofuu version */
    if (strcmp(cmd, "version") == 0 || strcmp(cmd, "--version") == 0 || strcmp(cmd, "-v") == 0) {
        print_version();
        return 0;
    }

    /* sofuu licenses */
    if (strcmp(cmd, "licenses") == 0 || strcmp(cmd, "--licenses") == 0) {
        print_licenses();
        return 0;
    }

    /* sofuu help */
    if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_help();
        return 0;
    }

    /* sofuu run <file> */
    if (strcmp(cmd, "run") == 0) {
        if (argc < 3) {
            fprintf(stderr, "\033[31mError:\033[0m 'sofuu run' requires a file argument\n");
            fprintf(stderr, "Usage: sofuu run <file.js>\n");
            return 1;
        }

        const char *file = argv[2];

        SofuuRuntime *rt = sofuu_init();
        if (!rt) {
            fprintf(stderr, "\033[31mFatal:\033[0m Failed to initialize Sofuu runtime\n");
            return 1;
        }

        int rc = sofuu_eval_file(rt, file);
        sofuu_destroy(rt);
        return rc;
    }

    /* sofuu eval "<code>" */
    if (strcmp(cmd, "eval") == 0) {
        if (argc < 3) {
            fprintf(stderr, "\033[31mError:\033[0m 'sofuu eval' requires a code string argument\n");
            return 1;
        }

        SofuuRuntime *rt = sofuu_init();
        if (!rt) {
            fprintf(stderr, "\033[31mFatal:\033[0m Failed to initialize Sofuu runtime\n");
            return 1;
        }

        int rc = sofuu_eval_string(rt, argv[2], "<eval>");
        sofuu_destroy(rt);
        return rc;
    }

    /* sofuu bundle <entry> [-o <output>] */
    if (strcmp(cmd, "bundle") == 0) {
        if (argc < 3) {
            fprintf(stderr, "\033[31mError:\033[0m 'sofuu bundle' requires an entry file\n");
            fprintf(stderr, "Usage: sofuu bundle <entry.js> [-o <out.js>]\n");
            return 1;
        }
        const char *entry  = argv[2];
        const char *output = "bundle.js"; /* default output */
        for (int i = 3; i < argc - 1; i++) {
            if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
                output = argv[i + 1];
            }
        }
        return sofuu_bundle(entry, output, 0);
    }

    /* sofuu install */
    if (strcmp(cmd, "install") == 0) {
        char cwd[4096];
        if (!getcwd(cwd, sizeof(cwd))) {
            fprintf(stderr, "\033[31mError:\033[0m Cannot determine current directory\n");
            return 1;
        }
        return npm_install_local_package_json(cwd) != 0 ? 1 : 0;
    }

    /* sofuu add <pkg> [@version] */
    if (strcmp(cmd, "add") == 0) {
        if (argc < 3) {
            fprintf(stderr, "\033[31mError:\033[0m 'sofuu add' requires a package name\n");
            fprintf(stderr, "Usage: sofuu add <package>[@version]\n");
            return 1;
        }
        /* Install all listed packages */
        char cwd[4096];
        if (!getcwd(cwd, sizeof(cwd))) {
            fprintf(stderr, "\033[31mError:\033[0m Cannot determine current directory\n");
            return 1;
        }
        int failed = 0;
        for (int i = 2; i < argc; i++) {
            printf("\n\033[1msofuu add %s\033[0m\n", argv[i]);
            if (npm_install(argv[i], cwd) != 0) failed = 1;
        }
        return failed ? 1 : 0;
    }

    fprintf(stderr, "\033[31mError:\033[0m Unknown command '%s'\n", cmd);
    fprintf(stderr, "Run 'sofuu help' for usage.\n");
    return 1;
}
