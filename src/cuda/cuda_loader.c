// src/cuda/cuda_loader.c — runtime dynamic loader for the CUDA Driver API.
//
// PLAN-GPU-CUDA C0. No CUDA headers, no libcudart, no link-time dependency:
// the NVIDIA display driver's libcuda.so.1 is opened with dlopen() and only
// the stable Driver API subset is resolved with dlsym().
//
// The loader is intentionally the ONLY place that knows driver symbol names,
// so the Rust engine (crates/sofuu-core/src/infer/cuda.rs) and the standalone
// Kaggle harness (src/cuda/parity/gpu_check.cpp) share identical resolution
// and fallback behaviour.

#include "cuda_loader.h"

#if defined(_WIN32)
// Sofuu's CUDA backend is Linux (and driver-probe capable on macOS for tests).
// Windows has no CUDA driver path in this milestone; keep the translation unit
// compilable so build systems can include it unconditionally.
sofuu_cuda_driver *sofuu_cuda_driver_open(const char *path) {
    (void)path;
    return 0;
}
void sofuu_cuda_driver_close(sofuu_cuda_driver *driver) { (void)driver; }
const char *sofuu_cuda_driver_path(const sofuu_cuda_driver *driver) {
    (void)driver;
    return "";
}
int sofuu_cuda_driver_version(const sofuu_cuda_driver *driver) {
    (void)driver;
    return 0;
}
int sofuu_cuda_driver_device_count(const sofuu_cuda_driver *driver) {
    (void)driver;
    return 0;
}
int sofuu_cuda_driver_device_info(const sofuu_cuda_driver *driver, int ordinal,
                                  char *name, size_t name_cap, int *cc_major,
                                  int *cc_minor, unsigned long long *total_mem,
                                  int *integrated, int *unified) {
    (void)driver; (void)ordinal; (void)name; (void)name_cap; (void)cc_major;
    (void)cc_minor; (void)total_mem; (void)integrated; (void)unified;
    return SOFUU_CUDA_ERR_LOAD;
}
sofuu_cuda_ctx *sofuu_cuda_ctx_create(sofuu_cuda_driver *driver, int ordinal) {
    (void)driver; (void)ordinal;
    return 0;
}
void sofuu_cuda_ctx_destroy(sofuu_cuda_ctx *ctx) { (void)ctx; }
int sofuu_cuda_ctx_load_ptx(sofuu_cuda_ctx *ctx, const char *ptx) {
    (void)ctx; (void)ptx;
    return SOFUU_CUDA_ERR_CONTEXT;
}
int sofuu_cuda_ctx_function(sofuu_cuda_ctx *ctx, const char *name, void **out) {
    (void)ctx; (void)name; (void)out;
    return SOFUU_CUDA_ERR_CONTEXT;
}
int sofuu_cuda_ctx_alloc(sofuu_cuda_ctx *ctx, unsigned long long bytes,
                         unsigned long long *out) {
    (void)ctx; (void)bytes; (void)out;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_free(sofuu_cuda_ctx *ctx, unsigned long long ptr) {
    (void)ctx; (void)ptr;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_upload(sofuu_cuda_ctx *ctx, unsigned long long dst,
                          const void *src, size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_download(sofuu_cuda_ctx *ctx, void *dst,
                            unsigned long long src, size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_upload_async(sofuu_cuda_ctx *ctx, unsigned long long dst,
                                const void *src, size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_download_async(sofuu_cuda_ctx *ctx, void *dst,
                                  unsigned long long src, size_t bytes) {
    (void)ctx; (void)dst; (void)src; (void)bytes;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_sync(sofuu_cuda_ctx *ctx) {
    (void)ctx;
    return SOFUU_CUDA_ERR_LAUNCH;
}
int sofuu_cuda_ctx_launch(sofuu_cuda_ctx *ctx, void *function, unsigned gx,
                          unsigned gy, unsigned gz, unsigned bx, unsigned by,
                          unsigned bz, unsigned shared, void **params) {
    (void)ctx; (void)function; (void)gx; (void)gy; (void)gz; (void)bx;
    (void)by; (void)bz; (void)shared; (void)params;
    return SOFUU_CUDA_ERR_LAUNCH;
}
int sofuu_cuda_ctx_host_register(sofuu_cuda_ctx *ctx, void *host, size_t bytes,
                                 unsigned long long *out) {
    (void)ctx; (void)host; (void)bytes; (void)out;
    return SOFUU_CUDA_ERR_MEMORY;
}
int sofuu_cuda_ctx_host_unregister(sofuu_cuda_ctx *ctx, void *host) {
    (void)ctx; (void)host;
    return SOFUU_CUDA_ERR_MEMORY;
}
const char *sofuu_cuda_last_error(void) {
    return "CUDA backend is not available on Windows";
}
int sofuu_cuda_driver_present(void) { return 0; }

#else // POSIX

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Driver ABI types (public cuda.h shapes, no header dependency) ──────────
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

#define SOFUU_CUDA_SUCCESS 0
#define SOFUU_CUDA_MEMHOSTREGISTER_DEVICEMAP 0x02u

// ── Driver function table ──────────────────────────────────────────────────
struct sofuu_cuda_driver {
    void *lib;
    char path[512];
    int version;

    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDriverGetVersion)(int *);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceTotalMem)(size_t *, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
    CUresult (*cuStreamCreate)(CUstream *, unsigned int);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, CUstream, void **,
                               void **);
    CUresult (*cuGetErrorString)(CUresult, const char **);
    CUresult (*cuGetErrorName)(CUresult, const char **);
    CUresult (*cuMemHostRegister)(void *, size_t, unsigned int);
    CUresult (*cuMemHostUnregister)(void *);
    CUresult (*cuMemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned int);
};

struct sofuu_cuda_ctx {
    sofuu_cuda_driver *driver; // borrowed
    CUcontext cu_ctx;
    CUmodule module;
    CUstream stream;
};

static _Thread_local char g_error[640];

static void set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

const char *sofuu_cuda_last_error(void) {
    return g_error[0] ? g_error : "no CUDA error";
}

static int debug_enabled(void) {
    const char *v = getenv("SOFUU_CUDA_DEBUG");
    return v && v[0] && strcmp(v, "0") != 0;
}

static const char *driver_error(const sofuu_cuda_driver *d, CUresult r,
                                char *buf, size_t cap) {
    const char *s = NULL;
    const char *n = NULL;
    if (d) {
        if (d->cuGetErrorString) d->cuGetErrorString(r, &s);
        if (d->cuGetErrorName) d->cuGetErrorName(r, &n);
    }
    if (s && n)
        snprintf(buf, cap, "%s (%s, code %d)", s, n, r);
    else if (s)
        snprintf(buf, cap, "%s (code %d)", s, r);
    else
        snprintf(buf, cap, "CUDA driver error %d", r);
    return buf;
}

#define FAIL_RESULT(d, what, r)                                                \
    do {                                                                       \
        char _eb[256];                                                         \
        set_error("%s: %s", (what), driver_error((d), (r), _eb, sizeof(_eb))); \
        return -1;                                                             \
    } while (0)

// Resolve the versioned name first, then the unversioned one. memcpy keeps the
// ISO C object/function-pointer distinction clean under -Wpedantic.
static void *resolve(void *lib, const char *versioned, const char *plain) {
    void *p = dlsym(lib, versioned);
    if (!p && plain) p = dlsym(lib, plain);
    return p;
}

#define LOAD_REQUIRED(field, versioned, plain)                                 \
    do {                                                                       \
        void *_p = resolve(lib, versioned, plain);                             \
        if (!_p) {                                                             \
            set_error("libcuda: missing required symbol %s%s%s", versioned,    \
                      (plain && strcmp(plain, versioned) != 0) ? " / " : "",   \
                      (plain && strcmp(plain, versioned) != 0) ? plain : "");  \
            dlclose(lib);                                                      \
            return NULL;                                                       \
        }                                                                      \
        memcpy(&d->field, &_p, sizeof(_p));                                    \
    } while (0)

#define LOAD_OPTIONAL(field, versioned, plain)                                 \
    do {                                                                       \
        void *_p = resolve(lib, versioned, plain);                             \
        if (_p) memcpy(&d->field, &_p, sizeof(_p));                            \
        else d->field = NULL;                                                  \
    } while (0)

static void *open_library(const char *path) {
    static const char *candidates[] = {
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/cuda/compat/libcuda.so.1",
    };
    if (path && path[0]) {
        void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!lib && debug_enabled())
            fprintf(stderr, "[cuda] dlopen(%s) failed: %s\n", path, dlerror());
        return lib;
    }
    const char *env = getenv("SOFUU_CUDA_LIB");
    if (env && env[0]) {
        void *lib = dlopen(env, RTLD_NOW | RTLD_LOCAL);
        if (lib) return lib;
        if (debug_enabled())
            fprintf(stderr, "[cuda] dlopen($SOFUU_CUDA_LIB=%s) failed: %s\n",
                    env, dlerror());
    }
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        void *lib = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) {
            if (debug_enabled())
                fprintf(stderr, "[cuda] loaded %s\n", candidates[i]);
            return lib;
        }
        if (debug_enabled())
            fprintf(stderr, "[cuda] dlopen(%s) failed: %s\n", candidates[i],
                    dlerror());
    }
    return NULL;
}

sofuu_cuda_driver *sofuu_cuda_driver_open(const char *path) {
    g_error[0] = '\0';
    void *lib = open_library(path);
    if (!lib) {
        set_error("no CUDA driver library found (tried libcuda.so.1 and the "
                  "documented distro paths; set SOFUU_CUDA_LIB to override)");
        return NULL;
    }

    sofuu_cuda_driver *d = (sofuu_cuda_driver *)calloc(1, sizeof(*d));
    if (!d) {
        dlclose(lib);
        set_error("out of memory allocating the CUDA driver table");
        return NULL;
    }
    d->lib = lib;
    const char *shown = (path && path[0]) ? path : getenv("SOFUU_CUDA_LIB");
    if (!shown) shown = "libcuda.so.1";
    snprintf(d->path, sizeof(d->path), "%s", shown);

    LOAD_REQUIRED(cuInit, "cuInit", NULL);
    LOAD_REQUIRED(cuDriverGetVersion, "cuDriverGetVersion", NULL);
    LOAD_REQUIRED(cuDeviceGetCount, "cuDeviceGetCount", NULL);
    LOAD_REQUIRED(cuDeviceGet, "cuDeviceGet", NULL);
    LOAD_REQUIRED(cuDeviceGetName, "cuDeviceGetName", NULL);
    LOAD_REQUIRED(cuDeviceTotalMem, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
    LOAD_REQUIRED(cuDeviceGetAttribute, "cuDeviceGetAttribute", NULL);
    LOAD_REQUIRED(cuCtxCreate, "cuCtxCreate_v2", "cuCtxCreate");
    LOAD_REQUIRED(cuCtxDestroy, "cuCtxDestroy_v2", "cuCtxDestroy");
    LOAD_REQUIRED(cuCtxSynchronize, "cuCtxSynchronize", NULL);
    LOAD_REQUIRED(cuModuleLoadData, "cuModuleLoadData", NULL);
    LOAD_REQUIRED(cuModuleGetFunction, "cuModuleGetFunction", NULL);
    LOAD_REQUIRED(cuModuleUnload, "cuModuleUnload", NULL);
    LOAD_REQUIRED(cuMemAlloc, "cuMemAlloc_v2", "cuMemAlloc");
    LOAD_REQUIRED(cuMemFree, "cuMemFree_v2", "cuMemFree");
    LOAD_REQUIRED(cuMemcpyHtoD, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    LOAD_REQUIRED(cuMemcpyDtoH, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    LOAD_REQUIRED(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2",
                  "cuMemcpyHtoDAsync");
    LOAD_REQUIRED(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2",
                  "cuMemcpyDtoHAsync");
    LOAD_REQUIRED(cuStreamCreate, "cuStreamCreate", NULL);
    LOAD_REQUIRED(cuStreamSynchronize, "cuStreamSynchronize", NULL);
    LOAD_REQUIRED(cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
    LOAD_REQUIRED(cuLaunchKernel, "cuLaunchKernel", NULL);

    LOAD_OPTIONAL(cuGetErrorString, "cuGetErrorString", NULL);
    LOAD_OPTIONAL(cuGetErrorName, "cuGetErrorName", NULL);
    LOAD_OPTIONAL(cuMemHostRegister, "cuMemHostRegister_v2", "cuMemHostRegister");
    LOAD_OPTIONAL(cuMemHostUnregister, "cuMemHostUnregister", NULL);
    LOAD_OPTIONAL(cuMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2",
                  "cuMemHostGetDevicePointer");

    CUresult r = d->cuInit(0);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuInit(0) failed: %s", driver_error(d, r, eb, sizeof(eb)));
        dlclose(lib);
        free(d);
        return NULL;
    }
    int ver = 0;
    if (d->cuDriverGetVersion(&ver) == SOFUU_CUDA_SUCCESS) d->version = ver;
    if (debug_enabled())
        fprintf(stderr, "[cuda] driver loaded from %s, version %d\n", d->path,
                d->version);
    return d;
}

void sofuu_cuda_driver_close(sofuu_cuda_driver *driver) {
    if (!driver) return;
    if (driver->lib) dlclose(driver->lib);
    driver->lib = NULL;
    free(driver);
}

const char *sofuu_cuda_driver_path(const sofuu_cuda_driver *driver) {
    return driver ? driver->path : "";
}

int sofuu_cuda_driver_version(const sofuu_cuda_driver *driver) {
    return driver ? driver->version : 0;
}

int sofuu_cuda_driver_device_count(const sofuu_cuda_driver *driver) {
    if (!driver) return 0;
    int count = 0;
    CUresult r = driver->cuDeviceGetCount(&count);
    if (r != SOFUU_CUDA_SUCCESS) {
        FAIL_RESULT(driver, "cuDeviceGetCount", r);
    }
    return count > 0 ? count : 0;
}

int sofuu_cuda_driver_device_info(const sofuu_cuda_driver *driver, int ordinal,
                                  char *name, size_t name_cap, int *cc_major,
                                  int *cc_minor, unsigned long long *total_mem,
                                  int *integrated, int *unified) {
    if (!driver) {
        set_error("CUDA driver is not open");
        return SOFUU_CUDA_ERR_LOAD;
    }
    CUdevice dev;
    CUresult r = driver->cuDeviceGet(&dev, ordinal);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuDeviceGet(%d) failed: %s", ordinal,
                  driver_error((sofuu_cuda_driver *)driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_DRIVER;
    }
    if (name && name_cap) {
        name[0] = '\0';
        r = driver->cuDeviceGetName(name, (int)name_cap, dev);
        if (r != SOFUU_CUDA_SUCCESS) {
            char eb[256];
            set_error("cuDeviceGetName failed: %s",
                      driver_error((sofuu_cuda_driver *)driver, r, eb,
                                   sizeof(eb)));
            return SOFUU_CUDA_ERR_DRIVER;
        }
        name[name_cap - 1] = '\0';
    }
    int major = -1, minor = -1, integ = 0, unif = 0;
    if (driver->cuDeviceGetAttribute(&major, SOFUU_CUDA_ATTR_CC_MAJOR, dev) !=
        SOFUU_CUDA_SUCCESS)
        major = -1;
    if (driver->cuDeviceGetAttribute(&minor, SOFUU_CUDA_ATTR_CC_MINOR, dev) !=
        SOFUU_CUDA_SUCCESS)
        minor = -1;
    if (driver->cuDeviceGetAttribute(&integ, SOFUU_CUDA_ATTR_INTEGRATED, dev) !=
        SOFUU_CUDA_SUCCESS)
        integ = 0;
    if (driver->cuDeviceGetAttribute(&unif, SOFUU_CUDA_ATTR_UNIFIED_ADDRESSING,
                                     dev) != SOFUU_CUDA_SUCCESS)
        unif = 0;
    if (cc_major) *cc_major = major;
    if (cc_minor) *cc_minor = minor;
    if (integrated) *integrated = integ;
    if (unified) *unified = unif;
    if (total_mem) {
        size_t bytes = 0;
        if (driver->cuDeviceTotalMem(&bytes, dev) == SOFUU_CUDA_SUCCESS)
            *total_mem = (unsigned long long)bytes;
        else
            *total_mem = 0;
    }
    return SOFUU_CUDA_OK;
}

sofuu_cuda_ctx *sofuu_cuda_ctx_create(sofuu_cuda_driver *driver, int ordinal) {
    g_error[0] = '\0';
    if (!driver) {
        set_error("CUDA driver is not open");
        return NULL;
    }
    CUdevice dev;
    CUresult r = driver->cuDeviceGet(&dev, ordinal);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuDeviceGet(%d) failed: %s", ordinal,
                  driver_error(driver, r, eb, sizeof(eb)));
        return NULL;
    }
    sofuu_cuda_ctx *ctx = (sofuu_cuda_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_error("out of memory allocating the CUDA context");
        return NULL;
    }
    ctx->driver = driver;
    r = driver->cuCtxCreate(&ctx->cu_ctx, 0, dev);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuCtxCreate failed: %s", driver_error(driver, r, eb,
                                                         sizeof(eb)));
        free(ctx);
        return NULL;
    }
    r = driver->cuStreamCreate(&ctx->stream, 0 /* CU_STREAM_DEFAULT */);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuStreamCreate failed: %s", driver_error(driver, r, eb,
                                                            sizeof(eb)));
        driver->cuCtxDestroy(ctx->cu_ctx);
        free(ctx);
        return NULL;
    }
    return ctx;
}

void sofuu_cuda_ctx_destroy(sofuu_cuda_ctx *ctx) {
    if (!ctx) return;
    sofuu_cuda_driver *d = ctx->driver;
    if (ctx->module) d->cuModuleUnload(ctx->module);
    if (ctx->stream) d->cuStreamDestroy(ctx->stream);
    if (ctx->cu_ctx) d->cuCtxDestroy(ctx->cu_ctx);
    ctx->module = NULL;
    ctx->stream = NULL;
    ctx->cu_ctx = NULL;
    free(ctx);
}

int sofuu_cuda_ctx_load_ptx(sofuu_cuda_ctx *ctx, const char *ptx) {
    if (!ctx || !ptx) {
        set_error("cuModuleLoadData: null context or PTX");
        return SOFUU_CUDA_ERR_ARG;
    }
    if (ctx->module) {
        ctx->driver->cuModuleUnload(ctx->module);
        ctx->module = NULL;
    }
    CUresult r = ctx->driver->cuModuleLoadData(&ctx->module, (const void *)ptx);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuModuleLoadData failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        ctx->module = NULL;
        return SOFUU_CUDA_ERR_CONTEXT;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_function(sofuu_cuda_ctx *ctx, const char *name,
                            void **out_function) {
    if (!ctx || !name || !out_function) {
        set_error("cuModuleGetFunction: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    if (!ctx->module) {
        set_error("cuModuleGetFunction: no PTX module loaded");
        return SOFUU_CUDA_ERR_CONTEXT;
    }
    CUfunction fn = NULL;
    CUresult r = ctx->driver->cuModuleGetFunction(&fn, ctx->module, name);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuModuleGetFunction(%s) failed: %s", name,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_CONTEXT;
    }
    *out_function = (void *)fn;
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_alloc(sofuu_cuda_ctx *ctx, unsigned long long bytes,
                         unsigned long long *out_ptr) {
    if (!ctx || !out_ptr || bytes == 0) {
        set_error("cuMemAlloc: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUdeviceptr p = 0;
    CUresult r = ctx->driver->cuMemAlloc(&p, (size_t)bytes);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemAlloc(%llu) failed: %s", bytes,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    *out_ptr = (unsigned long long)p;
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_free(sofuu_cuda_ctx *ctx, unsigned long long dev_ptr) {
    if (!ctx) return SOFUU_CUDA_ERR_ARG;
    if (!dev_ptr) return SOFUU_CUDA_OK;
    CUresult r = ctx->driver->cuMemFree((CUdeviceptr)dev_ptr);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemFree failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_upload(sofuu_cuda_ctx *ctx, unsigned long long dev_dst,
                          const void *host_src, size_t bytes) {
    if (!ctx || !host_src || !dev_dst || bytes == 0) {
        set_error("cuMemcpyHtoD: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuMemcpyHtoD((CUdeviceptr)dev_dst, host_src,
                                           bytes);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemcpyHtoD(%zu) failed: %s", bytes,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_download(sofuu_cuda_ctx *ctx, void *host_dst,
                            unsigned long long dev_src, size_t bytes) {
    if (!ctx || !host_dst || !dev_src || bytes == 0) {
        set_error("cuMemcpyDtoH: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuMemcpyDtoH(host_dst, (CUdeviceptr)dev_src,
                                           bytes);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemcpyDtoH(%zu) failed: %s", bytes,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_upload_async(sofuu_cuda_ctx *ctx, unsigned long long dev_dst,
                                const void *host_src, size_t bytes) {
    if (!ctx || !host_src || !dev_dst || bytes == 0) {
        set_error("cuMemcpyHtoDAsync: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuMemcpyHtoDAsync((CUdeviceptr)dev_dst, host_src,
                                                bytes, ctx->stream);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemcpyHtoDAsync(%zu) failed: %s", bytes,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_download_async(sofuu_cuda_ctx *ctx, void *host_dst,
                                  unsigned long long dev_src, size_t bytes) {
    if (!ctx || !host_dst || !dev_src || bytes == 0) {
        set_error("cuMemcpyDtoHAsync: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuMemcpyDtoHAsync(host_dst, (CUdeviceptr)dev_src,
                                                bytes, ctx->stream);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemcpyDtoHAsync(%zu) failed: %s", bytes,
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_sync(sofuu_cuda_ctx *ctx) {
    if (!ctx || !ctx->stream) {
        set_error("cuStreamSynchronize: no stream");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuStreamSynchronize(ctx->stream);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuStreamSynchronize failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_LAUNCH;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_launch(sofuu_cuda_ctx *ctx, void *function, unsigned gx,
                          unsigned gy, unsigned gz, unsigned bx, unsigned by,
                          unsigned bz, unsigned shared, void **params) {
    if (!ctx || !function || gx == 0 || bx == 0) {
        set_error("cuLaunchKernel: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    CUresult r = ctx->driver->cuLaunchKernel(
        (CUfunction)function, gx, gy, gz, bx, by, bz, shared, ctx->stream,
        params, NULL);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuLaunchKernel failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_LAUNCH;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_host_register(sofuu_cuda_ctx *ctx, void *host_ptr,
                                 size_t bytes, unsigned long long *out_dev_ptr) {
    if (!ctx || !host_ptr || !out_dev_ptr || bytes == 0) {
        set_error("cuMemHostRegister: invalid argument");
        return SOFUU_CUDA_ERR_ARG;
    }
    if (!ctx->driver->cuMemHostRegister ||
        !ctx->driver->cuMemHostGetDevicePointer) {
        set_error("cuMemHostRegister: driver lacks host-register symbols");
        return SOFUU_CUDA_ERR_LOAD;
    }
    CUresult r = ctx->driver->cuMemHostRegister(
        host_ptr, bytes, SOFUU_CUDA_MEMHOSTREGISTER_DEVICEMAP);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemHostRegister failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    CUdeviceptr dev = 0;
    r = ctx->driver->cuMemHostGetDevicePointer(&dev, host_ptr, 0);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        ctx->driver->cuMemHostUnregister(host_ptr);
        set_error("cuMemHostGetDevicePointer failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    *out_dev_ptr = (unsigned long long)dev;
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_ctx_host_unregister(sofuu_cuda_ctx *ctx, void *host_ptr) {
    if (!ctx || !host_ptr) return SOFUU_CUDA_ERR_ARG;
    if (!ctx->driver->cuMemHostUnregister) return SOFUU_CUDA_ERR_LOAD;
    CUresult r = ctx->driver->cuMemHostUnregister(host_ptr);
    if (r != SOFUU_CUDA_SUCCESS) {
        char eb[256];
        set_error("cuMemHostUnregister failed: %s",
                  driver_error(ctx->driver, r, eb, sizeof(eb)));
        return SOFUU_CUDA_ERR_MEMORY;
    }
    return SOFUU_CUDA_OK;
}

int sofuu_cuda_driver_present(void) {
    sofuu_cuda_driver *d = sofuu_cuda_driver_open(NULL);
    if (!d) return 0;
    int count = sofuu_cuda_driver_device_count(d);
    sofuu_cuda_driver_close(d);
    return count > 0;
}

#endif // _WIN32
