// src/cuda/cuda_loader.h — minimal, opaque CUDA Driver API loader for Sofuu.
//
// PLAN-GPU-CUDA C0. Sofuu never links the CUDA toolkit or libcudart. This
// translation unit dlopen()s the display driver's libcuda.so.1 at RUNTIME and
// resolves only the stable Driver API subset the engine needs. Every entry
// point below is safe to call on a machine with no NVIDIA driver: a missing
// library yields a NULL handle, not a crash or a link error.
//
// The same API is consumed by two callers:
//   * crates/sofuu-core/src/infer/cuda.rs   (safe Rust wrapper)
//   * src/cuda/parity/gpu_check.cpp         (standalone Kaggle T4 harness)
// so the driver-resolution rules can never drift between them.
//
// Symbol-name note: modern NVIDIA drivers export versioned symbols. CUDA's own
// cuda.h maps the public names onto the _v2 forms with #defines:
//
//   cuMemAlloc -> cuMemAlloc_v2        cuCtxCreate -> cuCtxCreate_v2
//   cuMemFree  -> cuMemFree_v2         cuMemcpyDtoH -> cuMemcpyDtoH_v2
//
// The loader therefore tries the _v2 name first and falls back to the plain
// name, which keeps it working on older drivers as well.

#ifndef SOFUU_CUDA_LOADER_H
#define SOFUU_CUDA_LOADER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sofuu_cuda_driver sofuu_cuda_driver;
typedef struct sofuu_cuda_ctx sofuu_cuda_ctx;

// Device attributes (values from the public cuda.h enum).
#define SOFUU_CUDA_ATTR_INTEGRATED 18
#define SOFUU_CUDA_ATTR_UNIFIED_ADDRESSING 41
#define SOFUU_CUDA_ATTR_CC_MAJOR 75
#define SOFUU_CUDA_ATTR_CC_MINOR 76
#define SOFUU_CUDA_ATTR_MULTIPROCESSOR_COUNT 16

// Status codes returned by this API. 0 means success.
#define SOFUU_CUDA_OK 0
#define SOFUU_CUDA_ERR_LOAD -1      // no driver library / missing symbol
#define SOFUU_CUDA_ERR_DRIVER -2    // cuInit / device query failed
#define SOFUU_CUDA_ERR_CONTEXT -3   // context / stream / module failure
#define SOFUU_CUDA_ERR_MEMORY -4    // allocation / copy failure
#define SOFUU_CUDA_ERR_LAUNCH -5    // kernel launch / synchronisation failure
#define SOFUU_CUDA_ERR_ARG -6       // invalid argument / unsupported request

// Open the driver. `path` may be NULL (search the documented list), an
// explicit library path, or a bare "libcuda.so.1". Returns NULL on failure;
// call sofuu_cuda_last_error() for the reason. On a machine with no NVIDIA
// driver this returns NULL without printing anything.
sofuu_cuda_driver *sofuu_cuda_driver_open(const char *path);

// Release the handle. Does not destroy contexts created from it; callers must
// destroy those first (sofuu_cuda_ctx_destroy). NULL is a no-op.
void sofuu_cuda_driver_close(sofuu_cuda_driver *driver);

// Resolved library path ("" when closed).
const char *sofuu_cuda_driver_path(const sofuu_cuda_driver *driver);

// Driver version (e.g. 12020 for 12.2) and device count. Both return 0 on a
// closed/absent driver.
int sofuu_cuda_driver_version(const sofuu_cuda_driver *driver);
int sofuu_cuda_driver_device_count(const sofuu_cuda_driver *driver);

// One-line human-readable identity, e.g.
// "NVIDIA Tesla T4 cc7.5 15109 MiB integrated=0 unified=1". Returns 0 on
// success, negative on failure. `name` is NUL-terminated.
int sofuu_cuda_driver_device_info(const sofuu_cuda_driver *driver, int ordinal,
                                  char *name, size_t name_cap,
                                  int *cc_major, int *cc_minor,
                                  unsigned long long *total_mem_bytes,
                                  int *integrated, int *unified_addressing);

// Create a context on `ordinal`, its module slot, and one async stream.
// Returns NULL on failure. The context becomes current on the calling thread
// (the CUDA driver API's cuCtxCreate semantics).
sofuu_cuda_ctx *sofuu_cuda_ctx_create(sofuu_cuda_driver *driver, int ordinal);

// Unload the module, destroy the stream, then the context. NULL is a no-op.
void sofuu_cuda_ctx_destroy(sofuu_cuda_ctx *ctx);

// JIT-load a PTX text module. `ptx` must be NUL-terminated and stay valid only
// for the duration of the call (the driver copies it). Loading a second module
// replaces the first.
int sofuu_cuda_ctx_load_ptx(sofuu_cuda_ctx *ctx, const char *ptx);

// Resolve a kernel entry point from the loaded module.
int sofuu_cuda_ctx_function(sofuu_cuda_ctx *ctx, const char *name,
                            void **out_function);

// Function attributes (the two values the occupancy arithmetic needs).
// Values from the public cuda.h CUfunction_attribute enum.
#define SOFUU_CUDA_FUNC_ATTR_MAX_THREADS 0
#define SOFUU_CUDA_FUNC_ATTR_NUM_REGS 4

// Query one attribute of a resolved kernel. `function` is the value returned by
// sofuu_cuda_ctx_function. Returns SOFUU_CUDA_ERR_LOAD when the driver does not
// export cuFuncGetAttribute (older drivers), so callers can treat the value as
// unknown rather than as zero.
int sofuu_cuda_ctx_function_attr(sofuu_cuda_ctx *ctx, void *function,
                                 int attrib, int *out_value);

// Device memory. `bytes` must be non-zero; cuMemAlloc returns 256-byte aligned
// addresses. Freeing 0 is a no-op.
int sofuu_cuda_ctx_alloc(sofuu_cuda_ctx *ctx, unsigned long long bytes,
                         unsigned long long *out_ptr);
int sofuu_cuda_ctx_free(sofuu_cuda_ctx *ctx, unsigned long long dev_ptr);

// Synchronous host<->device copies.
int sofuu_cuda_ctx_upload(sofuu_cuda_ctx *ctx, unsigned long long dev_dst,
                          const void *host_src, size_t bytes);
int sofuu_cuda_ctx_download(sofuu_cuda_ctx *ctx, void *host_dst,
                            unsigned long long dev_src, size_t bytes);

// Asynchronous copies on the context's stream (call ctx_sync before reading).
int sofuu_cuda_ctx_upload_async(sofuu_cuda_ctx *ctx, unsigned long long dev_dst,
                                const void *host_src, size_t bytes);
int sofuu_cuda_ctx_download_async(sofuu_cuda_ctx *ctx, void *host_dst,
                                  unsigned long long dev_src, size_t bytes);
int sofuu_cuda_ctx_sync(sofuu_cuda_ctx *ctx);

// ── CUDA graphs ────────────────────────────────────────────────────────────
// Capture a fixed sequence of launches once, then replay it with one driver
// call per token. All buffers must already exist at capture time, and any
// per-token scalar has to live in device memory (a kernel parameter is baked
// into the graph). Returns non-zero when the driver lacks the entry points.
int sofuu_cuda_ctx_graphs_supported(sofuu_cuda_ctx *ctx);
int sofuu_cuda_ctx_begin_capture(sofuu_cuda_ctx *ctx);
int sofuu_cuda_ctx_end_capture(sofuu_cuda_ctx *ctx, void **out_exec);
int sofuu_cuda_ctx_graph_launch(sofuu_cuda_ctx *ctx, void *exec);
int sofuu_cuda_ctx_graph_exec_free(sofuu_cuda_ctx *ctx, void *exec);

// Enqueue a kernel. `params` is the CUDA Driver API argument array: one entry
// per kernel parameter, each pointing at the host-side value/device-pointer
// variable. `extra` must be NULL.
int sofuu_cuda_ctx_launch(sofuu_cuda_ctx *ctx, void *function,
                          unsigned grid_x, unsigned grid_y, unsigned grid_z,
                          unsigned block_x, unsigned block_y, unsigned block_z,
                          unsigned shared_mem_bytes, void **params);

// Pin host memory for the Jetson/chiplet zero-copy path. Registers `bytes`
// starting at `host_ptr` with CU_MEMHOSTREGISTER_DEVICEMAP and returns the
// device-visible address. The caller keeps ownership of host_ptr and must call
// unregister before freeing it. Not used by the C1 kernels yet; exposed so the
// C3 unified-memory path does not need a second loader.
int sofuu_cuda_ctx_host_register(sofuu_cuda_ctx *ctx, void *host_ptr,
                                 size_t bytes, unsigned long long *out_dev_ptr);
int sofuu_cuda_ctx_host_unregister(sofuu_cuda_ctx *ctx, void *host_ptr);

// Last error for this thread. Always NUL-terminated. The buffer is overwritten
// by the next failing call.
const char *sofuu_cuda_last_error(void);

// Convenience for humans/tests: 1 when an NVIDIA driver library is locatable,
// 0 otherwise. Does not initialise CUDA and does not create a context.
int sofuu_cuda_driver_present(void);

// Non-zero when SOFUU_CUDA_DEBUG is set to anything but empty or "0". This is
// the single source of truth for CUDA tracing; the Rust side calls it rather
// than parsing the same variable a second time.
int sofuu_cuda_debug_enabled(void);

#ifdef __cplusplus
}
#endif

#endif // SOFUU_CUDA_LOADER_H
