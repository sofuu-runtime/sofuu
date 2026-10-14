// src/simd/infer_kernels.cpp — High-performance C++ inference microkernels for Sofuu
//
// Optimized for ARM NEON (Apple Silicon, ARM64) and x86 AVX2/FMA.
// Keeps compute-heavy inner loops in C++ where intrinsics and auto-vectorization
// beat Rust abstractions, while the rest of the runtime remains safe Rust.

#include "infer_kernels.h"
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdio>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SOFUU_NEON 1
#endif

#if defined(__aarch64__)
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define SOFUU_AVX2 1
#endif

#include "platform/platform_sync.h"
#include "platform/platform_topology.h"
#include "platform/platform_affinity.h"

// ── Multi-core CPU Worker Thread Pool (Lock-Free Spin-Wait Barrier) ──
// Eliminates the Linux/Android futex syscall storm.
// Instead of hundreds of pthread_cond_broadcast / futex syscalls per token,
// worker threads synchronize in user space via atomic counters and hardware yield pauses.
// When idle (>2ms), workers park on condition variables to save power.

struct alignas(64) AlignedAtomicU64 {
    std::atomic<uint64_t> val{0};
};

struct WorkerSlice {
    size_t start = 0;
    size_t end = 0;
};

class WorkerThreadPool {
public:
    WorkerThreadPool()
        : stop_(false),
          user_threads_(0),
          next_seq_(1),
          work_seq_(0),
          sleeping_count_(0),
          current_task_(nullptr)
    {
        int default_threads = detect_optimal_threads();
        set_thread_count(default_threads);
    }

    ~WorkerThreadPool() {
        stop_workers();
    }

    void set_thread_count(int n) {
        if (n <= 0) {
            n = detect_optimal_threads();
        }
        std::unique_lock<std::mutex> lock(config_mutex_);
        if (n == user_threads_ && !workers_.empty()) return;

        stop_workers_locked();

        user_threads_ = n;
        size_t bg_count = (n > 1) ? (size_t)(n - 1) : 0;
        slices_.resize(bg_count);
        worker_done_ = std::make_unique<AlignedAtomicU64[]>(bg_count);
        for (size_t i = 0; i < bg_count; ++i) {
            worker_done_[i].val.store(0, std::memory_order_relaxed);
        }

        stop_.store(false, std::memory_order_relaxed);
        work_seq_.store(0, std::memory_order_relaxed);
        next_seq_ = 1;
        sleeping_count_.store(0, std::memory_order_relaxed);

        sofuu_set_current_thread_affinity(0);
        for (size_t i = 0; i < bg_count; ++i) {
            workers_.emplace_back(&WorkerThreadPool::worker_loop, this, i);
            sofuu_set_worker_affinity(workers_.back(), (int)(i + 1));
        }
    }

    int get_thread_count() const {
        return user_threads_;
    }

    template <typename F>
    void parallel_for(size_t start, size_t end, F &&task) {
        size_t total = end - start;
        if (total == 0) return;

        if (total <= 64 || user_threads_ <= 1 || workers_.empty()) {
            task(start, end);
            return;
        }

        size_t n_workers = workers_.size() + 1;
        size_t chunk = (total + n_workers - 1) / n_workers;

        std::function<void(size_t, size_t)> fn = task;
        current_task_ = &fn;

        // Partition work across background threads
        for (size_t i = 0; i < workers_.size(); ++i) {
            size_t s = start + (i + 1) * chunk;
            size_t e = (s < end) ? std::min(s + chunk, end) : end;
            slices_[i].start = s;
            slices_[i].end = e;
        }

        // Publish task atomically with monotonic sequence number
        uint64_t target_seq = next_seq_++;
        work_seq_.store(target_seq, std::memory_order_release);

        // If any worker thread parked in deep sleep during idle periods, wake them
        if (sleeping_count_.load(std::memory_order_relaxed) > 0) {
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            cv_.notify_all();
        }

        // Main thread executes chunk 0
        size_t main_s = start;
        size_t main_e = std::min(start + chunk, end);
        if (main_s < main_e) {
            task(main_s, main_e);
        }

        // Barrier: Wait for all background workers to finish this specific job sequence
        for (size_t i = 0; i < workers_.size(); ++i) {
            int spins = 0;
            constexpr int MAIN_SPIN_LIMIT = 5000;
            while (worker_done_[i].val.load(std::memory_order_acquire) < target_seq) {
                if (++spins < MAIN_SPIN_LIMIT) {
                    sofuu_spin_pause();
                } else {
                    if (sleeping_count_.load(std::memory_order_relaxed) > 0) {
                        std::unique_lock<std::mutex> lock(sleep_mutex_);
                        cv_.notify_all();
                    }
                    std::this_thread::yield();
                }
            }
        }

        current_task_ = nullptr;
    }

private:
    void worker_loop(size_t worker_id) {
        uint64_t my_seq = 0;
        constexpr int IDLE_SPIN_LIMIT = 500000; // ~1-2ms spin before parking

        while (!stop_.load(std::memory_order_relaxed)) {
            int idle_spins = 0;
            while (work_seq_.load(std::memory_order_acquire) == my_seq) {
                if (stop_.load(std::memory_order_relaxed)) return;
                if (++idle_spins < IDLE_SPIN_LIMIT) {
                    sofuu_spin_pause();
                } else {
                    // Park worker on condition variable when idle to save CPU/battery
                    std::unique_lock<std::mutex> lock(sleep_mutex_);
                    sleeping_count_.fetch_add(1, std::memory_order_relaxed);
                    cv_.wait(lock, [this, my_seq]() {
                        return stop_.load(std::memory_order_relaxed) ||
                               work_seq_.load(std::memory_order_acquire) > my_seq;
                    });
                    sleeping_count_.fetch_sub(1, std::memory_order_relaxed);
                    idle_spins = 0;
                }
            }

            if (stop_.load(std::memory_order_relaxed)) return;

            uint64_t target_seq = work_seq_.load(std::memory_order_acquire);

            auto *task_ptr = current_task_;
            size_t s = slices_[worker_id].start;
            size_t e = slices_[worker_id].end;
            if (task_ptr && s < e) {
                (*task_ptr)(s, e);
            }

            // Signal slice completion for this target_seq atomically
            my_seq = target_seq;
            worker_done_[worker_id].val.store(target_seq, std::memory_order_release);
        }
    }

    void stop_workers() {
        std::unique_lock<std::mutex> lock(config_mutex_);
        stop_workers_locked();
    }

    void stop_workers_locked() {
        stop_.store(true, std::memory_order_release);
        work_seq_.fetch_add(1, std::memory_order_release);
        {
            std::unique_lock<std::mutex> lock(sleep_mutex_);
            cv_.notify_all();
        }
        for (auto &t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        worker_done_.reset();
    }

    std::vector<std::thread> workers_;
    std::vector<WorkerSlice> slices_;
    std::unique_ptr<AlignedAtomicU64[]> worker_done_;
    std::mutex config_mutex_;
    std::mutex sleep_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
    int user_threads_;
    uint64_t next_seq_;
    std::atomic<uint64_t> work_seq_;
    std::atomic<uint32_t> sleeping_count_;
    const std::function<void(size_t, size_t)> *current_task_;
};

static WorkerThreadPool g_pool;

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;       // IEEE 754 half-precision scale (16-bit)
    uint8_t  qs[16];  // 32 nibbles (4-bit values)
};

struct block_q8_0 {
    uint16_t d;       // IEEE 754 half-precision scale (16-bit)
    int8_t   qs[32];  // 32 signed int8 values
};

struct block_q4_K {
    uint16_t d;          // super-block scale (fp16)
    uint16_t dmin;       // super-block min (fp16)
    uint8_t scales[12];  // 6-bit scales and mins
    uint8_t qs[128];     // 4-bit quants (256 nibbles)
};

struct block_q6_K {
    uint8_t ql[128];     // quants, lower 4 bits (128 bytes)
    uint8_t qh[64];      // quants, upper 2 bits (64 bytes)
    int8_t  scales[16];  // scales, 8-bit signed (16 bytes)
    uint16_t d;          // super-block scale (fp16, 2 bytes)
};
#pragma pack(pop)

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

// Branch-free IEEE 754 half to single float conversion
static inline float f16_to_f32_val(uint16_t h) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 f = *(__fp16 *)&h;
    return (float)f;
#else
    uint32_t sign = ((uint32_t)h >> 15) & 1;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign << 31;
            float val;
            std::memcpy(&val, &bits, 4);
            return val;
        }
        uint32_t e = 113;
        uint32_t m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e--;
        }
        uint32_t bits = (sign << 31) | (e << 23) | ((m & 0x3ff) << 13);
        float val;
        std::memcpy(&val, &bits, 4);
        return val;
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | (0xff << 23) | (mant << 13);
        float val;
        std::memcpy(&val, &bits, 4);
        return val;
    }
    uint32_t exp_f32 = exp + 127 - 15;
    uint32_t mant_f32 = mant << 13;
    uint32_t bits = (sign << 31) | (exp_f32 << 23) | mant_f32;
    float val;
    std::memcpy(&val, &bits, 4);
    return val;
#endif
}

static inline uint16_t f32_to_f16_val(float val) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    __fp16 h = (__fp16)val;
    uint16_t r;
    std::memcpy(&r, &h, sizeof(r));
    return r;
#else
    uint32_t x;
    std::memcpy(&x, &val, 4);
    uint32_t sign = (x >> 31) & 1;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00);
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
#endif
}

#include "arch/arm_cortex_a76.h"

// ── Packed-tile GEMM/GEMV kernels (ported from llama.cpp repack *_generic) ──
#if defined(__aarch64__)
#include "arch/a76_repack.h"
#include <mutex>
#include <unordered_map>

struct sofuu_tile_entry {
    std::vector<uint8_t> data;
    std::vector<uint8_t> sc;   // q4_K decoded-scale sidecar (tile_sc_q4_Kx8 per block)
    unsigned type;
    size_t rows, cols;
    uint64_t fp;
    bool has_sc = false;
};
static std::mutex g_tile_mutex;
static std::unordered_map<const void *, sofuu_tile_entry> g_tile_map;

// 1 = decoded-scale sidecars disabled (default), 0 = enabled via
// SOFUU_SCALE_SIDECAR=1. Measured on Pi 5 (2026-10-14): prefill GEMM is
// unaffected (the in-loop 12-byte scale decode overlaps the NEON-pipe wall,
// removing it is not a prefill win) while the +33% tile bytes slow the
// DRAM-bound q4_K GEMV decode ~15-20%. Kept behind an opt-in because the
// kernels are bit-exact and useful as a harness for scale-path work.
// sofuu_tile_scale_sidecar_mode() flips it at runtime (tests).
static int &scale_sidecar_disabled_state() {
    static int v = getenv("SOFUU_SCALE_SIDECAR") ? 0 : 1;
    return v;
}

static uint64_t sofuu_tile_fingerprint(const void *w) {
    const uint64_t *p = (const uint64_t *)w;
    return p[0] ^ (p[1] * 3) ^ (p[2] * 7) ^ (p[3] * 11);
}

extern "C" void sofuu_tile_scale_sidecar_mode(int disable) {
    scale_sidecar_disabled_state() = disable ? 1 : 0;
}

extern "C" int sofuu_tile_warm(const void *w, size_t rows, size_t cols, unsigned type) {
    if (getenv("SOFUU_NO_TILES")) return -1; // runtime kill-switch
    bool aligned;
    const bool want_sc = (type == 12) && (scale_sidecar_disabled_state() == 0);
    if (type == 12 || type == 14) aligned = (rows % 8 == 0 && cols % 256 == 0);
    else if (type == 8) aligned = (rows % 4 == 0 && cols % 32 == 0);
    else return -1;
    if (!aligned || rows == 0 || cols == 0) return -1;
    const uint64_t fp = sofuu_tile_fingerprint(w);
    {
        std::lock_guard<std::mutex> lk(g_tile_mutex);
        auto it = g_tile_map.find(w);
        if (it != g_tile_map.end() && it->second.type == type && it->second.rows == rows &&
            it->second.cols == cols && it->second.fp == fp && it->second.has_sc == want_sc) return 0;
    }
    sofuu_tile_entry e;
    e.type = type; e.rows = rows; e.cols = cols; e.fp = fp;
    if (type == 12) {
        const int64_t nb = cols / 256, r8 = rows / 8;
        e.data.resize((size_t)r8 * nb * sizeof(tile_block_q4_Kx8));
        uint8_t *dst = e.data.data();
        const uint8_t *src = (const uint8_t *)w;
        // Repack is a pure per-row-group data shuffle, so splitting by
        // 8-row group is bit-identical to the serial whole-matrix call;
        // small matrices stay on the serial path to skip pool overhead.
        if (r8 >= 16) {
            g_pool.parallel_for(0, (size_t)r8, [=, &e](size_t g0, size_t g1) {
                sofuu_repack_q4k_8bl_range(dst, src, nb, g0, g1, 4);
            });
        } else {
            sofuu_repack_q4k_8bl(dst, src, rows, nb, 4);
        }
#if defined(__ARM_FEATURE_DOTPROD)
        if (want_sc) {
            static_assert(sizeof(tile_sc_q4_Kx8) == 384, "q4_K sidecar layout drift");
            const size_t n_blk = e.data.size() / sizeof(tile_block_q4_Kx8);
            e.sc.resize(n_blk * sizeof(tile_sc_q4_Kx8));
            tile_sc_q4_Kx8 *         scs  = (tile_sc_q4_Kx8 *)e.sc.data();
            const tile_block_q4_Kx8 *blks = (const tile_block_q4_Kx8 *)e.data.data();
            auto fill_sidecar = [=](size_t i0, size_t i1) {
                for (size_t i = i0; i < i1; i++) {
                    for (int sb = 0; sb < QK_K / 64; sb++) {
                        for (int h = 0; h < 2; h++) {
                            int8_t    aux[8];
                            int16x8_t mins;
                            decode_q_Kx8_6bit_scales(&blks[i].scales[sb * 24 + h * 12], &mins, aux);
                            float *scl = scs[i].sb_scales[sb * 2 + h];
                            for (int r = 0; r < 8; r++) scl[r] = (float)aux[r];
                            vst1q_s16(scs[i].sb_mins[sb * 2 + h], mins);
                        }
                    }
                }
            };
            if (n_blk >= 16) {
                g_pool.parallel_for(0, n_blk, fill_sidecar);
            } else {
                fill_sidecar(0, n_blk);
            }
            e.has_sc = true;
        }
#endif
    } else if (type == 14) {
        const int64_t nb = cols / 256, r8 = rows / 8;
        e.data.resize((size_t)r8 * nb * sizeof(tile_block_q6_Kx8));
        if (r8 >= 16) {
            g_pool.parallel_for(0, (size_t)r8, [=, &e](size_t g0, size_t g1) {
                sofuu_repack_q6k_8bl_range(e.data.data(), (const uint8_t *)w, nb, g0, g1, 4);
            });
        } else {
            sofuu_repack_q6k_8bl(e.data.data(), (const uint8_t *)w, rows, nb, 4);
        }
    } else {
        const int64_t nb = cols / 32, r4 = rows / 4;
        e.data.resize((size_t)r4 * nb * sizeof(tile_block_q8_0x4));
        if (r4 >= 16) {
            g_pool.parallel_for(0, (size_t)r4, [=, &e](size_t g0, size_t g1) {
                sofuu_repack_q8_0_4bl_range(e.data.data(), (const uint8_t *)w, nb, g0, g1, 4);
            });
        } else {
            sofuu_repack_q8_0_4bl(e.data.data(), (const uint8_t *)w, rows, nb, 4);
        }
    }
    std::lock_guard<std::mutex> lk(g_tile_mutex);
    g_tile_map[w] = std::move(e);
    return 0;
}

static const uint8_t * sofuu_tile_lookup(const void *w, unsigned type, size_t rows, size_t cols) {
    const uint64_t fp = sofuu_tile_fingerprint(w);
    std::lock_guard<std::mutex> lk(g_tile_mutex);
    auto it = g_tile_map.find(w);
    if (it == g_tile_map.end()) return nullptr;
    const sofuu_tile_entry &e = it->second;
    if (e.type == type && e.rows == rows && e.cols == cols && e.fp == fp) return e.data.data();
    return nullptr;
}

// Same keying as sofuu_tile_lookup, additionally hands back the
// decoded-scale sidecar when one was warmed for this entry (q4_K only;
// *sc_out stays nullptr otherwise and the caller uses the original kernels).
static const uint8_t * sofuu_tile_lookup_sc(const void *w, unsigned type, size_t rows, size_t cols, const uint8_t **sc_out) {
    const uint64_t fp = sofuu_tile_fingerprint(w);
    std::lock_guard<std::mutex> lk(g_tile_mutex);
    *sc_out = nullptr;
    auto it = g_tile_map.find(w);
    if (it == g_tile_map.end()) return nullptr;
    const sofuu_tile_entry &e = it->second;
    if (e.type == type && e.rows == rows && e.cols == cols && e.fp == fp) {
        *sc_out = e.has_sc ? e.sc.data() : nullptr;
        return e.data.data();
    }
    return nullptr;
}

// Row quantizer for the plain q8_0 tile-block layout (decode GEMV path).
static inline void sofuu_quantize_row_q8_0_tile(const float * __restrict x, tile_block_q8_0 * __restrict y, int64_t cols) {
    const int64_t nb = cols / 32;
    for (int64_t i = 0; i < nb; ++i) {
        const float *xb = x + i * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = MAX(amax, fabsf(xb[j]));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        for (int j = 0; j < 32; ++j) y[i].qs[j] = (int8_t)nearest_int(xb[j] * id);
    }
}

#define SOFUU_TILE_TYPES_OK(t) ((t) == 12 || (t) == 14 || (t) == 8)
#endif  // __aarch64__

#include <vector>

// Reusable per-thread scratch for batched GEMM paths: avoids a fresh
// zero-filled vector allocation (and its page faults) on every
// weight-matrix call during prefill.
// Templates must have C++ linkage, so this helper lives outside the
// extern "C" block below; the extern "C" kernel functions still call it.
template <typename T>
static T *tile_scratch(size_t n) {
    static thread_local std::vector<T> buf;
    buf.resize(n);
    return buf.data();
}

extern "C" {

void sofuu_set_threads(int n) {
    g_pool.set_thread_count(n);
}

int sofuu_get_threads(void) {
    return g_pool.get_thread_count();
}

// ── Basic & K-Quants Dequantization Routines ─────────────────────────
void sofuu_dequantize_row_q4_0(const void *src, float *dst, size_t k) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    size_t nb = k / 32;
    for (size_t b = 0; b < nb; ++b) {
        const block_q4_0 *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        float *y = dst + b * 32;
        const uint8_t *qs = blk->qs;
        #pragma GCC unroll 16
        for (int j = 0; j < 16; ++j) {
            uint8_t byte = qs[j];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)(byte >> 4) - 8;
            y[j]      = (float)v0 * d;
            y[j + 16] = (float)v1 * d;
        }
    }
}

void sofuu_dequantize_row_q8_0(const void *src, float *dst, size_t k) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    size_t nb = k / 32;
    for (size_t b = 0; b < nb; ++b) {
        const block_q8_0 *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        float *y = dst + b * 32;
        const int8_t *qs = blk->qs;
        #pragma GCC unroll 32
        for (int j = 0; j < 32; ++j) {
            y[j] = (float)qs[j] * d;
        }
    }
}

void sofuu_dequantize_row_q4_k(const void *src, float *dst, size_t k) {
    const block_q4_K *blocks = (const block_q4_K *)src;
    size_t nb = k / 256;
    float *y = dst;

    for (size_t b = 0; b < nb; ++b) {
        const block_q4_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        float min = f16_to_f32_val(blk->dmin);
        const uint8_t *q = blk->qs;

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = min * (float)m;
            get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = min * (float)m;

#if defined(SOFUU_NEON)
            float32x4_t vd1 = vdupq_n_f32(d1);
            float32x4_t vm1 = vdupq_n_f32(m1);
            float32x4_t vd2 = vdupq_n_f32(d2);
            float32x4_t vm2 = vdupq_n_f32(m2);

            for (int chunk = 0; chunk < 2; ++chunk) {
                uint8x16_t raw = vld1q_u8(q + chunk * 16);
                uint8x16_t lo_u8 = vandq_u8(raw, vdupq_n_u8(0x0f));
                uint8x16_t hi_u8 = vshrq_n_u8(raw, 4);

                uint16x8_t lo_16_0 = vmovl_u8(vget_low_u8(lo_u8));
                uint16x8_t lo_16_1 = vmovl_u8(vget_high_u8(lo_u8));

                float32x4_t f_lo_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_0)));
                float32x4_t f_lo_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_0)));
                float32x4_t f_lo_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_1)));
                float32x4_t f_lo_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_1)));

                vst1q_f32(y + chunk * 16 + 0,  vsubq_f32(vmulq_f32(f_lo_0, vd1), vm1));
                vst1q_f32(y + chunk * 16 + 4,  vsubq_f32(vmulq_f32(f_lo_1, vd1), vm1));
                vst1q_f32(y + chunk * 16 + 8,  vsubq_f32(vmulq_f32(f_lo_2, vd1), vm1));
                vst1q_f32(y + chunk * 16 + 12, vsubq_f32(vmulq_f32(f_lo_3, vd1), vm1));

                uint16x8_t hi_16_0 = vmovl_u8(vget_low_u8(hi_u8));
                uint16x8_t hi_16_1 = vmovl_u8(vget_high_u8(hi_u8));

                float32x4_t f_hi_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_0)));
                float32x4_t f_hi_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_0)));
                float32x4_t f_hi_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_1)));
                float32x4_t f_hi_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_1)));

                vst1q_f32(y + 32 + chunk * 16 + 0,  vsubq_f32(vmulq_f32(f_hi_0, vd2), vm2));
                vst1q_f32(y + 32 + chunk * 16 + 4,  vsubq_f32(vmulq_f32(f_hi_1, vd2), vm2));
                vst1q_f32(y + 32 + chunk * 16 + 8,  vsubq_f32(vmulq_f32(f_hi_2, vd2), vm2));
                vst1q_f32(y + 32 + chunk * 16 + 12, vsubq_f32(vmulq_f32(f_hi_3, vd2), vm2));
            }
#else
            #pragma GCC unroll 32
            for (int l = 0; l < 32; ++l) {
                y[l + 0]  = d1 * (float)(q[l] & 0x0F) - m1;
                y[l + 32] = d2 * (float)(q[l] >> 4)  - m2;
            }
#endif
            q += 32;
            y += 64;
            is += 2;
        }
    }
}

void sofuu_dequantize_row_q6_k(const void *src, float *dst, size_t k) {
    const block_q6_K *blocks = (const block_q6_K *)src;
    size_t nb = k / 256;
    float *y = dst;

    for (size_t b = 0; b < nb; ++b) {
        const block_q6_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t  *sc = blk->scales;

        for (int n = 0; n < 2; ++n) {
#if defined(SOFUU_NEON)
            for (int part = 0; part < 2; ++part) {
                int is = part;
                float32x4_t vd_sc1 = vdupq_n_f32(d * (float)sc[is + 0]);
                float32x4_t vd_sc2 = vdupq_n_f32(d * (float)sc[is + 2]);
                float32x4_t vd_sc3 = vdupq_n_f32(d * (float)sc[is + 4]);
                float32x4_t vd_sc4 = vdupq_n_f32(d * (float)sc[is + 6]);

                size_t l_base = (size_t)part * 16;
                uint8x16_t ql_0 = vld1q_u8(ql + l_base);
                uint8x16_t ql_32 = vld1q_u8(ql + l_base + 32);
                uint8x16_t qh_val = vld1q_u8(qh + l_base);

                // q1 = (ql[l+0] & 0x0F) | (((qh[l] >> 0) & 3) << 4) - 32
                uint8x16_t lo1 = vandq_u8(ql_0, vdupq_n_u8(0x0F));
                uint8x16_t hi1 = vshlq_n_u8(vandq_u8(qh_val, vdupq_n_u8(0x03)), 4);
                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), vdupq_n_s8(32));

                // q2 = (ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4) - 32
                uint8x16_t lo2 = vandq_u8(ql_32, vdupq_n_u8(0x0F));
                uint8x16_t hi2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_val, 2), vdupq_n_u8(0x03)), 4);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo2, hi2)), vdupq_n_s8(32));

                // q3 = (ql[l+0] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                uint8x16_t lo3 = vshrq_n_u8(ql_0, 4);
                uint8x16_t hi3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_val, 4), vdupq_n_u8(0x03)), 4);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo3, hi3)), vdupq_n_s8(32));

                // q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                uint8x16_t lo4 = vshrq_n_u8(ql_32, 4);
                uint8x16_t hi4 = vshlq_n_u8(vshrq_n_u8(qh_val, 6), 4);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo4, hi4)), vdupq_n_s8(32));

                int16x8_t q1_16_0 = vmovl_s8(vget_low_s8(q1));
                int16x8_t q1_16_1 = vmovl_s8(vget_high_s8(q1));
                vst1q_f32(y + l_base + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_0))), vd_sc1));
                vst1q_f32(y + l_base + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_0))), vd_sc1));
                vst1q_f32(y + l_base + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_1))), vd_sc1));
                vst1q_f32(y + l_base + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_1))), vd_sc1));

                int16x8_t q2_16_0 = vmovl_s8(vget_low_s8(q2));
                int16x8_t q2_16_1 = vmovl_s8(vget_high_s8(q2));
                vst1q_f32(y + l_base + 32 + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_16_0))), vd_sc2));
                vst1q_f32(y + l_base + 32 + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_16_0))), vd_sc2));
                vst1q_f32(y + l_base + 32 + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_16_1))), vd_sc2));
                vst1q_f32(y + l_base + 32 + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_16_1))), vd_sc2));

                int16x8_t q3_16_0 = vmovl_s8(vget_low_s8(q3));
                int16x8_t q3_16_1 = vmovl_s8(vget_high_s8(q3));
                vst1q_f32(y + l_base + 64 + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3_16_0))), vd_sc3));
                vst1q_f32(y + l_base + 64 + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3_16_0))), vd_sc3));
                vst1q_f32(y + l_base + 64 + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3_16_1))), vd_sc3));
                vst1q_f32(y + l_base + 64 + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3_16_1))), vd_sc3));

                int16x8_t q4_16_0 = vmovl_s8(vget_low_s8(q4));
                int16x8_t q4_16_1 = vmovl_s8(vget_high_s8(q4));
                vst1q_f32(y + l_base + 96 + 0,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q4_16_0))), vd_sc4));
                vst1q_f32(y + l_base + 96 + 4,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q4_16_0))), vd_sc4));
                vst1q_f32(y + l_base + 96 + 8,  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q4_16_1))), vd_sc4));
                vst1q_f32(y + l_base + 96 + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q4_16_1))), vd_sc4));
            }
#else
            #pragma GCC unroll 32
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l + 0]  = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
#endif
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ── Q4_0 Vector Dot Product ──────────────────────────────────────────
float sofuu_vec_dot_q4_0(const void *weight_bytes, const float *x, size_t n_blocks) {
    const block_q4_0 *blocks = (const block_q4_0 *)weight_bytes;
    float total_sum = 0.0f;

#if defined(SOFUU_NEON)
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_0 *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const float *xb = x + b * 32;

        uint8x16_t raw = vld1q_u8(blk->qs);
        uint8x16_t lo_u = vandq_u8(raw, vdupq_n_u8(0x0f));
        uint8x16_t hi_u = vshrq_n_u8(raw, 4);

        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(lo_u), vdupq_n_s8(8));
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(hi_u), vdupq_n_s8(8));

        // Convert low 16 nibbles to float32 and multiply with xb[0..15]
        int16x8_t lo_s16_0 = vmovl_s8(vget_low_s8(lo));
        int16x8_t lo_s16_1 = vmovl_s8(vget_high_s8(lo));

        float32x4_t lo_f_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_0)));
        float32x4_t lo_f_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16_0)));
        float32x4_t lo_f_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s16_1)));
        float32x4_t lo_f_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s16_1)));

        float32x4_t acc0 = vmulq_f32(lo_f_0, vld1q_f32(xb + 0));
        acc0 = vfmaq_f32(acc0, lo_f_1, vld1q_f32(xb + 4));
        acc0 = vfmaq_f32(acc0, lo_f_2, vld1q_f32(xb + 8));
        acc0 = vfmaq_f32(acc0, lo_f_3, vld1q_f32(xb + 12));

        // Convert high 16 nibbles to float32 and multiply with xb[16..31]
        int16x8_t hi_s16_0 = vmovl_s8(vget_low_s8(hi));
        int16x8_t hi_s16_1 = vmovl_s8(vget_high_s8(hi));

        float32x4_t hi_f_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_0)));
        float32x4_t hi_f_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16_0)));
        float32x4_t hi_f_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s16_1)));
        float32x4_t hi_f_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s16_1)));

        float32x4_t acc1 = vmulq_f32(hi_f_0, vld1q_f32(xb + 16));
        acc1 = vfmaq_f32(acc1, hi_f_1, vld1q_f32(xb + 20));
        acc1 = vfmaq_f32(acc1, hi_f_2, vld1q_f32(xb + 24));
        acc1 = vfmaq_f32(acc1, hi_f_3, vld1q_f32(xb + 28));

        float32x4_t blk_acc = vaddq_f32(acc0, acc1);
        float blk_sum = vgetq_lane_f32(blk_acc, 0) + vgetq_lane_f32(blk_acc, 1) +
                        vgetq_lane_f32(blk_acc, 2) + vgetq_lane_f32(blk_acc, 3);

        total_sum += blk_sum * d;
    }
#else
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_0 *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const float *xb = x + b * 32;

        float blk_sum = 0.0f;
        #pragma GCC unroll 16
        for (size_t j = 0; j < 16; ++j) {
            uint8_t byte = blk->qs[j];
            int8_t v0 = (int8_t)(byte & 0x0f) - 8;
            int8_t v1 = (int8_t)(byte >> 4) - 8;
            blk_sum += ((float)v0) * xb[j] + ((float)v1) * xb[j + 16];
        }
        total_sum += blk_sum * d;
    }
#endif

    return total_sum;
}

// ── Q8_0 Vector Dot Product ──────────────────────────────────────────
float sofuu_vec_dot_q8_0(const void *weight_bytes, const float *x, size_t n_blocks) {
    const block_q8_0 *blocks = (const block_q8_0 *)weight_bytes;
    float total_sum = 0.0f;

#if defined(SOFUU_NEON)
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q8_0 *blk = &blocks[b];
        #if defined(__aarch64__)
        __builtin_prefetch(&blocks[b + 4], 0, 0);
        #endif
        float d = f16_to_f32_val(blk->d);
        const float *xb = x + b * 32;

        int8x16_t q0 = vld1q_s8(blk->qs + 0);
        int8x16_t q1 = vld1q_s8(blk->qs + 16);

        int16x8_t q0_16_0 = vmovl_s8(vget_low_s8(q0));
        int16x8_t q0_16_1 = vmovl_s8(vget_high_s8(q0));
        int16x8_t q1_16_0 = vmovl_s8(vget_low_s8(q1));
        int16x8_t q1_16_1 = vmovl_s8(vget_high_s8(q1));

        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_16_0)));
        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_16_0)));
        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0_16_1)));
        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q0_16_1)));

        float32x4_t acc0 = vmulq_f32(f0, vld1q_f32(xb + 0));
        acc0 = vfmaq_f32(acc0, f1, vld1q_f32(xb + 4));
        acc0 = vfmaq_f32(acc0, f2, vld1q_f32(xb + 8));
        acc0 = vfmaq_f32(acc0, f3, vld1q_f32(xb + 12));

        float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_0)));
        float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_0)));
        float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_1)));
        float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_1)));

        float32x4_t acc1 = vmulq_f32(f4, vld1q_f32(xb + 16));
        acc1 = vfmaq_f32(acc1, f5, vld1q_f32(xb + 20));
        acc1 = vfmaq_f32(acc1, f6, vld1q_f32(xb + 24));
        acc1 = vfmaq_f32(acc1, f7, vld1q_f32(xb + 28));

        float32x4_t blk_acc = vaddq_f32(acc0, acc1);
        float blk_sum = vgetq_lane_f32(blk_acc, 0) + vgetq_lane_f32(blk_acc, 1) +
                        vgetq_lane_f32(blk_acc, 2) + vgetq_lane_f32(blk_acc, 3);

        total_sum += blk_sum * d;
    }
#else
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q8_0 *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const float *xb = x + b * 32;

        float blk_sum = 0.0f;
        #pragma GCC unroll 32
        for (size_t j = 0; j < 32; ++j) {
            blk_sum += ((float)blk->qs[j]) * xb[j];
        }
        total_sum += blk_sum * d;
    }
#endif

    return total_sum;
}

// ── Q4_K Vector Dot Product ──────────────────────────────────────────
float sofuu_vec_dot_q4_k(const void *weight_bytes, const float *x, size_t n_blocks) {
    const block_q4_K *blocks = (const block_q4_K *)weight_bytes;
#if defined(SOFUU_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_K *blk = &blocks[b];
        #if defined(__aarch64__)
        const char *p = (const char *)&blocks[b + 2];
        __builtin_prefetch(p + 0, 0, 0);
        __builtin_prefetch(p + 64, 0, 0);
        __builtin_prefetch(p + 128, 0, 0);
        #endif
        float d = f16_to_f32_val(blk->d);
        float min = f16_to_f32_val(blk->dmin);
        const uint8_t *q = blk->qs;
        const float *xb = x + b * 256;

        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = min * (float)m;
            get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = min * (float)m;

            float32x4_t vd1 = vdupq_n_f32(d1);
            float32x4_t vm1 = vdupq_n_f32(m1);
            float32x4_t vd2 = vdupq_n_f32(d2);
            float32x4_t vm2 = vdupq_n_f32(m2);

            float32x4_t dot_lo0 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo1 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo2 = vdupq_n_f32(0.0f);
            float32x4_t dot_lo3 = vdupq_n_f32(0.0f);
            float32x4_t sum_x_lo = vdupq_n_f32(0.0f);

            float32x4_t dot_hi0 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi1 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi2 = vdupq_n_f32(0.0f);
            float32x4_t dot_hi3 = vdupq_n_f32(0.0f);
            float32x4_t sum_x_hi = vdupq_n_f32(0.0f);

            for (int chunk = 0; chunk < 2; ++chunk) {
                uint8x16_t raw = vld1q_u8(q + chunk * 16);
                uint8x16_t lo_u8 = vandq_u8(raw, vdupq_n_u8(0x0f));
                uint8x16_t hi_u8 = vshrq_n_u8(raw, 4);

                uint16x8_t lo_16_0 = vmovl_u8(vget_low_u8(lo_u8));
                uint16x8_t lo_16_1 = vmovl_u8(vget_high_u8(lo_u8));

                float32x4_t f_lo_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_0)));
                float32x4_t f_lo_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_0)));
                float32x4_t f_lo_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_16_1)));
                float32x4_t f_lo_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_16_1)));

                float32x4_t x_lo_0 = vld1q_f32(xb + chunk * 16 + 0);
                float32x4_t x_lo_1 = vld1q_f32(xb + chunk * 16 + 4);
                float32x4_t x_lo_2 = vld1q_f32(xb + chunk * 16 + 8);
                float32x4_t x_lo_3 = vld1q_f32(xb + chunk * 16 + 12);

                dot_lo0 = vfmaq_f32(dot_lo0, f_lo_0, x_lo_0);
                dot_lo1 = vfmaq_f32(dot_lo1, f_lo_1, x_lo_1);
                dot_lo2 = vfmaq_f32(dot_lo2, f_lo_2, x_lo_2);
                dot_lo3 = vfmaq_f32(dot_lo3, f_lo_3, x_lo_3);

                sum_x_lo = vaddq_f32(sum_x_lo, vaddq_f32(vaddq_f32(x_lo_0, x_lo_1), vaddq_f32(x_lo_2, x_lo_3)));

                uint16x8_t hi_16_0 = vmovl_u8(vget_low_u8(hi_u8));
                uint16x8_t hi_16_1 = vmovl_u8(vget_high_u8(hi_u8));

                float32x4_t f_hi_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_0)));
                float32x4_t f_hi_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_0)));
                float32x4_t f_hi_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_16_1)));
                float32x4_t f_hi_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_16_1)));

                float32x4_t x_hi_0 = vld1q_f32(xb + 32 + chunk * 16 + 0);
                float32x4_t x_hi_1 = vld1q_f32(xb + 32 + chunk * 16 + 4);
                float32x4_t x_hi_2 = vld1q_f32(xb + 32 + chunk * 16 + 8);
                float32x4_t x_hi_3 = vld1q_f32(xb + 32 + chunk * 16 + 12);

                dot_hi0 = vfmaq_f32(dot_hi0, f_hi_0, x_hi_0);
                dot_hi1 = vfmaq_f32(dot_hi1, f_hi_1, x_hi_1);
                dot_hi2 = vfmaq_f32(dot_hi2, f_hi_2, x_hi_2);
                dot_hi3 = vfmaq_f32(dot_hi3, f_hi_3, x_hi_3);

                sum_x_hi = vaddq_f32(sum_x_hi, vaddq_f32(vaddq_f32(x_hi_0, x_hi_1), vaddq_f32(x_hi_2, x_hi_3)));
            }

            float32x4_t total_dot_lo = vaddq_f32(vaddq_f32(dot_lo0, dot_lo1), vaddq_f32(dot_lo2, dot_lo3));
            float32x4_t total_dot_hi = vaddq_f32(vaddq_f32(dot_hi0, dot_hi1), vaddq_f32(dot_hi2, dot_hi3));

            acc0 = vfmaq_f32(acc0, total_dot_lo, vd1);
            acc1 = vfmsq_f32(acc1, sum_x_lo, vm1);
            acc2 = vfmaq_f32(acc2, total_dot_hi, vd2);
            acc3 = vfmsq_f32(acc3, sum_x_hi, vm2);

            q += 32;
            xb += 64;
            is += 2;
        }
    }
    float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(total);
#else
    float total_sum = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q4_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        float min = f16_to_f32_val(blk->dmin);
        const uint8_t *q = blk->qs;
        const float *xb = x + b * 256;

        float b_sum = 0.0f;
        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk->scales, &sc, &m);
            float d1 = d * (float)sc;
            float m1 = min * (float)m;
            get_scale_min_k4(is + 1, blk->scales, &sc, &m);
            float d2 = d * (float)sc;
            float m2 = min * (float)m;

            #pragma GCC unroll 32
            for (int l = 0; l < 32; ++l) {
                float w1 = d1 * (float)(q[l] & 0x0F) - m1;
                float w2 = d2 * (float)(q[l] >> 4)  - m2;
                b_sum += w1 * xb[l] + w2 * xb[l + 32];
            }
            q += 32;
            xb += 64;
            is += 2;
        }
        total_sum += b_sum;
    }
    return total_sum;
#endif
}

// ── Q6_K Vector Dot Product ──────────────────────────────────────────
float sofuu_vec_dot_q6_k(const void *weight_bytes, const float *x, size_t n_blocks) {
    const block_q6_K *blocks = (const block_q6_K *)weight_bytes;
#if defined(SOFUU_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q6_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t  *sc = blk->scales;
        const float   *xb = x + b * 256;

        for (int n = 0; n < 2; ++n) {
            for (int part = 0; part < 2; ++part) {
                int is = part;
                float32x4_t vd_sc1 = vdupq_n_f32(d * (float)sc[is + 0]);
                float32x4_t vd_sc2 = vdupq_n_f32(d * (float)sc[is + 2]);
                float32x4_t vd_sc3 = vdupq_n_f32(d * (float)sc[is + 4]);
                float32x4_t vd_sc4 = vdupq_n_f32(d * (float)sc[is + 6]);

                size_t l_base = (size_t)part * 16;
                uint8x16_t ql_0 = vld1q_u8(ql + l_base);
                uint8x16_t ql_32 = vld1q_u8(ql + l_base + 32);
                uint8x16_t qh_val = vld1q_u8(qh + l_base);

                // q1 = (ql[l+0] & 0x0F) | (((qh[l] >> 0) & 3) << 4) - 32
                uint8x16_t lo1 = vandq_u8(ql_0, vdupq_n_u8(0x0F));
                uint8x16_t hi1 = vshlq_n_u8(vandq_u8(qh_val, vdupq_n_u8(0x03)), 4);
                int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo1, hi1)), vdupq_n_s8(32));

                // q2 = (ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4) - 32
                uint8x16_t lo2 = vandq_u8(ql_32, vdupq_n_u8(0x0F));
                uint8x16_t hi2 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_val, 2), vdupq_n_u8(0x03)), 4);
                int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo2, hi2)), vdupq_n_s8(32));

                // q3 = (ql[l+0] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                uint8x16_t lo3 = vshrq_n_u8(ql_0, 4);
                uint8x16_t hi3 = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_val, 4), vdupq_n_u8(0x03)), 4);
                int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo3, hi3)), vdupq_n_s8(32));

                // q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                uint8x16_t lo4 = vshrq_n_u8(ql_32, 4);
                uint8x16_t hi4 = vshlq_n_u8(vshrq_n_u8(qh_val, 6), 4);
                int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo4, hi4)), vdupq_n_s8(32));

                int16x8_t q1_16_0 = vmovl_s8(vget_low_s8(q1));
                int16x8_t q1_16_1 = vmovl_s8(vget_high_s8(q1));
                acc0 = vfmaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_0))), vd_sc1), vld1q_f32(xb + l_base + 0));
                acc1 = vfmaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_0))), vd_sc1), vld1q_f32(xb + l_base + 4));
                acc2 = vfmaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1_16_1))), vd_sc1), vld1q_f32(xb + l_base + 8));
                acc3 = vfmaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q1_16_1))), vd_sc1), vld1q_f32(xb + l_base + 12));

                int16x8_t q2_16_0 = vmovl_s8(vget_low_s8(q2));
                int16x8_t q2_16_1 = vmovl_s8(vget_high_s8(q2));
                acc0 = vfmaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_16_0))), vd_sc2), vld1q_f32(xb + l_base + 32 + 0));
                acc1 = vfmaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_16_0))), vd_sc2), vld1q_f32(xb + l_base + 32 + 4));
                acc2 = vfmaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_16_1))), vd_sc2), vld1q_f32(xb + l_base + 32 + 8));
                acc3 = vfmaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_16_1))), vd_sc2), vld1q_f32(xb + l_base + 32 + 12));

                int16x8_t q3_16_0 = vmovl_s8(vget_low_s8(q3));
                int16x8_t q3_16_1 = vmovl_s8(vget_high_s8(q3));
                acc0 = vfmaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3_16_0))), vd_sc3), vld1q_f32(xb + l_base + 64 + 0));
                acc1 = vfmaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3_16_0))), vd_sc3), vld1q_f32(xb + l_base + 64 + 4));
                acc2 = vfmaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q3_16_1))), vd_sc3), vld1q_f32(xb + l_base + 64 + 8));
                acc3 = vfmaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q3_16_1))), vd_sc3), vld1q_f32(xb + l_base + 64 + 12));

                int16x8_t q4_16_0 = vmovl_s8(vget_low_s8(q4));
                int16x8_t q4_16_1 = vmovl_s8(vget_high_s8(q4));
                acc0 = vfmaq_f32(acc0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q4_16_0))), vd_sc4), vld1q_f32(xb + l_base + 96 + 0));
                acc1 = vfmaq_f32(acc1, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q4_16_0))), vd_sc4), vld1q_f32(xb + l_base + 96 + 4));
                acc2 = vfmaq_f32(acc2, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q4_16_1))), vd_sc4), vld1q_f32(xb + l_base + 96 + 8));
                acc3 = vfmaq_f32(acc3, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q4_16_1))), vd_sc4), vld1q_f32(xb + l_base + 96 + 12));
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(total);
#else
    float total_sum = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const block_q6_K *blk = &blocks[b];
        float d = f16_to_f32_val(blk->d);
        const uint8_t *ql = blk->ql;
        const uint8_t *qh = blk->qh;
        const int8_t  *sc = blk->scales;
        const float   *xb = x + b * 256;
        float b_sum = 0.0f;

        for (int n = 0; n < 2; ++n) {
            #pragma GCC unroll 32
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                b_sum += (d * (float)sc[is + 0]) * (float)q1 * xb[l + 0]
                       + (d * (float)sc[is + 2]) * (float)q2 * xb[l + 32]
                       + (d * (float)sc[is + 4]) * (float)q3 * xb[l + 64]
                       + (d * (float)sc[is + 6]) * (float)q4 * xb[l + 96];
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        total_sum += b_sum;
    }
    return total_sum;
#endif
}

// ── Hardware SIMD Vector Dot Product (F32) ───────────────────────────
float sofuu_vec_dot_f32(const float *a, const float *b, size_t n) {
#if defined(SOFUU_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        sum2 = vfmaq_f32(sum2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        sum3 = vfmaq_f32(sum3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    sum0 = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    for (; i + 4 <= n; i += 4) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float total = vaddvq_f32(sum0);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#elif defined(SOFUU_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; i + 8 <= n; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }
    alignas(32) float buf[8];
    _mm256_storeu_ps(buf, sum0);
    float total = 0.0f;
    for (int k = 0; k < 8; ++k) total += buf[k];
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#else
    float total = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#endif
}

// ── 4-way SIMD Vector Dot Product (F32) ──────────────────────────────
void sofuu_vec_dot_f32_x4(const float *w, const float *x0, const float *x1, const float *x2, const float *x3, size_t n,
                          float *out0, float *out1, float *out2, float *out3) {
#if defined(SOFUU_NEON)
    float32x4_t sum0_0 = vdupq_n_f32(0.0f);
    float32x4_t sum0_1 = vdupq_n_f32(0.0f);
    float32x4_t sum1_0 = vdupq_n_f32(0.0f);
    float32x4_t sum1_1 = vdupq_n_f32(0.0f);
    float32x4_t sum2_0 = vdupq_n_f32(0.0f);
    float32x4_t sum2_1 = vdupq_n_f32(0.0f);
    float32x4_t sum3_0 = vdupq_n_f32(0.0f);
    float32x4_t sum3_1 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t vw0 = vld1q_f32(w + i + 0);
        float32x4_t vw1 = vld1q_f32(w + i + 4);

        sum0_0 = vfmaq_f32(sum0_0, vw0, vld1q_f32(x0 + i + 0));
        sum0_1 = vfmaq_f32(sum0_1, vw1, vld1q_f32(x0 + i + 4));

        sum1_0 = vfmaq_f32(sum1_0, vw0, vld1q_f32(x1 + i + 0));
        sum1_1 = vfmaq_f32(sum1_1, vw1, vld1q_f32(x1 + i + 4));

        sum2_0 = vfmaq_f32(sum2_0, vw0, vld1q_f32(x2 + i + 0));
        sum2_1 = vfmaq_f32(sum2_1, vw1, vld1q_f32(x2 + i + 4));

        sum3_0 = vfmaq_f32(sum3_0, vw0, vld1q_f32(x3 + i + 0));
        sum3_1 = vfmaq_f32(sum3_1, vw1, vld1q_f32(x3 + i + 4));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t vw = vld1q_f32(w + i);
        sum0_0 = vfmaq_f32(sum0_0, vw, vld1q_f32(x0 + i));
        sum1_0 = vfmaq_f32(sum1_0, vw, vld1q_f32(x1 + i));
        sum2_0 = vfmaq_f32(sum2_0, vw, vld1q_f32(x2 + i));
        sum3_0 = vfmaq_f32(sum3_0, vw, vld1q_f32(x3 + i));
    }

    float s0 = vaddvq_f32(vaddq_f32(sum0_0, sum0_1));
    float s1 = vaddvq_f32(vaddq_f32(sum1_0, sum1_1));
    float s2 = vaddvq_f32(vaddq_f32(sum2_0, sum2_1));
    float s3 = vaddvq_f32(vaddq_f32(sum3_0, sum3_1));

    for (; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i];
        s1 += wi * x1[i];
        s2 += wi * x2[i];
        s3 += wi * x3[i];
    }

    *out0 = s0;
    *out1 = s1;
    *out2 = s2;
    *out3 = s3;
#elif defined(SOFUU_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vw = _mm256_loadu_ps(w + i);
        sum0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x0 + i), sum0);
        sum1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x1 + i), sum1);
        sum2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x2 + i), sum2);
        sum3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x3 + i), sum3);
    }
    alignas(32) float b0[8], b1[8], b2[8], b3[8];
    _mm256_storeu_ps(b0, sum0);
    _mm256_storeu_ps(b1, sum1);
    _mm256_storeu_ps(b2, sum2);
    _mm256_storeu_ps(b3, sum3);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        s0 += b0[k]; s1 += b1[k]; s2 += b2[k]; s3 += b3[k];
    }
    for (; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i];
        s1 += wi * x1[i];
        s2 += wi * x2[i];
        s3 += wi * x3[i];
    }
    *out0 = s0;
    *out1 = s1;
    *out2 = s2;
    *out3 = s3;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i];
        s1 += wi * x1[i];
        s2 += wi * x2[i];
        s3 += wi * x3[i];
    }
    *out0 = s0;
    *out1 = s1;
    *out2 = s2;
    *out3 = s3;
#endif
}

// ── 8-way SIMD Vector Dot Product (F32) ──────────────────────────────
void sofuu_vec_dot_f32_x8(const float *w,
                          const float *x0, const float *x1, const float *x2, const float *x3,
                          const float *x4, const float *x5, const float *x6, const float *x7,
                          size_t n,
                          float *out0, float *out1, float *out2, float *out3,
                          float *out4, float *out5, float *out6, float *out7) {
#if defined(SOFUU_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    float32x4_t sum5 = vdupq_n_f32(0.0f);
    float32x4_t sum6 = vdupq_n_f32(0.0f);
    float32x4_t sum7 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vw = vld1q_f32(w + i);
        sum0 = vfmaq_f32(sum0, vw, vld1q_f32(x0 + i));
        sum1 = vfmaq_f32(sum1, vw, vld1q_f32(x1 + i));
        sum2 = vfmaq_f32(sum2, vw, vld1q_f32(x2 + i));
        sum3 = vfmaq_f32(sum3, vw, vld1q_f32(x3 + i));
        sum4 = vfmaq_f32(sum4, vw, vld1q_f32(x4 + i));
        sum5 = vfmaq_f32(sum5, vw, vld1q_f32(x5 + i));
        sum6 = vfmaq_f32(sum6, vw, vld1q_f32(x6 + i));
        sum7 = vfmaq_f32(sum7, vw, vld1q_f32(x7 + i));
    }

    float s0 = vaddvq_f32(sum0);
    float s1 = vaddvq_f32(sum1);
    float s2 = vaddvq_f32(sum2);
    float s3 = vaddvq_f32(sum3);
    float s4 = vaddvq_f32(sum4);
    float s5 = vaddvq_f32(sum5);
    float s6 = vaddvq_f32(sum6);
    float s7 = vaddvq_f32(sum7);

    for (; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i];
        s1 += wi * x1[i];
        s2 += wi * x2[i];
        s3 += wi * x3[i];
        s4 += wi * x4[i];
        s5 += wi * x5[i];
        s6 += wi * x6[i];
        s7 += wi * x7[i];
    }

    *out0 = s0;
    *out1 = s1;
    *out2 = s2;
    *out3 = s3;
    *out4 = s4;
    *out5 = s5;
    *out6 = s6;
    *out7 = s7;
#elif defined(SOFUU_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vw = _mm256_loadu_ps(w + i);
        sum0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x0 + i), sum0);
        sum1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x1 + i), sum1);
        sum2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x2 + i), sum2);
        sum3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x3 + i), sum3);
        sum4 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x4 + i), sum4);
        sum5 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x5 + i), sum5);
        sum6 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x6 + i), sum6);
        sum7 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x7 + i), sum7);
    }
    alignas(32) float b0[8], b1[8], b2[8], b3[8], b4[8], b5[8], b6[8], b7[8];
    _mm256_storeu_ps(b0, sum0); _mm256_storeu_ps(b1, sum1);
    _mm256_storeu_ps(b2, sum2); _mm256_storeu_ps(b3, sum3);
    _mm256_storeu_ps(b4, sum4); _mm256_storeu_ps(b5, sum5);
    _mm256_storeu_ps(b6, sum6); _mm256_storeu_ps(b7, sum7);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    for (int k = 0; k < 8; ++k) {
        s0 += b0[k]; s1 += b1[k]; s2 += b2[k]; s3 += b3[k];
        s4 += b4[k]; s5 += b5[k]; s6 += b6[k]; s7 += b7[k];
    }
    for (; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i]; s1 += wi * x1[i]; s2 += wi * x2[i]; s3 += wi * x3[i];
        s4 += wi * x4[i]; s5 += wi * x5[i]; s6 += wi * x6[i]; s7 += wi * x7[i];
    }
    *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
    *out4 = s4; *out5 = s5; *out6 = s6; *out7 = s7;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float wi = w[i];
        s0 += wi * x0[i]; s1 += wi * x1[i]; s2 += wi * x2[i]; s3 += wi * x3[i];
        s4 += wi * x4[i]; s5 += wi * x5[i]; s6 += wi * x6[i]; s7 += wi * x7[i];
    }
    *out0 = s0; *out1 = s1; *out2 = s2; *out3 = s3;
    *out4 = s4; *out5 = s5; *out6 = s6; *out7 = s7;
#endif
}

// ── Multi-Threaded Parallel Matrix-Vector (GEMV) Kernels ─────────────

// Does THIS cpu actually implement the ARMv8.2 dot-product (SDOT) extension?
//
// __ARM_FEATURE_DOTPROD only says the *translation unit* was compiled with
// -march=armv8.2-a+dotprod, which build.rs passes for every aarch64 target.
// On a device whose cpu predates ARMv8.2 that flag is still honoured at
// compile time and the sdot instructions are emitted unconditionally —
// so the kernel would SIGILL at the first weight block instead of taking
// the generic path.
//
// This asks the kernel instead. Android and every other aarch64 Linux
// userland are __linux__, so getauxval(AT_HWCAP) is available; the answer
// is a per-CPU constant, cached in a function-local static.
static bool sofuu_cpu_has_dotprod() {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
#if defined(__linux__)
    static const bool has = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return has;
#else
    // Not Linux: __ARM_FEATURE_DOTPROD was the only signal available, and
    // every non-Linux aarch64 target we build (Apple silicon) has sdot.
    return true;
#endif
#else
    return false;
#endif
}

void sofuu_gemv_q4_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    size_t n_blocks = cols / 32;
    size_t row_size = n_blocks * sizeof(block_q4_0);
    const uint8_t *w = (const uint8_t *)weight_bytes;

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            out[r] = sofuu_vec_dot_q4_0(w + r * row_size, x, n_blocks);
        }
    });
}

void sofuu_gemv_q8_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    size_t n_blocks = cols / 32;
    size_t row_size = n_blocks * sizeof(block_q8_0);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    if (sofuu_cpu_has_dotprod() && rows % 4 == 0 && cols % 32 == 0 && SOFUU_TILE_TYPES_OK(8)) {
        if (const uint8_t *tile = sofuu_tile_lookup(weight_bytes, 8, rows, cols)) {
            std::vector<tile_block_q8_0> qa(cols / 32);
            sofuu_quantize_row_q8_0_tile(x, qa.data(), cols);
            const tile_block_q8_0 *qa_ptr = qa.data();
            const size_t nb = cols / 32;
            g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemv_q8_0_4x4_q8_0(cols, out + g * 4, rows,
                        tile + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, 1, 4);
                }
            });
            return;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Compiled in, but only taken on a cpu that implements SDOT — see
    // sofuu_cpu_has_dotprod(). Anything older falls through to the generic
    // kernel below rather than faulting on the first weight block.
    if (sofuu_cpu_has_dotprod()) {
        thread_local std::vector<sofuu_q8_block_ext> tls_x_q8;
        if (tls_x_q8.size() < n_blocks) {
            tls_x_q8.resize(n_blocks);
        }
        sofuu_quantize_row_q8_a76(x, tls_x_q8.data(), n_blocks);

        const sofuu_q8_block_ext *x_q8_ptr = tls_x_q8.data();
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                out[r] = sofuu_vec_dot_q8_0_dotprod_a76((const block_q8_0 *)(w + r * row_size), x_q8_ptr, n_blocks);
            }
        });
        return;
    }
#endif
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            out[r] = sofuu_vec_dot_q8_0(w + r * row_size, x, n_blocks);
        }
    });
}

void sofuu_gemv_q4_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    size_t n_blocks = cols / 256;
    size_t row_size = n_blocks * sizeof(block_q4_K);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    if (sofuu_cpu_has_dotprod() && rows % 8 == 0 && cols % 256 == 0 && SOFUU_TILE_TYPES_OK(12)) {
        const uint8_t *sc = nullptr;
        if (const uint8_t *tile = sofuu_tile_lookup_sc(weight_bytes, 12, rows, cols, &sc)) {
            std::vector<tile_block_q8_K> qa(cols / 256);
            sofuu_quantize_row_q8_K(x, qa.data(), cols);
            const tile_block_q8_K *qa_ptr = qa.data();
            const size_t nb = cols / 256;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t = tile + g * nb * sizeof(tile_block_q4_Kx8);
                    if (sc) {
                        ggml_gemv_q4_K_8x4_q8_K_sc(cols, out + g * 8, rows, t, qa_ptr, 1, 8,
                            sc + g * nb * sizeof(tile_sc_q4_Kx8));
                    } else {
                        ggml_gemv_q4_K_8x4_q8_K(cols, out + g * 8, rows, t, qa_ptr, 1, 8);
                    }
                }
            });
            return;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (sofuu_cpu_has_dotprod()) {
        size_t q8_blocks = cols / 32;
        thread_local std::vector<sofuu_q8_block_ext> tls_x_q8;
        if (tls_x_q8.size() < q8_blocks) {
            tls_x_q8.resize(q8_blocks);
        }
        sofuu_quantize_row_q8_a76(x, tls_x_q8.data(), q8_blocks);

        const sofuu_q8_block_ext *x_q8_ptr = tls_x_q8.data();
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                out[r] = sofuu_vec_dot_q4_k_dotprod_a76((const block_q4_K *)(w + r * row_size), x_q8_ptr, n_blocks);
            }
        });
        return;
    }
#endif
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            out[r] = sofuu_vec_dot_q4_k(w + r * row_size, x, n_blocks);
        }
    });
}

void sofuu_gemv_q6_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    size_t n_blocks = cols / 256;
    size_t row_size = n_blocks * sizeof(block_q6_K);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    if (sofuu_cpu_has_dotprod() && rows % 8 == 0 && cols % 256 == 0 && SOFUU_TILE_TYPES_OK(14)) {
        if (const uint8_t *tile = sofuu_tile_lookup(weight_bytes, 14, rows, cols)) {
            std::vector<tile_block_q8_K> qa(cols / 256);
            sofuu_quantize_row_q8_K(x, qa.data(), cols);
            const tile_block_q8_K *qa_ptr = qa.data();
            const size_t nb = cols / 256;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemv_q6_K_8x4_q8_K(cols, out + g * 8, rows,
                        tile + g * nb * sizeof(tile_block_q6_Kx8), qa_ptr, 1, 8);
                }
            });
            return;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (sofuu_cpu_has_dotprod()) {
        size_t q8_blocks = cols / 32;
        thread_local std::vector<sofuu_q8_block_ext> tls_x_q8;
        if (tls_x_q8.size() < q8_blocks) {
            tls_x_q8.resize(q8_blocks);
        }
        sofuu_quantize_row_q8_a76(x, tls_x_q8.data(), q8_blocks);

        const sofuu_q8_block_ext *x_q8_ptr = tls_x_q8.data();
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                out[r] = sofuu_vec_dot_q6_k_dotprod_a76((const block_q6_K *)(w + r * row_size), x_q8_ptr, n_blocks);
            }
        });
        return;
    }
#endif
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            out[r] = sofuu_vec_dot_q6_k(w + r * row_size, x, n_blocks);
        }
    });
}

void sofuu_gemv_f32(const float *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            out[r] = sofuu_vec_dot_f32(weight_bytes + r * cols, x, cols);
        }
    });
}

void sofuu_gemv_f16(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    const uint8_t *w_base = (const uint8_t *)weight_bytes;
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
#if defined(SOFUU_NEON)
        for (size_t r = start; r < end; ++r) {
            const uint8_t *row = w_base + r * cols * 2;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            size_t c = 0;
            for (; c + 16 <= cols; c += 16) {
                uint16x8_t raw0 = vreinterpretq_u16_u8(vld1q_u8(row + c * 2));
                uint16x8_t raw1 = vreinterpretq_u16_u8(vld1q_u8(row + c * 2 + 16));
                float32x4_t f0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(raw0)));
                float32x4_t f1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(raw0)));
                float32x4_t f2 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(raw1)));
                float32x4_t f3 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(raw1)));
                acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x + c));
                acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x + c + 4));
                acc2 = vfmaq_f32(acc2, f2, vld1q_f32(x + c + 8));
                acc3 = vfmaq_f32(acc3, f3, vld1q_f32(x + c + 12));
            }
            float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            float sum = vaddvq_f32(total);
            for (; c < cols; ++c) {
                uint16_t val;
                std::memcpy(&val, row + c * 2, 2);
                sum += f16_to_f32_val(val) * x[c];
            }
            out[r] = sum;
        }
#else
        for (size_t r = start; r < end; ++r) {
            const uint8_t *row = w_base + r * cols * 2;
            float sum = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                uint16_t val;
                std::memcpy(&val, row + c * 2, 2);
                sum += f16_to_f32_val(val) * x[c];
            }
            out[r] = sum;
        }
#endif
    });
}

#if defined(SOFUU_NEON)
static inline float vec_dot_bf16_neon(const uint8_t *row_bytes, const float *x, size_t cols) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t c = 0;
    for (; c + 16 <= cols; c += 16) {
        uint16x8_t raw0 = vreinterpretq_u16_u8(vld1q_u8(row_bytes + c * 2));
        uint16x8_t raw1 = vreinterpretq_u16_u8(vld1q_u8(row_bytes + c * 2 + 16));

        uint32x4_t u0 = vshll_n_u16(vget_low_u16(raw0), 16);
        uint32x4_t u1 = vshll_n_u16(vget_high_u16(raw0), 16);
        uint32x4_t u2 = vshll_n_u16(vget_low_u16(raw1), 16);
        uint32x4_t u3 = vshll_n_u16(vget_high_u16(raw1), 16);

        float32x4_t f0 = vreinterpretq_f32_u32(u0);
        float32x4_t f1 = vreinterpretq_f32_u32(u1);
        float32x4_t f2 = vreinterpretq_f32_u32(u2);
        float32x4_t f3 = vreinterpretq_f32_u32(u3);

        acc0 = vfmaq_f32(acc0, f0, vld1q_f32(x + c + 0));
        acc1 = vfmaq_f32(acc1, f1, vld1q_f32(x + c + 4));
        acc2 = vfmaq_f32(acc2, f2, vld1q_f32(x + c + 8));
        acc3 = vfmaq_f32(acc3, f3, vld1q_f32(x + c + 12));
    }

    float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float sum = vgetq_lane_f32(total, 0) + vgetq_lane_f32(total, 1) +
                vgetq_lane_f32(total, 2) + vgetq_lane_f32(total, 3);

    for (; c < cols; ++c) {
        uint16_t val;
        std::memcpy(&val, row_bytes + c * 2, 2);
        uint32_t bits = (uint32_t)val << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        sum += f * x[c];
    }
    return sum;
}

static inline float vec_dot_mlx4_neon(
    const uint8_t *w_row,
    const uint8_t *s_row,
    const uint8_t *b_row,
    const float *x,
    size_t cols,
    size_t group_size,
    bool is_bf16
) {
    size_t num_groups = cols / group_size;
    size_t u32_per_group = group_size / 8;
    float total_sum = 0.0f;

    for (size_t g = 0; g < num_groups; ++g) {
        uint16_t s_u16, b_u16;
        std::memcpy(&s_u16, s_row + g * 2, 2);
        std::memcpy(&b_u16, b_row + g * 2, 2);

        float scale, bias;
        if (is_bf16) {
            uint32_t s_bits = (uint32_t)s_u16 << 16;
            uint32_t b_bits = (uint32_t)b_u16 << 16;
            std::memcpy(&scale, &s_bits, 4);
            std::memcpy(&bias, &b_bits, 4);
        } else {
            scale = f16_to_f32_val(s_u16);
            bias  = f16_to_f32_val(b_u16);
        }

        float32x4_t v_scale = vdupq_n_f32(scale);
        float32x4_t v_bias  = vdupq_n_f32(bias);

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        size_t base_col = g * group_size;
        const uint8_t *w_grp = w_row + (g * u32_per_group) * 4;
        const float *x_grp = x + base_col;

        for (size_t u = 0; u < u32_per_group; ++u) {
            uint32_t packed;
            std::memcpy(&packed, w_grp + u * 4, 4);

            float n0 = (float)((packed >> 0)  & 0x0F);
            float n1 = (float)((packed >> 4)  & 0x0F);
            float n2 = (float)((packed >> 8)  & 0x0F);
            float n3 = (float)((packed >> 12) & 0x0F);
            float n4 = (float)((packed >> 16) & 0x0F);
            float n5 = (float)((packed >> 20) & 0x0F);
            float n6 = (float)((packed >> 24) & 0x0F);
            float n7 = (float)((packed >> 28) & 0x0F);

            alignas(16) float f_lo[4] = { n0, n1, n2, n3 };
            alignas(16) float f_hi[4] = { n4, n5, n6, n7 };

            float32x4_t w0 = vfmaq_f32(v_bias, vld1q_f32(f_lo), v_scale);
            float32x4_t w1 = vfmaq_f32(v_bias, vld1q_f32(f_hi), v_scale);

            acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x_grp + u * 8 + 0));
            acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x_grp + u * 8 + 4));
        }

        float32x4_t acc = vaddq_f32(acc0, acc1);
        total_sum += vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                     vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
    }
    return total_sum;
}
#endif

static inline float vec_dot_bf16_scalar(const uint8_t *row_bytes, const float *x, size_t cols) {
    float sum = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
        uint16_t val;
        std::memcpy(&val, row_bytes + c * 2, 2);
        uint32_t bits = (uint32_t)val << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        sum += f * x[c];
    }
    return sum;
}

static inline float vec_dot_mlx4_scalar(
    const uint8_t *w_row,
    const uint8_t *s_row,
    const uint8_t *b_row,
    const float *x,
    size_t cols,
    size_t group_size,
    bool is_bf16
) {
    size_t num_groups = cols / group_size;
    size_t u32_per_group = group_size / 8;
    float sum = 0.0f;

    for (size_t g = 0; g < num_groups; ++g) {
        uint16_t s_u16, b_u16;
        std::memcpy(&s_u16, s_row + g * 2, 2);
        std::memcpy(&b_u16, b_row + g * 2, 2);

        float scale, bias;
        if (is_bf16) {
            uint32_t s_bits = (uint32_t)s_u16 << 16;
            uint32_t b_bits = (uint32_t)b_u16 << 16;
            std::memcpy(&scale, &s_bits, 4);
            std::memcpy(&bias, &b_bits, 4);
        } else {
            scale = f16_to_f32_val(s_u16);
            bias  = f16_to_f32_val(b_u16);
        }

        size_t base_col = g * group_size;
        const uint8_t *w_grp = w_row + (g * u32_per_group) * 4;
        const float *x_grp = x + base_col;

        for (size_t u = 0; u < u32_per_group; ++u) {
            uint32_t packed;
            std::memcpy(&packed, w_grp + u * 4, 4);
            size_t c_idx = u * 8;
            for (size_t i = 0; i < 8; ++i) {
                float val_4b = (float)((packed >> (i * 4)) & 0x0F);
                float w = val_4b * scale + bias;
                sum += w * x_grp[c_idx + i];
            }
        }
    }
    return sum;
}

void sofuu_gemv_bf16(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols) {
    const uint8_t *w = (const uint8_t *)weight_bytes;
    size_t row_bytes = cols * 2;
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
#if defined(SOFUU_NEON)
            out[r] = vec_dot_bf16_neon(w + r * row_bytes, x, cols);
#else
            out[r] = vec_dot_bf16_scalar(w + r * row_bytes, x, cols);
#endif
        }
    });
}

void sofuu_gemv_mlx4(
    const void *w_bytes,
    const void *scales_bytes,
    const void *biases_bytes,
    const float *x,
    float *out,
    size_t rows,
    size_t cols,
    size_t group_size,
    int is_bf16
) {
    const uint8_t *w = (const uint8_t *)w_bytes;
    const uint8_t *s = (const uint8_t *)scales_bytes;
    const uint8_t *b = (const uint8_t *)biases_bytes;
    size_t w_row_bytes = (cols / 8) * 4;
    size_t grp_row_bytes = (cols / group_size) * 2;
    bool bf16_flag = is_bf16 != 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (sofuu_cpu_has_dotprod()) {
        size_t q8_blocks = cols / 32;
        thread_local std::vector<sofuu_q8_block_ext> tls_x_q8;
        if (tls_x_q8.size() < q8_blocks) {
            tls_x_q8.resize(q8_blocks);
        }
        sofuu_quantize_row_q8_a76(x, tls_x_q8.data(), q8_blocks);
        const sofuu_q8_block_ext *x_q8_ptr = tls_x_q8.data();
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                out[r] = sofuu_vec_dot_mlx4_dotprod_a76(w + r * w_row_bytes, s + r * grp_row_bytes, b + r * grp_row_bytes, x, cols, group_size, x_q8_ptr, bf16_flag);
            }
        });
        return;
    }
#endif

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
#if defined(SOFUU_NEON)
            out[r] = vec_dot_mlx4_neon(w + r * w_row_bytes, s + r * grp_row_bytes, b + r * grp_row_bytes, x, cols, group_size, bf16_flag);
#else
            out[r] = vec_dot_mlx4_scalar(w + r * w_row_bytes, s + r * grp_row_bytes, b + r * grp_row_bytes, x, cols, group_size, bf16_flag);
#endif
        }
    });
}

void sofuu_gemv_mlx4_multi(
    const void *const *w_bytes,
    const void *const *scales_bytes,
    const void *const *biases_bytes,
    const float *x,
    float *const *outs,
    const size_t *rows,
    size_t num_mats,
    size_t cols,
    size_t group_size,
    int is_bf16
) {
    if (!w_bytes || !scales_bytes || !biases_bytes || !x || !outs || !rows || num_mats == 0 || !cols) return;
    size_t total_rows = 0;
    for (size_t m = 0; m < num_mats; ++m) total_rows += rows[m];
    if (!total_rows) return;

    size_t w_row_bytes = (cols / 8) * 4;
    size_t grp_row_bytes = (cols / group_size) * 2;
    bool bf16_flag = is_bf16 != 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (sofuu_cpu_has_dotprod()) {
        size_t q8_blocks = cols / 32;
        thread_local std::vector<sofuu_q8_block_ext> tls_x_q8;
        if (tls_x_q8.size() < q8_blocks) tls_x_q8.resize(q8_blocks);
        sofuu_quantize_row_q8_a76(x, tls_x_q8.data(), q8_blocks);
        const sofuu_q8_block_ext *x_q8_ptr = tls_x_q8.data();

        g_pool.parallel_for(0, total_rows, [=](size_t start, size_t end) {
            size_t mat = 0;
            size_t mat_start = 0;
            size_t mat_end = rows[0];
            for (size_t r = start; r < end; ++r) {
                while (r >= mat_end) {
                    ++mat;
                    mat_start = mat_end;
                    mat_end = mat_start + rows[mat];
                }
                size_t local = r - mat_start;
                outs[mat][local] = sofuu_vec_dot_mlx4_dotprod_a76(
                    (const uint8_t *)w_bytes[mat] + local * w_row_bytes,
                    (const uint8_t *)scales_bytes[mat] + local * grp_row_bytes,
                    (const uint8_t *)biases_bytes[mat] + local * grp_row_bytes,
                    x, cols, group_size, x_q8_ptr, bf16_flag);
            }
        });
        return;
    }
#endif

    g_pool.parallel_for(0, total_rows, [=](size_t start, size_t end) {
        size_t mat = 0;
        size_t mat_start = 0;
        size_t mat_end = rows[0];
        for (size_t r = start; r < end; ++r) {
            while (r >= mat_end) {
                ++mat;
                mat_start = mat_end;
                mat_end = mat_start + rows[mat];
            }
            size_t local = r - mat_start;
#if defined(SOFUU_NEON)
            outs[mat][local] = vec_dot_mlx4_neon(
                (const uint8_t *)w_bytes[mat] + local * w_row_bytes,
                (const uint8_t *)scales_bytes[mat] + local * grp_row_bytes,
                (const uint8_t *)biases_bytes[mat] + local * grp_row_bytes,
                x, cols, group_size, bf16_flag);
#else
            outs[mat][local] = vec_dot_mlx4_scalar(
                (const uint8_t *)w_bytes[mat] + local * w_row_bytes,
                (const uint8_t *)scales_bytes[mat] + local * grp_row_bytes,
                (const uint8_t *)biases_bytes[mat] + local * grp_row_bytes,
                x, cols, group_size, bf16_flag);
#endif
        }
    });
}

// ── Multi-Threaded Batched Matrix-Matrix (GEMM) Kernels ───────────────
void sofuu_gemm_q4_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_q4_0(weight_bytes, x, out, rows, cols);
        return;
    }
    size_t n_blocks = cols / 32;
    size_t row_size = n_blocks * sizeof(block_q4_0);
    const uint8_t *w = (const uint8_t *)weight_bytes;

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        alignas(32) float stack_buf[4096];
        std::vector<float> heap_buf;
        float *row_f32 = stack_buf;
        if (cols > 4096) {
            heap_buf.resize(cols);
            row_f32 = heap_buf.data();
        }

        for (size_t r = start; r < end; ++r) {
            sofuu_dequantize_row_q4_0(w + r * row_size, row_f32, cols);

            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row_f32, x + b * cols, cols);
            }
        }
    });
}

// Speculative batches arrive at 2-3 (occasionally 5-7) rows; the 4-row tile
// GEMM gate would push them to row-wise SDOT fallbacks that re-stream the
// whole weight set once per row. Pad activations up to a 4-multiple (capped
// at 8 padded rows) inside the tile dispatch instead: one weight stream per
// spec step, and the caller only ever sees its real rows.
static inline size_t gemm_tile_pad_batch(size_t batch_size) {
    const size_t rpad = (batch_size + 3) & ~(size_t)3;
    return (batch_size >= 2 && batch_size < rpad && rpad <= 8) ? rpad : batch_size;
}

void sofuu_gemm_q8_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_q8_0(weight_bytes, x, out, rows, cols);
        return;
    }
    size_t n_blocks = cols / 32;
    size_t row_size = n_blocks * sizeof(block_q8_0);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    // Packed-tile prefill (q8_0 4x8), extended to small speculative batches
    // by padding activations to a 4-multiple inside the call.
    const size_t tile_batch = batch_size % 4 == 0 ? batch_size : gemm_tile_pad_batch(batch_size);
    if (tile_batch >= 4 && tile_batch % 4 == 0 && rows % 4 == 0 && cols % 32 == 0 && SOFUU_TILE_TYPES_OK(8)) {
        if (const uint8_t *tile = sofuu_tile_lookup(weight_bytes, 8, rows, cols)) {
            const size_t nb = cols / 32;
            tile_block_q8_0x4 *qa_ptr = tile_scratch<tile_block_q8_0x4>((tile_batch / 4) * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            for (size_t g4 = 0; g4 < tile_batch / 4; ++g4) {
                ggml_quantize_mat_q8_0_4x4(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemm_q8_0_4x4_q8_0(cols, out_dst + g * 4, rows,
                        tile + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, tile_batch, 4);
                }
            });
            if (tile_batch != batch_size) std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
            return;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (optionally tiled 8x8 by gemm_q8_0_tile below).
    // Small-col matrices stay on the f32 x8 path: with cols < 2048 the
    // dequant-once + register-reuse pattern beats weight re-reads.
    if (sofuu_cpu_has_dotprod() && (cols % 32 == 0) && cols >= 2048) {
        const size_t q8_blocks = cols / 32;
        sofuu_q8_block_ext *xq8_base = tile_scratch<sofuu_q8_block_ext>(batch_size * q8_blocks);
        for (size_t b = 0; b < batch_size; ++b) {
            sofuu_quantize_row_q8_a76(x + b * cols, xq8_base + b * q8_blocks, q8_blocks);
        }
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                const block_q8_0 *row = (const block_q8_0 *)(w + r * row_size);
                for (size_t b = 0; b < batch_size; ++b) {
                    out[b * rows + r] = sofuu_vec_dot_q8_0_dotprod_a76(row, xq8_base + b * q8_blocks, q8_blocks);
                }
            }
        });
        return;
    }
#endif

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        alignas(32) float stack_buf[4096];
        std::vector<float> heap_buf;
        float *row_f32 = stack_buf;
        if (cols > 4096) {
            heap_buf.resize(cols);
            row_f32 = heap_buf.data();
        }

        for (size_t r = start; r < end; ++r) {
            sofuu_dequantize_row_q8_0(w + r * row_size, row_f32, cols);

            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row_f32, x + b * cols, cols);
            }
        }
    });
}

void sofuu_gemm_q4_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_q4_k(weight_bytes, x, out, rows, cols);
        return;
    }
    size_t n_blocks = cols / 256;
    size_t row_size = n_blocks * sizeof(block_q4_K);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    // Packed-tile prefill: repacked 8-row weight tiles + q8_K batch
    // activations, extended to small speculative batches by padding
    // activations to a 4-multiple inside the call.
    const size_t tile_batch = batch_size % 4 == 0 ? batch_size : gemm_tile_pad_batch(batch_size);
    if (tile_batch >= 4 && tile_batch % 4 == 0 && rows % 8 == 0 && cols % 256 == 0 && SOFUU_TILE_TYPES_OK(12)) {
        const uint8_t *sc = nullptr;
        if (const uint8_t *tile = sofuu_tile_lookup_sc(weight_bytes, 12, rows, cols, &sc)) {
            const size_t nb = cols / 256;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>((tile_batch / 4) * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            for (size_t g4 = 0; g4 < tile_batch / 4; ++g4) {
                ggml_quantize_mat_q8_K_4x4_generic(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t = tile + g * nb * sizeof(tile_block_q4_Kx8);
                    if (sc) {
                        ggml_gemm_q4_K_8x4_q8_K_sc(cols, out_dst + g * 8, rows, t, qa_ptr, tile_batch, 8,
                            sc + g * nb * sizeof(tile_sc_q4_Kx8));
                    } else {
                        ggml_gemm_q4_K_8x4_q8_K(cols, out_dst + g * 8, rows, t, qa_ptr, tile_batch, 8);
                    }
                }
            });
            if (tile_batch != batch_size) std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
            return;
        }
    }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (see gemm_q8_0 note): cols >= 2048 only.
    if (sofuu_cpu_has_dotprod() && (cols % 256 == 0) && cols >= 2048) {
        const size_t q8_blocks = cols / 32;
        sofuu_q8_block_ext *xq8_base = tile_scratch<sofuu_q8_block_ext>(batch_size * q8_blocks);
        for (size_t b = 0; b < batch_size; ++b) {
            sofuu_quantize_row_q8_a76(x + b * cols, xq8_base + b * q8_blocks, q8_blocks);
        }
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                const block_q4_K *row = (const block_q4_K *)(w + r * row_size);
                for (size_t b = 0; b < batch_size; ++b) {
                    out[b * rows + r] = sofuu_vec_dot_q4_k_dotprod_a76(row, xq8_base + b * q8_blocks, n_blocks);
                }
            }
        });
        return;
    }
#endif

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        alignas(32) float stack_buf[4096];
        std::vector<float> heap_buf;
        float *row_f32 = stack_buf;
        if (cols > 4096) {
            heap_buf.resize(cols);
            row_f32 = heap_buf.data();
        }

        for (size_t r = start; r < end; ++r) {
            sofuu_dequantize_row_q4_k(w + r * row_size, row_f32, cols);

            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row_f32, x + b * cols, cols);
            }
        }
    });
}

void sofuu_gemm_q6_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_q6_k(weight_bytes, x, out, rows, cols);
        return;
    }
    size_t n_blocks = cols / 256;
    size_t row_size = n_blocks * sizeof(block_q6_K);
    const uint8_t *w = (const uint8_t *)weight_bytes;

#if defined(__aarch64__)
    // Packed-tile prefill (q6_K 8x8), extended to small speculative batches
    // by padding activations to a 4-multiple inside the call.
    const size_t tile_batch = batch_size % 4 == 0 ? batch_size : gemm_tile_pad_batch(batch_size);
    if (tile_batch >= 4 && tile_batch % 4 == 0 && rows % 8 == 0 && cols % 256 == 0 && SOFUU_TILE_TYPES_OK(14)) {
        if (const uint8_t *tile = sofuu_tile_lookup(weight_bytes, 14, rows, cols)) {
            const size_t nb = cols / 256;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>((tile_batch / 4) * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            for (size_t g4 = 0; g4 < tile_batch / 4; ++g4) {
                ggml_quantize_mat_q8_K_4x4_generic(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemm_q6_K_8x4_q8_K(cols, out_dst + g * 8, rows,
                        tile + g * nb * sizeof(tile_block_q6_Kx8), qa_ptr, tile_batch, 8);
                }
            });
            if (tile_batch != batch_size) std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
            return;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (see gemm_q8_0 note): cols >= 2048 only.
    if (sofuu_cpu_has_dotprod() && (cols % 256 == 0) && cols >= 2048) {
        const size_t q8_blocks = cols / 32;
        sofuu_q8_block_ext *xq8_base = tile_scratch<sofuu_q8_block_ext>(batch_size * q8_blocks);
        for (size_t b = 0; b < batch_size; ++b) {
            sofuu_quantize_row_q8_a76(x + b * cols, xq8_base + b * q8_blocks, q8_blocks);
        }
        g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
            for (size_t r = start; r < end; ++r) {
                const block_q6_K *row = (const block_q6_K *)(w + r * row_size);
                for (size_t b = 0; b < batch_size; ++b) {
                    out[b * rows + r] = sofuu_vec_dot_q6_k_dotprod_a76(row, xq8_base + b * q8_blocks, n_blocks);
                }
            }
        });
        return;
    }
#endif

    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        alignas(32) float stack_buf[4096];
        std::vector<float> heap_buf;
        float *row_f32 = stack_buf;
        if (cols > 4096) {
            heap_buf.resize(cols);
            row_f32 = heap_buf.data();
        }

        for (size_t r = start; r < end; ++r) {
            sofuu_dequantize_row_q6_k(w + r * row_size, row_f32, cols);

            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row_f32, x + b * cols, cols);
            }
        }
    });
}

void sofuu_gemm_f32(const float *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_f32(weight_bytes, x, out, rows, cols);
        return;
    }
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
            const float *row = weight_bytes + r * cols;
            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row, x + b * cols, cols);
            }
        }
    });
}

void sofuu_gemm_f16(const uint16_t *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    if (rows == 0 || cols == 0 || batch_size == 0) return;
    if (batch_size == 1) {
        sofuu_gemv_f16(weight_bytes, x, out, rows, cols);
        return;
    }
    g_pool.parallel_for(0, rows, [=](size_t start, size_t end) {
        alignas(32) float stack_buf[4096];
        std::vector<float> heap_buf;
        float *row_f32 = stack_buf;
        if (cols > 4096) {
            heap_buf.resize(cols);
            row_f32 = heap_buf.data();
        }

        for (size_t r = start; r < end; ++r) {
            const uint16_t *row = weight_bytes + r * cols;
            for (size_t c = 0; c < cols; ++c) {
                row_f32[c] = f16_to_f32_val(row[c]);
            }
            size_t b = 0;
            for (; b + 7 < batch_size; b += 8) {
                float o0, o1, o2, o3, o4, o5, o6, o7;
                sofuu_vec_dot_f32_x8(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     x + (b + 4) * cols, x + (b + 5) * cols,
                                     x + (b + 6) * cols, x + (b + 7) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
                out[(b + 4) * rows + r] = o4;
                out[(b + 5) * rows + r] = o5;
                out[(b + 6) * rows + r] = o6;
                out[(b + 7) * rows + r] = o7;
            }
            for (; b + 3 < batch_size; b += 4) {
                float o0, o1, o2, o3;
                sofuu_vec_dot_f32_x4(row_f32,
                                     x + (b + 0) * cols, x + (b + 1) * cols,
                                     x + (b + 2) * cols, x + (b + 3) * cols,
                                     cols,
                                     &o0, &o1, &o2, &o3);
                out[(b + 0) * rows + r] = o0;
                out[(b + 1) * rows + r] = o1;
                out[(b + 2) * rows + r] = o2;
                out[(b + 3) * rows + r] = o3;
            }
            for (; b < batch_size; ++b) {
                out[b * rows + r] = sofuu_vec_dot_f32(row_f32, x + b * cols, cols);
            }
        }
    });
}

// ── RMSNorm ──────────────────────────────────────────────────────────
void sofuu_rmsnorm(float *out, const float *x, const float *weight, size_t n, float eps) {
    float sum_sq = 0.0f;

#if defined(SOFUU_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, vx, vx);
    }
    sum_sq = vgetq_lane_f32(vsum, 0) + vgetq_lane_f32(vsum, 1) +
             vgetq_lane_f32(vsum, 2) + vgetq_lane_f32(vsum, 3);
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
#elif defined(SOFUU_AVX2)
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    alignas(32) float buf[8];
    _mm256_storeu_ps(buf, vsum);
    for (int k = 0; k < 8; ++k) sum_sq += buf[k];
    for (; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
#endif

    float scale = 1.0f / sqrtf((sum_sq / (float)n) + eps);

#if defined(SOFUU_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    size_t j = 0;
    for (; j + 3 < n; j += 4) {
        float32x4_t vx = vld1q_f32(x + j);
        float32x4_t vw = vld1q_f32(weight + j);
        float32x4_t vo = vmulq_f32(vmulq_f32(vx, vscale), vw);
        vst1q_f32(out + j, vo);
    }
    for (; j < n; ++j) {
        out[j] = x[j] * scale * weight[j];
    }
#elif defined(SOFUU_AVX2)
    __m256 vscale = _mm256_set1_ps(scale);
    size_t j = 0;
    for (; j + 7 < n; j += 8) {
        __m256 vx = _mm256_loadu_ps(x + j);
        __m256 vw = _mm256_loadu_ps(weight + j);
        __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out + j, vo);
    }
    for (; j < n; ++j) {
        out[j] = x[j] * scale * weight[j];
    }
#else
    for (size_t j = 0; j < n; ++j) {
        out[j] = x[j] * scale * weight[j];
    }
#endif
}

// ── SwiGLU Activation ────────────────────────────────────────────────
void sofuu_swiglu(float *out, const float *gate, const float *up, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// ── Numerically Stable Softmax ───────────────────────────────────────
// ── Batched elementwise glue: one FFI call per layer instead of one per
// token; internally parallel over rows. Kills per-call FFI overhead that
// dominates small-elementwork in batched prefill. ─────────────────────────
void sofuu_batched_rmsnorm(float *out, const float *in, const float *weight, size_t rows, size_t cols, float eps) {
    if (rows == 0 || cols == 0) return;
    g_pool.parallel_for(0, rows, [=](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            sofuu_rmsnorm(out + r * cols, in + r * cols, weight, cols, eps);
        }
    });
}

void sofuu_batched_mad(float *dst, const float *src, float alpha, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    g_pool.parallel_for(0, rows, [=](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            sofuu_vec_mad_f32(dst + r * cols, src + r * cols, alpha, cols);
        }
    });
}

void sofuu_batched_add_bias(float *x, const float *bias, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    g_pool.parallel_for(0, rows, [=](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            float *row = x + r * cols;
            for (size_t i = 0; i < cols; ++i) {
                row[i] += bias[i];
            }
        }
    });
}

void sofuu_batched_swiglu(float *gate, const float *up, size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) return;
    g_pool.parallel_for(0, rows, [=](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            sofuu_swiglu(gate + r * cols, gate + r * cols, up + r * cols, cols);
        }
    });
}

void sofuu_batched_rope_kv(
    float *q_batch, float *k_batch, const float *v_batch,
    float *k_cache, float *v_cache,
    size_t b_size, size_t start_pos, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t max_seq_len, float freq_base, int rope_mode
) {
    if (b_size == 0) return;
    size_t q_dim = n_heads * head_dim;
    size_t kv_dim = n_kv_heads * head_dim;
    g_pool.parallel_for(0, b_size, [=](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            size_t pos = start_pos + b;
            sofuu_rope(q_batch + b * q_dim, n_heads, head_dim, pos, freq_base, rope_mode);
            sofuu_rope(k_batch + b * kv_dim, n_kv_heads, head_dim, pos, freq_base, rope_mode);
            size_t cache_pos = pos % max_seq_len;
            const float *v_row = v_batch + b * kv_dim;
            for (size_t kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
                size_t slot = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim + cache_pos * head_dim;
                memcpy(k_cache + slot, k_batch + b * kv_dim + kv_h * head_dim, head_dim * sizeof(float));
                memcpy(v_cache + slot, v_row + kv_h * head_dim, head_dim * sizeof(float));
            }
        }
    });
}

void sofuu_softmax(float *scores, size_t n) {
    if (n == 0) return;
    float max_val = scores[0];
    for (size_t i = 1; i < n; ++i) {
        if (scores[i] > max_val) max_val = scores[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float exp_v = expf(scores[i] - max_val);
        scores[i] = exp_v;
        sum += exp_v;
    }
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-8f);
    for (size_t i = 0; i < n; ++i) {
        scores[i] *= inv_sum;
    }
}

// ── Rotary Position Embedding (RoPE - NEOX / Qwen / LLaMA style) ─────
void sofuu_rope(float *vec, size_t n_heads, size_t head_dim, size_t pos, float freq_base, int rope_style) {
    size_t half_dim = head_dim / 2;
    for (size_t h = 0; h < n_heads; ++h) {
        float *head = vec + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
            float theta = (float)pos * freq;
            float c = cosf(theta);
            float s = sinf(theta);

            if (rope_style == 1) {
                // GGUF "normal" RoPE: consecutive head values form a pair.
                size_t even = 2 * i;
                size_t odd = even + 1;
                float v0 = head[even];
                float v1 = head[odd];
                head[even] = v0 * c - v1 * s;
                head[odd]  = v1 * c + v0 * s;
            } else {
                // NeoX/Gemma/Qwen-style RoPE: halves form a pair.
                float v0 = head[i];
                float v1 = head[i + half_dim];
                head[i]            = v0 * c - v1 * s;
                head[i + half_dim] = v0 * s + v1 * c;
            }
        }
    }
}

// ── Shift RoPE Position (delta_pos can be negative for left compaction) ──
void sofuu_rope_shift(float *vec, size_t n_heads, size_t head_dim, int64_t delta_pos, float freq_base, int rope_style) {
    if (delta_pos == 0 || n_heads == 0 || head_dim == 0) return;
    size_t half_dim = head_dim / 2;
    for (size_t h = 0; h < n_heads; ++h) {
        float *head = vec + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
            float theta = (float)delta_pos * freq;
            float c = cosf(theta);
            float s = sinf(theta);

            if (rope_style == 1) {
                size_t even = 2 * i;
                size_t odd = even + 1;
                float v0 = head[even];
                float v1 = head[odd];
                head[even] = v0 * c - v1 * s;
                head[odd]  = v1 * c + v0 * s;
            } else {
                float v0 = head[i];
                float v1 = head[i + half_dim];
                head[i]            = v0 * c - v1 * s;
                head[i + half_dim] = v0 * s + v1 * c;
            }
        }
    }
}

// ── In-place KV Cache Compaction across all layers ───────────────────
void sofuu_kv_compact(
    float *k_cache,
    float *v_cache,
    size_t n_layers,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t current_seq_len,
    size_t sink_tokens,
    size_t drop_tokens,
    float freq_base,
    int rope_style
) {
    if (drop_tokens == 0 || current_seq_len <= sink_tokens + drop_tokens) return;
    size_t surviving_recent = current_seq_len - sink_tokens - drop_tokens;
    int64_t delta_pos = -(int64_t)drop_tokens;

    auto compact_layer_range = [=](size_t start_layer, size_t end_layer) {
        for (size_t l = start_layer; l < end_layer; ++l) {
            for (size_t kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
                size_t head_base = (l * n_kv_heads + kv_h) * max_seq_len * head_dim;
                float *k_head = k_cache + head_base;
                float *v_head = v_cache + head_base;

                // 1. Shift V cache (raw activations, direct memmove)
                float *v_dst = v_head + sink_tokens * head_dim;
                const float *v_src = v_head + (sink_tokens + drop_tokens) * head_dim;
                std::memmove(v_dst, v_src, surviving_recent * head_dim * sizeof(float));

                // 2. Shift K cache (memmove)
                float *k_dst = k_head + sink_tokens * head_dim;
                const float *k_src = k_head + (sink_tokens + drop_tokens) * head_dim;
                std::memmove(k_dst, k_src, surviving_recent * head_dim * sizeof(float));

                // 3. Shift RoPE position on moved K cache keys
                for (size_t t = 0; t < surviving_recent; ++t) {
                    sofuu_rope_shift(k_dst + t * head_dim, 1, head_dim, delta_pos, freq_base, rope_style);
                }
            }
        }
    };

    if (n_layers >= 4) {
        g_pool.parallel_for(0, n_layers, compact_layer_range);
    } else {
        compact_layer_range(0, n_layers);
    }
}




// ── Hardware SIMD Multiply-Accumulate (dst[i] += src[i] * scalar) ─────
void sofuu_vec_mad_f32(float *dst, const float *src, float scalar, size_t n) {
#if defined(SOFUU_NEON)
    float32x4_t s = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t d2 = vld1q_f32(dst + i + 8);
        float32x4_t d3 = vld1q_f32(dst + i + 12);
        d0 = vmlaq_f32(d0, vld1q_f32(src + i), s);
        d1 = vmlaq_f32(d1, vld1q_f32(src + i + 4), s);
        d2 = vmlaq_f32(d2, vld1q_f32(src + i + 8), s);
        d3 = vmlaq_f32(d3, vld1q_f32(src + i + 12), s);
        vst1q_f32(dst + i, d0);
        vst1q_f32(dst + i + 4, d1);
        vst1q_f32(dst + i + 8, d2);
        vst1q_f32(dst + i + 12, d3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dst + i);
        d = vmlaq_f32(d, vld1q_f32(src + i), s);
        vst1q_f32(dst + i, d);
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scalar;
    }
#elif defined(SOFUU_AVX2)
    __m256 s = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 d0 = _mm256_loadu_ps(dst + i);
        __m256 d1 = _mm256_loadu_ps(dst + i + 8);
        d0 = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), s, d0);
        d1 = _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 8), s, d1);
        _mm256_storeu_ps(dst + i, d0);
        _mm256_storeu_ps(dst + i + 8, d1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        d = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), s, d);
        _mm256_storeu_ps(dst + i, d);
    }
    for (; i < n; ++i) {
        dst[i] += src[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i] * scalar;
    }
#endif
}

// ── Multi-core Parallel Attention with Sliding Window & Attention Sinks ──
void sofuu_batched_attention_ext(
    const float *q_batch,
    const float *k_cache,
    const float *v_cache,
    float *head_out_batch,
    size_t b_size,
    size_t start_pos,
    size_t layer_idx,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t sliding_window,
    size_t sink_tokens
) {
    if (b_size == 0 || n_heads == 0 || head_dim == 0) return;
    size_t q_dim = n_heads * head_dim;
    size_t n_rep = n_heads / n_kv_heads;
    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);
    size_t total_work = b_size * n_heads;

    auto compute_head_slice = [=](size_t start, size_t end) {
        alignas(32) float local_scores_stack[4096];
        std::vector<float> heap_scores;
        float *scores = local_scores_stack;
        if (max_seq_len > 4096) {
            heap_scores.resize(max_seq_len);
            scores = heap_scores.data();
        }

        for (size_t work_idx = start; work_idx < end; ++work_idx) {
            size_t b = work_idx / n_heads;
            size_t h = work_idx % n_heads;
            size_t kv_h = h / n_rep;
            size_t pos = start_pos + b;
            size_t seq_len = (pos + 1 < max_seq_len) ? (pos + 1) : max_seq_len;

            const float *q_head = q_batch + b * q_dim + h * head_dim;
            size_t head_base = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim;
            const float *k_past = k_cache + head_base;
            const float *v_past = v_cache + head_base;

            if (sliding_window > 0 && seq_len > sliding_window) {
                size_t window_start = seq_len - sliding_window;
                size_t sink_end = (sink_tokens < window_start) ? sink_tokens : window_start;
                size_t active_count = 0;

                for (size_t p = 0; p < sink_end; ++p) {
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }
                for (size_t p = window_start; p < seq_len; ++p) {
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }

                sofuu_softmax(scores, active_count);

                float *out_head = head_out_batch + b * q_dim + h * head_dim;
                std::memset(out_head, 0, head_dim * sizeof(float));

                size_t score_idx = 0;
                for (size_t p = 0; p < sink_end; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[score_idx++], head_dim);
                }
                for (size_t p = window_start; p < seq_len; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[score_idx++], head_dim);
                }
            } else {
                for (size_t p = 0; p < seq_len; ++p) {
                    scores[p] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }

                sofuu_softmax(scores, seq_len);

                float *out_head = head_out_batch + b * q_dim + h * head_dim;
                std::memset(out_head, 0, head_dim * sizeof(float));

                for (size_t p = 0; p < seq_len; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[p], head_dim);
                }
            }
        }
    };

    size_t effective_len = (sliding_window > 0 && start_pos > sliding_window) ? sliding_window : (start_pos + b_size / 2 + 1);
    bool run_parallel = (total_work >= 32) || (total_work >= 16 && effective_len >= 128);

    if (run_parallel) {
        g_pool.parallel_for(0, total_work, compute_head_slice);
    } else {
        compute_head_slice(0, total_work);
    }
}

void sofuu_batched_attention(
    const float *q_batch,
    const float *k_cache,
    const float *v_cache,
    float *head_out_batch,
    size_t b_size,
    size_t start_pos,
    size_t layer_idx,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len
) {
    sofuu_batched_attention_ext(
        q_batch, k_cache, v_cache, head_out_batch,
        b_size, start_pos, layer_idx, n_heads, n_kv_heads, head_dim, max_seq_len,
        0, 0
    );
}

void sofuu_tree_rope(
    float *q_batch,
    float *k_batch,
    const size_t *node_positions,
    size_t tree_size,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    float freq_base,
    int rope_style
) {
    if (tree_size == 0) return;
    size_t q_dim = n_heads * head_dim;
    size_t kv_dim = n_kv_heads * head_dim;
    g_pool.parallel_for(0, tree_size, [=](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) {
            size_t pos = node_positions[b];
            sofuu_rope(q_batch + b * q_dim, n_heads, head_dim, pos, freq_base, rope_style);
            sofuu_rope(k_batch + b * kv_dim, n_kv_heads, head_dim, pos, freq_base, rope_style);
        }
    });
}

void sofuu_tree_attention(
    const float *q_batch,
    const float *k_batch,
    const float *v_batch,
    const float *k_cache,
    const float *v_cache,
    float *head_out_batch,
    const uint8_t *ancestry_mask,
    size_t tree_size,
    size_t start_pos,
    size_t layer_idx,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    size_t max_seq_len,
    size_t sliding_window,
    size_t sink_tokens
) {
    if (tree_size == 0 || n_heads == 0 || head_dim == 0) return;
    size_t q_dim = n_heads * head_dim;
    size_t kv_dim = n_kv_heads * head_dim;
    size_t n_rep = n_heads / n_kv_heads;
    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);
    size_t total_work = tree_size * n_heads;

    auto compute_head_slice = [=](size_t start, size_t end) {
        alignas(32) float local_scores_stack[4096];
        std::vector<float> heap_scores;
        float *scores = local_scores_stack;
        if (max_seq_len + tree_size > 4096) {
            heap_scores.resize(max_seq_len + tree_size);
            scores = heap_scores.data();
        }

        size_t valid_j[128];

        for (size_t work_idx = start; work_idx < end; ++work_idx) {
            size_t i = work_idx / n_heads;
            size_t h = work_idx % n_heads;
            size_t kv_h = h / n_rep;

            const float *q_head = q_batch + i * q_dim + h * head_dim;
            size_t head_base = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim;
            const float *k_past = k_cache + head_base;
            const float *v_past = v_cache + head_base;

            size_t active_count = 0;

            // 1. Historical KV Cache
            if (sliding_window > 0 && start_pos > sliding_window) {
                size_t window_start = start_pos - sliding_window;
                size_t sink_end = (sink_tokens < window_start) ? sink_tokens : window_start;

                for (size_t p = 0; p < sink_end; ++p) {
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }
                for (size_t p = window_start; p < start_pos; ++p) {
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }
            } else {
                for (size_t p = 0; p < start_pos; ++p) {
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
                }
            }

            size_t hist_count = active_count;

            // 2. In-Tree Candidates (only valid ancestors)
            size_t num_valid_j = 0;
            for (size_t j = 0; j < tree_size; ++j) {
                bool is_valid = (ancestry_mask != nullptr) ? (ancestry_mask[i * tree_size + j] != 0) : (j <= i);
                if (is_valid) {
                    if (num_valid_j < 128) {
                        valid_j[num_valid_j++] = j;
                    }
                    const float *k_cand = k_batch + j * kv_dim + kv_h * head_dim;
                    scores[active_count++] = sofuu_vec_dot_f32(q_head, k_cand, head_dim) * inv_sqrt_head_dim;
                }
            }

            sofuu_softmax(scores, active_count);

            float *out_head = head_out_batch + i * q_dim + h * head_dim;
            std::memset(out_head, 0, head_dim * sizeof(float));

            // Accumulate historical V
            if (sliding_window > 0 && start_pos > sliding_window) {
                size_t window_start = start_pos - sliding_window;
                size_t sink_end = (sink_tokens < window_start) ? sink_tokens : window_start;
                size_t s_idx = 0;
                for (size_t p = 0; p < sink_end; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[s_idx++], head_dim);
                }
                for (size_t p = window_start; p < start_pos; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[s_idx++], head_dim);
                }
            } else {
                for (size_t p = 0; p < hist_count; ++p) {
                    sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[p], head_dim);
                }
            }

            // Accumulate in-tree V
            for (size_t idx = 0; idx < num_valid_j; ++idx) {
                size_t j = valid_j[idx];
                const float *v_cand = v_batch + j * kv_dim + kv_h * head_dim;
                sofuu_vec_mad_f32(out_head, v_cand, scores[hist_count + idx], head_dim);
            }
        }
    };

    size_t effective_len = (sliding_window > 0 && start_pos > sliding_window) ? sliding_window : start_pos;
    bool run_parallel = (total_work >= 32) || (total_work >= 16 && effective_len >= 128);

    if (run_parallel) {
        g_pool.parallel_for(0, total_work, compute_head_slice);
    } else {
        compute_head_slice(0, total_work);
    }
}

// ── Micro-bench: attribute prefill GEMM cost between the activation-quantize
// prologue and the tile kernel (Pi 5 diagnostics; SOFUU_BENCH_GEMM env hook).
// Synthetic bit-pattern weights (throughput is value-independent); the tile
// warm is excluded from timing.
extern "C" void sofuu_bench_gemm(unsigned type, int rows, int cols, int batch, int iters) {
#if defined(__aarch64__)
    if (batch < 4 || batch % 4 != 0 || iters < 1) return;
    const bool qk = (type == 12);
    if (!qk && type != 8) return;
    const size_t row_size = qk ? (size_t)(cols / 256) * 144 : (size_t)(cols / 32) * 34;
    std::vector<uint8_t> w((size_t)rows * row_size);
    std::vector<float> x((size_t)batch * cols);
    std::vector<float> out((size_t)batch * rows);
    uint32_t rng = 12345;
    for (auto &b : w) { rng = rng * 1664525u + 1013904223u; b = (uint8_t)(rng >> 24); }
    for (auto &v : x) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 16) % 2000 - 1000) / 700.0f; }
    if (sofuu_tile_warm(w.data(), rows, cols, qk ? 12 : 8) != 0) {
        fprintf(stderr, "[bench] tile warm failed\n");
        return;
    }
    double t_quant = 0, t_kernel = 0;
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point t1;
        if (qk) {
            const size_t nb = cols / 256;
            tile_block_q8_Kx4 *qa = tile_scratch<tile_block_q8_Kx4>((batch / 4) * nb);
            for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
                ggml_quantize_mat_q8_K_4x4_generic(x.data() + g4 * 4 * cols, qa + g4 * nb, cols);
            }
            t1 = std::chrono::steady_clock::now();
            const uint8_t *sc = nullptr;
            const uint8_t *tile = sofuu_tile_lookup_sc(w.data(), 12, rows, cols, &sc);
            g_pool.parallel_for(0, rows / 8, [&](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t = tile + g * nb * sizeof(tile_block_q4_Kx8);
                    if (sc) {
                        ggml_gemm_q4_K_8x4_q8_K_sc(cols, out.data() + g * 8, rows, t, qa, batch, 8,
                            sc + g * nb * sizeof(tile_sc_q4_Kx8));
                    } else {
                        ggml_gemm_q4_K_8x4_q8_K(cols, out.data() + g * 8, rows, t, qa, batch, 8);
                    }
                }
            });
        } else {
            const size_t nb = cols / 32;
            tile_block_q8_0x4 *qa = tile_scratch<tile_block_q8_0x4>((batch / 4) * nb);
            for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
                ggml_quantize_mat_q8_0_4x4(x.data() + g4 * 4 * cols, qa + g4 * nb, cols);
            }
            t1 = std::chrono::steady_clock::now();
            const uint8_t *tile = sofuu_tile_lookup(w.data(), 8, rows, cols);
            g_pool.parallel_for(0, rows / 4, [&](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemm_q8_0_4x4_q8_0(cols, out.data() + g * 4, rows,
                        tile + g * nb * sizeof(tile_block_q8_0x4), qa, batch, 4);
                }
            });
        }
        auto t2 = std::chrono::steady_clock::now();
        t_quant += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_kernel += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    const double macs = (double)batch * rows * cols * iters;
    fprintf(stderr,
        "[bench] %s %dx%d batch=%d iters=%d: quant %.1fms (%.2fms/iter) | kernel %.1fms (%.2fms/iter) = %.1f GMAC/s | quant share %.1f%%\n",
        qk ? "q4_K" : "q8_0", rows, cols, batch, iters,
        t_quant, t_quant / iters, t_kernel, t_kernel / iters,
        macs / (t_kernel * 1e6), 100.0 * t_quant / (t_quant + t_kernel));
#endif
}

} // extern "C"

