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
#include <memory>

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
    std::vector<uint8_t> data_p;  // q4_K nibble-plane tile (tile_block_q4_Kx8p per block)
    std::vector<uint8_t> data_s;  // q4_K pre-decoded scale section (tile_ss_q4_Kx8s per block)
    unsigned type;
    size_t rows, cols;
    uint64_t fp;
    bool has_sc = false;
    bool has_p = false;
    bool has_s = false;
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

// Nibble-plane q4_K tiles (prefill GEMM only). OPT-IN via SOFUU_NIBBLE_PLANES=1:
// measured on Pi 5 (2026-10-14) at +5.3-7.8% over the packed kernel (129-130 vs
// 120-122 GMAC/s) across real Qwen GEMM shapes — well short of the ≥15% ship
// bar. The remaining q4_K-vs-q8_0 gap (130 vs 185 GMAC/s) lives in the
// per-subblock scale-application chain (vmulq/vmovl/vcvtq int lanes), not the
// vand/vshr nibble unpacks the planes remove. Kept as bit-exact infrastructure
// (parity-tested; bench via SOFUU_BENCH_GEMM rows,cols,batch,iters,plane_iters).
// Planes cost +0.5 weight bytes/element over the packed tile, so the auto gate
// would cap each tensor at 16M elements with a global default budget of 768 MB
// (SOFUU_NIBBLE_PLANES_MB overrides; 0 = unlimited) — relevant only when opted in.
static bool nibble_planes_enabled() {
    static const int v = [] {
        const char * e = getenv("SOFUU_NIBBLE_PLANES");
        return e != nullptr && e[0] == '1';
    }();
    return v != 0;
}
static size_t nibble_planes_budget_bytes() {
    static const size_t v = [] {
        const char * e = getenv("SOFUU_NIBBLE_PLANES_MB");
        if (!e) return (size_t)768 << 20;
        return (size_t)strtoull(e, nullptr, 10) << 20;
    }();
    return v;
}
static bool nibble_planes_auto(size_t rows, size_t cols) {
    if (!nibble_planes_enabled()) return false;
    const size_t elems = rows * cols;
    if (elems > (size_t)16 << 20) return false;
    static std::atomic<size_t> used{0};
    const size_t need = elems / 2;  // one extra byte per element (lo+hi vs packed nibble)
    const size_t budget = nibble_planes_budget_bytes();
    size_t u = used.load(std::memory_order_relaxed);
    while (true) {
        if (budget != 0 && u + need > budget) return false;
        if (used.compare_exchange_weak(u, u + need, std::memory_order_relaxed)) return true;
    }
}

// T1 (2026-10-15): repack-time pre-decoded sub-block scales for q4_K, as a
// separate 192 B/block region the prefill GEMM loads beside the stock tile
// (bit-exact; +65% kernel GMAC/s on the Qwen FFN shape, +16.7% tile bytes).
// The decode GEMV never reads the region, so its bandwidth path is
// untouched. Default ON; `SOFUU_Q4_SS=0` kills it. Per-tensor cap 16M elems,
// global budget `SOFUU_Q4_SS_MB` (default 768 MB).
static bool q4_ss_enabled() {
    static const bool v = [] {
        const char * e = getenv("SOFUU_Q4_SS");
        return !(e != nullptr && e[0] == '0');
    }();
    return v;
}
static bool q4_ss_eligible(size_t rows, size_t cols) {
    return q4_ss_enabled() && rows * cols <= (size_t)16 << 20;
}
static bool q4_ss_admit(size_t rows, size_t cols) {
    if (!q4_ss_eligible(rows, cols)) return false;
    static std::atomic<size_t> used{0};
    const size_t need = (rows * cols) / 256 * 192;
    static const size_t budget = [] {
        const char * e = getenv("SOFUU_Q4_SS_MB");
        return e ? (size_t)strtoull(e, nullptr, 10) << 20 : (size_t)768 << 20;
    }();
    size_t u = used.load(std::memory_order_relaxed);
    while (true) {
        if (budget != 0 && u + need > budget) return false;
        if (used.compare_exchange_weak(u, u + need, std::memory_order_relaxed)) return true;
    }
}

extern "C" int sofuu_tile_warm(const void *w, size_t rows, size_t cols, unsigned type) {
    if (getenv("SOFUU_NO_TILES")) return -1; // runtime kill-switch
    bool aligned;
    const bool want_sc = (type == 12) && (scale_sidecar_disabled_state() == 0);
    const bool want_s = (type == 12) && q4_ss_eligible(rows, cols) && rows % 8 == 0 && cols % 256 == 0;
    if (type == 12 || type == 14) aligned = (rows % 8 == 0 && cols % 256 == 0);
    else if (type == 8) aligned = (rows % 4 == 0 && cols % 32 == 0);
    else return -1;
    if (!aligned || rows == 0 || cols == 0) return -1;
    const uint64_t fp = sofuu_tile_fingerprint(w);
    {
        std::lock_guard<std::mutex> lk(g_tile_mutex);
        auto it = g_tile_map.find(w);
        if (it != g_tile_map.end() && it->second.type == type && it->second.rows == rows &&
            it->second.cols == cols && it->second.fp == fp && it->second.has_sc == want_sc &&
            it->second.has_s == want_s) return 0;
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
        // Nibble-plane expansion for the prefill GEMM (bit-identical math,
        // vand/vshr unpack moved out of the hot loop). Gated per tensor; the
        // packed tile above always exists for the GEMV.
        if (nibble_planes_auto(rows, cols)) {
            static_assert(sizeof(tile_block_q4_Kx8p) == 2176, "q4_K plane tile layout drift");
            const size_t np = (size_t)r8 * nb;
            e.data_p.resize(np * sizeof(tile_block_q4_Kx8p));
            uint8_t * dstp = e.data_p.data();
            if (r8 >= 16) {
                g_pool.parallel_for(0, (size_t)r8, [=](size_t g0, size_t g1) {
                    sofuu_repack_q4k_8bl_planes_range(dstp, src, nb, g0, g1, 4);
                });
            } else {
                sofuu_repack_q4k_8bl_planes(dstp, src, rows, nb, 4);
            }
            e.has_p = true;
        }
#if defined(__ARM_FEATURE_DOTPROD)
        if (want_s && q4_ss_admit(rows, cols)) {
            static_assert(sizeof(tile_ss_q4_Kx8s) == 192, "q4_K ss-region layout drift");
            const size_t n_blk = (size_t)r8 * nb;
            e.data_s.resize(n_blk * sizeof(tile_ss_q4_Kx8s));
            tile_ss_q4_Kx8s *         outs = (tile_ss_q4_Kx8s *)e.data_s.data();
            const tile_block_q4_Kx8 * ins  = (const tile_block_q4_Kx8 *)e.data.data();
            auto fill_s = [=](size_t i0, size_t i1) {
                for (size_t i = i0; i < i1; i++) {
                    fill_q4k_s_block(&outs[i], &ins[i]);
                }
            };
            if (n_blk >= 16) {
                g_pool.parallel_for(0, n_blk, fill_s);
            } else {
                fill_s(0, n_blk);
            }
            e.has_s = true;
        }
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

// Nibble-plane tile for the q4_K prefill GEMM (same keying; only present
// when warm built one under the size/budget gate).
static const uint8_t * sofuu_tile_lookup_p(const void *w, size_t rows, size_t cols) {
    const uint64_t fp = sofuu_tile_fingerprint(w);
    std::lock_guard<std::mutex> lk(g_tile_mutex);
    auto it = g_tile_map.find(w);
    if (it == g_tile_map.end()) return nullptr;
    const sofuu_tile_entry &e = it->second;
    if (e.type == 12 && e.rows == rows && e.cols == cols && e.fp == fp && e.has_p) return e.data_p.data();
    return nullptr;
}

// T1 pre-decoded-scale section (tile_ss_q4_Kx8s region). Present when warm
// admitted this tensor under SOFUU_Q4_SS gates; the GEMM takes it as a second
// stream beside the stock tile. Same convention as lookup_sc: the return is
// the stock tile pointer and *ss_out receives the scale-section pointer
// (nullptr when no s-entry exists — callers fall back).
static const uint8_t * sofuu_tile_lookup_s(const void *w, size_t rows, size_t cols, const uint8_t **ss_out) {
    const uint64_t fp = sofuu_tile_fingerprint(w);
    std::lock_guard<std::mutex> lk(g_tile_mutex);
    *ss_out = nullptr;
    auto it = g_tile_map.find(w);
    if (it == g_tile_map.end()) return nullptr;
    const sofuu_tile_entry &e = it->second;
    if (e.type == 12 && e.rows == rows && e.cols == cols && e.fp == fp && e.has_s) {
        *ss_out = e.data_s.data();
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

// Separate host for the pad-to-4 activation copy: the pad paths hold p (the
// zero-padded x) while ALSO handing out_dst from the same thread_local
// tile_scratch vector — a second resize that grows past the first capacity
// reallocates and leaves p DANGLING (quantizing activations from freed
// memory). Bit corruption when rows > cols, e.g. Spark's fused-QKV prefill
// 78→80 pad with rows 4096 > cols 2560 — Qwen never pads (chunks are ÷4) so
// only the pad-bearing models were hit. A dedicated vector per role makes
// both pointers stable regardless of resize order.
template <typename T>
static T *tile_scratch_pad(size_t n) {
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

// In-place elementwise add: dst[i] += src[i]. Fallback kernel for the fused
// residual add on pad paths (same single IEEE add per element, bit-exact).
static inline void sofuu_f32_add_inplace(float * __restrict dst, const float * __restrict src, size_t n) {
#if defined(SOFUU_NEON)
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(dst + i,      vaddq_f32(vld1q_f32(dst + i),      vld1q_f32(src + i)));
        vst1q_f32(dst + i + 4,  vaddq_f32(vld1q_f32(dst + i + 4),  vld1q_f32(src + i + 4)));
        vst1q_f32(dst + i + 8,  vaddq_f32(vld1q_f32(dst + i + 8),  vld1q_f32(src + i + 8)));
        vst1q_f32(dst + i + 12, vaddq_f32(vld1q_f32(dst + i + 12), vld1q_f32(src + i + 12)));
    }
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
#endif
}

// Residual-fused GEMM core. Contract: in_acc must alias out and out must be
// preloaded with the residual; on return 0 out holds w*x + residual. Returns
// 1 when the fused tile path was not taken — out is then UNTOUCHED and the
// caller falls back to an unfused matmat (+ separate residual add).
static int sofuu_gemm_q8_0_impl(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size, const float * in_acc) {
    if (rows == 0 || cols == 0 || batch_size == 0) return in_acc ? 1 : 0;
    if (in_acc && in_acc != out) return 1;  // fused contract: add in place over the residual
    if (batch_size == 1) {
        sofuu_gemv_q8_0(weight_bytes, x, out, rows, cols);
        return 1;  // batch-1 callers fall back to the unfused matvec path
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
                float *p = tile_scratch_pad<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            for (size_t g4 = 0; g4 < tile_batch / 4; ++g4) {
                ggml_quantize_mat_q8_0_4x4(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            const float * acc_k = (in_acc && tile_batch == batch_size) ? out : nullptr;
            if (tile_batch % 8 == 0) {
                // 8-act panel: each weight-tile row + weight-scale convert
                // shared across two 4-act groups; fold order unchanged ->
                // bit-identical outputs, fewer loads per MAC.
                g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
                    for (size_t g = g0; g < g1; ++g) {
                        ggml_gemm_q8_0_8ax4_q8_0(cols, out_dst + g * 4, rows,
                            tile + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, tile_batch, 4, acc_k);
                    }
                });
            } else {
                g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
                    for (size_t g = g0; g < g1; ++g) {
                        ggml_gemm_q8_0_4x4_q8_0(cols, out_dst + g * 4, rows,
                            tile + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, tile_batch, 4, acc_k);
                    }
                });
            }
            if (tile_batch != batch_size) {
                if (in_acc) {
                    sofuu_f32_add_inplace(out, out_dst, batch_size * rows);  // out still holds the residual
                } else {
                    std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
                }
            }
            return 0;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (optionally tiled 8x8 by gemm_q8_0_tile below).
    // Small-col matrices stay on the f32 x8 path: with cols < 2048 the
    // dequant-once + register-reuse pattern beats weight re-reads.
    if (in_acc) return 1;
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
        return 0;
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
    return 0;
}

void sofuu_gemm_q8_0(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    sofuu_gemm_q8_0_impl(weight_bytes, x, out, rows, cols, batch_size, nullptr);
}

extern "C" int sofuu_gemm_q8_0_acc(const void *weight_bytes, const float *x, float *out_acc, size_t rows, size_t cols, size_t batch_size) {
    return sofuu_gemm_q8_0_impl(weight_bytes, x, out_acc, rows, cols, batch_size, out_acc);
}

static int sofuu_gemm_q4_k_impl(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size, const float * in_acc) {
    if (rows == 0 || cols == 0 || batch_size == 0) return in_acc ? 1 : 0;
    if (in_acc && in_acc != out) return 1;  // fused contract: add in place over the residual
    if (batch_size == 1) {
        sofuu_gemv_q4_k(weight_bytes, x, out, rows, cols);
        return 1;  // batch-1 callers fall back to the unfused matvec path
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
        // Nibble-plane tiles first: same act-quant prologue, aux-op-free kernel.
        if (const uint8_t *ptile = sofuu_tile_lookup_p(weight_bytes, rows, cols)) {
            const size_t nb = cols / 256;
            const size_t g4n = tile_batch / 4;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>(g4n * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch_pad<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            if (g4n >= 2) {
                g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
                    for (size_t g4 = a; g4 < b; ++g4) {
                        ggml_quantize_mat_q8_K_4x4_neon(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
                    }
                });
            } else {
                ggml_quantize_mat_q8_K_4x4_neon(gr, qa_ptr, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            const float * acc_k = (in_acc && tile_batch == batch_size) ? out : nullptr;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t = ptile + g * nb * sizeof(tile_block_q4_Kx8p);
                    ggml_gemm_q4_K_8x4_q8_K_p(cols, out_dst + g * 8, rows, t, qa_ptr, tile_batch, 8, acc_k);
                }
            });
            if (tile_batch != batch_size) {
                if (in_acc) {
                    sofuu_f32_add_inplace(out, out_dst, batch_size * rows);  // out still holds the residual
                } else {
                    std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
                }
            }
            return 0;
        }
        const uint8_t *ss = nullptr;
        const uint8_t *stile = sofuu_tile_lookup_s(weight_bytes, rows, cols, &ss);
        if (stile != nullptr && ss != nullptr) {
            const size_t nb = cols / 256;
            const size_t g4n = tile_batch / 4;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>(g4n * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch_pad<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            if (g4n >= 2) {
                g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
                    for (size_t g4 = a; g4 < b; ++g4) {
                        ggml_quantize_mat_q8_K_4x4_neon(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
                    }
                });
            } else {
                ggml_quantize_mat_q8_K_4x4_neon(gr, qa_ptr, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            const float * acc_k = (in_acc && tile_batch == batch_size) ? out : nullptr;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t   = stile + g * nb * sizeof(tile_block_q4_Kx8);
                    const uint8_t *ssg = ss + g * nb * sizeof(tile_ss_q4_Kx8s);
                    ggml_gemm_q4_K_8x4_q8_K_s(cols, out_dst + g * 8, rows, t, ssg, qa_ptr, tile_batch, 8, acc_k);
                }
            });
            if (tile_batch != batch_size) {
                if (in_acc) {
                    sofuu_f32_add_inplace(out, out_dst, batch_size * rows);
                } else {
                    std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
                }
            }
            return 0;
        }
        const uint8_t *sc = nullptr;
        if (const uint8_t *tile = sofuu_tile_lookup_sc(weight_bytes, 12, rows, cols, &sc)) {
            const size_t nb = cols / 256;
            const size_t g4n = tile_batch / 4;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>(g4n * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch_pad<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            // Act-quant on the pool: the scalar generic quantizer was the
            // single-threaded serial prologue (~22-37% of the q4_K path at
            // batch 64 on Pi 5).
            if (g4n >= 2) {
                g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
                    for (size_t g4 = a; g4 < b; ++g4) {
                        ggml_quantize_mat_q8_K_4x4_neon(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
                    }
                });
            } else {
                ggml_quantize_mat_q8_K_4x4_neon(gr, qa_ptr, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            const float * acc_k = (in_acc && tile_batch == batch_size) ? out : nullptr;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    const uint8_t *t = tile + g * nb * sizeof(tile_block_q4_Kx8);
                    if (sc) {
                        ggml_gemm_q4_K_8x4_q8_K_sc(cols, out_dst + g * 8, rows, t, qa_ptr, tile_batch, 8,
                            sc + g * nb * sizeof(tile_sc_q4_Kx8), acc_k);
                    } else {
                        ggml_gemm_q4_K_8x4_q8_K(cols, out_dst + g * 8, rows, t, qa_ptr, tile_batch, 8, acc_k);
                    }
                }
            });
            if (tile_batch != batch_size) {
                if (in_acc) {
                    sofuu_f32_add_inplace(out, out_dst, batch_size * rows);  // out still holds the residual
                } else {
                    std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
                }
            }
            return 0;
        }
    }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (see gemm_q8_0 note): cols >= 2048 only.
    if (in_acc) return 1;
    if (sofuu_cpu_has_dotprod() && (cols % 256 == 0) && cols >= 2048) {
        // sofuu_quantize_row_q8_a76 quantizes 32-element blocks, and the dot
        // kernel reads 8 ext-blocks per q4_K super-block: the buffer unit is
        // cols/32. This was cols/256 — the quantizer covered 1/8 of every
        // row and the dot read past the scratch into stale/foreign bytes
        // (the 9848995df-wave Spark-4B incoherence reached through this
        // fallback whenever the tile path was skipped).
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
        return 0;
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
    return 0;
}

void sofuu_gemm_q4_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    sofuu_gemm_q4_k_impl(weight_bytes, x, out, rows, cols, batch_size, nullptr);
}

extern "C" int sofuu_gemm_q4_k_acc(const void *weight_bytes, const float *x, float *out_acc, size_t rows, size_t cols, size_t batch_size) {
    return sofuu_gemm_q4_k_impl(weight_bytes, x, out_acc, rows, cols, batch_size, out_acc);
}

static int sofuu_gemm_q6_k_impl(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size, const float * in_acc) {
    if (rows == 0 || cols == 0 || batch_size == 0) return in_acc ? 1 : 0;
    if (in_acc && in_acc != out) return 1;  // fused contract: add in place over the residual
    if (batch_size == 1) {
        sofuu_gemv_q6_k(weight_bytes, x, out, rows, cols);
        return 1;  // batch-1 callers fall back to the unfused matvec path
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
            const size_t g4n = tile_batch / 4;
            tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>(g4n * nb);
            const float *gr = x;
            if (tile_batch != batch_size) {
                float *p = tile_scratch_pad<float>(tile_batch * cols);
                std::memset(p, 0, tile_batch * cols * sizeof(float));
                std::memcpy(p, x, batch_size * cols * sizeof(float));
                gr = p;
            }
            // Act-quant on the pool: the scalar generic quantizer was the
            // single-threaded serial prologue (~22-37% of the q4_K path at
            // batch 64 on Pi 5).
            if (g4n >= 2) {
                g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
                    for (size_t g4 = a; g4 < b; ++g4) {
                        ggml_quantize_mat_q8_K_4x4_neon(gr + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
                    }
                });
            } else {
                ggml_quantize_mat_q8_K_4x4_neon(gr, qa_ptr, cols);
            }
            float *out_dst = out;
            if (tile_batch != batch_size) out_dst = tile_scratch<float>(tile_batch * rows);
            const float * acc_k = (in_acc && tile_batch == batch_size) ? out : nullptr;
            g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemm_q6_K_8x4_q8_K(cols, out_dst + g * 8, rows,
                        tile + g * nb * sizeof(tile_block_q6_Kx8), qa_ptr, tile_batch, 8, acc_k);
                }
            });
            if (tile_batch != batch_size) {
                if (in_acc) {
                    sofuu_f32_add_inplace(out, out_dst, batch_size * rows);  // out still holds the residual
                } else {
                    std::memcpy(out, out_dst, batch_size * rows * sizeof(float));
                }
            }
            return 0;
        }
    }
#endif

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    // Prefill via int8 SDOT (see gemm_q8_0 note): cols >= 2048 only.
    if (in_acc) return 1;
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
        return 0;
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
    return 0;
}

void sofuu_gemm_q6_k(const void *weight_bytes, const float *x, float *out, size_t rows, size_t cols, size_t batch_size) {
    sofuu_gemm_q6_k_impl(weight_bytes, x, out, rows, cols, batch_size, nullptr);
}

extern "C" int sofuu_gemm_q6_k_acc(const void *weight_bytes, const float *x, float *out_acc, size_t rows, size_t cols, size_t batch_size) {
    return sofuu_gemm_q6_k_impl(weight_bytes, x, out_acc, rows, cols, batch_size, out_acc);
}

// ── Dual-output GEMM (gate & up share the activation-quant prologue) ──
// Quantize x once, then run both weight tiles back-to-back per row group:
// one act-quant pass and one pool dispatch instead of two full GEMM calls
// (the q8 activations for the second weight come from L2/hot registers).
// Integer math per output is identical to the separate calls, so out1/out2
// are bit-equal to two sofuu_gemm_* runs on the same inputs. Returns 1 when
// the fused tile path was not taken (out1/out2 UNTOUCHED).
extern "C" int sofuu_gemm_q4_k_pair(const void *w1, const void *w2, const float *x, float *out1, float *out2, size_t rows, size_t cols, size_t batch_size) {
#if defined(__aarch64__)
    if (rows == 0 || cols == 0 || batch_size < 4 || batch_size % 4 != 0 || rows % 8 != 0 || cols % 256 != 0 || !SOFUU_TILE_TYPES_OK(12)) return 1;
    const uint8_t *t1 = sofuu_tile_lookup(w1, 12, rows, cols);
    const uint8_t *t2 = sofuu_tile_lookup(w2, 12, rows, cols);
    if (!t1 || !t2) return 1;
    const size_t nb = cols / 256;
    tile_block_q8_Kx4 *qa_ptr = tile_scratch<tile_block_q8_Kx4>((batch_size / 4) * nb);
    if (batch_size >= 8) {
        const size_t g4n = batch_size / 4;
        g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
            for (size_t g4 = a; g4 < b; ++g4) {
                ggml_quantize_mat_q8_K_4x4_neon(x + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
        });
    } else {
        for (size_t g4 = 0; g4 < (size_t)batch_size / 4; ++g4) {
            ggml_quantize_mat_q8_K_4x4_neon(x + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
        }
    }
    g_pool.parallel_for(0, rows / 8, [=](size_t g0, size_t g1) {
        for (size_t g = g0; g < g1; ++g) {
            ggml_gemm_q4_K_8x4_q8_K(cols, out1 + g * 8, rows,
                t1 + g * nb * sizeof(tile_block_q4_Kx8), qa_ptr, batch_size, 8, nullptr);
            ggml_gemm_q4_K_8x4_q8_K(cols, out2 + g * 8, rows,
                t2 + g * nb * sizeof(tile_block_q4_Kx8), qa_ptr, batch_size, 8, nullptr);
        }
    });
    return 0;
#else
    (void) w1; (void) w2; (void) x; (void) out1; (void) out2; (void) rows; (void) cols; (void) batch_size;
    return 1;
#endif
}

extern "C" int sofuu_gemm_q8_0_pair(const void *w1, const void *w2, const float *x, float *out1, float *out2, size_t rows, size_t cols, size_t batch_size) {
#if defined(__aarch64__)
    if (rows == 0 || cols == 0 || batch_size < 4 || batch_size % 4 != 0 || rows % 4 != 0 || cols % 32 != 0 || !SOFUU_TILE_TYPES_OK(8)) return 1;
    const uint8_t *t1 = sofuu_tile_lookup(w1, 8, rows, cols);
    const uint8_t *t2 = sofuu_tile_lookup(w2, 8, rows, cols);
    if (!t1 || !t2) return 1;
    const size_t nb = cols / 32;
    tile_block_q8_0x4 *qa_ptr = tile_scratch<tile_block_q8_0x4>((batch_size / 4) * nb);
    if (batch_size >= 8) {
        const size_t g4n = batch_size / 4;
        g_pool.parallel_for(0, g4n, [=](size_t a, size_t b) {
            for (size_t g4 = a; g4 < b; ++g4) {
                ggml_quantize_mat_q8_0_4x4(x + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
            }
        });
    } else {
        for (size_t g4 = 0; g4 < (size_t)batch_size / 4; ++g4) {
            ggml_quantize_mat_q8_0_4x4(x + g4 * 4 * cols, qa_ptr + g4 * nb, cols);
        }
    }
    if (batch_size % 8 == 0) {
        // 8-act pair panel: both weight groups and both act groups in one
        // pass; fold order per output unchanged -> bit-identical.
        g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
            for (size_t g = g0; g < g1; ++g) {
                ggml_gemm_q8_0_8ax4_pair_q8_0(cols, out1 + g * 4, out2 + g * 4, rows,
                    t1 + g * nb * sizeof(tile_block_q8_0x4), t2 + g * nb * sizeof(tile_block_q8_0x4),
                    qa_ptr, batch_size, 4);
            }
        });
        return 0;
    }
    g_pool.parallel_for(0, rows / 4, [=](size_t g0, size_t g1) {
        for (size_t g = g0; g < g1; ++g) {
            ggml_gemm_q8_0_4x4_q8_0(cols, out1 + g * 4, rows,
                t1 + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, batch_size, 4, nullptr);
            ggml_gemm_q8_0_4x4_q8_0(cols, out2 + g * 4, rows,
                t2 + g * nb * sizeof(tile_block_q8_0x4), qa_ptr, batch_size, 4, nullptr);
        }
    });
    return 0;
#else
    (void) w1; (void) w2; (void) x; (void) out1; (void) out2; (void) rows; (void) cols; (void) batch_size;
    return 1;
#endif
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

#if defined(__aarch64__) && defined(SOFUU_NEON)
// Vector exp(x) for the softmax path: exp(x) = 2^(round(x·log2e) + frac) with
// the magic-number rounding and a two-part ln2, then a degree-5 Horner for
// e^frac on |frac| ≤ 0.3466. Max relative error vs libm expf measured ~1e-7;
// NOT bit-identical to the scalar expf, so parity vs the scalar reference is
// asserted at rel ≤ 5e-7 in sofuu_test_softmax_vec_parity (greedy decode is
// orders of magnitude inside its argmax margins everywhere it matters).
static inline float32x4_t sofuu_vexpq_f32(float32x4_t x) {
    const float32x4_t LOG2E  = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t LN2_HI = vdupq_n_f32(0.693145751953125f);      // ln2 high
    const float32x4_t LN2_LO = vdupq_n_f32(1.4286067653301870e-06f); // ln2 low
    const float32x4_t MAGIC  = vdupq_n_f32(12582912.0f);
    const int32x4_t   LOCLAMP  = vdupq_n_s32(-126);
    float32x4_t z  = vmulq_f32(x, LOG2E);
    float32x4_t kf = vsubq_f32(vaddq_f32(z, MAGIC), MAGIC); // round-half-even by magic add
    int32x4_t   ki = vcvtq_s32_f32(kf);
    // Matches scalar f32 expf, which underflows to exact 0 once the true
    // exponent drops below 2^-149 (x·log2e < -150); the mask also discards
    // poly residue from absurd inputs.
    uint32x4_t dead = vcltq_s32(ki, vdupq_n_s32(-150));
    ki = vmaxq_s32(ki, LOCLAMP); // guard the 2^ki exponent field
    float32x4_t f = vmlsq_f32(x, vcvtq_f32_s32(ki), LN2_HI);
    f = vmlsq_f32(f, vcvtq_f32_s32(ki), LN2_LO);
    // e^f on |f| <= ln2/2: Horner for 1 + f + f^2/2 + f^3/6 + f^4/24 + f^5/120.
    float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), f, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), f, p);
    p = vfmaq_f32(vdupq_n_f32(0.5f), f, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f), f, p);
    p = vfmaq_f32(vdupq_n_f32(1.0f), f, p);
    int32x4_t e = vaddq_s32(ki, vdupq_n_s32(127)); // f32 exponent field bias
    float32x4_t two_ki = vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
    float32x4_t res = vmulq_f32(p, two_ki);
    return vbslq_f32(dead, vdupq_n_f32(0.0f), res);
}

void sofuu_softmax_vec(float *scores, size_t n) {
    // max: vector scan
    float32x4_t vmax = vdupq_n_f32(scores[0]);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vmax = vmaxq_f32(vmax, vld1q_f32(scores + i));
        vmax = vmaxq_f32(vmax, vld1q_f32(scores + i + 4));
    }
    for (; i + 4 <= n; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(scores + i));
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; ++i) if (scores[i] > max_val) max_val = scores[i];

    const float32x4_t vm = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(scores + i), vm);
        float32x4_t e = sofuu_vexpq_f32(v);
        vst1q_f32(scores + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) {
        float exp_v = expf(scores[i] - max_val);
        scores[i] = exp_v;
        sum += exp_v;
    }
    const float inv_sum_s = 1.0f / (sum > 0.0f ? sum : 1e-8f);
    const float32x4_t vinv = vdupq_n_f32(inv_sum_s);
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        vst1q_f32(scores + j, vmulq_f32(vld1q_f32(scores + j), vinv));
    }
    for (; j < n; ++j) scores[j] *= inv_sum_s;
}
#endif

void sofuu_softmax(float *scores, size_t n) {
    if (n == 0) return;
#if defined(__aarch64__) && defined(SOFUU_NEON)
    sofuu_softmax_vec(scores, n);
#else
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
#endif
}

// ── Rotary Position Embedding (RoPE - NEOX / Qwen / LLaMA style) ─────
// RoPE recomputed 1/powf(freq_base, 2i/hd) + cosf + sinf per head per element
// — ~10M powf + 20M sincos for a 234-token Qwen prefill (~27% of prefill wall
// inside the profiler's "other" bucket). The table caches freq per
// (half_dim, freq_base) and cos/sin per (pos, i), computed with the exact
// same libm calls on the exact same float arguments ((float)pos * freq), so
// rotation results — and the model's logits — are bit-identical. Tables are
// shared across q/k heads and layers; rows fill lazily and never move, and a
// grow copies the valid prefix into a fresh larger table (kept alive in the
// list), so concurrent readers stay safe.
struct sofuu_rope_table {
    size_t half_dim = 0;
    float  freq_base = 0.0f;
    uint32_t fb_bits = 0;
    std::vector<float> freqs;   // half_dim
    std::vector<float> coss;    // cap * half_dim, rows [0, filled) valid
    std::vector<float> sins;
    size_t cap = 0;
    size_t filled = 0;
};
static std::mutex g_rope_tables_mutex;
// Exactly one LIVE table per (half_dim, freq_base) key. Retired tables are
// parked (never freed) for readers that still hold their row pointers.
static std::vector<std::shared_ptr<sofuu_rope_table>> g_rope_live;
static std::vector<std::shared_ptr<sofuu_rope_table>> g_rope_retired;

static uint32_t sofuu_float_bits(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

static void sofuu_rope_fill_rows(sofuu_rope_table *t, size_t from, size_t to) {
    const size_t hd = t->half_dim;
    for (size_t pos = from; pos < to; ++pos) {
        float *c = t->coss.data() + pos * hd;
        float *s = t->sins.data() + pos * hd;
        for (size_t i = 0; i < hd; ++i) {
            const float theta = (float)pos * t->freqs[i];
            c[i] = cosf(theta);
            s[i] = sinf(theta);
        }
    }
    t->filled = to;
}

// Returns cos/sin rows for `pos`; fills the table up to pos+margin as needed.
// Returned pointers stay valid for the duration of the call.
static void sofuu_rope_lookup(size_t half_dim, float freq_base, size_t pos,
                              const float ** cos_row, const float ** sin_row) {
    const bool rope_debug = getenv("SOFUU_ROPE_DEBUG") != nullptr;
    static std::atomic<uint64_t> dbg_calls{0};
    std::lock_guard<std::mutex> lk(g_rope_tables_mutex);
    const uint32_t fb = sofuu_float_bits(freq_base);
    size_t live_idx = (size_t)-1;
    for (size_t i = 0; i < g_rope_live.size(); ++i) {
        if (g_rope_live[i]->half_dim == half_dim && g_rope_live[i]->fb_bits == fb) {
            live_idx = i;
            break;
        }
    }
    if (live_idx == (size_t)-1) {
        auto t = std::make_shared<sofuu_rope_table>();
        t->half_dim = half_dim;
        t->freq_base = freq_base;
        t->fb_bits = fb;
        t->freqs.resize(half_dim);
        for (size_t i = 0; i < half_dim; ++i) {
            t->freqs[i] = 1.0f / powf(freq_base, (float)(2 * i) / (float)(half_dim * 2));
        }
        t->cap = pos + pos / 2 + 1024;
        t->coss.resize(t->cap * half_dim);
        t->sins.resize(t->cap * half_dim);
        live_idx = g_rope_live.size();
        g_rope_live.push_back(std::move(t));
    }
    sofuu_rope_table * fit = g_rope_live[live_idx].get();
    size_t grew = 0;
    if (pos >= fit->cap) {
        // Grow: copy the valid prefix into a fresh, larger table. The old one
        // is parked (kept alive) for any reader still holding its rows.
        auto t = std::make_shared<sofuu_rope_table>();
        t->half_dim = fit->half_dim;
        t->freq_base = fit->freq_base;
        t->fb_bits = fit->fb_bits;
        t->freqs = fit->freqs;
        t->cap = pos + pos / 2 + 1024;
        t->coss.resize(t->cap * half_dim);
        t->sins.resize(t->cap * half_dim);
        if (fit->filled > 0) {
            memcpy(t->coss.data(), fit->coss.data(), fit->filled * half_dim * sizeof(float));
            memcpy(t->sins.data(), fit->sins.data(), fit->filled * half_dim * sizeof(float));
            t->filled = fit->filled;
        }
        g_rope_retired.push_back(std::move(g_rope_live[live_idx]));
        fit = t.get();
        g_rope_live[live_idx] = std::move(t);
        grew = 1;
    }
    const size_t want = pos + 1;
    size_t did_fill = 0;
    if (want > fit->filled) {
        const size_t to = std::min(fit->cap, want + 256);
        const size_t from = fit->filled;
        sofuu_rope_fill_rows(fit, from, to);
        did_fill = to - from;
    }
    if (rope_debug) {
        const uint64_t n = dbg_calls.fetch_add(1, std::memory_order_relaxed);
        if (n < 40 || n % 20000 == 0) {
            fprintf(stderr, "[rope-dbg] call %llu: pos=%zu hd=%zu live=%zu cap=%zu filled=%zu grew=%zu fill=%zu\n",
                    (unsigned long long)n, pos, half_dim, g_rope_live.size(), fit->cap, fit->filled, grew, did_fill);
        }
    }
    *cos_row = fit->coss.data() + pos * half_dim;
    *sin_row = fit->sins.data() + pos * half_dim;
}

void sofuu_rope(float *vec, size_t n_heads, size_t head_dim, size_t pos, float freq_base, int rope_style) {
    if (head_dim < 2) return;
    size_t half_dim = head_dim / 2;
    static const bool no_rope_table = getenv("SOFUU_NO_ROPE_TABLE") != nullptr;
    const float *cos_row = nullptr;
    const float *sin_row = nullptr;
    if (!no_rope_table) {
        sofuu_rope_lookup(half_dim, freq_base, pos, &cos_row, &sin_row);
    }
    for (size_t h = 0; h < n_heads; ++h) {
        float *head = vec + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            float c;
            float s;
            if (cos_row) {
                c = cos_row[i];
                s = sin_row[i];
            } else {
                float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)head_dim);
                float theta = (float)pos * freq;
                c = cosf(theta);
                s = sinf(theta);
            }

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

// ── Attention QK/V inner loops, 4 positions per iteration ────────────
// Prefill attention was per-position single dots/mads; batching 4 positions
// keeps every per-position reduction chain identical to
// sofuu_vec_dot_f32/sofuu_vec_mad_f32 (4 lanes × 16-float strides, in-order
// position application), so results are bit-identical — the win is ILP (16
// independent FMA chains) and one shared Q load / dst round-trip per strip.
// K rows for a position range are contiguous (p·head_dim); `scores_out` are
// dense over the range. SWA callers use two ranges (sink, window).
static void sofuu_attn_qk_range(const float * __restrict q, const float * __restrict k_base,
                                size_t k_stride, size_t head_dim, size_t p_begin, size_t count,
                                float scale, float * __restrict scores_out) {
#if defined(SOFUU_NEON)
    // 2 positions × 4 lanes: same per-position reduction as sofuu_vec_dot_f32,
    // 16-float strips shared by both K rows. 8 live accumulators avoids the
    // register spilling that made a 4-position variant slower on A76.
    size_t p = 0;
    for (; p + 2 <= count; p += 2) {
        const float *k0 = k_base + (p_begin + p) * k_stride;
        const float *k1 = k_base + (p_begin + p + 1) * k_stride;
        float32x4_t a0s0 = vdupq_n_f32(0.0f), a0s1 = vdupq_n_f32(0.0f);
        float32x4_t a0s2 = vdupq_n_f32(0.0f), a0s3 = vdupq_n_f32(0.0f);
        float32x4_t a1s0 = vdupq_n_f32(0.0f), a1s1 = vdupq_n_f32(0.0f);
        float32x4_t a1s2 = vdupq_n_f32(0.0f), a1s3 = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 16 <= head_dim; i += 16) {
            const float32x4_t qv0 = vld1q_f32(q + i);
            const float32x4_t qv1 = vld1q_f32(q + i + 4);
            const float32x4_t qv2 = vld1q_f32(q + i + 8);
            const float32x4_t qv3 = vld1q_f32(q + i + 12);
            a0s0 = vfmaq_f32(a0s0, qv0, vld1q_f32(k0 + i));
            a0s1 = vfmaq_f32(a0s1, qv1, vld1q_f32(k0 + i + 4));
            a0s2 = vfmaq_f32(a0s2, qv2, vld1q_f32(k0 + i + 8));
            a0s3 = vfmaq_f32(a0s3, qv3, vld1q_f32(k0 + i + 12));
            a1s0 = vfmaq_f32(a1s0, qv0, vld1q_f32(k1 + i));
            a1s1 = vfmaq_f32(a1s1, qv1, vld1q_f32(k1 + i + 4));
            a1s2 = vfmaq_f32(a1s2, qv2, vld1q_f32(k1 + i + 8));
            a1s3 = vfmaq_f32(a1s3, qv3, vld1q_f32(k1 + i + 12));
        }
        // Exactly the single-dot tail sequence per position: combine the four
        // lanes, continue the 4-stride loop into the combined vector, vaddvq,
        // then scalar tail.
        float32x4_t c0 = vaddq_f32(vaddq_f32(a0s0, a0s1), vaddq_f32(a0s2, a0s3));
        float32x4_t c1 = vaddq_f32(vaddq_f32(a1s0, a1s1), vaddq_f32(a1s2, a1s3));
        size_t j = i;
        for (; j + 4 <= head_dim; j += 4) {
            const float32x4_t qv = vld1q_f32(q + j);
            c0 = vfmaq_f32(c0, qv, vld1q_f32(k0 + j));
            c1 = vfmaq_f32(c1, qv, vld1q_f32(k1 + j));
        }
        float t0 = vaddvq_f32(c0);
        float t1 = vaddvq_f32(c1);
        for (; j < head_dim; ++j) {
            t0 += q[j] * k0[j];
            t1 += q[j] * k1[j];
        }
        scores_out[p]     = t0 * scale;
        scores_out[p + 1] = t1 * scale;
    }
    for (; p < count; ++p) {
        scores_out[p] = sofuu_vec_dot_f32(q, k_base + (p_begin + p) * k_stride, head_dim) * scale;
    }
#else
    for (size_t p = 0; p < count; ++p) {
        scores_out[p] = sofuu_vec_dot_f32(q, k_base + (p_begin + p) * k_stride, head_dim) * scale;
    }
#endif
}

// 8-q-row batched QK: same per-dot arithmetic order as sofuu_attn_qk_range
// (4-lane strips, combined tail, vaddvq, scalar tail) but the two K strips
// are loaded once for EIGHT q rows — the plain prefill path was re-reading
// every K row once per batch row (64x). counts[] are the per-row causal seq
// bounds (nondecreasing with row index); dots beyond a row's count are
// stored nowhere. Bit-exact per dot.
static void sofuu_attn_qk_rows8(const float * __restrict q_batch, size_t q_stride,
                                const float * __restrict k_base, size_t k_stride,
                                size_t head_dim, size_t p_begin, const size_t *counts,
                                float * __restrict const * __restrict outs,
                                float scale, size_t nrows) {
#if defined(__aarch64__) && defined(SOFUU_NEON)
    if (nrows == 0) return;
    size_t max_count = counts[nrows - 1];
    for (size_t p = 0; p + 2 <= max_count; p += 2) {
        const float *k0 = k_base + (p_begin + p) * k_stride;
        const float *k1 = k_base + (p_begin + p + 1) * k_stride;
        for (size_t r = 0; r < nrows; ++r) {
            const float *q = q_batch + r * q_stride;
            float32x4_t a0s0 = vdupq_n_f32(0.0f), a0s1 = vdupq_n_f32(0.0f);
            float32x4_t a0s2 = vdupq_n_f32(0.0f), a0s3 = vdupq_n_f32(0.0f);
            float32x4_t a1s0 = vdupq_n_f32(0.0f), a1s1 = vdupq_n_f32(0.0f);
            float32x4_t a1s2 = vdupq_n_f32(0.0f), a1s3 = vdupq_n_f32(0.0f);
            size_t i = 0;
            for (; i + 16 <= head_dim; i += 16) {
                const float32x4_t qv0 = vld1q_f32(q + i);
                const float32x4_t qv1 = vld1q_f32(q + i + 4);
                const float32x4_t qv2 = vld1q_f32(q + i + 8);
                const float32x4_t qv3 = vld1q_f32(q + i + 12);
                a0s0 = vfmaq_f32(a0s0, qv0, vld1q_f32(k0 + i));
                a0s1 = vfmaq_f32(a0s1, qv1, vld1q_f32(k0 + i + 4));
                a0s2 = vfmaq_f32(a0s2, qv2, vld1q_f32(k0 + i + 8));
                a0s3 = vfmaq_f32(a0s3, qv3, vld1q_f32(k0 + i + 12));
                a1s0 = vfmaq_f32(a1s0, qv0, vld1q_f32(k1 + i));
                a1s1 = vfmaq_f32(a1s1, qv1, vld1q_f32(k1 + i + 4));
                a1s2 = vfmaq_f32(a1s2, qv2, vld1q_f32(k1 + i + 8));
                a1s3 = vfmaq_f32(a1s3, qv3, vld1q_f32(k1 + i + 12));
            }
            float32x4_t c0 = vaddq_f32(vaddq_f32(a0s0, a0s1), vaddq_f32(a0s2, a0s3));
            float32x4_t c1 = vaddq_f32(vaddq_f32(a1s0, a1s1), vaddq_f32(a1s2, a1s3));
            size_t j = i;
            for (; j + 4 <= head_dim; j += 4) {
                const float32x4_t qv = vld1q_f32(q + j);
                c0 = vfmaq_f32(c0, qv, vld1q_f32(k0 + j));
                c1 = vfmaq_f32(c1, qv, vld1q_f32(k1 + j));
            }
            float t0 = vaddvq_f32(c0);
            float t1 = vaddvq_f32(c1);
            for (; j < head_dim; ++j) {
                t0 += q[j] * k0[j];
                t1 += q[j] * k1[j];
            }
            const size_t p_abs = p_begin + p;
            if (p_abs < counts[r]) outs[r][p_abs] = t0 * scale;
            if (p_abs + 1 < counts[r]) outs[r][p_abs + 1] = t1 * scale;
        }
    }
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t p = (max_count - (max_count & 1)); p < counts[r]; ++p) {
            outs[r][p_begin + p] = sofuu_vec_dot_f32(q_batch + r * q_stride,
                                                     k_base + (p_begin + p) * k_stride, head_dim) * scale;
        }
    }
#else
    for (size_t r = 0; r < nrows; ++r) {
        const float *q = q_batch + r * q_stride;
        for (size_t p = 0; p < counts[r]; ++p) {
            outs[r][p] = sofuu_vec_dot_f32(q, k_base + (p_begin + p) * k_stride, head_dim) * scale;
        }
    }
#endif
}

static void sofuu_attn_vacc_range(float * __restrict out, const float * __restrict v_base,
                                  size_t v_stride, size_t head_dim, size_t p_begin, size_t count,
                                  const float * __restrict s) {
#if defined(SOFUU_NEON)
    size_t p = 0;
    for (; p + 4 <= count; p += 4) {
        const size_t i16 = head_dim - (head_dim & 15);
        for (size_t i = 0; i + 16 <= head_dim; i += 16) {
            float32x4_t d0 = vld1q_f32(out + i);
            float32x4_t d1 = vld1q_f32(out + i + 4);
            float32x4_t d2 = vld1q_f32(out + i + 8);
            float32x4_t d3 = vld1q_f32(out + i + 12);
            for (int a = 0; a < 4; ++a) {
                const float32x4_t sv   = vdupq_n_f32(s[p + a]);
                const float *     vrow = v_base + (p_begin + p + a) * v_stride;
                d0 = vmlaq_f32(d0, vld1q_f32(vrow + i), sv);
                d1 = vmlaq_f32(d1, vld1q_f32(vrow + i + 4), sv);
                d2 = vmlaq_f32(d2, vld1q_f32(vrow + i + 8), sv);
                d3 = vmlaq_f32(d3, vld1q_f32(vrow + i + 12), sv);
            }
            vst1q_f32(out + i, d0);
            vst1q_f32(out + i + 4, d1);
            vst1q_f32(out + i + 8, d2);
            vst1q_f32(out + i + 12, d3);
        }
        // Tail exactly like sofuu_vec_mad_f32 per position, positions in order.
        for (int a = 0; a < 4; ++a) {
            const float *     vrow = v_base + (p_begin + p + a) * v_stride;
            const float32x4_t sv   = vdupq_n_f32(s[p + a]);
            size_t it = i16;
            for (; it + 4 <= head_dim; it += 4) {
                float32x4_t d = vld1q_f32(out + it);
                d = vmlaq_f32(d, vld1q_f32(vrow + it), sv);
                vst1q_f32(out + it, d);
            }
            for (; it < head_dim; ++it) {
                out[it] += vrow[it] * s[p + a];
            }
        }
    }
    for (; p < count; ++p) {
        sofuu_vec_mad_f32(out, v_base + (p_begin + p) * v_stride, s[p], head_dim);
    }
#else
    for (size_t p = 0; p < count; ++p) {
        sofuu_vec_mad_f32(out, v_base + (p_begin + p) * v_stride, s[p], head_dim);
    }
#endif
}

// ── FA2-style block-tiled attention (online softmax) ─────────────────
// The previous per-slot path materialized all seq_len scores and made three
// scalar passes over them (dot / max+exp+sum / normalize) before a separate
// V-weighted-sum pass. The tiled path never materializes the full score
// vector: per 128-position K/V block it computes block scores (same x2
// lanes), one NEON block max, rescales the running max/sum/output once per
// block (FlashAttention-2's online-softmax identity), accumulates P·V into
// the output register set, and normalizes once at the end. Shared by every
// backend + Spark. Measured on Pi 5: LOSES to the two-pass everywhere (the
// scores vector never leaves L1 at our seq lengths, so FA2's
// don't-materialize-scores advantage is void and the block bookkeeping is
// pure overhead) — default OFF, `SOFUU_FA2=1` opts in (§Session 4 addendum 2).
static bool fa2_enabled() {
    static int v = getenv("SOFUU_FA2") ? strcmp(getenv("SOFUU_FA2"), "1") == 0 ? 1 : 0 : 0;
    return v != 0;
}

static inline float sofuu_f32_max_neon(const float * v, size_t n) {
#if defined(SOFUU_NEON)
    float32x4_t m = vdupq_n_f32(-3.4e38f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        m = vmaxq_f32(m, vmaxq_f32(vld1q_f32(v + i), vmaxq_f32(vld1q_f32(v + i + 4), vmaxq_f32(vld1q_f32(v + i + 8), vld1q_f32(v + i + 12)))));
    }
    float r = vmaxvq_f32(m);
    for (; i < n; ++i) r = r > v[i] ? r : v[i];
    return r;
#else
    float r = v[0];
    for (size_t i = 1; i < n; ++i) r = r > v[i] ? r : v[i];
    return r;
#endif
}

static inline float sofuu_f32_sum_neon(const float * v, size_t n) {
#if defined(SOFUU_NEON)
    float32x4_t s = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s = vaddq_f32(s, vaddq_f32(vld1q_f32(v + i), vaddq_f32(vaddq_f32(vld1q_f32(v + i + 4), vld1q_f32(v + i + 8)), vld1q_f32(v + i + 12))));
    }
    float r = vaddvq_f32(s);
    for (; i < n; ++i) r += v[i];
    return r;
#else
    float r = 0.0f;
    for (size_t i = 0; i < n; ++i) r += v[i];
    return r;
#endif
}

// One tiled online-softmax range over [p_begin, p_begin+count). Updates the
// running (m, l, acc_triuple) carried across ranges + blocks. exp: libm
// expf per position (block max keeps exponents ≤ 0 — no overflow, stable).
static void sofuu_attn_fa2_range(
    const float * __restrict q,
    const float * __restrict k_base,
    const float * __restrict v_base,
    size_t k_stride, size_t head_dim,
    size_t p_begin, size_t count,
    float scale,
    float * block_scores,   // ≥ 128 floats
    float * acc,            // head_dim accumulator
    float * m_run, float * l_run, int * have_max
) {
    size_t off = 0;
    while (off < count) {
        const size_t blk = (count - off) > 128 ? 128 : (count - off);
        sofuu_attn_qk_range(q, k_base, k_stride, head_dim, p_begin + off, blk, scale, block_scores);
        const float bm = sofuu_f32_max_neon(block_scores, blk);
        if (!*have_max) {
            *m_run = bm;
            *have_max = 1;
        }
        const float m_new = bm > *m_run ? bm : *m_run;
        const float rf = expf(*m_run - m_new); // 1.0 when m unchanged or first block (m==bm)
        for (size_t j = 0; j < blk; ++j) {
            block_scores[j] = expf(block_scores[j] - m_new);
        }
        // rescale running state
        if (rf != 1.0f) {
#if defined(SOFUU_NEON)
            const float32x4_t rf4 = vdupq_n_f32(rf);
            for (size_t i = 0; i + 16 <= head_dim; i += 16) {
                vst1q_f32(acc + i,      vmulq_f32(vld1q_f32(acc + i),      rf4));
                vst1q_f32(acc + i + 4,  vmulq_f32(vld1q_f32(acc + i + 4),  rf4));
                vst1q_f32(acc + i + 8,  vmulq_f32(vld1q_f32(acc + i + 8),  rf4));
                vst1q_f32(acc + i + 12, vmulq_f32(vld1q_f32(acc + i + 12), rf4));
            }
            for (size_t i = head_dim - (head_dim & 15); i < head_dim; ++i) acc[i] *= rf;
#else
            for (size_t i = 0; i < head_dim; ++i) acc[i] *= rf;
#endif
            *l_run *= rf;
        }
        *l_run += sofuu_f32_sum_neon(block_scores, blk);
#if defined(SOFUU_NEON)
        // P·V accumulate in place (position-ordered chunked FMAs, same op as
        // sofuu_attn_vacc_range but on the running acc — exp weights, unnormalized)
        const size_t i16 = head_dim - (head_dim & 15);
        for (size_t i = 0; i < i16; i += 16) {
            float32x4_t d0 = vld1q_f32(acc + i), d1 = vld1q_f32(acc + i + 4);
            float32x4_t d2 = vld1q_f32(acc + i + 8), d3 = vld1q_f32(acc + i + 12);
            for (size_t b = 0; b < blk; ++b) {
                const float32x4_t sv = vdupq_n_f32(block_scores[b]);
                const float * vrow = v_base + (p_begin + off + b) * k_stride;
                d0 = vmlaq_f32(d0, vld1q_f32(vrow + i), sv);
                d1 = vmlaq_f32(d1, vld1q_f32(vrow + i + 4), sv);
                d2 = vmlaq_f32(d2, vld1q_f32(vrow + i + 8), sv);
                d3 = vmlaq_f32(d3, vld1q_f32(vrow + i + 12), sv);
            }
            vst1q_f32(acc + i, d0); vst1q_f32(acc + i + 4, d1);
            vst1q_f32(acc + i + 8, d2); vst1q_f32(acc + i + 12, d3);
        }
        for (int a = 0; a < (int)blk; ++a) { // tail floats, position-ordered
            const float * vrow = v_base + (p_begin + off + a) * k_stride;
            for (size_t i = i16; i < head_dim; ++i) {
                acc[i] += vrow[i] * block_scores[a];
            }
        }
#else
        for (size_t b = 0; b < blk; ++b) {
            sofuu_vec_mad_f32(acc, v_base + (p_begin + off + b) * k_stride, block_scores[b], head_dim);
        }
#endif
        *m_run = m_new;
        off += blk;
    }
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
    // Plain-path rows go 8 batch rows per item so sofuu_attn_qk_rows8 can
    // share the two K strips across them; SWA/window and FA2 keep the
    // (row, head) domain (their per-row score ranges differ structurally).
    const bool use_rows8 = sliding_window == 0 && !(fa2_enabled() && head_dim <= 4096);
    size_t total_work = use_rows8 ? ((b_size + 7) / 8) * n_heads : b_size * n_heads;

    auto compute_head_slice = [=](size_t start, size_t end) {
        if (use_rows8) {
            alignas(32) float gscore_stack[8][4096];
            std::vector<float> heap_scores;
            float *gscore[8];
            for (size_t r = 0; r < 8; ++r) gscore[r] = gscore_stack[r];
            if (max_seq_len > 4096) {
                heap_scores.resize(8 * max_seq_len);
                for (size_t r = 0; r < 8; ++r) gscore[r] = heap_scores.data() + r * max_seq_len;
            }
            for (size_t work_idx = start; work_idx < end; ++work_idx) {
                const size_t grp = work_idx / n_heads;
                const size_t h   = work_idx % n_heads;
                const size_t b0  = grp * 8;
                const size_t b1  = (b0 + 8 < b_size) ? (b0 + 8) : b_size;
                const size_t nr  = b1 - b0;
                const size_t kv_h = h / n_rep;
                size_t head_base = (layer_idx * n_kv_heads + kv_h) * max_seq_len * head_dim;
                const float *k_past = k_cache + head_base;
                const float *v_past = v_cache + head_base;
                size_t counts[8];
                float *outs_p[8];
                for (size_t r = 0; r < nr; ++r) {
                    const size_t pos = start_pos + b0 + r;
                    counts[r] = (pos + 1 < max_seq_len) ? (pos + 1) : max_seq_len;
                    outs_p[r] = gscore[r];
                }
                sofuu_attn_qk_rows8(q_batch + b0 * q_dim + h * head_dim, q_dim, k_past,
                                    head_dim, head_dim, 0, counts, outs_p, inv_sqrt_head_dim, nr);
                for (size_t r = 0; r < nr; ++r) {
                    sofuu_softmax(outs_p[r], counts[r]);
                    float *out_head = head_out_batch + (b0 + r) * q_dim + h * head_dim;
                    std::memset(out_head, 0, head_dim * sizeof(float));
                    sofuu_attn_vacc_range(out_head, v_past, head_dim, head_dim, 0, counts[r], outs_p[r]);
                }
            }
            return;
        }
        alignas(32) float local_scores_stack[4096];
        std::vector<float> heap_scores;
        float *scores = local_scores_stack;
        if (max_seq_len > 4096) {
            heap_scores.resize(max_seq_len);
            scores = heap_scores.data();
        }
        const bool use_fa2 = fa2_enabled() && head_dim <= 4096;

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
                size_t active_count = sink_end + (seq_len - window_start);
                (void)active_count;

                if (use_fa2) {
                    // FA2 tile pass over the union softmax (sinks + window share
                    // one running max/sum), then one normalize.
                    alignas(32) float fa2_acc_stack[4096];
                    std::vector<float> fa2_acc_heap;
                    float *acc = fa2_acc_stack;
                    if (head_dim > 4096) {
                        fa2_acc_heap.resize(head_dim);
                        acc = fa2_acc_heap.data();
                    }
                    std::memset(acc, 0, head_dim * sizeof(float));
                    float m_run = 0.0f, l_run = 0.0f;
                    int have_max = 0;
                    sofuu_attn_fa2_range(q_head, k_past, v_past, head_dim, head_dim, 0, sink_end,
                                         inv_sqrt_head_dim, scores, acc, &m_run, &l_run, &have_max);
                    sofuu_attn_fa2_range(q_head, k_past, v_past, head_dim, head_dim, window_start,
                                         seq_len - window_start, inv_sqrt_head_dim, scores, acc, &m_run, &l_run, &have_max);
                    float *out_head = head_out_batch + b * q_dim + h * head_dim;
                    const float inv_l = 1.0f / (l_run > 0.0f ? l_run : 1e-8f);
                    for (size_t i = 0; i + 16 <= head_dim; i += 16) {
#if defined(SOFUU_NEON)
                        vst1q_f32(out_head + i,      vmulq_f32(vld1q_f32(acc + i),      vdupq_n_f32(inv_l)));
                        vst1q_f32(out_head + i + 4,  vmulq_f32(vld1q_f32(acc + i + 4),  vdupq_n_f32(inv_l)));
                        vst1q_f32(out_head + i + 8,  vmulq_f32(vld1q_f32(acc + i + 8),  vdupq_n_f32(inv_l)));
                        vst1q_f32(out_head + i + 12, vmulq_f32(vld1q_f32(acc + i + 12), vdupq_n_f32(inv_l)));
#else
                        for (size_t k = 0; k < 16; ++k) out_head[i + k] = acc[i + k] * inv_l;
#endif
                    }
                    for (size_t i = head_dim - (head_dim & 15); i < head_dim; ++i) out_head[i] = acc[i] * inv_l;
                } else {
                sofuu_attn_qk_range(q_head, k_past, head_dim, head_dim, 0, sink_end, inv_sqrt_head_dim, scores);
                sofuu_attn_qk_range(q_head, k_past, head_dim, head_dim, window_start, seq_len - window_start,
                                    inv_sqrt_head_dim, scores + sink_end);

                sofuu_softmax(scores, active_count);

                float *out_head = head_out_batch + b * q_dim + h * head_dim;
                std::memset(out_head, 0, head_dim * sizeof(float));

                sofuu_attn_vacc_range(out_head, v_past, head_dim, head_dim, 0, sink_end, scores);
                sofuu_attn_vacc_range(out_head, v_past, head_dim, head_dim, window_start, seq_len - window_start,
                                      scores + sink_end);
                }
            } else if (use_fa2) {
                alignas(32) float fa2_acc_stack[4096];
                std::vector<float> fa2_acc_heap;
                float *acc = fa2_acc_stack;
                if (head_dim > 4096) {
                    fa2_acc_heap.resize(head_dim);
                    acc = fa2_acc_heap.data();
                }
                std::memset(acc, 0, head_dim * sizeof(float));
                float m_run = 0.0f, l_run = 0.0f;
                int have_max = 0;
                sofuu_attn_fa2_range(q_head, k_past, v_past, head_dim, head_dim, 0, seq_len,
                                     inv_sqrt_head_dim, scores, acc, &m_run, &l_run, &have_max);
                float *out_head = head_out_batch + b * q_dim + h * head_dim;
                const float inv_l = 1.0f / (l_run > 0.0f ? l_run : 1e-8f);
                for (size_t i = 0; i + 16 <= head_dim; i += 16) {
#if defined(SOFUU_NEON)
                    vst1q_f32(out_head + i,      vmulq_f32(vld1q_f32(acc + i),      vdupq_n_f32(inv_l)));
                    vst1q_f32(out_head + i + 4,  vmulq_f32(vld1q_f32(acc + i + 4),  vdupq_n_f32(inv_l)));
                    vst1q_f32(out_head + i + 8,  vmulq_f32(vld1q_f32(acc + i + 8),  vdupq_n_f32(inv_l)));
                    vst1q_f32(out_head + i + 12, vmulq_f32(vld1q_f32(acc + i + 12), vdupq_n_f32(inv_l)));
#else
                    for (size_t k = 0; k < 16; ++k) out_head[i + k] = acc[i + k] * inv_l;
#endif
                }
                for (size_t i = head_dim - (head_dim & 15); i < head_dim; ++i) out_head[i] = acc[i] * inv_l;
            } else {
                sofuu_attn_qk_range(q_head, k_past, head_dim, head_dim, 0, seq_len, inv_sqrt_head_dim, scores);

                sofuu_softmax(scores, seq_len);

                float *out_head = head_out_batch + b * q_dim + h * head_dim;
                std::memset(out_head, 0, head_dim * sizeof(float));

                sofuu_attn_vacc_range(out_head, v_past, head_dim, head_dim, 0, seq_len, scores);
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

// Test fixtures are random bytes, and the GGUF block headers (q4_K d/dmin,
// q8_0 d) are reinterpreted as f16 there — ~3% of random headers are NaN/inf,
// which turns every downstream bitwise comparison into NaN-payload matching
// (kernel-vs-kernel passes vacuously, kernel-vs-scalar-add fails on payload
// drift). Stamp small finite f16 scales per block instead; payload bytes stay
// random so the dot-product math is still exercised.
static void sanitize_block_scale_headers(uint8_t * w, size_t n_blk, size_t block_bytes, size_t hdr_bytes) {
    for (size_t i = 0; i < n_blk; ++i) {
        uint8_t *     h  = w + i * block_bytes;
        const float   k  = (float)(i % 16);
        uint16_t      db = ggml_cpu_fp32_to_fp16(0.5f + 0.015625f * k);
        memcpy(h, &db, 2);
        if (hdr_bytes >= 4) {
            uint16_t mb = ggml_cpu_fp32_to_fp16(0.0625f + 0.0078125f * k);
            memcpy(h + 2, &mb, 2);
        }
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
    sanitize_block_scale_headers(w.data(), (size_t)rows * (qk ? (size_t)(cols / 256) : (size_t)(cols / 32)), qk ? 144 : 34, qk ? 4 : 2);
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
                ggml_quantize_mat_q8_K_4x4_neon(x.data() + g4 * 4 * cols, qa + g4 * nb, cols);
            }
            t1 = std::chrono::steady_clock::now();
            t_quant += std::chrono::duration<double, std::milli>(t1 - t0).count();
            auto tk = std::chrono::steady_clock::now();
            const uint8_t *sc = nullptr;
            const uint8_t *ss = nullptr;
            const uint8_t *stile = sofuu_tile_lookup_s(w.data(), rows, cols, &ss);
            if (stile != nullptr && ss != nullptr) {
                g_pool.parallel_for(0, rows / 8, [&](size_t g0, size_t g1) {
                    for (size_t g = g0; g < g1; ++g) {
                        ggml_gemm_q4_K_8x4_q8_K_s(cols, out.data() + g * 8, rows,
                            stile + g * nb * sizeof(tile_block_q4_Kx8),
                            ss + g * nb * sizeof(tile_ss_q4_Kx8s), qa, batch, 8, nullptr);
                    }
                });
            } else if (const uint8_t *tile = sofuu_tile_lookup_sc(w.data(), 12, rows, cols, &sc)) {
                g_pool.parallel_for(0, rows / 8, [&](size_t g0, size_t g1) {
                    for (size_t g = g0; g < g1; ++g) {
                        const uint8_t *t = tile + g * nb * sizeof(tile_block_q4_Kx8);
                        if (sc) {
                            ggml_gemm_q4_K_8x4_q8_K_sc(cols, out.data() + g * 8, rows, t, qa, batch, 8,
                                sc + g * nb * sizeof(tile_sc_q4_Kx8), nullptr);
                        } else {
                            ggml_gemm_q4_K_8x4_q8_K(cols, out.data() + g * 8, rows, t, qa, batch, 8, nullptr);
                        }
                    }
                });
            }
            t_kernel += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tk).count();
        } else {
            const size_t nb = cols / 32;
            tile_block_q8_0x4 *qa = tile_scratch<tile_block_q8_0x4>((batch / 4) * nb);
            for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
                ggml_quantize_mat_q8_0_4x4(x.data() + g4 * 4 * cols, qa + g4 * nb, cols);
            }
            t1 = std::chrono::steady_clock::now();
            t_quant += std::chrono::duration<double, std::milli>(t1 - t0).count();
            auto tk = std::chrono::steady_clock::now();
            const uint8_t *tile = sofuu_tile_lookup(w.data(), 8, rows, cols);
            g_pool.parallel_for(0, rows / 4, [&](size_t g0, size_t g1) {
                for (size_t g = g0; g < g1; ++g) {
                    ggml_gemm_q8_0_4x4_q8_0(cols, out.data() + g * 4, rows,
                        tile + g * nb * sizeof(tile_block_q8_0x4), qa, batch, 4, nullptr);
                }
            });
            t_kernel += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tk).count();
        }
    }
    const double macs = (double)batch * rows * cols * iters;
    const double it = (double)iters;
    if (qk) {
        fprintf(stderr, "[bench] q4_K %dx%d batch=%d iters=%d: quant %.1fms (%.2fms/iter) | kernel %.1fms (%.2fms/iter) = %.1f GMAC/s | quant share %.1f%%\n",
                rows, cols, batch, iters, t_quant, t_quant / it, t_kernel, t_kernel / it,
                macs / (t_kernel * 1e6), 100.0 * t_quant / (t_quant + t_kernel));
    } else {
        fprintf(stderr, "[bench] q8_0 %dx%d batch=%d iters=%d: quant %.1fms (%.2fms/iter) | kernel %.1fms (%.2fms/iter) = %.1f GMAC/s | quant share %.1f%%\n",
                rows, cols, batch, iters, t_quant, t_quant / it, t_kernel, t_kernel / it,
                macs / (t_kernel * 1e6), 100.0 * t_quant / (t_quant + t_kernel));
    }
#else
    (void) type; (void) rows; (void) cols; (void) batch; (void) iters;
#endif
}

extern "C" int sofuu_test_quant_q8_K_parity(int cols, int iters) {
#if defined(__aarch64__)
    if (cols % 256 != 0 || iters < 1) return -1;
    const size_t qsz = sizeof(tile_block_q8_Kx4) * (cols / 256);
    std::vector<float> x(4 * (size_t)cols);
    std::vector<uint8_t> a(qsz), b(qsz);
    uint32_t rng = 987;
    for (int it = 0; it < iters; it++) {
        for (auto &v : x) {
            rng = rng * 1664525u + 1013904223u;
            v = ((int)(rng >> 12) % 4000 - 2000) / 500.0f;
            if ((rng & 7) == 0) v = 0.0f;                       // zeros
            if ((rng & 31) == 5) v = ((rng >> 5) & 1) ? 12.75f : -12.75f; // signed maxes
        }
        memset(a.data(), 0xAA, qsz); memset(b.data(), 0x55, qsz);
        ggml_quantize_mat_q8_K_4x4_generic(x.data(), a.data(), cols);
        ggml_quantize_mat_q8_K_4x4_neon(x.data(), b.data(), cols);
        if (memcmp(a.data(), b.data(), qsz) != 0) {
            for (size_t i = 0; i < qsz; i++) {
                if (a[i] != b[i]) { fprintf(stderr, "[parity] byte %zu: %02x vs %02x\n", i, a[i], b[i]); break; }
            }
            return 1;
        }
    }
    return 0;
#else
    (void) cols; (void) iters;
    return 0;
#endif
}

// ── R3 bench: q4_K nibble-plane kernel vs packed kernel, kernel-only on
// identical synthetic weights/activations. rows*cols must satisfy the auto
// plane gate (<= 16M elems) so warm builds the plane tile.
extern "C" void sofuu_bench_gemm_plane(int rows, int cols, int batch, int iters) {
#if defined(__aarch64__)
    if (batch < 4 || batch % 4 != 0 || iters < 1) return;
    const size_t row_size = (size_t)(cols / 256) * 144;
    std::vector<uint8_t> w((size_t)rows * row_size);
    std::vector<float> x((size_t)batch * cols);
    std::vector<float> out((size_t)batch * rows);
    uint32_t rng = 12345;
    for (auto &b : w) { rng = rng * 1664525u + 1013904223u; b = (uint8_t)(rng >> 24); }
    for (auto &v : x) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 16) % 2000 - 1000) / 700.0f; }
    sanitize_block_scale_headers(w.data(), (size_t)rows * (size_t)(cols / 256), 144, 4);
    if (sofuu_tile_warm(w.data(), rows, cols, 12) != 0) {
        fprintf(stderr, "[bench-p] tile warm failed\n");
        return;
    }
    const uint8_t *ptile = sofuu_tile_lookup_p(w.data(), rows, cols);
    const uint8_t *ktile = sofuu_tile_lookup(w.data(), 12, rows, cols);
    if (!ptile || !ktile) {
        fprintf(stderr, "[bench-p] missing tiles (plane gate: rows*cols <= 16M elems)\n");
        return;
    }
    const size_t nb = cols / 256;
    tile_block_q8_Kx4 *qa = tile_scratch<tile_block_q8_Kx4>((batch / 4) * nb);
    for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
        ggml_quantize_mat_q8_K_4x4_neon(x.data() + g4 * 4 * cols, qa + g4 * nb, cols);
    }
    double t_k = 0, t_p = 0;
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        g_pool.parallel_for(0, rows / 8, [&](size_t g0, size_t g1) {
            for (size_t g = g0; g < g1; ++g) {
                ggml_gemm_q4_K_8x4_q8_K(cols, out.data() + g * 8, rows,
                    ktile + g * nb * sizeof(tile_block_q4_Kx8), qa, batch, 8, nullptr);
            }
        });
        auto t1 = std::chrono::steady_clock::now();
        t_k += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        g_pool.parallel_for(0, rows / 8, [&](size_t g0, size_t g1) {
            for (size_t g = g0; g < g1; ++g) {
                ggml_gemm_q4_K_8x4_q8_K_p(cols, out.data() + g * 8, rows,
                    ptile + g * nb * sizeof(tile_block_q4_Kx8p), qa, batch, 8, nullptr);
            }
        });
        auto t1 = std::chrono::steady_clock::now();
        t_p += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double macs = (double)batch * rows * cols * iters;
    fprintf(stderr, "[bench-p] q4_K %dx%d batch=%d iters=%d: packed %.1f GMAC/s | planes %.1f GMAC/s (%+.1f%%)\n",
        rows, cols, batch, iters, macs / (t_k * 1e6), macs / (t_p * 1e6), 100.0 * (t_k - t_p) / t_k);
#else
    (void) rows; (void) cols; (void) batch; (void) iters;
#endif
}

// ── Parity test: the nibble-plane q4_K GEMM must be bit-identical to the
// packed kernel on deterministic synthetic weights + activations (a few
// activation rounds incl. zeros and mixed magnitudes).
// ── Parity: the T1 pre-decoded-scale GEMM (stock tile + ss region) must be
// bit-identical to the stock packed-tile kernel — the reorganization only
// replaces the 6-bit group unpack and moves the scale multiply to the f32
// pipe where every product is exact (see tile_ss_q4_Kx8s comment). ──
extern "C" int sofuu_test_q4_k_s_bit_exact(int rows, int cols, int batch) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    if (rows % 8 != 0 || cols % 256 != 0 || batch < 4 || batch % 4 != 0) return -1;
    const size_t nb = cols / 256;
    const size_t r8 = rows / 8;
    std::vector<uint8_t> wsrc((size_t)rows * nb * 144);
    std::vector<uint8_t> wpack(r8 * nb * sizeof(tile_block_q4_Kx8));
    std::vector<uint8_t> ws(r8 * nb * sizeof(tile_ss_q4_Kx8s));
    uint32_t rng = 90210;
    for (auto &b : wsrc) { rng = rng * 1664525u + 1013904223u; b = (uint8_t)(rng >> 24); }
    sanitize_block_scale_headers(wsrc.data(), (size_t)rows * nb, 144, 4);
    sofuu_repack_q4k_8bl(wpack.data(), wsrc.data(), rows, nb, 4);
    {
        const tile_block_q4_Kx8 * ins = (const tile_block_q4_Kx8 *)wpack.data();
        tile_ss_q4_Kx8s *         outs = (tile_ss_q4_Kx8s *)ws.data();
        for (size_t i = 0; i < (size_t)r8 * nb; ++i) {
            fill_q4k_s_block(&outs[i], &ins[i]);
        }
    }
    std::vector<float> x((size_t)batch * cols);
    for (int it = 0; it < 3; ++it) {
        for (auto &v : x) {
            rng = rng * 1664525u + 1013904223u;
            v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f;
            if ((rng & 7) == 0) v = 0.0f;
        }
        std::vector<tile_block_q8_Kx4> qa((batch / 4) * nb);
        for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
            ggml_quantize_mat_q8_K_4x4_neon(x.data() + g4 * 4 * cols, qa.data() + g4 * nb, cols);
        }
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<float> oa((size_t)batch * rows, -1e30f), os((size_t)batch * rows, -1e30f);
            for (size_t g = 0; g < r8; ++g) {
                ggml_gemm_q4_K_8x4_q8_K(cols, oa.data() + g * 8, rows,
                    wpack.data() + g * nb * sizeof(tile_block_q4_Kx8), qa.data(), batch, 8,
                    mode ? oa.data() : nullptr);
            }
            for (size_t g = 0; g < r8; ++g) {
                ggml_gemm_q4_K_8x4_q8_K_s(cols, os.data() + g * 8, rows,
                    wpack.data() + g * nb * sizeof(tile_block_q4_Kx8),
                    ws.data() + g * nb * sizeof(tile_ss_q4_Kx8s), qa.data(), batch, 8,
                    mode ? os.data() : nullptr);
            }
            if (memcmp(oa.data(), os.data(), oa.size() * sizeof(float)) != 0) {
                for (size_t i = 0; i < oa.size(); ++i) {
                    if (memcmp(&oa[i], &os[i], sizeof(float)) != 0) {
                        fprintf(stderr, "[parity-s gemm m%d] iter %d: out[%zu] %.9g vs %.9g\n",
                                mode, it, i, oa[i], os[i]);
                        return 1;
                    }
                }
                return 2;
            }
        }
    }
    return 0;
#else
    (void) rows; (void) cols; (void) batch;
    return 0;
#endif
}

extern "C" int sofuu_test_gemm_q4_k_plane_parity(int rows, int cols, int batch) {
#if defined(__aarch64__)
    if (rows % 8 != 0 || cols % 256 != 0 || batch < 4 || batch % 4 != 0) return -1;
    const size_t nb = cols / 256;
    const size_t r8 = rows / 8;
    std::vector<uint8_t> wsrc((size_t)rows * nb * 144);
    std::vector<uint8_t> wpack(r8 * nb * sizeof(tile_block_q4_Kx8));
    std::vector<uint8_t> wplane(r8 * nb * sizeof(tile_block_q4_Kx8p));
    uint32_t rng = 4242;
    for (auto &b : wsrc) { rng = rng * 1664525u + 1013904223u; b = (uint8_t)(rng >> 24); }
    sanitize_block_scale_headers(wsrc.data(), (size_t)rows * nb, 144, 4);
    sofuu_repack_q4k_8bl(wpack.data(), wsrc.data(), rows, nb, 4);
    sofuu_repack_q4k_8bl_planes(wplane.data(), wsrc.data(), rows, nb, 4);
    std::vector<float> x((size_t)batch * cols);
    for (int it = 0; it < 3; ++it) {
        for (auto &v : x) {
            rng = rng * 1664525u + 1013904223u;
            v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f;
            if ((rng & 7) == 0) v = 0.0f;
        }
        std::vector<tile_block_q8_Kx4> qa((batch / 4) * nb);
        for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
            ggml_quantize_mat_q8_K_4x4_neon(x.data() + g4 * 4 * cols, qa.data() + g4 * nb, cols);
        }
        std::vector<float> oa((size_t)batch * rows, -1e30f), ob((size_t)batch * rows, -1e30f);
        for (size_t g = 0; g < r8; ++g) {
            ggml_gemm_q4_K_8x4_q8_K(cols, oa.data() + g * 8, rows,
                wpack.data() + g * nb * sizeof(tile_block_q4_Kx8), qa.data(), batch, 8, nullptr);
        }
        for (size_t g = 0; g < r8; ++g) {
            ggml_gemm_q4_K_8x4_q8_K_p(cols, ob.data() + g * 8, rows,
                wplane.data() + g * nb * sizeof(tile_block_q4_Kx8p), qa.data(), batch, 8, nullptr);
        }
        if (memcmp(oa.data(), ob.data(), oa.size() * sizeof(float)) != 0) {
            for (size_t i = 0; i < oa.size(); ++i) {
                if (memcmp(&oa[i], &ob[i], sizeof(float)) != 0) {
                    fprintf(stderr, "[parity-p] iter %d: out[%zu] %.9g vs %.9g\n", it, i, oa[i], ob[i]);
                    return 1;
                }
            }
            return 2;
        }
    }
    return 0;
#else
    (void) rows; (void) cols; (void) batch;
    return 0;
#endif
}

// ── Parity: the x4-position attention inner loops must be bit-identical to
// the original per-position dot/mad/softmax loops. Deterministic fixtures,
// several (batch, start, window, sinks) configs; compares full head_out_batch.
static void sofuu_attn_ref_slot(const float *q_head, const float *k_past, const float *v_past,
                                float *out_head, float *scores, size_t seq_len, size_t head_dim,
                                float inv_sqrt_head_dim, size_t sliding_window, size_t sink_tokens) {
    if (sliding_window > 0 && seq_len > sliding_window) {
        size_t window_start = seq_len - sliding_window;
        size_t sink_end = (sink_tokens < window_start) ? sink_tokens : window_start;
        size_t active_count = 0;
        for (size_t p = 0; p < sink_end; ++p)
            scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
        for (size_t p = window_start; p < seq_len; ++p)
            scores[active_count++] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
        sofuu_softmax(scores, active_count);
        std::memset(out_head, 0, head_dim * sizeof(float));
        size_t score_idx = 0;
        for (size_t p = 0; p < sink_end; ++p)
            sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[score_idx++], head_dim);
        for (size_t p = window_start; p < seq_len; ++p)
            sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[score_idx++], head_dim);
    } else {
        for (size_t p = 0; p < seq_len; ++p)
            scores[p] = sofuu_vec_dot_f32(q_head, k_past + p * head_dim, head_dim) * inv_sqrt_head_dim;
        sofuu_softmax(scores, seq_len);
        std::memset(out_head, 0, head_dim * sizeof(float));
        for (size_t p = 0; p < seq_len; ++p)
            sofuu_vec_mad_f32(out_head, v_past + p * head_dim, scores[p], head_dim);
    }
}

// ── Parity: the RoPE sincos table must be bit-identical to the inline
// powf/cosf/sinf computation across positions, styles and freq bases.
extern "C" int sofuu_test_rope_table_parity(int n_heads, int head_dim, int max_pos) {
    if (n_heads < 1 || head_dim < 4 || max_pos < 4) return -1;
    std::vector<float> a((size_t)n_heads * head_dim), b((size_t)n_heads * head_dim);
    const float bases[3] = {10000.0f, 500000.0f, 27.5f};
    const int styles[2] = {0, 1};
    uint32_t rng = 606;
    for (int st = 0; st < 2; ++st) {
        for (int bi = 0; bi < 3; ++bi) {
            for (size_t i = 0; i < a.size(); ++i) {
                rng = rng * 1664525u + 1013904223u;
                a[i] = b[i] = ((int)(rng >> 8) % 2000 - 1000) / 128.0f;
            }
            for (int pos = 0; pos < max_pos; pos += 1) {
                sofuu_rope(a.data(), (size_t)n_heads, (size_t)head_dim, (size_t)pos, bases[bi], styles[st]);
                // reference: the original inline per-(head,i) computation
                const size_t half_dim = (size_t)head_dim / 2;
                for (size_t h = 0; h < (size_t)n_heads; ++h) {
                    float *head = b.data() + h * (size_t)head_dim;
                    for (size_t i = 0; i < half_dim; ++i) {
                        float freq = 1.0f / powf(bases[bi], (float)(2 * i) / (float)head_dim);
                        float theta = (float)pos * freq;
                        float c = cosf(theta);
                        float s = sinf(theta);
                        if (styles[st] == 1) {
                            size_t even = 2 * i, odd = even + 1;
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
                if (memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                    for (size_t i = 0; i < a.size(); ++i) {
                        if (a[i] != b[i]) {
                            fprintf(stderr, "[parity-rope] style=%d base=%g pos=%d: out[%zu] %.9g vs %.9g\n",
                                    styles[st], bases[bi], pos, i, a[i], b[i]);
                            return 1;
                        }
                    }
                    return 2;
                }
            }
        }
    }
    return 0;
}

// Vectorized-softmax parity gate: for sizes around the real shapes (score
// rows of 1..4096+), compare sofuu_softmax against a scalar expf reference
// accumulated in double. Gate = 2e-6 PEAK-relative: the poly exp itself is
// ~1e-7, the remainder is f32 sum reassociation across 200+ lanes (4-lane
// vector sums vs the reference) — still far inside what the downstream
// V-weighted sum and greedy argmax can observe.
extern "C" int sofuu_test_softmax_vec_parity(void) {
    const size_t sizes[] = {1, 2, 3, 4, 5, 7, 31, 64, 128, 234, 4096, 5000};
    uint32_t rng = 20261015u;
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        const size_t n = sizes[si];
        std::vector<float> x(n), y(n);
        for (size_t r = 0; r < 4; ++r) {
            for (auto &v : x) {
                rng = rng * 1664525u + 1013904223u;
                v = (float)((int64_t)(rng >> 8) % 2000001 - 1000000) / 10000.0f;
                if ((rng & 15) == 0) v = -0.0f;                 // exact ties / zeros
                if ((rng & 15) == 1) v = -87.3f;                // deep underflow term
                if ((rng & 15) == 2) v = -1e30f;                // exp → 0 exactly
            }
            y = x;
            sofuu_softmax(y.data(), n);
            // Reference in f32-expf semantics: exp(x-m) in float (underflow to
            // 0 exactly like any f32 softmax — a double exp would resolve
            // exp(-185) where both scalar and vector f32 softmax give 0),
            // accumulated in double to keep the reference sum clean.
            double m = x[0];
            for (size_t k = 1; k < n; ++k) m = m > x[k] ? m : x[k];
            double s2 = 0.0;
            for (size_t k = 0; k < n; ++k) s2 += (double)expf((float)(x[k] - m));
            double inv = 1.0 / s2;
            double worst = 0.0;
            double peak = 0.0;
            bool nan_seen = false;
            for (size_t k = 0; k < n; ++k) {
                double ref = (double)expf((float)x[k] - (float)m) * inv;
                peak = peak > ref ? peak : ref;
            }
            for (size_t k = 0; k < n; ++k) {
                double ref = (double)expf((float)x[k] - (float)m) * inv;
                double got = (double)y[k];
                if (!std::isfinite(got)) { nan_seen = true; break; }
                // Peak-relative: sub-1e-7-of-peak weights (real softmax
                // underflow) may collapse to 0 without affecting the
                // normalized V-weighted sum; per-element rel would flag
                // exactly those harmless underflows.
                double err = std::fabs(got - ref) / (peak > 0.0 ? peak : 1.0);
                if (err > worst) worst = err;
            }
            if (nan_seen) {
                fprintf(stderr, "[softmax-vec] n=%zu r=%zu NaN\n", n, r);
                for (size_t k = 0; k < n; ++k)
                    fprintf(stderr, "  k=%zu x=%.9g y=%.9g\n", k, (double)x[k], (double)y[k]);
                return 10 + (int)si;
            }
            if (worst > 2e-6) {
                fprintf(stderr, "[softmax-vec] n=%zu r=%zu worst rel-peak %.3g\n", n, r, worst);
                if (n <= 8) {
                    for (size_t k = 0; k < n; ++k)
                        fprintf(stderr, "  k=%zu x=%.9g y=%.9g\n", k, (double)x[k], (double)y[k]);
                }
                return 20 + (int)si;
            }
        }
    }
    return 0;
}

extern "C" int sofuu_test_attention_x4_parity(int n_heads, int n_kv_heads, int head_dim, int max_seq) {    if (n_heads < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0 || head_dim < 32 || max_seq < 8) return -1;
    const size_t H = (size_t)n_heads, KV = (size_t)n_kv_heads, HD = (size_t)head_dim;
    const size_t rep = H / KV, qd = H * HD, kvd = KV * HD, ms = (size_t)max_seq;
    std::vector<float> q(2 * qd), kc(ms * kvd), vc(ms * kvd), oref(2 * qd), onew(2 * qd);
    struct Cfg { size_t b, start, window, sinks; };
    const Cfg cfgs[] = {
        {2, 0, 0, 0},
        {2, ms - 3, 0, 0},
        {2, ms - 3, 8, 3},
        {2, ms - 7, 16, 2},
        {1, ms - 1, ms / 2 + 1, 4},  // sink_end clipped by window_start
        {2, 11, 5, 0},
        {2, 13, 0, 0},
    };
    uint32_t rng = 777;
    for (auto &v : q)  { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    for (auto &v : kc) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    for (auto &v : vc) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    std::vector<float> scores(ms + 16);
    const float inv_sqrt = 1.0f / sqrtf((float)HD);
    for (const Cfg &c : cfgs) {
        for (size_t i = 0; i < oref.size(); ++i) { oref[i] = -1e30f; onew[i] = -1e30f; }
        for (size_t b = 0; b < c.b; ++b) {
            const size_t pos = c.start + b;
            const size_t seq_len = (pos + 1 < ms) ? (pos + 1) : ms;
            for (size_t h = 0; h < H; ++h) {
                const size_t kv_h = h / rep;
                const float *q_head = q.data() + b * qd + h * HD;
                const float *kp = kc.data() + kv_h * ms * HD;
                const float *vp = vc.data() + kv_h * ms * HD;
                float *ro = oref.data() + b * qd + h * HD;
                float *no = onew.data() + b * qd + h * HD;
                if (c.window > 0 && seq_len > c.window) {
                    const size_t ws = seq_len - c.window;
                    const size_t se = (c.sinks < ws) ? c.sinks : ws;
                    const size_t ac = se + (seq_len - ws);
                    sofuu_attn_qk_range(q_head, kp, HD, HD, 0, se, inv_sqrt, scores.data());
                    sofuu_attn_qk_range(q_head, kp, HD, HD, ws, seq_len - ws, inv_sqrt, scores.data() + se);
                    sofuu_softmax(scores.data(), ac);
                    std::memset(no, 0, HD * sizeof(float));
                    sofuu_attn_vacc_range(no, vp, HD, HD, 0, se, scores.data());
                    sofuu_attn_vacc_range(no, vp, HD, HD, ws, seq_len - ws, scores.data() + se);
                } else {
                    sofuu_attn_qk_range(q_head, kp, HD, HD, 0, seq_len, inv_sqrt, scores.data());
                    sofuu_softmax(scores.data(), seq_len);
                    std::memset(no, 0, HD * sizeof(float));
                    sofuu_attn_vacc_range(no, vp, HD, HD, 0, seq_len, scores.data());
                }
                sofuu_attn_ref_slot(q_head, kp, vp, ro, scores.data(), seq_len, HD, inv_sqrt, c.window, c.sinks);
            }
        }
        if (memcmp(onew.data(), oref.data(), oref.size() * sizeof(float)) != 0) {
            for (size_t i = 0; i < oref.size(); ++i) {
                if (onew[i] != oref[i]) {
                    fprintf(stderr, "[parity-attn] b=%zu start=%zu win=%zu sinks=%zu: out[%zu] %.9g vs %.9g\n",
                            c.b, c.start, c.window, c.sinks, i, onew[i], oref[i]);
                    return 1;
                }
            }
            return 2;
        }
    }
    return 0;
}

// ── Parity: FA2 tiled online-softmax vs the two-pass reference. NOT bit-exact
// (block-wise max/exp reassociates the float rounding): gate is relative
// element error ≤ 1e-4 on the same SWA/plain config matrix as the x4 test. ──
extern "C" int sofuu_test_attention_fa2_parity(int n_heads, int n_kv_heads, int head_dim, int max_seq) {
    if (n_heads < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0 || head_dim < 32 || max_seq < 8) return -1;
    const size_t H = (size_t)n_heads, KV = (size_t)n_kv_heads, HD = (size_t)head_dim;
    const size_t rep = H / KV, qd = H * HD, kvd = KV * HD, ms = (size_t)max_seq;
    std::vector<float> q(2 * qd), kc(ms * kvd), vc(ms * kvd), oref(2 * qd), onew(2 * qd);
    struct Cfg { size_t b, start, window, sinks; };
    const Cfg cfgs[] = {
        {2, 0, 0, 0},
        {2, ms - 3, 0, 0},
        {2, ms - 3, 8, 3},
        {2, ms - 7, 16, 2},
        {1, ms - 1, ms / 2 + 1, 4},  // sink_end clipped by window_start
        {2, 11, 5, 0},
        {2, 13, 0, 0},
    };
    uint32_t rng = 777;
    for (auto &v : q)  { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    for (auto &v : kc) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    for (auto &v : vc) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) % 2000 - 1000) / 512.0f; }
    std::vector<float> scores(ms + 16), acc(HD, 0.0f);
    const float inv_sqrt = 1.0f / sqrtf((float)HD);
    const float tol = 1e-4f;
    for (const Cfg &c : cfgs) {
        for (size_t i = 0; i < oref.size(); ++i) { oref[i] = -1e30f; onew[i] = -1e30f; }
        for (size_t b = 0; b < c.b; ++b) {
            const size_t pos = c.start + b;
            const size_t seq_len = (pos + 1 < ms) ? (pos + 1) : ms;
            for (size_t h = 0; h < H; ++h) {
                const size_t kv_h = h / rep;
                const float *q_head = q.data() + b * qd + h * HD;
                const float *kp = kc.data() + kv_h * ms * HD;
                const float *vp = vc.data() + kv_h * ms * HD;
                float *ro = oref.data() + b * qd + h * HD;
                float *no = onew.data() + b * qd + h * HD;
                std::fill(acc.begin(), acc.end(), 0.0f);
                float m_run = 0.0f, l_run = 0.0f;
                int have_max = 0;
                if (c.window > 0 && seq_len > c.window) {
                    const size_t ws = seq_len - c.window;
                    const size_t se = (c.sinks < ws) ? c.sinks : ws;
                    sofuu_attn_fa2_range(q_head, kp, vp, HD, HD, 0, se, inv_sqrt, scores.data(),
                                         acc.data(), &m_run, &l_run, &have_max);
                    sofuu_attn_fa2_range(q_head, kp, vp, HD, HD, ws, seq_len - ws, inv_sqrt, scores.data(),
                                         acc.data(), &m_run, &l_run, &have_max);
                } else {
                    sofuu_attn_fa2_range(q_head, kp, vp, HD, HD, 0, seq_len, inv_sqrt, scores.data(),
                                         acc.data(), &m_run, &l_run, &have_max);
                }
                const float inv_l = 1.0f / (l_run > 0.0f ? l_run : 1e-8f);
                for (size_t i = 0; i < HD; ++i) no[i] = acc[i] * inv_l;
                sofuu_attn_ref_slot(q_head, kp, vp, ro, scores.data(), seq_len, HD, inv_sqrt, c.window, c.sinks);
                float peak = 0.0f;
                for (size_t i = 0; i < HD; ++i) peak = peak > fabsf(ro[i]) ? peak : fabsf(ro[i]);
                if (peak <= 0.0f) peak = 1.0f;
                for (size_t i = 0; i < HD; ++i) {
                    // Peak-relative: near-zero outputs (tiny attention weight)
                    // flag at per-element rel even when the absolute gap is
                    // far below the softmax tolerance — same gate semantics
                    // as the softmax vec test.
                    const float rel = fabsf(no[i] - ro[i]) / peak;
                    if (!(rel <= tol)) {
                        fprintf(stderr,
                                "[parity-fa2] b=%zu start=%zu win=%zu sinks=%zu: out[%zu] %.9g vs %.9g (rel %.3g)\n",
                                c.b, c.start, c.window, c.sinks, i, no[i], ro[i], rel);
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

// ── Parity: residual-fused and dual-pair GEMM paths must be bit-identical to
// the unfused equivalents on the same inputs (q4_K + q8_0). ──
extern "C" int sofuu_test_gemm_resid_pair_parity(int rows, int cols, int batch) {
    if (rows < 8 || rows % 8 != 0 || cols % 256 != 0 || batch < 4 || batch % 4 != 0) return -1;
    const size_t nb4 = cols / 256, nb8 = cols / 32;
    const size_t wrow4 = nb4 * 144, wrow8 = nb8 * sizeof(block_q8_0);
    std::vector<uint8_t> w4((size_t)rows * wrow4), w4b((size_t)rows * wrow4);
    std::vector<uint8_t> w8((size_t)rows * wrow8), w8b((size_t)rows * wrow8);
    std::vector<float> x((size_t)batch * cols), resid((size_t)batch * rows);
    uint32_t rng = 991;
    for (auto &b : w4)  { rng = rng * 1664525u + 1013904223u; b = (uint8_t)(rng >> 24); }
    for (auto &v : x)      { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f; }
    for (auto &v : resid)  { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f; }
    for (size_t i = 0; i < w4b.size(); ++i)  w4b[i]  = (uint8_t)(w4[i] ^ (uint8_t)(i * 31 + 7));
    sanitize_block_scale_headers(w4.data(), (size_t)rows * nb4, 144, 4);
    sanitize_block_scale_headers(w4b.data(), (size_t)rows * nb4, 144, 4);
    for (int rr = 0; rr < 2; ++rr) {
        sanitize_block_scale_headers(rr ? w8b.data() : w8.data(), (size_t)rows * nb8, sizeof(block_q8_0), 2);
    }
    if (sofuu_tile_warm(w4.data(), rows, cols, 12) != 0) return -2;
    if (sofuu_tile_warm(w4b.data(), rows, cols, 12) != 0) return -2;
    if (sofuu_tile_warm(w8.data(), rows, cols, 8) != 0) return -2;
    if (sofuu_tile_warm(w8b.data(), rows, cols, 8) != 0) return -2;
    std::vector<float> ref(batch * rows), aug(batch * rows), p1(batch * rows), p2(batch * rows);

    // q4_K residual: fused vs gemm + scalar in-place add
    sofuu_gemm_q4_k(w4.data(), x.data(), ref.data(), rows, cols, batch);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = resid[i] + ref[i];
    aug = resid;
    if (sofuu_gemm_q4_k_acc(w4.data(), x.data(), aug.data(), rows, cols, batch) != 0) return 1;
    if (memcmp(ref.data(), aug.data(), ref.size() * sizeof(float)) != 0) {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (memcmp(&ref[i], &aug[i], sizeof(float)) != 0) {
                fprintf(stderr, "[parity-resid] q4_K at %zu: ref %.9g aug %.9g bits %08x/%08x\n", i,
                        ref[i], aug[i], *(unsigned *)&ref[i], *(unsigned *)&aug[i]);
                break;
            }
        }
        // split the halves: plain dispatch gemm vs zero-base acc gemm
        std::vector<float> gp(batch * rows);
        sofuu_gemm_q4_k(w4.data(), x.data(), gp.data(), rows, cols, batch);
        std::vector<float> zb(batch * rows, 0.0f);
        if (sofuu_gemm_q4_k_acc(w4.data(), x.data(), zb.data(), rows, cols, batch) != 0) { return 17; }
        for (size_t i = 0; i < ref.size(); ++i) {
            if (memcmp(&gp[i], &zb[i], sizeof(float)) != 0) {
                fprintf(stderr, "[parity-resid] plain vs zero-acc at %zu: %.9g vs %.9g bits %08x/%08x\n", i, gp[i], zb[i],
                        *(unsigned *)&gp[i], *(unsigned *)&zb[i]);
                break;
            }
        }
        // dispatch determinism: same inputs twice
        std::vector<float> gp2(batch * rows);
        sofuu_gemm_q4_k(w4.data(), x.data(), gp2.data(), rows, cols, batch);
        {
            int d = memcmp(gp.data(), gp2.data(), gp.size() * sizeof(float));
            fprintf(stderr, "[diag] dispatch run1 vs run2 memcmp=%d\n", d);
            for (size_t i = 0; i < gp.size(); ++i) {
                if (memcmp(&gp[i], &gp2[i], sizeof(float)) != 0) {
                    fprintf(stderr, "[diag] run1 vs run2 first diff at %zu: %08x/%08x\n", i, *(unsigned *)&gp[i], *(unsigned *)&gp2[i]);
                    break;
                }
            }
        }
        // T1 diagnostic: registry bytes vs locally-built tile+ss, and the
        // direct kernel called on the registry bytes with a local qa.
        {
            const size_t nbi = cols / 256, r8i = rows / 8;
            std::vector<uint8_t> wpack(r8i * nbi * sizeof(tile_block_q4_Kx8));
            std::vector<uint8_t> ws(r8i * nbi * sizeof(tile_ss_q4_Kx8s));
            sofuu_repack_q4k_8bl(wpack.data(), w4.data(), rows, nbi, 4);
            const tile_block_q4_Kx8 * ins  = (const tile_block_q4_Kx8 *)wpack.data();
            tile_ss_q4_Kx8s *         outs = (tile_ss_q4_Kx8s *)ws.data();
            for (size_t i = 0; i < (size_t)r8i * nbi; ++i) fill_q4k_s_block(&outs[i], &ins[i]);
            const uint8_t * tile_reg = nullptr, * ss_reg = nullptr;
            tile_reg = sofuu_tile_lookup_s(w4.data(), rows, cols, &ss_reg);
            fprintf(stderr, "[diag] lookup_s tile=%d ss=%d\n", tile_reg != nullptr, ss_reg != nullptr);
            if (tile_reg && ss_reg) {
                for (size_t i = 0; i < wpack.size(); ++i) {
                    if (wpack[i] != tile_reg[i]) { fprintf(stderr, "[diag] TILE byte %zu: local %02x reg %02x\n", i, wpack[i], tile_reg[i]); break; }
                }
                for (size_t i = 0; i < ws.size(); ++i) {
                    if (ws[i] != ss_reg[i]) {
                        fprintf(stderr, "[diag] SS byte %zu: local %02x reg %02x (block %zu off %zu)\n", i, ws[i], ss_reg[i], i / 192, i % 192);
                        break;
                    }
                }
                std::vector<tile_block_q8_Kx4> qa((batch / 4) * nbi);
                for (size_t g4 = 0; g4 < (size_t)batch / 4; ++g4) {
                    ggml_quantize_mat_q8_K_4x4_neon(x.data() + g4 * 4 * cols, qa.data() + g4 * nbi, cols);
                }
                {
                    uint32_t c = 2166136261u;
                    const uint8_t * qb = (const uint8_t *)qa.data();
                    for (size_t i = 0; i < qa.size() * sizeof(tile_block_q8_Kx4); ++i) { c = (c ^ qb[i]) * 16777619u; }
                    fprintf(stderr, "[qa-dump] probe  qa ffnv1a=%08x x0=%.9g x1=%.9g nb=%zu g4n=%zu\n", c, x[0], x[1], nbi, batch / 4);
                }
                std::vector<float> oa((size_t)batch * rows), oz((size_t)batch * rows);
                for (size_t g = 0; g < (size_t)r8i; ++g) {
                    ggml_gemm_q4_K_8x4_q8_K(cols, oa.data() + g * 8, rows,
                        wpack.data() + g * nbi * sizeof(tile_block_q4_Kx8), qa.data(), batch, 8, nullptr);
                    ggml_gemm_q4_K_8x4_q8_K_s(cols, oz.data() + g * 8, rows,
                        tile_reg + g * nbi * sizeof(tile_block_q4_Kx8),
                        ss_reg + g * nbi * sizeof(tile_ss_q4_Kx8s), qa.data(), batch, 8, nullptr);
                }
                for (size_t i = 0; i < oa.size(); ++i) {
                    if (memcmp(&oa[i], &oz[i], sizeof(float)) != 0) {
                        fprintf(stderr, "[diag] direct(regbytes) at %zu: stock %.9g s %.9g\n", i, oa[i], oz[i]);
                        break;
                    }
                }
                // f16 view of the first ss scales
                const __fp16 * sp0 = (const __fp16 *)ss_reg;
                fprintf(stderr, "[diag] ss_reg[0..7] f16: %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                        (float)sp0[0], (float)sp0[1], (float)sp0[2], (float)sp0[3],
                        (float)sp0[4], (float)sp0[5], (float)sp0[6], (float)sp0[7]);
            }
        }
        return 10;
    }

    // q4_K pair: dual vs two separate gemm calls
    if (sofuu_gemm_q4_k_pair(w4.data(), w4b.data(), x.data(), p1.data(), p2.data(), rows, cols, batch) != 0) return 2;
    std::vector<float> s1(batch * rows), s2(batch * rows);
    sofuu_gemm_q4_k(w4.data(), x.data(), s1.data(), rows, cols, batch);
    sofuu_gemm_q4_k(w4b.data(), x.data(), s2.data(), rows, cols, batch);
    if (memcmp(s1.data(), p1.data(), s1.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-pair] q4_K out1 mismatch\n"); return 11; }
    if (memcmp(s2.data(), p2.data(), s2.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-pair] q4_K out2 mismatch\n"); return 12; }

    // q8_0 residual + pair (rows % 8 also satisfies q8_0's % 4)
    ref.assign(batch * rows, 0.0f);
    sofuu_gemm_q8_0(w8.data(), x.data(), ref.data(), rows, cols, batch);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = resid[i] + ref[i];
    aug = resid;
    if (sofuu_gemm_q8_0_acc(w8.data(), x.data(), aug.data(), rows, cols, batch) != 0) return 3;
    if (memcmp(ref.data(), aug.data(), ref.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-resid] q8_0 mismatch\n"); return 13; }
    if (sofuu_gemm_q8_0_pair(w8.data(), w8b.data(), x.data(), p1.data(), p2.data(), rows, cols, batch) != 0) return 4;
    std::fill(s1.begin(), s1.end(), 0.0f);
    std::fill(s2.begin(), s2.end(), 0.0f);
    sofuu_gemm_q8_0(w8.data(), x.data(), s1.data(), rows, cols, batch);
    sofuu_gemm_q8_0(w8b.data(), x.data(), s2.data(), rows, cols, batch);
    if (memcmp(s1.data(), p1.data(), s1.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-pair] q8_0 out1 mismatch\n"); return 14; }
    if (memcmp(s2.data(), p2.data(), s2.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-pair] q8_0 out2 mismatch\n"); return 15; }

    // Pad path: batch 6 → tile_batch 8; the fused kernel runs on scratch and
    // the residual is added back with sofuu_f32_add_inplace — must equal the
    // unfused pad path + separate add.
    const int batch6 = 6;
    std::vector<float> x6((size_t)batch6 * cols), resid6((size_t)batch6 * rows);
    for (auto &v : x6)     { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f; }
    for (auto &v : resid6) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 10) % 4001 - 2000) / 512.0f; }
    std::vector<float> ref6((size_t)batch6 * rows), aug6((size_t)batch6 * rows);
    sofuu_gemm_q4_k(w4.data(), x6.data(), ref6.data(), rows, cols, batch6);
    for (size_t i = 0; i < ref6.size(); ++i) ref6[i] = resid6[i] + ref6[i];
    aug6 = resid6;
    if (sofuu_gemm_q4_k_acc(w4.data(), x6.data(), aug6.data(), rows, cols, batch6) != 0) return 5;
    if (memcmp(ref6.data(), aug6.data(), ref6.size() * sizeof(float)) != 0) { fprintf(stderr, "[parity-resid] q4_K pad batch mismatch\n"); return 16; }
    return 0;
}

} // extern "C"

