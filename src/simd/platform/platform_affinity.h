// src/simd/platform/platform_affinity.h — Thread affinity and CPU core pinning per platform.
#pragma once

#include <thread>

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>

static inline void sofuu_set_worker_affinity(std::thread &th, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
}

static inline void sofuu_set_current_thread_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>

static inline void sofuu_set_worker_affinity(std::thread &th, int /*core_id*/) {
    // macOS Apple Silicon uses QoS classes to steer execution to Performance cores (P-cores)
    pthread_t handle = th.native_handle();
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

static inline void sofuu_set_current_thread_affinity(int /*core_id*/) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#elif defined(__ANDROID__)
// Android: pin the inference pool onto the performance cores. The kernel
// pool's main chunk runs on whichever caller thread spawned inference, and
// workers run sustained GEMV; unpinned, EAS may let any of them drift onto
// the little cores where decode pays off badly. Cores are ranked from
// cpufreq (no other eBPF-style source is available); "performance" = all
// cores within 25% of the fastest. Logical index 0 (the calling chunk)
// lands on the fastest core, 1..n on the rest, and threads beyond the
// performance-cluster size stay unpinned so a deliberate over-thread
// experiment still uses the little cores through EAS.
// SOFUU_AFFINITY=0 disables everything (governor experiments).
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <utility>

namespace sofuu_pin {

// Performance cores, fastest first — read once, cached.
inline const std::vector<int> &perf_cores() {
    static const std::vector<int> cores = [] {
        std::vector<std::pair<int, long>> by_freq;
        for (int i = 0; i < 64; i++) {
            char p[128];
            snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
            FILE *f = fopen(p, "r");
            if (!f) continue;
            long khz = 0;
            if (fscanf(f, "%ld", &khz) == 1 && khz > 0) by_freq.push_back({i, khz});
            fclose(f);
        }
        std::sort(by_freq.begin(), by_freq.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });
        std::vector<int> out;
        if (!by_freq.empty()) {
            const long best = by_freq.front().second;
            for (const auto &c : by_freq) {
                if (c.second >= (best * 3) / 4) out.push_back(c.first);
            }
        }
        return out;
    }();
    return cores;
}

inline bool enabled() {
    // OPT-IN for now (SOFUU_AFFINITY=1): the 2026-10-16 tuning run measured
    // 16.3 tok/s with pinning on vs <50 unpinned the day before, and the
    // cpufreq detection could not be verified on the vivo kernel (SELinux
    // may zero the /sys files or report cluster-uniform maxima). Re-enable
    // by default once a run with the console file naming the detected
    // cores proves the mapping.
    static const bool on = getenv("SOFUU_AFFINITY") != nullptr;
    return on;
}

// Pin `logical` (0 = fastest) for either the current thread or a joinable
// std::thread. Out-of-range logical ids stay with the scheduler.
// Bionic has no pthread_setaffinity_np: the calling thread pins via
// sched_setaffinity(0, ...), workers need their syscall TID first.
template <typename ThreadLike>
inline void pin(ThreadLike *native, bool current, int logical) {
    if (!enabled()) return;
    const auto &cores = perf_cores();
    if (cores.empty() || logical < 0 || logical >= (int)cores.size()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cores[logical], &set);
    if (current) {
        sched_setaffinity(0, sizeof(set), &set);
    } else {
        sched_setaffinity(pthread_gettid_np(*native), sizeof(set), &set);
    }
}

}  // namespace sofuu_pin

static inline void sofuu_set_worker_affinity(std::thread &th, int core_id) {
    pthread_t native = th.native_handle();
    sofuu_pin::pin(&native, /*current=*/false, core_id);
}

static inline void sofuu_set_current_thread_affinity(int core_id) {
    sofuu_pin::pin<pthread_t>(nullptr, /*current=*/true, core_id);
}
#else
// Fallback for Windows, BSD
static inline void sofuu_set_worker_affinity(std::thread &/*th*/, int /*core_id*/) {}
static inline void sofuu_set_current_thread_affinity(int /*core_id*/) {}
#endif
