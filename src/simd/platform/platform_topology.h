// src/simd/platform/platform_topology.h — Device- and OS-specific CPU topology detection.
#pragma once

#include <thread>

#if defined(__APPLE__)
// macOS (Apple Silicon M-series & Intel Mac)
#include <sys/sysctl.h>

static inline int detect_optimal_threads() {
    int p_cores = 0;
    size_t size = sizeof(p_cores);
    // Four threads is the best measured default for dense CPU decode on
    // M-series laptops: it saturates memory bandwidth without adding
    // barrier/scheduling overhead from the remaining performance cores.
    if (sysctlbyname("hw.perflevel0.physicalcpu", &p_cores, &size, NULL, 0) == 0 && p_cores > 0) {
        return p_cores < 4 ? p_cores : 4;
    }
    int phys_cores = 0;
    if (sysctlbyname("hw.physicalcpu", &phys_cores, &size, NULL, 0) == 0 && phys_cores > 0) {
        return phys_cores;
    }
    int logical = (int)std::thread::hardware_concurrency();
    return (logical > 0) ? logical : 4;
}

#elif defined(__ANDROID__)
// Android (big.LITTLE Cortex clusters)
#include <unistd.h>

static inline int detect_optimal_threads() {
    // On mobile big.LITTLE SoCs (e.g. 4 big + 4 small), 4 threads saturate the big cluster without thermal throttling
    long n_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_online > 4) {
        return 4;
    }
    return (n_online > 0) ? (int)n_online : 4;
}

#elif defined(__linux__)
// Linux (Raspberry Pi 5, server ARM64, and x86 Linux)
#include <unistd.h>

static inline int detect_optimal_threads() {
    // Raspberry Pi 5 has 4 physical Cortex-A76 cores with no SMT.
    // Cache the first answer: musl resolves _SC_NPROCESSORS_ONLN through the
    // calling thread's affinity mask, so a later query from a thread the pool
    // pinned to one core reports 1 and would silently collapse the pool.
    static const int cached = [] {
        long n_online = sysconf(_SC_NPROCESSORS_ONLN);
        if (n_online > 0) {
            return (int)n_online;
        }
        int logical = (int)std::thread::hardware_concurrency();
        return (logical > 0) ? logical : 4;
    }();
    return cached;
}

#else
// Windows / Generic Fallback
static inline int detect_optimal_threads() {
    int logical = (int)std::thread::hardware_concurrency();
    return (logical > 0) ? logical : 4;
}
#endif
