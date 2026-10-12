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

#else
// Fallback for Android, Windows, BSD
static inline void sofuu_set_worker_affinity(std::thread &/*th*/, int /*core_id*/) {}
static inline void sofuu_set_current_thread_affinity(int /*core_id*/) {}
#endif
