// src/simd/platform/platform_sync.h — Hardware-specific spin-pause, memory barrier, and thread yield primitives.
#pragma once

#include <thread>

#if defined(__aarch64__) || defined(_M_ARM64)
// ARM64: Apple Silicon, Raspberry Pi 5 Cortex-A76, Android ARM64.
// Use ARM YIELD instruction to give up pipeline execution slots without flushing the pipeline.
static inline void sofuu_spin_pause() {
    asm volatile("yield" ::: "memory");
}
#elif defined(__x86_64__) || defined(_M_X64)
// x86_64: Intel, AMD.
#include <immintrin.h>
static inline void sofuu_spin_pause() {
    _mm_pause();
}
#elif defined(__arm__)
// 32-bit ARM: Raspberry Pi 2/3 32-bit, older Android.
static inline void sofuu_spin_pause() {
    asm volatile("yield" ::: "memory");
}
#else
// Generic fallback.
static inline void sofuu_spin_pause() {
    std::this_thread::yield();
}
#endif
