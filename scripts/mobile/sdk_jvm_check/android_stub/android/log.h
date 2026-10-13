/* Desktop stand-in for <android/log.h>, used only by the JVM smoke check
 * that compiles this same jni_bridge.c for the host OS. */
#ifndef SOFUU_ANDROID_LOG_STUB_H
#define SOFUU_ANDROID_LOG_STUB_H
#include <stdio.h>
#define ANDROID_LOG_ERROR 6
static inline int __android_log_print(int prio, const char *tag, const char *fmt, ...) {
    (void)prio;
    fprintf(stderr, "[%s] %s\n", tag ? tag : "sofuu", fmt ? fmt : "");
    return 0;
}
#endif
