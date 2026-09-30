/*
 * MSVC compatibility shim: <sys/time.h>.
 *
 * quickjs.c / quickjs-libc.c include <sys/time.h> unconditionally for
 * struct timeval + gettimeofday. Including winsock2.h here provides the
 * canonical Windows struct timeval AND defines _WINSOCKAPI_, which stops
 * windows.h (included later by quickjs-libc.c) from pulling in winsock.h
 * and redefining it. select()/fd_set for the os poll path come from the
 * same header.
 */
#ifndef SOFUU_MSVC_SYS_TIME_H
#define SOFUU_MSVC_SYS_TIME_H

#include <winsock2.h>
#include <time.h>

/* gettimeofday is not in the UCRT; timespec_get (C11) is. */
static __inline int gettimeofday(struct timeval *tv, void *tz)
{
    struct timespec ts;
    (void)tz;
    if (tv) {
        if (timespec_get(&ts, TIME_UTC) != TIME_UTC)
            return -1;
        tv->tv_sec = (long)ts.tv_sec;
        tv->tv_usec = (long)(ts.tv_nsec / 1000);
    }
    return 0;
}

#endif /* SOFUU_MSVC_SYS_TIME_H */
