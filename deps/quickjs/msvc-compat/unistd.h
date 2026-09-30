/*
 * MSVC compatibility shim: <unistd.h>.
 *
 * Only put on the include path when building the vendored QuickJS with
 * cl.exe (sofuu-ffi/build.rs gates it on CARGO_CFG_TARGET_ENV == "msvc");
 * with gcc/clang this directory is not added at all.
 *
 * Two layers:
 *   1. include the UCRT headers that carry the _-prefixed twins, with
 *      _CRT_DECLARE_NONSTDC_NAMES so O_*/struct stat/stat() resolve (the
 *      UCRT also defines some POSIX names as macros itself — identical
 *      to ours);
 *   2. map the remaining POSIX call names onto the twins. Every mapping
 *      is preceded by #undef so the result does not depend on which
 *      non-stdc names the UCRT happened to define (notably lseek: the
 *      code passes 64-bit offsets, so it must go to _lseeki64, not the
 *      UCRT's 32-bit _lseek).
 */
#ifndef SOFUU_MSVC_UNISTD_H
#define SOFUU_MSVC_UNISTD_H

#ifndef _CRT_DECLARE_NONSTDC_NAMES
#define _CRT_DECLARE_NONSTDC_NAMES 1
#endif

#include <io.h>        /* _open/_close/_read/_write/_isatty/_access   */
#include <process.h>   /* _getpid etc.                                */
#include <direct.h>    /* _getcwd/_mkdir/_rmdir                       */
#include <fcntl.h>     /* _O_* open() flags + O_* spellings           */
#include <stdint.h>    /* intptr_t for ssize_t                        */
#include <string.h>    /* _strdup                                     */
#include <stdlib.h>    /* _environ                                    */
#include <stdio.h>     /* _popen/_pclose declarations                 */
#include <errno.h>
#include <sys/stat.h>  /* struct stat / stat()                        */

/* UCRT has no ssize_t in the C headers. */
#ifndef _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#define _SSIZE_T_DEFINED
#endif

/* POSIX PATH_MAX: MSVC limits.h does not define it; 4096 matches the
 * quickjs-libc.c callers' buffer sizing on Unix. */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Windows stat() never reports these file types, but the os module flag
 * table exposes the POSIX spellings unconditionally. */
#ifndef S_IFBLK
#define S_IFBLK 0
#endif
#ifndef S_IFSOCK
#define S_IFSOCK 0
#endif
#ifndef S_IFLNK
#define S_IFLNK 0
#endif

/* ── name mappings (fseeko/ftello are not in the UCRT at all) ─── */
#undef open
#define open    _open
#undef close
#define close   _close
#undef read
#define read    _read
#undef write
#define write   _write
#undef lseek
#define lseek   _lseeki64
#undef isatty
#define isatty  _isatty
#undef access
#define access  _access
#undef unlink
#define unlink  _unlink
#undef getcwd
#define getcwd  _getcwd
#undef mkdir
#define mkdir   _mkdir       /* _WIN32 branch calls it without a mode */
#undef rmdir
#define rmdir   _rmdir
#undef strdup
#define strdup  _strdup
#undef popen
#define popen   _popen
#undef pclose
#define pclose  _pclose
#undef fseeko
#define fseeko  _fseeki64
#undef ftello
#define ftello  _ftelli64

/* js_std_getenviron() walks a NULL-terminated char**. */
#undef environ
#define environ _environ

#endif /* SOFUU_MSVC_UNISTD_H */
