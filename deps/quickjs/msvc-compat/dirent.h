/*
 * MSVC compatibility shim: <dirent.h>.
 *
 * Implements the subset of POSIX directory iteration that
 * quickjs-libc.c's os.readdir uses (opendir/readdir/closedir; only
 * d_name is ever read) on top of FindFirstFileA/FindNextFileA. One
 * entry is returned per call, "." first — same order POSIX yields.
 */
#ifndef SOFUU_MSVC_DIRENT_H
#define SOFUU_MSVC_DIRENT_H

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct dirent {
    char d_name[MAX_PATH];
};

typedef struct {
    HANDLE handle;
    WIN32_FIND_DATAA data;
    int first_pending;
    struct dirent ent;
} DIR;

static DIR *opendir(const char *path)
{
    char pattern[MAX_PATH];
    DIR *d;
    size_t n = strlen(path);

    if (n + 3 > sizeof(pattern)) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    memcpy(pattern, path, n);
    if (n == 0 || (path[n - 1] != '/' && path[n - 1] != '\\'))
        pattern[n++] = '\\';
    pattern[n++] = '*';
    pattern[n] = '\0';

    d = (DIR *)malloc(sizeof(DIR));
    if (!d) {
        errno = ENOMEM;
        return NULL;
    }
    d->handle = FindFirstFileA(pattern, &d->data);
    if (d->handle == INVALID_HANDLE_VALUE) {
        free(d);
        errno = ENOENT;
        return NULL;
    }
    d->first_pending = 1;
    return d;
}

static struct dirent *readdir(DIR *d)
{
    if (!d || d->handle == INVALID_HANDLE_VALUE)
        return NULL;
    if (d->first_pending) {
        d->first_pending = 0;
    } else if (!FindNextFileA(d->handle, &d->data)) {
        return NULL;
    }
    memcpy(d->ent.d_name, d->data.cFileName, MAX_PATH);
    return &d->ent;
}

static int closedir(DIR *d)
{
    if (!d)
        return -1;
    if (d->handle != INVALID_HANDLE_VALUE) {
        FindClose(d->handle);
        d->handle = INVALID_HANDLE_VALUE;
    }
    free(d);
    return 0;
}

#endif /* SOFUU_MSVC_DIRENT_H */
