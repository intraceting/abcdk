/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/general.h"

size_t abcdk_align(size_t size, size_t align)
{
    size_t pad = 0;

    if (align > 1)
    {
        pad = size % align;
        size += ((pad > 0) ? (align - pad) : 0);
    }

    return size;
}

int abcdk_once(volatile int *status, int (*routine)(void *opaque), void *opaque)
{
    int chk, ret;

    assert(status != NULL && routine != NULL);

    if (abcdk_atomic_compare_and_swap(status,0, 1))
    {
        ret = 0;
        chk = routine(opaque);
        abcdk_atomic_store(status, ((chk == 0) ? 2 : 0));
    }
    else
    {
        ret = 1;
        while (abcdk_atomic_load(status) == 1)
            sched_yield();
    }

    chk = ((abcdk_atomic_load(status) == 2) ? 0 : -1);

    return (chk == 0 ? ret : -1);
}

char *abcdk_bin2hex(char* dst,const void *src,size_t size, int ABC)
{
    assert(dst != NULL && src != NULL && size>0);

    for (size_t i = 0; i < size; i++)
    {
        if (ABC)
            sprintf(ABCDK_PTR2U8PTR(dst, i * 2),"%02X", ABCDK_PTR2U8(src,i));
        else
            sprintf(ABCDK_PTR2U8PTR(dst, i * 2),"%02x", ABCDK_PTR2U8(src,i));
    }  
    return dst;
}

void *abcdk_hex2bin(void *dst, const char *src, size_t size)
{
    assert(dst != NULL && src != NULL && size > 0);
    assert(size % 2 == 0);

    for (size_t i = 0; i < size / 2; i++)
    {
        sscanf(ABCDK_PTR2U8PTR(src, i * 2), "%2hhx", ABCDK_PTR2U8PTR(dst,i));
    }
  
    return dst;
}

const char *abcdk_match_env(const char *line, const char *name, uint8_t delim)
{
    const char *s, *d;

    s = line;
    d = name;

    while (*s && *d)
    {
        if (toupper(*s) != toupper(*d))
            break;

        s += 1;
        d += 1;
    }

    /* 提前结束，表示不匹配。*/
    if (*s == '\0' || (!isspace(*s) && *s != delim) || *d != '\0')
        return NULL;

    while (*s)
    {
        if (!isspace(*s) && *s != delim)
            return s;

        s += 1;
    }

    return NULL;
}

void abcdk_memcopy_1d(void *dst, size_t dst_offset, const void *src,size_t src_offset, size_t count)
{
    memcpy(ABCDK_PTR2VPTR(dst, dst_offset), ABCDK_PTR2VPTR(src, src_offset), count);
}

void abcdk_memcopy_2d(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y,
                      const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y,
                      size_t roi_width_bytes, size_t roi_height)
{
    /*一行一行的复制。*/
//#pragma omp parallel
    for (int h = 0; h < roi_height; h++)
        abcdk_memcopy_1d(dst, (h + dst_y) * dst_pitch + dst_x_bytes, src, (h + src_y) * src_pitch + src_x_bytes, roi_width_bytes);
}

pid_t abcdk_waitpid(pid_t pid, int options, int *exitcode, int *sigcode)
{
    int wstatus = 0;
    pid_t pid_chk = -1;

    assert(pid >= 0);

    pid_chk = waitpid(pid, &wstatus, options);

    if (pid_chk == pid)
    {
        if (WIFEXITED(wstatus) && exitcode != NULL)
            *exitcode = WEXITSTATUS(wstatus);

        if (WIFSIGNALED(wstatus) && sigcode != NULL)
            *sigcode = WTERMSIG(wstatus);

        if (WIFSTOPPED(wstatus) && sigcode != NULL)
            *sigcode = WSTOPSIG(wstatus);
    }

    return pid_chk;
}

pid_t abcdk_gettid()
{
	return syscall(SYS_gettid);
} 

uint64_t abcdk_sequence_num()
{
    static volatile uint64_t num = 1;

    return abcdk_atomic_fetch_and_add(&num,1);
}

