/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/hexdump.h"

int _abcdk_hexdump_print(size_t *total, FILE *fd, const char *fmt, ...)
{
    ssize_t wsize2 = -1;

    va_list vaptr;
    va_start(vaptr, fmt);

    wsize2 = vfprintf(fd, fmt, vaptr);

    va_end(vaptr);

    if (wsize2 > 0)
        *total += wsize2;

    return (wsize2 > 0 ? 0 : -1);
}

ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size)
{
    size_t wsize = 0;
    size_t width = 16;
    size_t lines = 0;
    size_t remainder = 0;
    const int8_t *p = NULL;
    size_t off = 0;
    size_t repeat = 0;
    int chk;

    assert(fd != NULL && data != NULL && size > 0);

    lines = size / width;
    remainder = size % width;
    p = ABCDK_PTR2U8PTR(data, 0);

    for (off = 0; off < lines; off++, p += width)
    {
        if (off > 0 && off < lines - 1)
        {
            /*检查是否与上一行重复。*/
            if (memcmp(p, p - width, width) == 0)
                repeat += 1;
            else
                repeat = 0;

            /*当与上一行重复用*号代替，但多行重复也仅打印一行。*/
            if (repeat == 1)
            {
                chk = _abcdk_hexdump_print(&wsize, fd, "*\n");
                if (chk != 0)
                    return wsize;
            }

            /*下一行。*/
            if (repeat > 0)
                continue;
        }

        if (size <= UINT32_MAX)
            chk = _abcdk_hexdump_print(&wsize, fd, "%08lx | ", off * width);
        else
            chk = _abcdk_hexdump_print(&wsize, fd, "%016lx | ", off * width);

        if (chk != 0)
            return wsize;

        for (size_t j = 0; j < width; j++)
        {
            chk = _abcdk_hexdump_print(&wsize, fd, "%02hhx", p[j]);
            if (chk != 0)
                return wsize;

            chk = _abcdk_hexdump_print(&wsize, fd, " ");
            if (chk != 0)
                return wsize;
        }

        chk = _abcdk_hexdump_print(&wsize, fd, "| ");
        if (chk != 0)
            return wsize;

        for (size_t j = 0; j < width; j++)
        {
            char c = p[j];
            chk = _abcdk_hexdump_print(&wsize, fd, "%c", (isprint(c) ? c : '.'));
            if (chk != 0)
                return wsize;
        }

        chk = _abcdk_hexdump_print(&wsize, fd, " |\n");
        if (chk != 0)
            return wsize;
    }

    if (remainder <= 0)
        goto final;


    if (size <= UINT32_MAX)
        chk = _abcdk_hexdump_print(&wsize, fd, "%08lx | ", off * width);
    else
        chk = _abcdk_hexdump_print(&wsize, fd, "%016lx | ", off * width);

    if (chk != 0)
        return wsize;

    for (size_t j = 0; j < width; j++)
    {
        if (j < remainder)
            chk = _abcdk_hexdump_print(&wsize, fd, "%02hhx", p[j]);
        else
            chk = _abcdk_hexdump_print(&wsize, fd, "  ");

        if (chk != 0)
            return wsize;
        
        chk = _abcdk_hexdump_print(&wsize, fd, " ");
        if (chk != 0)
            return wsize;
    }

    chk = _abcdk_hexdump_print(&wsize, fd, "| ");
    if (chk != 0)
        return wsize;

    for (size_t j = 0; j < width; j++)
    {
        if (j < remainder)
        {
            char c = p[j];
            chk = _abcdk_hexdump_print(&wsize, fd, "%c", (isprint(c) ? c : '.'));
        }
        else
        {
            chk = _abcdk_hexdump_print(&wsize, fd, " ");
        }
        if (chk != 0)
            return wsize;
    }

    chk = _abcdk_hexdump_print(&wsize, fd, " |\n");
    if (chk != 0)
        return wsize;


final:

    return wsize;
}

ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size)
{
    FILE *fp = NULL;
    ssize_t wsize = 0;

    assert(file != NULL && data != NULL && size > 0);

    fp = fopen(file, "w");
    if (fp)
    {
        wsize = abcdk_hexdump(fp, data, size);
        fclose(fp);
    }

    return wsize;
}
