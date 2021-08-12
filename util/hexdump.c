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

int _abcdk_hexdump_print_char(size_t *total, FILE *fd, uint8_t c, size_t off, size_t size,
                              const char *color, size_t color_s, size_t color_e, int flag)
{
    int chk = -1;

    if (off < size)
    {
        if (color != NULL && off >= color_s && off < color_e)
        {
            chk = _abcdk_hexdump_print(total, fd, color);
            if (chk != 0)
                return chk;
        }

        if (flag)
            chk = _abcdk_hexdump_print(total, fd, "%02hhx", c);
        else
            chk = _abcdk_hexdump_print(total, fd, "%c", (isprint(c) ? c : '.'));

        if (chk != 0)
            return chk;

        if (color != NULL && off >= color_s && off < color_e)
        {
            chk = _abcdk_hexdump_print(total, fd, ABCDK_ANSI_COLOR_RESET);
            if (chk != 0)
                return chk;
        }
    }
    else
    {
        if (flag)
            chk = _abcdk_hexdump_print(total, fd, "  ");
        else
            chk = _abcdk_hexdump_print(total, fd, " ");
    }

    chk = (flag ? _abcdk_hexdump_print(total, fd, " ") : 0);
    if (chk != 0)
        return chk;

    return chk;
}

ssize_t _abcdk_hexdump_match_keyword(const void *data, size_t size, abcdk_allocator_t *keywords)
{
    if (!keywords)
        return -1UL;

    for (size_t i = 0; i < keywords->numbers; i++)
    {
        if (memcmp(data, keywords->pptrs[i], ABCDK_MIN(keywords->sizes[i], size)) == 0)
            return i;
    }

    return -1UL;
}

const char *_abcdk_hexdump_select_color(size_t kwidx, abcdk_allocator_t *palette)
{
    if (!palette)
        return NULL;

    if (palette->numbers > 0)
        return palette->pptrs[kwidx % palette->numbers];

    return NULL;
}

ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size,
                      abcdk_allocator_t *keywords, abcdk_allocator_t *palette)
{
    size_t wsize = 0;
    size_t size_a = 0;
    size_t width = 16;
    size_t col = 0;
    size_t row = 0;
    size_t repeat = 0;
    size_t kwidx = -1UL;
    const char *color = NULL;
    size_t color_s = -1UL;
    size_t color_e = -1UL;
    const uint8_t *p = NULL, *q = NULL;
    int chk;

    assert(fd != NULL && data != NULL && size > 0);

    size_a = abcdk_align(size, width);
    p = (uint8_t *)data;

    for (size_t i = 0; i < size_a;)
    {
        row = i / width;
        col = i % width;

        if (row > 0 && col == 0 && i < size)
        {
            /*检查是否与上一行重复。*/
            if (memcmp(p, p - width, ABCDK_MIN(width, size - i)) == 0)
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
            {
                /**/
                kwidx = color_s = color_e = -1UL;
                /**/
                i += width;
                p += width;
                continue;
            }
        }

        if (col == 0)
        {
            if (size <= UINT32_MAX)
                chk = _abcdk_hexdump_print(&wsize, fd, "%08lx | ", row * width);
            else
                chk = _abcdk_hexdump_print(&wsize, fd, "%016lx | ", row * width);

            if (chk != 0)
                return wsize;
        }

        if (i < size && palette != NULL)
        {
            if (kwidx == -1UL || i == color_e)
            {
                color_s = color_e = -1UL;
                kwidx = _abcdk_hexdump_match_keyword(p, size - i, keywords);
                if (kwidx < keywords->numbers)
                {
                    color_s = i;
                    color_e = i + keywords->sizes[kwidx];
                }
            }
        }
        else
        {
            kwidx = color_s = color_e = -1UL;
        }

        /*从调色板选取颜色。*/
        color = _abcdk_hexdump_select_color(kwidx, palette);

        chk = _abcdk_hexdump_print_char(&wsize, fd, *p, i, size, color, color_s, color_e, 1);
        if (chk != 0)
            return wsize;

        if (col == (width - 1))
        {
            chk = _abcdk_hexdump_print(&wsize, fd, "| ");
            if (chk != 0)
                return wsize;

            q = p - col;
            for (size_t j = 0; j < width;)
            {
                chk = _abcdk_hexdump_print_char(&wsize, fd, *q, i - col + j, size, NULL, -1UL, -1UL, 0);
                if (chk != 0)
                    return wsize;

                j += 1;
                q += 1;
            }

            chk = _abcdk_hexdump_print(&wsize, fd, " |\n");
            if (chk != 0)
                return wsize;
        }

        i += 1;
        p += 1;
    }

final:

    return wsize;
}

ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size,
                       abcdk_allocator_t *keywords, abcdk_allocator_t *palette)
{
    FILE *fp = NULL;
    ssize_t wsize = 0;

    assert(file != NULL && data != NULL && size > 0);

    fp = fopen(file, "w");
    if (fp)
    {
        wsize = abcdk_hexdump(fp, data, size, keywords, palette);
        fclose(fp);
    }

    return wsize;
}
