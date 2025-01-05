/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/hexdump.h"

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

int _abcdk_hexdump_print2(size_t *total, FILE *fd, uint8_t c, size_t off, size_t size,
                          const char *color, size_t color_s, size_t color_e, int iscode,int base)
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

        if (iscode)
        {
            if (base == ABCDK_HEXDEMP_BASE_DEC)
                chk = _abcdk_hexdump_print(total, fd, "%03hhu", c);
            else if (base == ABCDK_HEXDEMP_BASE_OCT)
                chk = _abcdk_hexdump_print(total, fd, "%03hho", c);
            else /*if(base == ABCDK_HEXDEMP_BASE_HEX)*/
                chk = _abcdk_hexdump_print(total, fd, "%02hhx", c);
        }
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
        if (iscode)
            chk = _abcdk_hexdump_print(total, fd, "  ");
        else
            chk = _abcdk_hexdump_print(total, fd, " ");
    }

    chk = (iscode ? _abcdk_hexdump_print(total, fd, " ") : 0);
    if (chk != 0)
        return chk;

    return chk;
}

ssize_t _abcdk_hexdump_match_keyword(const void *data, size_t size, const abcdk_hexdump_option_t *opt)
{
    if (!opt->keyword)
        return -1UL;

    for (size_t i = 0; i < opt->keyword->numbers; i++)
    {
        if(!opt->keyword->pptrs[i] || opt->keyword->sizes[i]==0)
            continue;

        if (size < opt->keyword->sizes[i])
            continue;

        if (memcmp(data, opt->keyword->pptrs[i],opt->keyword->sizes[i]) == 0)
            return i;
    }

    return -1UL;
}

const char *_abcdk_hexdump_select_color(size_t kwidx, const abcdk_hexdump_option_t *opt)
{
    if (!opt->palette)
        return NULL;

    if (opt->palette->numbers > 0)
        return opt->palette->pptrs[kwidx % opt->palette->numbers];

    return NULL;
}

ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size, size_t offset, const abcdk_hexdump_option_t *opt)
{
    size_t wsize = 0;
    size_t size_a = 0;
    size_t width = 16;
    size_t col = 0;
    size_t row = 0;
    size_t repeat = 0;
    size_t kwidx = -1UL,kwidx2 = -1UL;
    const char *color = NULL, *color2 = NULL;
    size_t color_s = -1UL, color2_s = -1UL;
    size_t color_e = -1UL, color2_e = -1UL;
    const uint8_t *p = NULL, *q = NULL;
    abcdk_tree_t *stack = NULL,*stack_p = NULL;
    abcdk_hexdump_option_t opt_default = {0};
    int chk;

    assert(fd != NULL && data != NULL && size > 0);

    if (!opt)
    {
        opt_default.flag = ABCDK_HEXDEMP_SHOW_ADDR | ABCDK_HEXDEMP_SHOW_CHAR;
        opt = &opt_default;
    }

    if (opt->width > 0)
        width = opt->width;

    size_a = abcdk_align(size, width);
    p = (uint8_t *)data;


    for (size_t i = 0; i < size_a;)
    {
        row = i / width;
        col = i % width;

        if (row > 0 && col == 0 && i+width < size)
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
                    goto final;
            }

            /*下一行。*/
            if (repeat > 0)
            {
                /**/
                kwidx = color_s = color_e = -1UL;
                kwidx2 = color2_s = color2_e = -1UL;

                /**/
                i += width;
                p += width;
                continue;
            }
        }

        if ((col == 0) && (opt->flag & ABCDK_HEXDEMP_SHOW_ADDR))
        {
            if (offset + size <= UINT32_MAX)
                chk = _abcdk_hexdump_print(&wsize, fd, "%08lx | ", offset + row * width);
            else
                chk = _abcdk_hexdump_print(&wsize, fd, "%016lx | ", offset + row * width);

            if (chk != 0)
                goto final;
        }

        if (i < size && opt->keyword != NULL && opt->palette != NULL)
        {
            if (kwidx == -1UL || i >= color_e)
            {
                color_s = color_e = -1UL;
                kwidx = _abcdk_hexdump_match_keyword(p, size - i, opt);
                if (kwidx < opt->keyword->numbers)
                {
                    color_s = i;
                    color_e = i + opt->keyword->sizes[kwidx];

                    /*可能存在多段不同颜色，因这里准备颜色队列。*/
                    if(!stack)
                    {
                        stack = abcdk_tree_alloc3(1);
                        if (!stack)
                            goto final;
                    }

                    stack_p = abcdk_tree_alloc2(NULL,3,0);
                    if (!stack)
                        goto final;

                    stack_p->obj->sizes[0] = kwidx;
                    stack_p->obj->sizes[1] = color_s;
                    stack_p->obj->sizes[2] = color_e;

                    /*按顺序压入队列。*/
                    abcdk_tree_insert2(stack,stack_p,0);
                    stack_p = NULL;
                }
            }
        }
        else
        {
            kwidx = color_s = color_e = -1UL;
        }

        /*从调色板选取颜色。*/
        color = _abcdk_hexdump_select_color(kwidx, opt);

        chk = _abcdk_hexdump_print2(&wsize, fd, *p, i, size, color, color_s, color_e, 1,opt->base);
        if (chk != 0)
            goto final;

        if (col == (width - 1))
        {
            if (opt->flag & ABCDK_HEXDEMP_SHOW_CHAR)
            {
                chk = _abcdk_hexdump_print(&wsize, fd, "| ");
                if (chk != 0)
                    goto final;

                q = p - col;
                for (size_t j = 0; j < width;)
                {
                    size_t i2 = i - col + j;

                    /*从颜色队列中取出颜色，如果存在的话。*/
                    while(stack)
                    {
                        kwidx2 = color2_s = color2_e = -1UL;
                        
                        stack_p = abcdk_tree_child(stack,1);
                        if(!stack_p)
                            break;

                        if( i2 < stack_p->obj->sizes[1])
                            break;
                        
                        if( i2 >= stack_p->obj->sizes[2])
                        {
                            abcdk_tree_unlink(stack_p);
                            abcdk_tree_free(&stack_p);
                        }
                        else
                        {
                            kwidx2 = stack_p->obj->sizes[0];
                            color2_s = stack_p->obj->sizes[1];
                            color2_e = stack_p->obj->sizes[2];
                            break;
                        }
                    }

                    /*从调色板选取颜色。*/
                    color2 = _abcdk_hexdump_select_color(kwidx2, opt);

                    chk = _abcdk_hexdump_print2(&wsize, fd, *q, i2, size, color2, color2_s, color2_e, 0,opt->base);
                    if (chk != 0)
                        goto final;

                    j += 1;
                    q += 1;
                }

                chk = _abcdk_hexdump_print(&wsize, fd, " |");
                if (chk != 0)
                    goto final;
            }

            chk = _abcdk_hexdump_print(&wsize, fd, "\n");
            if (chk != 0)
                goto final;

            
        }


        i += 1;
        p += 1;
    }

final:

    abcdk_tree_free(&stack);

    return wsize;
}

ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size,size_t offset, const abcdk_hexdump_option_t *opt)
{
    FILE *fp = NULL;
    ssize_t wsize = 0;

    assert(file != NULL && data != NULL && size > 0 && opt != NULL);

    fp = fopen(file, "w");
    if (fp)
    {
        wsize = abcdk_hexdump(fp, data, size, offset, opt);
        fclose(fp);
    }

    return wsize;
}
