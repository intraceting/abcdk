/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

void _abcdk_hd_print_usage(abcdk_option_t *args, int only_version)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的十六进制格式查看工具.\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息.\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\t文件(包括路径).\n");

    fprintf(stderr, "\n\t--offset < OFFSET >\n");
    fprintf(stderr, "\t\t偏移量(0为起始).默认: 0\n");

    fprintf(stderr, "\n\t--size < SIZE >\n");
    fprintf(stderr, "\t\t长度(字节).默认: 1024\n");

    fprintf(stderr, "\n\t--width < WIDTH >\n");
    fprintf(stderr, "\t\t宽度.默认: 16\n");

    fprintf(stderr, "\n\t--base < BASE >\n");
    fprintf(stderr, "\t\t进制.默认: %d\n",ABCDK_HEXDEMP_BASE_HEX);

    fprintf(stderr, "\n\t\t%d: 十六进制\n",ABCDK_HEXDEMP_BASE_HEX);
    fprintf(stderr, "\t\t%d: 十进制\n",ABCDK_HEXDEMP_BASE_DEC);
    fprintf(stderr, "\t\t%d: 八进制\n",ABCDK_HEXDEMP_BASE_OCT);

    fprintf(stderr, "\n\t--show-addr\n");
    fprintf(stderr, "\t\t显示地址.默认: 不显示\n");

    fprintf(stderr, "\n\t--show-char\n");
    fprintf(stderr, "\t\t显示字符.默认: 不显示\n");

    fprintf(stderr, "\n\t--palette < COLOR [COLOR ...] >\n");
    fprintf(stderr, "\t\t调色板.默认: ");
    fprintf(stderr, ABCDK_ANSI_COLOR_RED "\\x1b\[31m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, ABCDK_ANSI_COLOR_GREEN "\\x1b\[32m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, ABCDK_ANSI_COLOR_YELLOW "\\x1b\[33m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, ABCDK_ANSI_COLOR_BLUE "\\x1b\[34m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, ABCDK_ANSI_COLOR_MAGENTA "\\x1b\[35m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, ABCDK_ANSI_COLOR_CYAN "\\x1b\[36m " ABCDK_ANSI_COLOR_RESET);
    fprintf(stderr, "\n");

    fprintf(stderr, "\n\t--keyword < STRING [STRING ...] >\n");
    fprintf(stderr, "\t\t关键字.\n");

    fprintf(stderr, "\n\t--keyword16 < HEXSTRING [HEXSTRING ...] >\n");
    fprintf(stderr, "\t\t关键字(16进制).每个关键字的长度为2的倍数, 否则忽略.如: 0102 0a0b \n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径).默认: 终端\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_hd_keyword_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    for(size_t i = 0;i<alloc->numbers;i++)
    {
        if(!alloc->pptrs[i])
            continue;

        abcdk_heap_free(alloc->pptrs[i]);
    }
}

void _abcdk_hd_work(abcdk_option_t *args)
{
    int err = 0;
    abcdk_object_t *mfile = NULL;
    abcdk_hexdump_option_t opt = {0};
    const char *file = NULL;
    const char *outfile = NULL;
    size_t offset = 0;
    size_t size = -1UL;
    size_t width = 16;
    ssize_t palettes = 0;
    ssize_t keywords = 0, keywords_a = 0, keywords_b = 0;

    file = abcdk_option_get(args, "--file", 0, NULL);
    offset = abcdk_option_get_llong(args, "--offset", 0, 0);
    size = abcdk_option_get_llong(args, "--size", 0, 1024);
    width = abcdk_option_get_int(args, "--width", 0, 16);
    outfile = abcdk_option_get(args, "--output", 0, NULL);

    opt.base = abcdk_option_get_int(args, "--base", 0, ABCDK_HEXDEMP_BASE_HEX);

    if (abcdk_option_exist(args, "--show-addr"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_ADDR;
    if (abcdk_option_exist(args, "--show-char"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_CHAR;

    opt.width = width;

    palettes = abcdk_option_count(args, "--palette");
    if (palettes <= 0)
    {
        opt.palette = abcdk_object_alloc(NULL, 6, 0);
        if (!opt.palette)
            goto final;

        opt.palette->pptrs[0] = (uint8_t*)ABCDK_ANSI_COLOR_RED;
        opt.palette->pptrs[1] = (uint8_t*)ABCDK_ANSI_COLOR_GREEN;
        opt.palette->pptrs[2] = (uint8_t*)ABCDK_ANSI_COLOR_YELLOW;
        opt.palette->pptrs[3] = (uint8_t*)ABCDK_ANSI_COLOR_BLUE;
        opt.palette->pptrs[4] = (uint8_t*)ABCDK_ANSI_COLOR_MAGENTA;
        opt.palette->pptrs[5] = (uint8_t*)ABCDK_ANSI_COLOR_CYAN;
    }
    else
    {
        opt.palette = abcdk_object_alloc(NULL, palettes, 0);
        if (!opt.palette)
            goto final;

        for (size_t i = 0; i < opt.palette->numbers; i++)
            opt.palette->pptrs[i] = (uint8_t *)abcdk_option_get(args, "--palette", i, NULL);
    }

    if (abcdk_option_count(args, "--keyword") > 0)
        keywords += (keywords_a = abcdk_option_count(args, "--keyword"));
    if (abcdk_option_count(args, "--keyword16") > 0)
        keywords += (keywords_b = abcdk_option_count(args, "--keyword16"));

    if (keywords > 0)
    {
        opt.keyword = abcdk_object_alloc(NULL, keywords, 0);
        if (!opt.keyword)
            goto final;

        abcdk_object_atfree(opt.keyword,_abcdk_hd_keyword_destroy_cb,NULL);

        for (size_t i = 0; i < keywords; i++)
        {
            if (i < keywords_a)
            {
                opt.keyword->pptrs[i] = abcdk_strdup(abcdk_option_get(args,"--keyword",i,""));
                if(!opt.keyword->pptrs[i])
                    goto final;

                opt.keyword->sizes[i] = strlen(opt.keyword->pptrs[i]);
            }
            else if (i - keywords_a < keywords_b)
            {
                size_t i2 = i - keywords_a;
                const char *p = abcdk_option_get(args, "--keyword16", i2, "");
                size_t l = strlen(p);
                if (l % 2)
                    continue;

                opt.keyword->pptrs[i] = abcdk_heap_alloc(l/2);
                if(!opt.keyword->pptrs[i])
                    goto final;
                    
                opt.keyword->sizes[i] = l/2;
                abcdk_hex2bin(opt.keyword->pptrs[i],p,l);
            }
        }
    }

    /*Clear errno.*/
    errno = 0;

    if (!file || !*file)
    {
        fprintf(stderr, "'--file FILE' 不能省略, 且不能为空.\n");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(file, R_OK) != 0)
    {
        fprintf(stderr, "'%s' %s.\n", file, strerror(errno));
        goto final;
    }

    mfile = abcdk_mmap_filename(file,0, 0, 0,0);
    if (!mfile)
    {
        fprintf(stderr, "'%s' %s.\n", file, strerror(errno));
        goto final;
    }

    if (offset >= mfile->sizes[0])
    {
        fprintf(stderr, "'--offset OFFSET' 不能超过文件长度(%zu).\n", mfile->sizes[0]);
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (size > mfile->sizes[0] - offset)
        size = mfile->sizes[0] - offset;

    if (outfile && *outfile)
    {
        if (abcdk_reopen(STDOUT_FILENO, outfile, 1, 0, 1) < 0)
        {
            fprintf(stderr, "'%s' %s.\n", outfile, strerror(errno));
            goto final;
        }
    }

    abcdk_hexdump(stdout, mfile->pptrs[0] + offset, size, offset, &opt);
    
    fflush(stdout);

    ABCDK_ERRNO_AND_GOTO1(0, final);

final:

    err = errno;
    abcdk_object_unref(&mfile);
    abcdk_object_unref(&opt.keyword);
    abcdk_object_unref(&opt.palette);
    errno = err;
}

int abcdk_tool_hexdump(abcdk_option_t *args)
{
    if (abcdk_option_exist(args, "--help"))
    {
        _abcdk_hd_print_usage(args, 0);
    }
    else
    {
        _abcdk_hd_work(args);
    }

    return errno;
}