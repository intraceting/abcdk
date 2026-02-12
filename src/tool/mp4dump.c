/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

void _abcdk_m4d_print_usage(abcdk_option_t *args, int only_version)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的MP4结构查看工具.\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息.\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\t文件(包括路径).\n");

    fprintf(stderr, "\n\t--offset < OFFSET >\n");
    fprintf(stderr, "\t\t偏移量.默认: 0\n");

    fprintf(stderr, "\n\t--all\n");
    fprintf(stderr, "\t\t打印全部结构.默认: 打印基本结构.\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径).默认: 终端\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_m4d_work(abcdk_option_t *args)
{
    int err;
    const char *file = NULL;
    const char *outfile = NULL;
    int all = 0;
    int fd = -1;
    uint64_t offset = 0;
    abcdk_tree_t *doc = NULL;

    file = abcdk_option_get(args, "--file", 0, NULL);
    outfile = abcdk_option_get(args, "--output", 0, NULL);
    all = abcdk_option_exist(args,"--all");
    offset = abcdk_option_get_llong(args,"--offset",0,0);
    

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

    fd = abcdk_open(file, 0, 0, 0);
    if (fd < 0)
        goto final;

    doc = abcdk_mp4_read_probe2(fd, offset, -1UL, (all ? 0 : ABCDK_MP4_ATOM_TYPE_MOOV));
    if (!doc)
        goto final;

    if (outfile && *outfile)
    {
        if (abcdk_reopen(STDOUT_FILENO, outfile, 1, 0, 1) < 0)
        {
            fprintf(stderr, "'%s' %s.\n", outfile, strerror(errno));
            goto final;
        }
    }

    if(!abcdk_mp4_find2(doc,ABCDK_MP4_ATOM_TYPE_FTYP,1,1))
    {
        fprintf(stderr, "'%s' 可能不是MP4文件, 或尚未支持此格式.\n", file);
        ABCDK_ERRNO_AND_GOTO1(EPERM, final);
    }

    abcdk_mp4_dump(stdout,doc);

    errno = 0;

final:

    err = errno;
    abcdk_closep(&fd);
    abcdk_tree_free(&doc);
    errno = err;

}

int abcdk_tool_mp4dump(abcdk_option_t *args)
{
    if (abcdk_option_exist(args, "--help"))
    {
        _abcdk_m4d_print_usage(args, 0);
    }
    else
    {
        _abcdk_m4d_work(args);
    }

    return errno;
}