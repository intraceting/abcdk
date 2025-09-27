/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

typedef struct _abcdk_lsmmc
{
    int errcode;
    abcdk_option_t *args;

    int fmt;
    const char *outfile;

    abcdk_tree_t *list;

}abcdk_lsmmc_t;

/** 输出格式。*/
enum _abcdk_lsmmc_fmt
{
    /** 文本。*/
    ABCDK_LSMMC_FMT_TEXT = 1,
#define ABCDK_LSMMC_FMT_TEXT ABCDK_LSMMC_FMT_TEXT

    /** XML。*/
    ABCDK_LSMMC_FMT_XML = 2,
#define ABCDK_LSMMC_FMT_XML ABCDK_LSMMC_FMT_XML

    /** JSON。*/
    ABCDK_LSMMC_FMT_JSON = 3
#define ABCDK_LSMMC_FMT_JSON ABCDK_LSMMC_FMT_JSON

};

void _abcdk_lsmmc_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tMMC设备枚举器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--fmt < FORMAT >\n");
    fprintf(stderr, "\t\t报表格式。默认: %d\n", ABCDK_LSMMC_FMT_TEXT);

    fprintf(stderr, "\n\t\t%d: TEXT。\n",ABCDK_LSMMC_FMT_TEXT);
    fprintf(stderr, "\t\t%d: XML。\n",ABCDK_LSMMC_FMT_XML);
    fprintf(stderr, "\t\t%d: JSON。\n",ABCDK_LSMMC_FMT_JSON);

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_lsmmc_work(abcdk_lsmmc_t *ctx)
{
    ctx->outfile = abcdk_option_get(ctx->args, "--output", 0, NULL);
    ctx->fmt = abcdk_option_get_int(ctx->args, "--fmt", 0, ABCDK_LSMMC_FMT_TEXT);

    abcdk_mmc_watch(&ctx->list,NULL,NULL);
    if(!ctx->list)
        goto final;

    if (ctx->outfile && *ctx->outfile)
    {
        if (abcdk_reopen(STDOUT_FILENO, ctx->outfile, 1, 0, 1) < 0)
        {
            fprintf(stderr, "'%s' %s.\n", ctx->outfile, strerror(errno));
            goto final;
        }
    }

    abcdk_mmc_format(ctx->list,ctx->fmt,stdout);
    fflush(stdout);

final:

    abcdk_tree_free(&ctx->list);
}

int abcdk_tool_lsmmc(abcdk_option_t *args)
{
    abcdk_lsmmc_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_lsmmc_print_usage(ctx.args);
    }
    else
    {
        _abcdk_lsmmc_work(&ctx);
    }

    return ctx.errcode;
}