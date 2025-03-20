/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

typedef struct _abcdk_mcdump
{
    int errcode;
    abcdk_option_t *args;

    const char *outfile;

}abcdk_mcdump_t;

void _abcdk_mcdump_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t获取硬件散列值。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--stuff < STUFF >\n");
    fprintf(stderr, "\t\t填充物。\n");
}

void _abcdk_mcdump_work(abcdk_mcdump_t *ctx)
{
    const char *out = abcdk_option_get(ctx->args,"--output",0,NULL);
    const char *stuff = abcdk_option_get(ctx->args,"--stuff",0,NULL);

    uint8_t uuid[16] = {0};
    char str[33] = {0};


    if(out && *out)
    {
        if(abcdk_reopen(STDOUT_FILENO,out,1,0,1)<0)
        {
            fprintf(stderr, "'%s' %s.\n",out, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
        }
    }

    abcdk_dmi_hash(uuid,stuff);

    abcdk_bin2hex(str,uuid,16,0);

    fprintf(stdout,"%s\n",str);

final:

    return;
}

int abcdk_tool_mcdump(abcdk_option_t *args)
{
    abcdk_mcdump_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_mcdump_print_usage(ctx.args);
    }
    else
    {
        _abcdk_mcdump_work(&ctx);
    }

    return ctx.errcode;
}