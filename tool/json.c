/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "entry.h"

typedef struct _abcdk_json
{
    int errcode;

    abcdk_option_t *args;

    const char *file;
    int readable;
    const char *outfile;

#ifdef _json_h_
    json_object *obj;
#endif //_json_h_

}abcdk_json_t;

#ifdef _json_h_

void _abcdk_json_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的JSON格式化工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\tJSON文件。\n");

    fprintf(stderr, "\n\t--readable\n");
    fprintf(stderr, "\t\t使文本便于阅读(截断超过80字节的文本)。默认：原文\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到指定的文件(包括路径)。默认：终端\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_json_wrok(abcdk_json_t *ctx)
{
    ctx->file = abcdk_option_get(ctx->args,"--file",0,NULL);
    ctx->readable = abcdk_option_exist(ctx->args,"--readable");
    ctx->outfile = abcdk_option_get(ctx->args,"--output",0,NULL);

    if (!ctx->file || !*ctx->file)
    {
        fprintf(stderr, "'--file FILE' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    if (access(ctx->file, R_OK) != 0)
    {
        fprintf(stderr, "'%s' %s。\n", ctx->file, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
    }


    if(ctx->outfile && *ctx->outfile)
    {
        if(abcdk_reopen(STDOUT_FILENO,ctx->outfile,1,0,1)<0)
        {
            fprintf(stderr, "'%s' %s.\n", ctx->outfile, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
        }
    }

    ctx->obj = json_object_from_file(ctx->file);
    if(!ctx->obj)
    {
        fprintf(stderr, "'%s' %s。\n", ctx->file, strerror(ESPIPE));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ESPIPE,final);
    }

    abcdk_json_readable(stdout,ctx->readable,0,ctx->obj);
    
    fprintf(stdout, "\n");
    fflush(stdout);

final:

    abcdk_json_unref(&ctx->obj);
}

#endif //_json_h_

int abcdk_tool_json(abcdk_option_t *args)
{
    abcdk_json_t ctx = {0};

#ifdef _json_h_

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_json_print_usage(ctx.args);
    }
    else
    {
        _abcdk_json_wrok(&ctx);
    }

#else 
    
    fprintf(stderr, "当前构建版本未包含此工具。\n");
    ctx.errcode = EPERM;

#endif //_json_h_

    return ctx.errcode;
}