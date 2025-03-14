/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

typedef struct _abcdk_bc
{
    int errcode;
    abcdk_option_t *args;

    int decode;
    uint8_t base;
    const char *in;
    const char *out;
    int no_amb;

} abcdk_bc_t;

void _abcdk_bc_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tbasecode编解码工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--decode\n");
    fprintf(stderr, "\t\t解码。如果未指定，则启用编码。\n");

    fprintf(stderr, "\n\t--base < CODE >\n");
    fprintf(stderr, "\t\t基值（32|64）。默认：64\n");

    fprintf(stderr, "\n\t--in < FILE | STRING >\n");
    fprintf(stderr, "\t\t输入的文件或字符串。\n");

    fprintf(stderr, "\n\t--out < FILE >\n");
    fprintf(stderr, "\t\t输出的文件。默认: 终端\n");

    fprintf(stderr, "\n\t--no-ambiguity\n");
    fprintf(stderr, "\t\t不使用有歧义的字符。注：仅base32有效\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

uint8_t _abcdk_bc_en_table32_sn(uint8_t n)
{
    return "ABCDEFGH"
           "JKLMN"
           "PQRSTUVWXYZ"
           "23456789"[n];
}

uint8_t _abcdk_bc_de_table32_sn(uint8_t c)
{
    if (c <= '9')
        return (uint8_t)(c - '2' + 24);
    else if (c <= 'H')
        return (uint8_t)(c - 'A');
    else if (c <= 'N')
        return (uint8_t)(c - 'J' + 8);
    else if (c <= 'Z')
        return (uint8_t)(c - 'P' + 13);

    return c;
}

void _abcdk_bc_work(abcdk_bc_t *ctx)
{
    abcdk_object_t *inbuf = NULL,*outbuf = NULL;
    ssize_t outsize = 0,outsize2 = 0;
    abcdk_basecode_t bc = {0};

    ctx->decode = abcdk_option_exist(ctx->args, "--decode");
    ctx->base = abcdk_option_get_int(ctx->args, "--base", 0, 64);
    ctx->in = abcdk_option_get(ctx->args, "--in", 0, "");
    ctx->out = abcdk_option_get(ctx->args, "--out", 0, "");
    ctx->no_amb = abcdk_option_exist(ctx->args, "--no-ambiguity");

    if (ctx->base != 32 && ctx->base != 64)
    {
        fprintf(stderr, "仅支持base32或base64。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (strlen(ctx->in) <= 0)
    {
        fprintf(stderr, "'--in < FILE | STRING >' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }
    else if (access(ctx->in, R_OK) == 0)
    {
        inbuf = abcdk_mmap_filename(ctx->in,0, 0, 0,0);
        if (!inbuf)
        {
            fprintf(stderr, "'%s' %s。\n", ctx->in, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }
    }
    else
    {
        inbuf = abcdk_object_alloc2(0);
        if (!inbuf)
        {
            fprintf(stderr, "%s。\n", strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }

        inbuf->pptrs[0] = (uint8_t *)ctx->in;
        inbuf->sizes[0] = strlen(ctx->in);
    }

    outbuf = abcdk_object_alloc2(inbuf->sizes[0] * 4);
    if (!outbuf)
    {
        fprintf(stderr, "%s。\n", strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    abcdk_basecode_init(&bc, ctx->base);

    if (ctx->base == 32 && ctx->no_amb)
    {
        bc.encode_table_cb = _abcdk_bc_en_table32_sn;
        bc.decode_table_cb = _abcdk_bc_de_table32_sn;
    }

    if (ctx->decode)
        outsize = abcdk_basecode_decode(&bc, inbuf->pptrs[0], inbuf->sizes[0], outbuf->pptrs[0], outbuf->sizes[0]);
    else
        outsize = abcdk_basecode_encode(&bc, inbuf->pptrs[0], inbuf->sizes[0], outbuf->pptrs[0], outbuf->sizes[0]);

    if (strlen(ctx->out) <= 0)
    {
        outsize2 = fprintf(stdout,"%s",outbuf->pptrs[0]);
        fprintf(stdout,"\n");
    }
    else
    {
        outsize2 = abcdk_save(ctx->out,outbuf->pptrs[0],outsize,0);
    }

    if(outsize2 != outsize)
    {
        fprintf(stderr, "空间不足或无法格式化输出。\n");
        ctx->errcode = ENOSPC;
    }

final:

    abcdk_object_unref(&inbuf);
    abcdk_object_unref(&outbuf);
}

int abcdk_tool_basecode(abcdk_option_t *args)
{
    abcdk_bc_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_bc_print_usage(ctx.args);
    }
    else
    {
        _abcdk_bc_work(&ctx);
    }

    return ctx.errcode;
}