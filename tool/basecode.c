/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "util/general.h"
#include "util/getargs.h"
#include "util/basecode.h"
#include "util/mmap.h"
#include "entry.h"

typedef struct _abcdkbc_ctx
{
    int errcode;
    abcdk_tree_t *args;

    uint8_t base;
    int sn;
    const char *in;
    const char *out;

}abcdkbc_ctx;

void _abcdkbc_print_usage(abcdk_tree_t *args)
{

}

uint8_t _abcdkbc_en_table32_sn(uint8_t n)
{
    return "ABCDEFGH"
           "JKLMN"
           "PQRSTUVWXYZ"
           "23456789"[n];
}

uint8_t _abcdkbc_de_table32_sn(uint8_t c)
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

void _abcdkbc_work(abcdkbc_ctx *ctx)
{
    abcdk_allocator_t *in_buf = NULL;
    abcdk_basecode_t bc = {0};

    ctx->base = abcdk_option_get_int(ctx->args,"--base",0,64,0);
    ctx->sn = abcdk_option_exist(ctx->args,"--sn");
    ctx->in = abcdk_option_get(ctx->args,"--in",0,"");
    ctx->out = abcdk_option_get(ctx->args,"--out",0,"");

    if(ctx->base != 32 && ctx->base != 64)
    {
        syslog(LOG_ERR, "仅支持base64或base32。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }

    if (strlen(ctx->in)<=0)
    {
        syslog(LOG_ERR, "'--in < FILE | STRING >' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL,final);
    }
    else if (access(ctx->in, R_OK) == 0)
    {
        in_buf = abcdk_mmap2(ctx->in,0,0);
        if(!in_buf)
        {
            syslog(LOG_ERR, "'%s' %s。", ctx->in, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
        }
    }
    else
    {
        in_buf = abcdk_allocator_alloc2(0);
        if(!in_buf)
        {
            syslog(LOG_ERR, "%s。", strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
        }
        
        in_buf->pptrs[0] = (uint8_t*)ctx->in;
        in_buf->sizes[0] = strlen(ctx->in);
    }

    if(!in_buf)
        goto final;

    
    abcdk_basecode_init(&bc,ctx->base);

    if(ctx->sn)
    {
        bc.encode_table_cb = _abcdkbc_en_table32_sn;
        bc.decode_table_cb = _abcdkbc_de_table32_sn;
    }

final:


    abcdk_allocator_unref(&in_buf);
}


int abcdk_tool_basecode(abcdk_tree_t *args)
{
    abcdkbc_ctx ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkbc_print_usage(ctx.args);
    }
    else
    {
        _abcdkbc_work(&ctx);
    }

    return ctx.errcode;
}