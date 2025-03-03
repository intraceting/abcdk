/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/packet.h"

void abcdk_torch_packet_free(abcdk_torch_packet_t **ctx)
{
    abcdk_torch_packet_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->buf);
    abcdk_heap_free(ctx_p);
}


abcdk_torch_packet_t *abcdk_torch_packet_alloc()
{
    abcdk_torch_packet_t *ctx;

    ctx = (abcdk_torch_packet_t*)abcdk_heap_alloc(sizeof(abcdk_torch_packet_t));
    if(!ctx)
        return NULL;

    ctx->data = NULL;
    ctx->size = 0;
    ctx->dts = -1;
    ctx->pts = -1;

    return ctx;
}

int abcdk_torch_packet_reset(abcdk_torch_packet_t **ctx, size_t size)
{
    abcdk_torch_packet_t *ctx_p;
    int chk;

    assert(ctx != NULL && size > 0);

    ctx_p = *ctx;

    if(!ctx_p)
    {
        *ctx = abcdk_torch_packet_alloc();
        if(!*ctx)
            return -1;
        
        chk = abcdk_torch_packet_reset(ctx, size);
        if(chk != 0)
            abcdk_torch_packet_free(ctx);

        return chk;
    }
    
    abcdk_object_unref(&ctx_p->buf);

    ctx_p->data = NULL;
    ctx_p->size = 0;
    ctx_p->dts = -1;
    ctx_p->pts = -1;

    ctx_p->buf = abcdk_object_alloc2(size);
    if (!ctx_p->buf)
        return -1;

    ctx_p->data = ctx_p->buf->pptrs[0];
    ctx_p->size = ctx_p->buf->sizes[0];

    return 0;
}

abcdk_torch_packet_t *abcdk_torch_packet_create(size_t size)
{
    abcdk_torch_packet_t *ctx = NULL;
    int chk;

    assert(size > 0);

    chk = abcdk_torch_packet_reset(&ctx,size);
    if(chk != 0)
        return NULL;

    return ctx;
}