/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/vcodec.h"


static void _abcdk_torch_vcodec_private_free_cb(void **ctx, uint8_t encoder)
{

}

void abcdk_torch_vcodec_free(abcdk_torch_vcodec_t **ctx)
{
    abcdk_torch_vcodec_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);

    abcdk_heap_free(ctx_p);
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc(uint32_t tag)
{
    abcdk_torch_vcodec_t *ctx;

    assert(tag == ABCDK_TORCH_TAG_HOST || tag == ABCDK_TORCH_TAG_CUDA);

    ctx = (abcdk_torch_vcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = tag;
    ctx->private_ctx = _abcdk_torch_vcodec_private_free_cb;

    return ctx;
}