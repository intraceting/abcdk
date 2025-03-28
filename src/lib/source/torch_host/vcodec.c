/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/vcodec.h"


void abcdk_torch_vcodec_free_host(abcdk_torch_vcodec_t **ctx)
{
    abcdk_torch_vcodec_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;
    
    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    abcdk_heap_free(ctx_p);
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_host()
{
    abcdk_torch_vcodec_t *ctx;

    ctx = (abcdk_torch_vcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;
    ctx->private_ctx = NULL;

    return ctx;
}