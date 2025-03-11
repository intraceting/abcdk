/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/frame.h"


void abcdk_torch_frame_free(abcdk_torch_frame_t **ctx)
{
    abcdk_torch_frame_t *ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_torch_image_free(&ctx_p->img);
    abcdk_heap_free(ctx_p);
}


abcdk_torch_frame_t *abcdk_torch_frame_alloc()
{
    abcdk_torch_frame_t *ctx;

    ctx = (abcdk_torch_frame_t*)abcdk_heap_alloc(sizeof(abcdk_torch_frame_t));
    if(!ctx)
        return NULL;

    ctx->img = NULL;
    ctx->dts = (int64_t)UINT64_C(0x8000000000000000);
    ctx->pts = (int64_t)UINT64_C(0x8000000000000000);

    return ctx;
}