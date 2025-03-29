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

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if(ctx_p->img)
    {
        if(ctx_p->img->tag == ABCDK_TORCH_TAG_HOST)
            abcdk_torch_image_free_host(&ctx_p->img);
        else if(ctx_p->img->tag == ABCDK_TORCH_TAG_CUDA)
            abcdk_torch_image_free_cuda(&ctx_p->img);
    }

    abcdk_heap_free(ctx_p);
}


abcdk_torch_frame_t *abcdk_torch_frame_alloc()
{
    abcdk_torch_frame_t *ctx;

    ctx = (abcdk_torch_frame_t*)abcdk_heap_alloc(sizeof(abcdk_torch_frame_t));
    if(!ctx)
        return NULL;

    ctx->dts = (int64_t)UINT64_C(0x8000000000000000);
    ctx->pts = (int64_t)UINT64_C(0x8000000000000000);
    ctx->img = NULL;

    return ctx;
}

int abcdk_torch_frame_reset_host(abcdk_torch_frame_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_torch_frame_t *ctx_p;
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt >= 0);

    ctx_p = *ctx;

    if (!ctx_p)
    {
        *ctx = abcdk_torch_frame_alloc();
        if (!*ctx)
            return -1;

        chk = abcdk_torch_frame_reset_host(ctx, width, height, pixfmt, align);
        if (chk != 0)
            abcdk_torch_frame_free(ctx);

        return chk;
    }

    if(ctx_p->img)
    {
        if(ctx_p->img->tag == ABCDK_TORCH_TAG_HOST)
            abcdk_torch_image_free_host(&ctx_p->img);
        else if(ctx_p->img->tag == ABCDK_TORCH_TAG_CUDA)
            abcdk_torch_image_free_cuda(&ctx_p->img);
    }

    ctx_p->dts = (int64_t)UINT64_C(0x8000000000000000);
    ctx_p->pts = (int64_t)UINT64_C(0x8000000000000000);

    ctx_p->img = abcdk_torch_image_create_host(width,height,pixfmt,align);
    if(!ctx_p->img)
    {
        abcdk_torch_frame_free(&ctx_p);
        return -1;
    }

    return 0;
}

abcdk_torch_frame_t *abcdk_torch_frame_create_host(int width, int height, int pixfmt, int align)
{
    abcdk_torch_frame_t *ctx = NULL;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_torch_frame_reset_host(&ctx,width,height,pixfmt,align);
    if(chk != 0)
        return NULL;

    return ctx;
}