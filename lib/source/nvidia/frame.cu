/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/frame.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_cuda_frame_reset(abcdk_torch_frame_t **ctx, int width, int height, int pixfmt, int align)
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

        chk = abcdk_cuda_frame_reset(ctx, width, height, pixfmt, align);
        if (chk != 0)
            abcdk_torch_frame_free(ctx);

        return chk;
    }

    abcdk_torch_image_free(&ctx_p->img);

    ctx_p->dts = (int64_t)UINT64_C(0x8000000000000000);
    ctx_p->pts = (int64_t)UINT64_C(0x8000000000000000);

    ctx_p->img = abcdk_cuda_image_create(width,height,pixfmt,align);
    if(!ctx_p->img)
    {
        abcdk_torch_frame_free(&ctx_p);
        return -1;
    }

    return 0;
}

abcdk_torch_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align)
{
    abcdk_torch_frame_t *ctx = NULL;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_cuda_frame_reset(&ctx,width,height,pixfmt,align);
    if(chk != 0)
        return NULL;

    return ctx;
}

#else // __cuda_cuda_h__

int abcdk_cuda_frame_reset(abcdk_torch_frame_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_torch_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__

__END_DECLS