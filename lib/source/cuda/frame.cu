/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/frame.h"

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_frame_private_free_cb(void **ctx)
{
    abcdk_media_image_free((abcdk_media_image_t **)ctx);
}

static int _abcdk_cuda_frame_image_upload_cb(void *ctx, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *dst_p = NULL;
    int chk;

    dst_p = (abcdk_media_image_t *)ctx;

    chk = abcdk_media_image_reset(dst_p, src->width, src->height, src->pixfmt, 1);
    if (chk != 0)
        return -1;

    abcdk_cuda_image_copy(dst_p, src);

    return 0;
}

static int _abcdk_cuda_frame_image_download_cb(void *ctx, abcdk_media_image_t **dst)
{
    abcdk_media_image_t *src_p = NULL, *dst_p = NULL;

    src_p = (abcdk_media_image_t *)ctx;

    if (*dst)
        dst_p = (abcdk_media_image_t *)*dst;
    else
        dst_p = *dst = abcdk_cuda_image_create(src_p->width, src_p->height, src_p->pixfmt, 1);

    if (!dst_p)
        return -1;

    abcdk_cuda_image_copy(dst_p, src_p);

    return 0;
}

abcdk_media_frame_t *abcdk_cuda_frame_alloc()
{
    abcdk_media_frame_t *ctx;
    
    ctx = abcdk_media_frame_alloc(ABCDK_MEDIA_TAG_CUDA);
    if(!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_frame_private_free_cb;
    ctx->image_upload_cb = _abcdk_cuda_frame_image_upload_cb;
    ctx->image_download_cb = _abcdk_cuda_frame_image_download_cb;

    /*创建私有环境。*/
    ctx->private_ctx = abcdk_media_image_alloc(ABCDK_MEDIA_TAG_CUDA);
    if (!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_media_frame_free(&ctx);
    return NULL;
}

#else // __cuda_cuda_h__

abcdk_media_frame_t *abcdk_cuda_frame_alloc()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif // __cuda_cuda_h__