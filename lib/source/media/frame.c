/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/frame.h"

void abcdk_media_frame_free(abcdk_media_frame_t **ctx)
{
    abcdk_media_frame_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->buf);
    abcdk_object_unref(&ctx_p->hw_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_media_frame_t *abcdk_media_frame_alloc()
{
    abcdk_media_frame_t *ctx;

    ctx = (abcdk_media_frame_t*)abcdk_heap_alloc(sizeof(abcdk_media_frame_t));
    if(!ctx)
        return NULL;

    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->format = -1;
    ctx->width = -1;
    ctx->height = -1;
    ctx->dts = -1;
    ctx->pts = -1;

    return ctx;
}

abcdk_media_frame_t *abcdk_media_frame_alloc2(int width, int height, int pixfmt, int align)
{
    abcdk_media_frame_t *ctx;
    int buf_size,chk_size;
    int block;

    assert(width > 0 && height > 0 && pixfmt > 0);

    ctx = abcdk_media_frame_alloc();
    if (!ctx)
        return NULL;

    block = abcdk_media_image_fill_stride(ctx->stride, width, pixfmt, align);
    if(block <=0)
        goto ERR;

    buf_size = abcdk_media_image_size(ctx->stride,height,pixfmt);
    if(buf_size <=0)
        goto ERR;

    ctx->buf = abcdk_object_alloc2(buf_size);
    if (!ctx->buf)
        goto ERR;

    chk_size = abcdk_media_image_fill_pointer(ctx->data, ctx->stride, height, pixfmt, ctx->buf->pptrs[0]);
    assert(chk_size == buf_size);

    return ctx;

ERR:

    abcdk_media_frame_free(&ctx);
    return NULL;
}

abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src)
{

}


abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt)
{

}