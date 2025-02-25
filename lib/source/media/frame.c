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
    abcdk_object_unref(&ctx_p->private_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag)
{
    abcdk_media_frame_t *ctx;

    assert(tag == ABCDK_MEDIA_FRAME_TAG_HOST ||
           tag == ABCDK_MEDIA_FRAME_TAG_CUDA);

    ctx = (abcdk_media_frame_t*)abcdk_heap_alloc(sizeof(abcdk_media_frame_t));
    if(!ctx)
        return NULL;

    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->pixfmt = ABCDK_MEDIA_PIXFMT_NONE;
    ctx->width = -1;
    ctx->height = -1;
    ctx->dts = -1;
    ctx->pts = -1;
    ctx->tag = tag;

    return ctx;
}

abcdk_media_frame_t *abcdk_media_frame_alloc2(int width, int height, int pixfmt, int align)
{
    abcdk_media_frame_t *ctx;
    int buf_size,chk_size;
    int block;

    assert(width > 0 && height > 0 && pixfmt > 0);

    ctx = abcdk_media_frame_alloc(ABCDK_MEDIA_FRAME_TAG_HOST);
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

    ctx->width = width;
    ctx->height = height;
    ctx->pixfmt = pixfmt;

    return ctx;

ERR:

    abcdk_media_frame_free(&ctx);
    return NULL;
}

abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src)
{
    abcdk_media_frame_t *dst;

    assert(src != NULL);

    dst = abcdk_media_frame_alloc2(src->width, src->height, src->pixfmt, 1);
    if (!dst)
        return NULL;

    abcdk_media_image_copy(dst->data, dst->stride, (const uint8_t **)src->data, src->stride, src->width, src->height, src->pixfmt);

    return dst;
}

abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt)
{
    abcdk_media_frame_t *dst;

    assert(src_data != NULL && src_stride != NULL && src_width > 0 && src_height > 0 && src_pixfmt > 0);

    dst = abcdk_media_frame_alloc2(src_width, src_height, src_pixfmt, 1);
    if (!dst)
        return NULL;

    abcdk_media_image_copy(dst->data, dst->stride, src_data, src_stride, src_width, src_height, src_pixfmt);

    return dst;
}

int abcdk_media_frame_save(const char *dst, const abcdk_media_frame_t *src)
{
    int src_bit_depth;
    int chk;

    assert(dst != NULL && src != NULL);

    src_bit_depth = abcdk_media_pixfmt_channels(src->pixfmt) * 8;

    assert(src_bit_depth == 24 || src_bit_depth == 32);

    /*BMP图像默认是倒投影存储。这里高度传入负值，使图像正投影存储。*/
    chk = abcdk_bmp_save_file(dst, src->data[0], src->stride[0], src->width, -src->height, src_bit_depth);
    if (chk != 0)
        return -1;

    return 0;
}
