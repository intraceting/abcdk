/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/frame.h"

static void _abcdk_media_frame_buffer_free_cb(void **ptr, int size)
{
    abcdk_heap_freep(ptr);
}

static int _abcdk_media_frame_buffer_alloc_cb(void **ptr, int size)
{
    *ptr = abcdk_heap_alloc(size);
    if (*ptr)
        return 0;

    return -1;
}

static void _abcdk_media_frame_clear(abcdk_media_frame_t *ctx)
{
    if (ctx->buffer_free_cb)
        ctx->buffer_free_cb(&ctx->buf_ptr, ctx->buf_size);

    ctx->data[0] = ctx->data[1] = ctx->data[2] = ctx->data[3] = NULL;
    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->width = -1;
    ctx->height = -1;
    ctx->pixfmt = ABCDK_MEDIA_PIXFMT_NONE;
    ctx->buf = NULL;
    ctx->dts = -1;
    ctx->pts = -1;
}

void abcdk_media_frame_free(abcdk_media_frame_t **ctx)
{
    abcdk_media_frame_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    _abcdk_media_frame_clear(ctx_p);

    abcdk_heap_free(ctx_p);
}

abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag)
{
    abcdk_media_frame_t *ctx;

    assert(tag == ABCDK_MEDIA_TAG_HOST || tag == ABCDK_MEDIA_TAG_CUDA);

    ctx = (abcdk_media_frame_t *)abcdk_heap_alloc(sizeof(abcdk_media_frame_t));
    if (!ctx)
        return NULL;
    
    _abcdk_media_frame_clear(ctx);

    ctx->tag = tag;
    ctx->buffer_free_cb = _abcdk_media_frame_buffer_free_cb;
    ctx->buffer_alloc_cb = _abcdk_media_frame_buffer_alloc_cb;

    return ctx;
}

int abcdk_media_frame_reset(abcdk_media_frame_t *ctx, int width, int height, int pixfmt, int align)
{
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt > 0);
    assert(ctx->tag == ABCDK_MEDIA_TAG_HOST || ctx->tag == ABCDK_MEDIA_TAG_CUDA);

    if(ctx->width == width || ctx->height == height || ctx->pixfmt == pixfmt)
        return 0;

    _abcdk_media_frame_clear(ctx);

    chk = abcdk_media_image_fill_stride(ctx->stride, width, pixfmt, align);
    if (chk <= 0)
        goto ERR;

    ctx->buf_size = abcdk_media_image_size(ctx->stride,height,pixfmt);
    if(ctx->buf_size <= 0)
        goto ERR;

    chk = ctx->buffer_alloc_cb(&ctx->buf_ptr,ctx->buf_size);
    if (chk != 0)
        goto ERR;

    chk = abcdk_media_image_fill_pointer(ctx->data, ctx->stride, height, pixfmt, ctx->buf_ptr);
    assert(chk == ctx->buf_size);

    ctx->width = width;
    ctx->height = height;
    ctx->pixfmt = pixfmt;

    return 0;

ERR:

    _abcdk_media_frame_clear(ctx);

    return -1;
}

abcdk_media_frame_t *abcdk_media_frame_create(int width, int height, int pixfmt, int align)
{
    abcdk_media_frame_t *ctx;
    int chk;

    assert(width > 0 && height > 0 && pixfmt > 0);

    ctx = abcdk_media_frame_alloc(ABCDK_MEDIA_TAG_HOST);
    if (!ctx)
        return NULL;

    chk = abcdk_media_frame_reset(ctx, width, height, pixfmt, 1);
    if(chk != 0)
    {
        abcdk_media_frame_free(&ctx);
        return NULL;
    }

    return ctx;
}

int abcdk_media_frame_save(const char *dst, const abcdk_media_frame_t *src)
{
    int src_bit_depth;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);

    src_bit_depth = abcdk_media_pixfmt_channels(src->pixfmt) * 8;

    assert(src_bit_depth == 24 || src_bit_depth == 32);

    /*BMP图像默认是倒投影存储。这里高度传入负值，使图像正投影存储。*/
    chk = abcdk_bmp_save_file(dst, src->data[0], src->stride[0], src->width, -src->height, src_bit_depth);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_media_frame_copy(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    int chk;
    
    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);
    assert(dst->width ==  src->width);
    assert(dst->height ==  src->height);
    assert(dst->pixfmt ==  src->pixfmt);

    /*复制图像数据。*/
    abcdk_media_image_copy(dst->data, dst->stride, (const uint8_t **)src->data, src->stride, src->width, src->height, src->pixfmt);
    
    /*复制其它数据。*/
    dst->dts = src->dts;
    dst->pts = src->pts;

    return 0;
}

abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src)
{
    abcdk_media_frame_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);

    dst = abcdk_media_frame_create(src->width, src->height, src->pixfmt, 1);
    if(!dst)
        return NULL;
    
    abcdk_media_frame_copy(dst,src);

    return dst;
}

abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt)
{
    abcdk_media_frame_t *dst;
    int chk;

    assert(src_data != NULL && src_stride != NULL && src_width > 0 && src_height > 0 && src_pixfmt > 0);

    dst = abcdk_media_frame_create(src_width, src_height, src_pixfmt, 1);
    if (!dst)
        return NULL;

    abcdk_media_image_copy(dst->data, dst->stride, src_data, src_stride, src_width, src_height, src_pixfmt);

    return dst;
}
