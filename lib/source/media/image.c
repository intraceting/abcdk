/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/image.h"

static void _abcdk_media_image_buffer_free_cb(void **ptr, int size)
{
    abcdk_heap_freep(ptr);
}

static int _abcdk_media_image_buffer_alloc_cb(void **ptr, int size)
{
    *ptr = abcdk_heap_alloc(size);
    if (*ptr)
        return 0;

    return -1;
}

void abcdk_media_image_free(abcdk_media_image_t **ctx)
{
    abcdk_media_image_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_MEDIA_TAG_HOST || ctx_p->tag == ABCDK_MEDIA_TAG_CUDA);

    if (ctx_p->buffer_free_cb)
        ctx_p->buffer_free_cb(&ctx_p->buf_ptr, ctx_p->buf_size);

    abcdk_heap_free(ctx_p);
}

abcdk_media_image_t *abcdk_media_image_alloc(uint32_t tag)
{
    abcdk_media_image_t *ctx;

    assert(tag == ABCDK_MEDIA_TAG_HOST || tag == ABCDK_MEDIA_TAG_CUDA);

    ctx = (abcdk_media_image_t *)abcdk_heap_alloc(sizeof(abcdk_media_image_t));
    if (!ctx)
        return NULL;

    ctx->data[0] = ctx->data[1] = ctx->data[2] = ctx->data[3] = NULL;
    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->width = -1;
    ctx->height = -1;
    ctx->pixfmt = ABCDK_MEDIA_PIXFMT_NONE;
    ctx->tag = tag;
    ctx->buffer_free_cb = _abcdk_media_image_buffer_free_cb;
    ctx->buffer_alloc_cb = _abcdk_media_image_buffer_alloc_cb;

    return ctx;
}

int abcdk_media_image_reset(abcdk_media_image_t *ctx, int width, int height, int pixfmt, int align)
{
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt >= 0);
    assert(ctx->tag == ABCDK_MEDIA_TAG_HOST || ctx->tag == ABCDK_MEDIA_TAG_CUDA);

    if (ctx->width == width || ctx->height == height || ctx->pixfmt == pixfmt)
        return 0;

    if (ctx->buffer_free_cb)
        ctx->buffer_free_cb(&ctx->buf_ptr, ctx->buf_size);

    ctx->data[0] = ctx->data[1] = ctx->data[2] = ctx->data[3] = NULL;
    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->width = -1;
    ctx->height = -1;
    ctx->pixfmt = ABCDK_MEDIA_PIXFMT_NONE;

    chk = abcdk_media_imgutil_fill_stride(ctx->stride, width, pixfmt, align);
    if (chk <= 0)
        return -1;

    ctx->buf_size = abcdk_media_imgutil_size(ctx->stride, height, pixfmt);
    if (ctx->buf_size <= 0)
        return -1;

    chk = ctx->buffer_alloc_cb(&ctx->buf_ptr, ctx->buf_size);
    if (chk != 0)
        return -1;

    chk = abcdk_media_imgutil_fill_pointer(ctx->data, ctx->stride, height, pixfmt, ctx->buf_ptr);
    assert(chk == ctx->buf_size);

    ctx->width = width;
    ctx->height = height;
    ctx->pixfmt = pixfmt;

    return 0;
}

abcdk_media_image_t *abcdk_media_image_create(int width, int height, int pixfmt, int align)
{
    abcdk_media_image_t *ctx;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    ctx = abcdk_media_image_alloc(ABCDK_MEDIA_TAG_HOST);
    if (!ctx)
        return NULL;

    chk = abcdk_media_image_reset(ctx, width, height, pixfmt, align);
    if(chk != 0)
    {
        abcdk_media_image_free(&ctx);
        return NULL;
    }

    return ctx;
}

int abcdk_media_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src)
{
    int chk;
    
    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);
    assert(dst->width ==  src->width);
    assert(dst->height ==  src->height);
    assert(dst->pixfmt ==  src->pixfmt);

    /*复制图像数据。*/
    abcdk_media_imgutil_copy(dst->data, dst->stride, (const uint8_t **)src->data, src->stride, src->width, src->height, src->pixfmt);

    return 0;
}

abcdk_media_image_t *abcdk_media_image_clone(const abcdk_media_image_t *src)
{
    abcdk_media_image_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);

    dst = abcdk_media_image_create(src->width, src->height, src->pixfmt, 1);
    if(!dst)
        return NULL;
    
    abcdk_media_image_copy(dst,src);

    return dst;
}

int abcdk_media_image_save(const char *dst, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *tmp_src;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST);

    if(src->pixfmt != ABCDK_MEDIA_PIXFMT_BGR32)
    {
        tmp_src = abcdk_media_image_create(src->width,src->height,ABCDK_MEDIA_PIXFMT_BGR32,1);
        if(!tmp_src)
            return -1;
        
        chk = abcdk_media_image_convert(tmp_src,src);

        /*转格式成功后继续执行保存操作。*/
        if(chk == 0)
            chk = abcdk_media_image_save(dst,tmp_src);

        abcdk_media_image_free(&tmp_src);
        return chk;
    }

    /*BMP图像默认是倒投影存储。这里高度传入负值，使图像正投影存储。*/
    chk = abcdk_bmp_save_file(dst, src->data[0], src->stride[0], src->width, -src->height, 32);
    if (chk != 0)
        return -1;

    return 0;
}