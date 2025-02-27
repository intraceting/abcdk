/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/image.h"

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_image_buffer_free_cb(void **ptr, int size)
{
    abcdk_cuda_free(ptr);
}

static int _abcdk_cuda_image_buffer_alloc_cb(void **ptr, int size)
{
    *ptr = abcdk_cuda_alloc(size);
    if (*ptr)
        return 0;

    return -1;
}

abcdk_media_image_t *abcdk_cuda_image_alloc()
{
    abcdk_media_image_t *ctx;
    
    ctx = abcdk_media_image_alloc(ABCDK_MEDIA_TAG_CUDA);
    if(!ctx)
        return NULL;

    ctx->buffer_free_cb = _abcdk_cuda_image_buffer_free_cb;
    ctx->buffer_alloc_cb = _abcdk_cuda_image_buffer_alloc_cb;

    return ctx;
}

abcdk_media_image_t *abcdk_cuda_image_create(int width, int height, int pixfmt, int align)
{
    abcdk_media_image_t *ctx;
    int chk;

    assert(width > 0 && height > 0 && pixfmt > 0);

    ctx = abcdk_cuda_image_alloc();
    if (!ctx)
        return NULL;

    chk = abcdk_media_image_reset(ctx, width, height, pixfmt, 1);
    if(chk != 0)
    {
        abcdk_media_image_free(&ctx);
        return NULL;
    }

    return ctx;
}

int abcdk_cuda_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST || dst->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->pixfmt == src->pixfmt);

    /*复制图像数据。*/
    chk = abcdk_cuda_imgutil_copy(dst->data, dst->stride, (dst->tag == ABCDK_MEDIA_TAG_HOST),
                                  (const uint8_t **)src->data, src->stride, (src->tag == ABCDK_MEDIA_TAG_HOST),
                                  src->width, src->height, src->pixfmt);
    if (chk != 0)
        return -1;

    return 0;
}

abcdk_media_image_t *abcdk_cuda_image_clone(int dst_in_host, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);

    if (dst_in_host)
        dst = abcdk_media_image_create(src->width, src->height, src->pixfmt, 1);
    else
        dst = abcdk_cuda_image_create(src->width, src->height, src->pixfmt, 1);

    if (!dst)
        return NULL;

    chk = abcdk_cuda_image_copy(dst, src);
    if (chk != 0)
    {
        abcdk_media_image_free(&dst);
        return NULL;
    }

    return dst;
}


int abcdk_cuda_image_save(const char *dst, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *tmp_src;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);

    if(src->pixfmt != ABCDK_MEDIA_PIXFMT_BGR32)
    {
        tmp_src = abcdk_cuda_image_create(src->width,src->height,ABCDK_MEDIA_PIXFMT_BGR32,1);
        if(!tmp_src)
            return -1;
        
        chk = abcdk_cuda_image_convert(tmp_src,src);

        /*转格式成功后继续执行保存操作。*/
        if(chk == 0)
            chk = abcdk_cuda_image_save(dst,tmp_src);

        abcdk_media_image_free(&tmp_src);
        return chk;
    }

    if(src->tag == ABCDK_MEDIA_TAG_CUDA)
    {
        tmp_src = abcdk_cuda_image_clone(1,src);
        if(!tmp_src)
            return -1;

        chk = abcdk_cuda_image_save(dst,tmp_src);
        abcdk_media_image_free(&tmp_src);
        return chk;
    }

    chk = abcdk_media_image_save(dst,src);
    if(chk != 0)
        return -1;

    return 0;
}

#else //__cuda_cuda_h__

abcdk_media_image_t *abcdk_cuda_image_alloc()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_image_reset(abcdk_media_image_t *ctx, int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_image_t *abcdk_cuda_image_create(int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_image_t *abcdk_cuda_image_clone(int dst_in_host, const abcdk_media_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_image_save(const char *dst, const abcdk_media_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__