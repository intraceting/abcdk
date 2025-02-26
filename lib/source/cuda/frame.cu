/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/frame.h"

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_frame_buffer_free_cb(void **ptr, int size)
{
    abcdk_cuda_free(ptr);
}

static int _abcdk_cuda_frame_buffer_alloc_cb(void **ptr, int size)
{
    *ptr = abcdk_cuda_alloc(size);
    if (*ptr)
        return 0;

    return -1;
}

abcdk_media_frame_t *abcdk_cuda_frame_alloc()
{
    abcdk_media_frame_t *ctx;
    
    ctx = abcdk_media_frame_alloc(ABCDK_MEDIA_TAG_CUDA);
    if(!ctx)
        return NULL;

    ctx->buffer_free_cb = _abcdk_cuda_frame_buffer_free_cb;
    ctx->buffer_alloc_cb = _abcdk_cuda_frame_buffer_alloc_cb;

    return ctx;
}

int abcdk_cuda_frame_reset(abcdk_media_frame_t *ctx, int width, int height, int pixfmt, int align)
{
    return abcdk_media_frame_reset(ctx, width, height, pixfmt, align);
}

abcdk_media_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align)
{
    abcdk_media_frame_t *ctx;
    int chk;

    assert(width > 0 && height > 0 && pixfmt > 0);

    ctx = abcdk_cuda_frame_alloc();
    if (!ctx)
        return NULL;

    chk = abcdk_cuda_frame_reset(ctx, width, height, pixfmt, 1);
    if(chk != 0)
    {
        abcdk_media_frame_free(&ctx);
        return NULL;
    }

    return ctx;
}


int abcdk_cuda_frame_copy(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    int dst_in_host, src_in_host;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST || dst->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    dst_in_host = (dst->tag == ABCDK_MEDIA_TAG_HOST);
    src_in_host = (src->tag == ABCDK_MEDIA_TAG_HOST);

    /*复制图像数据。*/
    chk = abcdk_cuda_image_copy(dst->data, dst->linesize, dst_in_host,
                                (const uint8_t **)src->data, src->st, src_in_host,
                                src->width, src->height, src->format);
    if (chk != 0)
        return -1;
    
    /*复制其它数据。*/
    dst->dts = src->dts;
    dst->pts = src->pts;

    return 0;
}

abcdk_media_frame_t *abcdk_cuda_frame_clone(int dst_in_host, const abcdk_media_frame_t *src)
{
    abcdk_media_frame_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);

    if (dst_in_host)
        dst = abcdk_media_frame_create(src->width, src->height, src->format, 1);
    else
        dst = abcdk_cuda_frame_create(src->width, src->height, src->format, 1);
    if (!dst)
        return NULL;

    chk = abcdk_cuda_frame_copy(dst, src);
    if (chk != 0)
    {
        abcdk_media_frame_free(&dst);
        return NULL;
    }

    return dst;
}

abcdk_media_frame_t *abcdk_cuda_frame_clone2(int dst_in_host, const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt)
{
    abcdk_media_frame_t *dst;
    int chk;

    assert(src_data != NULL && src_stride != NULL && src_width > 0 && src_height > 0 && src_pixfmt > 0);

    if (dst_in_host)
        dst = abcdk_media_frame_create(src_width, src_height, src_pixfmt, 1);
    else
        dst = abcdk_cuda_frame_create(src_width, src_height, src_pixfmt, 1);
    if (!dst)
        return NULL;

    abcdk_cuda_image_copy(dst->data, dst->stride,dst_in_host, src_data, src_stride,1, src_width, src_height, src_pixfmt);

    return dst;
}

int abcdk_cuda_frame_save(const char *dst, const abcdk_media_frame_t *src)
{
    abcdk_media_frame_t *tmp_src;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);

    if(src->tag == ABCDK_MEDIA_TAG_CUDA)
    {
        tmp_src = abcdk_cuda_frame_clone(1,src);
        if(!tmp_src)
            return -1;

        chk = abcdk_cuda_frame_save(dst,tmp_src);
        abcdk_media_frame_free(&tmp_src);

        return chk;
    }

    chk = abcdk_media_frame_save(dst,src);
    if(chk != 0)
        return -1;

    return 0;
}

#else //__cuda_cuda_h__

abcdk_media_frame_t *abcdk_cuda_frame_alloc()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_frame_reset(abcdk_media_frame_t *ctx, int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_frame_copy(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_frame_t *abcdk_cuda_frame_clone(int dst_in_host, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_media_frame_t *abcdk_cuda_frame_clone2(int dst_in_host, const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_frame_save(const char *dst, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__