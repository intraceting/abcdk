/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/image.h"

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_image_private_ctx_free_cb(void **ctx)
{
    abcdk_cuda_free(ctx);
}

abcdk_media_image_t *abcdk_cuda_image_alloc()
{
    abcdk_media_image_t *ctx;
    
    ctx = abcdk_media_image_alloc(ABCDK_MEDIA_TAG_CUDA);
    if(!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_image_private_ctx_free_cb;

    return ctx;
}

int abcdk_cuda_image_reset(abcdk_media_image_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_media_image_t *ctx_p;
    int buf_size;
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt >= 0);

    ctx_p = *ctx;

    if (!ctx_p)
    {
        *ctx = abcdk_cuda_image_alloc();
        if (!*ctx)
            return -1;

        chk = abcdk_cuda_image_reset(ctx, width, height, pixfmt, align);
        if (chk != 0)
            abcdk_media_image_free(ctx);

        return chk;
    }
    
    assert(ctx_p->tag == ABCDK_MEDIA_TAG_CUDA);

    if (ctx_p->width == width || ctx_p->height == height || ctx_p->pixfmt == pixfmt)
        return 0;

    if (ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);

    ctx_p->data[0] = ctx_p->data[1] = ctx_p->data[2] = ctx_p->data[3] = NULL;
    ctx_p->stride[0] = ctx_p->stride[1] = ctx_p->stride[2] = ctx_p->stride[3] = -1;
    ctx_p->width = -1;
    ctx_p->height = -1;
    ctx_p->pixfmt = ABCDK_MEDIA_PIXFMT_NONE;

    chk = abcdk_media_imgutil_fill_stride(ctx_p->stride, width, pixfmt, align);
    if (chk <= 0)
        return -1;

    buf_size = abcdk_media_imgutil_size(ctx_p->stride, height, pixfmt);
    if (buf_size <= 0)
        return -1;

    ctx_p->private_ctx = abcdk_cuda_alloc_z(buf_size);
    if (!ctx_p->private_ctx)
        return -1;

    chk = abcdk_media_imgutil_fill_pointer(ctx_p->data, ctx_p->stride, height, pixfmt, ctx_p->private_ctx);
    assert(chk == buf_size);

    ctx_p->width = width;
    ctx_p->height = height;
    ctx_p->pixfmt = pixfmt;

    return 0;
}

abcdk_media_image_t *abcdk_cuda_image_create(int width, int height, int pixfmt, int align)
{
    abcdk_media_image_t *ctx = NULL;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_cuda_image_reset(&ctx, width, height, pixfmt, align);
    if(chk != 0)
        return NULL;
    

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

void abcdk_cuda_image_copy_plane(abcdk_media_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk, chk_stride, chk_height;

    assert(dst != NULL && dst_plane >= 0);
    assert(dst->tag == ABCDK_MEDIA_TAG_HOST || dst->tag == ABCDK_MEDIA_TAG_CUDA);
    assert(src_data != NULL && src_stride > 0);

    chk_stride = abcdk_media_imgutil_fill_stride(real_stride, dst->width, dst->pixfmt, 1);
    chk_height = abcdk_media_imgutil_fill_height(real_height, dst->height, dst->pixfmt);
    chk = ABCDK_MIN(chk_stride, chk_height);

    assert(dst_plane < chk);

    abcdk_cuda_memcpy_2d(dst->data[dst_plane], dst->stride[dst_plane], 0, 0, (dst->tag == ABCDK_MEDIA_TAG_HOST),
                         src_data, src_stride, 0, 0, 1,
                         real_stride[dst_plane], real_height[dst_plane]);
}

abcdk_media_image_t *abcdk_cuda_image_clone(int dst_in_host, const abcdk_media_image_t *src)
{
    abcdk_media_image_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_MEDIA_TAG_HOST || src->tag == ABCDK_MEDIA_TAG_CUDA);

    if(dst_in_host)
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

    if(src->pixfmt != ABCDK_MEDIA_PIXFMT_BGR24)
    {
        tmp_src = abcdk_cuda_image_create(src->width,src->height,ABCDK_MEDIA_PIXFMT_BGR24,4);
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

int abcdk_cuda_image_reset(abcdk_media_image_t **ctx, int width, int height, int pixfmt, int align)
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

void abcdk_cuda_image_copy_plane(abcdk_media_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
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