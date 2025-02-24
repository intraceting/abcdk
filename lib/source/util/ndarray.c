/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/ndarray.h"

size_t abcdk_ndarray_size(abcdk_ndarray_t *ctx)
{
    size_t size = 0;

    assert(ctx != NULL);
    assert(ctx->fmt == ABCDK_NDARRAY_NCHW || ctx->fmt == ABCDK_NDARRAY_NHWC);
    assert(ctx->block > 0 && ctx->width > 0 && ctx->height > 0 && ctx->depth > 0);
    assert(ctx->stride >= ctx->width && ctx->cell > 0);

    if (ctx->fmt == ABCDK_NDARRAY_NCHW)
    {
        size = ctx->block * ctx->depth * ctx->height * ctx->stride;
    }
    else if (ctx->fmt == ABCDK_NDARRAY_NHWC)
    {
        size = ctx->block * ctx->height * ctx->stride;
    }

    return size;
}

void abcdk_ndarray_set_stride(abcdk_ndarray_t *ctx, size_t align)
{
    assert(ctx != NULL);
    assert(ctx->fmt == ABCDK_NDARRAY_NCHW || ctx->fmt == ABCDK_NDARRAY_NHWC);
    assert(ctx->block > 0 && ctx->width > 0 && ctx->height > 0 && ctx->depth > 0);
    assert(ctx->cell > 0);

    if (ctx->fmt == ABCDK_NDARRAY_NCHW)
    {
        ctx->stride = abcdk_align(ctx->width * ctx->cell, align);
    }
    else if (ctx->fmt == ABCDK_NDARRAY_NHWC)
    {
        ctx->stride = abcdk_align(ctx->width * ctx->depth * ctx->cell, align);
    }
}

size_t abcdk_ndarray_offset(abcdk_ndarray_t *ctx, size_t n, size_t x, size_t y, size_t z, int flag)
{
    size_t off = SIZE_MAX;

    assert(ctx != NULL);
    assert(ctx->fmt == ABCDK_NDARRAY_NCHW || ctx->fmt == ABCDK_NDARRAY_NHWC);
    assert(ctx->block > 0 && ctx->width > 0 && ctx->height > 0 && ctx->depth > 0);
    assert(ctx->stride >= ctx->width && ctx->cell > 0);
    assert(n < ctx->block && x < ctx->width && y < ctx->height && z < ctx->depth);

    if (flag == ABCDK_NDARRAY_FLIP_H)
    {
        off = abcdk_ndarray_offset(ctx, n, x, ctx->height - y - 1, z, 0);
    }
    else if (flag == ABCDK_NDARRAY_FLIP_V)
    {
        off = abcdk_ndarray_offset(ctx, n, ctx->width - x - 1, y, z, 0);
    }
    else
    {
        if (ctx->fmt == ABCDK_NDARRAY_NCHW)
        {
            off = x * ctx->cell;
            off += y * ctx->stride;
            off += z * ctx->height * ctx->stride;
            off += n * ctx->depth * ctx->height * ctx->stride;
        }
        else if (ctx->fmt == ABCDK_NDARRAY_NHWC)
        {
            off = x * ctx->depth * ctx->cell;
            off += z * ctx->cell;
            off += y * ctx->stride;
            off += n * ctx->height * ctx->stride;
        }
    }

    return off;
}

void *abcdk_ndarray_seek(abcdk_ndarray_t *ctx, size_t n, size_t x, size_t y, size_t z, int flag)
{
    size_t off = abcdk_ndarray_offset(ctx,n,x,y,z,flag);
}

void abcdk_ndarray_free(abcdk_ndarray_t **ctx)
{
    abcdk_ndarray_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->buf);
    abcdk_object_unref(&ctx_p->hw_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_ndarray_t *abcdk_ndarray_alloc(int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align, int no_space)
{
    abcdk_ndarray_t *ctx;
    int buf_size = 0;
    void *buf_ptr = NULL;

    assert(fmt == ABCDK_NDARRAY_NCHW || fmt == ABCDK_NDARRAY_NHWC);
    assert(block > 0 && width > 0 && height > 0 && depth > 0);

    ctx = (abcdk_ndarray_t *)abcdk_heap_alloc(sizeof(abcdk_ndarray_t));
    if (!ctx)
        return NULL;

    ctx->fmt = fmt;
    ctx->block = block;
    ctx->width = width;
    ctx->height = height;
    ctx->depth = depth;
    ctx->stride = 0;
    ctx->cell = cell;
    ctx->data = NULL;
    ctx->size = 0;
    ctx->hw_ctx = NULL;
    ctx->buf = NULL;

    abcdk_ndarray_set_stride(ctx, align);

    /*如果不需要空间直接返回。*/
    if (no_space)
        return ctx;

    buf_size = abcdk_ndarray_size(ctx);

    ctx->buf = abcdk_object_alloc2(buf_size);
    if (!ctx->buf)
    {
        abcdk_ndarray_free(&ctx);
        return NULL;
    }

    ctx->data = ctx->buf->pptrs[0];
    ctx->size = buf_size;

    return ctx;
}

int abcdk_ndarray_copy(abcdk_ndarray_t *dst, const abcdk_ndarray_t *src)
{
    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->fmt == src->fmt);
    assert(dst->fmt == ABCDK_NDARRAY_NCHW || dst->fmt == ABCDK_NDARRAY_NHWC);

    if (dst->fmt == ABCDK_NDARRAY_NHWC)
    {
        abcdk_memcpy_2d(dst->data, dst->stride, 0, 0,
                        src->data, src->stride, 0, 0,
                        src->cell * src->depth * src->width, src->block * src->height);
    }
    else if (dst->fmt == ABCDK_NDARRAY_NCHW)
    {
        abcdk_memcpy_2d(dst->data, dst->stride, 0, 0,
                        src->data, src->stride, 0, 0,
                        src->cell * src->width, src->block * src->depth * src->height);
    }
    else
    {
        return -1;
    }

    return 0;
}

abcdk_ndarray_t *abcdk_ndarray_clone(const abcdk_ndarray_t *src)
{
    abcdk_ndarray_t *dst;
    int chk;

    assert(src != NULL);

    dst = abcdk_ndarray_alloc(src->fmt, src->block, src->width, src->height, src->depth, src->cell, 1, 0);
    if (!dst)
        return NULL;

    chk = abcdk_ndarray_copy(dst, src);
    if (chk != 0)
    {
        abcdk_ndarray_free(&dst);
        return NULL;
    }

    return dst;
}

abcdk_ndarray_t *abcdk_ndarray_clone2(const uint8_t *src_data, const int src_stride,
                                      int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell)
{
    abcdk_ndarray_t *dst, tmp_src = {0};
    int chk;

    assert(src_data != NULL && src_stride >0);

    dst = abcdk_ndarray_alloc(fmt, block, width, height, depth, cell, 1, 0);
    if (!dst)
        return NULL;

    tmp_src.fmt = fmt;
    tmp_src.block = block;
    tmp_src.width = width;
    tmp_src.height = height;
    tmp_src.depth = depth;
    tmp_src.cell = cell;
    tmp_src.stride = src_stride;

    tmp_src.data = (void *)src_data;
    tmp_src.size = abcdk_ndarray_size(&tmp_src);

    chk = abcdk_ndarray_copy(dst, &tmp_src);
    if (chk != 0)
    {
        abcdk_ndarray_free(&dst);
        return NULL;
    }

    return dst;
}