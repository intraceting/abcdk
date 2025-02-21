/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/ndarray.h"

size_t abcdk_ndarray_size(abcdk_ndarray_t *ndarray)
{
    size_t size = 0;

    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->stride >= ndarray->width && ndarray->cell > 0);

    if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
    {
        size = ndarray->block * ndarray->depth * ndarray->height * ndarray->stride;
    }
    else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
    {
        size = ndarray->block * ndarray->height * ndarray->stride;
    }

    return size;
}

void abcdk_ndarray_set_stride(abcdk_ndarray_t *ndarray,size_t align)
{
    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->cell > 0);

    if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
    {
        ndarray->stride = abcdk_align(ndarray->width * ndarray->cell, align);
    }
    else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
    {
        ndarray->stride = abcdk_align(ndarray->width * ndarray->depth * ndarray->cell, align);
    }
}

size_t abcdk_ndarray_offset(abcdk_ndarray_t *ndarray, size_t n, size_t x, size_t y, size_t z,int flag)
{
    size_t off = SIZE_MAX;

    assert(ndarray != NULL);
    assert(ndarray->fmt == ABCDK_NDARRAY_NCHW || ndarray->fmt == ABCDK_NDARRAY_NHWC);
    assert(ndarray->block > 0 && ndarray->width > 0 && ndarray->height > 0 && ndarray->depth > 0);
    assert(ndarray->stride >= ndarray->width && ndarray->cell > 0);
    assert(n < ndarray->block && x < ndarray->width && y < ndarray->height && z < ndarray->depth);

    if (flag == ABCDK_NDARRAY_FLIP_H)
    {
        off = abcdk_ndarray_offset(ndarray, n, x, ndarray->height - y - 1, z, 0);
    }
    else if (flag == ABCDK_NDARRAY_FLIP_V)
    {
        off = abcdk_ndarray_offset(ndarray, n, ndarray->width - x - 1, y, z, 0);
    }
    else
    {
        if (ndarray->fmt == ABCDK_NDARRAY_NCHW)
        {
            off = x * ndarray->cell;
            off += y * ndarray->stride;
            off += z * ndarray->height * ndarray->stride;
            off += n * ndarray->depth * ndarray->height * ndarray->stride;
        }
        else if (ndarray->fmt == ABCDK_NDARRAY_NHWC)
        {
            off = x * ndarray->depth * ndarray->cell;
            off += z * ndarray->cell;
            off += y * ndarray->stride;
            off += n * ndarray->height * ndarray->stride;
        }
    }

    return off;
}

void abcdk_ndarray_free(abcdk_ndarray_t **ctx)
{
    abcdk_ndarray_t *ctx_p;

    if(!ctx || !*ctx)
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
    if(no_space)
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