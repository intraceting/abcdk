/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/ndarray.h"
#include "../generic/invoke.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__


CUmemorytype abcdk_cuda_ndarray_memory_type(const abcdk_ndarray_t *src)
{
    assert(src != NULL);

    if (src->hw_ctx && ABCDK_PTR2I64(src->hw_ctx->pptrs[0],0) == 'cuda')
        return CU_MEMORYTYPE_DEVICE;

    return CU_MEMORYTYPE_UNIFIED;
}

static void _abcdk_cuda_ndarray_free(abcdk_object_t *obj, void *opaque)
{
    abcdk_cuda_free((void **)&obj->pptrs[0]);
}

abcdk_ndarray_t *abcdk_cuda_ndarray_alloc(int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_ndarray_t *ctx;
    int buf_size;
    void *buf_ptr = NULL;

    ctx = abcdk_ndarray_alloc(fmt,block,width,height,depth,cell,align,1);
    if (!ctx)
        return NULL;

    ctx->buf = abcdk_object_alloc2(0);
    if (!ctx->buf)
    {
        abcdk_ndarray_free(&ctx);
        return NULL;
    }

    ctx->hw_ctx = abcdk_object_alloc2(sizeof(int64_t));
    if (!ctx->hw_ctx)
    {
        abcdk_ndarray_free(&ctx);
        return NULL;
    }

    buf_size = abcdk_ndarray_size(ctx);

    buf_ptr = abcdk_cuda_alloc_z(buf_size);
    if(!buf_ptr)
    {
        abcdk_ndarray_free(&ctx);
        return NULL;
    }

    /*标志已经占用。*/
    ABCDK_PTR2I64(ctx->hw_ctx->pptrs[0],0) = 'cuda';
    
    ctx->data = buf_ptr;
    ctx->size = buf_size;

    ctx->buf->pptrs[0] = (uint8_t *)buf_ptr;
    ctx->buf->sizes[0] = buf_size;
    abcdk_object_atfree(ctx->buf, _abcdk_cuda_ndarray_free, NULL);

    return ctx;
}

int abcdk_cuda_ndarray_copy(abcdk_ndarray_t *dst, const abcdk_ndarray_t *src)
{
    int dst_in_host, src_in_host;
    int chk;

    assert(dst != NULL && src != NULL);

    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->fmt == src->fmt);
    assert(dst->fmt == ABCDK_NDARRAY_NCHW || dst->fmt == ABCDK_NDARRAY_NHWC);

    dst_in_host = (abcdk_cuda_ndarray_memory_type(dst) != CU_MEMORYTYPE_DEVICE);
    src_in_host = (abcdk_cuda_ndarray_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if (dst->fmt == ABCDK_NDARRAY_NHWC)
    {
        chk = abcdk_cuda_memcpy_2d(dst->data, dst->stride, 0, 0, dst_in_host,
                                   src->data, src->stride, 0, 0, src_in_host,
                                   src->cell * src->depth * src->width, src->block * src->height);

        if(chk != 0)
            return -1;
    }
    else if (dst->fmt == ABCDK_NDARRAY_NCHW)
    {
        chk = abcdk_cuda_memcpy_2d(dst->data, dst->stride, 0, 0, dst_in_host,
                                   src->data, src->stride, 0, 0, src_in_host,
                                   src->cell * src->width, src->block * src->depth * src->height);

        if (chk != 0)
            return -1;
    }
    else
    {
        return -1;
    }

    return 0;
}

abcdk_ndarray_t *abcdk_cuda_ndarray_clone(int dst_in_host, const abcdk_ndarray_t *src)
{
    abcdk_ndarray_t *dst;
    int chk;

    assert(src != NULL);

    if (dst_in_host)
        dst = abcdk_ndarray_alloc(src->fmt, src->block, src->width, src->height, src->depth, src->cell, 1, 0);
    else
        dst = abcdk_cuda_ndarray_alloc(src->fmt, src->block, src->width, src->height, src->depth, src->cell, 1);
    if (!dst)
        return NULL;

    chk = abcdk_cuda_ndarray_copy(dst, src);
    if (chk != 0)
    {
        abcdk_ndarray_free(&dst);
        return NULL;
    }

    return dst;
}

abcdk_ndarray_t *abcdk_cuda_ndarray_clone2(int dst_in_host,
                                           const uint8_t *src_data, const int src_stride, int src_in_host,
                                           int fmt, size_t block, size_t width, size_t height, size_t depth, size_t cell)
{
    abcdk_ndarray_t *dst, tmp_src = {0};
    int chk;

    assert(src != NULL);

    if (dst_in_host)
        dst = abcdk_ndarray_alloc(fmt, block, width, height, depth, cell, 1, 0);
    else
        dst = abcdk_cuda_ndarray_alloc(fmt, block, width, height, depth, cell, 1);
    if (!dst)
        return NULL;

    tmp_src.fmt = fmt;
    tmp_src.block = block;
    tmp_src.width = width;
    tmp_src.height = height;
    tmp_src.depth = depth;
    tmp_src.cell = cell;
    tmp_src.stride = src_stride;

    tmp_src.data = src_data;
    tmp_src.size = abcdk_ndarray_size(&tmp_src);

    chk = abcdk_cuda_ndarray_copy(dst, &tmp_src);
    if (chk != 0)
        return -1;

    return 0;
}

#endif //__cuda_cuda_h__
