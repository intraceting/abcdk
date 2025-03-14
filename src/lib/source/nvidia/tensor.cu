/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/tensor.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

static void _abcdk_cuda_tensor_private_ctx_free_cb(void **ctx)
{
    abcdk_cuda_free(ctx);
}

abcdk_torch_tensor_t *abcdk_cuda_tensor_alloc()
{
    abcdk_torch_tensor_t *ctx;

    ctx = abcdk_torch_tensor_alloc(ABCDK_TORCH_TAG_CUDA);
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_tensor_private_ctx_free_cb;

    return ctx;
}

int abcdk_cuda_tensor_reset(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_torch_tensor_t *ctx_p;
    int buf_size;
    int chk;

    assert(ctx != NULL);
    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(block > 0 && width > 0 && height > 0 && depth > 0 && cell);

    ctx_p = *ctx;

    if(!ctx_p)
    {
        *ctx = abcdk_cuda_tensor_alloc();
        if(!*ctx)
            return -1;

        chk = abcdk_cuda_tensor_reset(ctx, format, block, width, height, depth, cell, align);
        if (chk != 0)
            abcdk_torch_tensor_free(ctx);

        return chk;
    }

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    if(ctx_p->format == format && ctx_p->block == block  && ctx_p->width == width && ctx_p->height == height && ctx_p->depth == depth && ctx_p->cell == cell)
        return 0;

    if(ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);

    ctx_p->data = NULL;
    ctx_p->format = -1;
    ctx_p->block = ctx_p->width = ctx_p->stride = ctx_p->height = ctx_p->depth = ctx_p->cell = 0;

    ctx_p->stride = abcdk_torch_tenutil_stride(format, width, depth, cell, align);
    buf_size = abcdk_torch_tenutil_size(format, block, width, ctx_p->stride, height, depth);

    ctx_p->private_ctx = abcdk_cuda_alloc_z(buf_size);
    if (!ctx_p->private_ctx)
        return -1;

    ctx_p->data = (uint8_t *)ctx_p->private_ctx;
    ctx_p->format = format;
    ctx_p->width = width;
    ctx_p->height = height;
    ctx_p->depth = depth;
    ctx_p->cell = cell;

    return 0;
}

abcdk_torch_tensor_t *abcdk_cuda_tensor_create(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_torch_tensor_t *ctx = NULL;
    int chk;

    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(block > 0 && width > 0 && height > 0 && depth > 0 && cell);

    chk = abcdk_cuda_tensor_reset(&ctx, format, block, width, height, depth, cell, align);
    if (chk != 0)
        return NULL;

    return ctx;
}

void abcdk_cuda_tensor_copy(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST || dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_HOST || src->tag == ABCDK_TORCH_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    if (dst->format == ABCDK_TORCH_TENFMT_NHWC)
    {
        abcdk_cuda_memcpy_2d(dst->data, dst->stride, 0, 0, (dst->tag == ABCDK_TORCH_TAG_HOST),
                             src->data, src->stride, 0, 0, (src->tag == ABCDK_TORCH_TAG_HOST),
                             src->cell * src->depth * src->width, src->block * src->height);
    }
    else if (dst->format == ABCDK_TORCH_TENFMT_NCHW)
    {
        abcdk_cuda_memcpy_2d(dst->data, dst->stride, 0, 0, (dst->tag == ABCDK_TORCH_TAG_HOST),
                             src->data, src->stride, 0, 0, (src->tag == ABCDK_TORCH_TAG_HOST),
                             src->cell * src->width, src->block * src->depth * src->height);
    }
}

void abcdk_cuda_tensor_copy_block(abcdk_torch_tensor_t *dst, int dst_block, const uint8_t *src_data, int src_stride)
{
    size_t dst_off;
    uint8_t *dst_data;

    assert(dst != NULL && dst_block >= 0);
    assert(src_data != NULL && src_stride > 0);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST || dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(dst->block > dst_block);
    assert(dst->stride <= src_stride);

    dst_off = abcdk_torch_tensor_offset(dst->format, dst->block, dst->width, dst->stride, dst->height, dst->depth, dst->cell, dst_block, 0, 0, 0);
    dst_data = ABCDK_PTR2U8PTR(dst->data, dst_off);

    if (dst->format == ABCDK_TORCH_TENFMT_NHWC)
    {
        abcdk_cuda_memcpy_2d(dst_data, dst->stride, 0, 0, (dst->tag == ABCDK_TORCH_TAG_HOST),
                             src_data, src_stride, 0, 0, 1,
                             dst->cell * dst->depth * dst->width, dst->block * dst->height);
    }
    else if (dst->format == ABCDK_TORCH_TENFMT_NCHW)
    {
        abcdk_cuda_memcpy_2d(dst_data, dst->stride, 0, 0, (dst->tag == ABCDK_TORCH_TAG_HOST),
                             src_data, src_stride, 0, 0, 1,
                             dst->cell * dst->width, dst->block * dst->depth * dst->height);
    }
}

abcdk_torch_tensor_t *abcdk_cuda_tensor_clone(int dst_in_host, const abcdk_torch_tensor_t *src)
{
    abcdk_torch_tensor_t *dst;

    assert(src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_HOST || src->tag == ABCDK_TORCH_TAG_CUDA);

    if (dst_in_host)
        dst = abcdk_torch_tensor_create(src->format, src->block, src->width, src->height, src->depth, src->cell, 1);
    else
        dst = abcdk_cuda_tensor_create(src->format, src->block, src->width, src->height, src->depth, src->cell, 1);
    if (!dst)
        return NULL;

    abcdk_cuda_tensor_copy(dst, src);

    return dst;
}


#else //__cuda_cuda_h__


abcdk_torch_tensor_t *abcdk_cuda_tensor_alloc()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_cuda_tensor_reset(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_tensor_t *abcdk_cuda_tensor_create(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

void abcdk_cuda_tensor_copy(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return ;
}

void abcdk_cuda_tensor_copy_block(abcdk_torch_tensor_t *dst, int dst_block, const uint8_t *src_data, int src_stride)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return;
}

abcdk_torch_tensor_t *abcdk_cuda_tensor_clone(int dst_in_host, const abcdk_torch_tensor_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__

__END_DECLS