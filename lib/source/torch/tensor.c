/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tensor.h"

static void _abcdk_torch_tensor_private_ctx_free_cb(void **ctx)
{
    abcdk_heap_freep(ctx);
}

void abcdk_torch_tensor_free(abcdk_torch_tensor_t **ctx)
{
    abcdk_torch_tensor_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if(ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);


    abcdk_heap_free(ctx_p);
}

abcdk_torch_tensor_t *abcdk_torch_tensor_alloc(uint32_t tag)
{
    abcdk_torch_tensor_t *ctx;
    int buf_size = 0;
    void *buf_ptr = NULL;

    assert(tag == ABCDK_TORCH_TAG_HOST || tag == ABCDK_TORCH_TAG_CUDA);

    ctx = (abcdk_torch_tensor_t *)abcdk_heap_alloc(sizeof(abcdk_torch_tensor_t));
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_torch_tensor_private_ctx_free_cb;

    return ctx;
}

int abcdk_torch_tensor_reset(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
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
        *ctx = abcdk_torch_tensor_alloc(ABCDK_TORCH_TAG_HOST);
        if(!*ctx)
            return -1;

        chk = abcdk_torch_tensor_reset(ctx, format, block, width, height, depth, cell, align);
        if (chk != 0)
            abcdk_torch_tensor_free(ctx);

        return chk;
    }

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    if(ctx_p->format == format && ctx_p->block == block  && ctx_p->width == width && ctx_p->height == height && ctx_p->depth == depth && ctx_p->cell == cell)
        return 0;

    if(ctx_p->private_ctx_free_cb)
        ctx_p->private_ctx_free_cb(&ctx_p->private_ctx);

    ctx_p->data = NULL;
    ctx_p->format = -1;
    ctx_p->block = ctx_p->width = ctx_p->stride = ctx_p->height = ctx_p->depth = ctx_p->cell = 0;

    ctx_p->stride = abcdk_torch_tenutil_stride(format, width, depth, cell, align);
    buf_size = abcdk_torch_tenutil_size(format, block, width, ctx_p->stride, height, depth);

    ctx_p->private_ctx = abcdk_heap_alloc(buf_size);
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

abcdk_torch_tensor_t *abcdk_torch_tensor_create(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align)
{
    abcdk_torch_tensor_t *ctx = NULL;
    int chk;

    assert(format == ABCDK_TORCH_TENFMT_NCHW || format == ABCDK_TORCH_TENFMT_NHWC);
    assert(block > 0 && width > 0 && height > 0 && depth > 0 && cell);

    chk = abcdk_torch_tensor_reset(&ctx, format, block, width, height, depth, cell, align);
    if (chk != 0)
        return NULL;

    return ctx;
}

void abcdk_torch_tensor_copy(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->format == src->format);

    if (dst->format == ABCDK_TORCH_TENFMT_NHWC)
    {
        abcdk_memcpy_2d(dst->data, dst->stride, 0, 0,
                        src->data, src->stride, 0, 0,
                        src->cell * src->depth * src->width, src->block * src->height);
    }
    else if (dst->format == ABCDK_TORCH_TENFMT_NCHW)
    {
        abcdk_memcpy_2d(dst->data, dst->stride, 0, 0,
                        src->data, src->stride, 0, 0,
                        src->cell * src->width, src->block * src->depth * src->height);
    }
}

abcdk_torch_tensor_t *abcdk_torch_tensor_clone(const abcdk_torch_tensor_t *src)
{
    abcdk_torch_tensor_t *dst;

    assert(src != NULL);

    dst = abcdk_torch_tensor_create(src->format, src->block, src->width, src->height, src->depth, src->cell, 1);
    if (!dst)
        return NULL;

    abcdk_torch_tensor_copy(dst, src);

    return dst;
}
