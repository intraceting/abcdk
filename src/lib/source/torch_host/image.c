/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"


void abcdk_torch_image_free_host(abcdk_torch_image_t **ctx)
{
    abcdk_torch_image_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    abcdk_heap_freep(&ctx_p->private_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_image_t *abcdk_torch_image_alloc_host()
{
    abcdk_torch_image_t *ctx;

    ctx = (abcdk_torch_image_t *)abcdk_heap_alloc(sizeof(abcdk_torch_image_t));
    if (!ctx)
        return NULL;

    ctx->data[0] = ctx->data[1] = ctx->data[2] = ctx->data[3] = NULL;
    ctx->stride[0] = ctx->stride[1] = ctx->stride[2] = ctx->stride[3] = -1;
    ctx->width = -1;
    ctx->height = -1;
    ctx->pixfmt = ABCDK_TORCH_PIXFMT_NONE;
    ctx->tag = ABCDK_TORCH_TAG_HOST;
    ctx->private_ctx = NULL;

    return ctx;
}

int abcdk_torch_image_reset_host(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_torch_image_t *ctx_p;
    int buf_size;
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt >= 0);

    ctx_p = *ctx;

    if (!ctx_p)
    {
        *ctx = abcdk_torch_image_alloc_host();
        if (!*ctx)
            return -1;

        chk = abcdk_torch_image_reset_host(ctx, width, height, pixfmt, align);
        if (chk != 0)
            abcdk_torch_image_free_host(ctx);

        return chk;
    }
    
    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    if (ctx_p->width == width && ctx_p->height == height && ctx_p->pixfmt == pixfmt)
        return 0;

    abcdk_heap_freep(&ctx_p->private_ctx);

    ctx_p->data[0] = ctx_p->data[1] = ctx_p->data[2] = ctx_p->data[3] = NULL;
    ctx_p->stride[0] = ctx_p->stride[1] = ctx_p->stride[2] = ctx_p->stride[3] = -1;
    ctx_p->width = -1;
    ctx_p->height = -1;
    ctx_p->pixfmt = ABCDK_TORCH_PIXFMT_NONE;

    chk = abcdk_torch_imgutil_fill_stride(ctx_p->stride, width, pixfmt, align);
    if (chk <= 0)
        return -1;

    buf_size = abcdk_torch_imgutil_size(ctx_p->stride, height, pixfmt);
    if (buf_size <= 0)
        return -1;

    ctx_p->private_ctx = abcdk_heap_alloc(buf_size);
    if (!ctx_p->private_ctx)
        return -1;

    chk = abcdk_torch_imgutil_fill_pointer(ctx_p->data, ctx_p->stride, height, pixfmt, ctx_p->private_ctx);
    assert(chk == buf_size);

    ctx_p->width = width;
    ctx_p->height = height;
    ctx_p->pixfmt = pixfmt;

    return 0;
}

abcdk_torch_image_t *abcdk_torch_image_create_host(int width, int height, int pixfmt, int align)
{
    abcdk_torch_image_t *ctx = NULL;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_torch_image_reset_host(&ctx, width, height, pixfmt, align);
    if(chk != 0)
        return NULL;

    return ctx;
}

void abcdk_torch_image_copy_host(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);
    assert(dst->width ==  src->width);
    assert(dst->height ==  src->height);
    assert(dst->pixfmt ==  src->pixfmt);

    /*复制图像数据。*/
    abcdk_torch_imgutil_copy_host(dst->data, dst->stride, 1, (const uint8_t **)src->data, src->stride, 1, src->width, src->height, src->pixfmt);

}

void abcdk_torch_image_copy_plane_host(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk, chk_stride, chk_height;

    assert(dst != NULL && dst_plane >= 0);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src_data != NULL && src_stride > 0);

    chk_stride = abcdk_torch_imgutil_fill_stride(real_stride, dst->width, dst->pixfmt,1);
    chk_height = abcdk_torch_imgutil_fill_height(real_height, dst->height, dst->pixfmt);
    chk = ABCDK_MIN(chk_stride, chk_height);

    assert(dst_plane < chk);

    abcdk_memcpy_2d(dst->data[dst_plane], dst->stride[dst_plane], 0, 0,
                    src_data, src_stride, 0, 0,
                    real_stride[dst_plane], real_height[dst_plane]);
}

abcdk_torch_image_t *abcdk_torch_image_clone_host(int dst_in_host, const abcdk_torch_image_t *src)
{
    abcdk_torch_image_t *dst;
    int chk;

    assert(dst_in_host != 0 && src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    dst = abcdk_torch_image_create_host(src->width, src->height, src->pixfmt, 1);
    if(!dst)
        return NULL;
    
    abcdk_torch_image_copy_host(dst,src);

    return dst;
}
