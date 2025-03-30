/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

void abcdk_torch_image_free_cuda(abcdk_torch_image_t **ctx)
{
    abcdk_torch_image_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    abcdk_torch_free_cuda(&ctx_p->private_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_image_t *abcdk_torch_image_alloc_cuda()
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
    ctx->tag = ABCDK_TORCH_TAG_CUDA;
    ctx->private_ctx = NULL;

    return ctx;
}

int abcdk_torch_image_reset_cuda(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_torch_image_t *ctx_p;
    int buf_size;
    int chk;

    assert(ctx != NULL && width > 0 && height > 0 && pixfmt >= 0);

    ctx_p = *ctx;

    if (!ctx_p)
    {
        *ctx = abcdk_torch_image_alloc_cuda();
        if (!*ctx)
            return -1;

        chk = abcdk_torch_image_reset_cuda(ctx, width, height, pixfmt, align);
        if (chk != 0)
            abcdk_torch_image_free_cuda(ctx);

        return chk;
    }

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    if (ctx_p->width == width && ctx_p->height == height && ctx_p->pixfmt == pixfmt)
        return 0;

    abcdk_torch_free_cuda(&ctx_p->private_ctx);

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

    ctx_p->private_ctx = abcdk_torch_alloc_z_cuda(buf_size);
    if (!ctx_p->private_ctx)
        return -1;

    chk = abcdk_torch_imgutil_fill_pointer(ctx_p->data, ctx_p->stride, height, pixfmt, ctx_p->private_ctx);
    assert(chk == buf_size);

    ctx_p->width = width;
    ctx_p->height = height;
    ctx_p->pixfmt = pixfmt;

    return 0;
}

abcdk_torch_image_t *abcdk_torch_image_create_cuda(int width, int height, int pixfmt, int align)
{
    abcdk_torch_image_t *ctx = NULL;
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_torch_image_reset_cuda(&ctx, width, height, pixfmt, align);
    if (chk != 0)
        return NULL;

    return ctx;
}

int abcdk_torch_image_copy_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    int chk;

    assert(dst != NULL && src != NULL);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST || dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_HOST || src->tag == ABCDK_TORCH_TAG_CUDA);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->pixfmt == src->pixfmt);

    /*复制图像数据。*/
    chk = abcdk_torch_imgutil_copy_cuda(dst->data, dst->stride, (dst->tag == ABCDK_TORCH_TAG_HOST),
                                        (const uint8_t **)src->data, src->stride, (src->tag == ABCDK_TORCH_TAG_HOST),
                                        src->width, src->height, src->pixfmt);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_torch_image_copy_plane_cuda(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk, chk_stride, chk_height;

    assert(dst != NULL && dst_plane >= 0);
    assert(dst->tag == ABCDK_TORCH_TAG_HOST || dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src_data != NULL && src_stride > 0);

    chk_stride = abcdk_torch_imgutil_fill_stride(real_stride, dst->width, dst->pixfmt, 1);
    chk_height = abcdk_torch_imgutil_fill_height(real_height, dst->height, dst->pixfmt);
    chk = ABCDK_MIN(chk_stride, chk_height);

    assert(dst_plane < chk);

    abcdk_torch_memcpy_2d_cuda(dst->data[dst_plane], dst->stride[dst_plane], 0, 0, (dst->tag == ABCDK_TORCH_TAG_HOST),
                               src_data, src_stride, 0, 0, 1,
                               real_stride[dst_plane], real_height[dst_plane]);

    return 0;
}

abcdk_torch_image_t *abcdk_torch_image_clone_cuda(int dst_in_host, const abcdk_torch_image_t *src)
{
    abcdk_torch_image_t *dst;
    int chk;

    assert(src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_HOST || src->tag == ABCDK_TORCH_TAG_CUDA);

    if (dst_in_host)
        dst = abcdk_torch_image_create_host(src->width, src->height, src->pixfmt, 1);
    else
        dst = abcdk_torch_image_create_cuda(src->width, src->height, src->pixfmt, 1);

    if (!dst)
        return NULL;

    chk = abcdk_torch_image_copy_cuda(dst, src);
    if (chk != 0)
    {
        abcdk_torch_image_free_cuda(&dst);
        return NULL;
    }

    return dst;
}

#else //__cuda_cuda_h__

void abcdk_torch_image_free_cuda(abcdk_torch_image_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return;
}

abcdk_torch_image_t *abcdk_torch_image_alloc_cuda()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_image_reset_cuda(abcdk_torch_image_t **ctx, int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_image_t *abcdk_torch_image_create_cuda(int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_image_copy_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_image_copy_plane_cuda(abcdk_torch_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_image_t *abcdk_torch_image_clone_cuda(int dst_in_host, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_image_convert_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_image_dump_cuda(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}


#endif //__cuda_cuda_h__

__END_DECLS