/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/jpeg.h"
#include "jpeg_decode.cu.hxx"
#include "jpeg_encode_x86_64.cu.hxx"
#include "jpeg_encode_aarch64.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

/**JPEG编解码器。*/
typedef struct _abcdk_cuda_jpeg
{
    /** !0 编码，0 解码。*/
    int encode;

    /**编码器。*/
    abcdk::cuda::jpeg::encoder *encodec_ctx;

} abcdk_cuda_jpeg_t;

void abcdk_cuda_jpeg_destroy(abcdk_cuda_jpeg_t **ctx)
{
    abcdk_cuda_jpeg_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->encode)
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::encoder_x86_64::destory(&ctx_p->encodec_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::encoder_aarch64::destory(&ctx_p->encodec_ctx);
#endif //__x86_64__ || __aarch64__
    }
    else
    {
    }

    abcdk_heap_free(ctx_p);
}

abcdk_cuda_jpeg_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg)
{
    abcdk_cuda_jpeg_t *ctx;
    int chk;

    ctx = (abcdk_cuda_jpeg_t *)abcdk_heap_alloc(sizeof(abcdk_cuda_jpeg_t));
    if (!ctx)
        return NULL;

    if (ctx->encode = encode)
    {
#ifdef __x86_64__
        ctx->encodec_ctx = abcdk::cuda::jpeg::encoder_x86_64::create();
#elif defined(__aarch64__)
        ctx->encodec_ctx = abcdk::cuda::jpeg::encoder_aarch64::create();
#endif //__x86_64__ || __aarch64__

        if (!ctx->encodec_ctx)
            goto ERR;

        chk = ctx->encodec_ctx->open(cfg);
        if (chk != 0)
            goto ERR;
    }
    else
    {
    }

    return ctx;
ERR:

    abcdk_cuda_jpeg_destroy(&ctx);

    return NULL;
}

abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_cuda_jpeg_t *ctx, const AVFrame *src)
{
    abcdk_object_t *dst;
    AVFrame *tmp_src = NULL;
    int src_in_host;

    assert(ctx != NULL && src != NULL);

    ABCDK_ASSERT(ctx->encode, "解码器不能用于编码。");

    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

    if(src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(0, src);
        if(!tmp_src)
            return NULL;

        dst = abcdk_cuda_jpeg_encode(ctx,tmp_src);
        av_frame_free(&tmp_src);

        return dst;
    }

    return ctx->encodec_ctx->update(src);
}

AVFrame *abcdk_cuda_jpeg_decode(abcdk_cuda_jpeg_t *ctx, const void *src, int src_size)
{
    assert(ctx != NULL && src != NULL && src_size > 0);

    ABCDK_ASSERT(!ctx->encode, "编码器不能用于解码。");

    return NULL;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__
