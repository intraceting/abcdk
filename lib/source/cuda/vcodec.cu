/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/vcodec.h"
#include "vcodec_decoder_ffnv.cu.hxx"
#include "vcodec_decoder_aarch64.cu.hxx"
#include "vcodec_encoder_ffnv.cu.hxx"
#include "vcodec_encoder_aarch64.cu.hxx"

#ifdef __cuda_cuda_h__

/** CUDA视频编/解码器。*/
typedef struct _abcdk_cuda_vcodec
{
    /**是否为编码器。!0 是，0 否。*/
    uint8_t encoder;

    /**编码器。*/
    abcdk::cuda::vcodec::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::cuda::vcodec::decoder *decoder_ctx;

} abcdk_cuda_vcodec_t;

static void _abcdk_cuda_vcodec_private_free_cb(void **ctx)
{
    abcdk_cuda_vcodec_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_cuda_vcodec_t *)*ctx;
    *ctx = NULL;

    if (ctx_p->encoder)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::vcodec::encoder_ffnv::destory(&ctx_p->encoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::vcodec::encoder_aarch64::destory(&ctx_p->encoder_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::vcodec::decoder_ffnv::destory(&ctx_p->decoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::vcodec::decoder_aarch64::destory(&ctx_p->decoder_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }

    abcdk_heap_free(ctx_p);
}

abcdk_media_vcodec_t *abcdk_cuda_vcodec_alloc(int encoder,CUcontext cuda_ctx)
{
    abcdk_media_vcodec_t *ctx;
    abcdk_cuda_vcodec_t *ctx_p;

    assert(cuda_ctx != NULL);

    ctx = abcdk_media_vcodec_alloc(ABCDK_MEDIA_TAG_CUDA);
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_vcodec_private_free_cb;

    /*创建内部对象。*/
    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_cuda_vcodec_t));
    if(!ctx->private_ctx)
        goto ERR;

    ctx_p = (abcdk_cuda_vcodec_t *)ctx->private_ctx;
    
    if (ctx_p->encoder = encoder)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx_p->encoder_ctx = abcdk::cuda::vcodec::encoder_ffnv::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx_p->encoder_ctx = abcdk::cuda::vcodec::encoder_aarch64::create(cuda_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx_p->encoder_ctx)
            goto ERR;
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx_p->decoder_ctx = abcdk::cuda::vcodec::decoder_ffnv::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx_p->decoder_ctx = abcdk::cuda::vcodec::decoder_aarch64::create(cuda_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx_p->decoder_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_media_vcodec_free(&ctx);

    return NULL;
}

int abcdk_cuda_vcodec_start(abcdk_media_vcodec_t *ctx, abcdk_media_vcodec_param_t *param)
{
    abcdk_cuda_vcodec_t *ctx_p;
    int chk;

    assert(ctx != NULL);

    ctx_p = (abcdk_cuda_vcodec_t *)ctx->private_ctx;

    if (ctx_p->encoder)
    {
        chk = ctx_p->encoder_ctx->open(param);
        if (chk != 0)
            return -1;
    }
    else
    {
        chk = ctx_p->decoder_ctx->open(param);
        if (chk != 0)
            return -1;
    }

    return 0;
}

int abcdk_cuda_vcodec_encode(abcdk_media_vcodec_t *ctx,abcdk_media_packet_t **dst, const abcdk_media_frame_t *src)
{
    abcdk_cuda_vcodec_t *ctx_p;

    assert(ctx != NULL && dst != NULL);

    ctx_p = (abcdk_cuda_vcodec_t *)ctx->private_ctx;

    ABCDK_ASSERT(ctx_p->encoder, "解码器不能用于编码。");

    return ctx_p->encoder_ctx->update(dst,src);
}

int abcdk_cuda_vcodec_decode(abcdk_media_vcodec_t *ctx,abcdk_media_frame_t **dst, const abcdk_media_packet_t *src)
{
    abcdk_cuda_vcodec_t *ctx_p;

    assert(ctx != NULL && dst != NULL);

    ctx_p = (abcdk_cuda_vcodec_t *)ctx->private_ctx;

    ABCDK_ASSERT(!ctx_p->encoder, "编码器不能用于解码。");

    return ctx_p->decoder_ctx->update(dst, src);
}

#else //__cuda_cuda_h__

void abcdk_cuda_vcodec_destroy(abcdk_media_vcodec_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return ;
}

abcdk_media_vcodec_t *abcdk_cuda_vcodec_create(int encode, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_vcodec_start(abcdk_media_vcodec_t *ctx, abcdk_media_vcodec_param_t *param)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_vcodec_encode(abcdk_media_vcodec_t *ctx,abcdk_media_packet_t **dst, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_vcodec_decode(abcdk_media_vcodec_t *ctx,abcdk_media_frame_t **dst, const abcdk_media_packet_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__
