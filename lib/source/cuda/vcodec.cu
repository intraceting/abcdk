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

static void _abcdk_cuda_vcodec_private_free_cb(void **ctx, uint8_t encoder)
{
    abcdk::cuda::video::encoder *encoder_ctx_p;
    abcdk::cuda::video::decoder *decoder_ctx_p;

    if (!ctx || !*ctx)
        return;

    if (encode)
    {
        encoder_ctx_p = *ctx;
        *ctx = NULL;

#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::video::encoder_ffnv::destory(&encoder_ctx_p);
#elif defined(__aarch64__)
        abcdk::cuda::video::encoder_aarch64::destory(&encoder_ctx_p);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }
    else
    {
        decoder_ctx_p = *ctx;
        *ctx = NULL;

#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::video::decoder_ffnv::destory(&decoder_ctx_p);
#elif defined(__aarch64__)
        abcdk::cuda::video::decoder_aarch64::destory(&decoder_ctx_p);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

    }
}

abcdk_media_vcodec_t *abcdk_cuda_vcodec_alloc(CUcontext cuda_ctx)
{
    abcdk_media_vcodec_t *ctx;
    int chk;

    assert(cuda_ctx != NULL);

    ctx = abcdk_media_vcodec_alloc(ABCDK_MEDIA_TAG_CUDA);
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_vcodec_private_free_cb;
    
    if (ctx->encoder)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx->private_ctx = abcdk::cuda::video::encoder_ffnv::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->private_ctx = abcdk::cuda::video::encoder_aarch64::create(cuda_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx->private_ctx)
            goto ERR;
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx->private_ctx = abcdk::cuda::video::decoder_ffnv::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->private_ctx = abcdk::cuda::video::decoder_aarch64::create(cuda_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx->private_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_media_vcodec_free(&ctx);

    return NULL;
}

int abcdk_cuda_vcodec_start(abcdk_media_vcodec_t *ctx, abcdk_media_vcodec_param_t *param)
{
    abcdk::cuda::video::encoder *encoder_ctx;
    abcdk::cuda::video::decoder *decoder_ctx;
    int chk;

    assert(ctx != NULL);

    if (ctx->encoder)
    {
        encoder_ctx = (abcdk::cuda::video::encoder *)ctx->private_ctx;

        chk = encoder_ctx->open(param);
        if(chk != 0)
            return -1;
    }
    else
    {
        decoder_ctx = (abcdk::cuda::video::decoder *)ctx->private_ctx;

        chk = decoder_ctx->open(param);
        if(chk != 0)
            return -1;
    }

    return 0;
}

int abcdk_cuda_vcodec_encode(abcdk_media_vcodec_t *ctx,abcdk_media_packet_t **dst, const abcdk_media_frame_t *src)
{
    abcdk::cuda::video::encoder *encoder_ctx;
    abcdk_media_frame_t *tmp_src = NULL;
    int src_in_host;
    int chk;

    assert(ctx != NULL && dst != NULL);

    ABCDK_ASSERT(ctx->encode, "解码器不能用于编码。");

    if (src)
    {
        src_in_host = (src->tag == ABCDK_MEDIA_TAG_HOST);

        if (src_in_host)
        {
            tmp_src = abcdk_cuda_frame_clone(0, src);
            if (!tmp_src)
                return -1;

            chk = abcdk_cuda_video_encode(ctx, dst, tmp_src);
            abcdk_media_frame_free(&tmp_src);

            return chk;
        }
    }

    encoder_ctx = (abcdk::cuda::video::encoder *)ctx->private_ctx;

    return encoder_ctx->update(dst,src);
}

int abcdk_cuda_video_decode(abcdk_media_vcodec_t *ctx,abcdk_media_frame_t **dst, const abcdk_media_packet_t *src)
{
    abcdk::cuda::video::decoder *decoder_ctx;

    assert(ctx != NULL && dst != NULL);

    ABCDK_ASSERT(!ctx->encode, "编码器不能用于解码。");

    decoder_ctx = (abcdk::cuda::video::encoder *)ctx->private_ctx;

    return decoder_ctx->update(dst,src);
}

#else //__cuda_cuda_h__

void abcdk_cuda_vcodec_destroy(abcdk_media_vcodec_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return ;
}

abcdk_media_vcodec_t *abcdk_cuda_vcodec_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_vcodec_sync(abcdk_media_vcodec_t *ctx,AVCodecContext *opt)
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
