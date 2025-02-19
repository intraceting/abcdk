/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/video.h"
#include "video_decoder_ffnv.cu.hxx"
#include "video_decoder_aarch64.cu.hxx"
#include "video_encoder_ffnv.cu.hxx"
#include "video_encoder_aarch64.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

/**JPEG编解码器。*/
typedef struct _abcdk_cuda_video
{
    /** !0 编码，0 解码。*/
    int encode;

    /**编码器。*/
    abcdk::cuda::video::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::cuda::video::decoder *decoder_ctx;

} abcdk_cuda_video_t;

void abcdk_cuda_video_destroy(abcdk_cuda_video_t **ctx)
{
    abcdk_cuda_video_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->encode)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::video::encoder_ffnv::destory(&ctx_p->encoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::video::encoder_aarch64::destory(&ctx_p->encoder_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::cuda::video::decoder_ffnv::destory(&ctx_p->decoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::video::decoder_aarch64::destory(&ctx_p->decoder_ctx);
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }

    abcdk_heap_free(ctx_p);
}

abcdk_cuda_video_t *abcdk_cuda_video_create(int encode, abcdk_option_t *cfg)
{
    abcdk_cuda_video_t *ctx;
    int chk;

    ctx = (abcdk_cuda_video_t *)abcdk_heap_alloc(sizeof(abcdk_cuda_video_t));
    if (!ctx)
        return NULL;

    if (ctx->encode = encode)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx->encoder_ctx = abcdk::cuda::video::encoder_ffnv::create();
#elif defined(__aarch64__)
        ctx->encoder_ctx = abcdk::cuda::video::encoder_aarch64::create();
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx->encoder_ctx)
            goto ERR;

        chk = ctx->encoder_ctx->open(cfg);
        if (chk != 0)
            goto ERR;
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        ctx->decoder_ctx = abcdk::cuda::video::decoder_ffnv::create();
#elif defined(__aarch64__)
        ctx->decoder_ctx = abcdk::cuda::video::decoder_aarch64::create();
#endif //FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!ctx->decoder_ctx)
            goto ERR;

        chk = ctx->decoder_ctx->open(cfg);
        if (chk != 0)
            goto ERR;
    }

    return ctx;
ERR:

    abcdk_cuda_video_destroy(&ctx);

    return NULL;
}

int abcdk_cuda_video_sync(abcdk_cuda_video_t *ctx,AVCodecContext *opt)
{
    int chk;

    assert(ctx != NULL && opt != NULL);

    if (ctx->encode)
    {
        chk = ctx->encoder_ctx->sync(opt);
    }
    else
    {
        chk = ctx->decoder_ctx->sync(opt);
    }

    return chk;
}

int abcdk_cuda_video_encode(abcdk_cuda_video_t *ctx,AVPacket **dst, const AVFrame *src)
{
    AVFrame *tmp_src = NULL;
    int src_in_host;
    int chk;

    assert(ctx != NULL && dst != NULL);

    ABCDK_ASSERT(ctx->encode, "解码器不能用于编码。");

    if (src)
    {
        src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);

        if (src_in_host)
        {
            tmp_src = abcdk_cuda_avframe_clone(0, src);
            if (!tmp_src)
                return -1;

            chk = abcdk_cuda_video_encode(ctx, dst, tmp_src);
            av_frame_free(&tmp_src);

            return chk;
        }
    }

    return ctx->encoder_ctx->update(dst,src);
}

int abcdk_cuda_video_decode(abcdk_cuda_video_t *ctx,AVFrame **dst, const AVPacket *src)
{
    assert(ctx != NULL && dst != NULL);

    ABCDK_ASSERT(!ctx->encode, "编码器不能用于解码。");

    return ctx->decoder_ctx->update(dst,src);
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__
