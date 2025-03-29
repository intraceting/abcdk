/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/torch/vcodec.h"
#include "abcdk/torch/nvidia.h"
#include "vcodec_decoder_ffnv.hxx"
#include "vcodec_decoder_aarch64.hxx"
#include "vcodec_encoder_ffnv.hxx"
#include "vcodec_encoder_aarch64.hxx"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

/** CUDA视频编/解码器。*/
typedef struct _abcdk_torch_vcodec_cuda
{
    /**是否为编码器。!0 是，0 否。*/
    uint8_t encoder;

    /**编码器。*/
    abcdk::torch_cuda::vcodec::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::torch_cuda::vcodec::decoder *decoder_ctx;

} abcdk_torch_vcodec_cuda_t;

void abcdk_torch_vcodec_free_cuda(abcdk_torch_vcodec_t **ctx)
{
    abcdk_torch_vcodec_t *ctx_p;
    abcdk_torch_vcodec_cuda_t *cu_ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_torch_vcodec_t *)*ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_vcodec_cuda_t *)ctx_p->private_ctx;

    if (cu_ctx_p->encoder)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::torch_cuda::vcodec::encoder_ffnv::destory(&cu_ctx_p->encoder_ctx);
#elif defined(__aarch64__)
        abcdk::torch_cuda::vcodec::encoder_aarch64::destory(&cu_ctx_p->encoder_ctx);
#endif // FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        abcdk::torch_cuda::vcodec::decoder_ffnv::destory(&cu_ctx_p->decoder_ctx);
#elif defined(__aarch64__)
        abcdk::torch_cuda::vcodec::decoder_aarch64::destory(&cu_ctx_p->decoder_ctx);
#endif // FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__
    }

    abcdk_heap_free(cu_ctx_p);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_cuda(int encoder)
{
    abcdk_torch_vcodec_t *ctx;
    abcdk_torch_vcodec_cuda_t *cu_ctx_p;

    ctx = (abcdk_torch_vcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_CUDA;

    /*创建内部对象。*/
    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_torch_vcodec_t));
    if (!ctx->private_ctx)
        goto ERR;

    cu_ctx_p = (abcdk_torch_vcodec_cuda_t *)ctx->private_ctx;

    if (cu_ctx_p->encoder = encoder)
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        cu_ctx_p->encoder_ctx = abcdk::torch_cuda::vcodec::encoder_ffnv::create((CUcontext)(abcdk_torch_context_current_get_cuda()->private_ctx));
#elif defined(__aarch64__)
        cu_ctx_p->encoder_ctx = abcdk::torch_cuda::vcodec::encoder_aarch64::create((CUcontext)(abcdk_torch_context_current_get_cuda()->private_ctx));
#endif // FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!cu_ctx_p->encoder_ctx)
            goto ERR;
    }
    else
    {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
        cu_ctx_p->decoder_ctx = abcdk::torch_cuda::vcodec::decoder_ffnv::create((CUcontext)(abcdk_torch_context_current_get_cuda()->private_ctx));
#elif defined(__aarch64__)
        cu_ctx_p->decoder_ctx = abcdk::torch_cuda::vcodec::decoder_aarch64::create((CUcontext)(abcdk_torch_context_current_get_cuda()->private_ctx));
#endif // FFNV_CUDA_DYNLINK_LOADER_H || __aarch64__

        if (!cu_ctx_p->decoder_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_torch_vcodec_free_cuda(&ctx);

    return NULL;
}

int abcdk_torch_vcodec_start_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_vcodec_param_t *param)
{
    abcdk_torch_vcodec_cuda_t *cu_ctx_p;
    int chk;

    assert(ctx != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_vcodec_cuda_t *)ctx->private_ctx;

    if (cu_ctx_p->encoder)
    {
        chk = cu_ctx_p->encoder_ctx->open(param);
        if (chk != 0)
            return -1;
    }
    else
    {
        chk = cu_ctx_p->decoder_ctx->open(param);
        if (chk != 0)
            return -1;
    }

    return 0;
}

int abcdk_torch_vcodec_encode_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
{
    abcdk_torch_vcodec_cuda_t *cu_ctx_p;

    assert(ctx != NULL && dst != NULL && src != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->img->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_vcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(cu_ctx_p->encoder, TT("解码器不能用于编码。"));

    return cu_ctx_p->encoder_ctx->update(dst, src);
}

int abcdk_torch_vcodec_decode_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
{
    abcdk_torch_vcodec_cuda_t *cu_ctx_p;

    assert(ctx != NULL && dst != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_vcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(!cu_ctx_p->encoder, TT("编码器不能用于解码。"));

    return cu_ctx_p->decoder_ctx->update(dst, src);
}

#ifdef AVCODEC_AVCODEC_H

int abcdk_torch_vcodec_encode_to_ffmpeg_cuda(abcdk_torch_vcodec_t *ctx, AVPacket **dst, const abcdk_torch_frame_t *src)
{
    abcdk_torch_packet_t *tmp_dst = NULL;
    AVPacket *dst_p;
    int chk;

    assert(ctx != NULL && dst != NULL);

    chk = abcdk_torch_vcodec_encode_cuda(ctx, &tmp_dst, src);
    if (chk > 0)
    {
        dst_p = *dst;

        if (dst_p)
            av_packet_unref(dst_p);
        else
            dst_p = *dst = av_packet_alloc();

        if (!dst_p)
        {
            abcdk_torch_packet_free(&tmp_dst);
            return -1;
        }

        av_new_packet(dst_p, tmp_dst->size);
        memcpy(dst_p->data, tmp_dst->data, tmp_dst->size);
    }

    abcdk_torch_packet_free(&tmp_dst);

    return chk;
}

int abcdk_torch_vcodec_decode_from_ffmpeg_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const AVPacket *src)
{
    abcdk_torch_packet_t tmp_src = {0};

    assert(ctx != NULL && dst != NULL);

    if (src)
    {
        tmp_src.data = src->data;
        tmp_src.size = src->size;
        tmp_src.pts = src->pts;
    }

    return abcdk_torch_vcodec_decode_cuda(ctx, dst, (src ? &tmp_src : NULL));
}

#endif // AVCODEC_AVCODEC_H

#else //__cuda_cuda_h__

void abcdk_torch_vcodec_free_cuda(abcdk_torch_vcodec_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return;
}

abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc_cuda(int encode)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_vcodec_start_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_vcodec_param_t *param)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_vcodec_encode_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_vcodec_decode_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

#ifdef AVCODEC_AVCODEC_H

int abcdk_torch_vcodec_encode_to_ffmpeg_cuda(abcdk_torch_vcodec_t *ctx, AVPacket **dst, const abcdk_torch_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

int abcdk_torch_vcodec_decode_from_ffmpeg_cuda(abcdk_torch_vcodec_t *ctx, abcdk_torch_frame_t **dst, const AVPacket *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含FFmpeg工具。"));
    return -1;
}

#endif // AVCODEC_AVCODEC_H

#endif //__cuda_cuda_h__

__END_DECLS
