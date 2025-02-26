/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/jpeg.h"
#include "jpeg_decoder_general.cu.hxx"
#include "jpeg_decoder_aarch64.cu.hxx"
#include "jpeg_encoder_general.cu.hxx"
#include "jpeg_encoder_aarch64.cu.hxx"

#ifdef __cuda_cuda_h__

/**JPEG编解码器。*/
typedef struct _abcdk_cuda_jpeg
{
    /** !0 编码，0 解码。*/
    int encode;

    /**编码器。*/
    abcdk::cuda::jpeg::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::cuda::jpeg::decoder *decoder_ctx;

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
        abcdk::cuda::jpeg::encoder_general::destory(&ctx_p->encoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::encoder_aarch64::destory(&ctx_p->encoder_ctx);
#endif //__x86_64__ || __aarch64__
    }
    else
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::decoder_general::destory(&ctx_p->decoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::decoder_aarch64::destory(&ctx_p->decoder_ctx);
#endif //__x86_64__ || __aarch64__
    }

    abcdk_heap_free(ctx_p);
}

abcdk_cuda_jpeg_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_cuda_jpeg_t *ctx;
    int chk;

    assert(cuda_ctx != NULL);

    ctx = (abcdk_cuda_jpeg_t *)abcdk_heap_alloc(sizeof(abcdk_cuda_jpeg_t));
    if (!ctx)
        return NULL;

    if (ctx->encode = encode)
    {
#ifdef __x86_64__
        ctx->encoder_ctx = abcdk::cuda::jpeg::encoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->encoder_ctx = abcdk::cuda::jpeg::encoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx->encoder_ctx)
            goto ERR;

        chk = ctx->encoder_ctx->open(cfg);
        if (chk != 0)
            goto ERR;
    }
    else
    {
#ifdef __x86_64__
        ctx->decoder_ctx = abcdk::cuda::jpeg::decoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->decoder_ctx = abcdk::cuda::jpeg::decoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx->decoder_ctx)
            goto ERR;

        chk = ctx->decoder_ctx->open(cfg);
        if (chk != 0)
            goto ERR;
    }

    return ctx;
    
ERR:

    abcdk_cuda_jpeg_destroy(&ctx);

    return NULL;
}

abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_cuda_jpeg_t *ctx, const abcdk_media_frame_t *src)
{
    abcdk_object_t *dst;
    abcdk_media_frame_t *tmp_src = NULL;
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

    return ctx->encoder_ctx->update(src);
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_cuda_jpeg_t *ctx, const char *dst, const abcdk_media_frame_t *src)
{
    int chk;

    assert(ctx != NULL && dst != NULL && src != NULL);

    ABCDK_ASSERT(ctx->encode, "解码器不能用于编码。");

    chk = ctx->encoder_ctx->update(dst,src);
    if(chk != 0)
        return -1;

    return 0;
}

abcdk_media_frame_t *abcdk_cuda_jpeg_decode(abcdk_cuda_jpeg_t *ctx, const void *src, int src_size)
{
    assert(ctx != NULL && src != NULL && src_size > 0);

    ABCDK_ASSERT(!ctx->encode, "编码器不能用于解码。");

    return ctx->decoder_ctx->update(src,src_size);
}

abcdk_media_frame_t *abcdk_cuda_jpeg_decode_from_file(abcdk_cuda_jpeg_t *ctx,const void *src)
{
    assert(ctx != NULL && src != NULL);

    ABCDK_ASSERT(!ctx->encode, "编码器不能用于解码。");

    return ctx->decoder_ctx->update(src);
}

#else // __cuda_cuda_h__

void abcdk_cuda_jpeg_destroy(abcdk_cuda_jpeg_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return;
}

abcdk_cuda_jpeg_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}


abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_cuda_jpeg_t *ctx, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_cuda_jpeg_t *ctx, const char *dst, const abcdk_media_frame_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_frame_t *abcdk_cuda_jpeg_decode(abcdk_cuda_jpeg_t *ctx, const void *src, int src_size)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_media_frame_t *abcdk_cuda_jpeg_decode_from_file(abcdk_cuda_jpeg_t *ctx,const void *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__
