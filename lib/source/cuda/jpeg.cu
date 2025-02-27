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

static void _abcdk_cuda_jpeg_private_free_cb(void **ctx, uint8_t encoder)
{
    if (!ctx || !*ctx)
        return;

    if (encoder)
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::encoder_general::destory((abcdk::cuda::jpeg::encoder **)ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::encoder_aarch64::destory((abcdk::cuda::jpeg::encoder **)ctx);
#endif //__x86_64__ || __aarch64__
    }
    else
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::decoder_general::destory((abcdk::cuda::jpeg::decoder **)ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::decoder_aarch64::destory((abcdk::cuda::jpeg::decoder **)ctx);
#endif //__x86_64__ || __aarch64__
    }
}

abcdk_media_jcodec_t *abcdk_cuda_jpeg_create(int encoder,CUcontext cuda_ctx)
{
    abcdk_media_jcodec_t *ctx;

    assert(cuda_ctx != NULL);

    ctx = abcdk_media_jcodec_alloc(ABCDK_MEDIA_TAG_CUDA);
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_jpeg_private_free_cb;

    if (ctx->encoder = encoder)
    {
#ifdef __x86_64__
        ctx->private_ctx = abcdk::cuda::jpeg::encoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->private_ctx = abcdk::cuda::jpeg::encoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx->private_ctx)
            goto ERR;
    }
    else
    {
#ifdef __x86_64__
        ctx->private_ctx = abcdk::cuda::jpeg::decoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx->private_ctx = abcdk::cuda::jpeg::decoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx->private_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_media_jcodec_free(&ctx);

    return NULL;
}

int abcdk_cuda_jpeg_start(abcdk_media_jcodec_t *ctx, abcdk_media_jcodec_param_t *param)
{
    abcdk::cuda::jpeg::encoder *encoder_ctx_p;
    abcdk::cuda::jpeg::decoder *decoder_ctx_p;
    int chk;

    if (ctx->encoder)
    {
        encoder_ctx_p = (abcdk::cuda::jpeg::encoder *)ctx->private_ctx;

        chk = encoder_ctx_p->open(param);
        if (chk != 0)
            return -1;
    }
    else
    {
        decoder_ctx_p = (abcdk::cuda::jpeg::decoder *)ctx->private_ctx;

        chk = decoder_ctx_p->open(param);
        if (chk != 0)
            return -1;
    }

    return 0;
}

abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_media_jcodec_t *ctx, const abcdk_media_image_t *src)
{
    abcdk::cuda::jpeg::encoder *encoder_ctx_p;
    abcdk_media_image_t *tmp_src = NULL;
    abcdk_object_t *dst = NULL;

    assert(ctx != NULL && src != NULL);

    ABCDK_ASSERT(ctx->encoder, "解码器不能用于编码。");

    encoder_ctx_p = (abcdk::cuda::jpeg::encoder *)ctx->private_ctx;

    if (src->tag == ABCDK_MEDIA_TAG_HOST)
    {
        tmp_src = abcdk_cuda_image_clone(0, src);
        if (!tmp_src)
            return NULL;

        dst = abcdk_cuda_jpeg_encode(ctx, tmp_src);
        abcdk_media_image_free(&tmp_src);

        return dst;
    }

    return encoder_ctx_p->update(src);
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_media_jcodec_t *ctx, const char *dst, const abcdk_media_image_t *src)
{
    abcdk::cuda::jpeg::encoder *encoder_ctx_p;
    int chk;

    assert(ctx != NULL && dst != NULL && src != NULL);

    ABCDK_ASSERT(ctx->encoder, "解码器不能用于编码。");

    encoder_ctx_p = (abcdk::cuda::jpeg::encoder *)ctx->private_ctx;

    chk = encoder_ctx_p->update(dst, src);
    if (chk != 0)
        return -1;

    return 0;
}

abcdk_media_image_t *abcdk_cuda_jpeg_decode(abcdk_media_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk::cuda::jpeg::decoder *decoder_ctx_p;

    assert(ctx != NULL && src != NULL && src_size > 0);

    ABCDK_ASSERT(!ctx->encoder, "编码器不能用于解码。");

    decoder_ctx_p = (abcdk::cuda::jpeg::decoder *)ctx->private_ctx;

    return decoder_ctx_p->update(src, src_size);
}

abcdk_media_image_t *abcdk_cuda_jpeg_decode_from_file(abcdk_media_jcodec_t *ctx, const void *src)
{
    abcdk::cuda::jpeg::decoder *decoder_ctx_p;

    assert(ctx != NULL && src != NULL);

    ABCDK_ASSERT(!ctx->encoder, "编码器不能用于解码。");

    decoder_ctx_p = (abcdk::cuda::jpeg::decoder *)ctx->private_ctx;

    return decoder_ctx_p->update(src);
}

#else // __cuda_cuda_h__

abcdk_media_jcodec_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_media_packet_t *abcdk_cuda_jpeg_encode(abcdk_media_jcodec_t *ctx, const abcdk_media_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_media_jcodec_t *ctx, const char *dst, const abcdk_media_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_media_image_t *abcdk_cuda_jpeg_decode(abcdk_media_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_media_image_t *abcdk_cuda_jpeg_decode_from_file(abcdk_media_jcodec_t *ctx, const void *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__
