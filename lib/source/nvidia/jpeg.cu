/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/jpeg.h"
#include "jpeg_decoder_general.cu.hxx"
#include "jpeg_decoder_aarch64.cu.hxx"
#include "jpeg_encoder_general.cu.hxx"
#include "jpeg_encoder_aarch64.cu.hxx"

#ifdef __cuda_cuda_h__

/** JPEG编/解码器。*/
typedef struct _abcdk_cuda_jpeg
{
    /**是否为编码器。!0 是，0 否。*/
    uint8_t encoder;

    /**编码器。*/
    abcdk::cuda::jpeg::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::cuda::jpeg::decoder *decoder_ctx;

} abcdk_cuda_jpeg_t;

static void _abcdk_cuda_jpeg_private_free_cb(void **ctx)
{
    abcdk_cuda_jpeg_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_cuda_jpeg_t *)*ctx;
    *ctx = NULL;

    if (ctx_p->encoder)
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

abcdk_torch_jcodec_t *abcdk_cuda_jpeg_create(int encoder,CUcontext cuda_ctx)
{
    abcdk_torch_jcodec_t *ctx;
    abcdk_cuda_jpeg_t *ctx_p;

    assert(cuda_ctx != NULL);

    ctx = abcdk_torch_jcodec_alloc(ABCDK_TORCH_TAG_CUDA);
    if (!ctx)
        return NULL;

    ctx->private_ctx_free_cb = _abcdk_cuda_jpeg_private_free_cb;

    /*创建内部对象。*/
    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_cuda_jpeg_t));
    if(!ctx->private_ctx)
        goto ERR;

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

    if (ctx_p->encoder = encoder)
    {
#ifdef __x86_64__
        ctx_p->encoder_ctx = abcdk::cuda::jpeg::encoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx_p->encoder_ctx = abcdk::cuda::jpeg::encoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx_p->encoder_ctx)
            goto ERR;
    }
    else
    {
#ifdef __x86_64__
        ctx_p->decoder_ctx = abcdk::cuda::jpeg::decoder_general::create(cuda_ctx);
#elif defined(__aarch64__)
        ctx_p->decoder_ctx = abcdk::cuda::jpeg::decoder_aarch64::create(cuda_ctx);
#endif //__x86_64__ || __aarch64__

        if (!ctx_p->decoder_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_torch_jcodec_free(&ctx);

    return NULL;
}

int abcdk_cuda_jpeg_start(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param)
{
    abcdk_cuda_jpeg_t *ctx_p;
    int chk;

    assert(ctx != NULL);

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

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

abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_cuda_jpeg_t *ctx_p;
    abcdk_torch_image_t *tmp_src = NULL;
    abcdk_object_t *dst = NULL;

    assert(ctx != NULL && src != NULL);

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

    ABCDK_ASSERT(ctx_p->encoder, "解码器不能用于编码。");

    if (src->tag == ABCDK_TORCH_TAG_HOST)
    {
        tmp_src = abcdk_cuda_image_clone(0, src);
        if (!tmp_src)
            return NULL;

        dst = abcdk_cuda_jpeg_encode(ctx, tmp_src);
        abcdk_torch_image_free(&tmp_src);

        return dst;
    }

    return ctx_p->encoder_ctx->update(src);
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_cuda_jpeg_t *ctx_p;
    int chk;

    assert(ctx != NULL && dst != NULL && src != NULL);

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

    ABCDK_ASSERT(ctx_p->encoder, "解码器不能用于编码。");

    chk = ctx_p->encoder_ctx->update(dst, src);
    if (chk != 0)
        return -1;

    return 0;
}

abcdk_torch_image_t *abcdk_cuda_jpeg_decode(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_cuda_jpeg_t *ctx_p;

    assert(ctx != NULL && src != NULL && src_size > 0);

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

    ABCDK_ASSERT(!ctx_p->encoder, "编码器不能用于解码。");

    return ctx_p->decoder_ctx->update(src, src_size);
}

abcdk_torch_image_t *abcdk_cuda_jpeg_decode_from_file(abcdk_torch_jcodec_t *ctx, const void *src)
{
    abcdk_cuda_jpeg_t *ctx_p;

    assert(ctx != NULL && src != NULL);

    ctx_p = (abcdk_cuda_jpeg_t *)ctx->private_ctx;

    ABCDK_ASSERT(!ctx_p->encoder, "编码器不能用于解码。");

    return ctx_p->decoder_ctx->update(src);
}

#else // __cuda_cuda_h__

abcdk_torch_jcodec_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_jpeg_encode_to_file(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

abcdk_torch_image_t *abcdk_cuda_jpeg_decode(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

abcdk_torch_image_t *abcdk_cuda_jpeg_decode_from_file(abcdk_torch_jcodec_t *ctx, const void *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

#endif //__cuda_cuda_h__
