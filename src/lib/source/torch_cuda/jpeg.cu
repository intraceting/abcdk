/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/jcodec.h"
#include "jpeg_decoder_general.cu.hxx"
#include "jpeg_decoder_aarch64.cu.hxx"
#include "jpeg_encoder_general.cu.hxx"
#include "jpeg_encoder_aarch64.cu.hxx"


__BEGIN_DECLS

#ifdef __cuda_cuda_h__

/** JPEG编/解码器。*/
typedef struct _abcdk_torch_jcodec_cuda
{
    /**是否为编码器。!0 是，0 否。*/
    uint8_t encoder;

    /**编码器。*/
    abcdk::cuda::jpeg::encoder *encoder_ctx;

    /**解码器。*/
    abcdk::cuda::jpeg::decoder *decoder_ctx;

} abcdk_torch_jcodec_cuda_t;

void abcdk_torch_jcodec_free_cuda(abcdk_torch_jcodec_t **ctx)
{
    abcdk_torch_jcodec_t *ctx_p;
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_torch_jcodec_t *)*ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx_p->private_ctx;

    if (cu_ctx_p->encoder)
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::encoder_general::destory(&cu_ctx_p->encoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::encoder_aarch64::destory(&cu_ctx_p->encoder_ctx);
#endif //__x86_64__ || __aarch64__
    }
    else
    {
#ifdef __x86_64__
        abcdk::cuda::jpeg::decoder_general::destory(&cu_ctx_p->decoder_ctx);
#elif defined(__aarch64__)
        abcdk::cuda::jpeg::decoder_aarch64::destory(&cu_ctx_p->decoder_ctx);
#endif //__x86_64__ || __aarch64__
    }

    abcdk_heap_free(cu_ctx_p);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_cuda(int encoder)
{
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;

    ctx = (abcdk_torch_jcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_jcodec_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_CUDA;

    /*创建内部对象。*/
    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_torch_jcodec_cuda_t));
    if(!ctx->private_ctx)
        goto ERR;

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

    if (cu_ctx_p->encoder = encoder)
    {
#ifdef __x86_64__
        cu_ctx_p->encoder_ctx = abcdk::cuda::jpeg::encoder_general::create(abcdk_torch_ctx_getspecific());
#elif defined(__aarch64__)
        cu_ctx_p->encoder_ctx = abcdk::cuda::jpeg::encoder_aarch64::create(abcdk_torch_ctx_getspecific());
#endif //__x86_64__ || __aarch64__

        if (!ctx_p->encoder_ctx)
            goto ERR;
    }
    else
    {
#ifdef __x86_64__
        cu_ctx_p->decoder_ctx = abcdk::cuda::jpeg::decoder_general::create(abcdk_torch_ctx_getspecific());
#elif defined(__aarch64__)
        cu_ctx_p->decoder_ctx = abcdk::cuda::jpeg::decoder_aarch64::create(abcdk_torch_ctx_getspecific());
#endif //__x86_64__ || __aarch64__

        if (!cu_ctx_p->decoder_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_torch_jcodec_free_cuda(&ctx);

    return NULL;
}

int abcdk_torch_jcodec_start_cuda(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param)
{
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;
    int chk;

    assert(ctx != NULL);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

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

abcdk_object_t *abcdk_torch_jcodec_encode_cuda(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;
    abcdk_torch_image_t *tmp_src = NULL;
    abcdk_object_t *dst = NULL;

    assert(ctx != NULL && src != NULL);

    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(cu_ctx_p->encoder, TT("解码器不能用于编码。"));

    if (src->tag == ABCDK_TORCH_TAG_HOST)
    {
        tmp_src = abcdk_torch_image_clone_cuda(0, src);
        if (!tmp_src)
            return NULL;

        dst = abcdk_torch_jcodec_encode_cuda(ctx, tmp_src);
        abcdk_torch_image_free_cuda(&tmp_src);

        return dst;
    }

    return cu_ctx_p->encoder_ctx->update(src);
}

int abcdk_torch_jcodec_encode_to_file_cuda(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;
    int chk;

    assert(ctx != NULL && dst != NULL && src != NULL);

    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(cu_ctx_p->encoder, TT("解码器不能用于编码。"));

    chk = cu_ctx_p->encoder_ctx->update(dst, src);
    if (chk != 0)
        return -1;

    return 0;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_cuda(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;

    assert(ctx != NULL && src != NULL && src_size > 0);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(!cu_ctx_p->encoder, TT("编码器不能用于解码。"));

    return cu_ctx_p->decoder_ctx->update(src, src_size);
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_cuda(abcdk_torch_jcodec_t *ctx, const void *src)
{
    abcdk_torch_jcodec_cuda_t *cu_ctx_p;

    assert(ctx != NULL && src != NULL);

    cu_ctx_p = (abcdk_torch_jcodec_cuda_t *)ctx->private_ctx;

    ABCDK_ASSERT(!cu_ctx_p->encoder, TT("编码器不能用于解码。"));

    return cu_ctx_p->decoder_ctx->update(src);
}

#else // __cuda_cuda_h__

void abcdk_torch_jcodec_free_cuda(void **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return ;
}

abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_cuda(int encode, abcdk_option_t *cfg, CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

abcdk_object_t *abcdk_torch_jcodec_encode_cuda(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

int abcdk_torch_jcodec_encode_to_file_cuda(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_cuda(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_cuda(abcdk_torch_jcodec_t *ctx, const void *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return NULL;
}

#endif //__cuda_cuda_h__


__END_DECLS