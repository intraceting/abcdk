/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/infer.h"
#include "infer_engine.hxx"

__BEGIN_DECLS

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

void abcdk_torch_infer_free_cuda(abcdk_torch_infer_t **ctx)
{
    abcdk_torch_infer_t *ctx_p;
    abcdk::torch_cuda::infer::engine *cu_ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_torch_infer_t *)*ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk::torch_cuda::infer::engine *)ctx_p->private_ctx;

    abcdk::torch::memory::delete_object(&cu_ctx_p);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_infer_t *abcdk_torch_infer_alloc_cuda()
{
    abcdk_torch_infer_t *ctx;

    ctx = (abcdk_torch_infer_t *)abcdk_heap_alloc(sizeof(abcdk_torch_infer_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_CUDA;

    ctx->private_ctx = new abcdk::torch_cuda::infer::engine();
    if(!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_infer_free_cuda(&ctx);
    return NULL;
}

int abcdk_torch_infer_load_model_cuda(abcdk_torch_infer_t *ctx, const char *file, abcdk_option_t *opt)
{
    abcdk::torch_cuda::infer::engine *cu_ctx_p;
    int chk;

    assert(ctx != NULL && file != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk::torch_cuda::infer::engine *)ctx->private_ctx;

    chk = cu_ctx_p->load(file);
    if(chk != 0)
        return -1;

    chk = cu_ctx_p->prepare(opt);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_torch_infer_execute_cuda(abcdk_torch_infer_t *ctx, int count, abcdk_torch_image_t *img[])
{
    abcdk::torch_cuda::infer::engine *cu_ctx_p;
    std::vector<abcdk_torch_image_t *> tmp_img;
    int chk;

    assert(ctx != NULL && count > 0 && img != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_CUDA);

    cu_ctx_p = (abcdk::torch_cuda::infer::engine *)ctx->private_ctx;

    tmp_img.resize(count);
    for (int i = 0; i < count; i++)
    {
        tmp_img[i] = img[i];
    }

    chk = cu_ctx_p->execute(tmp_img);
    if(chk != 0)
        return -1;

    return 0;
}

#else // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)


void abcdk_torch_infer_free_cuda(abcdk_torch_infer_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return ;
}

abcdk_torch_infer_t *abcdk_torch_infer_alloc_cuda()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return NULL;
}

int abcdk_torch_infer_load_model_cuda(abcdk_torch_infer_t *ctx, const char *file, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return -1;
}

int abcdk_torch_infer_execute_cuda(abcdk_torch_infer_t *ctx, int count, abcdk_torch_image_t *img[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return -1;
}

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

__END_DECLS