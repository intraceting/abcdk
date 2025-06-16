/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"
#include "abcdk/torch/dnnengine.h"
#include "dnn_engine.hxx"

__BEGIN_DECLS

#ifdef ORT_API_VERSION

void abcdk_torch_dnn_engine_free_host(abcdk_torch_dnn_engine_t **ctx)
{
    abcdk_torch_dnn_engine_t *ctx_p;
    abcdk::torch_host::dnn::engine *cu_ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = (abcdk_torch_dnn_engine_t *)*ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    cu_ctx_p = (abcdk::torch_host::dnn::engine *)ctx_p->private_ctx;

    abcdk::torch::memory::delete_object(&cu_ctx_p);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_host()
{
    abcdk_torch_dnn_engine_t *ctx;

    ctx = (abcdk_torch_dnn_engine_t *)abcdk_heap_alloc(sizeof(abcdk_torch_dnn_engine_t));
    if (!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;

    ctx->private_ctx = new abcdk::torch_host::dnn::engine();
    if (!ctx->private_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_dnn_engine_free_host(&ctx);
    return NULL;
}

int abcdk_torch_dnn_engine_load_model_host(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt)
{
    abcdk::torch_host::dnn::engine *ht_ctx_p;
    int chk;

    assert(ctx != NULL && file != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch_host::dnn::engine *)ctx->private_ctx;

    chk = ht_ctx_p->load(file);
    if (chk != 0)
        return -1;

    chk = ht_ctx_p->prepare(opt);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_torch_dnn_engine_fetch_tensor_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor_t tensor[])
{
    abcdk::torch_host::dnn::engine *ht_ctx_p;
    int chk_count = 0;

    assert(ctx != NULL && count > 0 && tensor != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch_host::dnn::engine *)ctx->private_ctx;

    for (int i = 0; i < count; i++)
    {
        abcdk::torch_host::dnn::tensor *src_p = ht_ctx_p->tensor_ptr(i);
        if (!src_p)
            break;

        abcdk_torch_dnn_tensor_t *dst_p = &tensor[i];

        dst_p->index = src_p->index();
        dst_p->name_p = src_p->name();
        dst_p->mode = (src_p->input() ? 1 : 2);

        dst_p->dims.nb = src_p->dims().size();

        for (int i = 0; i < dst_p->dims.nb; i++)
            dst_p->dims.d[i] = (int)src_p->dims()[i];

        dst_p->data_p = (src_p->input() ? NULL : src_p->data(1));

        chk_count += 1;
    }

    return chk_count;
}

int abcdk_torch_dnn_engine_infer_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[])
{
    abcdk::torch_host::dnn::engine *ht_ctx_p;
    std::vector<abcdk_torch_image_t *> tmp_img;
    int chk;

    assert(ctx != NULL && count > 0 && img != NULL);

    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk::torch_host::dnn::engine *)ctx->private_ctx;

    tmp_img.resize(count);
    for (int i = 0; i < count; i++)
    {
        if (!img[i])
            continue;

        assert(img[i]->tag == ABCDK_TORCH_TAG_HOST);

        tmp_img[i] = img[i];
    }

    chk = ht_ctx_p->execute(tmp_img);
    if (chk != 0)
        return -1;

    return 0;
}

#else // ORT_API_VERSION

void abcdk_torch_dnn_engine_free_host(abcdk_torch_dnn_engine_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含ONNXRuntime工具。"));
    return ;
}

abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_host()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含ONNXRuntime工具。"));
    return NULL;
}

int abcdk_torch_dnn_engine_load_model_host(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含ONNXRuntime工具。"));
    return -1; 
}

int abcdk_torch_dnn_engine_fetch_tensor_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor_t info[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含ONNXRuntime工具。"));
    return -1;
}

int abcdk_torch_dnn_engine_infer_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含ONNXRuntime工具。"));
    return -1; 
}

#endif // ORT_API_VERSION

__END_DECLS