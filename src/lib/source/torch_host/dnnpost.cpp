/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/torch/dnnpost.h"
#include "../torch/memory.hxx"
#include "dnn_yolo.hxx"

__BEGIN_DECLS

/** DNN后处理环境。*/
struct _abcdk_torch_dnn_post
{
    /**模型名字。*/
    const char *model_name;

    /**模型环境。*/
    abcdk::torch_host::dnn::model *model_ctx;
}; // abcdk_torch_dnn_post_t;

void abcdk_torch_dnn_post_free(abcdk_torch_dnn_post_t **ctx)
{
    abcdk_torch_dnn_post_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (abcdk_strcmp(ctx_p->model_name, "yolo-11", 0) == 0)
        abcdk::torch::memory::delete_object((abcdk::torch_host::dnn::yolo_11 **)&ctx_p->model_ctx);
    else
        abcdk::torch::memory::delete_object(&ctx_p->model_ctx);

    abcdk_heap_free((void*)ctx_p->model_name);

    abcdk_heap_free(ctx_p);
}

abcdk_torch_dnn_post_t *abcdk_torch_dnn_post_alloc()
{
    abcdk_torch_dnn_post_t *ctx;

    ctx = (abcdk_torch_dnn_post_t *)abcdk_heap_alloc(sizeof(abcdk_torch_dnn_post_t));
    if (!ctx)
        return NULL;

    return ctx;

ERR:

    abcdk_torch_dnn_post_free(&ctx);
    return NULL;
}

int abcdk_torch_dnn_post_init(abcdk_torch_dnn_post_t *ctx, const char *name, abcdk_option_t *opt)
{
    assert(ctx != NULL && name != NULL);

    ABCDK_ASSERT(ctx->model_ctx == NULL, TT("仅允许初始化一次。"));

    if (abcdk_strcmp(name, "yolo-11", 0) == 0)
    {
        ctx->model_ctx = new abcdk::torch_host::dnn::yolo_11();
        if (!ctx->model_ctx)
            return -1;

        ctx->model_name = abcdk_strdup_safe(name);

        return 0;
    }
    else
    {
        abcdk_trace_printf(LOG_WARNING, TT("尚未支持的模型(%s)。"), name);
    }

    return -1;
}

int abcdk_torch_dnn_post_process(abcdk_torch_dnn_post_t *ctx, int count, abcdk_torch_dnn_tensor tensor[], float score_threshold, float nms_threshold)
{
    std::vector<abcdk_torch_dnn_tensor> tmp_tensor;

    assert(ctx != NULL && count > 0 && tensor != NULL);
    ABCDK_ASSERT(ctx->model_ctx != NULL,TT("未初始化，不能执行后处理操作。"));

    tmp_tensor.resize(count);

    for (int i = 0; i < count; i++)
    {
        tmp_tensor[i] = tensor[i];
    }

    ctx->model_ctx->process(tmp_tensor, score_threshold, nms_threshold);

    return 0;
}

int abcdk_torch_dnn_post_fetch(abcdk_torch_dnn_post_t *ctx, int index, int count, abcdk_torch_dnn_object_t object[])
{
    std::vector<abcdk_torch_dnn_object_t> dst_object;
    int chk_count = 0;

    assert(ctx != NULL && index >= 0 && count > 0 && object != NULL);
    ABCDK_ASSERT(ctx->model_ctx != NULL,TT("未初始化，不能执行获取结果操作。"));

    ctx->model_ctx->fetch(dst_object, index);

    for (int i = 0; i < count; i++)
    {
        if (i >= dst_object.size())
            break;

        object[i] = dst_object[i];
        chk_count += 1;
    }

    return chk_count;
}

__END_DECLS