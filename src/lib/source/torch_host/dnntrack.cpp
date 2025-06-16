/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/torch/dnntrack.h"
#include "../torch/memory.hxx"
#include "dnn_mot_bytetrack.hxx"


__BEGIN_DECLS

/** DNN后处理环境。*/
struct _abcdk_torch_dnn_track
{
    /**环境。*/
    abcdk::torch_host::dnn::mot *mot_ctx;
}; // abcdk_torch_dnn_track_t;

void abcdk_torch_dnn_track_free(abcdk_torch_dnn_track_t **ctx)
{
    abcdk_torch_dnn_track_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*可能未初始化。*/
    if(!ctx_p->mot_ctx)
        goto END;

    if (abcdk_strcmp(ctx_p->mot_ctx->name(), "bytetrack", 0) == 0)
#ifdef __BYTETRACK__
        abcdk::torch::memory::delete_object((abcdk::torch_host::dnn::mot_bytetrack **)&ctx_p->mot_ctx);
#endif //__BYTETRACK__
    else
        abcdk::torch::memory::delete_object(&ctx_p->mot_ctx);

END:

    abcdk_heap_free(ctx_p);
}

abcdk_torch_dnn_track_t *abcdk_torch_dnn_track_alloc()
{
    abcdk_torch_dnn_track_t *ctx;

    ctx = (abcdk_torch_dnn_track_t *)abcdk_heap_alloc(sizeof(abcdk_torch_dnn_track_t));
    if (!ctx)
        return NULL;

    return ctx;

ERR:

    abcdk_torch_dnn_track_free(&ctx);
    return NULL;
}

int abcdk_torch_dnn_track_init(abcdk_torch_dnn_track_t *ctx, const char *name, abcdk_option_t *opt)
{
    assert(ctx != NULL && name != NULL && opt != NULL);

    ABCDK_ASSERT(ctx->mot_ctx == NULL, TT("仅允许初始化一次。"));

    if (abcdk_strcmp(name, "bytetrack", 0) == 0)
    {
#ifdef __BYTETRACK__
        ctx->mot_ctx = new abcdk::torch_host::dnn::mot_bytetrack(name);
#else 
        abcdk_trace_printf(LOG_WARNING,TT("当前环境在构建时未包含bytetrack工具。"));
#endif //__BYTETRACK__
        if (!ctx->mot_ctx)
            return -1;
    }
    else
    {
        abcdk_trace_printf(LOG_WARNING, TT("尚未支持的追踪器(%s)。"), name);
        return -1;
    }

    ctx->mot_ctx->prepare(opt);

    return 0;
}

int abcdk_torch_dnn_track_update(abcdk_torch_dnn_track_t *ctx, int count, abcdk_torch_dnn_object_t object[])
{
    assert(ctx != NULL && count > 0 && object != NULL);
    ABCDK_ASSERT(ctx->mot_ctx != NULL,TT("未初始化，不能执行此操作。"));

    ctx->mot_ctx->update(count,object);

    return 0;
}

__END_DECLS