/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/atomic.h"
#include "abcdk/util/object.h"
#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/xpu/dnn_track.h"
#include "runtime.in.h"

#if defined(__XPU_GENERAL__)
#include "common/dnn_mot_bytetrack.hxx"
#endif //#if defined(__XPU_GENERAL__)

typedef struct _abcdk_xpu_dnn_track
{
#if defined(__XPU_GENERAL__)
    abcdk_xpu::common::dnn::mot *mot_ctx;
#endif //#if defined(__XPU_GENERAL__)
} abcdk_xpu_dnn_track_t;

void abcdk_xpu_dnn_track_free(abcdk_xpu_dnn_track_t **ctx)
{
    abcdk_xpu_dnn_track_t *ctx_p;

#if defined(__XPU_GENERAL__)

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->mot_ctx)
    {
        if (abcdk_strcmp(ctx_p->mot_ctx->name(), "bytetrack", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::mot_bytetrack **)&ctx_p->mot_ctx);
        else
            abcdk_xpu::common::util::delete_object(&ctx_p->mot_ctx);
    }

    abcdk_xpu::common::util::delete_object(&ctx_p);

#endif // #if defined(__XPU_GENERAL__)
}

abcdk_xpu_dnn_track_t *abcdk_xpu_dnn_track_alloc()
{
    abcdk_xpu_dnn_track_t *ctx = NULL;

#if defined(__XPU_GENERAL__)

    ctx = new abcdk_xpu_dnn_track_t;
    if (!ctx)
        return NULL;

    ctx->mot_ctx = NULL;


#endif //#if defined(__XPU_GENERAL__)

    return ctx;
}

int abcdk_xpu_dnn_track_init(abcdk_xpu_dnn_track_t *ctx, const char *name, abcdk_option_t *opt)
{
    assert(ctx != NULL && name != NULL && opt != NULL);

#if defined(__XPU_GENERAL__)

    ABCDK_TRACE_ASSERT(ctx->mot_ctx == NULL, ABCDK_GETTEXT("仅允许初始化一次."));

    if (abcdk_strcmp(name, "bytetrack", 0) == 0)
    {
        ctx->mot_ctx = new abcdk_xpu::common::dnn::mot_bytetrack(name);
        if (!ctx->mot_ctx)
            return -1;
    }
    else
    {
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("尚未支持的追踪器(%s)."), name);
        return -1;
    }

    ctx->mot_ctx->prepare(opt);
    return 0;
#else //#if defined(__XPU_GENERAL__)
    return -1;
#endif //#if defined(__XPU_GENERAL__)
}

int abcdk_xpu_dnn_track_update(abcdk_xpu_dnn_track_t *ctx, int count, abcdk_xpu_dnn_object_t object[])
{
    assert(ctx != NULL && count > 0 && object != NULL);

#if defined(__XPU_GENERAL__)

    ABCDK_TRACE_ASSERT(ctx->mot_ctx != NULL,ABCDK_GETTEXT("未初始化, 不能执行此操作."));

    ctx->mot_ctx->update(count,object);
    return 0;
#else //#if defined(__XPU_GENERAL__)
    return -1;
#endif //#if defined(__XPU_GENERAL__)
}
