/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/dnn_post.h"
#include "runtime.in.h"
#include "common/dnn_yolo_v11_obb.hxx"
#include "common/dnn_yolo_v11_pose.hxx"
#include "common/dnn_yolo_v11_seg.hxx"
#include "common/dnn_yolo_v11.hxx"
#include "common/dnn_face_yunet.hxx"
#include "common/dnn_face_sface.hxx"

typedef struct _abcdk_xpu_dnn_post
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
    abcdk_xpu::common::dnn::model *model_ctx;
#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
} abcdk_xpu_dnn_post_t;

void abcdk_xpu_dnn_post_free(abcdk_xpu_dnn_post_t **ctx)
{
    abcdk_xpu_dnn_post_t *ctx_p;

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->model_ctx)
    {
        if (abcdk_strcmp(ctx_p->model_ctx->name(), "yolo-v11", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::yolo_v11 **)&ctx_p->model_ctx);
        else if (abcdk_strcmp(ctx_p->model_ctx->name(), "yolo-v11-obb", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::yolo_v11_obb **)&ctx_p->model_ctx);
        else if (abcdk_strcmp(ctx_p->model_ctx->name(), "yolo-v11-pose", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::yolo_v11_pose **)&ctx_p->model_ctx);
        else if (abcdk_strcmp(ctx_p->model_ctx->name(), "yolo-v11-seg", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::yolo_v11_seg **)&ctx_p->model_ctx);
        else if (abcdk_strcmp(ctx_p->model_ctx->name(), "face-yunet", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::face_yunet **)&ctx_p->model_ctx);
        else if (abcdk_strcmp(ctx_p->model_ctx->name(), "face-sface", 0) == 0)
            abcdk_xpu::common::util::delete_object((abcdk_xpu::common::dnn::face_sface **)&ctx_p->model_ctx);
        else
            abcdk_xpu::common::util::delete_object(&ctx_p->model_ctx);
    }

    abcdk_xpu::common::util::delete_object(&ctx_p);

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
}

abcdk_xpu_dnn_post_t *abcdk_xpu_dnn_post_alloc()
{
    abcdk_xpu_dnn_post_t *ctx = NULL;

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    ctx = new abcdk_xpu_dnn_post_t;
    if (!ctx)
        return NULL;

    ctx->model_ctx = NULL;

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return ctx;
}

int abcdk_xpu_dnn_post_init(abcdk_xpu_dnn_post_t *ctx, const char *name, abcdk_option_t *opt)
{
    assert(ctx != NULL && name != NULL && opt != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    ABCDK_TRACE_ASSERT(ctx->model_ctx == NULL, ABCDK_GETTEXT("仅允许初始化一次."));

    if (abcdk_strcmp(name, "yolo-v11", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::yolo_v11(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else if (abcdk_strcmp(name, "yolo-v11-obb", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::yolo_v11_obb(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else if (abcdk_strcmp(name, "yolo-v11-pose", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::yolo_v11_pose(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else if (abcdk_strcmp(name, "yolo-v11-seg", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::yolo_v11_seg(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else  if (abcdk_strcmp(name, "face-yunet", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::face_yunet(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else  if (abcdk_strcmp(name, "face-sface", 0) == 0)
    {
        ctx->model_ctx = new abcdk_xpu::common::dnn::face_sface(name);
        if (!ctx->model_ctx)
            return -1;
    }
    else
    {
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("尚未支持的模型(%s)."), name);
        return -1;
    }

    ctx->model_ctx->prepare(opt);
    return 0;
#else //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
    return -1;
#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
}

int abcdk_xpu_dnn_post_process(abcdk_xpu_dnn_post_t *ctx, int count, abcdk_xpu_dnn_tensor_t tensor[], float score_threshold, float nms_threshold)
{
    std::vector<abcdk_xpu_dnn_tensor_t> tmp_tensor;

    assert(ctx != NULL && count > 0 && tensor != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    ABCDK_TRACE_ASSERT(ctx->model_ctx != NULL,ABCDK_GETTEXT("未初始化, 不能执行此操作."));

    tmp_tensor.resize(count);

    for (int i = 0; i < count; i++)
    {
        tmp_tensor[i] = tensor[i];
    }

    ctx->model_ctx->process(tmp_tensor, score_threshold, nms_threshold);
    return 0;
#else //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
    return -1;
#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
}

int abcdk_xpu_dnn_post_fetch(abcdk_xpu_dnn_post_t *ctx, int index, int count, abcdk_xpu_dnn_object_t object[])
{
    std::vector<abcdk_xpu_dnn_object_t> dst_object;
    int chk_count = 0;

    assert(ctx != NULL && index >= 0 && count > 0 && object != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    ABCDK_TRACE_ASSERT(ctx->model_ctx != NULL,ABCDK_GETTEXT("未初始化, 不能执行此操作."));

    ctx->model_ctx->fetch(dst_object, index);

    for (int i = 0; i < count; i++)
    {
        if (i >= dst_object.size())
            break;

        object[i] = dst_object[i];
        chk_count += 1;
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return chk_count;
}
