/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/xpu/dnn_infer.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/dnn_infer.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/dnn_infer.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

void abcdk_xpu_dnn_infer_free(abcdk_xpu_dnn_infer_t **ctx)
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::dnn::infer::free((abcdk_xpu::general::dnn::infer::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #if !defined(__XPU_NVIDIA__)
        abcdk_xpu::nvidia::dnn::infer::free((abcdk_xpu::nvidia::dnn::infer::metadata_t **)ctx);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return;
}

abcdk_xpu_dnn_infer_t *abcdk_xpu_dnn_infer_alloc()
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_dnn_infer_t *)abcdk_xpu::general::dnn::infer::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_dnn_infer_t *)abcdk_xpu::nvidia::dnn::infer::alloc();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_dnn_infer_load_model(abcdk_xpu_dnn_infer_t *ctx, const char *file, abcdk_option_t *opt)
{
    assert(ctx != NULL && file != NULL && opt != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::dnn::infer::load_model((abcdk_xpu::general::dnn::infer::metadata_t *)ctx, file, opt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::dnn::infer::load_model((abcdk_xpu::nvidia::dnn::infer::metadata_t *)ctx, file, opt);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_dnn_infer_fetch_tensor(abcdk_xpu_dnn_infer_t *ctx, int count, abcdk_xpu_dnn_tensor_t tensor[])
{
    assert(ctx != NULL && count > 0 && tensor != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::dnn::infer::fetch_tensor((abcdk_xpu::general::dnn::infer::metadata_t *)ctx, count, tensor);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::dnn::infer::fetch_tensor((abcdk_xpu::nvidia::dnn::infer::metadata_t *)ctx, count, tensor);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_dnn_infer_forward(abcdk_xpu_dnn_infer_t *ctx, int count, abcdk_xpu_image_t *img[])
{
    assert(ctx != NULL && count > 0 && img != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::dnn::infer::forward((abcdk_xpu::general::dnn::infer::metadata_t *)ctx, count, (abcdk_xpu::general::image::metadata_t **)img);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::dnn::infer::forward((abcdk_xpu::nvidia::dnn::infer::metadata_t *)ctx, count, (abcdk_xpu::general::image::metadata_t **)img);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}