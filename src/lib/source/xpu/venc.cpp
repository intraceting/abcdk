/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/venc.h"
#include "runtime.in.h"
#include "context.in.h"
#include "general/venc.hxx"
#include "nvidia/venc.hxx"

void abcdk_xpu_venc_free(abcdk_xpu_venc_t **ctx)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::venc::free((abcdk_xpu::general::venc::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::venc::free((abcdk_xpu::nvidia::venc::metadata_t **)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return;
}

abcdk_xpu_venc_t *abcdk_xpu_venc_alloc()
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_venc_t *)abcdk_xpu::general::venc::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_venc_t *)abcdk_xpu::nvidia::venc::alloc();
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_venc_setup(abcdk_xpu_venc_t *ctx, const abcdk_xpu_vcodec_params_t *params)
{
    assert(ctx != NULL && params != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::venc::setup((abcdk_xpu::general::venc::metadata_t *)ctx, params, (abcdk_xpu::general::context::metadata_t *)_abcdk_xpu_context_current_get());
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::venc::setup((abcdk_xpu::nvidia::venc::metadata_t *)ctx, params, (abcdk_xpu::nvidia::context::metadata_t *)_abcdk_xpu_context_current_get());
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_venc_get_params(abcdk_xpu_venc_t *ctx, abcdk_xpu_vcodec_params_t *params)
{
    assert(ctx != NULL && params != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::venc::get_params((abcdk_xpu::general::venc::metadata_t *)ctx, params);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::venc::get_params((abcdk_xpu::nvidia::venc::metadata_t *)ctx, params);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_venc_recv_packet(abcdk_xpu_venc_t *ctx ,abcdk_object_t **dst, int64_t *ts)
{
    assert(ctx != NULL && dst != NULL && ts != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::venc::recv_packet((abcdk_xpu::general::venc::metadata_t *)ctx, dst ,ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::venc::recv_packet((abcdk_xpu::nvidia::venc::metadata_t *)ctx, dst, ts);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_venc_send_frame(abcdk_xpu_venc_t *ctx, const abcdk_xpu_image_t *src,int64_t ts)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::venc::send_frame((abcdk_xpu::general::venc::metadata_t *)ctx, (const abcdk_xpu::general::image::metadata_t *)src ,ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::venc::send_frame((abcdk_xpu::nvidia::venc::metadata_t *)ctx, (const abcdk_xpu::nvidia::image::metadata_t *)src,ts);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}