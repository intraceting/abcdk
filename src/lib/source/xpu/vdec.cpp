/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/vdec.h"
#include "runtime.in.h"
#include "context.in.h"
#include "general/vdec.hxx"
#include "nvidia/vdec.hxx"

void abcdk_xpu_vdec_free(abcdk_xpu_vdec_t **ctx)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::vdec::free((abcdk_xpu::general::vdec::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::vdec::free((abcdk_xpu::nvidia::vdec::metadata_t **)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return;
}

abcdk_xpu_vdec_t *abcdk_xpu_vdec_alloc()
{

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_vdec_t *)abcdk_xpu::general::vdec::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_vdec_t *)abcdk_xpu::nvidia::vdec::alloc();
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_vdec_setup(abcdk_xpu_vdec_t *ctx, const abcdk_xpu_vcodec_params_t *params)
{
    assert(ctx != NULL && params != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::setup((abcdk_xpu::general::vdec::metadata_t *)ctx, params, (abcdk_xpu::general::context::metadata_t *)_abcdk_xpu_context_current_get());
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::vdec::setup((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, params, (abcdk_xpu::nvidia::context::metadata_t *)_abcdk_xpu_context_current_get());
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_vdec_send_packet(abcdk_xpu_vdec_t *ctx, const void *src_data, size_t src_size, int64_t ts)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::send_packet((abcdk_xpu::general::vdec::metadata_t *)ctx, src_data, src_size, ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::vdec::send_packet((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, src_data, src_size, ts);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_vdec_recv_frame(abcdk_xpu_vdec_t *ctx, abcdk_xpu_image_t **dst, int64_t *ts)
{
    assert(ctx != NULL && dst != NULL && ts != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::recv_frame((abcdk_xpu::general::vdec::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t **)dst, ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::vdec::recv_frame((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, (abcdk_xpu::nvidia::image::metadata_t **)dst, ts);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}