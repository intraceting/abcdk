/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/xpu/vdec.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/vdec.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/vdec.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

void abcdk_xpu_vdec_free(abcdk_xpu_vdec_t **ctx)
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::vdec::free((abcdk_xpu::general::vdec::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #if !defined(__XPU_NVIDIA__)
        abcdk_xpu::nvidia::vdec::free((abcdk_xpu::nvidia::vdec::metadata_t **)ctx);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return;
}

abcdk_xpu_vdec_t *abcdk_xpu_vdec_alloc()
{

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_vdec_t *)abcdk_xpu::general::vdec::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_vdec_t *)abcdk_xpu::nvidia::vdec::alloc();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_vdec_setup(abcdk_xpu_vdec_t *ctx, const abcdk_xpu_vcodec_params_t *params)
{
    assert(ctx != NULL && params != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::setup((abcdk_xpu::general::vdec::metadata_t *)ctx, params, (abcdk_xpu::general::context::metadata_t *)_abcdk_xpu_context_current_get());
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::vdec::setup((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, params, (abcdk_xpu::nvidia::context::metadata_t *)_abcdk_xpu_context_current_get());
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_vdec_send_packet(abcdk_xpu_vdec_t *ctx, const void *src_data, size_t src_size, int64_t ts)
{
    assert(ctx != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::send_packet((abcdk_xpu::general::vdec::metadata_t *)ctx, src_data, src_size, ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::vdec::send_packet((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, src_data, src_size, ts);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_vdec_recv_frame(abcdk_xpu_vdec_t *ctx, abcdk_xpu_image_t **dst, int64_t *ts)
{
    assert(ctx != NULL && dst != NULL && ts != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::vdec::recv_frame((abcdk_xpu::general::vdec::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t **)dst, ts);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::vdec::recv_frame((abcdk_xpu::nvidia::vdec::metadata_t *)ctx, (abcdk_xpu::nvidia::image::metadata_t **)dst, ts);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}