/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/pixfmt.h"
#include "runtime.in.h"
#include "general/pixfmt.hxx"
#include "nvidia/pixfmt.hxx"

int abcdk_xpu_pixfmt_get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_bit(pixfmt, have_pad);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::pixfmt::get_bit(pixfmt, have_pad);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

const char *abcdk_xpu_pixfmt_get_name(abcdk_xpu_pixfmt_t pixfmt)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_name(pixfmt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return "";
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::pixfmt::get_name(pixfmt);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return "";
}

int abcdk_xpu_pixfmt_get_channel(abcdk_xpu_pixfmt_t pixfmt)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_channel(pixfmt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::pixfmt::get_channel(pixfmt);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}