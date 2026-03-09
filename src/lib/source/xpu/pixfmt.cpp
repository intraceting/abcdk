/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/sws.h"
#include "abcdk/xpu/pixfmt.h"
#include "runtime.in.h"

#if defined(__XPU_GENERAL__)
#include "general/pixfmt.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/pixfmt.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

int abcdk_xpu_pixfmt_get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad)
{
#if defined(__XPU_GENERAL__)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_bit(pixfmt, have_pad);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::pixfmt::get_bit(pixfmt, have_pad);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

const char *abcdk_xpu_pixfmt_get_name(abcdk_xpu_pixfmt_t pixfmt)
{
#if defined(__XPU_GENERAL__)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_name(pixfmt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return "";
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::pixfmt::get_name(pixfmt);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return "";
}

int abcdk_xpu_pixfmt_get_channel(abcdk_xpu_pixfmt_t pixfmt)
{
#if defined(__XPU_GENERAL__)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::pixfmt::get_channel(pixfmt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::pixfmt::get_channel(pixfmt);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}