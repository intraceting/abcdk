/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/xpu/runtime.h"
#include "runtime.in.h"

#if defined(__XPU_GENERAL__)
#if defined(__XPU_NVIDIA__)
#include "nvidia/runtime.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

static int _abcdk_xpu_hwaccel_current = ABCDK_XPU_HWACCEL_NONE;

void _abcdk_xpu_hwaccel_set(int hwaccel)
{
    _abcdk_xpu_hwaccel_current = hwaccel;
}

int _abcdk_xpu_hwaccel_get()
{
    return _abcdk_xpu_hwaccel_current;
}

int abcdk_xpu_runtime_deinit()
{
#if defined(__XPU_GENERAL__)

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return 0;
    }
    else 
    if(_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::runtime::deinit();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_runtime_init(int hwaccel, ...)
{
    int chk;

    assert(hwaccel == ABCDK_XPU_HWACCEL_NONE ||
           hwaccel == ABCDK_XPU_HWACCEL_NVIDIA ||
           hwaccel == ABCDK_XPU_HWACCEL_SOPHON ||
           hwaccel == ABCDK_XPU_HWACCEL_ROCKCHIP);

#if defined(__XPU_GENERAL__)

    _abcdk_xpu_hwaccel_set(hwaccel);

    if (hwaccel == ABCDK_XPU_HWACCEL_NONE)
    {
        return 0;
    }
    else if (hwaccel == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::runtime::init();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return 1;
}