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
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/util.h"
#include "abcdk/xpu/image.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/image.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/image.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

void abcdk_xpu_image_free(abcdk_xpu_image_t **ctx)
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::image::free((abcdk_xpu::general::image::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #if !defined(__XPU_NVIDIA__)
        abcdk_xpu::nvidia::image::free((abcdk_xpu::nvidia::image::metadata_t **)ctx);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return;
}

abcdk_xpu_image_t *abcdk_xpu_image_alloc()
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_image_t *)abcdk_xpu::general::image::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_image_t *)abcdk_xpu::nvidia::image::alloc();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_image_reset(abcdk_xpu_image_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
{
    assert(ctx != NULL && width > 0 && height > 0 && pixfmt > ABCDK_XPU_PIXFMT_NONE && align >= 0);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::reset((abcdk_xpu::general::image::metadata_t **)ctx, width, height, pixfmt, align);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::reset((abcdk_xpu::nvidia::image::metadata_t **)ctx, width, height, pixfmt, align, 0);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

abcdk_xpu_image_t *abcdk_xpu_image_create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
{
    assert(width > 0 && height > 0 && pixfmt > ABCDK_XPU_PIXFMT_NONE && align >= 0);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_image_t *)abcdk_xpu::general::image::create(width, height, pixfmt, align);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_image_t *)abcdk_xpu::nvidia::image::create( width, height, pixfmt, align, 0);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_image_copy(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::copy((abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t *)dst);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::copy((abcdk_xpu::nvidia::image::metadata_t *)src, 0, (abcdk_xpu::nvidia::image::metadata_t *)dst, 0);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_image_get_width(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_width((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::get_width((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_image_get_height(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_height((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::get_height((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

abcdk_xpu_pixfmt_t abcdk_xpu_image_get_pixfmt(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_pixfmt((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return ABCDK_XPU_PIXFMT_NONE;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::get_pixfmt((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return ABCDK_XPU_PIXFMT_NONE;
}

int abcdk_xpu_image_upload(const uint8_t *src_data[4], const int src_linesize[4], abcdk_xpu_image_t *dst)
{
    assert(src_data != NULL && src_linesize != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::upload(src_data, src_linesize, (abcdk_xpu::general::image::metadata_t *)dst);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::upload(src_data, src_linesize, (abcdk_xpu::nvidia::image::metadata_t *)dst);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_image_download(const abcdk_xpu_image_t *src, uint8_t *dst_data[4], int dst_linesize[4])
{
    assert(src != NULL && dst_data != NULL && dst_linesize != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::download((const abcdk_xpu::general::image::metadata_t *)src, dst_data, dst_linesize);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::image::download((const abcdk_xpu::nvidia::image::metadata_t *)src, dst_data, dst_linesize);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_image_empty(const abcdk_xpu_image_t *src)
{
    int w, h;

    assert(src != NULL);

    w = abcdk_xpu_image_get_width(src);
    h = abcdk_xpu_image_get_height(src);

    return ((w * h <= 0) ? 1 : 0);
}

int abcdk_xpu_image_clone(const abcdk_xpu_image_t *src, abcdk_xpu_image_t **dst, int dst_align)
{
    abcdk_xpu_pixfmt_t f;
    int w, h;
    int chk;

    assert(src != NULL && dst != NULL && dst_align >= 0);

    if (!abcdk_xpu_image_empty(src))
        return -1;

    w = abcdk_xpu_image_get_width(src);
    h = abcdk_xpu_image_get_height(src);
    f = abcdk_xpu_image_get_pixfmt(src);

    chk = abcdk_xpu_image_reset(dst, w, h, f, dst_align);
    if (chk != 0)
        return chk;

    chk = abcdk_xpu_image_copy(src, *dst);
    if (chk != 0)
        return chk;

    return 0;
}