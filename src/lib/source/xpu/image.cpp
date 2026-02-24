/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/image.h"
#include "runtime.in.h"
#include "context.in.h"
#include "general/image.hxx"
#include "nvidia/image.hxx"

void abcdk_xpu_image_free(abcdk_xpu_image_t **ctx)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::image::free((abcdk_xpu::general::image::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::image::free((abcdk_xpu::nvidia::image::metadata_t **)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return;
}

abcdk_xpu_image_t *abcdk_xpu_image_alloc()
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_image_t *)abcdk_xpu::general::image::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_image_t *)abcdk_xpu::nvidia::image::alloc();
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_image_reset(abcdk_xpu_image_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
{
    assert(ctx != NULL && width > 0 && height > 0 && pixfmt > ABCDK_XPU_PIXFMT_NONE && align >= 0);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::reset((abcdk_xpu::general::image::metadata_t **)ctx, width, height, pixfmt, align);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::reset((abcdk_xpu::nvidia::image::metadata_t **)ctx, width, height, pixfmt, align, 0);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

abcdk_xpu_image_t *abcdk_xpu_image_create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
{
    assert(width > 0 && height > 0 && pixfmt > ABCDK_XPU_PIXFMT_NONE && align >= 0);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_image_t *)abcdk_xpu::general::image::create(width, height, pixfmt, align);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_image_t *)abcdk_xpu::nvidia::image::create( width, height, pixfmt, align, 0);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_image_copy(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst)
{
    assert(src != NULL && dst != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::copy((abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t *)dst);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::copy((abcdk_xpu::nvidia::image::metadata_t *)src, 0, (abcdk_xpu::nvidia::image::metadata_t *)dst, 0);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_image_get_width(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_width((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::get_width((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_image_get_height(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_height((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::get_height((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

abcdk_xpu_pixfmt_t abcdk_xpu_image_get_pixfmt(const abcdk_xpu_image_t *src)
{
    assert(src != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::get_pixfmt((abcdk_xpu::general::image::metadata_t *)src);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return ABCDK_XPU_PIXFMT_NONE;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::get_pixfmt((abcdk_xpu::nvidia::image::metadata_t *)src);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return ABCDK_XPU_PIXFMT_NONE;
}

int abcdk_xpu_image_upload(const uint8_t *src_data[4], const int src_linesize[4], abcdk_xpu_image_t *dst)
{
    assert(src_data != NULL && src_linesize != NULL && dst != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::upload(src_data, src_linesize, (abcdk_xpu::general::image::metadata_t *)dst);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::upload(src_data, src_linesize, (abcdk_xpu::nvidia::image::metadata_t *)dst);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_image_download(const abcdk_xpu_image_t *src, uint8_t *dst_data[4], int dst_linesize[4])
{
    assert(src != NULL && dst_data != NULL && dst_linesize != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::image::download((const abcdk_xpu::general::image::metadata_t *)src, dst_data, dst_linesize);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::image::download((const abcdk_xpu::nvidia::image::metadata_t *)src, dst_data, dst_linesize);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}