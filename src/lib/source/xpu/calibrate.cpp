/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/calibrate.h"
#include "runtime.in.h"
#include "context.in.h"
#include "general/calibrate.hxx"
#include "nvidia/calibrate.hxx"

void abcdk_xpu_calibrate_free(abcdk_xpu_calibrate_t **ctx)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::calibrate::free((abcdk_xpu::general::calibrate::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::calibrate::free((abcdk_xpu::nvidia::calibrate::metadata_t **)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return;
}

abcdk_xpu_calibrate_t *abcdk_xpu_calibrate_alloc()
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_calibrate_t *)abcdk_xpu::general::calibrate::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_calibrate_t *)abcdk_xpu::nvidia::calibrate::alloc();
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

void abcdk_xpu_calibrate_setup(abcdk_xpu_calibrate_t *ctx, int board_cols, int board_rows, int grid_width, int grid_height)
{
    assert(ctx != NULL && board_cols >= 2 && board_rows >= 2 && grid_width >= 5 && grid_height >= 5);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::calibrate::setup((abcdk_xpu::general::calibrate::metadata_t *)ctx, board_cols, board_rows, grid_width, grid_height);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::calibrate::setup((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx, board_cols, board_rows, grid_width, grid_height);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return;
}

int abcdk_xpu_calibrate_detect_corners(abcdk_xpu_calibrate_t *ctx, const abcdk_xpu_image_t *img)
{
    assert(ctx != NULL && img != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::detect_corners((abcdk_xpu::general::calibrate::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t *)img);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::detect_corners((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t *)img);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

double abcdk_xpu_calibrate_estimate_parameters(abcdk_xpu_calibrate_t *ctx)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::estimate_parameters((abcdk_xpu::general::calibrate::metadata_t *)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return 1.0;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::estimate_parameters((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return 1.0;
}

int abcdk_xpu_calibrate_build_parameters(abcdk_xpu_calibrate_t *ctx, double alpha)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::build_parameters((abcdk_xpu::general::calibrate::metadata_t *)ctx,alpha);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::build_parameters((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx,alpha);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

abcdk_object_t *abcdk_xpu_calibrate_dump_parameters(abcdk_xpu_calibrate_t *ctx, const char *magic)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::dump_parameters((abcdk_xpu::general::calibrate::metadata_t *)ctx, magic);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::dump_parameters((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx, magic);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_calibrate_dump_parameters_to_file(abcdk_xpu_calibrate_t *ctx, const char *dst, const char *magic)
{
    abcdk_object_t *tmp_dst;
    ssize_t wr_len;
    int chk;

    assert(ctx != NULL && dst != NULL);

    tmp_dst = abcdk_xpu_calibrate_dump_parameters(ctx, magic);
    if (!tmp_dst)
        return -1;

    if (access(dst, F_OK) == 0)
    {
        chk = truncate(dst, 0);
        if (chk != 0)
            return -1;
    }

    wr_len = abcdk_save(dst, tmp_dst->pptrs[0], tmp_dst->sizes[0], 0);
    chk = (wr_len == tmp_dst->sizes[0] ? 0 : -1);
    abcdk_object_unref(&tmp_dst);

    return chk;
}

int abcdk_xpu_calibrate_load_parameters(abcdk_xpu_calibrate_t *ctx, const char *src, const char *magic)
{
    assert(ctx != NULL && src != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::load_parameters((abcdk_xpu::general::calibrate::metadata_t *)ctx, src, magic);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::load_parameters((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx, src, magic);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}

int abcdk_xpu_calibrate_load_parameters_from_file(abcdk_xpu_calibrate_t *ctx, const char *src, const char *magic)
{
    abcdk_object_t *tmp_src;
    ssize_t wr_len;
    int chk;

    assert(ctx != NULL && src != NULL);

    tmp_src = abcdk_mmap_filename(src,0,0,0,0);
    if (!tmp_src)
        return -1;

    chk = abcdk_xpu_calibrate_load_parameters(ctx,tmp_src->pstrs[0],magic);
    abcdk_object_unref(&tmp_src);

    return chk;
}

int abcdk_xpu_calibrate_undistort(abcdk_xpu_calibrate_t *ctx, const abcdk_xpu_image_t*src, abcdk_xpu_image_t **dst, abcdk_xpu_inter_t inter_mode)
{
    assert(ctx != NULL && src != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::calibrate::undistort((abcdk_xpu::general::calibrate::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t **)dst, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #ifndef HAVE_CUDA
        return abcdk_xpu::nvidia::calibrate::undistort((abcdk_xpu::nvidia::calibrate::metadata_t *)ctx, (abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t **)dst,inter_mode);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return -1;
}
