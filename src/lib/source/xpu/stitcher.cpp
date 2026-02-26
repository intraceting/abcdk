/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/stitcher.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/stitcher.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/stitcher.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

void abcdk_xpu_stitcher_free(abcdk_xpu_stitcher_t **ctx)
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::stitcher::free((abcdk_xpu::general::stitcher::metadata_t **)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #if !defined(__XPU_NVIDIA__)
        abcdk_xpu::nvidia::stitcher::free((abcdk_xpu::nvidia::stitcher::metadata_t **)ctx);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return;
}

abcdk_xpu_stitcher_t *abcdk_xpu_stitcher_alloc()
{
#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_stitcher_t *)abcdk_xpu::general::stitcher::alloc();
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_stitcher_t *)abcdk_xpu::nvidia::stitcher::alloc();
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_stitcher_set_feature_finder(abcdk_xpu_stitcher_t *ctx, const char *name)
{
    assert(ctx != NULL && name != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::set_feature_finder((abcdk_xpu::general::stitcher::metadata_t *)ctx, name);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::set_feature_finder((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, name);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_stitcher_set_warper(abcdk_xpu_stitcher_t *ctx, const char *name)
{
    assert(ctx != NULL && name != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::set_warper((abcdk_xpu::general::stitcher::metadata_t *)ctx, name);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::set_warper((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, name);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_stitcher_estimate_parameters(abcdk_xpu_stitcher_t *ctx, int count, const abcdk_xpu_image_t *img[], const abcdk_xpu_image_t *mask[], float threshold)
{
    assert(ctx != NULL && count >= 2 && img != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::estimate_parameters((abcdk_xpu::general::stitcher::metadata_t *)ctx, count, (const abcdk_xpu::general::image::metadata_t **)img,
                                                             (const abcdk_xpu::general::image::metadata_t **)mask, threshold);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::estimate_parameters((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, count, (const abcdk_xpu::nvidia::image::metadata_t **)img,
                                                            (const abcdk_xpu::nvidia::image::metadata_t **)mask, threshold);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_stitcher_build_parameters(abcdk_xpu_stitcher_t *ctx)
{
    assert(ctx != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::build_parameters((abcdk_xpu::general::stitcher::metadata_t *)ctx);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::build_parameters((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

abcdk_object_t *abcdk_xpu_stitcher_dump_parameters(abcdk_xpu_stitcher_t *ctx, const char *magic)
{
    assert(ctx != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::dump_parameters((abcdk_xpu::general::stitcher::metadata_t *)ctx, magic);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::dump_parameters((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, magic);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_stitcher_dump_parameters_to_file(abcdk_xpu_stitcher_t *ctx, const char *dst, const char *magic)
{
    abcdk_object_t *tmp_dst;
    ssize_t wr_len;
    int chk;

    assert(ctx != NULL && dst != NULL);

    tmp_dst = abcdk_xpu_stitcher_dump_parameters(ctx, magic);
    if (!tmp_dst)
        return -1;

    wr_len = abcdk_dump(dst, tmp_dst->pptrs[0], tmp_dst->sizes[0]);
    chk = (wr_len == tmp_dst->sizes[0] ? 0 : -1);
    abcdk_object_unref(&tmp_dst);

    return chk;
}

int abcdk_xpu_stitcher_load_parameters(abcdk_xpu_stitcher_t *ctx, const char *src, const char *magic)
{
    assert(ctx != NULL && src != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::load_parameters((abcdk_xpu::general::stitcher::metadata_t *)ctx, src, magic);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::load_parameters((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, src, magic);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_stitcher_load_parameters_from_file(abcdk_xpu_stitcher_t *ctx, const char *src, const char *magic)
{
    abcdk_object_t *tmp_src;
    ssize_t wr_len;
    int chk;

    assert(ctx != NULL && src != NULL);

    tmp_src = abcdk_mmap_filename(src, 0, 0, 0, 0);
    if (!tmp_src)
        return -1;

    chk = abcdk_xpu_stitcher_load_parameters(ctx, tmp_src->pstrs[0], magic);
    abcdk_object_unref(&tmp_src);

    return chk;
}

int abcdk_xpu_stitcher_compose(abcdk_xpu_stitcher_t *ctx, int count, const abcdk_xpu_image_t *img[], abcdk_xpu_image_t **out, int optimize_seam)
{
    assert(ctx != NULL && count >= 2 && img != NULL && out != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::stitcher::compose((abcdk_xpu::general::stitcher::metadata_t *)ctx, count, (const abcdk_xpu::general::image::metadata_t **)img,
                                                 (abcdk_xpu::general::image::metadata_t **)out, optimize_seam);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::stitcher::compose((abcdk_xpu::nvidia::stitcher::metadata_t *)ctx, count, (const abcdk_xpu::nvidia::image::metadata_t **)img,
                                                (abcdk_xpu::nvidia::image::metadata_t **)out, optimize_seam);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}