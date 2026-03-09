/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"
#include "abcdk/xpu/imgproc.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/imgproc.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/imgproc.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

int abcdk_xpu_imgproc_convert(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::convert((abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t *)dst);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::convert((abcdk_xpu::nvidia::image::metadata_t *)src, (abcdk_xpu::nvidia::image::metadata_t *)dst);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_convert2(abcdk_xpu_image_t **dst, abcdk_xpu_pixfmt_t pixfmt)
{
    assert(dst != NULL && pixfmt > ABCDK_XPU_PIXFMT_NONE);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::convert2((abcdk_xpu::general::image::metadata_t **)dst, pixfmt);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::convert2((abcdk_xpu::nvidia::image::metadata_t **)dst, pixfmt);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;  
}

int abcdk_xpu_imgproc_resize(const abcdk_xpu_image_t *src, const abcdk_xpu_rect_t *src_roi, abcdk_xpu_image_t *dst, abcdk_xpu_inter_t inter_mode)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::resize((abcdk_xpu::general::image::metadata_t *)src, src_roi, (abcdk_xpu::general::image::metadata_t *)dst, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::resize((abcdk_xpu::nvidia::image::metadata_t *)src, src_roi, (abcdk_xpu::nvidia::image::metadata_t *)dst, inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_stuff(abcdk_xpu_image_t *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar)
{
    assert(dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::stuff((abcdk_xpu::general::image::metadata_t *)dst, roi, scalar);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::stuff((abcdk_xpu::nvidia::image::metadata_t *)dst, roi, scalar);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_brightness(abcdk_xpu_image_t *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate)
{
    assert(dst != NULL && alpha != NULL && bate != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::brightness((abcdk_xpu::general::image::metadata_t *)dst, alpha, bate);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::brightness((abcdk_xpu::nvidia::image::metadata_t *)dst, alpha, bate);
#endif // #ifndef panorama
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_rectangle(abcdk_xpu_image_t *dst, const abcdk_xpu_rect_t *rect, int weight, const abcdk_xpu_scalar_t *color)
{
    assert(dst != NULL && rect != NULL && weight > 0 && color != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::rectangle((abcdk_xpu::general::image::metadata_t *)dst, rect, weight, color);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::rectangle((abcdk_xpu::nvidia::image::metadata_t *)dst, rect, weight, color);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_warp(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst, const abcdk_xpu_matrix_3x3_t *coeffs, int warp_mode, abcdk_xpu_inter_t inter_mode)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::warp((abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t *)dst,
                                             coeffs, warp_mode, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::warp((abcdk_xpu::nvidia::image::metadata_t *)src, (abcdk_xpu::nvidia::image::metadata_t *)dst,
                                            coeffs, warp_mode, inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_warp_quad2quad(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t src_quad[4],
                                 abcdk_xpu_image_t *dst, const abcdk_xpu_point_t dst_quad[4],
                                 int warp_mode, abcdk_xpu_inter_t inter_mode)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::warp_quad2quad((abcdk_xpu::general::image::metadata_t *)src, src_quad,
                                                       (abcdk_xpu::general::image::metadata_t *)dst, dst_quad,
                                                       warp_mode, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::warp_quad2quad((abcdk_xpu::nvidia::image::metadata_t *)src, src_quad,
                                                      (abcdk_xpu::nvidia::image::metadata_t *)dst, dst_quad,
                                                      warp_mode, inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_remap(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst,
                        const abcdk_xpu_image_t *xmap, const abcdk_xpu_image_t *ymap,
                        abcdk_xpu_inter_t inter_mode)
{

    assert(src != NULL && dst != NULL && xmap != NULL && ymap != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::remap((abcdk_xpu::general::image::metadata_t *)src, (abcdk_xpu::general::image::metadata_t *)dst,
                                              (abcdk_xpu::general::image::metadata_t *)xmap, (abcdk_xpu::general::image::metadata_t *)ymap,
                                              inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::remap((abcdk_xpu::nvidia::image::metadata_t *)src, (abcdk_xpu::nvidia::image::metadata_t *)dst,
                                             (abcdk_xpu::nvidia::image::metadata_t *)xmap, (abcdk_xpu::nvidia::image::metadata_t *)ymap,
                                             inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_undistort(const abcdk_xpu_size_t *size, double alpha,
                            const abcdk_xpu_matrix_3x3_t *camera_matrix,
                            const abcdk_xpu_scalar_t *dist_coeffs,
                            abcdk_xpu_image_t **xmap, abcdk_xpu_image_t **ymap)
{
    assert(size != NULL && camera_matrix != NULL && dist_coeffs != NULL && xmap != NULL && ymap != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::undistort(size, alpha, camera_matrix, dist_coeffs, (abcdk_xpu::general::image::metadata_t **)xmap, (abcdk_xpu::general::image::metadata_t **)ymap);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::undistort(size, alpha, camera_matrix, dist_coeffs, (abcdk_xpu::nvidia::image::metadata_t **)xmap, (abcdk_xpu::nvidia::image::metadata_t **)ymap);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_line(abcdk_xpu_image_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                       const abcdk_xpu_scalar_t *color, int weight)
{
    assert(dst != NULL && p1 != NULL && p2 != NULL && color != NULL && weight > 0);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::line((abcdk_xpu::general::image::metadata_t *)dst, p1, p2, color, weight);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::line((abcdk_xpu::nvidia::image::metadata_t *)dst, p1, p2, color, weight);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_mask(abcdk_xpu_image_t *dst, const abcdk_xpu_image_t *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not)
{
    assert(dst != NULL && feature != NULL && color != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::mask((abcdk_xpu::general::image::metadata_t *)dst, (abcdk_xpu::general::image::metadata_t *)feature, threshold, color, less_or_not);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::mask((abcdk_xpu::nvidia::image::metadata_t *)dst, (abcdk_xpu::nvidia::image::metadata_t *)feature, threshold, color, less_or_not);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_quad2rect(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t src_quad[4], abcdk_xpu_image_t *dst, abcdk_xpu_inter_t inter_mode)
{
    assert(src != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::quad2rect((abcdk_xpu::general::image::metadata_t *)src, src_quad, (abcdk_xpu::general::image::metadata_t *)dst, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::quad2rect((abcdk_xpu::nvidia::image::metadata_t *)src, src_quad, (abcdk_xpu::nvidia::image::metadata_t *)dst, inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}

int abcdk_xpu_imgproc_face_warp(const abcdk_xpu_image_t *src, const abcdk_xpu_point_t face_kpt[5], abcdk_xpu_image_t **dst, abcdk_xpu_inter_t inter_mode)
{
    assert(src != NULL && face_kpt != NULL && dst != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgproc::face_warp((abcdk_xpu::general::image::metadata_t *)src, face_kpt, (abcdk_xpu::general::image::metadata_t **)dst, inter_mode);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return -1;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgproc::face_warp((abcdk_xpu::nvidia::image::metadata_t *)src, face_kpt, (abcdk_xpu::nvidia::image::metadata_t **)dst, inter_mode);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return -1;
}