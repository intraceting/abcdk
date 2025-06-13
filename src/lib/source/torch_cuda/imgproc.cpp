/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"


#ifdef __cuda_cuda_h__

__BEGIN_DECLS

__END_DECLS

#else //__cuda_cuda_h__

__BEGIN_DECLS

int abcdk_torch_imgproc_brightness_cuda(abcdk_torch_image_t *dst, float alpha[], float bate[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_compose_cuda(abcdk_torch_image_t *panorama, abcdk_torch_image_t *compose,
                                     uint32_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_defog_cuda(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_drawrect_cuda(abcdk_torch_image_t *dst, uint32_t color[], int weight, int corner[4])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_remap_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                   const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                   const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                   int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_undistort_cuda(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src, const float camera_matrix[3][3], const float dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_line_cuda(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1; 
}

int abcdk_torch_imgproc_resize_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                    const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                    int keep_aspect_ratio, int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_stuff_cuda(abcdk_torch_image_t *dst, uint32_t scalar[], const abcdk_torch_rect_t *roi)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_warp_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi, const abcdk_torch_point_t dst_quad[4],
                                  const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi, const abcdk_torch_point_t src_quad[4],
                                  int warp_mode, int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_torch_imgproc_drawmask_cuda(abcdk_torch_image_t *dst, abcdk_torch_image_t *mask, float threshold, uint32_t color[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

__END_DECLS

#endif //__cuda_cuda_h__
