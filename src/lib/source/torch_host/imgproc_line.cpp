/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS

#ifdef OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_line_host(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight)
{
    return -1;
}

#else // OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_line_host(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_IMGPROC_HPP

__END_DECLS