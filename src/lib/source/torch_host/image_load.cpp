/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS


#ifdef OPENCV_IMGCODECS_HPP

abcdk_torch_image_t *abcdk_torch_image_load_host(const char *src, int gray)
{
    abcdk_torch_image_t *dst = NULL;
    cv::Mat tmp_src;

    assert(src != NULL);

    tmp_src = cv::imread(src, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (tmp_src.empty())
        return NULL;

    dst = abcdk_torch_image_create_host(tmp_src.cols, tmp_src.rows, (gray ? ABCDK_TORCH_PIXFMT_GRAY8 : ABCDK_TORCH_PIXFMT_BGR24), 1);
    if (!dst)
        return NULL;

    abcdk_torch_image_copy_plane_host(dst, 0, tmp_src.data, tmp_src.step);

    return dst;
}

#else //OPENCV_IMGCODECS_HPP

abcdk_torch_image_t *abcdk_torch_image_load_host(const char *src, int gray)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

#endif //OPENCV_IMGCODECS_HPP

__END_DECLS