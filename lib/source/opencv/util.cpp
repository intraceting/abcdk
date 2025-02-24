/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/util.h"

#ifdef OPENCV_CORE_HPP

abcdk_ndarray_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_ndarray_t *dst;
    cv::Mat tmp_src;

    assert(file != NULL);

    tmp_src = cv::imread(file, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (tmp_src.empty())
        return NULL;

    dst = abcdk_ndarray_clone2(tmp_src.data, tmp_src.step, ABCDK_NDARRAY_NHWC, 1, tmp_src.cols, tmp_src.rows, tmp_src.channels(), 1);
    if (!dst)
        return NULL;

    return dst;
}

#else //OPENCV_CORE_HPP

abcdk_ndarray_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

#endif //OPENCV_CORE_HPP