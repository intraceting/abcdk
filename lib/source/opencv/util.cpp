/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/util.h"

#ifdef OPENCV_CORE_HPP

abcdk_media_image_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_media_image_t *dst = NULL, tmp_src = {0};
    cv::Mat src;

    assert(file != NULL);

    src = cv::imread(file, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (src.empty())
        return NULL;

    tmp_src.tag = ABCDK_MEDIA_TAG_HOST;
    tmp_src.data[0] = src.data;
    tmp_src.data[1] = tmp_src.data[2] = tmp_src.data[3] = NULL;
    tmp_src.stride[0] = src.step;
    tmp_src.stride[1] = tmp_src.stride[2] = tmp_src.stride[3] = -1;
    tmp_src.width = src.cols;
    tmp_src.height = src.rows;
    tmp_src.pixfmt = (gray ? ABCDK_MEDIA_PIXFMT_GRAY8 : ABCDK_MEDIA_PIXFMT_BGR24);

    dst = abcdk_media_image_clone(&tmp_src);
    if (!dst)
        return NULL;

    return dst;
}

#else //OPENCV_CORE_HPP

abcdk_media_image_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

#endif //OPENCV_CORE_HPP