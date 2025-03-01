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
    abcdk_media_image_t *dst = NULL;
    cv::Mat src;

    assert(file != NULL);

    src = cv::imread(file, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (src.empty())
        return NULL;

    dst = abcdk_media_image_create(src.cols, src.rows, (gray ? ABCDK_MEDIA_PIXFMT_GRAY8 : ABCDK_MEDIA_PIXFMT_BGR24), 1);
    if (!dst)
        return NULL;

    abcdk_media_image_copy_plane(dst,0, src.data, src.step);

    return dst;
}

#else //OPENCV_CORE_HPP

abcdk_media_image_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

#endif //OPENCV_CORE_HPP