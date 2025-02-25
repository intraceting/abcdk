/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/util.h"

#ifdef OPENCV_CORE_HPP

abcdk_media_frame_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_media_frame_t *dst;
    cv::Mat tmp_src;
    uint8_t *src_data[4] = {NULL, NULL, NULL, NULL};
    int src_stride[4] = {-1, -1, -1, -1};

    assert(file != NULL);

    tmp_src = cv::imread(file, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (tmp_src.empty())
        return NULL;

    src_data[0] = tmp_src.data;
    src_stride[0] = tmp_src.step;

    dst = abcdk_media_frame_clone2((const uint8_t **)src_data, src_stride, tmp_src.cols, tmp_src.rows, (gray ? ABCDK_MEDIA_PIXFMT_GRAY8 : ABCDK_MEDIA_PIXFMT_BGR24));
    if (!dst)
        return NULL;

    

    return dst;
}

#else //OPENCV_CORE_HPP

abcdk_media_frame_t *abcdk_opencv_image_load(const char *file, int gray)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

#endif //OPENCV_CORE_HPP