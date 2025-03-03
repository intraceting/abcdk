/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/util.h"

#ifdef OPENCV_CORE_HPP

abcdk_torch_image_t *abcdk_opencv_image_load(const char *src, int gray)
{
    abcdk_torch_image_t *dst = NULL;
    cv::Mat tmp_src;

    assert(src != NULL);

    tmp_src = cv::imread(src, (gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR));
    if (tmp_src.empty())
        return NULL;

    dst = abcdk_torch_image_create(tmp_src.cols, tmp_src.rows, (gray ? ABCDK_TORCH_PIXFMT_GRAY8 : ABCDK_TORCH_PIXFMT_BGR24), 1);
    if (!dst)
        return NULL;

    abcdk_torch_image_copy_plane(dst, 0, tmp_src.data, tmp_src.step);

    return dst;
}

int abcdk_opencv_image_save(const char *dst, abcdk_torch_image_t *src)
{
    cv::Mat tmp_src;
    bool chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    int src_depth = abcdk_torch_pixfmt_channels(src->pixfmt);

    tmp_src = cv::Mat(src->height,src->width,CV_8UC(src_depth),src->data[0]);

    chk = cv::imwrite(dst,tmp_src);
    if(!chk)
        return -1;

    return 0;
}

#else //OPENCV_CORE_HPP

abcdk_torch_image_t *abcdk_opencv_image_load(const char *src, int gray)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

int abcdk_opencv_image_save(const char *dst, abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

#endif //OPENCV_CORE_HPP