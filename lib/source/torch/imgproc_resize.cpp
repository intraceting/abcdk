/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"

__BEGIN_DECLS

#ifdef OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_resize_8u(int channels, int packed,
                                  uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                  const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                  int keep_aspect_ratio, int inter_mode)
{
    abcdk_torch_size_t tmp_src_size = {0};
    abcdk_torch_rect_t tmp_dst_roi = {0}, tmp_src_roi = {0};
    abcdk_resize_t tmp_param = {0};
    cv::Mat tmp_dst, tmp_src;
    cv::Mat tmp_dst2;
    cv::Size tmp_dst2_size;

    assert(channels == 1 || channels == 3 || channels == 4);
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0);
    assert(src != NULL && src_w > 0 && src_ws > 0 && src_h > 0);

    tmp_dst_roi.x = (dst_roi ? dst_roi->x : 0);
    tmp_dst_roi.y = (dst_roi ? dst_roi->y : 0);
    tmp_dst_roi.width = (dst_roi ? dst_roi->width : dst_w);
    tmp_dst_roi.height = (dst_roi ? dst_roi->height : dst_h);

    tmp_src_size.width = src_w;
    tmp_src_size.height = src_h;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src_w);
    tmp_src_roi.height = (src_roi ? src_roi->height : src_h);

    abcdk_resize_ratio_2d(&tmp_param,tmp_src_roi.width,tmp_src_roi.height,tmp_dst_roi.width,tmp_dst_roi.height,keep_aspect_ratio);

    tmp_dst = cv::Mat(dst_h, dst_w, CV_8UC(channels), (void *)dst, dst_ws);
    tmp_src = cv::Mat(src_h, src_w, CV_8UC(channels), (void *)src, src_ws);

    tmp_dst2_size.width = abcdk_resize_src2dst_2d(&tmp_param,src_w,1) - tmp_param.x_shift;
    tmp_dst2_size.height = abcdk_resize_src2dst_2d(&tmp_param,src_h,0) - tmp_param.y_shift;

    tmp_dst2 = tmp_dst(cv::Rect(tmp_param.x_shift, tmp_param.y_shift, tmp_dst2_size.width, tmp_dst2_size.height));

    cv::resize(tmp_src, tmp_dst2, tmp_dst2_size, 0, 0, inter_mode);

    return 0;
}


#else // OPENCV_IMGPROC_HPP

int abcdk_torch_imgproc_resize_8u(int channels, int packed,
                                  uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                  const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                  int keep_aspect_ratio, int inter_mode)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

#endif //OPENCV_IMGPROC_HPP

__END_DECLS