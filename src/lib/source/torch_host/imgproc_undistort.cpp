/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS

#ifdef OPENCV_CALIB3D_HPP

static int _abcdk_torch_imgproc_undistort_8u_host(int channels, int packed,
                                                  uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                  const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h,
                                                  const float camera_matrix[3][3], const float dist_coeffs[5])
{
    cv::Mat tmp_dst, tmp_src;
    cv::Mat tmp_camera_matrix, tmp_dist_coeffs;

    tmp_dst = cv::Mat(dst_h, dst_w, CV_8UC(channels), (void *)dst, dst_ws);
    tmp_src = cv::Mat(src_h, src_w, CV_8UC(channels), (void *)src, src_ws);

    tmp_camera_matrix = cv::Mat(3, 3, CV_32FC1, (void *)camera_matrix, 3 * sizeof(float));
    tmp_dist_coeffs = cv::Mat(1, 5, CV_32FC1, (void *)dist_coeffs, 5 * sizeof(float));

    cv::undistort(tmp_src, tmp_dst, tmp_camera_matrix, tmp_dist_coeffs);

    return 0;
}

int abcdk_torch_imgproc_undistort_host(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src, const float camera_matrix[3][3], const float dist_coeffs[5])
{
    int dst_depth;

    assert(dst != NULL && src != NULL && camera_matrix != NULL && dist_coeffs != NULL);
    assert(dst->pixfmt == src->pixfmt);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_undistort_8u_host(dst_depth, true,
                                                  dst->data[0], dst->width, dst->stride[0], dst->height,
                                                  src->data[0], src->width, src->stride[0], src->height,
                                                  camera_matrix, dist_coeffs);
}

#else // OPENCV_CALIB3D_HPP

int abcdk_torch_imgproc_undistort_host(abcdk_torch_image_t *dst, const abcdk_torch_image_t *src, const float camera_matrix[3][3], const float dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_CALIB3D_HPP

__END_DECLS