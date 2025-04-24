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

int abcdk_torch_imgproc_undistort_buildmap_host(abcdk_torch_image_t **xmap, abcdk_torch_image_t **ymap,
                                                abcdk_torch_size_t *size, double alpha,
                                                const double camera_matrix[3][3], const double dist_coeffs[5])
{
    cv::Size tmp_size;
    cv::Mat tmp_camera_matrix, tmp_dist_coeffs;
    cv::Mat tmp_xmap, tmp_ymap;
    int chk;

    assert(xmap != NULL && ymap != NULL && size != NULL && alpha >= 0.0 && alpha <= 1.0 && camera_matrix != NULL && dist_coeffs != NULL);
    assert(size->width > 0 && size->height > 1);

    tmp_size.width = size->width;
    tmp_size.height = size->height;

    tmp_camera_matrix = cv::Mat(3, 3, CV_64FC1, (void *)camera_matrix, 3 * sizeof(double));
    tmp_dist_coeffs = cv::Mat(1, 5, CV_64FC1, (void *)dist_coeffs, 5 * sizeof(double));

    cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // 不做旋转。
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(tmp_camera_matrix, tmp_dist_coeffs, tmp_size, alpha, tmp_size, 0);

    // 生成映射表，注意 xmap 和 ymap 都是 CV_32FC1（CUDA兼容）
    cv::initUndistortRectifyMap(tmp_camera_matrix, tmp_dist_coeffs, R, newCameraMatrix, tmp_size, CV_32FC1, tmp_xmap, tmp_ymap);

    if (tmp_xmap.empty() || tmp_ymap.empty())
        return -1;

    chk = abcdk_torch_image_reset_host(xmap, tmp_xmap.cols, tmp_xmap.rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);
    if (chk != 0)
        return -2;

    chk = abcdk_torch_image_reset_host(ymap, tmp_ymap.cols, tmp_ymap.rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);
    if (chk != 0)
        return -3;

    abcdk_torch_image_copy_plane_host(*xmap, 0, tmp_xmap.data, tmp_xmap.step);
    abcdk_torch_image_copy_plane_host(*ymap, 0, tmp_ymap.data, tmp_ymap.step);

    return 0;
}

#else // OPENCV_CALIB3D_HPP

int abcdk_torch_imgproc_undistort_buildmap_host(abcdk_torch_image_t **xmap, abcdk_torch_image_t **ymap,
                                                abcdk_torch_size_t *size, double alpha,
                                                const double camera_matrix[3][3], const double dist_coeffs[5])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_CALIB3D_HPP

__END_DECLS