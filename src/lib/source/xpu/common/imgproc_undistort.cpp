/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            int undistort(const abcdk_xpu_size_t *size, double alpha, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, cv::Mat &xmap, cv::Mat &ymap)
            {
                cv::Size tmp_size;
                int chk;

                tmp_size.width = size->width;
                tmp_size.height = size->height;

                cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // 不做旋转.
                cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, tmp_size, alpha, tmp_size, 0);

                // 生成映射表, xmap 和 ymap 都是 CV_32FC1(CUDA兼容).
                cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, R, newCameraMatrix, tmp_size, CV_32FC1, xmap, ymap);

                return 0;
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
