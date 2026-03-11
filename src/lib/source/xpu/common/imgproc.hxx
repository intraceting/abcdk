/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_HXX
#define ABCDK_XPU_COMMON_IMGPROC_HXX

#include "abcdk/xpu/imgproc.h"
#include "../base.in.h"
#include "imgproc_brightness.hxx"
#include "imgproc_compose.hxx"
#include "imgproc_rectangle.hxx"
#include "imgproc_stuff.hxx"
#include "imgproc_line.hxx"
#include "imgproc_mask.hxx"
#include "imgproc_blob.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            cv::InterpolationFlags inter_local_to_opencv(abcdk_xpu_inter_t mode);

            int convert(const AVFrame *src, AVFrame *dst);

            int resize(const cv::Mat &src, const abcdk_xpu_rect_t *src_roi, cv::Mat &dst, cv::InterpolationFlags inter_mode);

            int warp(const cv::Mat &src, cv::Mat &dst, const cv::Mat &coeffs, int warp_mode, cv::InterpolationFlags inter_mode);

            int remap(const cv::Mat &src, cv::Mat &dst, const cv::Mat &xmap, const cv::Mat &ymap, cv::InterpolationFlags inter_mode);

            int undistort(const abcdk_xpu_size_t *size, double alpha, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, cv::Mat &xmap, cv::Mat &ymap);

            cv::Mat find_homography(const abcdk_xpu_point_t src_quad[4], const abcdk_xpu_point_t dst_quad[4]);

            void find_homography(const abcdk_xpu_point_t src_quad[4], const abcdk_xpu_point_t dst_quad[4], abcdk_xpu_matrix_3x3_t *coeffs);

            cv::Mat find_homography_face_112x112(const abcdk_xpu_point_t face_kpt[5]);

            void find_homography_face_112x112(const abcdk_xpu_point_t face_kpt[5], abcdk_xpu_matrix_3x3_t *coeffs);
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_HXX