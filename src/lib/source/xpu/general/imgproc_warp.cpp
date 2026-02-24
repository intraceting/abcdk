/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace imgproc
        {
            static int _warp(const image::metadata_t *src, image::metadata_t *dst, const abcdk_xpu_matrix_3x3_t *coeffs, int warp_mode, abcdk_xpu_inter_t inter_mode)
            {
                cv::Mat tmp_dst, tmp_src, tmp_coeffs;

                assert(src->format == dst->format);
                
                tmp_dst = common::util::AVFrame2cvMat(dst);
                tmp_src = common::util::AVFrame2cvMat(src);

                tmp_coeffs.create(3,3,CV_64FC1);
                for (int y = 0; y < tmp_coeffs.rows; y++)
                {
                    for (int x = 0; x < tmp_coeffs.cols; x++)
                    {
                        tmp_coeffs.at<double>(y, x)= coeffs->f64[y][x];
                    }
                }

                return common::imgproc::warp(tmp_src, tmp_dst, tmp_coeffs, warp_mode, inter_local_to_opencv(inter_mode));
            }

            int warp(const image::metadata_t *src, image::metadata_t *dst, const abcdk_xpu_matrix_3x3_t *coeffs, int warp_mode, abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _warp(src, dst, coeffs, warp_mode, inter_mode);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
