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
            static int _warp_quad2quad(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4],
                                       image::metadata_t *dst, const abcdk_xpu_point_t dst_quad[4],
                                       int warp_mode, abcdk_xpu_inter_t inter_mode)
            {
                
                cv::Mat tmp_dst, tmp_src;
                abcdk_xpu_point_t tmp_dst_quad[4], tmp_src_quad[4];
                cv::Mat coeffs;

                assert(src->format == dst->format);

                tmp_dst = common::util::AVFrame2cvMat(dst);
                tmp_src = common::util::AVFrame2cvMat(src);
                
                if (!dst_quad)
                {
                    tmp_dst_quad[0].x = 0;
                    tmp_dst_quad[0].y = 0;
                    tmp_dst_quad[1].x = dst->width - 1;
                    tmp_dst_quad[1].y = 0;
                    tmp_dst_quad[2].x = dst->width - 1;
                    tmp_dst_quad[2].y = dst->height - 1;
                    tmp_dst_quad[3].x = 0;
                    tmp_dst_quad[3].y = dst->height - 1;

                    return _warp_quad2quad(src, src_quad, dst, tmp_dst_quad, warp_mode, inter_mode);
                }

                if (!src_quad)
                {
                    tmp_src_quad[0].x = 0;
                    tmp_src_quad[0].y = 0;
                    tmp_src_quad[1].x = src->width - 1;
                    tmp_src_quad[1].y = 0;
                    tmp_src_quad[2].x = src->width - 1;
                    tmp_src_quad[2].y = src->height - 1;
                    tmp_src_quad[3].x = 0;
                    tmp_src_quad[3].y = src->height - 1;

                    return _warp_quad2quad(src, tmp_src_quad, dst, dst_quad, warp_mode, inter_mode);
                }

                coeffs = common::imgproc::find_homography(src_quad, dst_quad);

                return common::imgproc::warp(tmp_src, tmp_dst, coeffs, warp_mode, inter_local_to_opencv(inter_mode));
            }

            int warp_quad2quad(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4],
                               image::metadata_t *dst, const abcdk_xpu_point_t dst_quad[4],
                               int warp_mode, abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _warp_quad2quad(src, src_quad, dst, dst_quad, warp_mode, inter_mode);
            }

        } // namespace image
    } // namespace general
} // namespace abcdk_xpu
