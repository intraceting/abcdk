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
            static int _resize(const image::metadata_t *src, const abcdk_xpu_rect_t *src_roi, image::metadata_t *dst,abcdk_xpu_inter_t inter_mode)
            {
                cv::Mat tmp_dst, tmp_src;

                tmp_dst = common::util::AVFrame2cvMat(dst);
                tmp_src = common::util::AVFrame2cvMat(src);

                return common::imgproc::resize(tmp_src, src_roi, tmp_dst, inter_local_to_opencv(inter_mode));
            }

            int resize(const image::metadata_t *src, const abcdk_xpu_rect_t *src_roi, image::metadata_t *dst, abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _resize(src, src_roi, dst, inter_mode);
            }
        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
