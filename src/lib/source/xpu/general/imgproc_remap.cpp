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
            static int _remap(const image::metadata_t *src, image::metadata_t *dst,
                              const image::metadata_t *xmap, const image::metadata_t *ymap,
                              abcdk_xpu_inter_t inter_mode)
            {
                cv::Mat tmp_src, tmp_dst;
                cv::Mat tmp_xmap, tmp_ymap;
                int chk;

                
                tmp_dst = common::util::AVFrame2cvMat(dst);
                tmp_src = common::util::AVFrame2cvMat(src);

                tmp_xmap = common::util::AVFrame2cvMat(xmap);
                tmp_ymap = common::util::AVFrame2cvMat(ymap);

                return common::imgproc::remap(tmp_src, tmp_dst, tmp_xmap, tmp_ymap, inter_local_to_opencv(inter_mode));
            }

            int remap(const image::metadata_t *src, image::metadata_t *dst,
                      const image::metadata_t *xmap, const image::metadata_t *ymap,
                      abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                assert(xmap->format == AV_PIX_FMT_GRAYF32 && ymap->format == AV_PIX_FMT_GRAYF32);

                return _remap(src, dst, xmap, ymap, inter_mode);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
