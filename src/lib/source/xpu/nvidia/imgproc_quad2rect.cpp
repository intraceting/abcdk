/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgproc
        {
            static int _quad2rect(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4], image::metadata_t *dst, abcdk_xpu_inter_t inter_mode)
            {
                return warp_quad2quad(src,src_quad,dst,NULL,1,inter_mode);
            }

            int quad2rect(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4], image::metadata_t *dst, abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _quad2rect(src, src_quad, dst, inter_mode);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
