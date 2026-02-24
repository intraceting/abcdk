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

            int line(image::metadata_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                     const abcdk_xpu_scalar_t *color, int weight)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return common::imgproc::line(dst, p1, p2, color, weight);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
