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
            int mask(image::metadata_t *dst, const image::metadata_t *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                assert(feature->format == AV_PIX_FMT_GRAYF32);

                return common::imgproc::mask(dst, feature, threshold, color ,less_or_not);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
