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
            int compose(image::metadata_t *panorama, const image::metadata_t *part,
                        size_t overlap_x, size_t overlap_y, size_t overlap_w,
                        const abcdk_xpu_scalar_t *scalar, int optimize_seam)
            {
                assert(panorama->format == AV_PIX_FMT_GRAY8 ||
                       panorama->format == AV_PIX_FMT_RGB24 ||
                       panorama->format == AV_PIX_FMT_BGR24 ||
                       panorama->format == AV_PIX_FMT_RGB32 ||
                       panorama->format == AV_PIX_FMT_BGR32);

                return common::imgproc::compose(panorama, part, overlap_x, overlap_y, overlap_w, scalar, optimize_seam);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
