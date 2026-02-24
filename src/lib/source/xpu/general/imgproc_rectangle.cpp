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
            static int _rectangle(image::metadata_t *dst, const abcdk_xpu_rect_t *rect, int weight, const abcdk_xpu_scalar_t *color)
            {
                abcdk_xpu_scalar_t corner = {0};
                
                corner.i32[0] = rect->x;
                corner.i32[1] = rect->y;
                corner.i32[2] = rect->x + rect->width;
                corner.i32[3] = rect->y + rect->height;

                return common::imgproc::rectangle(dst,&corner,weight,color);
            }

            int rectangle(image::metadata_t *dst, const abcdk_xpu_rect_t *rect, int weight, const abcdk_xpu_scalar_t *color)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return _rectangle(dst, rect, weight, color);
            }

        } // namespace image
    } // namespace general

} // namespace abcdk_xpu
