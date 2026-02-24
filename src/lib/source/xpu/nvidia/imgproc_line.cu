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
            __ABCDK_XPU_INVOKE_HOST int _line(image::metadata_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                                          const abcdk_xpu_scalar_t *color, int weight)
            {
                abcdk_xpu_scalar_t *gpu_color = NULL;
                int chk;

                gpu_color = memory::clone(0, color, sizeof(abcdk_xpu_scalar_t), 1);
                if (!gpu_color)
                {
                    memory::free(gpu_color, 0);
                    return -1;
                }

                chk = common::imgproc::line(dst, p1, p2, gpu_color, weight);
                memory::free(gpu_color, 0);

                return chk;
            }

            int line(image::metadata_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                     const abcdk_xpu_scalar_t *color, int weight)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32 ||
                       dst->format == AV_PIX_FMT_GRAYF32);

                return _line(dst, p1, p2, color, weight);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
