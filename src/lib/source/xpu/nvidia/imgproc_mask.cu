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
            __ABCDK_XPU_INVOKE_HOST int _mask(image::metadata_t *dst, const image::metadata_t *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not)
            {
                abcdk_xpu_scalar_t *gpu_color = NULL;
                int chk;

                gpu_color = memory::clone(0, color, sizeof(abcdk_xpu_scalar_t), 1);
                if (!gpu_color)
                {
                    memory::free(gpu_color, 0);
                    return -1;
                }

                chk = common::imgproc::mask(dst, feature, threshold, gpu_color, less_or_not);
                memory::free(gpu_color, 0);

                return chk;
            }

            int mask(image::metadata_t *dst, const image::metadata_t *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                assert(feature->format == AV_PIX_FMT_GRAYF32);

                return _mask(dst, feature, threshold, color, less_or_not);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
