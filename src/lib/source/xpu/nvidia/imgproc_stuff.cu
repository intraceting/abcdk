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
            __ABCDK_XPU_INVOKE_HOST int _stuff(image::metadata_t *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar)
            {
                abcdk_xpu_scalar_t *gpu_scalar = NULL;
                int chk;

                gpu_scalar = memory::clone(0, scalar, sizeof(abcdk_xpu_scalar_t), 1);
                if (!gpu_scalar)
                {
                    memory::free(gpu_scalar, 0);
                    return -1;
                }

                chk = common::imgproc::stuff(dst, roi, gpu_scalar);
                memory::free(gpu_scalar, 0);

                return chk;
            }

            int stuff(image::metadata_t *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return _stuff(dst, roi, scalar);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
