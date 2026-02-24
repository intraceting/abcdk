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
            __ABCDK_XPU_INVOKE_HOST int _brightness(image::metadata_t *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate)
            {
                abcdk_xpu_scalar_t *gpu_alpha = NULL;
                abcdk_xpu_scalar_t *gpu_bate = NULL;
                int chk;

                gpu_alpha = memory::clone(0, alpha, sizeof(abcdk_xpu_scalar_t), 1);
                gpu_bate = memory::clone(0, bate, sizeof(abcdk_xpu_scalar_t), 1);

                if (!gpu_alpha || !gpu_bate)
                {
                    memory::free(gpu_alpha, 0);
                    memory::free(gpu_bate, 0);
                    return -1;
                }

                chk = common::imgproc::brightness(dst, gpu_alpha, gpu_bate);

                memory::free(gpu_alpha, 0);
                memory::free(gpu_bate, 0);

                return chk;
            }

            int brightness(image::metadata_t *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return _brightness(dst, alpha, bate);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
