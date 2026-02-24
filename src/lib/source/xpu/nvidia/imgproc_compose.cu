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
            __ABCDK_XPU_INVOKE_HOST int _compose(image::metadata_t *panorama, const image::metadata_t *part,
                                             size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                             const abcdk_xpu_scalar_t *scalar, int optimize_seam)
            {
                abcdk_xpu_scalar_t *gpu_scalar = NULL;
                int chk;

                gpu_scalar = memory::clone(0, scalar, sizeof(abcdk_xpu_scalar_t), 1);
                if (!gpu_scalar)
                {
                    memory::free(gpu_scalar, 0);
                    return -1;
                }

                chk = common::imgproc::compose(panorama, part, overlap_x, overlap_y, overlap_w, gpu_scalar, optimize_seam);
                memory::free(gpu_scalar, 0);

                return chk;
            }

            int compose(image::metadata_t *panorama, const image::metadata_t *part,
                        size_t overlap_x, size_t overlap_y, size_t overlap_w,
                        const abcdk_xpu_scalar_t *scalar, int optimize_seam)
            {
                assert(panorama->format == AV_PIX_FMT_GRAY8 ||
                       panorama->format == AV_PIX_FMT_RGB24 ||
                       panorama->format == AV_PIX_FMT_BGR24 ||
                       panorama->format == AV_PIX_FMT_RGB32 ||
                       panorama->format == AV_PIX_FMT_BGR32);

                return _compose(panorama, part, overlap_x, overlap_y, overlap_w, scalar, optimize_seam);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
