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
            template <typename DT, typename ST>
            __ABCDK_XPU_INVOKE_HOST int _blob(int dst_packed, DT *dst, size_t dst_ws, int dst_c_invert,
                                          int src_packed, const ST *src, size_t src_ws, int src_c_invert,
                                          size_t b, size_t w, size_t h, size_t c,
                                          const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std,
                                          bool revert)
            {
                abcdk_xpu_scalar_t *gpu_scale = NULL;
                abcdk_xpu_scalar_t *gpu_mean = NULL;
                abcdk_xpu_scalar_t *gpu_std = NULL;
                int chk;

                gpu_scale = memory::clone(0, scale, sizeof(abcdk_xpu_scalar_t), 1);
                gpu_mean = memory::clone(0, mean, sizeof(abcdk_xpu_scalar_t), 1);
                gpu_std = memory::clone(0, std, sizeof(abcdk_xpu_scalar_t), 1);
                if (!gpu_scale || !gpu_mean || !gpu_std)
                {
                    memory::free(gpu_scale, 0);
                    memory::free(gpu_mean, 0);
                    memory::free(gpu_std, 0);
                    return -1;
                }

                chk = common::imgproc::blob<DT, ST, float>(dst_packed, dst, dst_ws, dst_c_invert,
                                                           src_packed, (ST *)src, src_ws, src_c_invert,
                                                           b, w, h, c,
                                                           (float *)gpu_scale, (float *)gpu_mean, (float *)gpu_std,
                                                           revert);
                memory::free(gpu_scale, 0);
                memory::free(gpu_mean, 0);
                memory::free(gpu_std, 0);

                return chk;
            }

            int blob_8u_to_32f(int dst_packed, float *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const uint8_t *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std)
            {
                return _blob<float, uint8_t>(dst_packed, dst, dst_ws, dst_c_invert,
                                             src_packed, src, src_ws, src_c_invert,
                                             b, w, h, c,
                                             scale, mean, std,
                                             false);
            }

            int blob_32f_to_8u(int dst_packed, uint8_t *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const float *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std)
            {
                return _blob<uint8_t, float>(dst_packed, dst, dst_ws, dst_c_invert,
                                             src_packed, src, src_ws, src_c_invert,
                                             b, w, h, c,
                                             scale, mean, std,
                                             true);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
