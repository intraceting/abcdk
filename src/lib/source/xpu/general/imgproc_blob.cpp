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
            int blob_8u_to_32f(int dst_packed, float *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const uint8_t *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std)
            {
                return common::imgproc::blob<float, uint8_t, float>(dst_packed, (float *)dst, dst_ws, dst_c_invert,
                                                                    src_packed, (uint8_t *)src, src_ws, src_c_invert,
                                                                    b, w, h, c,
                                                                    (float *)scale, (float *)mean, (float *)std,
                                                                    false);
            }

            int blob_32f_to_8u(int dst_packed, uint8_t *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const float *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std)
            {
                return common::imgproc::blob<uint8_t, float, float>(dst_packed, (uint8_t *)dst, dst_ws, dst_c_invert,
                                                                    src_packed, (float *)src, src_ws, src_c_invert,
                                                                    b, w, h, c,
                                                                    (float *)scale, (float *)mean, (float *)std,
                                                                    true);
            }
        } // namespace imgproc
    } // namespace general
} // namespace abcdk_xpu
