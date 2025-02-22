/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_TENSORPROC_BLOB_HXX
#define ABCDK_GENERIC_TENSORPROC_BLOB_HXX

#include "util.hxx"

namespace abcdk
{
    namespace generic
    {
        namespace tensorproc
        {
            template <typename ST, typename DT>
            ABCDK_INVOKE_DEVICE void blob_kernel(int channels, bool revert,
                                                 bool dst_packed, DT *dst, size_t dst_ws,
                                                 bool src_packed, ST *src, size_t src_ws,
                                                 size_t w, size_t h, float *scale, float *mean, float *std,
                                                 size_t tid)
            {

                size_t y = tid / w;
                size_t x = tid % w;

                if (x >= w || y >= h)
                    return;

                for (size_t z = 0; z < channels; z++)
                {
                    size_t src_of = abcdk::generic::util::off<ST>(src_packed, w, src_ws, h, channels, 0, x, y, z);
                    size_t dst_of = abcdk::generic::util::off<DT>(dst_packed, w, dst_ws, h, channels, 0, x, y, z);

                    ST *src_p = abcdk::generic::util::ptr<ST>(src, src_of);
                    DT *dst_p = abcdk::generic::util::ptr<DT>(dst, dst_of);

                    if (revert)
                        *dst_p = (DT)((((float)(*src_p) * std[z]) + mean[z]) * scale[z]);
                    else
                        *dst_p = (DT)((((float)(*src_p) / scale[z]) - mean[z]) / std[z]);
                }
            }

            /**
             * 数值转换。
             *
             * @param [in] scale 系数。
             * @param [in] mean 均值。
             * @param [in] std 方差。
             */
            template <typename ST, typename DT>
            ABCDK_INVOKE_HOST void blob(int channels, bool revert,
                                        DT *dst, size_t dst_ws, bool dst_nchw,
                                        ST *src, size_t src_ws, bool src_nchw,
                                        size_t c, size_t w, size_t h,
                                        float *scale, float *mean, float *std)
            {
                for (size_t i = 0; i < w * h; i++)
                {
                    blob_kernel<ST, DT>(channels, revert, dst, dst_ws, dst_nchw, src, src_ws, src_nchw, c, w, h, scale, mean, std, i);
                }
            }
        } // namespace tensorproc
    } // namespace generic
} // namespace abcdk

#endif // ABCDK_GENERIC_TENSORPROC_BLOB_HXX