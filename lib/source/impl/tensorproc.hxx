/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_TENSORPROC_HXX
#define ABCDK_IMPL_TENSORPROC_HXX

#include "general.hxx"

namespace abcdk
{
    namespace tensorproc
    {
        template <class ST, class DT>
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
                size_t src_of = abcdk::general::off<ST>(src_packed, w, src_ws, h, channels, 0, x, y, z);
                size_t dst_of = abcdk::general::off<DT>(dst_packed, w, dst_ws, h, channels, 0, x, y, z);

                ST *src_p = abcdk::general::ptr<ST>(src, src_of);
                DT *dst_p = abcdk::general::ptr<DT>(dst, dst_of);

                if (revert)
                    *dst_p = (((DT)*src_p * std[z]) + mean[z]) * scale[z];
                else
                    *dst_p = (((DT)*src_p / scale[z]) - mean[z]) / std[z];
            }
        }
    } // namespace tensorproc

} // namespace abcdk

#endif // ABCDK_IMPL_TENSORPROC_HXX