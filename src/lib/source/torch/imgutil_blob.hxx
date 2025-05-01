/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMGUTIL_BLOB_HXX
#define ABCDK_TORCH_IMGUTIL_BLOB_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace imgutil
        {

            /**
             * 数值转换。
             *
             * @param [in] scale 系数。
             * @param [in] mean 均值。
             * @param [in] std 方差。
             */
            template <typename DT, typename ST, typename BT>
            ABCDK_TORCH_INVOKE_DEVICE void blob(bool dst_packed, DT *dst, size_t dst_ws,
                                                bool src_packed, ST *src, size_t src_ws,
                                                size_t b, size_t w, size_t h, size_t c,
                                                BT *scale, BT *mean, BT *std,
                                                bool revert, size_t tid)
            {
                size_t n, x, y, z;

                abcdk::torch::util::idx2nyxz(tid,h,w,c,n,y,x,z);

                if (n >= b || x >= w || y >= h || z >= c)
                    return;

                /*源和目标索引算法必须一样。*/

                size_t src_of = abcdk::torch::util::off<ST>(src_packed, w, src_ws, h, c, n, x, y, z);
                size_t dst_of = abcdk::torch::util::off<DT>(dst_packed, w, dst_ws, h, c, n, x, y, z);

                ST *src_p = abcdk::torch::util::ptr<ST>(src, src_of);
                DT *dst_p = abcdk::torch::util::ptr<DT>(dst, dst_of);

                if (revert)
                    *dst_p = (DT)((((BT)(*src_p) * std[z]) + mean[z]) * scale[z]);
                else
                    *dst_p = (DT)((((BT)(*src_p) / scale[z]) - mean[z]) / std[z]);
            }

        } // namespace imgutil
    } // namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_IMGUTIL_BLOB_HXX