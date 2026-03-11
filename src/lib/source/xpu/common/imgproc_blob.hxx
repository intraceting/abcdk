/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_BLOB_HXX
#define ABCDK_XPU_COMMON_IMGPROC_BLOB_HXX

#include "abcdk/xpu/imgproc.h"
#include "../base.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {

            /**
             * 数值转换.
             *
             * @param [in] scale 系数.
             * @param [in] mean 均值.
             * @param [in] std 方差.
             */
            template <typename DT, typename ST, typename BT>
            __ABCDK_XPU_INVOKE_DEVICE void blob_kernel(bool dst_packed, DT *dst, size_t dst_ws, bool dst_c_invert,
                                                   bool src_packed, ST *src, size_t src_ws, bool src_c_invert,
                                                   size_t b, size_t w, size_t h, size_t c,
                                                   BT *scale, BT *mean, BT *std,
                                                   bool revert, size_t tid)
            {
                size_t n, x, y, z;

                /*源和目标索引算法必须一样.*/
                util::idx2nyxz(tid, h, w, c, n, y, x, z);

                if (n >= b || x >= w || y >= h || z >= c)
                    return;

                size_t src_z = (src_c_invert ? c - z : z);
                size_t dst_z = (dst_c_invert ? c - z : z);

                size_t src_of = util::off<ST>(src_packed, w, src_ws, h, c, n, x, y, src_z);
                size_t dst_of = util::off<DT>(dst_packed, w, dst_ws, h, c, n, x, y, dst_z);

                ST *src_p = util::ptr<ST>(src, src_of);
                DT *dst_p = util::ptr<DT>(dst, dst_of);

                if (revert)
                    *dst_p = (DT)((((BT)(*src_p) * std[z]) + mean[z]) * scale[z]);
                else
                    *dst_p = (DT)((((BT)(*src_p) / scale[z]) - mean[z]) / std[z]);
            }

            template <typename DT, typename ST, typename BT>
            __ABCDK_XPU_INVOKE_GLOBAL void blob_3d3d(bool dst_packed, DT *dst, size_t dst_ws, bool dst_c_invert,
                                                 bool src_packed, ST *src, size_t src_ws, bool src_c_invert,
                                                 size_t b, size_t w, size_t h, size_t c,
                                                 BT *scale, BT *mean, BT *std,
                                                 bool revert, size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                blob_kernel<DT, ST, BT>(dst_packed, dst, dst_ws, dst_c_invert,
                                        src_packed, src, src_ws, src_c_invert,
                                        b, w, h, c,
                                        scale, mean, std,
                                        revert, tid);
            }

            template <typename DT, typename ST, typename BT>
            __ABCDK_XPU_INVOKE_HOST int blob(bool dst_packed, DT *dst, size_t dst_ws, bool dst_c_invert,
                                         bool src_packed, ST *src, size_t src_ws, bool src_c_invert,
                                         size_t b, size_t w, size_t h, size_t c,
                                         BT *scale, BT *mean, BT *std,
                                         bool revert)
            {

#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid, block, b * w * h * c);

                blob_3d3d<DT, ST, BT><<<grid, block>>>(dst_packed, dst, dst_ws, dst_c_invert,
                                                       src_packed, src, src_ws, src_c_invert,
                                                       b, w, h, c,
                                                       scale, mean, std,
                                                       revert);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < b * w * h * c; tid++)
                {
                    blob_3d3d<DT, ST, BT>(dst_packed, dst, dst_ws, dst_c_invert,
                                          src_packed, src, src_ws, src_c_invert,
                                          b, w, h, c,
                                          scale, mean, std,
                                          revert, tid);
                }
#endif //__NVCC__

                return 0;
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_BLOB_HXX