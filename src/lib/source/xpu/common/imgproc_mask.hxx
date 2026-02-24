/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_MASK_HXX
#define ABCDK_XPU_COMMON_IMGPROC_MASK_HXX

#include "abcdk/xpu/imgproc.h"
#include "../runtime.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**画掩码. */
            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_DEVICE void mask_kernel(int channels, bool packed,
                                                   DT *dst, size_t dst_ws, MT *feature, size_t feature_ws, size_t w, size_t h, MT threshold, DT *color, int less_or_not,
                                                   size_t tid)
            {
                size_t y = tid / w;
                size_t x = tid % w;

                if (x >= w || y >= h)
                    return;
                
                //
                MT pot = util::obj<MT>(feature, packed, w, feature_ws, h, 1, 0, x, y, 0);

                /*超出范围的不要.*/
                if (less_or_not ? (pot < threshold) : (pot > threshold))
                    return;

                /*填充颜色.*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t dst_off = util::off<DT>(packed, w, dst_ws, h, channels, 0, x, y, z);
                    *util::ptr<DT>(dst, dst_off) = (util::obj<DT>(dst, dst_off) * 0.5 + util::pixel<DT>(color[z]) * 0.5);
                }
            }

            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_GLOBAL void mask_3d3d(int channels, bool packed,
                                                 DT *dst, size_t dst_ws, MT *feature, size_t feature_ws, size_t w, size_t h, MT threshold, DT *color,int less_or_not,
                                                 size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                mask_kernel<DT,MT>(channels, packed, dst, dst_ws, feature, feature_ws, w, h, threshold, color, less_or_not, tid);
            }

            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_HOST int mask(int pixfmt,  DT *dst, size_t dst_ws, MT *feature, size_t feature_ws, size_t w, size_t h, MT threshold, DT *color, int less_or_not)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;

#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid,block,w*h);

                mask_3d3d<DT,MT><<<grid, block>>>(channel, true, dst, dst_ws, feature, feature_ws, w, h, threshold, color, less_or_not);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < w * h; tid++)
                {
                    mask_3d3d<DT,MT>(channel, true, dst, dst_ws, feature, feature_ws, w, h, threshold, color, less_or_not, tid);
                }
#endif //__NVCC__

                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int mask(AVFrame *dst, const AVFrame *feature, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                assert(feature->format == AV_PIX_FMT_GRAYF32);

                return mask<uint8_t, float>(dst->format,
                                            (uint8_t *)dst->data[0], dst->linesize[0],
                                            (float *)feature->data[0], feature->linesize[0],
                                            dst->width, dst->height, threshold, (uint8_t *)color, less_or_not);
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_DRAWMASK_HXX