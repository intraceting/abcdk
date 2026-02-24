/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_BRIGHTNESS_HXX
#define ABCDK_XPU_COMMON_IMGPROC_BRIGHTNESS_HXX

#include "abcdk/xpu/imgproc.h"
#include "../runtime.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**亮度. */
            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_DEVICE void brightness_kernel(int channels, bool packed,
                                                         DT *dst, size_t dst_w, size_t dst_ws, size_t dst_h, MT *alpha, MT *bate,
                                                         size_t tid)
            {
                size_t y = tid / dst_w;
                size_t x = tid % dst_w;

                if (x >= dst_w || y >= dst_h)
                    return;

                for (size_t z = 0; z < channels; z++)
                {
                    size_t dst_offset = util::off<DT>(packed, dst_w, dst_ws, dst_h, channels, 0, x, y, z);

                    *util::ptr<DT>(dst, dst_offset) = util::pixel<DT>((MT)util::obj<DT>(dst, dst_offset) * alpha[z] + bate[z]);
                }
            }

            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_GLOBAL void brightness_3d3d(int channels, bool packed,
                                                       DT *dst, size_t dst_w, size_t dst_ws, size_t dst_h, MT *alpha, MT *bate,
                                                       size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                brightness_kernel<DT, MT>(channels, packed, dst, dst_w, dst_ws, dst_h, alpha, bate,tid);
            }

            template <typename DT, typename MT>
            __ABCDK_XPU_INVOKE_HOST int brightness(int pixfmt, DT *dst, size_t dst_w, size_t dst_ws, size_t dst_h, MT *alpha, MT *bate)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;
#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid,block,dst_w*dst_h);

                brightness_3d3d<DT, MT><<<grid, block>>>(channel, true, dst, dst_w, dst_ws, dst_h, alpha, bate);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < dst_w * dst_h; tid++)
                {
                    brightness_3d3d<DT, MT>(channel, true, dst, dst_w, dst_ws, dst_h, alpha, bate, tid);
                }
#endif //__NVCC__
                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int brightness(AVFrame *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return brightness<uint8_t,float>(dst->format, dst->data[0],dst->width, dst->linesize[0],dst->height,(float*)alpha, (float*)bate);
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_BRIGHTNESS_HXX