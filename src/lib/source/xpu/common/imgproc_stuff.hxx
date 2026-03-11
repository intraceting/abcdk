/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_STUFF_HXX
#define ABCDK_XPU_COMMON_IMGPROC_STUFF_HXX

#include "abcdk/xpu/imgproc.h"
#include "../base.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**填充.*/
            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE void stuff_kernel(int channels, bool packed,
                                                    T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, T *scalar,
                                                    size_t roi_x, size_t roi_y, size_t roi_w, size_t roi_h,
                                                    size_t tid)
            {

                size_t y = tid / dst_w;
                size_t x = tid % dst_w;

                if (x >= dst_w || y >= dst_h)
                    return;

                if (x < roi_x || x > roi_x + roi_w)
                    return;

                if (y < roi_y || y > roi_y + roi_h)
                    return;

                for (size_t z = 0; z < channels; z++)
                {
                    size_t dst_off = util::off<T>(packed, dst_w, dst_ws, dst_h, channels, 0, x, y, z);
                    *util::ptr<T>(dst, dst_off) = util::pixel<T>(scalar[z]);
                }
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_GLOBAL void stuff_3d3d(int channels, bool packed,
                                                  T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, T *scalar,
                                                  size_t roi_x, size_t roi_y, size_t roi_w, size_t roi_h,
                                                  size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                stuff_kernel<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, roi_x, roi_y, roi_w, roi_h, tid);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_HOST int stuff(int pixfmt,
                                          T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, T *scalar,
                                          size_t roi_x, size_t roi_y, size_t roi_w, size_t roi_h)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;
#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid,block,dst_w*dst_h);

                stuff_3d3d<T><<<grid, block>>>(channel, true, dst, dst_w, dst_ws, dst_h, scalar, roi_x, roi_y, roi_w, roi_h);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < dst_w * dst_h; tid++)
                {
                    stuff_3d3d<T>(channel, true, dst, dst_w, dst_ws, dst_h, scalar, roi_x, roi_y, roi_w, roi_h, tid);
                }
#endif //__NVCC__
                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int stuff(AVFrame *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return stuff<uint8_t>(dst->format, dst->data[0], dst->width, dst->linesize[0], dst->height, (uint8_t*)scalar, roi->x, roi->y, roi->width, roi->height);
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_STUFF_HXX