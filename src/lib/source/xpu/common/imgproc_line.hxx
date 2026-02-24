/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_LINE_HXX
#define ABCDK_XPU_COMMON_IMGPROC_LINE_HXX

#include "abcdk/xpu/imgproc.h"
#include "../runtime.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**画线段.*/
            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE void line_kernel(int channels, bool packed,
                                                   T *dst, size_t w, size_t ws, size_t h,
                                                   int x1, int y1, int x2, int y2,
                                                   T *color, int weight,
                                                   size_t tid)
            {
                int y = tid / w; // 必须是有符号的.
                int x = tid % w; // 必须是有符号的.

                if (x >= w || y >= h)
                    return;

                bool chk_bool = util::point_on_line(x1, y1, x2, y2, x, y, weight);
                if (!chk_bool)
                    return;

                /*填充颜色.*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t off = util::off<T>(packed, w, ws, h, channels, 0, x, y, z);
                    *util::ptr<T>(dst, off) = util::pixel<T>(color[z]);
                }
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_GLOBAL void line_3d3d(int channels, bool packed,
                                                 T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                 int x1, int y1, int x2, int y2,
                                                 T *color, int weight,
                                                 size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                line_kernel<T>(channels, packed, dst, dst_w, dst_ws, dst_h, x1, y1, x2, y2, color, weight, tid);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_HOST int line(int pixfmt,
                                         T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                         int x1, int y1, int x2, int y2,
                                         T *color, int weight)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;
#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid, block, dst_w * dst_h);

                line_3d3d<T><<<grid, block>>>(channel, true, dst, dst_w, dst_ws, dst_h, x1, y1, x2, y2, color, weight);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < dst_w * dst_h; tid++)
                {
                    line_3d3d<T>(channel, true, dst, dst_w, dst_ws, dst_h, x1, y1, x2, y2, color, weight, tid);
                }
#endif //__NVCC__
                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int line(AVFrame *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                                         const abcdk_xpu_scalar_t *color, int weight)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return line<uint8_t>(dst->format, dst->data[0], dst->width, dst->linesize[0], dst->height, p1->x, p1->y, p2->x, p2->y, (uint8_t *)color, weight);
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_DRAWLINE_HXX