/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_RECTANGLE_HXX
#define ABCDK_XPU_COMMON_IMGPROC_RECTANGLE_HXX

#include "abcdk/xpu/imgproc.h"
#include "../runtime.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**
             * 画矩形框.
             *
             * @param corner 左上, 右下.[x1][y1][x2][y2]
             */
            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE void rectangle_kernel(int channels, bool packed,
                                                        T *dst, size_t w, size_t ws, size_t h, T *color, int weight, int *corner,
                                                        size_t tid)
            {
                int y = tid / w; // 必须是有符号的.
                int x = tid % w; // 必须是有符号的.

                if (x >= w || y >= h)
                    return;

                int x1 = corner[0];
                int y1 = corner[1];
                int x2 = corner[2];
                int y2 = corner[3];

                int chk = 0x00;

                /*上边*/
                if (x >= x1 && y >= y1 && x <= x2 && y <= y1 + weight && y <= y2)
                    chk |= 0x01;

                /*下边*/
                if (x >= x1 && y <= y2 && x <= x2 && y >= y2 - weight && y >= y1)
                    chk |= 0x02;

                /*左边*/
                if (x >= x1 && y >= y1 && x <= x1 + weight && y <= y2 && x <= x2)
                    chk |= 0x04;

                /*右边*/
                if (x >= x2 - weight && y >= y1 && x <= x2 && y <= y2 && x >= x1)
                    chk |= 0x08;

                /*为0表示不需要填充颜色.*/
                if (chk == 0)
                    return;

                /*填充颜色.*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t off = util::off<T>(packed, w, ws, h, channels, 0, x, y, z);
                    *util::ptr<T>(dst, off) = util::pixel<T>(color[z]);
                }
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_GLOBAL void rectangle_3d3d(int channels, bool packed,
                                                      T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, T *color, int weight, int *corner,
                                                      size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                rectangle_kernel<T>(channels, packed, dst, dst_w, dst_ws, dst_h, color, weight, corner, tid);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_HOST int rectangle(int pixfmt,
                                              T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, T *color, int weight, int *corner)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;
#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid,block,dst_w*dst_h);

                rectangle_3d3d<T><<<grid, block>>>(channel, true, dst, dst_w, dst_ws, dst_h, color, weight, corner);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < dst_w * dst_h; tid++)
                {
                    rectangle_3d3d<T>(channel, true, dst, dst_w, dst_ws, dst_h, color, weight, corner, tid);
                }
#endif //__NVCC__
                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int rectangle(AVFrame *dst, const abcdk_xpu_scalar_t *corner, int weight, const abcdk_xpu_scalar_t *color)
            {
                assert(dst->format == AV_PIX_FMT_GRAY8 ||
                       dst->format == AV_PIX_FMT_RGB24 ||
                       dst->format == AV_PIX_FMT_BGR24 ||
                       dst->format == AV_PIX_FMT_RGB32 ||
                       dst->format == AV_PIX_FMT_BGR32);

                return rectangle<uint8_t>(dst->format, dst->data[0], dst->width, dst->linesize[0], dst->height, (uint8_t *)color, weight, (int *)corner);
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_RECTANGLE_HXX