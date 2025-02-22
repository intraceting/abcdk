/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_IMAGEPROC_DRAWRECT_HXX
#define ABCDK_GENERIC_IMAGEPROC_DRAWRECT_HXX

#include "util.hxx"

namespace abcdk
{
    namespace generic
    {
        namespace imageproc
        {
            template <typename T>
            ABCDK_INVOKE_DEVICE void drawrect_kernel(int channels, bool packed,
                                                     T *dst, size_t w, size_t ws, size_t h,
                                                     T *color, int weight, int *corner,
                                                     size_t tid)
            {
                size_t y = tid / w;
                size_t x = tid % w;

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

                /*为0表示不需要填充颜色。*/
                if (chk == 0)
                    return;

                /*填充颜色。*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t off = abcdk::generic::util::off<T>(packed, w, ws, h, channels, 0, x, y, z);
                    *abcdk::generic::util::ptr<T>(dst, off) = color[z];
                }
            }

            /**
             * 画矩形框。
             *
             * @param corner 左上，右下。[x1][y1][x2][y2]
             */
            template <typename T>
            ABCDK_INVOKE_HOST void drawrect(int channels, bool packed, T *dst, size_t w, size_t ws, size_t h, T *color, int weight, int *corner)
            {

                // #pragma omp parallel
                for (size_t i = 0; i < w * h; i++)
                {
                    drawrect_kernel<T>(channels, packed, dst, w, ws, h, color, weight, corner, i);
                }
            }

        } // namespace imageproc
    } //    namespace generic
} // namespace abcdk

#endif // ABCDK_GENERIC_IMAGEPROC_DRAWRECT_HXX