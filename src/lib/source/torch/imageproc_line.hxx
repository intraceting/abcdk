/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMAGEPROC_LINE_HXX
#define ABCDK_TORCH_IMAGEPROC_LINE_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace imageproc
        {
            /** 画线段。*/
            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE void line(int channels, bool packed,
                                                T *dst, size_t w, size_t ws, size_t h,
                                                int x1, int y1, int x2, int y2,
                                                uint32_t *color, int weight,
                                                size_t tid)
            {
                int y = tid / w; // 必须是有符号的。
                int x = tid % w; // 必须是有符号的。

                if (x >= w || y >= h)
                    return;

                bool chk_bool = abcdk::torch::util::point_on_line(x1, y1, x2, y2, x, y, weight);
                if (!chk_bool)
                    return;

                /*填充颜色。*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t off = abcdk::torch::util::off<T>(packed, w, ws, h, channels, 0, x, y, z);
                    *abcdk::torch::util::ptr<T>(dst, off) = abcdk::torch::util::pixel<T>(color[z]);
                }
            }

        } // namespace imageproc
    } //    namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_IMAGEPROC_LINE_HXX