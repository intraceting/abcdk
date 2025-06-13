/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMAGEPROC_DRAWMASK_HXX
#define ABCDK_TORCH_IMAGEPROC_DRAWMASK_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace imageproc
        {
            /**
             * 画掩码。
             *
             */
            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE void drawmask(int channels, bool packed,
                                                    T *dst, size_t dst_ws, float* mask, size_t mask_ws, size_t w, size_t h, float threshold, uint32_t *color, 
                                                    size_t tid)
            {
                size_t y = tid / w; 
                size_t x = tid % w;

                if (x >= w || y >= h)
                    return;

                /*小于阈值的不需要。*/
                if(abcdk::torch::util::obj<float>(mask, packed, w, mask_ws, h, 1, 0, x, y, 0) < threshold)
                    return;

                /*填充颜色。*/
                for (size_t z = 0; z < channels; z++)
                {
                    size_t dst_off = abcdk::torch::util::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);
                    *abcdk::torch::util::ptr<T>(dst, dst_off) = (abcdk::torch::util::obj<T>(dst, dst_off) * 0.5 + abcdk::torch::util::pixel<T>(color[z]) * 0.5);
                }
            }

        } // namespace imageproc
    } //    namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_IMAGEPROC_DRAWMASK_HXX