/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMAGEPROC_BRIGHTNESS_HXX
#define ABCDK_TORCH_IMAGEPROC_BRIGHTNESS_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace imageproc
        {
            /**调整亮度。*/
            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE void brightness(int channels, bool packed,
                                                T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, float *alpha, float *bate, 
                                                size_t tid)
            {
                size_t y = tid / dst_w;
                size_t x = tid % dst_w;

                if (x >= dst_w || y >= dst_h)
                    return;

                for (size_t z = 0; z < channels; z++)
                {
                    size_t dst_offset = abcdk::torch::util::off<T>(packed, dst_w, dst_ws, dst_h, channels, 0, x, y, z);

                    *abcdk::torch::util::ptr<T>(dst, dst_offset) = abcdk::torch::util::pixel<T>(abcdk::torch::util::obj<T>(dst, dst_offset) * alpha[z] + bate[z]);
                }
            }

        } // namespace imageproc
    } // namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_IMAGEPROC_BRIGHTNESS_HXX