/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_IMAGEPROC_BRIGHTNESS_HXX
#define ABCDK_IMPL_IMAGEPROC_BRIGHTNESS_HXX

#include "general.hxx"

namespace abcdk
{
    namespace imageproc
    {
        template <typename T>
        ABCDK_INVOKE_DEVICE void brightness_kernel(int channels, bool packed,
                                                   T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                   size_t w, size_t h, float *alpha, float *bate,
                                                   size_t tid)
        {
            size_t y = tid / w;
            size_t x = tid % w;

            if (x >= w || y >= h)
                return;

            for (size_t z = 0; z < channels; z++)
            {
                size_t src_offset = abcdk::general::off<T>(packed, w, src_ws, h, channels, 0, x, y, z);
                size_t dst_offset = abcdk::general::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);

                *abcdk::general::ptr<T>(dst, dst_offset) = (T)abcdk::general::pixel_clamp<float>(abcdk::general::obj<T>(src, src_offset) * alpha[z] + bate[z]);
            }
        }

        /**调整亮度。*/
        template <typename T>
        ABCDK_INVOKE_HOST void brightness(int channels, bool packed, T *dst, size_t dst_ws, T *src, size_t src_ws,
                                          size_t w, size_t h, float *alpha, float *bate)
        {
            for (size_t i = 0; i < w * h; i++)
            {
                brightness_kernel<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, alpha, bate, i);
            }
        }
        
    } // namespace imageproc

} // namespace abcdk

#endif // ABCDK_IMPL_IMAGEPROC_BRIGHTNESS_HXX