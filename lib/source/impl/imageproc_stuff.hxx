/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_IMAGEPROC_STUFF_HXX
#define ABCDK_IMPL_IMAGEPROC_STUFF_HXX

#include "general.hxx"

namespace abcdk
{
    namespace imageproc
    {
        template <typename T>
        ABCDK_INVOKE_DEVICE void stuff_kernel(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar, size_t tid)
        {

            size_t y = tid / width;
            size_t x = tid % width;

            if (x >= width || y >= height)
                return;

            for (size_t i = 0; i < channels; i++)
            {
                size_t offset = abcdk::general::off<T>(packed, width, pitch, height, channels, 0, x, y, i);

                *abcdk::general::ptr<T>(dst,offset) = (scalar ? scalar[i] : (T)0);
            }
        }

        /**填充颜色。*/
        template <typename T>
        ABCDK_INVOKE_HOST void stuff(int channels, bool packed,
                                     T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                     T *scalar)
        {
            for (size_t i = 0; i < dst_w * dst_h; i++)
            {
                stuff_kernel<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, i);
            }
        }

    } // namespace imageproc

} // namespace abcdk

#endif // ABCDK_IMPL_IMAGEPROC_STUFF_HXX