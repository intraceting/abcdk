/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_IMAGEPROC_STUFF_HXX
#define ABCDK_GENERIC_IMAGEPROC_STUFF_HXX

#include "util.hxx"

namespace abcdk
{
    namespace generic
    {
        namespace imageproc
        {
            template <typename T>
            ABCDK_INVOKE_DEVICE void stuff(int channels, bool packed,
                                           T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, uint32_t *scalar,
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

                for (size_t i = 0; i < channels; i++)
                {
                    size_t dst_off = abcdk::generic::util::off<T>(packed, dst_w, dst_ws, dst_h, channels, 0, x, y, i);

                    *abcdk::generic::util::ptr<T>(dst, dst_off) = abcdk::generic::util::pixel<T>(scalar[i]);
                }
            }


        } // namespace imageproc
    } //    namespace generic
} // namespace abcdk

#endif // ABCDK_GENERIC_IMAGEPROC_STUFF_HXX