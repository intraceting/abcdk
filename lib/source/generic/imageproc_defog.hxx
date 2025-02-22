/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_IMAGEPROC_DEFOG_HXX
#define ABCDK_GENERIC_IMAGEPROC_DEFOG_HXX

#include "util.hxx"

namespace abcdk
{
    namespace generic
    {
        namespace imageproc
        {
            template <typename T>
            ABCDK_INVOKE_DEVICE void defog_kernel(int channels, bool packed,
                                                  T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                  size_t w, size_t h, float dack_m, T dack_a, float dack_w,
                                                  size_t tid)
            {

                size_t y = tid / w;
                size_t x = tid % w;

                if (x >= w || y >= h)
                    return;

                T dack_c = (T)abcdk::generic::util::pixel_clamp<uint32_t>(0xffffffff);
                size_t src_of[4] = {0, 0, 0, 0};
                size_t dst_of[4] = {0, 0, 0, 0};

                for (size_t z = 0; z < channels; z++)
                {
                    src_of[z] = abcdk::generic::util::off<T>(packed, w, src_ws, h, channels, 0, x, y, z);
                    dst_of[z] = abcdk::generic::util::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);

                    if (dack_c > abcdk::generic::util::obj<T>(src, src_of[z]))
                        dack_c = abcdk::generic::util::obj<T>(src, src_of[z]);
                }

                float t = abcdk::generic::util::max<float>(dack_m, (1.0 - dack_w / dack_a * dack_c));

                for (size_t z = 0; z < channels; z++)
                {
                    *abcdk::generic::util::ptr<T>(dst, dst_of[z]) = abcdk::generic::util::pixel_clamp<T>(((abcdk::generic::util::obj<T>(src, src_of[z]) - dack_a) / t + dack_a));
                }
            }

            /**暗通道除雾。*/
            template <typename T>
            ABCDK_INVOKE_HOST void defog(int channels, bool packed,
                                         T *dst, size_t dst_ws, T *src, size_t src_ws,
                                         size_t w, size_t h, float dack_m = 0.35, T dack_a = 220, float dack_w = 0.9)
            {
                for (size_t i = 0; i < w * h; i++)
                {
                    defog_kernel<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_m, dack_a, dack_w, i);
                }
            }

        } // namespace imageproc
    } //    namespace generic
} // namespace abcdk

#endif // ABCDK_GENERIC_IMAGEPROC_DEFOG_HXX