/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_IMAGEPROC_DEFOG_HXX
#define ABCDK_TORCH_IMAGEPROC_DEFOG_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace imageproc
        {
            /**
             * 暗通道除雾。
             * 建议：a=220,m=0.35,w=0.9
             */
            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE void defog(int channels, bool packed,
                                                 T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                 uint32_t dack_a, float dack_m, float dack_w,
                                                 size_t tid)
            {

                size_t y = tid / dst_w;
                size_t x = tid % dst_w;

                if (x >= dst_w || y >= dst_h)
                    return;

                dack_a = abcdk::torch::util::pixel<T>(dack_a);

                T dack_c = abcdk::torch::util::pixel<T>(0xffffffff);
                size_t dst_of[4] = {0, 0, 0, 0};

                for (size_t z = 0; z < channels; z++)
                {
                    dst_of[z] = abcdk::torch::util::off<T>(packed, dst_w, dst_ws, dst_h, channels, 0, x, y, z);

                    if (dack_c > abcdk::torch::util::obj<T>(dst, dst_of[z]))
                        dack_c = abcdk::torch::util::obj<T>(dst, dst_of[z]);
                }

                float t = abcdk::torch::util::max<float>(dack_m, (1.0 - dack_w / dack_a * dack_c));

                for (size_t z = 0; z < channels; z++)
                {
                    *abcdk::torch::util::ptr<T>(dst, dst_of[z]) = abcdk::torch::util::pixel<T>(((abcdk::torch::util::obj<T>(dst, dst_of[z]) - dack_a) / t + dack_a));
                }
            }

        } // namespace imageproc
    } //    namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_IMAGEPROC_DEFOG_HXX