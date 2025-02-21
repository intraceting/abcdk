/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_TENSORPROC_RESHAPE_HXX
#define ABCDK_IMPL_TENSORPROC_RESHAPE_HXX

#include "general.hxx"

namespace abcdk
{
    namespace tensorproc
    {
        template <typename T>
        ABCDK_INVOKE_DEVICE void reshape_kernel(bool dst_packed, T *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                                bool src_packed, T *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c,
                                                size_t tid)
        {
            size_t dst_n, dst_x, dst_y, dst_z;
            size_t src_n, src_x, src_y, src_z;

            if (dst_packed)
            {
                dst_n = tid / (dst_h * dst_w * dst_c);
                dst_y = (tid / (dst_w * dst_c)) % dst_h;
                dst_x = (tid / dst_c) % dst_w;
                dst_z = tid % dst_c;
            }
            else
            {
                dst_n = tid / (dst_c * dst_h * dst_w);
                dst_z = (tid / (dst_h * dst_w)) % dst_c;
                dst_h = (tid / dst_w) % dst_h;
                dst_x = tid % dst_w;
            }

            if (src_packed)
            {
                src_n = tid / (src_h * src_w * src_c);
                src_y = (tid / (src_w * src_c)) % src_h;
                src_x = (tid / src_c) % src_w;
                src_z = tid % src_c;
            }
            else
            {
                src_n = tid / (src_c * src_h * src_w);
                src_z = (tid / (src_h * src_w)) % src_c;
                src_h = (tid / src_w) % src_h;
                src_x = tid % src_w;
            }

            if (dst_n >= dst_b || dst_x >= dst_w || dst_y >= dst_h || dst_z >= dst_c)
                return;

            if (src_n >= src_b || src_x >= src_w || src_y >= src_h || src_z >= src_c)
                return;

            size_t src_of = abcdk::general::off<T>(src_packed, src_w, src_ws, src_h, src_c, src_n, src_x, src_y, src_z);
            size_t dst_of = abcdk::general::off<T>(dst_packed, dst_w, dst_ws, dst_h, dst_c, dst_n, dst_x, dst_y, dst_z);

            *abcdk::general::ptr<T>(dst, dst_of) = abcdk::general::obj<T>(src, src_of);
        }

        template <typename T>
        ABCDK_INVOKE_HOST void reshape(bool dst_packed, T *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                       bool src_packed, T *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c)
        {
            size_t dst_total = dst_b * dst_w * dst_h * dst_c;
            size_t src_total = src_b * src_w * src_h * src_c;

            assert(dst_total == src_total);

            for (size_t i = 0; i < dst_total; i++)
                reshape_kernel<T>(dst_packed, dst, dst_b, dst_w, dst_ws, dst_h, dst_c, src_packed, src, src_b, src_w, src_ws, src_h, src_c);
        }

    } // namespace tensorproc

} // namespace abcdk

#endif // ABCDK_IMPL_TENSORPROC_RESHAPE_HXX