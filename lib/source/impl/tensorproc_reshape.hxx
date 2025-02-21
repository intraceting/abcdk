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
            size_t hw;

            /*源和目标索引算法必须一样。*/

            dst_n = tid / (dst_w * dst_h * dst_c);  // 块索引
            hw = tid % (dst_w * dst_h * dst_c);     // 块余数
            dst_y = hw / (dst_w * dst_c);           // 高度索引
            dst_x = (hw % (dst_w * dst_c)) / dst_c; // 宽度索引
            dst_z = hw % dst_c;                     // 通道索引

            src_n = tid / (src_w * src_h * src_c);  // 块索引
            hw = tid % (src_w * src_h * src_c);     // 块余数
            src_y = hw / (src_w * src_c);           // 高度索引
            src_x = (hw % (src_w * src_c)) / src_c; // 宽度索引
            src_z = hw % src_c;                     // 通道索引

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
            size_t dst_total, src_total;
        
            assert(dst != NULL && dst_b > 0 && dst_w > 0 && dst_ws > 0 && dst_h > 0 && dst_c > 0);
            assert(dst != NULL && src_b > 0 && src_w > 0 && src_ws > 0 && src_h > 0 && src_c > 0);
        
            assert(dst_packed ? (dst_ws >= dst_w * dst_c * sizeof(T)) : (dst_ws >= dst_w * sizeof(T)));
            assert(src_packed ? (src_ws >= src_w * src_c * sizeof(T)) : (src_ws >= src_w * sizeof(T)));
        
            dst_total = dst_b * dst_w * dst_h * dst_c;
            src_total = src_b * src_w * src_h * src_c;
        
            assert(dst_total == src_total);

            for (size_t i = 0; i < dst_total; i++)
                reshape_kernel<T>(dst_packed, dst, dst_b, dst_w, dst_ws, dst_h, dst_c, src_packed, src, src_b, src_w, src_ws, src_h, src_c, i);
        }

    } // namespace tensorproc

} // namespace abcdk

#endif // ABCDK_IMPL_TENSORPROC_RESHAPE_HXX