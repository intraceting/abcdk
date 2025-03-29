/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TENSORPROC_RESHAPE_HXX
#define ABCDK_TORCH_TENSORPROC_RESHAPE_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace tensorproc
        {
            ABCDK_TORCH_INVOKE_DEVICE void reshape(bool dst_packed, uint8_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                             bool src_packed, uint8_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c,
                                             size_t cell, size_t tid)
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

                size_t src_of = abcdk::torch::util::off<uint8_t>(src_packed, src_w, src_ws, src_h, src_c, src_n, src_x, src_y, src_z);
                size_t dst_of = abcdk::torch::util::off<uint8_t>(dst_packed, dst_w, dst_ws, dst_h, dst_c, dst_n, dst_x, dst_y, dst_z);

                for (int i = 0; i < cell; i++)
                    *abcdk::torch::util::ptr<uint8_t>(dst, dst_of + i) = abcdk::torch::util::obj<uint8_t>(src, src_of + i);
            }
        } // namespace tensorproc
    } //   namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_TENSORPROC_RESHAPE_HXX