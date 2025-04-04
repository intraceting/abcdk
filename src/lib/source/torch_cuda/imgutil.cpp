/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_torch_imgutil_copy_cuda(uint8_t *dst_data[4], int dst_stride[4], int dst_in_host,
                                  const uint8_t *src_data[4], const int src_stride[4], int src_in_host,
                                  int width, int height, int pixfmt)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk, chk_plane, chk_stride, chk_height;

    assert(dst_data != NULL && dst_stride != NULL);
    assert(src_data != NULL && src_stride != NULL);
    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk_stride = abcdk_torch_imgutil_fill_stride(real_stride, width, pixfmt, 1);
    chk_height = abcdk_torch_imgutil_fill_height(real_height, height, pixfmt);
    chk_plane = ABCDK_MIN(chk_stride, chk_height);

    for (int i = 0; i < chk_plane; i++)
    {
        chk = abcdk_torch_memcpy_2d_cuda(dst_data[i], dst_stride[i], 0, 0, dst_in_host,
                                         src_data[i], src_stride[i], 0, 0, src_in_host,
                                         real_stride[i], real_height[i]);
        if (chk != 0)
            return -1;
    }

    return 0;
}

#else //__cuda_cuda_h__

int abcdk_torch_imgutil_copy_cuda(uint8_t *dst_data[4], int dst_stride[4], int dst_in_host,
                                  const uint8_t *src_data[4], const int src_stride[4], int src_in_host,
                                  int width, int height, int pixfmt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

#endif //__cuda_cuda_h__

__END_DECLS