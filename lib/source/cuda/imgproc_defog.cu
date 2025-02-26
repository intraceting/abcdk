/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "../generic/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_defog_2d2d(int channels, bool packed,
                                                        T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                        size_t w, size_t h, float dack_m, T dack_a, float dack_w)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::defog_kernel<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_m, dack_a, dack_w, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_defog(int channels, bool packed,
                                                T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                size_t w, size_t h, T dack_a, float dack_m, float dack_w)
{
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_cuda_imgproc_defog_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_a, dack_m, dack_w);

    return 0;
}

int abcdk_cuda_imgproc_defog_8u(int channels, int packed,
                                uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                size_t w, size_t h, uint8_t dack_a, float dack_m, float dack_w)
{
    return _abcdk_cuda_imgproc_defog(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_a, dack_m, dack_w);
}

#else // __cuda_cuda_h__

int abcdk_cuda_imgproc_defog_8u(int channels, int packed,
                                uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                size_t w, size_t h, uint8_t dack_a, float dack_m, float dack_w)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif // __cuda_cuda_h__