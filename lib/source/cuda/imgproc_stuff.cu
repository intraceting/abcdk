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
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_stuff_2d2d(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::stuff_kernel<T>(channels,packed,dst,width,pitch,height,scalar,tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_stuff(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar)
{
    void *gpu_scalar;
    uint3 dim[2];

    assert(dst != NULL && width > 0 && pitch > 0 && height > 0 && scalar != NULL);

    gpu_scalar = abcdk_cuda_copyfrom(scalar, channels * sizeof(T), 1);
    if (!gpu_scalar)
        return -1;

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, width * height, 64);

    _abcdk_cuda_imgproc_stuff_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, width, pitch, height, (T*)gpu_scalar);
    abcdk_cuda_free(&gpu_scalar);

    return 0;
}

int abcdk_cuda_imgproc_stuff_8u(int channels, int packed,uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[])
{
    return _abcdk_cuda_imgproc_stuff<uint8_t>(channels, packed, dst, width, pitch, height, scalar);
}

#else // __cuda_cuda_h__

int abcdk_cuda_imgproc_stuff_8u(int channels, int packed,uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[])
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif // __cuda_cuda_h__
