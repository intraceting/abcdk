/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "grid.cu.hxx"
#include "util.cu.hxx"

#ifdef HAVE_CUDA

template <typename T>
ABCDK_CUDA_GLOBAL void _abcdk_cuda_imgproc_stuff_2d2d(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar)
{
    size_t tid = abcdk::cuda::grid_get_tid(2, 2);

    size_t y = tid / width;
    size_t x = tid % width;

    if (x >= width || y >= height)
        return;

    for (size_t i = 0; i < channels; i++)
    {
        size_t offset = abcdk::cuda::off<T>(packed, width, pitch, height, channels, 0, x, y, i);

        dst[offset] = (scalar ? scalar[i] : (T)0);
    }
}

template <typename T>
ABCDK_CUDA_HOST int _abcdk_cuda_imgproc_stuff(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar)
{
    T *gpu_scalar;
    uint3 dim[2];

    gpu_scalar = (T *)abcdk_cuda_copyfrom(scalar, channels * sizeof(T), 1);
    if (!gpu_scalar)
        return -1;

    /*2D-2D*/
    abcdk::cuda::grid_make_2d2d(dim, width * height, 64);

    _abcdk_cuda_imgproc_stuff_2d2d<T><<<dim[0], dim[1]>>>(3, true, dst, width, pitch, height, gpu_scalar);
    abcdk_cuda_free((void **)&gpu_scalar);

    return -1;
}

int abcdk_cuda_imgproc_stuff_8u_c1r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[0])
{
    assert(dst != NULL && width > 0 && pitch > 0 && height > 0 && scalar != NULL);

    return _abcdk_cuda_imgproc_stuff<uint8_t>(1, true, dst, width, pitch, height, scalar);
}

int abcdk_cuda_imgproc_stuff_8u_c3r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[3])
{
    assert(dst != NULL && width > 0 && pitch > 0 && height > 0 && scalar != NULL);

    return _abcdk_cuda_imgproc_stuff<uint8_t>(3, true, dst, width, pitch, height, scalar);
}

int abcdk_cuda_imgproc_stuff_8u_c4r(uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[4])
{
    assert(dst != NULL && width > 0 && pitch > 0 && height > 0 && scalar != NULL);

    return _abcdk_cuda_imgproc_stuff<uint8_t>(4, true, dst, width, pitch, height, scalar);
}

#endif // HAVE_CUDA
