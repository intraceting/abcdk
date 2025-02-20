/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "../impl/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_brightness_2d2d(int channels, bool packed,
                                                             T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                             size_t w, size_t h, float *alpha, float *bate)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::imageproc::brightness_kernel<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, alpha, bate, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_brightness(int channels, bool packed,
                                                     T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                     size_t w, size_t h, float *alpha, float *bate)
{
    void *gpu_alpha = NULL, *gpu_bate = NULL;
    uint3 dim[2];

    gpu_alpha = abcdk_cuda_copyfrom(alpha, channels * sizeof(float), 1);
    gpu_bate = abcdk_cuda_copyfrom(bate, channels * sizeof(float), 1);

    if (!gpu_alpha || !gpu_bate)
    {
        abcdk_cuda_free(&gpu_alpha);
        abcdk_cuda_free(&gpu_bate);
        return -1;
    }

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_cuda_imgproc_brightness_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_ws, src, src_ws, w, h, (float *)gpu_alpha, (float *)gpu_bate);

    abcdk_cuda_free(&gpu_alpha);
    abcdk_cuda_free(&gpu_bate);
    return 0;
}

int abcdk_cuda_imgproc_brightness_8u_c1r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(alpha != NULL && bate != NULL);

    return _abcdk_cuda_imgproc_brightness(1, true, dst, dst_ws, src, src_ws, w, h, alpha, bate);
}

int abcdk_cuda_imgproc_brightness_8u_c3r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(alpha != NULL && bate != NULL);

    return _abcdk_cuda_imgproc_brightness(3, true, dst, dst_ws, src, src_ws, w, h, alpha, bate);
}

int abcdk_cuda_imgproc_brightness_8u_c4r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float *alpha, float *bate)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(alpha != NULL && bate != NULL);

    return _abcdk_cuda_imgproc_brightness(4, true, dst, dst_ws, src, src_ws, w, h, alpha, bate);
}

#endif // __cuda_cuda_h__