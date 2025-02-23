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
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_drawrect_2d2d(int channels, bool packed,
                                                           T *dst, size_t w, size_t ws, size_t h,
                                                           T *color, int weight, int *corner)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::drawrect_kernel<T>(channels, packed, dst, w, ws, h, color, weight, corner, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_drawrect(int channels, bool packed,
                                                   T *dst, size_t w, size_t ws, size_t h,
                                                   T *color, int weight, int *corner)
{
    void *gpu_color = NULL, *gpu_conrer = NULL;
    uint3 dim[2];

    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0 && corner != NULL);

    gpu_color = abcdk_cuda_copyfrom(color, channels * sizeof(T), 1);
    gpu_conrer = abcdk_cuda_copyfrom(corner, 4 * sizeof(int), 1);

    if (!gpu_color || !gpu_conrer)
    {
        abcdk_cuda_free(&gpu_color);
        abcdk_cuda_free(&gpu_conrer);
        return -1;
    }

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_cuda_imgproc_drawrect_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, w, ws, h, (T *)gpu_color, weight, (int *)gpu_conrer);
    abcdk_cuda_free(&gpu_color);
    abcdk_cuda_free(&gpu_conrer);

    return 0;
}

int abcdk_cuda_imgproc_drawrect_8u(int channels, int packed,
                                   uint8_t *dst, size_t w, size_t ws, size_t h,
                                   uint8_t color[], int weight, int corner[4])
{
    return _abcdk_cuda_imgproc_drawrect<uint8_t>(channels, packed, dst, w, ws, h, color, weight, corner);
}

#endif // __cuda_cuda_h__