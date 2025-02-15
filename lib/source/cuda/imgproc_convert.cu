/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "grid.cu.hxx"
#include "util.cu.hxx"

#ifdef __cuda_cuda_h__

template <class ST, class DT>
ABCDK_CUDA_GLOBAL void _abcdk_cuda_imgproc_convert_2d2d(int channels,
                                                        bool dst_packed, DT *dst, size_t dst_ws,
                                                        bool src_packed, ST *src, size_t src_ws,
                                                        size_t w, size_t h,
                                                        float *scale, float *mean, float *std)
{
    size_t tid = abcdk::cuda::grid_get_tid(2, 2);

    size_t y = tid / w;
    size_t x = tid % w;

    if (x >= w || y >= h)
        return;

    for (size_t z = 0; z < channels; z++)
    {
        size_t src_of = abcdk::cuda::off<ST>(src_packed, w, src_ws, h, channels, 0, x, y, z);
        size_t dst_of = abcdk::cuda::off<DT>(dst_packed, w, dst_ws, h, channels, 0, x, y, z);

        ST *src_p = abcdk::cuda::ptr<ST>(src, src_of);
        DT *dst_p = abcdk::cuda::ptr<DT>(dst, dst_of);
        *dst_p = (((DT)*src_p / scale[z]) - mean[z]) / std[z];
    }
}

template <class ST, class DT>
ABCDK_CUDA_HOST int _abcdk_cuda_imgproc_convert(int channels,
                                                bool dst_packed, DT *dst, size_t dst_ws,
                                                bool src_packed, ST *src, size_t src_ws,
                                                size_t w, size_t h,
                                                float *scale, float *mean, float *std)
{
    void *gpu_scale = NULL, *gpu_mean = NULL, *gpu_std = NULL;
    uint3 dim[2];

    gpu_scale = abcdk_cuda_copyfrom(scale, channels * sizeof(float), 1);
    gpu_mean = abcdk_cuda_copyfrom(mean, channels * sizeof(float), 1);
    gpu_std = abcdk_cuda_copyfrom(std, channels * sizeof(float), 1);

    if (!gpu_scale || !gpu_mean || !gpu_std)
    {
        abcdk_cuda_free(&gpu_scale);
        abcdk_cuda_free(&gpu_mean);
        abcdk_cuda_free(&gpu_std);
        return -1;
    }

    /*2D-2D*/
    abcdk::cuda::grid_make_2d2d(dim, w * h, 64);

    _abcdk_cuda_imgproc_convert_2d2d<ST, DT><<<dim[0], dim[1]>>>(channels, dst_packed, dst, dst_ws, src_packed, src, src_ws, w, h, (float *)gpu_scale, (float *)gpu_mean, (float *)gpu_std);

    abcdk_cuda_free(&gpu_scale);
    abcdk_cuda_free(&gpu_mean);
    abcdk_cuda_free(&gpu_std);
    return 0;
}

int abcdk_cuda_imgproc_convert_8u_to_32f_3r(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t w, size_t h, float scale[3], float mean[3], float std[3])
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

    return _abcdk_cuda_imgproc_convert<uint8_t, float>(3, dst_packed, dst, dst_ws, src_packed, src, src_ws, w, h, scale, mean, std);
}

#endif // __cuda_cuda_h__
