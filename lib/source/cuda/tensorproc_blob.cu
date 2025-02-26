/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/tensorproc.h"
#include "../generic/tensorproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename ST, typename DT>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_tensorproc_blob_2d2d(int channels, bool revert,
                                                          bool dst_packed, DT *dst, size_t dst_ws,
                                                          bool src_packed, ST *src, size_t src_ws,
                                                          size_t w, size_t h, float *scale, float *mean, float *std)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::tensorproc::blob_kernel<ST, DT>(channels, revert, dst_packed, dst, dst_ws, src_packed, src, src_ws, w, h, scale, mean, std, tid);
}

template <typename ST, typename DT>
ABCDK_INVOKE_HOST int _abcdk_cuda_tensorproc_blob(int channels, bool revert,
                                                  bool dst_packed, DT *dst, size_t dst_ws,
                                                  bool src_packed, ST *src, size_t src_ws,
                                                  size_t w, size_t h, float *scale, float *mean, float *std)
{
    void *gpu_scale = NULL, *gpu_mean = NULL, *gpu_std = NULL;
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

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
    abcdk::cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_cuda_tensorproc_blob_2d2d<ST, DT><<<dim[0], dim[1]>>>(channels, revert,
                                                                 dst_packed, dst, dst_ws,
                                                                 src_packed, src, src_ws,
                                                                 w, h, (float *)gpu_scale, (float *)gpu_mean, (float *)gpu_std);
    abcdk_cuda_free(&gpu_scale);
    abcdk_cuda_free(&gpu_mean);
    abcdk_cuda_free(&gpu_std);
    
    return 0;
}

int abcdk_cuda_tensorproc_blob_8u_to_32f(int channels,
                                         int dst_packed, float *dst, size_t dst_ws,
                                         int src_packed, uint8_t *src, size_t src_ws,
                                         size_t w, size_t h, float scale[], float mean[], float std[])
{
    return _abcdk_cuda_tensorproc_blob<uint8_t, float>(channels, false, dst_packed, dst, dst_ws, src_packed, src, src_ws, w, h, scale, mean, std);
}

int abcdk_cuda_tensorproc_blob_32f_to_8u(int channels,
                                         int dst_packed, uint8_t *dst, size_t dst_ws,
                                         int src_packed, float *src, size_t src_ws,
                                         size_t w, size_t h, float scale[], float mean[], float std[])
{
    return _abcdk_cuda_tensorproc_blob<float, uint8_t>(channels, true, dst_packed, dst, dst_ws, src_packed, src, src_ws, w, h, scale, mean, std);
}

#else //__cuda_cuda_h__

int abcdk_cuda_tensorproc_blob_8u_to_32f(int channels,
    int dst_packed, float *dst, size_t dst_ws,
    int src_packed, uint8_t *src, size_t src_ws,
    size_t w, size_t h, float scale[], float mean[], float std[])
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_tensorproc_blob_32f_to_8u(int channels,
    int dst_packed, uint8_t *dst, size_t dst_ws,
    int src_packed, float *src, size_t src_ws,
    size_t w, size_t h, float scale[], float mean[], float std[])
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}


#endif // __cuda_cuda_h__
