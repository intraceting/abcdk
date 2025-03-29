/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tensorproc.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/tensorproc.hxx"
#include "grid.hxx"

#ifdef __cuda_cuda_h__

template <typename DT, typename ST>
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_tensorproc_blob_2d2d_cuda(bool dst_packed, DT *dst, size_t dst_ws,
                                                                bool src_packed, ST *src, size_t src_ws,
                                                                size_t b, size_t w, size_t h, size_t c,
                                                                bool revert, float *scale, float *mean, float *std)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::tensorproc::blob<DT, ST>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c,
                                             revert, scale, mean, std, tid);
}

template <typename DT, typename ST>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_tensorproc_blob_cuda(bool dst_packed, DT *dst, size_t dst_ws,
                                                        bool src_packed, ST *src, size_t src_ws,
                                                        size_t b, size_t w, size_t h, size_t c,
                                                        bool revert, float *scale, float *mean, float *std)
{
    void *gpu_scale = NULL, *gpu_mean = NULL, *gpu_std = NULL;
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(b > 0 && w > 0 && h > 0 && c > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

    gpu_scale = abcdk_torch_copyfrom_cuda(scale, c * sizeof(float), 1);
    gpu_mean = abcdk_torch_copyfrom_cuda(mean, c * sizeof(float), 1);
    gpu_std = abcdk_torch_copyfrom_cuda(std, c * sizeof(float), 1);

    if (!gpu_scale || !gpu_mean || !gpu_std)
    {
        abcdk_torch_free_cuda(&gpu_scale);
        abcdk_torch_free_cuda(&gpu_mean);
        abcdk_torch_free_cuda(&gpu_std);
        return -1;
    }

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, b * w * h * c, 64);

    _abcdk_torch_tensorproc_blob_2d2d_cuda<DT, ST><<<dim[0], dim[1]>>>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c,
                                                                       revert, (float *)gpu_scale, (float *)gpu_mean, (float *)gpu_std);

    abcdk_torch_free_cuda(&gpu_scale);
    abcdk_torch_free_cuda(&gpu_mean);
    abcdk_torch_free_cuda(&gpu_std);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_tensorproc_blob_8u_to_32f_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    assert(dst != NULL && src != NULL);
    assert(scale != NULL && mean != NULL && std != NULL);
    assert(dst->block == src->block);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->depth == src->depth);
    assert(dst->cell == sizeof(float) && src->cell == sizeof(uint8_t));

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    return _abcdk_torch_tensorproc_blob_cuda<float, uint8_t>((dst->format == ABCDK_TORCH_TENFMT_NHWC), (float *)dst->data, dst->stride,
                                                             (src->format == ABCDK_TORCH_TENFMT_NHWC), src->data, src->stride,
                                                             dst->block, dst->width, dst->height, dst->depth,
                                                             false, scale, mean, std);
}

int abcdk_torch_tensorproc_blob_32f_to_8u_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    assert(dst != NULL && src != NULL);
    assert(scale != NULL && mean != NULL && std != NULL);
    assert(dst->block == src->block);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->depth == src->depth);
    assert(dst->cell == sizeof(uint8_t) && src->cell == sizeof(float));

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    return _abcdk_torch_tensorproc_blob_cuda<uint8_t, float>((dst->format == ABCDK_TORCH_TENFMT_NHWC), dst->data, dst->stride,
                                                             (src->format == ABCDK_TORCH_TENFMT_NHWC), (float *)src->data, src->stride,
                                                             dst->block, dst->width, dst->height, dst->depth,
                                                             true, scale, mean, std);
}

__END_DECLS

#endif // __cuda_cuda_h__
