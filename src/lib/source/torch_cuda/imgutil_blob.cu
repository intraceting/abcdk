/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/imgutil.hxx"
#include "grid.hxx"

#ifdef __cuda_cuda_h__

template <typename DT, typename ST, typename BT>
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_imgutil_blob_2d2d_cuda(bool dst_packed, DT *dst, size_t dst_ws,
                                                                   bool src_packed, ST *src, size_t src_ws,
                                                                   size_t b, size_t w, size_t h, size_t c,
                                                                   BT *scale, BT *mean, BT *std,
                                                                   bool revert)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::imgutil::blob<DT, ST, BT>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c, scale, mean, std, revert, tid);
}

template <typename DT, typename ST, typename BT>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgutil_blob_cuda(bool dst_packed, DT *dst, size_t dst_ws,
                                                           bool src_packed, ST *src, size_t src_ws,
                                                           size_t b, size_t w, size_t h, size_t c,
                                                           BT *scale, BT *mean, BT *std,
                                                           bool revert)
{
    void *gpu_scale = NULL, *gpu_mean = NULL, *gpu_std = NULL;
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(b > 0 && w > 0 && h > 0 && c > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

    gpu_scale = abcdk_torch_copyfrom_cuda(scale, c * sizeof(BT), 1);
    gpu_mean = abcdk_torch_copyfrom_cuda(mean, c * sizeof(BT), 1);
    gpu_std = abcdk_torch_copyfrom_cuda(std, c * sizeof(BT), 1);

    if (!gpu_scale || !gpu_mean || !gpu_std)
    {
        abcdk_torch_free_cuda(&gpu_scale);
        abcdk_torch_free_cuda(&gpu_mean);
        abcdk_torch_free_cuda(&gpu_std);
        return -1;
    }

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, b * w * h * c, 64);

    _abcdk_torch_imgutil_blob_2d2d_cuda<DT, ST, BT><<<dim[0], dim[1]>>>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c, (BT *)gpu_scale, (BT *)gpu_mean, (BT *)gpu_std, revert);

    abcdk_torch_free_cuda(&gpu_scale);
    abcdk_torch_free_cuda(&gpu_mean);
    abcdk_torch_free_cuda(&gpu_std);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgutil_blob_8u_to_32f_cuda(int dst_packed, float *dst, size_t dst_ws,
                                            int src_packed, uint8_t *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[])
{
    return _abcdk_torch_imgutil_blob_cuda<float, uint8_t, float>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c, scale, mean, std, false);
}

int abcdk_torch_imgutil_blob_32f_to_8u_cuda(int dst_packed, uint8_t *dst, size_t dst_ws,
                                            int src_packed, float *src, size_t src_ws,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[])
{
    return _abcdk_torch_imgutil_blob_cuda<uint8_t, float, float>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c, scale, mean, std, true);
}

__END_DECLS

#endif // __cuda_cuda_h__
