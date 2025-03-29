/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tensorproc.h"
#include "../torch/tensorproc.hxx"

template <typename DT, typename ST>
ABCDK_TORCH_INVOKE_HOST void _abcdk_torch_tensorproc_blob_1d(bool dst_packed, DT *dst, size_t dst_ws,
                                                       bool src_packed, ST *src, size_t src_ws,
                                                       size_t b, size_t w, size_t h, size_t c,
                                                       bool revert, float *scale, float *mean, float *std)
{
    // #pragma omp parallel
    for (size_t i = 0; i < b * w * h * c; i++)
    {
        abcdk::torch::tensorproc::blob<DT, ST>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c,
                                                 revert, scale, mean, std, i);
    }
}

template <typename DT, typename ST>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_tensorproc_blob(bool dst_packed, DT *dst, size_t dst_ws,
                                                   bool src_packed, ST *src, size_t src_ws,
                                                   size_t b, size_t w, size_t h, size_t c,
                                                   bool revert, float *scale, float *mean, float *std)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(b > 0 && w > 0 && h > 0 && c > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

    _abcdk_torch_tensorproc_blob_1d<DT, ST>(dst_packed, dst, dst_ws, src_packed, src, src_ws, b, w, h, c,
                                            revert, scale, mean, std);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_tensorproc_blob_8u_to_32f_host(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    assert(dst != NULL && src != NULL);
    assert(scale != NULL && mean != NULL && std != NULL);
    assert(dst->block == src->block);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->depth == src->depth);
    assert(dst->cell == sizeof(float) && src->cell == sizeof(uint8_t));

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    return _abcdk_torch_tensorproc_blob<float, uint8_t>((dst->format == ABCDK_TORCH_TENFMT_NHWC), (float *)dst->data, dst->stride,
                                                        (src->format == ABCDK_TORCH_TENFMT_NHWC), src->data, src->stride,
                                                        dst->block, dst->width, dst->height, dst->depth,
                                                        false, scale, mean, std);
}

int abcdk_torch_tensorproc_blob_32f_to_8u_host(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src, float scale[], float mean[], float std[])
{
    assert(dst != NULL && src != NULL);
    assert(scale != NULL && mean != NULL && std != NULL);
    assert(dst->block == src->block);
    assert(dst->width == src->width);
    assert(dst->height == src->height);
    assert(dst->depth == src->depth);
    assert(dst->cell == sizeof(uint8_t) && src->cell == sizeof(float));
    
    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    return _abcdk_torch_tensorproc_blob<uint8_t, float>((dst->format == ABCDK_TORCH_TENFMT_NHWC), dst->data, dst->stride,
                                                        (src->format == ABCDK_TORCH_TENFMT_NHWC), (float *)src->data, src->stride,
                                                        dst->block, dst->width, dst->height, dst->depth,
                                                        true, scale, mean, std);
}

__END_DECLS
