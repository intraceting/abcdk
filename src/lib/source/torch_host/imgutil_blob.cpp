/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgutil.h"
#include "../torch/imgutil.hxx"

template <typename DT, typename ST, typename BT>
ABCDK_TORCH_INVOKE_HOST void _abcdk_torch_imgutil_blob_1d_host(bool dst_packed, DT *dst, size_t dst_ws, bool dst_c_invert,
                                                               bool src_packed, ST *src, size_t src_ws, bool src_c_invert,
                                                               size_t b, size_t w, size_t h, size_t c,
                                                               BT *scale, BT *mean, BT *std,
                                                               bool revert)
{
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(abcdk_align(cpus / 2, 1))
    for (size_t i = 0; i < b * w * h * c; i++)
    {
        abcdk::torch::imgutil::blob<DT, ST, BT>(dst_packed, dst, dst_ws, dst_c_invert,
                                                src_packed, src, src_ws, src_c_invert,
                                                b, w, h, c,
                                                scale, mean, std,
                                                revert, i);
    }
}

template <typename DT, typename ST, typename BT>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgutil_blob_host(bool dst_packed, DT *dst, size_t dst_ws, bool dst_c_invert,
                                                           bool src_packed, ST *src, size_t src_ws, bool src_c_invert,
                                                           size_t b, size_t w, size_t h, size_t c,
                                                           BT *scale, BT *mean, BT *std,
                                                           bool revert)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(b > 0 && w > 0 && h > 0 && c > 0);
    assert(scale != NULL && mean != NULL && std != NULL);

    _abcdk_torch_imgutil_blob_1d_host<DT, ST, BT>(dst_packed, dst, dst_ws, dst_c_invert,
                                                  src_packed, src, src_ws, src_c_invert,
                                                  b, w, h, c,
                                                  scale, mean, std,
                                                  revert);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgutil_blob_8u_to_32f_host(int dst_packed, float *dst, size_t dst_ws, int dst_c_invert,
                                            int src_packed, uint8_t *src, size_t src_ws, int src_c_invert,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[])
{
    return _abcdk_torch_imgutil_blob_host<float, uint8_t, float>(dst_packed, dst, dst_ws, dst_c_invert,
                                                                 src_packed, src, src_ws, src_c_invert,
                                                                 b, w, h, c,
                                                                 scale, mean, std,
                                                                 false);
}

int abcdk_torch_imgutil_blob_32f_to_8u_host(int dst_packed, uint8_t *dst, size_t dst_ws, int dst_c_invert,
                                            int src_packed, float *src, size_t src_ws, int src_c_invert,
                                            size_t b, size_t w, size_t h, size_t c,
                                            float scale[], float mean[], float std[])
{
    return _abcdk_torch_imgutil_blob_host<uint8_t, float, float>(dst_packed, dst, dst_ws, dst_c_invert, src_packed,
                                                                 src, src_ws, src_c_invert,
                                                                 b, w, h, c,
                                                                 scale, mean, std,
                                                                 true);
}

__END_DECLS
