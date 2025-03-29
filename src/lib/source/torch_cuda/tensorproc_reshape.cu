/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/tensorproc.h"
#include "../torch/tensorproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_tensorproc_reshape_2d2d_cuda(bool dst_packed, uint8_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                                                   bool src_packed, uint8_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c,
                                                                   size_t cell)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::tensorproc::reshape(dst_packed, dst, dst_b, dst_w, dst_ws, dst_h, dst_c,
                                        src_packed, src, src_b, src_w, src_ws, src_h, src_c,
                                        cell, tid);
}

ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_tensorproc_reshape_cuda(bool dst_packed, uint8_t *dst, size_t dst_b, size_t dst_w, size_t dst_ws, size_t dst_h, size_t dst_c,
                                                           bool src_packed, uint8_t *src, size_t src_b, size_t src_w, size_t src_ws, size_t src_h, size_t src_c,
                                                           size_t cell)
{
    size_t dst_total, src_total;
    uint3 dim[2];

    assert(dst != NULL && dst_b > 0 && dst_w > 0 && dst_ws > 0 && dst_h > 0 && dst_c > 0);
    assert(dst != NULL && src_b > 0 && src_w > 0 && src_ws > 0 && src_h > 0 && src_c > 0);
    assert(cell > 0);

    assert(dst_packed ? (dst_ws >= dst_w * dst_c * cell) : (dst_ws >= dst_w * cell));
    assert(src_packed ? (src_ws >= src_w * src_c * cell) : (src_ws >= src_w * cell));

    dst_total = dst_b * dst_w * dst_h * dst_c;
    src_total = src_b * src_w * src_h * src_c;

    assert(dst_total == src_total);

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, dst_total, 64);

    _abcdk_torch_tensorproc_reshape_2d2d_cuda<<<dim[0], dim[1]>>>(dst_packed, dst, dst_b, dst_w, dst_ws, dst_h, dst_c,
                                                                  src_packed, src, src_b, src_w, src_ws, src_h, src_c,
                                                                  cell);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_tensorproc_reshape_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    assert(dst != NULL && src != NULL);
    assert(dst->cell == src->cell);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);

    return _abcdk_torch_tensorproc_reshape_cuda((dst->format == ABCDK_TORCH_TENFMT_NHWC), dst->data, dst->block, dst->width, dst->stride, dst->height, dst->depth,
                                                (src->format == ABCDK_TORCH_TENFMT_NHWC), src->data, src->block, src->width, src->stride, src->height, src->depth,
                                                dst->cell);
}

__END_DECLS

#else //__cuda_cuda_h__

__BEGIN_DECLS

int abcdk_torch_tensorproc_reshape_cuda(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

__END_DECLS

#endif // __cuda_cuda_h__
