/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_defog_2d2d(int channels, bool packed,
                                                        T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                        uint32_t dack_a, float dack_m, float dack_w)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::defog<T>(channels, packed, dst, dst_w, dst_ws, dst_h, dack_m, dack_a, dack_w, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_defog(int channels, bool packed,
                                                T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                uint32_t dack_a, float dack_m, float dack_w)
{
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(dst_w > 0 && dst_h > 0);

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, dst_w * dst_h, 64);

    _abcdk_cuda_imgproc_defog_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_w, dst_ws, dst_h, dack_a, dack_m, dack_w);

    return 0;
}

__BEGIN_DECLS


int abcdk_cuda_imgproc_defog(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w)
{
    int dst_depth;

    assert(dst != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_cuda_imgproc_defog<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, dack_a, dack_m, dack_w);
}

__END_DECLS

#else // __cuda_cuda_h__

__BEGIN_DECLS

int abcdk_cuda_imgproc_defog(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

__END_DECLS

#endif // __cuda_cuda_h__