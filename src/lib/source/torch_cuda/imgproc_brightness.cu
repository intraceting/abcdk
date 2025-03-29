/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../generic/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_torch_imgproc_brightness_2d2d_cuda(int channels, bool packed,
                                                                   T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, float *alpha, float *bate)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::brightness<T>(channels, packed, dst, dst_ws, dst_ws, dst_h, alpha, bate, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_brightness_cuda(int channels, bool packed,
                                                           T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, float *alpha, float *bate)
{
    void *gpu_alpha = NULL, *gpu_bate = NULL;
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0);
    assert(dst_w > 0 && dst_h > 0);
    assert(alpha != NULL && bate != NULL);

    gpu_alpha = abcdk_torch_copyfrom_cuda(alpha, channels * sizeof(float), 1);
    gpu_bate = abcdk_torch_copyfrom_cuda(bate, channels * sizeof(float), 1);

    if (!gpu_alpha || !gpu_bate)
    {
        abcdk_torch_free_cuda(&gpu_alpha);
        abcdk_torch_free_cuda(&gpu_bate);
        return -1;
    }

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, dst_w * dst_h, 64);

    _abcdk_torch_imgproc_brightness_2d2d_cuda<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_ws, dst_ws, dst_h, (float *)gpu_alpha, (float *)gpu_bate);
    abcdk_torch_free_cuda(&gpu_alpha);
    abcdk_torch_free_cuda(&gpu_bate);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_brightness_cuda(abcdk_torch_image_t *dst, float alpha[], float bate[])
{
    int dst_depth;

    assert(dst != NULL && alpha != NULL && bate != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_brightness_cuda<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, alpha, bate);
}

__END_DECLS

#else // __cuda_cuda_h__

__BEGIN_DECLS

int abcdk_torch_imgproc_brightness_cuda(abcdk_torch_image_t *dst, float alpha[], float bate[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

__END_DECLS

#endif // __cuda_cuda_h__