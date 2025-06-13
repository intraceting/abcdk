/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/imageproc.hxx"
#include "grid.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_imgproc_drawmask_2d2d_cuda(int channels, bool packed,
                                                                       T *dst, size_t dst_ws, float *mask, size_t mask_ws, size_t w, size_t h, float threshold, uint32_t *color)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::imageproc::drawmask<T>(channels, packed, dst, dst_ws, mask, mask_ws, w, h, threshold, color, tid);
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_drawmask_cuda(int channels, bool packed,
                                                               T *dst, size_t dst_ws, float *mask, size_t mask_ws, size_t w, size_t h, float threshold, uint32_t *color)
{
    void *gpu_color = NULL;
    uint3 dim[2];

    assert(dst != NULL && dst_ws > 0 && mask != NULL && mask_ws > 0 && w > 0 && h > 0);
    assert(color != NULL);

    gpu_color = abcdk_torch_copyfrom_cuda(color, channels * sizeof(uint32_t), 1);

    if (!gpu_color)
    {
        abcdk_torch_free_cuda(&gpu_color);
        return -1;
    }

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_torch_imgproc_drawmask_2d2d_cuda<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_ws, mask, mask_ws, w, h, threshold, (uint32_t *)gpu_color);
    abcdk_torch_free_cuda(&gpu_color);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_drawmask_cuda(abcdk_torch_image_t *dst, abcdk_torch_image_t *mask, float threshold, uint32_t color[])
{
    int dst_depth;

    assert(dst != NULL && mask != NULL && color != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(mask->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_drawmask_cuda<uint8_t>(dst_depth, true, dst->data[0], dst->stride[0], (float *)mask->data[0], mask->stride[0], dst->width, dst->height, threshold, color);
}

__END_DECLS


#endif // __cuda_cuda_h__