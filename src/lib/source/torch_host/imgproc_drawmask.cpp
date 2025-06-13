/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../torch/imageproc.hxx"

template <typename T>
ABCDK_TORCH_INVOKE_HOST void _abcdk_torch_imgproc_drawmask_1d_host(int channels, bool packed,
                                                                   T *dst, size_t dst_ws, float *mask, size_t mask_ws, size_t w, size_t h, float threshold, uint32_t *color)
{
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(abcdk_align(cpus / 2, 1))
    for (size_t i = 0; i < w * h; i++)
    {
        abcdk::torch::imageproc::drawmask<T>(channels, packed, dst, dst_ws, mask, mask_ws, w, h, threshold, color, i);
    }
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_drawmask_host(int channels, bool packed,
                                                         T *dst, size_t dst_ws, float *mask, size_t mask_ws, size_t w, size_t h, float threshold, uint32_t *color)
{
    assert(dst != NULL && dst_ws > 0 && mask != NULL && mask_ws > 0 && w > 0 && h > 0);
    assert(color != NULL);

    _abcdk_torch_imgproc_drawmask_1d_host<T>(channels, packed, dst, dst_ws, mask, mask_ws, w, h, threshold, color);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_drawmask_host(abcdk_torch_image_t *dst, abcdk_torch_image_t *mask, float threshold, uint32_t color[])
{
    int dst_depth;

    assert(dst != NULL && mask != NULL && color != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);
    assert(mask->tag == ABCDK_TORCH_TAG_HOST);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_drawmask_host<uint8_t>(dst_depth, true, dst->data[0], dst->stride[0], (float *)mask->data[0], mask->stride[0], dst->width, dst->height, threshold, color);
}

__END_DECLS