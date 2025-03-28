/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_brightness_1d_host(int channels, bool packed,
                                                               T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, float *alpha, float *bate)
{
    for (size_t i = 0; i < dst_w * dst_h; i++)
    {
        abcdk::generic::imageproc::brightness<T>(channels, packed, dst, dst_ws, dst_ws, dst_h, alpha, bate, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_brightness_host(int channels, bool packed,
                                                           T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, float *alpha, float *bate)
{

    assert(dst != NULL && dst_ws > 0);
    assert(dst_w > 0 && dst_h > 0);
    assert(alpha != NULL && bate != NULL);

    _abcdk_torch_imgproc_brightness_1d_host<T>(channels, packed, dst, dst_ws, dst_ws, dst_h, alpha, bate);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_brightness_host(abcdk_torch_image_t *dst, float alpha[], float bate[])
{
    int dst_depth;

    assert(dst != NULL && alpha != NULL && bate != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_brightness_host<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, alpha, bate);
}

__END_DECLS