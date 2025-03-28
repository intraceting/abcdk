/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_defog_1d_host(int channels, bool packed,
                                                          T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                          uint32_t dack_a, float dack_m, float dack_w)
{
    for (size_t i = 0; i < dst_w * dst_h; i++)
    {
        abcdk::generic::imageproc::defog<T>(channels, packed, dst, dst_w, dst_ws, dst_h, dack_a, dack_m, dack_w, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_defog_host(int channels, bool packed,
                                                      T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                      uint32_t dack_a, float dack_m, float dack_w)
{
    assert(dst != NULL && dst_ws > 0);
    assert(dst_w > 0 && dst_h > 0);

    _abcdk_torch_imgproc_defog_1d_host<T>(channels, packed, dst, dst_w, dst_ws, dst_h, dack_a, dack_m, dack_w);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_defog_host(abcdk_torch_image_t *dst, uint32_t dack_a, float dack_m, float dack_w)
{
    int dst_depth;

    assert(dst != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_defog_host<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, dack_a, dack_m, dack_w);
}

__END_DECLS