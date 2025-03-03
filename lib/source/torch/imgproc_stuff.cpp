/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_stuff_1d(int channels, bool packed,
                                                     T *dst, size_t dst_w, size_t dst_ws, size_t dst_h,
                                                     T *scalar)
{
    for (size_t i = 0; i < dst_w * dst_h; i++)
    {
        abcdk::generic::imageproc::stuff<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_stuff(int channels, bool packed,
                                                 T *dst, size_t width, size_t pitch, size_t height,
                                                 T *scalar)
{
    assert(dst != NULL && width > 0 && pitch > 0 && height > 0 && scalar != NULL);

    _abcdk_torch_imgproc_stuff_1d<T>(channels, packed, dst, width, pitch, height, scalar);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_stuff_8u(int channels, int packed, uint8_t *dst, size_t width, size_t pitch, size_t height, uint8_t scalar[])
{
    return _abcdk_torch_imgproc_stuff<uint8_t>(channels, packed, dst, width, pitch, height, scalar);
}

__END_DECLS