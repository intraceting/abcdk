/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_brightness_1d(int channels, bool packed,
                                                          T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                          size_t w, size_t h, float *alpha, float *bate)
{
    for (size_t i = 0; i < w * h; i++)
    {
        abcdk::generic::imageproc::brightness<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, alpha, bate, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_brightness(int channels, bool packed,
                                                      T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                      size_t w, size_t h, float *alpha, float *bate)
{

    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);
    assert(alpha != NULL && bate != NULL);

    _abcdk_torch_imgproc_brightness_1d<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, alpha, bate);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_brightness_8u(int channels, int packed,
                                      uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                      size_t w, size_t h, float *alpha, float *bate)
{
    return _abcdk_torch_imgproc_brightness<uint8_t>(channels, packed, dst, dst_ws, src, src_ws, w, h, alpha, bate);
}

__END_DECLS