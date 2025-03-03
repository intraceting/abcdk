/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_defog_1d(int channels, bool packed,
                                                     T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                     size_t w, size_t h, float dack_m = 0.35, T dack_a = 220, float dack_w = 0.9)
{
    for (size_t i = 0; i < w * h; i++)
    {
        abcdk::generic::imageproc::defog<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_m, dack_a, dack_w, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_defog(int channels, bool packed,
                                                 T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                 size_t w, size_t h, float dack_m, T dack_a, float dack_w)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);

    _abcdk_torch_imgproc_defog_1d<T>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_m, dack_a, dack_w);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_defog_8u(int channels, int packed,
                                 uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                 size_t w, size_t h, uint8_t dack_a, float dack_m, float dack_w)
{
    return _abcdk_torch_imgproc_defog<uint8_t>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_a, dack_m, dack_w);
}

__END_DECLS