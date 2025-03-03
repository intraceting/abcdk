/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"

template <typename T>
ABCDK_INVOKE_HOST void _abcdk_torch_imgproc_compose_1d(int channels, bool packed,
                                                       T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                       T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                       T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{

    for (size_t i = 0; i < compose_w * compose_h; i++)
    {
        abcdk::generic::imageproc::compose<T>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                              compose, compose_w, compose_ws, compose_h,
                                              scalar, overlap_x, overlap_y, overlap_w, optimize_seam, i);
    }
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_torch_imgproc_compose(int channels, bool packed,
                                                   T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                   T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                   T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    assert(panorama != NULL && panorama_w > 0 && panorama_ws > 0 && panorama_h > 0);
    assert(compose != NULL && compose_w > 0 && compose_ws > 0 && compose_h > 0);
    assert(scalar != NULL && overlap_x > 0 && overlap_y > 0 && overlap_w > 0);

    _abcdk_torch_imgproc_compose_1d<T>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                       compose, compose_w, compose_ws, compose_h,
                                       scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_compose_8u(int channels, int packed,
                                   uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                   uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                   uint8_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    return _abcdk_torch_imgproc_compose<uint8_t>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                                 compose, compose_w, compose_ws, compose_h,
                                                 scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
}

__END_DECLS