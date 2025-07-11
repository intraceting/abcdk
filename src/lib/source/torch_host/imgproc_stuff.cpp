/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../torch/imageproc.hxx"

template <typename T>
ABCDK_TORCH_INVOKE_HOST void _abcdk_torch_imgproc_stuff_1d_host(int channels, bool packed,
                                                          T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, uint32_t *scalar,
                                                          size_t roi_x, size_t roi_y, size_t roi_w, size_t roi_h)
{
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(abcdk_align(cpus / 2, 1))
    for (size_t i = 0; i < dst_w * dst_h; i++)
    {
        abcdk::torch::imageproc::stuff<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, roi_x, roi_y, roi_w, roi_h, i);
    }
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_stuff_host(int channels, bool packed,
                                                      T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, uint32_t *scalar,
                                                      const abcdk_torch_rect_t *roi)
{
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0 && scalar != NULL);

    if (roi)
        _abcdk_torch_imgproc_stuff_1d_host<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, roi->x, roi->y, roi->width, roi->height);
    else
        _abcdk_torch_imgproc_stuff_1d_host<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, 0, 0, dst_w, dst_h);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_stuff_host(abcdk_torch_image_t *dst, uint32_t scalar[], const abcdk_torch_rect_t *roi)
{
    int dst_depth;

    assert(dst != NULL && scalar != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_stuff_host<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, scalar, roi);
}

__END_DECLS