/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "../torch/imageproc.hxx"

template <typename T>
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_imgproc_line_1d_host(int channels, bool packed,
                                                                   T *dst, size_t w, size_t ws, size_t h,
                                                                   int x1, int y1, int x2, int y2,
                                                                   uint32_t *color, int weight)
{
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(abcdk_align(cpus / 2, 1))
    for (size_t i = 0; i < w * h; i++)
    {
        abcdk::torch::imageproc::line<T>(channels, packed, dst, w, ws, h, x1, y1, x2, y2, color, weight, i);
    }
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_line_host(int channels, bool packed,
                                                           T *dst, size_t w, size_t ws, size_t h,
                                                           int x1, int y1, int x2, int y2,
                                                           uint32_t *color, int weight)
{
    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0);


    _abcdk_torch_imgproc_line_1d_host<T>(channels, packed, dst, w, ws, h, x1, y1, x2, y2, color, weight);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_line_host(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight)
{
    int dst_depth;

    assert(dst != NULL && color != NULL && weight > 0);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_HOST);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_line_host<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, p1->x, p1->y, p2->x, p2->y, color, weight);
}

__END_DECLS