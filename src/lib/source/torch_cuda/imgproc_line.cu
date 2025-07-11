
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
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_imgproc_line_2d2d_cuda(int channels, bool packed,
                                                                   T *dst, size_t w, size_t ws, size_t h,
                                                                   int x1, int y1, int x2, int y2,
                                                                   uint32_t *color, int weight)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::imageproc::line<T>(channels, packed, dst, w, ws, h, x1, y1, x2, y2, color, weight, tid);
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_line_cuda(int channels, bool packed,
                                                           T *dst, size_t w, size_t ws, size_t h,
                                                           int x1, int y1, int x2, int y2,
                                                           uint32_t *color, int weight)
{
    void *gpu_color = NULL;
    uint3 dim[2];

    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0);

    gpu_color = abcdk_torch_copyfrom_cuda(color, channels * sizeof(uint32_t), 1);

    if (!gpu_color)
    {
        abcdk_torch_free_cuda(&gpu_color);
        return -1;
    }

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, w * h, 64);

    _abcdk_torch_imgproc_line_2d2d_cuda<T><<<dim[0], dim[1]>>>(channels, packed, dst, w, ws, h, x1, y1, x2, y2, (uint32_t *)gpu_color, weight);
    abcdk_torch_free_cuda(&gpu_color);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_line_cuda(abcdk_torch_image_t *dst, const abcdk_torch_point_t *p1, const abcdk_torch_point_t *p2, uint32_t color[], int weight)
{
    int dst_depth;

    assert(dst != NULL && color != NULL && weight > 0);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_line_cuda<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, p1->x, p1->y, p2->x, p2->y, color, weight);
}

__END_DECLS

#endif // __cuda_cuda_h__
