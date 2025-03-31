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
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_imgproc_stuff_2d2d_cuda(int channels, bool packed,
                                                              T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, uint32_t *scalar,
                                                              size_t roi_x, size_t roi_y, size_t roi_w, size_t roi_h)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    abcdk::torch::imageproc::stuff<T>(channels, packed, dst, dst_w, dst_ws, dst_h, scalar, roi_x, roi_y, roi_w, roi_h, tid);
}

template <typename T>
ABCDK_TORCH_INVOKE_HOST int _abcdk_torch_imgproc_stuff_cuda(int channels, bool packed,
                                                      T *dst, size_t dst_w, size_t dst_ws, size_t dst_h, uint32_t *scalar,
                                                      const abcdk_torch_rect_t *roi)
{
    void *gpu_scalar;
    uint3 dim[2];

    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0 && scalar != NULL);

    gpu_scalar = abcdk_torch_copyfrom_cuda(scalar, channels * sizeof(uint32_t), 1);
    if (!gpu_scalar)
        return -1;

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, dst_w * dst_h, 64);

    if (roi)
        _abcdk_torch_imgproc_stuff_2d2d_cuda<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_w, dst_ws, dst_h, (uint32_t *)gpu_scalar, roi->x, roi->y, roi->width, roi->height);
    else
        _abcdk_torch_imgproc_stuff_2d2d_cuda<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_w, dst_ws, dst_h, (uint32_t *)gpu_scalar, 0, 0, dst_w, dst_h);

    abcdk_torch_free_cuda(&gpu_scalar);

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_stuff_cuda(abcdk_torch_image_t *dst, uint32_t scalar[], const abcdk_torch_rect_t *roi)
{
    int dst_depth;

    assert(dst != NULL && scalar != NULL);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_stuff_cuda<uint8_t>(dst_depth, true, dst->data[0], dst->width, dst->stride[0], dst->height, scalar, roi);
}

__END_DECLS


#endif // __cuda_cuda_h__
