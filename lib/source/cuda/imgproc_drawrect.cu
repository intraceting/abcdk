/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "kernel_1.cu.hxx"
#include "kernel_2.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_CUDA_GLOBAL void _abcdk_cuda_imgproc_drawrect_2d2d(int channels, bool packed,
                                                         T *dst, size_t w, size_t ws, size_t h,
                                                         T *color, int weight, int *corner)
{
    size_t tid = abcdk::cuda::kernel::grid_get_tid(2, 2);

    size_t y = tid / w;
    size_t x = tid % w;

    if (x >= w || y >= h)
        return;

    int x1 = corner[0];
    int y1 = corner[1];
    int x2 = corner[2];
    int y2 = corner[3];

    int chk = 0x00;

    /*上边*/
    if (x >= x1 && y >= y1 && x <= x2 && y <= y1 + weight && y <= y2)
        chk |= 0x01;

    /*下边*/
    if (x >= x1 && y <= y2 && x <= x2 && y >= y2 - weight && y >= y1)
        chk |= 0x02;

    /*左边*/
    if (x >= x1 && y >= y1 && x <= x1 + weight && y <= y2 && x <= x2)
        chk |= 0x04;

    /*右边*/
    if (x >= x2 - weight && y >= y1 && x <= x2 && y <= y2 && x >= x1)
        chk |= 0x08;

    /*为0表示不需要填充颜色。*/
    if (chk == 0)
        return;

    /*填充颜色。*/
    for (size_t z = 0; z < channels; z++)
    {
        size_t off = abcdk::cuda::kernel::off<T>(packed, w, ws, h, channels, 0, x, y, z);
        dst[off] = color[z];
    }
}

template <typename T>
ABCDK_CUDA_HOST int _abcdk_cuda_imgproc_drawrect(int channels, bool packed,
                                                 T *dst, size_t w, size_t ws, size_t h,
                                                 T *color, int weight, int *corner)
{
    void *gpu_color = NULL, *gpu_conrer = NULL;
    uint3 dim[2];

    gpu_color = abcdk_cuda_copyfrom(color, channels * sizeof(T), 1);
    gpu_conrer = abcdk_cuda_copyfrom(corner, 4 * sizeof(int), 1);

    if (!gpu_color || !gpu_conrer)
    {
        abcdk_cuda_free(&gpu_color);
        abcdk_cuda_free(&gpu_conrer);
        return -1;
    }

    /*2D-2D*/
    abcdk::cuda::kernel::grid_make_2d2d(dim, w * h, 64);

    _abcdk_cuda_imgproc_drawrect_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, w, ws, h, (T *)gpu_color, weight, (int *)gpu_conrer);

    abcdk_cuda_free(&gpu_color);
    abcdk_cuda_free(&gpu_conrer);
    return 0;
}

int abcdk_cuda_imgproc_drawrect_8u_c1r(uint8_t *dst, size_t w, size_t ws, size_t h,
                                       uint8_t color[1], int weight, int corner[4])
{
    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0 && corner != NULL);

    return _abcdk_cuda_imgproc_drawrect<uint8_t>(1, true, dst, w, ws, h, color, weight, corner);
}

int abcdk_cuda_imgproc_drawrect_8u_c3r(uint8_t *dst, size_t w, size_t ws, size_t h,
                                       uint8_t color[3], int weight, int corner[4])
{
    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0 && corner != NULL);

    return _abcdk_cuda_imgproc_drawrect<uint8_t>(3, true, dst, w, ws, h, color, weight, corner);
}

int abcdk_cuda_imgproc_drawrect_8u_c4r(uint8_t *dst, size_t w, size_t ws, size_t h,
                                       uint8_t color[4], int weight, int corner[4])
{
    assert(dst != NULL && w > 0 && ws > 0 && h > 0);
    assert(color != NULL && weight > 0 && corner != NULL);

    return _abcdk_cuda_imgproc_drawrect<uint8_t>(4, true, dst, w, ws, h, color, weight, corner);
}

#endif // __cuda_cuda_h__