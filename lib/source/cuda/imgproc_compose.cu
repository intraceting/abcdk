/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "grid.cu.hxx"
#include "util.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_CUDA_GLOBAL void _abcdk_cuda_imgproc_compose_2d2d(int channels, bool packed,
                                                        T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                        T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                        T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    size_t tid = abcdk::cuda::grid_get_tid(2, 2);

    size_t y = tid / compose_w;
    size_t x = tid % compose_w;

    size_t panorama_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
    size_t compose_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
    size_t panorama_scalars = 0;
    size_t compose_scalars = 0;

    /*融合权重。-1.0 ～ 0～ 1.0 。*/
    double scale = 0;

    if (x >= compose_w || y >= compose_h)
        return;

    for (size_t i = 0; i < channels; i++)
    {
        panorama_offset[i] = abcdk::cuda::off<T>(packed, panorama_w, panorama_ws, panorama_h, channels, 0, x + overlap_x, y + overlap_y, i);
        compose_offset[i] = abcdk::cuda::off<T>(packed, compose_w, compose_ws, compose_h, channels, 0, x, y, i);

        /*计算融合图象素是否为填充色。*/
        panorama_scalars += (panorama[panorama_offset[i]] == scalar[i] ? 1 : 0);

        /*计算融合图象素是否为填充色。*/
        compose_scalars += (compose[compose_offset[i]] == scalar[i] ? 1 : 0);
    }

    if (panorama_scalars == channels)
    {
        /*全景图象素为填充色，只取融合图象素。*/
        scale = 0;
    }
    else if (compose_scalars == channels)
    {
        /*融合图象素为填充色，只取全景图象素。*/
        scale = 1;
    }
    else
    {
        /*判断是否在图像重叠区域，从左到右渐进分配融合重叠区域的顔色权重。*/
        if (x + overlap_x <= overlap_x + overlap_w)
        {
            scale = ((overlap_w - x) / (double)overlap_w);
            /*按需优化接缝线。*/
            scale = (optimize_seam ? scale : 1 - scale);
        }
        else
        {
            /*不在重叠区域，只取融合图象素。*/
            scale = 0;
        }
    }

    for (size_t i = 0; i < channels; i++)
    {
        panorama[panorama_offset[i]] = abcdk::cuda::blend<T>(panorama[panorama_offset[i]], compose[compose_offset[i]], scale);
    }
}

template <typename T>
ABCDK_CUDA_HOST int _abcdk_cuda_imgproc_compose(int channels, bool packed,
                                                T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    T *gpu_scalar;
    uint3 dim[2];

    gpu_scalar = (T *)abcdk_cuda_copyfrom(scalar, channels * sizeof(T), 1);
    if (!gpu_scalar)
        return -1;

    /*2D-2D*/
    abcdk::cuda::grid_make_2d2d(dim, compose_w * compose_h, 64);

    _abcdk_cuda_imgproc_compose_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                                            compose, compose_w, compose_ws, compose_h,
                                                            gpu_scalar, overlap_x, overlap_y, overlap_w, optimize_seam);

    abcdk_cuda_free((void **)&gpu_scalar);

    return -1;
}

int abcdk_cuda_imgproc_compose_8u_c1r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[1], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    assert(panorama != NULL && panorama_w > 0 && panorama_ws > 0 && panorama_h > 0);
    assert(compose != NULL && compose_w > 0 && compose_ws > 0 && compose_h > 0);
    assert(scalar != NULL && overlap_x > 0 && overlap_y > 0 && overlap_w > 0);

    return _abcdk_cuda_imgproc_compose(1, true, panorama, panorama_w, panorama_ws, panorama_h,
                                       compose, compose_w, compose_ws, compose_h,
                                       scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
}

int abcdk_cuda_imgproc_compose_8u_c3r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[3], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    assert(panorama != NULL && panorama_w > 0 && panorama_ws > 0 && panorama_h > 0);
    assert(compose != NULL && compose_w > 0 && compose_ws > 0 && compose_h > 0);
    assert(scalar != NULL && overlap_x > 0 && overlap_y > 0 && overlap_w > 0);

    return _abcdk_cuda_imgproc_compose(3, true, panorama, panorama_w, panorama_ws, panorama_h,
                                       compose, compose_w, compose_ws, compose_h,
                                       scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
}

int abcdk_cuda_imgproc_compose_8u_c4r(uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                      uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                      uint8_t scalar[4], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    assert(panorama != NULL && panorama_w > 0 && panorama_ws > 0 && panorama_h > 0);
    assert(compose != NULL && compose_w > 0 && compose_ws > 0 && compose_h > 0);
    assert(scalar != NULL && overlap_x > 0 && overlap_y > 0 && overlap_w > 0);

    return _abcdk_cuda_imgproc_compose(4, true, panorama, panorama_w, panorama_ws, panorama_h,
                                       compose, compose_w, compose_ws, compose_h,
                                       scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
}

#endif // __cuda_cuda_h__