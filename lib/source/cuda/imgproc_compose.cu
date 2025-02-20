/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/imgproc.h"
#include "../impl/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_compose_2d2d(int channels, bool packed,
                                                          T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                          T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                          T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::imageproc::compose_kernel<T>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h, compose, compose_w, compose_ws, compose_h, scalar,
                                        overlap_x, overlap_y, overlap_w, optimize_seam, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_compose(int channels, bool packed,
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
    abcdk::cuda::grid::make_dim_dim(dim, compose_w * compose_h, 64);

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