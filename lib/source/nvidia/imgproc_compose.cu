/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/imgproc.h"
#include "../generic/imageproc.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_imgproc_compose_2d2d(int channels, bool packed,
                                                          T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                          T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                          T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    abcdk::generic::imageproc::compose<T>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h, compose, compose_w, compose_ws, compose_h, scalar,
                                          overlap_x, overlap_y, overlap_w, optimize_seam, tid);
}

template <typename T>
ABCDK_INVOKE_HOST int _abcdk_cuda_imgproc_compose(int channels, bool packed,
                                                  T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                  T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                  T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam)
{
    void *gpu_scalar;
    uint3 dim[2];

    assert(panorama != NULL && panorama_w > 0 && panorama_ws > 0 && panorama_h > 0);
    assert(compose != NULL && compose_w > 0 && compose_ws > 0 && compose_h > 0);
    assert(scalar != NULL);// && overlap_x >= 0 && overlap_y >= 0 && overlap_w >= 0);

    gpu_scalar = abcdk_cuda_copyfrom(scalar, channels * sizeof(T), 1);
    if (!gpu_scalar)
        return -1;

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, compose_w * compose_h, 64);

    _abcdk_cuda_imgproc_compose_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                                            compose, compose_w, compose_ws, compose_h,
                                                            (T*)gpu_scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
    abcdk_cuda_free(&gpu_scalar);

    return 0;
}

__BEGIN_DECLS

int abcdk_cuda_imgproc_compose_8u(int channels, int packed,
                                  uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                  uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                  uint8_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    return _abcdk_cuda_imgproc_compose<uint8_t>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                                compose, compose_w, compose_ws, compose_h,
                                                scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
}

__END_DECLS

#else // __cuda_cuda_h__

__BEGIN_DECLS

int abcdk_cuda_imgproc_compose_8u(int channels, int packed,
                                  uint8_t *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                  uint8_t *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                  uint8_t scalar[], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

__END_DECLS

#endif // __cuda_cuda_h__