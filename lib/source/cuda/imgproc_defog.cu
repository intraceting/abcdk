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
ABCDK_CUDA_GLOBAL void _abcdk_cuda_imgproc_defog_2d2d(int channels, bool packed,
                                                      T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                      size_t w, size_t h, float dack_m, T dack_a, float dack_w)
{
    size_t tid = abcdk::cuda::kernel::grid_get_tid(2, 2);

    size_t y = tid / w;
    size_t x = tid % w;

    if (x >= w || y >= h)
        return;

    T dack_c = (T)abcdk::cuda::kernel::pixel_clamp<uint32_t>(0xffffffff);
    size_t src_of[4] = {0, 0, 0, 0};
    size_t dst_of[4] = {0, 0, 0, 0};

    for (size_t z = 0; z < channels; z++)
    {
        src_of[z] = abcdk::cuda::kernel::off<T>(packed, w, src_ws, h, channels, 0, x, y, z);
        dst_of[z] = abcdk::cuda::kernel::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);

        if (dack_c > src[src_of[z]])
            dack_c = src[src_of[z]];
    }

    float t = abcdk::cuda::kernel::max<float>(dack_m, (1.0 - dack_w / dack_a * dack_c));

    for (size_t z = 0; z < channels; z++)
    {
        dst[dst_of[z]] = abcdk::cuda::kernel::pixel_clamp<T>(((src[src_of[z]] - dack_a) / t + dack_a));
    }
}

template <typename T>
ABCDK_CUDA_HOST int _abcdk_cuda_imgproc_defog(int channels, bool packed,
                                              T *dst, size_t dst_ws, T *src, size_t src_ws,
                                              size_t w, size_t h, T dack_a, float dack_m, float dack_w)
{
    uint3 dim[2];

    /*2D-2D*/
    abcdk::cuda::kernel::grid_make_2d2d(dim, w * h, 64);

    _abcdk_cuda_imgproc_defog_2d2d<T><<<dim[0], dim[1]>>>(channels, packed, dst, dst_ws, src, src_ws, w, h, dack_a, dack_m, dack_w);

    return 0;
}

int abcdk_cuda_imgproc_defog_8u_c3r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                    size_t w, size_t h, uint8_t dack_a, float dack_m, float dack_w)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);

    return _abcdk_cuda_imgproc_defog(3,true,dst,dst_ws,src,src_ws,w,h,dack_a,dack_m,dack_w);
}

int abcdk_cuda_imgproc_defog_8u_c4r(uint8_t *dst, size_t dst_ws, uint8_t *src, size_t src_ws,
                                    size_t w, size_t h, uint8_t dack_a, float dack_m, float dack_w)
{
    assert(dst != NULL && dst_ws > 0);
    assert(src != NULL && src_ws > 0);
    assert(w > 0 && h > 0);

    return _abcdk_cuda_imgproc_defog(4, true, dst, dst_ws, src, src_ws, w, h, dack_a, dack_m, dack_w);
}

#endif // __cuda_cuda_h__