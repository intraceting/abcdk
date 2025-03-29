/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/memory.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/invoke.hxx"
#include "grid.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_TORCH_INVOKE_GLOBAL void _abcdk_torch_memset_2d2d_cuda(T *data, T value, size_t size)
{
    size_t tid = abcdk::torch_cuda::grid::get_tid(2, 2);

    if (tid >= size)
        return;

    data[tid] = value;
}

__BEGIN_DECLS

void *abcdk_torch_memset_cuda(void *dst, int val, size_t size)
{
    uint3 dim[2];

    /*2D-2D*/
    abcdk::torch_cuda::grid::make_dim_dim(dim, size, 64);

    _abcdk_torch_memset_2d2d_cuda<uint8_t><<<dim[0], dim[1]>>>((uint8_t *)dst, (uint8_t)val, size);

    return dst;
}

__END_DECLS

#endif //__cuda_cuda_h__
