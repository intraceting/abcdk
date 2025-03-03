/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/memory.h"
#include "../generic/invoke.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_memset_2d2d(T *data, T value, size_t size)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    if (tid >= size)
        return;

    data[tid] = value;
}

__BEGIN_DECLS

void *abcdk_cuda_memset(void *dst, int val, size_t size)
{
    uint3 dim[2];

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, size, 64);

    _abcdk_cuda_memset_2d2d<uint8_t><<<dim[0], dim[1]>>>((uint8_t *)dst, (uint8_t)val, size);

    return dst;
}

__END_DECLS

#else //__cuda_cuda_h__

__BEGIN_DECLS

void *abcdk_cuda_memset(void *dst, int val, size_t size)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

__END_DECLS

#endif //__cuda_cuda_h__
