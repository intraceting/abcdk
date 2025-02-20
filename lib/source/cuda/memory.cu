/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/memory.h"
#include "../impl/invoke.hxx"
#include "grid.cu.hxx"

#ifdef __cuda_cuda_h__

void abcdk_cuda_free(void **data)
{
    void *data_p;

    if (!data || !*data)
        return;

    data_p = *data;
    *data = NULL;

    cudaFree(data_p);
}

void *abcdk_cuda_alloc(size_t size)
{
    void *data;
    cudaError_t chk;

    assert(size > 0);

    chk = cudaMalloc(&data, size);
    if (chk != cudaSuccess)
        return NULL;

    return data;
}

void *abcdk_cuda_alloc_z(size_t size)
{
    void *data;
    
    assert(size > 0);

    data = abcdk_cuda_alloc(size);
    if(!data)
        return NULL;

    abcdk_cuda_memset(data, 0, size);

    return data;
}

template <typename T>
ABCDK_INVOKE_GLOBAL void _abcdk_cuda_memset_2d2d(T *data, T value, size_t size)
{
    size_t tid = abcdk::cuda::grid::get_tid(2, 2);

    if (tid >= size)
        return;

    data[tid] = value;
}

void *abcdk_cuda_memset(void *dst, int val, size_t size)
{
    uint3 dim[2];

    /*2D-2D*/
    abcdk::cuda::grid::make_dim_dim(dim, size, 64);

    _abcdk_cuda_memset_2d2d<uint8_t><<<dim[0], dim[1]>>>((uint8_t *)dst, (uint8_t)val, size);

    return dst;
}

int abcdk_cuda_memcpy(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size)
{
    cudaMemcpyKind kind = cudaMemcpyDefault;
    cudaError_t chk;

    assert(dst != NULL && src != NULL && size > 0);

    if (src_in_host && dst_in_host)
        kind = cudaMemcpyHostToHost;
    else if (src_in_host)
        kind = cudaMemcpyHostToDevice;
    else if (dst_in_host)
        kind = cudaMemcpyDeviceToHost;
    else
        kind = cudaMemcpyDeviceToDevice;

    chk = cudaMemcpy(dst, src, size, kind);

    if (chk != cudaSuccess)
        return -1;

    return 0;
}

int abcdk_cuda_memcpy_2d(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                         const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                         size_t roi_width_bytes, size_t roi_height)
{
    CUDA_MEMCPY2D copy_args = {0};
    CUresult chk;

    assert(dst != NULL && src != NULL && roi_width_bytes > 0 && roi_height > 0);

    copy_args.dstXInBytes = dst_x_bytes;
    copy_args.dstY = dst_y;
    copy_args.dstMemoryType = (dst_in_host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE);
    copy_args.dstHost = (dst_in_host ? dst : NULL);
    copy_args.dstDevice = (CUdeviceptr)(dst_in_host ? NULL : dst);
    copy_args.dstPitch = dst_pitch;

    copy_args.srcXInBytes = src_x_bytes;
    copy_args.srcY = src_y;
    copy_args.srcMemoryType = (src_in_host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE);
    copy_args.srcHost = (src_in_host ? src : NULL);
    copy_args.srcDevice = (CUdeviceptr)(src_in_host ? NULL : src);
    copy_args.srcPitch = src_pitch;

    copy_args.WidthInBytes = roi_width_bytes;
    copy_args.Height = roi_height;

    chk = cuMemcpy2D(&copy_args);

    if (chk != CUDA_SUCCESS)
        return -1;

    return 0;
}

void *abcdk_cuda_copyfrom(const void *src, size_t size, int src_in_host)
{
    void *dst;
    int chk;

    assert(src != NULL && size > 0);

    dst = abcdk_cuda_alloc(size);
    if (!dst)
        return NULL;

    chk = abcdk_cuda_memcpy(dst, 0, src, src_in_host, size);
    if (chk == 0)
        return dst;

    abcdk_cuda_free(&dst);
    return NULL;
}

#endif //__cuda_cuda_h__
