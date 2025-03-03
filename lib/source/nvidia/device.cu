/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/nvidia/device.h"


#ifdef __cuda_cuda_h__

int abcdk_cuda_get_device()
{
    int device = -1;
    cudaError_t chk;

    chk = cudaGetDevice(&device);
    if(chk != cudaSuccess)
        return -1;

    return device;
}

int abcdk_cuda_set_device(int device)
{
    cudaError_t chk;

    assert(device >=0);

    chk = cudaSetDevice(device);
    if(chk != cudaSuccess)
        return -1;

    return 0;
}

int abcdk_cuda_get_device_name(char name[256], int device)
{
    struct cudaDeviceProp prop;
    cudaError_t chk;

    assert(name != NULL && device >= 0);

    chk = cudaGetDeviceProperties(&prop, device);
    if (chk != cudaSuccess)
        return -1;

    strncpy(name, prop.name, 256);

    return 0;
}

int abcdk_cuda_get_runtime_version(int *minor)
{
    int num_ver = 0;
    int major = -1;
    cudaError_t chk;

    chk = cudaRuntimeGetVersion(&num_ver);
    if (chk != cudaSuccess)
        return -1;
    
    major = num_ver / 1000;

    if (minor)
        *minor = (num_ver % 1000) / 10;

    return major;
}

void abcdk_cuda_ctx_destroy(CUcontext *ctx)
{
    CUcontext ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    cuCtxDestroy(ctx_p);
}

CUcontext abcdk_cuda_ctx_create(int device, int flag)
{
    CUcontext ctx;
    CUdevice cuda_dev;
    CUresult chk;

    chk = cuDeviceGet(&cuda_dev, device);
    if (chk != CUDA_SUCCESS)
        return NULL;

    chk = cuCtxCreate(&ctx, flag, cuda_dev);
    if (chk != CUDA_SUCCESS)
        return NULL;

    return ctx;
}

int abcdk_cuda_ctx_push_current(CUcontext ctx)
{
    CUresult chk;

    assert(ctx != NULL);

    chk = cuCtxPushCurrent(ctx);
    if (chk != CUDA_SUCCESS)
        return -1;

    return 0;
}

int abcdk_cuda_ctx_pop_current(CUcontext *ctx)
{
    CUresult chk;

    assert(ctx != NULL);

    chk = cuCtxPopCurrent(ctx);
    if (chk != CUDA_SUCCESS)
        return -1;

    return 0;
}

#else //__cuda_cuda_h__

int abcdk_cuda_get_device()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_set_device(int device)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_get_device_name(char name[256], int device)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_get_runtime_version(int *minor)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

void abcdk_cuda_ctx_destroy(CUcontext *ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
}

CUcontext abcdk_cuda_ctx_create(int device, int flag)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return NULL;
}

int abcdk_cuda_ctx_push_current(CUcontext ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

int abcdk_cuda_ctx_pop_current(CUcontext *ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含CUDA工具。");
    return -1;
}

#endif //__cuda_cuda_h__