/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/nvidia/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_cuda_init(uint32_t flags)
{
    CUresult chk;
    chk = cuInit(flags);
    if(chk != CUDA_SUCCESS)
        return -1;
    
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

#else // __cuda_cuda_h__

int abcdk_cuda_ctx_init(uint32_t flags)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_cuda_get_runtime_version(int *minor)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}

int abcdk_cuda_get_device_name(char name[256], int device)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA工具。"));
    return -1;
}


#endif //__cuda_cuda_h__


__END_DECLS