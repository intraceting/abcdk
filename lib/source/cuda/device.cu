/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/device.h"
#include "grid.cu.hxx"

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

#endif //__cuda_cuda_h__