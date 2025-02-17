/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CUDA_CUDA_H
#define ABCDK_CUDA_CUDA_H

#include "abcdk/util/general.h"

#ifdef HAVE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <npp.h>

#ifdef __x86_64__
#include <nvjpeg.h>

#ifdef HAVE_FFNVCODEC
#include "ffnvcodec/dynlink_loader.h"
#include "ffnvcodec/dynlink_nvcuvid.h"
#endif //HAVE_FFNVCODEC

#endif //__x86_64__

#endif //HAVE_CUDA

/** CUDA函数修饰。*/
#ifdef __NVCC__
#define ABCDK_CUDA_DEVICE __device__ __forceinline__
#define ABCDK_CUDA_HOST __host__ __forceinline__
#define ABCDK_CUDA_GLOBAL __global__
#else //__NVCC__
#define ABCDK_CUDA_DEVICE
#define ABCDK_CUDA_HOST 
#define ABCDK_CUDA_GLOBAL 
#endif //__NVCC__


#endif //ABCDK_CUDA_CUDA_H