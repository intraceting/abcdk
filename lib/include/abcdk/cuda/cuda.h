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
#endif //__x86_64__
#endif //HAVE_CUDA

#ifndef __cuda_cuda_h__
/**/
typedef struct CUctx_st *CUcontext;
/**Memory types */
typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST    = 0x01,    /**< Host memory */
    CU_MEMORYTYPE_DEVICE  = 0x02,    /**< Device memory */
    CU_MEMORYTYPE_ARRAY   = 0x03,    /**< Array memory */
    CU_MEMORYTYPE_UNIFIED = 0x04     /**< Unified device or host memory */
} CUmemorytype;
#endif //__cuda_cuda_h__

#endif //ABCDK_CUDA_CUDA_H