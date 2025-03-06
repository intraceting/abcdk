/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_INVOKE_HXX
#define ABCDK_GENERIC_INVOKE_HXX

#include "abcdk/util/defs.h"

#ifdef __NVCC__
#define ABCDK_INVOKE_DEVICE  __device__ __forceinline__
#define ABCDK_INVOKE_HOST __host__ __forceinline__
#define ABCDK_INVOKE_GLOBAL __global__
#else //__NVCC__
#define ABCDK_INVOKE_DEVICE static inline
#define ABCDK_INVOKE_HOST static inline
#define ABCDK_INVOKE_GLOBAL static inline
#endif //__NVCC__

#endif //ABCDK_GENERIC_INVOKE_HXX