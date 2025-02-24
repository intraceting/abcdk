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

#endif //ABCDK_CUDA_CUDA_H