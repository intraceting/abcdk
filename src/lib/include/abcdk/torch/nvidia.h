/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_H
#define ABCDK_TORCH_NVIDIA_H

/*CUDA interface.*/
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

/*TensorRT interface.*/
#ifdef HAVE_TENSORRT
#ifdef __cplusplus
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#endif //__cplusplus
#endif //HAVE_TENSORRT

__BEGIN_DECLS

#ifndef __cuda_cuda_h__



#endif //__cuda_cuda_h__


__END_DECLS

#endif //ABCDK_TORCH_NVIDIA_H