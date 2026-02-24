/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_RUNTIME_IN_H
#define ABCDK_XPU_RUNTIME_IN_H

#include "abcdk/xpu/runtime.h"
#include "abcdk/ffmpeg/ffmpeg.h"

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
/*TensorRT interface.*/
#ifdef HAVE_TENSORRT
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#endif //HAVE_TENSORRT
#else //#ifdef HAVE_CUDA
/*Cancel TensorRT interface.*/
#undef HAVE_TENSORRT
#endif //HAVE_CUDA

#ifdef HAVE_OPENCV
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#endif // HAVE_OPENCV_XFEATURES2D
#ifdef HAVE_OPENCV_FREETYPE
#include "opencv2/freetype.hpp"
#endif //HAVE_OPENCV_FREETYPE
#endif //#ifdef HAVE_OPENCV

#ifdef HAVE_ONNX
#ifndef ONNX_ML
#define ONNX_ML
#endif // ONNX_ML
#include "onnx/onnx_pb.h"
#endif //#ifdef HAVE_ONNX


#ifdef __NVCC__
#define __ABCDK_XPU_INVOKE_DEVICE  __device__ __forceinline__
#define __ABCDK_XPU_INVOKE_HOST __host__ __forceinline__
#define __ABCDK_XPU_INVOKE_GLOBAL __global__
#else //__NVCC__
#define __ABCDK_XPU_INVOKE_DEVICE  static inline
#define __ABCDK_XPU_INVOKE_HOST static inline
#define __ABCDK_XPU_INVOKE_GLOBAL static inline
#endif //__NVCC__

void _abcdk_xpu_hwaccel_set(int hwaccel);
int _abcdk_xpu_hwaccel_get();


#endif //ABCDK_XPU_RUNTIME_IN_H