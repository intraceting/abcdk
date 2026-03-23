/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_RUNTIME_IN_H
#define ABCDK_XPU_RUNTIME_IN_H

#include "abcdk/xpu/runtime.h"

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG) && defined(HAVE_ONNX) && defined(HAVE_EIGEN3)
#define __XPU_GENERAL__
#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG) && defined(HAVE_ONNX) && defined(HAVE_EIGEN)

#if defined(HAVE_CUDA) && defined(HAVE_TENSORRT)
#define __XPU_NVIDIA__
#endif //#if defined(HAVE_CUDA) && defined(HAVE_TENSORRT)

void _abcdk_xpu_hwaccel_set(int hwaccel);
int _abcdk_xpu_hwaccel_get();

#endif //ABCDK_XPU_RUNTIME_IN_H