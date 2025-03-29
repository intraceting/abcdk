/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_JPEG_UTIL_HXX
#define ABCDK_TORCH_NVIDIA_JPEG_UTIL_HXX

#include "abcdk/torch/image.h"
#include "context_robot.cu.hxx"

#ifdef __aarch64__
#include "jetson/nvmpi.h"
#include "jetson/NvJpegDecoder.h"
#include "jetson/NvJpegEncoder.h"
#endif //__aarch64__

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace jpeg
        {

        } // namespace jpeg
    } // namespace torch_cuda
} // namespace abcdk


#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_JPEG_UTIL_HXX