/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_VCODEC_UTIL_HXX
#define ABCDK_TORCH_NVIDIA_VCODEC_UTIL_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/torch/nvidia.h"
#include "context_robot.hxx"

#ifdef __cuda_cuda_h__

#ifdef __x86_64__
#ifdef HAVE_FFNVCODEC
#include <ffnvcodec/dynlink_loader.h> //在指定路径和系统路径中查找。
#include <ffnvcodec/dynlink_nvcuvid.h>
#else 
#include "ffnvcodec/dynlink_loader.h" //在本级路径查找。
#include "ffnvcodec/dynlink_nvcuvid.h"
#endif //HAVE_FFNVCODEC
#endif //__x86_64__

#ifdef __aarch64__
#include "jetson/nvmpi.h"
#endif //__aarch64__


namespace abcdk
{
    namespace torch_cuda
    {
        namespace vcodec
        {
            int vcodec_to_nvcodec(int format)
            {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
                switch (format)
                {
                case ABCDK_TORCH_VCODEC_H264:
                    return cudaVideoCodec_H264;
                case ABCDK_TORCH_VCODEC_HEVC:
                    return cudaVideoCodec_HEVC;
                case ABCDK_TORCH_VCODEC_MJPEG:
                    return cudaVideoCodec_JPEG;
                case ABCDK_TORCH_VCODEC_MPEG1VIDEO:
                    return cudaVideoCodec_MPEG1;
                case ABCDK_TORCH_VCODEC_MPEG2VIDEO:
                    return cudaVideoCodec_MPEG2;
                case ABCDK_TORCH_VCODEC_MPEG4:
                    return cudaVideoCodec_MPEG4;
                case ABCDK_TORCH_VCODEC_VC1:
                    return cudaVideoCodec_VC1;
                case ABCDK_TORCH_VCODEC_VP8:
                    return cudaVideoCodec_VP8;
                case ABCDK_TORCH_VCODEC_VP9:
                    return cudaVideoCodec_VP9;
                case ABCDK_TORCH_VCODEC_WMV3:
                    return cudaVideoCodec_VC1;
                }
#elif defined(__aarch64__)
                switch (format)
                {
                case ABCDK_TORCH_VCODEC_H264:
                    return NV_VIDEO_CodingH264;
                case ABCDK_TORCH_VCODEC_HEVC:
                    return NV_VIDEO_CodingHEVC;
                case ABCDK_TORCH_VCODEC_VP8:
                    return NV_VIDEO_CodingVP8;
                case ABCDK_TORCH_VCODEC_VP9:
                    return NV_VIDEO_CodingVP9;
                case ABCDK_TORCH_VCODEC_MPEG4:
                    return NV_VIDEO_CodingMPEG4;
                case ABCDK_TORCH_VCODEC_MPEG2VIDEO:
                    return NV_VIDEO_CodingMPEG2;
                default:
                    return NV_VIDEO_CodingUnused;
                }
#endif //__aarch64__
                return -1;
            }
            
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
            static bool operator==(const GUID &guid1, const GUID &guid2)
            {
                return !memcmp(&guid1, &guid2, sizeof(GUID));
            }

            static bool operator!=(const GUID &guid1, const GUID &guid2)
            {
                return !(guid1 == guid2);
            }
#endif //FFNV_CUDA_DYNLINK_LOADER_H
        } // namespace video
    } // namespace torch_cuda
} // namespace abcdk


#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_VIDEO_UTIL_HXX