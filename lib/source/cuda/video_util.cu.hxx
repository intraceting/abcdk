/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_UTIL_HXX
#define ABCDK_CUDA_VIDEO_UTIL_HXX

#include "abcdk/cuda/cuda.h"
#include "abcdk/media/cdcfmt.h"
#include "context_robot.cu.hxx"

#ifdef __x86_64__
#ifdef HAVE_FFNVCODEC
#include "ffnvcodec/dynlink_loader.h"
#include "ffnvcodec/dynlink_nvcuvid.h"
#endif //HAVE_FFNVCODEC
#endif //__x86_64__

#ifdef __aarch64__
#include "jetson/nvmpi.h"
#include "jetson/NvJpegDecoder.h"
#include "jetson/NvJpegEncoder.h"
#endif //__aarch64__

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            int cdcfmt_to_nvcodec(int cdcfmt)
            {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
                switch (id)
                {
                case ABCDK_MEDIA_CDCFMT_H264:
                    return cudaVideoCodec_H264;
                case ABCDK_MEDIA_CDCFMT_HEVC:
                    return cudaVideoCodec_HEVC;
                case ABCDK_MEDIA_CDCFMT_MJPEG:
                    return cudaVideoCodec_JPEG;
                case ABCDK_MEDIA_CDCFMT_MPEG1VIDEO:
                    return cudaVideoCodec_MPEG1;
                case ABCDK_MEDIA_CDCFMT_MPEG2VIDEO:
                    return cudaVideoCodec_MPEG2;
                case ABCDK_MEDIA_CDCFMT_MPEG4:
                    return cudaVideoCodec_MPEG4;
                case ABCDK_MEDIA_CDCFMT_VC1:
                    return cudaVideoCodec_VC1;
                case ABCDK_MEDIA_CDCFMT_VP8:
                    return cudaVideoCodec_VP8;
                case ABCDK_MEDIA_CDCFMT_VP9:
                    return cudaVideoCodec_VP9;
                case ABCDK_MEDIA_CDCFMT_WMV3:
                    return cudaVideoCodec_VC1;
                }
#elif defined(__aarch64__)
                switch (id)
                {
                case ABCDK_MEDIA_CDCFMT_H264:
                    return NV_VIDEO_CodingH264;
                case ABCDK_MEDIA_CDCFMT_HEVC:
                    return NV_VIDEO_CodingHEVC;
                case ABCDK_MEDIA_CDCFMT_VP8:
                    return NV_VIDEO_CodingVP8;
                case ABCDK_MEDIA_CDCFMT_VP9:
                    return NV_VIDEO_CodingVP9;
                case ABCDK_MEDIA_CDCFMT_MPEG4:
                    return NV_VIDEO_CodingMPEG4;
                case ABCDK_MEDIA_CDCFMT_MPEG2VIDEO:
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
    } // namespace cuda
} // namespace abcdk


#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_UTIL_HXX