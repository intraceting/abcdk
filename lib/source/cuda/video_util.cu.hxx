/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_UTIL_HXX
#define ABCDK_CUDA_VIDEO_UTIL_HXX

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
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
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            int codecid_ffmpeg_to_nvcodec(enum AVCodecID id)
            {
#ifdef FFNV_CUDA_DYNLINK_LOADER_H
                switch (id)
                {
                case AV_CODEC_ID_H264:
                    return cudaVideoCodec_H264;
                case AV_CODEC_ID_HEVC:
                    return cudaVideoCodec_HEVC;
                case AV_CODEC_ID_MJPEG:
                    return cudaVideoCodec_JPEG;
                case AV_CODEC_ID_MPEG1VIDEO:
                    return cudaVideoCodec_MPEG1;
                case AV_CODEC_ID_MPEG2VIDEO:
                    return cudaVideoCodec_MPEG2;
                case AV_CODEC_ID_MPEG4:
                    return cudaVideoCodec_MPEG4;
                case AV_CODEC_ID_VC1:
                    return cudaVideoCodec_VC1;
                case AV_CODEC_ID_VP8:
                    return cudaVideoCodec_VP8;
                case AV_CODEC_ID_VP9:
                    return cudaVideoCodec_VP9;
                case AV_CODEC_ID_WMV3:
                    return cudaVideoCodec_VC1;
                }
#elif defined(__aarch64__)
                switch (id)
                {
                case AV_CODEC_ID_H264:
                    return NV_VIDEO_CodingH264;
                case AV_CODEC_ID_HEVC:
                    return NV_VIDEO_CodingHEVC;
                case AV_CODEC_ID_VP8:
                    return NV_VIDEO_CodingVP8;
                case AV_CODEC_ID_VP9:
                    return NV_VIDEO_CodingVP9;
                case AV_CODEC_ID_MPEG4:
                    return NV_VIDEO_CodingMPEG4;
                case AV_CODEC_ID_MPEG2VIDEO:
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


#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_UTIL_HXX