/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_VCODEC_UTIL_HXX
#define ABCDK_XPU_NVIDIA_VCODEC_UTIL_HXX

#include "abcdk/xpu/types.h"
#include "../base.in.h"

#ifdef __x86_64__
#include "ffnvcodec/dynlink_loader.h"
#include "ffnvcodec/dynlink_nvcuvid.h"
#endif // #ifdef __x86_64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace vcodec
        {
#ifdef __x86_64__
            static inline cudaVideoCodec local_to_nvcodec(abcdk_xpu_vcodec_id_t id)
            {
                switch (id)
                {
                case ABCDK_XPU_VCODEC_ID_MJPEG:
                    return cudaVideoCodec_JPEG;
                case ABCDK_XPU_VCODEC_ID_MPEG1VIDEO:
                    return cudaVideoCodec_MPEG1;
                case ABCDK_XPU_VCODEC_ID_MPEG2VIDEO:
                    return cudaVideoCodec_MPEG2;
                case ABCDK_XPU_VCODEC_ID_MPEG4:
                    return cudaVideoCodec_MPEG4;
                case ABCDK_XPU_VCODEC_ID_H264:
                    return cudaVideoCodec_H264;
                case ABCDK_XPU_VCODEC_ID_HEVC:
                    return cudaVideoCodec_HEVC;
                case ABCDK_XPU_VCODEC_ID_VC1:
                    return cudaVideoCodec_VC1;
                case ABCDK_XPU_VCODEC_ID_VP8:
                    return cudaVideoCodec_VP8;
                case ABCDK_XPU_VCODEC_ID_VP9:
                    return cudaVideoCodec_VP9;
                default:
                    return cudaVideoCodec_NumCodecs;
                }
            }
#endif // #ifdef __x86_64__

#ifdef __aarch64__
            static inline int local_to_nvcodec(abcdk_xpu_vcodec_id_t id)
            {
                switch (id)
                {
                case ABCDK_XPU_VCODEC_ID_MPEG2VIDEO:
                    return 1;
                case ABCDK_XPU_VCODEC_ID_MPEG4:
                    return 2;
                case ABCDK_XPU_VCODEC_ID_H264:
                    return 3;
                case ABCDK_XPU_VCODEC_ID_HEVC:
                    return 4;
                case ABCDK_XPU_VCODEC_ID_VP8:
                    return 5;
                case ABCDK_XPU_VCODEC_ID_VP9:
                    return 6;
                default:
                    return -1;
                }
            }
#endif // #ifdef __aarch64__

        } // namespace vcodec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_VCODEC_UTIL_HXX