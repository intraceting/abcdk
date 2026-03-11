/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_VCODEC_UTIL_HXX
#define ABCDK_XPU_GENERAL_VCODEC_UTIL_HXX

#include "abcdk/xpu/types.h"
#include "../base.in.h"

namespace abcdk_xpu
{
    namespace general
    {
        namespace vcodec
        {
            static inline AVCodecID local_to_ffmpeg(abcdk_xpu_vcodec_id_t id)
            {
                switch (id)
                {
                case ABCDK_XPU_VCODEC_ID_MJPEG:
                    return AV_CODEC_ID_MJPEG;
                case ABCDK_XPU_VCODEC_ID_MPEG1VIDEO:
                    return AV_CODEC_ID_MPEG1VIDEO;
                case ABCDK_XPU_VCODEC_ID_MPEG2VIDEO:
                    return AV_CODEC_ID_MPEG2VIDEO;
                case ABCDK_XPU_VCODEC_ID_MPEG4:
                    return AV_CODEC_ID_MPEG4;
                case ABCDK_XPU_VCODEC_ID_H264:
                    return AV_CODEC_ID_H264;
                case ABCDK_XPU_VCODEC_ID_HEVC:
                    return AV_CODEC_ID_H265;
                case ABCDK_XPU_VCODEC_ID_VC1:
                    return AV_CODEC_ID_VC1;
                case ABCDK_XPU_VCODEC_ID_VP8:
                    return AV_CODEC_ID_VP8;
                case ABCDK_XPU_VCODEC_ID_VP9:
                    return AV_CODEC_ID_VP9;
                default:
                    return AV_CODEC_ID_NONE;
                }
            }

            static inline AVCodecID local_to_ffmpeg(int id)
            {
                return local_to_ffmpeg((abcdk_xpu_vcodec_id_t)id);
            }
        } // namespace vcodec
    } // namespace general
} // namespace abcdk_xpu

#endif // ABCDK_XPU_GENERAL_VCODEC_UTIL_HXX