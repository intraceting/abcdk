/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RTSP_H
#define ABCDK_RTSP_RTSP_H

#include "abcdk/util/defs.h"

#ifdef HAVE_LIVE555
#ifdef __cplusplus
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>
#include <GroupsockHelper.hh>
#include <RTSPServer.hh>
#endif //__cplusplus
#endif // HAVE_LIVE555

__BEGIN_DECLS

/**视频编码常量。*/
typedef enum _abcdk_rtsp_codec_constant
{
    ABCDK_RTSP_CODEC_NONE = -1,
#define ABCDK_RTSP_CODEC_NONE ABCDK_RTSP_CODEC_NONE

    ABCDK_RTSP_CODEC_H264 = 1,
#define ABCDK_RTSP_CODEC_H264 ABCDK_RTSP_CODEC_H264

    ABCDK_RTSP_CODEC_HEVC,
#define ABCDK_RTSP_CODEC_HEVC ABCDK_RTSP_CODEC_HEVC
#define ABCDK_RTSP_CODEC_H265 ABCDK_RTSP_CODEC_HEVC

    ABCDK_RTSP_CODEC_VP8,
#define ABCDK_RTSP_CODEC_VP8 ABCDK_RTSP_CODEC_VP8

    ABCDK_RTSP_CODEC_VP9,
#define ABCDK_RTSP_CODEC_VP9 ABCDK_RTSP_CODEC_VP9

    ABCDK_RTSP_CODEC_ACC = 100,
#define ABCDK_RTSP_CODEC_ACC ABCDK_RTSP_CODEC_ACC

} abcdk_rtsp_codec_constant_t;


__END_DECLS

#endif //ABCDK_RTSP_RTSP_H